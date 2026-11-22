"""Pull human-corrected sleep frames out of Cloudflare D1 + R2.

The labels live in the remote D1 table ``sleep_frame`` (written by the review UI);
the images live in the configured R2 bucket under ``frames/{userId}/{date}/{epochMs}.jpg``.

Two facts drive the design:

- **R2 expires old frames.** Every run reconciles D1 against the actual R2 listing and
  reports which nights were dropped, rather than failing on the first 404.
- **Every reviewed row is a model mistake.** The review UI only records a correction
  when a human touches a frame, so ``corrected_label != label`` holds for all of
  them. That makes this set hard examples, not a random sample.

Credentials and environment settings are read from ``train/.env`` first, then
``os.environ``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import config
import r2_store


@dataclass(frozen=True)
class Frame:
    frame_id: int
    date: str
    camera: str
    timestamp: str
    image_key: str
    predicted_label: str
    label: str  # the human-corrected label


def _cloudflare_account_id() -> str:
    account_id = config.get_env("CLOUDFLARE_ACCOUNT_ID")
    if account_id:
        return account_id
    endpoint = config.get_env("R2_ENDPOINT_URL")
    match = re.search(r"https?://([0-9a-f]+)\.r2\.cloudflarestorage\.com", endpoint)
    if match:
        return match.group(1)
    raise SystemExit(
        "CLOUDFLARE_ACCOUNT_ID is not set and could not be derived from "
        "R2_ENDPOINT_URL. Set CLOUDFLARE_ACCOUNT_ID or R2_ENDPOINT_URL in "
        "os.environ or train/.env."
    )


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def d1_query(sql: str, *, d1_database_id: str) -> list[dict]:
    """Run one read-only SQL statement against the remote D1 database."""
    env = dict(os.environ, CLOUDFLARE_ACCOUNT_ID=_cloudflare_account_id())
    raw = _run(
        [
            "pnpx",
            "--allow-build=workerd",
            "cf",
            "d1",
            "query",
            d1_database_id,
            "--sql",
            sql,
            "-q",
        ],
        env=env,
    )
    start = raw.find("[")
    if start < 0:
        raise SystemExit(f"unexpected cf d1 query output:\n{raw[:2000]}")
    payload = json.loads(raw[start:])
    results: list[dict] = []
    for block in payload:
        if not block.get("success", True):
            raise SystemExit(f"D1 query failed: {block}")
        results.extend(block.get("results", []))
    return results


def fetch_corrected_rows(*, d1_database_id: str) -> list[Frame]:
    """All reviewed frames whose label a human actually changed.

    Filters on ``corrected_label != label`` rather than ``in_dataset``: the review UI
    sets ``in_dataset`` even when a reviewer confirms an already-correct frame, and
    those no-op rows carry no learning signal.
    """
    rows = d1_query(
        "select id, date, camera, timestamp, image_key, label, corrected_label "
        "from sleep_frame "
        "where in_dataset = 1 and corrected_label is not null "
        "and corrected_label != label and image_key is not null "
        "order by date, timestamp",
        d1_database_id=d1_database_id,
    )
    frames = []
    for row in rows:
        label = str(row["corrected_label"])
        if label not in config.LABELS:
            raise SystemExit(
                f"frame {row['id']} has corrected_label={label!r}, which is not one of "
                f"{list(config.LABELS)}. A new class would renumber the label indices "
                f"and invalidate every existing checkpoint."
            )
        frames.append(
            Frame(
                frame_id=int(row["id"]),
                date=str(row["date"]),
                camera=str(row.get("camera") or ""),
                timestamp=str(row.get("timestamp") or ""),
                image_key=str(row["image_key"]),
                predicted_label=str(row["label"]),
                label=label,
            )
        )
    return frames


def download(
    frames: list[Frame], out_dir: Path, store: r2_store.R2Store, jobs: int
) -> list[Frame]:
    """Fetch each frame's JPEG via R2Store. Returns frames that landed on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(frame: Frame) -> Frame | None:
        # Flatten the key so the local filename is unique and traceable back to R2.
        dest = out_dir / f"{frame.date}__{frame.frame_id}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            return frame
        try:
            ok = store.download_raw(frame.image_key, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[fetch] WARN {frame.image_key}: {exc}", file=sys.stderr)
            return None
        if not ok:
            print(f"[fetch] WARN {frame.image_key}: not found", file=sys.stderr)
            return None
        return frame if dest.exists() and dest.stat().st_size > 0 else None

    got: list[Frame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(one, frames):
            if result is not None:
                got.append(result)
    return got


def _report(available: list[Frame], missing: list[Frame]) -> str:
    lines = ["## corrections", ""]
    by_date: dict[str, Counter] = {}
    for frame in available:
        by_date.setdefault(frame.date, Counter())[frame.label] += 1
    missing_by_date = Counter(frame.date for frame in missing)

    lines.append("| date | total | empty | exist | sleeping | note |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for date in sorted(set(by_date) | set(missing_by_date)):
        counts = by_date.get(date, Counter())
        total = sum(counts.values())
        note = ""
        if missing_by_date.get(date):
            note = f"{missing_by_date[date]} 件は R2 に画像なし（期限切れ削除）"
        lines.append(
            f"| {date} | {total} | {counts['empty']} | {counts['exist']} | "
            f"{counts['sleeping']} | {note} |"
        )
    lines.append("")

    flips = Counter((frame.predicted_label, frame.label) for frame in available)
    lines.append("### 誤りの内訳（モデル予測 → 人間の修正）")
    lines.append("")
    lines.append("| 予測 | 修正後 | 件数 |")
    lines.append("|---|---|---:|")
    for (predicted, corrected), count in flips.most_common():
        lines.append(f"| {predicted} | {corrected} | {count} |")
    lines.append("")
    lines.append(f"- 取得できた修正フレーム: **{len(available)}**")
    lines.append(f"- 画像が無く使えなかった修正: **{len(missing)}**")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="reconcile D1 against R2 and print the report without downloading",
    )
    args = parser.parse_args(argv)

    d1_database_id = config.require_env("D1_DATABASE_ID")

    print("[fetch] querying D1 for corrected frames...", flush=True)
    frames = fetch_corrected_rows(d1_database_id=d1_database_id)
    print(f"[fetch] D1 rows with a real correction: {len(frames)}", flush=True)

    store = r2_store.R2Store()
    store.verify()
    print(
        f"[fetch] listing R2 {store.bucket}/{config.R2_FRAME_PREFIX} ...",
        flush=True,
    )
    keys = store.list_bucket_keys(config.R2_FRAME_PREFIX)
    print(f"[fetch] R2 objects under the frame prefix: {len(keys)}", flush=True)

    present = [frame for frame in frames if frame.image_key in keys]
    missing = [frame for frame in frames if frame.image_key not in keys]

    report = _report(present, missing)
    print()
    print(report)

    if args.dry_run:
        return 0

    out_dir = config.CORRECTIONS_DIR
    images_dir = out_dir / "images"
    print(f"\n[fetch] downloading {len(present)} images -> {images_dir}", flush=True)
    got = download(present, images_dir, store, config.FETCH_JOBS)
    if len(got) != len(present):
        print(
            f"[fetch] WARN downloaded {len(got)}/{len(present)}; "
            f"the rest failed and are excluded",
            file=sys.stderr,
        )

    manifest = {
        "source": {
            "d1_database_id": d1_database_id,
            "r2_bucket": store.bucket,
            "filter": "in_dataset=1 AND corrected_label IS NOT NULL AND corrected_label != label",
        },
        "counts": {
            "d1_corrections": len(frames),
            "image_available": len(present),
            "image_missing": len(missing),
            "downloaded": len(got),
        },
        "missing_image_keys": [frame.image_key for frame in missing],
        "frames": [
            {
                "id": frame.frame_id,
                "date": frame.date,
                "camera": frame.camera,
                "timestamp": frame.timestamp,
                "image_key": frame.image_key,
                "file": f"images/{frame.date}__{frame.frame_id}.jpg",
                "predicted_label": frame.predicted_label,
                "label": frame.label,
            }
            for frame in got
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (out_dir / "report.md").write_text(report + "\n")
    print(f"[fetch] manifest: {manifest_path}")
    print(f"[fetch] report:   {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
