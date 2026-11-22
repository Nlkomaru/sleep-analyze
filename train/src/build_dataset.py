"""Assemble the training image pool: existing labels + freshly fetched corrections.

Produces a **flat pool** at ``train/build/dataset/{label}/{date}__{NNN}__{variant}.jpg``
plus ``dataset_manifest.json`` carrying the date -> split assignment. The splits are
not materialized as directories: resolving them at load time keeps one copy of every
image on disk, lets the decoded-image cache be shared across splits, and makes
rotating which nights validate a runtime argument instead of a dataset rebuild.

Three things this script exists to get right:

**Geometry.** Production serves uncropped full frames (2304x1296), but ~89% of the
historical labels are bed crops (1518x540). Training uses both — the bed crop acts as
a strong zoom augmentation — while every eval split is scored on full frames only, so
the reported numbers reflect what production actually sees. Newly fetched full frames
also get a bed crop generated from them, matching the historical ratios, so the
augmentation-like signal stays balanced.

**Split by night, never by file.** A bed crop and a full frame of the same moment are
near-duplicates. Splitting per file would leak; every variant of a date shares a split.

**A rule, not a date list.** This runs monthly, so the newest correction nights become
the held-out regression check and older ones flow into training. Freezing a date list
would let the hard-example split go stale while new nights piled into train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import config


@dataclass(frozen=True)
class Item:
    """One image destined for the pool."""

    source: Path
    date: str
    label: str
    variant: str  # "full" | "bed"
    origin: str  # "labeled" | "correction"
    # Set when the file must be produced by cropping a full frame.
    crop_from: Path | None = None


def _variant_from_size(size: tuple[int, int], path: Path) -> str:
    if size == config.FULL_SIZE:
        return config.VARIANT_FULL
    if size == config.BED_SIZE:
        return config.VARIANT_BED
    raise SystemExit(
        f"{path}: unexpected geometry {size}. Known: {list(config.KNOWN_SIZES)}. "
        f"An unknown crop would train and score on a different field of view than "
        f"production."
    )


def collect_labeled(labeled_dir: Path) -> list[Item]:
    """The existing hand-labeled pool.

    The April/May files are untagged (``2026-04-10__001.jpg``) but are bed crops; the
    variant is taken from the pixel size rather than trusted from the filename.
    """
    items: list[Item] = []
    for label in config.LABELS:
        label_dir = labeled_dir / label
        if not label_dir.is_dir():
            raise SystemExit(f"missing label directory: {label_dir}")
        for path in sorted(label_dir.glob("*.jpg")):
            date = path.stem.split("__", 1)[0]
            with Image.open(path) as image:
                variant = _variant_from_size(image.size, path)
            items.append(
                Item(
                    source=path,
                    date=date,
                    label=label,
                    variant=variant,
                    origin="labeled",
                )
            )
    return items


def collect_corrections(corrections_dir: Path, *, make_bed_crops: bool) -> list[Item]:
    manifest_path = corrections_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — run fetch_corrections.py first.")
    manifest = json.loads(manifest_path.read_text())
    items: list[Item] = []
    for entry in manifest["frames"]:
        path = corrections_dir / entry["file"]
        if not path.exists():
            continue
        with Image.open(path) as image:
            size = image.size
        if _variant_from_size(size, path) != config.VARIANT_FULL:
            raise SystemExit(
                f"{path}: corrections are expected to be full frames "
                f"({config.FULL_SIZE}), got {size}"
            )
        items.append(
            Item(
                source=path,
                date=entry["date"],
                label=entry["label"],
                variant=config.VARIANT_FULL,
                origin="correction",
            )
        )
        if make_bed_crops:
            items.append(
                Item(
                    source=path,
                    date=entry["date"],
                    label=entry["label"],
                    variant=config.VARIANT_BED,
                    origin="correction",
                    crop_from=path,
                )
            )
    return items


def plan_splits(
    *,
    labeled_dates: set[str],
    correction_full_counts: dict[str, int],
    min_hard_size: int,
    val_correction_nights: int,
) -> dict[str, str]:
    """Resolve every date to a split. Returns ``{date: split}``."""
    assignment: dict[str, str] = {}
    for date in config.FIXED_TEST_LEGACY_DATES:
        assignment[date] = "test_legacy"
    for date in config.FIXED_TEST_FULL_DATES:
        assignment[date] = "test_full"
    for date in config.FIXED_VAL_JUNE_DATES:
        assignment[date] = "val"

    pinned = set(assignment)
    missing = sorted(pinned - labeled_dates)
    if missing:
        raise SystemExit(
            f"pinned split dates absent from the labeled pool: {missing}. These anchor "
            f"the comparable scores and cannot be silently dropped."
        )

    # Newest correction nights first: fill test_hard until it is big enough, then take
    # the val night(s), then everything older trains.
    remaining = sorted(
        (d for d in correction_full_counts if d not in pinned), reverse=True
    )
    hard_size = 0
    index = 0
    while index < len(remaining) and hard_size < min_hard_size:
        assignment[remaining[index]] = "test_hard"
        hard_size += correction_full_counts[remaining[index]]
        index += 1
    if hard_size < min_hard_size:
        raise SystemExit(
            f"test_hard would only have {hard_size} scored images (< {min_hard_size}). "
            f"Review more frames, or lower --min-hard-size deliberately."
        )
    for _ in range(val_correction_nights):
        if index >= len(remaining):
            break
        assignment[remaining[index]] = "val"
        index += 1
    for date in remaining[index:]:
        assignment[date] = "train"

    for date in sorted(labeled_dates - set(assignment)):
        assignment[date] = "train"
    return assignment


def _crop_bed(image: Image.Image) -> Image.Image:
    cx, cy, cw, ch = config.BED_CROP_RATIOS
    width, height = image.size
    box_w, box_h = cw * width, ch * height
    left = round(cx * width - box_w / 2)
    top = round(cy * height - box_h / 2)
    return image.crop((left, top, left + round(box_w), top + round(box_h)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(items: list[Item], out_dir: Path) -> list[dict]:
    """Write the flat pool. Returns one record per image."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for label in config.LABELS:
        (out_dir / label).mkdir(parents=True, exist_ok=True)

    counter: dict[tuple[str, str], int] = defaultdict(int)
    records: list[dict] = []
    for item in sorted(items, key=lambda i: (i.date, i.variant, str(i.source))):
        counter[(item.date, item.variant)] += 1
        name = (
            f"{item.date}__{counter[(item.date, item.variant)]:04d}__{item.variant}.jpg"
        )
        dest = out_dir / item.label / name
        if item.crop_from is not None:
            with Image.open(item.crop_from) as image:
                _crop_bed(image.convert("RGB")).save(dest, quality=92)
        else:
            shutil.copy2(item.source, dest)
        records.append(
            {
                "label": item.label,
                "date": item.date,
                "variant": item.variant,
                "origin": item.origin,
                "file": f"{item.label}/{name}",
                "source": str(item.source),
                "size": dest.stat().st_size,
                "sha256": _sha256(dest),
            }
        )
    return records


def summarize(records: list[dict], assignment: dict[str, str]) -> dict:
    per_split: dict[str, Counter] = defaultdict(Counter)
    per_variant: dict[str, Counter] = defaultdict(Counter)
    scored: dict[str, Counter] = defaultdict(Counter)
    origins: Counter = Counter()
    # Per-date facts the fold rotation needs: whether a night has full frames at all,
    # and whether it came from the correction stream (hard examples) or normal labeling.
    date_info: dict[str, dict] = {}
    for record in records:
        split = assignment[record["date"]]
        per_split[split][record["label"]] += 1
        per_variant[split][record["variant"]] += 1
        origins[record["origin"]] += 1
        info = date_info.setdefault(
            record["date"], {"full": 0, "bed": 0, "origins": [], "split": split}
        )
        info[record["variant"]] += 1
        if record["origin"] not in info["origins"]:
            info["origins"].append(record["origin"])
        pattern = config.EVAL_PATTERNS.get(split)
        # Mirror the loader's filter: only ``*__full.jpg`` counts for the full-frame
        # splits, everything counts for test_legacy and train.
        if pattern is None or Path(record["file"]).match(pattern):
            scored[split][record["label"]] += 1
    return {
        "labels": list(config.LABELS),
        "split_assignment": assignment,
        "split_dates": {
            split: sorted(d for d, s in assignment.items() if s == split)
            for split in config.ALL_SPLITS
        },
        "split_counts": {s: dict(per_split[s]) for s in config.ALL_SPLITS},
        "split_variants": {s: dict(per_variant[s]) for s in config.ALL_SPLITS},
        "scored_counts": {s: dict(scored[s]) for s in config.ALL_SPLITS},
        "date_info": date_info,
        "eval_patterns": dict(config.EVAL_PATTERNS),
        "train_pattern": config.TRAIN_PATTERN,
        "origins": dict(origins),
        "total": len(records),
    }


def run_gates(
    summary: dict, records: list[dict], out_dir: Path, *, min_hard_size: int
) -> list[str]:
    """Every check that must pass before spending money on a GPU."""
    failures: list[str] = []

    # 1. Label directories and index order.
    present = sorted(d.name for d in out_dir.iterdir() if d.is_dir())
    if present != sorted(config.LABELS):
        failures.append(f"pool class dirs {present} != {sorted(config.LABELS)}")
    if sorted(config.LABELS) != list(config.LABELS):
        failures.append(
            f"LABELS {config.LABELS} is not in alphabetical order; any loader that "
            f"sorts directory names would disagree with the checkpoint's label order"
        )

    # 2. Every eval split resolves to a NON-EMPTY scored set under its own glob. This
    #    is the check that catches applying '*__full.jpg' to the untagged legacy split.
    for split, pattern in config.EVAL_PATTERNS.items():
        total = sum(summary["scored_counts"][split].values())
        if total == 0:
            failures.append(
                f"{split}: zero images match its pattern {pattern!r} — the score would "
                f"be meaningless rather than absent"
            )
        elif split == "test_hard" and total < min_hard_size:
            failures.append(
                f"test_hard: only {total} scored images (< {min_hard_size})"
            )

    # 3. Every date resolves to exactly one split (the loader assumes this).
    for split in config.ALL_SPLITS:
        for date in summary["split_dates"][split]:
            if summary["split_assignment"][date] != split:
                failures.append(f"date {date} assigned inconsistently")

    # 4. No identical file content across splits (bed/full near-duplicate leak, or a
    #    genuinely duplicated source image).
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record["file"])
    for digest, files in by_hash.items():
        splits = {
            summary["split_assignment"][Path(f).stem.split("__", 1)[0]] for f in files
        }
        if len(splits) > 1:
            failures.append(
                f"identical content in multiple splits ({sorted(splits)}): "
                f"{files[:3]} (sha256 {digest[:12]})"
            )

    # 5. Train must carry all three classes, or the class weights are undefined.
    for label in config.LABELS:
        if not summary["split_counts"]["train"].get(label):
            failures.append(f"train has no {label!r} examples")

    # 6. Filenames must be unique across the pool: the cache key is
    #    "<label>/<name>:<size>", so a collision would alias two images.
    names = Counter(record["file"] for record in records)
    collisions = [name for name, count in names.items() if count > 1]
    if collisions:
        failures.append(f"duplicate pool filenames: {collisions[:5]}")
    return failures


def render_report(summary: dict, *, failures: list[str]) -> str:
    lines = ["# dataset build report", ""]
    lines.append(
        f"- 総枚数: **{summary['total']}**（フラット pool、split はマニフェストで指定）"
    )
    origins = summary["origins"]
    lines.append(
        f"- 出典: labeled={origins.get('labeled', 0)} / correction={origins.get('correction', 0)}"
    )
    lines.append("")
    lines.append("## split 別")
    lines.append("")
    lines.append(
        "| split | empty | exist | sleeping | full | bed | 採点glob | 採点枚数 | 日付数 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for split in config.ALL_SPLITS:
        counts = summary["split_counts"][split]
        variants = summary["split_variants"][split]
        pattern = summary["eval_patterns"].get(split, summary["train_pattern"])
        scored = sum(summary["scored_counts"][split].values())
        lines.append(
            f"| {split} | {counts.get('empty', 0)} | {counts.get('exist', 0)} | "
            f"{counts.get('sleeping', 0)} | {variants.get('full', 0)} | "
            f"{variants.get('bed', 0)} | `{pattern}` | {scored} | "
            f"{len(summary['split_dates'][split])} |"
        )
    lines.append("")
    lines.append("## 採点対象のクラス内訳（eval split）")
    lines.append("")
    lines.append("| split | empty | exist | sleeping |")
    lines.append("|---|---:|---:|---:|")
    for split in config.EVAL_SPLITS:
        scored = summary["scored_counts"][split]
        lines.append(
            f"| {split} | {scored.get('empty', 0)} | {scored.get('exist', 0)} | "
            f"{scored.get('sleeping', 0)} |"
        )
    lines.append("")
    lines.append("## 解決された日付割り当て")
    lines.append("")
    for split in config.ALL_SPLITS:
        dates = summary["split_dates"][split]
        lines.append(f"- **{split}** ({len(dates)}日): {', '.join(dates) or '(なし)'}")
    lines.append("")
    if failures:
        lines.append("## ❌ hard gate 失敗")
        lines.append("")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("## ✅ hard gate 全通過")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-hard-size",
        type=int,
        default=config.MIN_HARD_SIZE,
        help="grow test_hard from the newest night backwards until it reaches this",
    )
    args = parser.parse_args(argv)

    out_dir = config.DATASET_DIR
    manifest_path = config.MANIFEST_PATH
    report_path = config.BUILD_DIR / "report.md"

    labeled = collect_labeled(config.LABELED_DIR)
    corrections = collect_corrections(config.CORRECTIONS_DIR, make_bed_crops=True)
    print(f"[build] labeled={len(labeled)} corrections={len(corrections)}", flush=True)

    # Size the hard split on scored (full-frame) images only.
    correction_full_counts = Counter(
        item.date for item in corrections if item.variant == config.VARIANT_FULL
    )
    assignment = plan_splits(
        labeled_dates={item.date for item in labeled},
        correction_full_counts=dict(correction_full_counts),
        min_hard_size=args.min_hard_size,
        val_correction_nights=config.VAL_CORRECTION_NIGHTS,
    )

    records = materialize(labeled + corrections, out_dir)
    summary = summarize(records, assignment)
    failures = run_gates(summary, records, out_dir, min_hard_size=args.min_hard_size)

    report = render_report(summary, failures=failures)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)

    summary["dataset_hash"] = hashlib.sha256(
        json.dumps(
            sorted((r["file"], r["sha256"]) for r in records), sort_keys=True
        ).encode()
    ).hexdigest()[:12]
    summary["items"] = records
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[build] manifest: {manifest_path}")
    print(f"[build] dataset_hash: {summary['dataset_hash']}")

    if failures:
        print(f"\n[build] {len(failures)} hard gate(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
