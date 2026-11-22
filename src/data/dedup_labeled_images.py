"""Remove duplicate labeled images. Three strategies as subcommands.

# 連続フレームの近傍重複（知覚的、推奨）。確認だけなら --delete を外す。
uv run python src/data/dedup_labeled_images.py near --threshold 0.95 --delete

# 完全一致（内容ハッシュ）の重複を一覧表示する。
uv run python src/data/dedup_labeled_images.py exact

# ファイル名マーカー (*__dup*.jpg) を削除する。
uv run python src/data/dedup_labeled_images.py marker --delete
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


LABELS = ("empty", "exist", "sleeping")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LABELED_DIR = Path("labeled")
COMPARISON_SIZE = 64
REPORT_PATH = Path("near_duplicate_labeled_images.csv")
EXACT_GLOB = "**/*"
HASH_ALGO = "sha256"
CHUNK_SIZE = 1024 * 1024
MARKER_PATTERN = "*__dup*.jpg"
# date__index[__variant] 形式。variant は画角タグ ("full" / "bed" 等、省略可)。
SEQUENCE_PATTERN = re.compile(r"^(?P<date>.+?)__(?P<index>\d+)(?:__(?P<variant>.+))?$")


# --------------------------------------------------------------------------- #
# near: 連続フレームの知覚的な近傍重複
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabeledImage:
    path: Path
    label: str
    date: str
    index: int
    variant: str


@dataclass(frozen=True)
class DuplicateCandidate:
    kept: LabeledImage
    removed: LabeledImage
    mean_absolute_difference: float


def _parse_image(path: Path, label: str) -> LabeledImage | None:
    match = SEQUENCE_PATTERN.match(path.stem)
    if match is None:
        return None
    return LabeledImage(
        path=path,
        label=label,
        date=match.group("date"),
        index=int(match.group("index")),
        variant=match.group("variant") or "",
    )


def _load_comparison_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L")
        resized = grayscale.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)


def _difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def find_near_duplicates(
    *,
    root: Path,
    threshold: float,
    comparison_size: int,
) -> tuple[list[DuplicateCandidate], int, int]:
    # 画角(variant)ごとに分けて重複判定する。タグを無視して date だけで束ねると、
    # 同じ瞬間の bed(寄り) と full(引き) が隣接比較され、追加した full フレームが
    # 重複として誤削除されうるため。
    images_by_group: dict[tuple[str, str], list[LabeledImage]] = {}
    skipped_names = 0

    for label in LABELS:
        label_dir = root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            item = _parse_image(path, label)
            if item is None:
                skipped_names += 1
                continue
            images_by_group.setdefault((item.date, item.variant), []).append(item)

    candidates: list[DuplicateCandidate] = []
    scanned = 0
    for date_images in images_by_group.values():
        date_images.sort(key=lambda item: (item.index, item.path.name))
        last_kept: LabeledImage | None = None
        last_kept_pixels: np.ndarray | None = None

        for item in date_images:
            scanned += 1
            pixels = _load_comparison_image(item.path, comparison_size)
            if last_kept is None or item.label != last_kept.label:
                last_kept = item
                last_kept_pixels = pixels
                continue

            assert last_kept_pixels is not None
            difference = _difference(last_kept_pixels, pixels)
            if difference <= threshold:
                candidates.append(
                    DuplicateCandidate(
                        kept=last_kept,
                        removed=item,
                        mean_absolute_difference=difference,
                    )
                )
                continue

            last_kept = item
            last_kept_pixels = pixels

    return candidates, scanned, skipped_names


def _write_report(path: Path, candidates: list[DuplicateCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["date", "label", "kept", "removed", "mean_absolute_difference"]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.removed.date,
                    candidate.removed.label,
                    candidate.kept.path,
                    candidate.removed.path,
                    f"{candidate.mean_absolute_difference:.6f}",
                ]
            )


def run_near(args: argparse.Namespace) -> int:
    root = LABELED_DIR
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    if args.threshold < 0:
        raise SystemExit("--threshold must be non-negative")

    candidates, scanned, skipped_names = find_near_duplicates(
        root=root,
        threshold=args.threshold,
        comparison_size=COMPARISON_SIZE,
    )
    _write_report(REPORT_PATH, candidates)

    deleted = 0
    if args.delete:
        for candidate in candidates:
            candidate.removed.path.unlink()
            deleted += 1

    for candidate in candidates[:20]:
        print(
            f"difference={candidate.mean_absolute_difference:.4f} "
            f"keep={candidate.kept.path} remove={candidate.removed.path}"
        )
    if len(candidates) > 20:
        print(f"... {len(candidates) - 20} more candidates; see {REPORT_PATH}")

    print(
        f"scanned={scanned} candidates={len(candidates)} deleted={deleted} "
        f"skipped_names={skipped_names} threshold={args.threshold:.4f} "
        f"report={REPORT_PATH}"
    )
    return 0


# --------------------------------------------------------------------------- #
# exact: 内容ハッシュによる完全一致重複
# --------------------------------------------------------------------------- #
def _hash_file(path: Path, algo: str, chunk_size: int) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_exact(args: argparse.Namespace) -> int:
    del args
    root = LABELED_DIR
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    groups: dict[str, list[Path]] = defaultdict(list)
    files_scanned = 0
    bytes_scanned = 0

    for p in sorted(root.glob(EXACT_GLOB)):
        if not p.is_file():
            continue
        files_scanned += 1
        try:
            bytes_scanned += p.stat().st_size
        except OSError:
            pass
        digest = _hash_file(p, algo=HASH_ALGO, chunk_size=CHUNK_SIZE)
        groups[digest].append(p)

    items = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shown_groups = 0
    shown_files = 0

    for digest, paths in items:
        if len(paths) <= 1:
            continue
        shown_groups += 1
        shown_files += len(paths)
        print(f"{digest}  count={len(paths)}")
        for p in sorted(paths):
            try:
                print(f"  {p.relative_to(root)}")
            except ValueError:
                print(f"  {p}")

    dup_groups = sum(1 for ps in groups.values() if len(ps) > 1)
    dup_files = sum(len(ps) for ps in groups.values() if len(ps) > 1)
    print(
        f"scanned_files={files_scanned} scanned_bytes={bytes_scanned} "
        f"unique_hashes={len(groups)} dup_groups={dup_groups} dup_files={dup_files} "
        f"shown_groups={shown_groups} shown_files={shown_files}"
    )
    return 0


# --------------------------------------------------------------------------- #
# marker: ファイル名マーカー (*__dup*.jpg) の削除
# --------------------------------------------------------------------------- #
def run_marker(args: argparse.Namespace) -> int:
    root = LABELED_DIR
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    paths = sorted(root.rglob(MARKER_PATTERN))
    for p in paths:
        if args.delete:
            p.unlink(missing_ok=True)
        else:
            print(p)

    print(f"deleted={len(paths) if args.delete else 0} matched={len(paths)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_near = sub.add_parser(
        "near", help="連続フレームの知覚的な近傍重複を検出・削除する（推奨）。"
    )
    p_near.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help=(
            "0-255 スケールの平均絶対グレースケール差の上限（default: 1.0）。"
            "大きいほど多く削除する。"
        ),
    )
    p_near.add_argument(
        "--delete",
        action="store_true",
        help="候補を削除する。付けなければレポートのみ。",
    )
    p_near.set_defaults(func=run_near)

    p_exact = sub.add_parser(
        "exact", help="内容ハッシュで完全一致の重複を一覧表示する。"
    )
    p_exact.set_defaults(func=run_exact)

    p_marker = sub.add_parser(
        "marker", help="ファイル名マーカー (*__dup*.jpg) を削除する。"
    )
    p_marker.add_argument(
        "--delete", action="store_true", help="削除する。付けなければ一覧表示のみ。"
    )
    p_marker.set_defaults(func=run_marker)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
