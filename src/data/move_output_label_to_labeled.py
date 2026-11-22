from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


FROM_DIR = Path("output-label/result")
TO_DIR = Path("labeled")
LABELS = ("empty", "exist", "sleeping")
GLOB = "*.jpg"
ON_CONFLICT = "overwrite"


@dataclass(frozen=True)
class Stats:
    moved: int = 0
    skipped: int = 0
    overwritten: int = 0
    renamed: int = 0

    def __add__(self, other: "Stats") -> "Stats":
        return Stats(
            moved=self.moved + other.moved,
            skipped=self.skipped + other.skipped,
            overwritten=self.overwritten + other.overwritten,
            renamed=self.renamed + other.renamed,
        )


def _unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__dup{i:04d}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Failed to find unique path for: {p}")


def move_tree(
    *,
    src_label_dir: Path,
    dst_label_dir: Path,
    glob: str,
    dry_run: bool,
    on_conflict: str,
) -> Stats:
    if not src_label_dir.exists():
        return Stats()
    if not src_label_dir.is_dir():
        raise ValueError(f"Source is not a directory: {src_label_dir}")

    dst_label_dir.mkdir(parents=True, exist_ok=True)

    st = Stats()
    for src in sorted(src_label_dir.glob(glob)):
        if not src.is_file():
            continue
        dst = dst_label_dir / src.name

        if dst.exists():
            if on_conflict == "skip":
                st = st + Stats(skipped=1)
                continue
            if on_conflict == "overwrite":
                if not dry_run:
                    dst.unlink()
                st = st + Stats(overwritten=1)
            elif on_conflict == "rename":
                dst = _unique_path(dst)
                st = st + Stats(renamed=1)
            else:
                raise ValueError(f"Unknown on_conflict: {on_conflict}")

        if not dry_run:
            shutil.move(str(src), str(dst))
        st = st + Stats(moved=1)

    return st


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move labeled results from output-label/result into labeled/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without moving files.",
    )
    args = parser.parse_args(argv)

    src_root = FROM_DIR
    dst_root = TO_DIR

    if not src_root.exists():
        raise SystemExit(f"Source dir not found: {src_root}")
    if not src_root.is_dir():
        raise SystemExit(f"Source is not a directory: {src_root}")

    total = Stats()
    for label in LABELS:
        st = move_tree(
            src_label_dir=src_root / label,
            dst_label_dir=dst_root / label,
            glob=GLOB,
            dry_run=args.dry_run,
            on_conflict=ON_CONFLICT,
        )
        total = total + st

    if args.dry_run:
        print("[dry-run] done")

    print(
        f"moved={total.moved} skipped={total.skipped} "
        f"overwritten={total.overwritten} renamed={total.renamed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
