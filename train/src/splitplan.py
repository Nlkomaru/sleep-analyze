"""Turn the dataset manifest into concrete train/eval splits, optionally rotated.

Splits live here rather than in the directory tree so that *which nights validate* is
a runtime argument. Two consumers want different things:

- ``--mode hpo`` uses the manifest's assignment verbatim. Hyperparameters are not that
  sensitive to the exact validation nights, and a single fixed split keeps trials
  comparable to each other, which is the whole point of a search.
- ``--mode refresh`` rotates the validation nights per fold. The nine ensemble
  candidates then differ by *both* seed and validation split, which is where the
  diversity that makes an ensemble worth building actually comes from. Rotating costs
  no extra GPU time — the same nine runs happen either way.

The three scoring anchors — ``test_full``, ``test_hard``, ``test_legacy`` — never move.
They are what makes numbers comparable across folds, across runs and across months.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from trainer import SplitSpec

# Splits that are never rotated.
FROZEN_SPLITS = ("test_full", "test_hard", "test_legacy")


@dataclass(frozen=True)
class ResolvedSplits:
    train: SplitSpec
    evals: dict[str, SplitSpec]
    fold: int | None
    # Human-readable note for the run config and the Discord summary.
    description: str

    def date_counts(self) -> dict[str, int]:
        return {"train": len(self.train.dates)} | {
            name: len(spec.dates) for name, spec in self.evals.items()
        }


def _rotate(pool: list[str], fold: int, take: int) -> list[str]:
    """Pick ``take`` items starting at ``fold * take``, wrapping around."""
    if not pool or take <= 0:
        return []
    take = min(take, len(pool))
    start = (fold * take) % len(pool)
    return [pool[(start + offset) % len(pool)] for offset in range(take)]


def resolve(
    manifest: dict,
    *,
    fold: int | None = None,
    val_june_nights: int = 2,
    val_correction_nights: int = 1,
) -> ResolvedSplits:
    """Build the split specs. ``fold=None`` uses the manifest assignment as-is."""
    assignment: dict[str, str] = manifest["split_assignment"]
    patterns = manifest.get("eval_patterns", config.EVAL_PATTERNS)
    train_pattern = manifest.get("train_pattern", config.TRAIN_PATTERN)

    if fold is None:
        by_split: dict[str, list[str]] = {}
        for date, split in assignment.items():
            by_split.setdefault(split, []).append(date)
        evals = {
            name: SplitSpec(tuple(sorted(by_split.get(name, ()))), patterns[name])
            for name in config.EVAL_SPLITS
        }
        return ResolvedSplits(
            train=SplitSpec(tuple(sorted(by_split.get("train", ()))), train_pattern),
            evals=evals,
            fold=None,
            description="manifest split (fold rotation off)",
        )

    if "date_info" not in manifest:
        raise SystemExit(
            "dataset manifest has no 'date_info' block, so fold rotation cannot tell "
            "which nights have full frames. Re-run build_dataset.py to regenerate it."
        )
    date_info: dict[str, dict] = manifest["date_info"]
    frozen = {date for date, split in assignment.items() if split in FROZEN_SPLITS}

    # A night can only validate if it actually has full frames — the val split is
    # scored on ``*__full.jpg`` and would otherwise resolve to nothing.
    candidates = [
        date
        for date in sorted(assignment)
        if date not in frozen and date_info.get(date, {}).get("full", 0) > 0
    ]
    # Keep the composition of the validation set stable across folds: some ordinary
    # nights plus some correction nights. A val set made only of model mistakes would
    # select for a distribution production does not have.
    ordinary = [d for d in candidates if "correction" not in date_info[d]["origins"]]
    corrections = sorted(
        (d for d in candidates if "correction" in date_info[d]["origins"]), reverse=True
    )
    val = sorted(
        _rotate(ordinary, fold, val_june_nights)
        + _rotate(corrections, fold, val_correction_nights)
    )
    if not val:
        raise SystemExit(
            "fold rotation produced an empty val split: no non-frozen night has full "
            "frames. Check the dataset manifest."
        )

    evals = {"val": SplitSpec(tuple(val), patterns["val"])}
    for name in FROZEN_SPLITS:
        dates = tuple(sorted(d for d, s in assignment.items() if s == name))
        evals[name] = SplitSpec(dates, patterns[name])

    val_set = set(val)
    train = tuple(sorted(d for d in assignment if d not in frozen and d not in val_set))
    return ResolvedSplits(
        train=SplitSpec(train, train_pattern),
        evals=evals,
        fold=fold,
        description=f"fold {fold}: val={val}",
    )


def shrink(splits: ResolvedSplits, keep_dates: int) -> ResolvedSplits:
    """Keep only the ``keep_dates`` most recent nights per split.

    Smoke tests only. Exercises the exact same code path in seconds instead of the
    ~40 minutes a real run needs, so the wiring can be checked before paying for a GPU.
    """

    def trim(spec: SplitSpec) -> SplitSpec:
        return SplitSpec(tuple(sorted(spec.dates)[-keep_dates:]), spec.pattern)

    return ResolvedSplits(
        train=trim(splits.train),
        evals={name: trim(spec) for name, spec in splits.evals.items()},
        fold=splits.fold,
        description=f"{splits.description} [shrunk to {keep_dates} date(s)/split]",
    )


def describe(splits: ResolvedSplits) -> str:
    lines = [splits.description]
    for name, spec in splits.evals.items():
        lines.append(f"  {name}: {len(spec.dates)}日 {spec.pattern} {list(spec.dates)}")
    lines.append(f"  train: {len(splits.train.dates)}日")
    return "\n".join(lines)
