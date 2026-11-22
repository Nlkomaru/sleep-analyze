"""Shared constants and environment helpers for the cloud training pipeline.

Everything that two or more scripts must agree on lives here, so a rename or a
re-split cannot drift between the local dataset build and the cloud trainer.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Label contract
# --------------------------------------------------------------------------- #

# Index order is a hard contract with the deployed checkpoint. Never reorder, and
# never add a class without retraining from scratch: the head's output units are
# positional.
LABELS = ("empty", "exist", "sleeping")

# --------------------------------------------------------------------------- #
# Repo layout
# --------------------------------------------------------------------------- #

# train/src/config.py -> train/src -> train -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = REPO_ROOT / "train"
BUILD_DIR = TRAIN_ROOT / "build"
# Downloaded corrections land here (images + manifest), kept out of git.
CORRECTIONS_DIR = BUILD_DIR / "corrections"
# The assembled flat image pool that gets tarred and uploaded. Splits are NOT
# directories here — they are resolved by date from the manifest at load time.
DATASET_DIR = BUILD_DIR / "dataset"
MANIFEST_PATH = BUILD_DIR / "dataset_manifest.json"
RUNS_DIR = BUILD_DIR / "runs"
CACHE_FILE = BUILD_DIR / "base_image_cache.pt"
BEST_HPARAMS_PATH = BUILD_DIR / "best_hparams.json"
ENSEMBLE_DIR = BUILD_DIR / "ensemble"
BASELINE_CACHE_PATH = BUILD_DIR / "baseline_production.json"
# Existing hand-labeled pool in the repo root (gitignored).
LABELED_DIR = REPO_ROOT / "labeled"

# --------------------------------------------------------------------------- #
# Environment / secrets (os.environ / Colab Secrets first, then train/.env)
# --------------------------------------------------------------------------- #

_DOTENV_CACHE: dict[str, str] | None = None


def load_env(*, reload: bool = False) -> dict[str, str]:
    """Parse ``train/.env`` into a dict (cached).

    Environment variables (including Google Colab Secrets exported to
    ``os.environ``) always take precedence over ``train/.env`` in :func:`get_env`.
    """
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None and not reload:
        return _DOTENV_CACHE
    path = TRAIN_ROOT / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            if key:
                values[key] = value.strip().strip("'\"")
    _DOTENV_CACHE = values
    return values


def get_env(name: str, default: str = "") -> str:
    """Read ``name`` from ``os.environ`` first, then ``train/.env``."""
    env_val = os.environ.get(name, "").strip()
    if env_val:
        return env_val
    dotenv_val = load_env().get(name, "").strip()
    if dotenv_val:
        return dotenv_val
    return default


def require_env(name: str) -> str:
    """Return ``name`` from env/dotenv or exit with a clear configuration error."""
    value = get_env(name)
    if not value:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            f"Set it in os.environ (or Colab Secrets) or {TRAIN_ROOT / '.env'}."
        )
    return value


PRODUCTION_MODEL_PATH = Path(
    get_env("PRODUCTION_MODEL_PATH")
    or str(Path.home() / "data/models/sleep_classifier_ensemble.pt")
)

# --------------------------------------------------------------------------- #
# Image geometry
# --------------------------------------------------------------------------- #

IMAGE_SIZE = 224
BASE_CACHE_SIZE = 256

# The two geometries that exist in the labeled pool. Production serves FULL frames,
# while historical labels are mostly bed crops — which is why eval splits are scored
# on full frames only.
FULL_SIZE = (2304, 1296)
BED_SIZE = (1518, 540)
KNOWN_SIZES = (FULL_SIZE, BED_SIZE)

# (center_x, center_y, width, height) as ratios of the full frame. Copied from
# ``src/marimo/label_segments.py`` crop_presets["bed (1518x540)"] so newly fetched
# full frames can be cropped identically to the historical bed-crop labels.
BED_CROP_RATIOS = (0.315, 0.56, 0.659, 0.417)

VARIANT_FULL = "full"
VARIANT_BED = "bed"

# --------------------------------------------------------------------------- #
# Split definition
# --------------------------------------------------------------------------- #

# Pinned anchors. These never move, so scores stay comparable across months.
FIXED_TEST_LEGACY_DATES = (
    "2026-04-12",
    "2026-04-21",
    "2026-05-14",
    "2026-05-18",
    "2026-05-19",
)
FIXED_TEST_FULL_DATES = ("2026-06-23", "2026-06-27")
FIXED_VAL_JUNE_DATES = ("2026-06-24", "2026-06-26")

# Per-split filename glob. This MUST be per-split: test_legacy's files are untagged
# ``{date}__{NNN}.jpg`` and match nothing under ``*__full.jpg``, which would silently
# score an empty set and read as "no regression".
EVAL_PATTERNS = {
    "val": f"*__{VARIANT_FULL}.jpg",
    "test_full": f"*__{VARIANT_FULL}.jpg",
    "test_hard": f"*__{VARIANT_FULL}.jpg",
    "test_legacy": "*.jpg",
}
EVAL_SPLITS = tuple(EVAL_PATTERNS)
ALL_SPLITS = ("train", *EVAL_SPLITS)

# Training uses every variant (the bed crop acts as a strong zoom augmentation).
TRAIN_PATTERN = "*.jpg"

# Newest correction nights are held out as the regression check; keep adding older
# nights until the set is at least this big, so a thin final night is not a problem.
MIN_HARD_SIZE = 80
# How many correction nights join the val set (the rest go to train).
VAL_CORRECTION_NIGHTS = 1

# --------------------------------------------------------------------------- #
# Training / ensemble / R2 constants
# --------------------------------------------------------------------------- #

EPOCHS = 40
HPO_TRIALS = 100
HPO_SEED = 0
SAMPLER_SEED = 20260725
TOPK_KEEP = 8
HPO_MODEL_NAMES = (
    "convnext_small.in12k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "convnext_small.fb_in1k",
)
REFRESH_CONFIGS = 3
REFRESH_SEEDS = (101, 202, 303, 404, 505, 606)
ENSEMBLE_MAX_MEMBERS = 5
ENSEMBLE_BASELINE_K = 5
BUNDLE_DTYPE = "fp16"
KEEP_GENERATIONS = 2

R2_FRAME_PREFIX = "frames/"
FETCH_JOBS = 8
DEFAULT_R2_TRAIN_PREFIX = "train"
