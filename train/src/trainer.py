"""Self-contained trainer for the cloud training pipeline.

Kept self-contained so the cloud training loop can vary hyperparameters and resolve
splits from a flat pool without touching production inference code. The augmentation
and caching logic is preserved, with:

- every hyperparameter living in :class:`HParams` instead of module constants, so a
  search can vary it;
- class indices pinned to ``LABELS`` order rather than inherited from
  ``ImageFolder``'s sorted-directory-name behaviour;
- eval splits carrying a per-split filename pattern, because the full-frame splits are
  scored on ``*__full.jpg`` while the legacy bed-crop split has untagged filenames;
- the best epoch's softmax probabilities written out, so ensemble selection can
  run offline on CPU instead of re-running inference on a GPU.

The checkpoint schema is identical to the production contract, so a bundle built from
these runs is a drop-in replacement for ``sleep_classifier_ensemble.pt``.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Index order is a hard contract with the deployed checkpoint. Never reorder.
LABELS = ("empty", "exist", "sleeping")

IMAGE_SIZE = 224
# Side length of the fixed square uint8 image kept in the cache. Augmentation
# (random-resized-crop etc.) runs on the GPU per batch on top of this base image,
# so it just needs some margin above IMAGE_SIZE to crop/zoom from.
BASE_CACHE_SIZE = 256
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
# Luminance weights (ITU-R BT.601) for grayscale / saturation on the GPU.
GRAY_WEIGHTS = (0.299, 0.587, 0.114)

# In-process tensor cache shared across runs (e.g. across sweep trials).
GLOBAL_TENSOR_CACHE: dict[str, torch.Tensor] = {}


@dataclass(frozen=True)
class HParams:
    """Everything the hyperparameter search is allowed to vary."""

    model_name: str = "convnext_small.in12k_ft_in1k"
    image_size: int = IMAGE_SIZE
    batch_size: int = 128
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    patience: int = 5
    label_smoothing: float = 0.0
    # ``None`` means "don't pass it to timm" — not every backbone accepts it.
    drop_path_rate: float | None = None
    # Augmentation (train only).
    rrc_scale_min: float = 0.3
    rrc_scale_max: float = 1.0
    rrc_ratio_min: float = 0.5
    rrc_ratio_max: float = 2.0
    rotation_deg: float = 8.0
    jitter_brightness: float = 0.4
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.3
    grayscale_p: float = 0.1

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> HParams:
        """Build from a dict, rejecting unknown keys loudly.

        A typo in a search space or in ``best_hparams.json`` must not silently fall
        back to a default — that would make a run untraceable to its config.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown hparams: {unknown} (known: {sorted(known)})")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    # confusion[true][pred]. For the hard-example split this, not macro-F1, is the
    # number that answers "did the sleeping->empty mistakes go away".
    confusion: list[list[int]] = field(default_factory=list)
    # Softmax probabilities [N, len(LABELS)] plus matching targets, kept only when
    # requested (eval at the best epoch) so ensemble selection can run on CPU.
    probs: np.ndarray | None = None
    targets: np.ndarray | None = None

    def to_json(self) -> dict[str, Any]:
        """JSON-safe view (drops the numpy arrays)."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "confusion": self.confusion,
            "support": [sum(row) for row in self.confusion],
        }


# --------------------------------------------------------------------------- #
# GPU augmentation
# --------------------------------------------------------------------------- #


# Built once per device: this is called three times per augmented batch, and
# rebuilding it from a Python tuple each time is a pageable host->device copy.
_GRAY_WEIGHTS_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def _grayscale(x: torch.Tensor) -> torch.Tensor:
    """Per-pixel luminance of a float batch [B,3,H,W] -> [B,1,H,W]."""
    weights = _GRAY_WEIGHTS_BY_DEVICE.get(x.device)
    if weights is None:
        weights = torch.tensor(GRAY_WEIGHTS, device=x.device).view(1, 3, 1, 1)
        _GRAY_WEIGHTS_BY_DEVICE[x.device] = weights
    return (x * weights).sum(dim=1, keepdim=True)


def _uniform(n: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(low, high)


def _gpu_random_resized_crop(
    x: torch.Tensor,
    out_size: int,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rotation_deg: float,
) -> torch.Tensor:
    """Per-sample random-resized-crop + flip + small rotation via one affine warp.

    ``affine_grid`` + ``grid_sample`` so every image in the batch gets independent
    random parameters in a single GPU op. ``theta`` maps output normalized coords ->
    input normalized coords, so a scale < 1 samples a sub-region (zoom in); a
    negative x-scale flips horizontally.
    """
    b = x.shape[0]
    device = x.device
    area = _uniform(b, scale[0], scale[1], device)
    log_ratio = _uniform(b, math.log(ratio[0]), math.log(ratio[1]), device)
    aspect = torch.exp(log_ratio)
    w = torch.sqrt(area * aspect).clamp(0.05, 1.0)
    h = torch.sqrt(area / aspect).clamp(0.05, 1.0)
    flip = torch.where(torch.rand(b, device=device) < 0.5, -1.0, 1.0)
    angle = _uniform(b, -rotation_deg, rotation_deg, device) * (math.pi / 180.0)
    tx = (1.0 - w) * _uniform(b, -1.0, 1.0, device)
    ty = (1.0 - h) * _uniform(b, -1.0, 1.0, device)
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    theta = torch.stack(
        [
            torch.stack([cos * w * flip, -sin * h, tx], dim=1),
            torch.stack([sin * w * flip, cos * h, ty], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(
        theta, (b, x.shape[1], out_size, out_size), align_corners=False
    )
    return F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )


def _gpu_color_jitter(
    x: torch.Tensor, *, brightness: float, contrast: float, saturation: float
) -> torch.Tensor:
    """Per-sample brightness/contrast/saturation jitter on a float batch in [0,1]."""
    b = x.shape[0]
    device = x.device
    x = (
        x * _uniform(b, 1 - brightness, 1 + brightness, device).view(b, 1, 1, 1)
    ).clamp(0.0, 1.0)
    mean = _grayscale(x).mean(dim=(2, 3), keepdim=True)
    x = (
        (x - mean) * _uniform(b, 1 - contrast, 1 + contrast, device).view(b, 1, 1, 1)
        + mean
    ).clamp(0.0, 1.0)
    gray = _grayscale(x)
    x = (
        gray
        + (x - gray)
        * _uniform(b, 1 - saturation, 1 + saturation, device).view(b, 1, 1, 1)
    ).clamp(0.0, 1.0)
    return x


def _gpu_random_grayscale(x: torch.Tensor, p: float) -> torch.Tensor:
    b = x.shape[0]
    gray = _grayscale(x).expand(-1, 3, -1, -1)
    mask = (torch.rand(b, device=x.device) < p).view(b, 1, 1, 1)
    return torch.where(mask, gray, x)


class GpuTransform:
    """Batched GPU transform applied inside the training loop.

    The dataset yields fixed-size uint8 images; this turns a uint8 batch on the
    device into a normalized float batch ready for the model. Training applies
    random augmentation per sample; eval is deterministic and mirrors the production
    inference pipeline (resize-to-square + ImageNet normalization).
    """

    def __init__(
        self, device: torch.device, *, train: bool, hp: HParams | None = None
    ) -> None:
        self.train = train
        self.hp = hp or HParams()
        self.mean = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        hp = self.hp
        size = hp.image_size
        x = x_uint8.to(dtype=torch.float32).div_(255.0)
        if self.train:
            x = _gpu_random_resized_crop(
                x,
                size,
                scale=(hp.rrc_scale_min, hp.rrc_scale_max),
                ratio=(hp.rrc_ratio_min, hp.rrc_ratio_max),
                rotation_deg=hp.rotation_deg,
            )
            x = _gpu_color_jitter(
                x,
                brightness=hp.jitter_brightness,
                contrast=hp.jitter_contrast,
                saturation=hp.jitter_saturation,
            )
            x = _gpu_random_grayscale(x, hp.grayscale_p)
        elif x.shape[-1] != size or x.shape[-2] != size:
            x = F.interpolate(
                x, size=(size, size), mode="bilinear", align_corners=False
            )
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _to_base_tensor(image: Image.Image) -> torch.Tensor:
    """Decode a PIL image to a fixed ``BASE_CACHE_SIZE`` square uint8 CHW tensor."""
    image = image.convert("RGB").resize(
        (BASE_CACHE_SIZE, BASE_CACHE_SIZE), Image.BILINEAR
    )
    array = np.array(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def date_of(path: Path | str) -> str:
    """``2026-07-21__0007__full.jpg`` -> ``2026-07-21``."""
    return Path(path).stem.split("__", 1)[0]


@dataclass(frozen=True)
class SplitSpec:
    """Which images make up one split.

    Splits are resolved at load time from a flat image pool rather than baked into a
    directory tree. That keeps one copy of every image on disk, lets the decoded-image
    cache be shared across splits, and — the reason it matters here — makes rotating
    which nights are validation a runtime argument instead of a dataset rebuild.
    """

    dates: tuple[str, ...]
    pattern: str = "*.jpg"

    def date_set(self) -> set[str]:
        return set(self.dates)


class LabeledImageFolder(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """Flat-pool dataset with class indices pinned to ``LABELS`` order.

    Reads ``<pool>/<label>/<date>__<NNN>__<variant>.jpg`` and keeps only the files
    whose date is in ``dates`` and whose name matches ``pattern``.

    Three deliberate differences from ``torchvision.datasets.ImageFolder``:

    1. Class indices come from ``LABELS``, not from sorted directory names.
       Alphabetical order agrees today (``empty`` < ``exist`` < ``sleeping``), but a
       class such as ``sitting`` would silently renumber the existing ones while the
       checkpoint kept claiming the old order. An unexpected class directory is a
       hard error here.
    2. A per-split filename ``pattern`` selects which files are scored, and a class
       with no matching file is allowed (``ImageFolder`` raises). A single held-out
       night legitimately may contain no ``exist`` frames.
    3. Date filtering happens here, so every variant (bed crop and full frame) of a
       given night always lands in the same split — splitting per file would leak
       near-duplicates across the train/eval boundary.
    """

    def __init__(
        self,
        pool: Path,
        *,
        cache: dict[str, torch.Tensor],
        dates: set[str] | None = None,
        pattern: str = "*.jpg",
    ) -> None:
        self.pool = pool
        self.pattern = pattern
        self.dates = dates
        self._cache = cache
        unexpected = sorted(
            d.name for d in pool.iterdir() if d.is_dir() and d.name not in LABELS
        )
        if unexpected:
            raise SystemExit(
                f"unexpected class directories under {pool}: {unexpected}. "
                f"Expected exactly {list(LABELS)} — adding a class changes the label "
                f"index order and invalidates every existing checkpoint."
            )
        self.samples: list[tuple[str, int]] = []
        for index, label in enumerate(LABELS):
            label_dir = pool / label
            if not label_dir.is_dir():
                continue
            for image_path in sorted(label_dir.glob(pattern)):
                if dates is not None and date_of(image_path) not in dates:
                    continue
                self.samples.append((str(image_path), index))

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(LABELS)}

    @property
    def targets(self) -> list[int]:
        return [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def cache_key(path: str) -> str:
        """Machine-portable cache key: ``<label>/<filename>:<size>``.

        Deliberately excludes the absolute path and the mtime. A downloaded cache then
        still hits after the dataset tar is unpacked somewhere else with fresh
        timestamps, which is what makes a ~900 MB cache worth keeping in R2 at all.
        Filenames inside the pool are unique by construction (date + index + variant),
        and the size guards against a same-named file having different content.
        """
        image_path = Path(path)
        return f"{image_path.parent.name}/{image_path.name}:{image_path.stat().st_size}"

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        key = self.cache_key(path)
        base = self._cache.get(key)
        if base is None:
            with Image.open(path) as image:
                base = _to_base_tensor(image)
            self._cache[key] = base
        return base, int(target)

    def populate_cache(self) -> int:
        before = len(self._cache)
        for index in range(len(self)):
            self[index]
        return len(self._cache) - before

    def fingerprint(self) -> str:
        """Hash of the ordered file list.

        Saved probability matrices are only comparable row-for-row between two runs if
        both scored the exact same files in the exact same order. Because fold rotation
        gives each ensemble member a *different* validation split, the ensemble builder
        has to verify it is combining aligned matrices rather than assume it.
        """
        joined = "\n".join(
            f"{Path(path).parent.name}/{Path(path).name}" for path, _ in self.samples
        )
        return hashlib.sha256(joined.encode()).hexdigest()[:16]


def make_dataset(
    pool: Path,
    *,
    cache: dict[str, torch.Tensor],
    spec: SplitSpec | None = None,
    pattern: str = "*.jpg",
) -> LabeledImageFolder | None:
    """Build one split from the flat pool. ``None`` when it resolves to no images."""
    if not pool.is_dir():
        return None
    dataset = LabeledImageFolder(
        pool,
        cache=cache,
        dates=None if spec is None else spec.date_set(),
        pattern=pattern if spec is None else spec.pattern,
    )
    return dataset if len(dataset) > 0 else None


# Python 3.14 のマルチプロセス既定は 'forkserver'。これだと (a) 親プロセスの in-process
# テンソルキャッシュをワーカーが共有できず再デコードが走り、(b) キャッシュ済みテンソルを
# fd 経由で渡そうとして "too many fds" で落ちる。'fork' なら COW でキャッシュを共有でき
# fd も渡さない。ワーカーは CPU 変換のみで CUDA を触らないため fork で安全。
_FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


def _make_loader(
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    extra: dict[str, Any] = {}
    if num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 2
        if _FORK_AVAILABLE:
            extra["multiprocessing_context"] = "fork"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **extra,
    )


class _GpuBatches:
    """Batch source that stages a whole split's base images in VRAM.

    The entire decoded pool is ~850 MiB of uint8 at ``BASE_CACHE_SIZE`` — under 2% of an
    L40S. Once the tensor cache is warm ``num_workers`` is 0, so the DataLoader's collate
    and its pageable host-to-device copy run on the main process *between* optimizer
    steps and never overlap with the GPU. Staging the split once turns every subsequent
    batch into a device-side gather, which is what removes that serial section.

    Iteration order matches ``DataLoader(shuffle=...)`` in effect, not bit-for-bit: the
    permutation comes from the CUDA generator that :func:`set_seed` seeds, so a run is
    still reproducible from its seed.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset[tuple[torch.Tensor, int]],
        *,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        stage_chunk: int = 256,
    ) -> None:
        count = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.images = torch.empty(
            (count, 3, BASE_CACHE_SIZE, BASE_CACHE_SIZE),
            dtype=torch.uint8,
            device=device,
        )
        targets = torch.empty(count, dtype=torch.int64)
        # Chunked so the staging copy is a few hundred large transfers rather than one
        # per image, without ever materialising a second full copy on the host.
        for start in range(0, count, stage_chunk):
            stop = min(start + stage_chunk, count)
            items = [dataset[index] for index in range(start, stop)]
            self.images[start:stop].copy_(torch.stack([base for base, _ in items]))
            targets[start:stop] = torch.tensor(
                [target for _, target in items], dtype=torch.int64
            )
        self.targets = targets.to(device)

    def __len__(self) -> int:
        return (self.images.size(0) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        count = self.images.size(0)
        order = (
            torch.randperm(count, device=self.images.device) if self.shuffle else None
        )
        for start in range(0, count, self.batch_size):
            stop = min(start + self.batch_size, count)
            if order is None:
                yield self.images[start:stop], self.targets[start:stop]
            else:
                picked = order[start:stop]
                yield (
                    self.images.index_select(0, picked),
                    self.targets.index_select(0, picked),
                )


def _make_batch_source(
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
):
    """VRAM-resident batches on CUDA, the ordinary DataLoader everywhere else."""
    if device.type == "cuda":
        return _GpuBatches(
            dataset, batch_size=batch_size, shuffle=shuffle, device=device
        )
    return _make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=False,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(targets: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets), minlength=len(LABELS)).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean()


def macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    scores = []
    for cls in range(len(LABELS)):
        pp, tp_mask = preds == cls, targets == cls
        tp = (pp & tp_mask).sum().item()
        fp = (pp & ~tp_mask).sum().item()
        fn = (~pp & tp_mask).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return sum(scores) / len(LABELS)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor) -> list[list[int]]:
    n = len(LABELS)
    matrix = torch.zeros(n, n, dtype=torch.long)
    for true, pred in zip(targets.tolist(), preds.tolist(), strict=True):
        matrix[true][pred] += 1
    return matrix.tolist()


def _run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    gpu_transform: GpuTransform,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    collect_probs: bool = False,
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)
    # Accumulated on the device: reading any of this per step (``.item()``/``.cpu()``)
    # forces a sync that stalls the whole pipeline on a ~12-step epoch. One transfer at
    # the end costs the same bytes and none of the stalls.
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_examples = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for images, targets in loader:
        # images: fixed-size uint8 batch -> normalized float (augmented on GPU).
        images = gpu_transform(images.to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(is_train):
            with autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        preds = logits.argmax(dim=1)
        bs = images.size(0)
        total_loss += loss.detach().double() * bs
        total_examples += bs
        all_preds.append(preds.detach())
        all_targets.append(targets.detach())
        if collect_probs:
            all_probs.append(torch.softmax(logits.detach().float(), dim=1))

    if total_examples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0, macro_f1=0.0)

    # The single sync point of the epoch.
    preds_cat = torch.cat(all_preds).cpu()
    targets_cat = torch.cat(all_targets).cpu()
    return EpochMetrics(
        loss=(total_loss / total_examples).item(),
        accuracy=(preds_cat == targets_cat).float().mean().item(),
        macro_f1=macro_f1(preds_cat, targets_cat),
        confusion=confusion_matrix(preds_cat, targets_cat),
        probs=torch.cat(all_probs).cpu().numpy().astype(np.float16)
        if collect_probs
        else None,
        targets=targets_cat.numpy().astype(np.int8) if collect_probs else None,
    )


def _evaluate(
    *,
    name: str,
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, int]] | None,
    batch_size: int,
    num_workers: int,
    model: nn.Module,
    gpu_transform: GpuTransform,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
    collect_probs: bool = False,
    source=None,
) -> EpochMetrics | None:
    if dataset is None or len(dataset) == 0:
        print(f"[eval] {name}: skipped", flush=True)
        return None
    metrics = _run_epoch(
        model=model,
        # ``val`` is scored every epoch, so its source is staged once by the caller and
        # handed in here rather than rebuilt per epoch.
        loader=source
        if source is not None
        else _make_batch_source(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=device,
        ),
        gpu_transform=gpu_transform,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp=amp,
        amp_dtype=amp_dtype,
        channels_last=channels_last,
        collect_probs=collect_probs,
    )
    print(
        f"[eval] {name}: n={len(dataset)} loss={metrics.loss:.4f} "
        f"acc={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(
    hp: HParams, *, channels_last: bool, device: torch.device, pretrained: bool = True
) -> nn.Module:
    kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": len(LABELS)}
    if hp.drop_path_rate is not None:
        kwargs["drop_path_rate"] = hp.drop_path_rate
    model = timm.create_model(hp.model_name, **kwargs)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


def _load_init_state(
    model: nn.Module,
    *,
    checkpoint_path: Path,
    member_index: int | None,
    expected_model_name: str,
    device: torch.device,
) -> str:
    """Warm-start ``model`` from a previous checkpoint or ensemble bundle member.

    Refuses to load across architectures: the state dict would partially match and
    leave the model silently broken.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    found = checkpoint.get("model_name")
    if found != expected_model_name:
        raise SystemExit(
            f"init checkpoint architecture mismatch: {checkpoint_path} is {found!r} "
            f"but training {expected_model_name!r}"
        )
    if checkpoint.get("ensemble"):
        members = checkpoint["members"]
        index = 0 if member_index is None else member_index % len(members)
        state_dict = members[index]
        source = f"{checkpoint_path}#member{index}of{len(members)}"
    else:
        state_dict = checkpoint["state_dict"]
        source = str(checkpoint_path)
    model.load_state_dict(state_dict)
    print(f"[train] warm start from {source}", flush=True)
    return source


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def run_training(
    *,
    pool_dir: Path,
    output_dir: Path,
    device_str: str,
    seed: int,
    epochs: int,
    hparams: HParams,
    train_spec: SplitSpec,
    eval_specs: dict[str, SplitSpec],
    num_workers: int = 0,
    cache: dict[str, torch.Tensor] | None = None,
    init_checkpoint: Path | None = None,
    init_member_index: int | None = None,
    epoch_callback: Callable[[int, float], None] | None = None,
    save_probs: bool = True,
    tb_log_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one model. Returns the metrics summary (also written as JSON).

    Splits are resolved from the flat image pool by date, so which nights validate is
    a runtime choice. ``eval_specs`` maps a split name to its dates and filename
    pattern; the pattern is per-split on purpose, because the full-frame splits are
    scored with ``*__full.jpg`` while ``test_legacy`` has untagged
    ``{date}__{NNN}.jpg`` filenames. One shared glob would resolve a split to zero
    images and report a meaningless score, so every named split must be non-empty —
    that is the check which catches such a mistake.

    A split named ``val`` drives early stopping and is the value handed to
    ``epoch_callback(epoch, val_macro_f1)``, which may raise to abort the run — that
    is how Optuna pruning is wired in.
    """
    hp = hparams
    set_seed(seed)
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    on_cuda = device.type == "cuda"
    amp = on_cuda
    # channels_last is essential: depthwise convs (convnext/efficientnet) are several
    # times slower in the default contiguous (NCHW) format.
    channels_last = on_cuda
    # fp16 autocast (with GradScaler), not bf16: cuDNN 9.2 has no channels_last engine
    # for *bf16* depthwise convs and raises "FIND was unable to find an engine".
    amp_dtype = torch.float16
    if on_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    tensor_cache = GLOBAL_TENSOR_CACHE if cache is None else cache
    gpu_train_transform = GpuTransform(device, train=True, hp=hp)
    gpu_eval_transform = GpuTransform(device, train=False, hp=hp)

    overlap = sorted(
        set(train_spec.dates) & {d for spec in eval_specs.values() for d in spec.dates}
    )
    if overlap:
        raise SystemExit(
            f"dates appear in both train and an eval split: {overlap}. A bed crop and "
            f"a full frame of the same moment are near-duplicates, so this would leak."
        )

    train_dataset = make_dataset(pool_dir, cache=tensor_cache, spec=train_spec)
    if train_dataset is None:
        raise SystemExit(
            f"train split resolved to zero images in {pool_dir} "
            f"({len(train_spec.dates)} date(s), pattern {train_spec.pattern!r})"
        )
    eval_datasets = {
        name: make_dataset(pool_dir, cache=tensor_cache, spec=spec)
        for name, spec in eval_specs.items()
    }
    empty = sorted(name for name, ds in eval_datasets.items() if ds is None)
    if empty:
        raise SystemExit(
            f"eval splits resolved to zero images: {empty}. Check the per-split "
            f"pattern — test_legacy has untagged filenames and matches '*.jpg', "
            f"not '*__full.jpg'."
        )
    val_dataset = eval_datasets.get("val")

    cache_added = train_dataset.populate_cache()
    for dataset in eval_datasets.values():
        if dataset is not None:
            cache_added += dataset.populate_cache()
    if cache_added:
        print(
            f"[train] populated tensor cache: added={cache_added} "
            f"total={len(tensor_cache)}",
            flush=True,
        )
    # With everything decoded in-process, workers would only add fork overhead.
    loader_num_workers = 0 if tensor_cache else num_workers

    stage_started = time.perf_counter()
    train_loader = _make_batch_source(
        train_dataset,
        batch_size=hp.batch_size,
        num_workers=loader_num_workers,
        shuffle=True,
        device=device,
    )
    # Staged once for the whole run, not once per epoch.
    val_source = (
        None
        if val_dataset is None
        else _make_batch_source(
            val_dataset,
            batch_size=hp.batch_size,
            num_workers=loader_num_workers,
            shuffle=False,
            device=device,
        )
    )
    stage_seconds = time.perf_counter() - stage_started

    model = build_model(hp, channels_last=channels_last, device=device)
    init_source = None
    if init_checkpoint is not None:
        init_source = _load_init_state(
            model,
            checkpoint_path=init_checkpoint,
            member_index=init_member_index,
            expected_model_name=hp.model_name,
            device=device,
        )

    class_weights = _class_weights(list(train_dataset.targets)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=hp.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
    )
    warmup_epochs = min(hp.warmup_epochs, max(0, epochs - 1))
    schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    milestones: list[int] = []
    if warmup_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
        )
        milestones.append(warmup_epochs)
    schedulers.append(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
    )
    scheduler = (
        torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=schedulers, milestones=milestones
        )
        if len(schedulers) > 1
        else schedulers[0]
    )
    scaler = GradScaler(enabled=amp and on_cuda)

    best_val_f1 = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    best_model_path = output_dir / "sleep_classifier.pt"
    best_state: dict[str, torch.Tensor] | None = None
    best_scalars: dict[str, float] = {}
    history: list[dict[str, object]] = []
    # Where a trial's wall clock actually goes. A sweep is budgeted in trials per hour,
    # so this is the number that decides whether tuning the loop is worth anything.
    timing = {"stage": round(stage_seconds, 3), "train": 0.0, "val": 0.0}
    epochs_run = 0

    eval_sizes = " ".join(
        f"{name}={0 if ds is None else len(ds)}" for name, ds in eval_datasets.items()
    )
    print(
        f"[train] train={len(train_dataset)} {eval_sizes} device={device} "
        f"model={hp.model_name} seed={seed} epochs={epochs} batch={hp.batch_size} "
        f"lr={hp.learning_rate:.3e} wd={hp.weight_decay:.3e} "
        f"drop_path={hp.drop_path_rate} ls={hp.label_smoothing:.3f} "
        f"warmup={warmup_epochs} workers={loader_num_workers} "
        f"amp_dtype={str(amp_dtype).rsplit('.', 1)[-1]} "
        f"channels_last={bool(channels_last)} cached={len(tensor_cache)}",
        flush=True,
    )
    print(
        "[train] class_weights="
        + ", ".join(
            f"{lbl}={class_weights[i].item():.4f}" for i, lbl in enumerate(LABELS)
        ),
        flush=True,
    )

    split_dates = {"train": list(train_spec.dates)} | {
        name: list(spec.dates) for name, spec in eval_specs.items()
    }
    checkpoint_meta = {
        "model_name": hp.model_name,
        "labels": LABELS,
        "image_size": hp.image_size,
        "seed": seed,
        "epochs": epochs,
        # Kept for compatibility with the existing checkpoint readers, which expect
        # exactly these three keys.
        "train_dates": split_dates["train"],
        "val_dates": split_dates.get("val", []),
        "test_dates": split_dates.get("test_full", []),
        "split_dates": split_dates,
        "split_patterns": {"train": train_spec.pattern}
        | {name: spec.pattern for name, spec in eval_specs.items()},
        # Lets the ensemble builder prove two members' probability matrices line up.
        "eval_fingerprints": {
            name: dataset.fingerprint()
            for name, dataset in eval_datasets.items()
            if dataset is not None
        },
    }

    writer = SummaryWriter(log_dir=str(tb_log_dir)) if tb_log_dir is not None else None
    try:
        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            epoch_started = time.perf_counter()
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                gpu_transform=gpu_train_transform,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=bool(amp),
                amp_dtype=amp_dtype,
                channels_last=bool(channels_last),
            )
            timing["train"] += time.perf_counter() - epoch_started
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch={epoch}/{epochs} lr={current_lr:.2e} "
                f"loss={train_metrics.loss:.4f} acc={train_metrics.accuracy:.4f} "
                f"macro_f1={train_metrics.macro_f1:.4f}",
                flush=True,
            )
            scheduler.step()

            # Only ``val`` is scored per epoch; the other splits are scored once at the
            # end with the best weights, so a pruned trial pays nothing for them.
            val_started = time.perf_counter()
            val_metrics = _evaluate(
                name="val",
                dataset=val_dataset,
                batch_size=hp.batch_size,
                num_workers=loader_num_workers,
                model=model,
                gpu_transform=gpu_eval_transform,
                criterion=criterion,
                device=device,
                amp=bool(amp),
                amp_dtype=amp_dtype,
                channels_last=bool(channels_last),
                source=val_source,
            )
            timing["val"] += time.perf_counter() - val_started
            score = (
                val_metrics.macro_f1 if val_metrics is not None else train_metrics.macro_f1
            )
            if writer is not None:
                writer.add_scalar("loss/train", train_metrics.loss, epoch)
                writer.add_scalar("accuracy/train", train_metrics.accuracy, epoch)
                writer.add_scalar("macro_f1/train", train_metrics.macro_f1, epoch)
                writer.add_scalar("lr", current_lr, epoch)
                if val_metrics is not None:
                    writer.add_scalar("loss/val", val_metrics.loss, epoch)
                    writer.add_scalar("accuracy/val", val_metrics.accuracy, epoch)
                    writer.add_scalar("macro_f1/val", val_metrics.macro_f1, epoch)

            if score > best_val_f1:
                best_val_f1 = score
                best_epoch = epoch
                stale_epochs = 0
                best = val_metrics if val_metrics is not None else train_metrics
                # Kept in VRAM rather than written out here: convnext_small is ~200 MB and
                # val improves on most early epochs, so serialising every time spends
                # seconds per trial on disk writes for a file only the last one survives.
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                best_scalars = {
                    "best_epoch": epoch,
                    "best_acc": best.accuracy,
                    "best_loss": best.loss,
                    "best_f1": best.macro_f1,
                }
            else:
                stale_epochs += 1

            history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics.to_json(),
                    "val": None if val_metrics is None else val_metrics.to_json(),
                }
            )

            # After recording history, so a pruned trial still leaves its curve behind.
            if epoch_callback is not None:
                epoch_callback(epoch, score)

            if val_dataset is not None and hp.patience >= 0 and stale_epochs >= hp.patience:
                print(
                    f"[train] early stop at epoch={epoch} (best_epoch={best_epoch} "
                    f"best_val_macro_f1={best_val_f1:.4f})",
                    flush=True,
                )
                break

        # The one write of the run. Done before scoring so an interrupted final eval still
        # leaves the checkpoint the epochs earned.
        save_started = time.perf_counter()
        if best_state is None:
            raise SystemExit(
                "training produced no best epoch — every epoch scored worse than -inf, "
                "which means the eval loop never ran."
            )
        torch.save(
            {
                **checkpoint_meta,
                "state_dict": {key: value.cpu() for key, value in best_state.items()},
                **best_scalars,
            },
            best_model_path,
        )
        timing["save"] = round(time.perf_counter() - save_started, 3)

        # Score every split with the best weights. ``val`` is re-scored here too, so the
        # saved probability matrix corresponds to the checkpoint we actually ship.
        model.load_state_dict(best_state)
        final_eval_started = time.perf_counter()
        evals: dict[str, dict[str, Any]] = {}
        for name, dataset in eval_datasets.items():
            metrics = _evaluate(
                name=name,
                dataset=dataset,
                batch_size=hp.batch_size,
                num_workers=loader_num_workers,
                model=model,
                gpu_transform=gpu_eval_transform,
                criterion=criterion,
                device=device,
                amp=bool(amp),
                amp_dtype=amp_dtype,
                channels_last=bool(channels_last),
                collect_probs=save_probs,
                source=val_source if name == "val" else None,
            )
            if metrics is None:
                continue
            evals[name] = metrics.to_json()
            if save_probs and metrics.probs is not None:
                np.save(output_dir / f"probs_{name}.npy", metrics.probs)
                np.save(output_dir / f"targets_{name}.npy", metrics.targets)

        if writer is not None:
            for name, m in evals.items():
                writer.add_scalar(f"final/macro_f1/{name}", m["macro_f1"], best_epoch)
                writer.add_scalar(f"final/accuracy/{name}", m["accuracy"], best_epoch)
            if "test_hard" in evals:
                conf = evals["test_hard"]["confusion"]
                false_sleeping = conf[LABELS.index("empty")][LABELS.index("sleeping")]
                writer.add_scalar(
                    "final/test_hard_false_sleeping", false_sleeping, best_epoch
                )
            hparam_dict = {
                k: (-1.0 if v is None else v) for k, v in hp.to_dict().items()
            }
            metric_dict = {"hparam/best_val_macro_f1": float(best_val_f1)}
            for name, m in evals.items():
                metric_dict[f"hparam/{name}_macro_f1"] = float(m["macro_f1"])
            writer.add_hparams(hparam_dict, metric_dict)
    finally:
        if writer is not None:
            writer.close()

    timing["final_eval"] = round(time.perf_counter() - final_eval_started, 3)
    timing["train"] = round(timing["train"], 3)
    timing["val"] = round(timing["val"], 3)
    timing["epochs_run"] = epochs_run
    print(
        "[train] timing: "
        + " ".join(f"{key}={value}" for key, value in timing.items())
        + f" (per-epoch train={timing['train'] / max(1, epochs_run):.2f}s)",
        flush=True,
    )

    summary = {
        **checkpoint_meta,
        "model_path": str(best_model_path),
        "hparams": hp.to_dict(),
        "init_source": init_source,
        "norm_mean": NORM_MEAN,
        "norm_std": NORM_STD,
        "class_weights": {lbl: class_weights[i].item() for i, lbl in enumerate(LABELS)},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "evals": evals,
        "timing": timing,
    }
    summary_path = output_dir / "sleep_classifier_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[train] metrics: {summary_path}", flush=True)

    del model, optimizer, scaler, criterion, train_loader, val_source, best_state
    if on_cuda:
        torch.cuda.empty_cache()

    return summary
