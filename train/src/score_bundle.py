"""Score an existing checkpoint or ensemble bundle on the frozen eval splits.

This is what makes the deploy gate readable. ``test_hard`` is re-resolved to the newest
correction nights every month, so no prior number for it exists anywhere — a confusion
matrix for the new model alone has nothing to be compared against. Running the *current
production bundle* over the same splits from the same pool produces the missing baseline.

Both sides go through ``trainer``'s eval transform, not production's
``Resize((224,224))``, so the comparison isolates the model rather than mixing in a
preprocessing difference. (The two are the same geometry but not bit-identical: training
caches at 256 and interpolates down to 224.)

Runs on CPU in minutes — 144 + 540 + 118 images — so it needs no GPU and no cloud.

    uv run python train/src/score_bundle.py --bundle ~/data/models/sleep_classifier_ensemble.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
import splitplan
import trainer


def load_bundle(path: Path, device: torch.device) -> dict[str, Any]:
    """Read a checkpoint, accepting both a bundle and a single-model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    labels = tuple(checkpoint["labels"])
    if labels != trainer.LABELS:
        raise SystemExit(
            f"{path}: labels {labels} != {trainer.LABELS}. The head's output units are "
            f"positional, so a different order means the scores would be meaningless."
        )
    state_dicts = (
        checkpoint["members"]
        if checkpoint.get("ensemble")
        else [checkpoint["state_dict"]]
    )
    return {
        "model_name": checkpoint["model_name"],
        "image_size": int(checkpoint["image_size"]),
        "state_dicts": state_dicts,
        "member_val": checkpoint.get("member_val", []),
        "path": str(path),
    }


@torch.no_grad()
def score_split(
    bundle: dict[str, Any],
    dataset: trainer.LabeledImageFolder,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform softmax average over members. Returns (probs [N,3], targets [N])."""
    hp = trainer.HParams(
        model_name=bundle["model_name"],
        image_size=bundle["image_size"],
        batch_size=batch_size,
    )
    on_cuda = device.type == "cuda"
    model = trainer.build_model(
        hp, channels_last=on_cuda, device=device, pretrained=False
    )
    model.eval()
    transform = trainer.GpuTransform(device, train=False, hp=hp)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    prob_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    for images, targets in loader:
        batch = transform(images.to(device))
        if on_cuda:
            batch = batch.contiguous(memory_format=torch.channels_last)
        total = None
        # One model instance, weights swapped per member — matches how production
        # evaluates a bundle, and keeps peak memory to a single model.
        for state_dict in bundle["state_dicts"]:
            model.load_state_dict(state_dict)
            member = torch.softmax(model(batch).float(), dim=1)
            total = member if total is None else total + member
        prob_chunks.append((total / len(bundle["state_dicts"])).cpu())
        target_chunks.append(targets)
    probs = torch.cat(prob_chunks).numpy().astype(np.float16)
    all_targets = torch.cat(target_chunks)
    return probs, all_targets.numpy().astype(np.int8)


def score_bundle(
    bundle_path: Path,
    *,
    pool_dir: Path,
    manifest: dict,
    split_names: tuple[str, ...],
    device: str = "cpu",
    batch_size: int = 64,
    cache: dict[str, torch.Tensor] | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Score ``bundle_path`` on the named frozen splits.

    Fold rotation only moves ``val``; the frozen splits are identical for every fold, so
    resolving without a fold gives exactly the sets the ensemble members were scored on.
    """
    torch_device = torch.device(device)
    bundle = load_bundle(bundle_path, torch_device)
    resolved = splitplan.resolve(manifest, fold=None)
    shared_cache = {} if cache is None else cache

    result: dict[str, Any] = {
        "bundle": bundle["path"],
        "model_name": bundle["model_name"],
        "image_size": bundle["image_size"],
        "member_count": len(bundle["state_dicts"]),
        "splits": {},
    }
    for name in split_names:
        spec = resolved.evals.get(name)
        if spec is None:
            continue
        dataset = trainer.make_dataset(pool_dir, cache=shared_cache, spec=spec)
        if dataset is None:
            print(f"[score] {name}: no images, skipped", flush=True)
            continue
        probs, targets = score_split(
            bundle, dataset, device=torch_device, batch_size=batch_size
        )
        preds = torch.from_numpy(probs.astype(np.float32)).argmax(dim=1)
        target_tensor = torch.from_numpy(targets.astype(np.int64))
        result["splits"][name] = {
            "n": int(len(dataset)),
            "macro_f1": trainer.macro_f1(preds, target_tensor),
            "accuracy": float((preds == target_tensor).float().mean()),
            "confusion": trainer.confusion_matrix(preds, target_tensor),
        }
        print(
            f"[score] {name}: n={len(dataset)} "
            f"macro_f1={result['splits'][name]['macro_f1']:.4f}",
            flush=True,
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"probs_{name}.npy", probs)
            np.save(out_dir / f"targets_{name}.npy", targets)
    return result


def render_confusion(confusion: list[list[int]], labels: list[str]) -> str:
    width = max(len(label) for label in labels)
    lines = [" " * (width + 3) + " ".join(f"{label[:6]:>6}" for label in labels)]
    for index, label in enumerate(labels):
        cells = " ".join(f"{value:>6}" for value in confusion[index])
        lines.append(f"{label:<{width}} | {cells}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=config.PRODUCTION_MODEL_PATH,
        help="checkpoint or ensemble bundle to score (default: the production model)",
    )
    args = parser.parse_args(argv)

    pool_dir = config.DATASET_DIR
    manifest_path = config.MANIFEST_PATH
    split_names = ("test_hard", "test_legacy", "test_full")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    if not args.bundle.exists():
        raise SystemExit(f"{args.bundle} not found")
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — run build_dataset.py first")

    manifest = json.loads(manifest_path.read_text())
    result = score_bundle(
        args.bundle,
        pool_dir=pool_dir,
        manifest=manifest,
        split_names=split_names,
        device=device,
        batch_size=batch_size,
    )
    result["dataset_hash"] = manifest.get("dataset_hash")

    print()
    print(f"# {args.bundle}  ({result['model_name']}, {result['member_count']} member)")
    for name, metrics in result["splits"].items():
        print()
        print(f"## {name}  n={metrics['n']}  macro_f1={metrics['macro_f1']:.4f}")
        print(render_confusion(metrics["confusion"], list(trainer.LABELS)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
