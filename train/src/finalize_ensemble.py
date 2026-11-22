"""Assemble the final ensemble bundle. Runs on CPU in seconds.

Reads the ``members/`` artifacts a refresh run left in R2 — the saved softmax
probability matrices, not the weights — searches for the best member subset, and only
then downloads the weights it actually needs. No GPU, no re-inference.

**Which set selects the members.** Fold rotation means each member validated on
different nights, so ``val`` probability matrices are not row-aligned and cannot be
combined. The frozen splits are identical for every member, so selection happens on
``test_full``. Each set therefore has exactly one job:

    val         (rotates)  early stopping + the Optuna objective
    test_full   (frozen)   ensemble member selection  <- this script
    test_hard   (frozen)   deploy gate: did the false-sleeping errors actually drop
    test_legacy (frozen)   regression gate vs the current production model

``test_hard`` and ``test_legacy`` are never used to choose anything here, which is what
keeps them honest as gates.

**Uniform averaging only.** Production averages member softmax outputs uniformly and
ignores any extra keys (``src/analyze_day.py``). Subset selection gets most of the
benefit with zero serving changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import config
import notify_discord as discord
import r2_store
import score_bundle
import trainer

# The split whose probability matrices drive subset selection.
SELECT_SPLIT = "test_full"
# Reported but never optimized against.
GATE_SPLITS = ("test_hard", "test_legacy")


@dataclass
class Member:
    member_id: str
    meta: dict
    probs: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]
    local_dir: Path

    @property
    def val_score(self) -> float:
        return float(self.meta.get("val_macro_f1", 0.0))


def _macro_f1(probs: np.ndarray, targets: np.ndarray) -> float:
    preds = torch.from_numpy(probs.astype(np.float32)).argmax(dim=1)
    return trainer.macro_f1(preds, torch.from_numpy(targets.astype(np.int64)))


def _confusion(probs: np.ndarray, targets: np.ndarray) -> list[list[int]]:
    preds = torch.from_numpy(probs.astype(np.float32)).argmax(dim=1)
    return trainer.confusion_matrix(preds, torch.from_numpy(targets.astype(np.int64)))


def _average(members: list[Member], split: str) -> np.ndarray:
    stacked = np.stack([m.probs[split].astype(np.float32) for m in members])
    return stacked.mean(axis=0)


def _load_one(member_id: str, local: Path, meta: dict) -> Member | None:
    probs: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for split in (SELECT_SPLIT, *GATE_SPLITS):
        probs_path = local / f"probs_{split}.npy"
        targets_path = local / f"targets_{split}.npy"
        if not (probs_path.exists() and targets_path.exists()):
            continue
        probs[split] = np.load(probs_path)
        targets[split] = np.load(targets_path)
    missing = [s for s in (SELECT_SPLIT, *GATE_SPLITS) if s not in probs]
    if missing:
        print(
            f"[ens] {member_id}: missing probs for {missing}, skipping", file=sys.stderr
        )
        return None
    return Member(
        member_id=member_id, meta=meta, probs=probs, targets=targets, local_dir=local
    )


def load_members_local(members_dir: Path) -> list[Member]:
    """Read candidates straight off disk (a refresh run's work dir, or a test)."""
    members: list[Member] = []
    for local in sorted(p for p in members_dir.iterdir() if p.is_dir()):
        meta_path = local / "meta.json"
        metrics_path = local / "sleep_classifier_metrics.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        elif metrics_path.exists():
            summary = json.loads(metrics_path.read_text())
            meta = {
                "model_name": summary["model_name"],
                "labels": summary["labels"],
                "image_size": summary["image_size"],
                "val_macro_f1": summary["best_val_macro_f1"],
                "dataset_hash": summary.get("dataset_hash"),
                "eval_fingerprints": summary.get("eval_fingerprints", {}),
            }
        else:
            continue
        member = _load_one(local.name, local, meta)
        if member is not None:
            members.append(member)
    if not members:
        raise SystemExit(f"no usable members under {members_dir}")
    return members


def load_members(store: r2_store.R2Store, work_dir: Path) -> list[Member]:
    """Download every ``members/`` artifact except the weights."""
    prefix = f"{store.run_prefix}/members/"
    keys = store.list_keys(prefix)
    member_ids = sorted(
        {
            key[len(prefix) :].split("/", 1)[0]
            for key in keys
            if "/" in key[len(prefix) :]
        }
    )
    if not member_ids:
        raise SystemExit(
            f"no members under {store.uri(prefix)} — run `--mode refresh` first "
            f"(a refresh run is what produces the ensemble candidates)."
        )
    members: list[Member] = []
    for member_id in member_ids:
        local = work_dir / member_id
        local.mkdir(parents=True, exist_ok=True)
        meta = store.get_json(f"{prefix}{member_id}/meta.json")
        if meta is None:
            print(f"[ens] {member_id}: no meta.json, skipping", file=sys.stderr)
            continue
        for split in (SELECT_SPLIT, *GATE_SPLITS):
            store.download(
                f"{prefix}{member_id}/probs_{split}.npy", local / f"probs_{split}.npy"
            )
            store.download(
                f"{prefix}{member_id}/targets_{split}.npy",
                local / f"targets_{split}.npy",
            )
        member = _load_one(member_id, local, meta)
        if member is not None:
            members.append(member)
    return members


def check_alignment(members: list[Member]) -> None:
    """Refuse to average matrices that did not score the same files in the same order."""
    for split in (SELECT_SPLIT, *GATE_SPLITS):
        shapes = {m.member_id: m.probs[split].shape for m in members}
        if len({s for s in shapes.values()}) != 1:
            raise SystemExit(f"{split}: probability matrices differ in shape: {shapes}")
        reference = members[0].targets[split]
        for member in members[1:]:
            if not np.array_equal(member.targets[split], reference):
                raise SystemExit(
                    f"{split}: {member.member_id} scored a different label sequence "
                    f"than {members[0].member_id} — the two are not comparable"
                )
        fingerprints = {
            m.member_id: (m.meta.get("eval_fingerprints") or {}).get(split)
            for m in members
        }
        distinct = {f for f in fingerprints.values() if f}
        if len(distinct) > 1:
            raise SystemExit(
                f"{split}: eval file-list fingerprints disagree: {fingerprints}"
            )

    architectures = {m.meta["model_name"] for m in members}
    if len(architectures) > 1:
        raise SystemExit(
            f"mixed architectures {sorted(architectures)}: production loads ONE timm "
            f"model and swaps state dicts into it, so every member must match"
        )
    labels = {tuple(m.meta["labels"]) for m in members}
    if labels != {tuple(trainer.LABELS)}:
        raise SystemExit(
            f"member label order {labels} != {tuple(trainer.LABELS)} — the head's "
            f"output units are positional and this would silently permute classes"
        )


def greedy_forward_selection(
    members: list[Member], *, max_members: int
) -> list[Member]:
    """Add the member that most improves the selection split, until nothing helps."""
    targets = members[0].targets[SELECT_SPLIT]
    chosen: list[Member] = []
    best_score = -1.0
    while len(chosen) < max_members:
        candidate_best: tuple[float, Member] | None = None
        for member in members:
            score = _macro_f1(_average([*chosen, member], SELECT_SPLIT), targets)
            if candidate_best is None or score > candidate_best[0]:
                candidate_best = (score, member)
        if candidate_best is None or candidate_best[0] <= best_score:
            break
        best_score, winner = candidate_best
        chosen.append(winner)
        print(
            f"[ens] + {winner.member_id} -> {SELECT_SPLIT} macro_f1={best_score:.4f}",
            flush=True,
        )
    return chosen


def top_by_val(members: list[Member], k: int) -> list[Member]:
    return sorted(members, key=lambda m: m.val_score, reverse=True)[:k]


def _false_sleeping(confusion: list[list[int]]) -> int:
    """Frames that are really ``empty`` but get predicted ``sleeping``."""
    empty = trainer.LABELS.index("empty")
    sleeping = trainer.LABELS.index("sleeping")
    return int(confusion[empty][sleeping])


def _compare(new_result: dict, baseline: dict | None) -> dict | None:
    """New vs production, per split, plus the one error count that decides the deploy."""
    if baseline is None:
        return None
    out: dict[str, Any] = {}
    for split, new_metrics in new_result["splits"].items():
        old_metrics = baseline["splits"].get(split)
        if old_metrics is None:
            continue
        entry = {
            "macro_f1_old": old_metrics["macro_f1"],
            "macro_f1_new": new_metrics["macro_f1"],
            "macro_f1_delta": new_metrics["macro_f1"] - old_metrics["macro_f1"],
        }
        if split == "test_hard":
            old_errors = _false_sleeping(old_metrics["confusion"])
            new_errors = _false_sleeping(new_metrics["confusion"])
            entry["false_sleeping_old"] = old_errors
            entry["false_sleeping_new"] = new_errors
            entry["false_sleeping_delta"] = new_errors - old_errors
        out[split] = entry
    gates = {
        "test_hard_false_sleeping_improved": (
            out.get("test_hard", {}).get("false_sleeping_delta", 1) < 0
        ),
        "test_legacy_not_regressed": (
            out.get("test_legacy", {}).get("macro_f1_delta", -1.0) >= -0.005
        ),
    }
    out["gates"] = gates
    out["verdict"] = "PASS" if all(gates.values()) else "REVIEW"
    return out


def resolve_baseline() -> dict | None:
    """Score the model currently in production over the same frozen splits."""
    baseline_bundle = config.PRODUCTION_MODEL_PATH
    if not baseline_bundle.exists():
        print(f"[ens] baseline skipped: {baseline_bundle} not found", file=sys.stderr)
        return None
    if not config.MANIFEST_PATH.exists():
        print(
            f"[ens] baseline skipped: {config.MANIFEST_PATH} not found (the baseline needs the "
            f"local image pool)",
            file=sys.stderr,
        )
        return None

    manifest = json.loads(config.MANIFEST_PATH.read_text())
    dataset_hash = manifest.get("dataset_hash")
    stamp = {"bundle": str(baseline_bundle), "dataset_hash": dataset_hash}
    cache_path = config.BASELINE_CACHE_PATH
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cached = None
        if cached and cached.get("_key") == stamp:
            print(f"[ens] baseline reused from {cache_path}", flush=True)
            return {k: v for k, v in cached.items() if k != "_key"}

    print(f"[ens] scoring baseline {baseline_bundle} (CPU, ~10min) ...", flush=True)
    scores = score_bundle.score_bundle(
        baseline_bundle,
        pool_dir=config.DATASET_DIR,
        manifest=manifest,
        split_names=(SELECT_SPLIT, *GATE_SPLITS),
        device="cpu",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({**scores, "_key": stamp}, ensure_ascii=False, indent=2)
    )
    return scores


def evaluate(members: list[Member], label: str) -> dict:
    result = {"label": label, "members": [m.member_id for m in members], "splits": {}}
    for split in (SELECT_SPLIT, *GATE_SPLITS):
        averaged = _average(members, split)
        targets = members[0].targets[split]
        result["splits"][split] = {
            "macro_f1": _macro_f1(averaged, targets),
            "confusion": _confusion(averaged, targets),
        }
    return result


def build_bundle(
    members: list[Member],
    store: r2_store.R2Store | None,
    out_path: Path,
    *,
    dtype: str,
) -> dict:
    """Fetch the chosen members' weights and write the production bundle."""
    state_dicts = []
    sources = []
    cast = torch.float16 if dtype == "fp16" else torch.float32
    with tempfile.TemporaryDirectory() as tmp:
        for member in members:
            if store is None:
                local = next(
                    (
                        candidate
                        for candidate in (
                            member.local_dir / "model.fp16.pt",
                            member.local_dir / "sleep_classifier.pt",
                        )
                        if candidate.exists()
                    ),
                    None,
                )
                if local is None:
                    raise SystemExit(f"no weights found in {member.local_dir}")
                source = str(local)
            else:
                local = Path(tmp) / f"{member.member_id}.pt"
                key = f"{store.run_prefix}/members/{member.member_id}/model.fp16.pt"
                if not store.download(key, local):
                    raise SystemExit(f"could not download {store.uri(key)}")
                source = store.uri(key)
            checkpoint = torch.load(local, map_location="cpu", weights_only=False)
            state_dicts.append(
                {
                    name: (value.to(cast) if value.is_floating_point() else value)
                    for name, value in checkpoint["state_dict"].items()
                }
            )
            sources.append(source)

    first = members[0].meta
    bundle = {
        # This schema is the contract with src/analyze_day.py.
        # Keys and types must stay exactly as they are.
        "ensemble": True,
        "model_name": first["model_name"],
        "labels": tuple(trainer.LABELS),
        "image_size": int(first["image_size"]),
        "members": state_dicts,
        "member_val": [m.val_score for m in members],
        "member_sources": sources,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "dtype": dtype,
        "model_name": bundle["model_name"],
        "image_size": bundle["image_size"],
        "member_count": len(state_dicts),
        "member_sources": sources,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--local-members",
        type=Path,
        default=None,
        help="read candidates from this directory instead of R2 (offline / testing)",
    )
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args(argv)

    max_members = config.ENSEMBLE_MAX_MEMBERS
    baseline_k = config.ENSEMBLE_BASELINE_K
    dtype = config.BUNDLE_DTYPE
    work_dir = config.ENSEMBLE_DIR / args.run_name

    if args.local_members is not None:
        store = None
        members = load_members_local(args.local_members)
    else:
        store = r2_store.R2Store(args.run_name)
        store.verify()
        members = load_members(store, work_dir)
    print(f"[ens] loaded {len(members)} member(s)", flush=True)
    check_alignment(members)

    greedy = greedy_forward_selection(members, max_members=max_members)
    baseline = top_by_val(members, baseline_k)
    greedy_result = evaluate(greedy, "greedy(test_full)")
    baseline_result = evaluate(baseline, f"top{baseline_k}-by-val")

    def gate_sum(result: dict) -> float:
        return sum(result["splits"][split]["macro_f1"] for split in GATE_SPLITS)

    chosen, chosen_result, other_result = (
        (greedy, greedy_result, baseline_result)
        if gate_sum(greedy_result) >= gate_sum(baseline_result)
        else (baseline, baseline_result, greedy_result)
    )
    print(
        f"[ens] selected: {chosen_result['label']} "
        f"(gates {gate_sum(chosen_result):.4f} vs {gate_sum(other_result):.4f})",
        flush=True,
    )

    baseline_scores = None if args.no_baseline else resolve_baseline()

    out_path = work_dir / "sleep_classifier_ensemble.pt"
    bundle_info = build_bundle(chosen, store, out_path, dtype=dtype)

    singles = {
        m.member_id: {
            "val_macro_f1": m.val_score,
            **{
                split: _macro_f1(m.probs[split], m.targets[split])
                for split in (SELECT_SPLIT, *GATE_SPLITS)
            },
        }
        for m in members
    }
    report = {
        "run_name": args.run_name,
        "dataset_hash": members[0].meta.get("dataset_hash"),
        "selection_split": SELECT_SPLIT,
        "gate_splits": list(GATE_SPLITS),
        "selected": chosen_result,
        "rejected": other_result,
        "single_members": singles,
        "bundle": bundle_info,
        "baseline": baseline_scores,
        "comparison": _compare(chosen_result, baseline_scores),
        "deploy_gate_note": (
            "判定は new と baseline（現本番モデル）の比較で行う。"
            "(1) test_hard で「実際は empty なのに sleeping と誤答」した件数が減っているか。"
            "(2) test_legacy の macro-F1 が baseline から落ちていないか。"
            "test_full は選抜用にのみ使っている。"
        ),
    }
    report_path = out_path.parent / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[ens] bundle: {out_path} ({bundle_info['bytes'] / 1e6:.0f} MB)")
    print(f"[ens] report: {report_path}")

    bundle_uri = None
    if store is not None:
        bundle_uri = store.upload(
            out_path, f"{store.run_prefix}/ensemble/sleep_classifier_ensemble.pt"
        )
        store.put_json(report, f"{store.run_prefix}/ensemble/report.json")

    fields = [
        (
            "メンバー",
            "\n".join(f"`{m}`" for m in chosen_result["members"]),
            False,
        ),
        (
            "macro-F1",
            " ".join(
                f"{split}={chosen_result['splits'][split]['macro_f1']:.4f}"
                for split in (SELECT_SPLIT, *GATE_SPLITS)
            ),
            False,
        ),
        (
            "test_hard 混同行列 — NEW（行=正解）",
            discord.format_confusion(
                chosen_result["splits"]["test_hard"]["confusion"],
                list(trainer.LABELS),
            ),
            False,
        ),
    ]
    comparison = report["comparison"]
    if baseline_scores is not None and comparison is not None:
        fields.append(
            (
                "test_hard 混同行列 — 現本番モデル（行=正解）",
                discord.format_confusion(
                    baseline_scores["splits"]["test_hard"]["confusion"],
                    list(trainer.LABELS),
                ),
                False,
            )
        )
        hard = comparison.get("test_hard", {})
        legacy = comparison.get("test_legacy", {})
        fields.append(
            (
                f"判定: {comparison['verdict']}",
                "```\n"
                f"empty→sleeping誤答: {hard.get('false_sleeping_old', '?')}"
                f" -> {hard.get('false_sleeping_new', '?')}"
                f" ({hard.get('false_sleeping_delta', 0):+d})\n"
                f"test_legacy F1 : {legacy.get('macro_f1_old', 0):.4f}"
                f" -> {legacy.get('macro_f1_new', 0):.4f}"
                f" ({legacy.get('macro_f1_delta', 0):+.4f})\n"
                "```\n"
                "empty を sleeping と誤答する件数が減り、test_legacy が落ちていなければ PASS。",
                False,
            )
        )
    else:
        fields.append(
            (
                "⚠ ベースラインなし",
                "現本番モデルを採点していないので、混同行列に比較対象がありません。"
                "`train/src/score_bundle.py` を回して old vs new を見てください。",
                False,
            )
        )
    if bundle_uri:
        fields.append(("bundle", f"`{bundle_uri}`", False))
    verdict = (report["comparison"] or {}).get("verdict")
    discord.send_embed(
        title="🎁 アンサンブル完成",
        description=(
            f"{chosen_result['label']} を採用（比較: {other_result['label']}）"
            f" • {bundle_info['member_count']} メンバー"
            f" • {bundle_info['bytes'] / 1e6:.0f} MB {dtype}"
        ),
        fields=fields,
        color=(discord.COLOR_GOOD if verdict == "PASS" else discord.COLOR_WARN),
        footer=f"run={args.run_name}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
