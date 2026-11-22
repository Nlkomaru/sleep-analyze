"""Colab + Cloudflare R2 training runner (HPO and ensemble refresh).

Two modes, one entry point so interruption handling, R2 sync, TensorBoard logging,
and Discord notifications live in one place:

``--mode hpo``
    Optuna TPE search with a MedianPruner over a fixed dataset split. Writes metrics
    for every trial, keeps only the top-K checkpoints, and settles the best configs
    into ``best_hparams.json``.

``--mode refresh``
    Reads ``best_hparams.json`` and trains each of its top configs across multiple
    seeds and validation folds, producing ensemble candidates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

import config
import notify_discord as discord
import r2_store
import splitplan
import trainer

# Batch sizes we allow, and the VRAM needed to use them for convnext_small @224.
_BATCH_CHOICES = (64, 128, 256)
_VRAM_BATCH_CAP = ((40, 256), (20, 128), (0, 64))  # (GiB threshold, max batch)


class Interrupted(Exception):
    """Raised when SIGINT or SIGTERM arrives."""


def max_batch_for_device(device_str: str) -> int:
    if not device_str.startswith("cuda") or not torch.cuda.is_available():
        return _BATCH_CHOICES[0]
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    for threshold, cap in _VRAM_BATCH_CAP:
        if total_gib >= threshold:
            print(
                f"[sweep] {torch.cuda.get_device_name(0)} {total_gib:.0f}GiB "
                f"-> batch cap {cap}",
                flush=True,
            )
            return cap
    return _BATCH_CHOICES[0]


def ensure_dataset(store: r2_store.R2Store) -> None:
    """Fetch and unpack the latest dataset from R2 if missing or outdated."""
    if not store.enabled:
        return
    latest = store.get_json(store.latest_manifest_key())
    if latest is None:
        return
    remote_hash = latest.get("dataset_hash")
    if not remote_hash:
        return

    local_hash = None
    if config.MANIFEST_PATH.exists():
        try:
            local_hash = json.loads(config.MANIFEST_PATH.read_text()).get("dataset_hash")
        except json.JSONDecodeError:
            local_hash = None

    dataset_ready = (config.DATASET_DIR / config.LABELS[0]).is_dir()
    if dataset_ready and local_hash == remote_hash:
        print(f"[sweep] dataset up to date (hash={local_hash})", flush=True)
        return

    print(
        f"[sweep] syncing dataset from R2 (local={local_hash} -> remote={remote_hash})",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / f"dataset-{remote_hash}.tar"
        tar_key = store.dataset_tar_key(remote_hash)
        if not store.download(tar_key, tar_path):
            raise SystemExit(f"failed to download dataset tar: {store.uri(tar_key)}")
        if config.DATASET_DIR.exists():
            shutil.rmtree(config.DATASET_DIR)
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(config.DATASET_DIR, filter="data")

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not store.download(store.dataset_manifest_key(remote_hash), config.MANIFEST_PATH):
        config.MANIFEST_PATH.write_text(json.dumps(latest, ensure_ascii=False, indent=2))
    print(f"[sweep] dataset synced (hash={remote_hash})", flush=True)


class Sweep:
    def __init__(
        self,
        args: argparse.Namespace,
        store: r2_store.R2Store,
        init_checkpoint: Path | None,
    ) -> None:
        self.args = args
        self.store = store
        self.init_checkpoint = init_checkpoint
        self.pool_dir = config.DATASET_DIR
        if not config.MANIFEST_PATH.exists():
            raise SystemExit(
                f"{config.MANIFEST_PATH} not found — run build_dataset.py or enable R2 sync."
            )
        self.manifest = json.loads(config.MANIFEST_PATH.read_text())
        self.dataset_hash = self.manifest.get("dataset_hash", "unknown")
        self.work_dir = config.RUNS_DIR / args.run_name
        self.tb_root = self.work_dir / "tensorboard"
        self.notify_enabled = not args.smoke
        self.cache: dict[str, torch.Tensor] = {}

        if args.smoke:
            self.device = "cpu"
            self.epochs = 2
            self.n_trials = 2
            self.limit_dates = 2
            self.model_names = ["mobilenetv3_small_100"]
            self.configs = 2
            self.seeds = [101, 202]
            self.allow_default_hparams = True
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.epochs = config.EPOCHS
            self.n_trials = config.HPO_TRIALS
            self.limit_dates = 0
            self.model_names = list(config.HPO_MODEL_NAMES)
            self.configs = config.REFRESH_CONFIGS
            self.seeds = list(config.REFRESH_SEEDS)
            self.allow_default_hparams = False

        self.max_batch = max_batch_for_device(self.device)
        self.batch_choices = [b for b in _BATCH_CHOICES if b <= self.max_batch] or [
            _BATCH_CHOICES[0]
        ]
        self.interrupted = False
        self.completed_this_launch = 0
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)

    # ------------------------------------------------------------------ signals

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True
        print(
            f"[sweep] signal {signum} received: will stop after this unit", flush=True
        )

    def should_stop(self) -> str | None:
        if self.interrupted:
            return "interrupted (SIGINT/SIGTERM)"
        return None

    # ------------------------------------------------------------------- notify

    def notify_text(self, text: str) -> None:
        if self.notify_enabled:
            discord.send_text(text)
        else:
            print(f"[notify] {text}")

    def notify_embed(self, **kwargs) -> None:
        if self.notify_enabled:
            discord.send_embed(**kwargs)
        else:
            print(f"[notify] {kwargs.get('title')}: {kwargs.get('description', '')}")

    # -------------------------------------------------------------------- cache

    def splits(self, *, fold: int | None) -> splitplan.ResolvedSplits:
        resolved = splitplan.resolve(self.manifest, fold=fold)
        if self.limit_dates:
            resolved = splitplan.shrink(resolved, self.limit_dates)
        return resolved

    def prime_cache(self) -> None:
        """Restore the decoded-image cache from R2, or build and publish it."""
        key = self.store.cache_key(self.dataset_hash)
        local = config.CACHE_FILE
        if local.exists():
            print(f"[sweep] using local cache {local}", flush=True)
        elif self.store.download(key, local):
            print(f"[sweep] restored cache from {self.store.uri(key)}", flush=True)
        if local.exists():
            try:
                self.cache = torch.load(local, map_location="cpu", weights_only=True)
                print(f"[sweep] cache entries: {len(self.cache)}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - a bad cache must never be fatal
                print(f"[sweep] cache unreadable ({exc}); rebuilding", flush=True)
                self.cache = {}

        started = time.time()
        total = 0
        dataset = trainer.make_dataset(
            self.pool_dir, cache=self.cache, pattern=config.TRAIN_PATTERN
        )
        if dataset is not None:
            total = dataset.populate_cache()
        print(
            f"[sweep] built cache: {total} images in {time.time() - started:.0f}s",
            flush=True,
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cache, local)
        self.store.upload(local, key)
        self.store.prune_generations("cache", keep=config.KEEP_GENERATIONS)

    def sync_tensorboard_down(self) -> None:
        if not self.store.enabled:
            return
        count = self.store.download_dir(
            f"{self.store.run_prefix}/tensorboard", self.tb_root
        )
        if count:
            print(f"[sweep] downloaded {count} TensorBoard event file(s)", flush=True)

    # ----------------------------------------------------------------- training

    def train_one(
        self,
        *,
        member_id: str,
        hparams: trainer.HParams,
        seed: int,
        group: str,
        splits: splitplan.ResolvedSplits,
        epoch_callback=None,
        extra_meta: dict | None = None,
    ) -> dict[str, Any]:
        """Train one candidate, archive it, and report it. Returns its summary."""
        started = time.time()
        out_dir = self.work_dir / group / member_id
        tb_dir = self.tb_root / member_id
        summary = trainer.run_training(
            pool_dir=self.pool_dir,
            output_dir=out_dir,
            device_str=self.device,
            seed=seed,
            epochs=self.epochs,
            hparams=hparams,
            train_spec=splits.train,
            eval_specs=splits.evals,
            cache=self.cache,
            init_checkpoint=self.init_checkpoint,
            init_member_index=seed,
            epoch_callback=epoch_callback,
            save_probs=True,
            tb_log_dir=tb_dir,
        )
        elapsed = time.time() - started
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["member_id"] = member_id
        summary["group"] = group
        summary["dataset_hash"] = self.dataset_hash
        summary["fold"] = splits.fold
        if extra_meta:
            summary.update(extra_meta)

        self.store.put_trial(member_id, summary)
        self.store.upload_dir(tb_dir, f"{self.store.run_prefix}/tensorboard/{member_id}")

        meta = {
            "member_id": member_id,
            "group": group,
            "model_name": hparams.model_name,
            "labels": list(trainer.LABELS),
            "image_size": hparams.image_size,
            "norm_mean": trainer.NORM_MEAN,
            "norm_std": trainer.NORM_STD,
            "seed": seed,
            "hparams": hparams.to_dict(),
            "val_macro_f1": summary["best_val_macro_f1"],
            "best_epoch": summary["best_epoch"],
            "evals": summary["evals"],
            "dataset_hash": self.dataset_hash,
            "fold": splits.fold,
            "split_dates": summary["split_dates"],
            "split_patterns": summary["split_patterns"],
            "eval_fingerprints": summary["eval_fingerprints"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        fp16 = r2_store.save_fp16_state(
            out_dir / "sleep_classifier.pt", out_dir / "model.fp16.pt"
        )
        uri = self.store.upload_member(
            out_dir, group=group, member_id=member_id, model_fp16=fp16, meta=meta
        )
        summary["r2_model_uri"] = uri
        if self.store.enabled:
            (out_dir / "sleep_classifier.pt").unlink(missing_ok=True)
            fp16.unlink(missing_ok=True)
        else:
            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )
        self.completed_this_launch += 1
        return summary

    def report_member(
        self, summary: dict, *, is_best: bool, progress: str, dropped: list[str]
    ) -> None:
        evals = summary["evals"]
        fields = [
            ("macro-F1", discord.metrics_line(evals), False),
            (
                "hparams",
                "```"
                + json.dumps(
                    {
                        k: v
                        for k, v in summary["hparams"].items()
                        if k not in {"image_size", "patience"}
                    },
                    indent=1,
                )
                + "```",
                False,
            ),
        ]
        if "test_hard" in evals:
            fields.append(
                (
                    "test_hard 混同行列（行=正解）",
                    discord.format_confusion(
                        evals["test_hard"]["confusion"], list(trainer.LABELS)
                    ),
                    False,
                )
            )
        uri = summary.get("r2_model_uri")
        if uri:
            fields.append(("model", f"`{uri}`", False))
        if dropped:
            fields.append(("top-K から追い出した", ", ".join(dropped), False))
        self.notify_embed(
            title=f"{'🏆 新ベスト ' if is_best else ''}{summary['member_id']} 完了",
            description=(
                f"{progress} • best_epoch={summary['best_epoch']} "
                f"• {summary['elapsed_seconds']:.0f}s"
            ),
            fields=fields,
            color=discord.COLOR_GOOD if is_best else discord.COLOR_INFO,
            footer=f"run={self.args.run_name} dataset={self.dataset_hash}",
        )

    # ---------------------------------------------------------------- mode: hpo

    def suggest(self, trial) -> trainer.HParams:
        return trainer.HParams(
            model_name=trial.suggest_categorical("model_name", self.model_names),
            batch_size=trial.suggest_categorical("batch_size", self.batch_choices),
            learning_rate=trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            warmup_epochs=trial.suggest_int("warmup_epochs", 1, 5),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.15),
            drop_path_rate=trial.suggest_float("drop_path_rate", 0.0, 0.4),
            rrc_scale_min=trial.suggest_float("rrc_scale_min", 0.2, 0.7),
            rotation_deg=trial.suggest_float("rotation_deg", 0.0, 15.0),
            jitter_brightness=(js := trial.suggest_float("jitter_strength", 0.0, 0.6)),
            jitter_contrast=js,
            jitter_saturation=js * 0.75,
            grayscale_p=trial.suggest_float("grayscale_p", 0.0, 0.3),
        )

    def run_hpo(self) -> int:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        journal = self.work_dir / "optuna" / "journal.log"
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal_key = f"{self.store.run_prefix}/optuna/journal.log"
        if not journal.exists():
            if self.store.download(journal_key, journal):
                print(f"[sweep] resumed Optuna journal from {journal_key}", flush=True)

        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(journal))
        )
        study = optuna.create_study(
            study_name=self.args.run_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=config.SAMPLER_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        already = len(study.trials)
        splits = self.splits(fold=None)
        print(
            f"[sweep] hpo: {already} trial(s) already in the study, "
            f"target {self.n_trials}",
            flush=True,
        )
        print(splitplan.describe(splits), flush=True)

        def objective(trial) -> float:
            hparams = self.suggest(trial)
            member_id = f"trial{trial.number:04d}"

            def on_epoch(epoch: int, score: float) -> None:
                trial.report(score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"pruned at epoch {epoch} (val {score:.4f})"
                    )

            summary = self.train_one(
                member_id=member_id,
                hparams=hparams,
                seed=config.HPO_SEED,
                group="topk",
                splits=splits,
                epoch_callback=on_epoch,
                extra_meta={"trial_number": trial.number},
            )
            dropped = self.store.evict_topk(config.TOPK_KEEP)
            best_so_far = max(
                [t.value for t in study.trials if t.value is not None] + [float("-inf")]
            )
            score = summary["best_val_macro_f1"]
            self.report_member(
                summary,
                is_best=score > best_so_far,
                progress=(
                    f"trial {trial.number + 1}/{self.n_trials} "
                    f"(this launch: {self.completed_this_launch})"
                ),
                dropped=dropped,
            )
            return score

        stop_reason = None
        while len(study.trials) < self.n_trials:
            stop_reason = self.should_stop()
            if stop_reason:
                break
            try:
                study.optimize(objective, n_trials=1, catch=())
            except Interrupted:
                stop_reason = "interrupted mid-trial"
                break
            finally:
                self.store.upload(journal, journal_key)
                try:
                    self.publish_best_hparams(study, quiet=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[sweep] best_hparams publish skipped: {exc}", flush=True)
            pruned = [
                t
                for t in study.trials[-1:]
                if t.state == optuna.trial.TrialState.PRUNED
            ]
            if pruned:
                self.notify_text(
                    f"`{self.args.run_name}` trial {pruned[0].number} pruned "
                    f"(枝刈り) • 累計 {len(study.trials)}/{self.n_trials}"
                )
        else:
            stop_reason = "target trial count reached"

        self.store.upload(journal, journal_key)
        self.publish_best_hparams(study)
        self.report_finish(stop_reason or "done", total=len(study.trials))
        return 0

    def publish_best_hparams(self, study, *, quiet: bool = False) -> None:
        """Write the current top configs to R2 and local fallback."""
        import optuna

        completed = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not completed:
            if not quiet:
                print("[sweep] no completed trials; best_hparams.json not written")
            return
        completed.sort(key=lambda t: t.value, reverse=True)
        top = completed[: config.REFRESH_CONFIGS]
        payload = {
            "run_name": self.args.run_name,
            "dataset_hash": self.dataset_hash,
            "trials_completed": len(completed),
            "epochs": self.epochs,
            "configs": [
                {
                    "rank": rank,
                    "trial_number": t.number,
                    "val_macro_f1": t.value,
                    "hparams": self._hparams_from_trial(t).to_dict(),
                }
                for rank, t in enumerate(top)
            ],
        }
        self.store.put_json(payload, self.store.best_hparams_key())
        config.BEST_HPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.BEST_HPARAMS_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
        if not quiet:
            print(
                f"[sweep] best_hparams -> {self.store.best_hparams_key()} "
                f"({len(completed)} completed trial(s))",
                flush=True,
            )

    @staticmethod
    def _hparams_from_trial(trial) -> trainer.HParams:
        params = dict(trial.params)
        jitter = params.pop("jitter_strength", 0.4)
        return trainer.HParams(
            jitter_brightness=jitter,
            jitter_contrast=jitter,
            jitter_saturation=jitter * 0.75,
            **params,
        )

    # ------------------------------------------------------------ mode: refresh

    def run_refresh(self) -> int:
        payload = self.store.get_json(self.store.best_hparams_key())
        if payload is None and config.BEST_HPARAMS_PATH.exists():
            payload = json.loads(config.BEST_HPARAMS_PATH.read_text())
        if payload is None:
            if not self.allow_default_hparams:
                self.notify_embed(
                    title="❌ refresh を開始できません",
                    description=(
                        "`best_hparams.json` が R2 にありません。先に "
                        "`--mode hpo` を回してください。"
                    ),
                    color=discord.COLOR_BAD,
                )
                print(
                    "[sweep] no best_hparams.json; refusing to guess", file=sys.stderr
                )
                return 2
            payload = {
                "configs": [
                    {
                        "rank": 0,
                        "val_macro_f1": None,
                        "hparams": trainer.HParams(
                            model_name=self.model_names[0]
                        ).to_dict(),
                    }
                ]
            }

        configs = payload["configs"][: self.configs]
        seeds = self.seeds
        done = self.store.completed_trial_ids()
        plan = [
            (entry, seed, fold, f"cfg{entry['rank']}_seed{seed}_fold{fold}")
            for entry in configs
            for fold, seed in enumerate(seeds)
        ]
        skipped = [m for *_, m in plan if m in done]
        print(
            f"[sweep] refresh: {len(plan)} member(s), {len(skipped)} already done",
            flush=True,
        )
        self.notify_embed(
            title="🔁 refresh 開始",
            description=(
                f"{len(configs)} config × {len(seeds)} seed/fold = {len(plan)} 本 "
                f"（既に完了 {len(skipped)} 本はスキップ）\n"
                f"seed と val fold の両方を振るので、メンバーは2軸で異なります。"
            ),
            fields=[
                ("dataset", f"`{self.dataset_hash}`", True),
                ("device", self.device, True),
                ("batch cap", str(self.max_batch), True),
            ],
            color=discord.COLOR_INFO,
            footer=f"run={self.args.run_name}",
        )

        best = float("-inf")
        stop_reason = "all members complete"
        for index, (entry, seed, fold, member_id) in enumerate(plan, start=1):
            if member_id in done:
                continue
            stop_reason = self.should_stop()
            if stop_reason:
                break
            hparams = trainer.HParams.from_dict(entry["hparams"])
            if hparams.batch_size > self.max_batch:
                hparams = trainer.HParams.from_dict(
                    {**entry["hparams"], "batch_size": self.max_batch}
                )
            splits = self.splits(fold=fold)
            print(splitplan.describe(splits), flush=True)
            summary = self.train_one(
                member_id=member_id,
                hparams=hparams,
                seed=seed,
                group="members",
                splits=splits,
                extra_meta={
                    "config_rank": entry["rank"],
                    "config_val": entry.get("val_macro_f1"),
                },
            )
            score = summary["best_val_macro_f1"]
            self.report_member(
                summary,
                is_best=score > best,
                progress=f"member {index}/{len(plan)} (this launch: {self.completed_this_launch})",
                dropped=[],
            )
            best = max(best, score)
        else:
            stop_reason = "all members complete"

        self.report_finish(stop_reason, total=len(plan))
        return 0

    # ------------------------------------------------------------------- finish

    def report_finish(self, reason: str, *, total: int) -> None:
        resume = (
            f"uv run python train/src/sweep.py --mode {self.args.mode} "
            f"--run-name {self.args.run_name}"
        )
        color = discord.COLOR_WARN if self.interrupted else discord.COLOR_GOOD
        self.notify_embed(
            title="⏹ 停止",
            description=f"理由: **{reason}**",
            fields=[
                ("この launch で完了", f"{self.completed_this_launch} 本", True),
                ("計画本数", str(total), True),
                ("再開コマンド", f"```\n{resume}\n```", False),
            ],
            color=color,
            footer=f"run={self.args.run_name} dataset={self.dataset_hash}",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["hpo", "refresh"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--init-checkpoint-key",
        default=None,
        help="R2 object key of a checkpoint/bundle to warm-start from",
    )
    parser.add_argument(
        "--no-r2",
        action="store_true",
        help="disable R2 reads/writes (local offline run)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="fast CPU preset (2 epochs, 2 trials/configs, 2 dates, mobilenetv3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = r2_store.R2Store(args.run_name, enabled=not (args.no_r2 or args.smoke))
    store.verify()
    ensure_dataset(store)

    if not (config.DATASET_DIR / config.LABELS[0]).is_dir():
        print(
            f"[sweep] image pool missing at {config.DATASET_DIR}; "
            f"run build_dataset.py or upload dataset to R2 first",
            file=sys.stderr,
        )
        return 2

    init_checkpoint: Path | None = None
    if args.init_checkpoint_key:
        if not store.enabled:
            raise SystemExit("--init-checkpoint-key requires R2 (cannot use --no-r2)")
        init_checkpoint = config.BUILD_DIR / "init_checkpoint.pt"
        if not store.download_raw(args.init_checkpoint_key, init_checkpoint):
            raise SystemExit(
                f"init checkpoint not found in R2: {args.init_checkpoint_key}"
            )
        print(
            f"[sweep] downloaded init checkpoint: {args.init_checkpoint_key}",
            flush=True,
        )

    sweep = Sweep(args, store, init_checkpoint)
    store.put_json(
        {
            "mode": args.mode,
            "run_name": args.run_name,
            "dataset_hash": sweep.dataset_hash,
            "epochs": sweep.epochs,
            "device": sweep.device,
            "batch_cap": sweep.max_batch,
            "git_sha": _git_sha(),
        },
        f"{store.run_prefix}/config.json",
    )
    sweep.sync_tensorboard_down()
    sweep.prime_cache()
    return sweep.run_hpo() if args.mode == "hpo" else sweep.run_refresh()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
