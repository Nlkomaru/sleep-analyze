"""Cloudflare R2 artifact store built on ``boto3``.

Supports a no-op ``enabled=False`` mode for offline/smoke runs, plus raw-bucket
helpers for fetching correction frames from ``frames/`` without the training
prefix.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
import torch

import config


def save_fp16_state(src: Path, dst: Path) -> Path:
    """Write a compact fp16 copy of ``sleep_classifier.pt`` for R2 archival."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    fp16_state = {
        name: (value.half() if value.is_floating_point() else value)
        for name, value in checkpoint["state_dict"].items()
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": checkpoint["model_name"],
            "labels": checkpoint["labels"],
            "image_size": checkpoint["image_size"],
            "state_dict": fp16_state,
        },
        dst,
    )
    return dst


def tar_dataset(dataset_dir: Path, tar_path: Path) -> Path:
    """Archive the flat dataset pool into ``tar_path``."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for entry in sorted(dataset_dir.iterdir()):
            tar.add(entry, arcname=entry.name)
    return tar_path


class R2Store:
    """Read/write training artifacts in Cloudflare R2 via S3-compatible API."""

    def __init__(
        self,
        run_name: str = "",
        *,
        enabled: bool = True,
        bucket: str | None = None,
        prefix: str | None = None,
    ) -> None:
        self.run_name = run_name
        self.enabled = enabled
        raw_prefix = (
            config.get_env("R2_TRAIN_PREFIX", config.DEFAULT_R2_TRAIN_PREFIX)
            if prefix is None
            else prefix
        )
        self.prefix = raw_prefix.strip("/")
        if not enabled:
            self.bucket = bucket or ""
            self.client = None
            return

        endpoint_url = config.require_env("R2_ENDPOINT_URL")
        access_key = config.require_env("R2_ACCESS_KEY_ID")
        secret_key = config.require_env("R2_SECRET_ACCESS_KEY")
        self.bucket = bucket or config.require_env("R2_BUCKET")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def _key(self, *parts: str) -> str:
        joined = "/".join(part.strip("/") for part in parts if part and part.strip("/"))
        if not self.prefix:
            return joined
        return f"{self.prefix}/{joined}" if joined else self.prefix

    @property
    def run_prefix(self) -> str:
        return self._key("runs", self.run_name)

    def best_hparams_key(self) -> str:
        return self._key("best_hparams.json")

    def dataset_tar_key(self, dataset_hash: str) -> str:
        return self._key("data", f"dataset-{dataset_hash}.tar")

    def dataset_manifest_key(self, dataset_hash: str) -> str:
        return self._key("data", f"dataset-{dataset_hash}.manifest.json")

    def latest_manifest_key(self) -> str:
        return self._key("data", "latest.manifest.json")

    def cache_key(self, dataset_hash: str) -> str:
        return self._key("cache", f"base_image_cache-{dataset_hash}.pt")

    def uri(self, key: str) -> str:
        return f"r2://{self.bucket}/{key}"

    def verify(self) -> None:
        """Verify credentials and bucket reachability."""
        if not self.enabled:
            return
        assert self.client is not None
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except (ClientError, BotoCoreError) as exc:
            raise SystemExit(
                f"R2 bucket verification failed for {self.bucket!r}: {exc}"
            ) from exc

    def upload(self, local: Path, key: str) -> str:
        if not self.enabled:
            return ""
        assert self.client is not None
        self.client.upload_file(str(local), self.bucket, key)
        return self.uri(key)

    def upload_dir(self, local_dir: Path, key_prefix: str) -> int:
        if not self.enabled or not local_dir.is_dir():
            return 0
        count = 0
        base = key_prefix.strip("/")
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            self.upload(path, f"{base}/{rel}" if base else rel)
            count += 1
        return count

    def download(self, key: str, local: Path) -> bool:
        if not self.enabled:
            return False
        return self.download_raw(key, local)

    def download_raw(self, key: str, local: Path) -> bool:
        """Download an exact bucket key (no ``R2_TRAIN_PREFIX`` added)."""
        if not self.enabled:
            return False
        assert self.client is not None
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(local))
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def download_dir(self, key_prefix: str, local_dir: Path) -> int:
        if not self.enabled:
            return 0
        base = key_prefix.strip("/")
        prefix = f"{base}/" if base else ""
        keys = self.list_keys(prefix)
        count = 0
        for key in keys:
            rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if not rel or rel.endswith("/"):
                continue
            if self.download(key, local_dir / rel):
                count += 1
        return count

    def list_keys(self, prefix: str) -> list[str]:
        if not self.enabled:
            return []
        assert self.client is not None
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def list_bucket_keys(self, prefix: str) -> set[str]:
        """List raw bucket keys under ``prefix`` (no ``R2_TRAIN_PREFIX`` added)."""
        return set(self.list_keys(prefix))

    def delete_prefix(self, prefix: str) -> int:
        if not self.enabled:
            return 0
        assert self.client is not None
        keys = self.list_keys(prefix)
        if not keys:
            return 0
        deleted = 0
        for start in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[start : start + 1000]]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
            deleted += len(batch)
        return deleted

    def put_json(self, data: Any, key: str) -> str:
        if not self.enabled:
            return ""
        assert self.client is not None
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return self.uri(key)

    def get_json(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        assert self.client is not None
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def completed_trial_ids(self) -> set[str]:
        prefix = f"{self.run_prefix}/trials/"
        return {
            Path(key).stem for key in self.list_keys(prefix) if key.endswith(".json")
        }

    def put_trial(self, member_id: str, summary: dict[str, Any]) -> str:
        return self.put_json(summary, f"{self.run_prefix}/trials/{member_id}.json")

    def upload_member(
        self,
        out_dir: Path,
        *,
        group: str,
        member_id: str,
        model_fp16: Path,
        meta: dict[str, Any],
    ) -> str:
        if not self.enabled:
            return ""
        member_prefix = f"{self.run_prefix}/{group}/{member_id}"
        self.put_json(meta, f"{member_prefix}/meta.json")
        model_uri = self.upload(model_fp16, f"{member_prefix}/model.fp16.pt")
        for path in sorted(out_dir.glob("probs_*.npy")):
            self.upload(path, f"{member_prefix}/{path.name}")
        for path in sorted(out_dir.glob("targets_*.npy")):
            self.upload(path, f"{member_prefix}/{path.name}")
        metrics_path = out_dir / "sleep_classifier_metrics.json"
        if metrics_path.exists():
            self.upload(metrics_path, f"{member_prefix}/{metrics_path.name}")
        return model_uri

    def evict_topk(self, keep: int) -> list[str]:
        if not self.enabled:
            return []
        prefix = f"{self.run_prefix}/topk/"
        keys = self.list_keys(prefix)
        member_ids = sorted(
            {
                key[len(prefix) :].split("/", 1)[0]
                for key in keys
                if "/" in key[len(prefix) :]
            }
        )
        if len(member_ids) <= keep:
            return []
        scored: list[tuple[float, str]] = []
        for mid in member_ids:
            meta = self.get_json(f"{prefix}{mid}/meta.json")
            score = float(meta.get("val_macro_f1", -1.0)) if meta else -1.0
            scored.append((score, mid))
        scored.sort(key=lambda item: item[0], reverse=True)
        evict = [mid for _, mid in scored[keep:]]
        for mid in evict:
            self.delete_prefix(f"{prefix}{mid}/")
        return evict

    def prune_generations(
        self, subdir: str, keep: int = config.KEEP_GENERATIONS
    ) -> list[str]:
        """Keep only the ``keep`` newest generations under ``<prefix>/<subdir>/``.

        For ``data``, ``latest.manifest.json``'s ``dataset_hash`` is always protected.
        """
        if not self.enabled:
            return []
        assert self.client is not None
        prefix = self._key(subdir) + "/"
        paginator = self.client.get_paginator("list_objects_v2")
        objects: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects.extend(page.get("Contents", []))

        protected_hash: str | None = None
        if subdir.strip("/") == "data":
            latest = self.get_json(self.latest_manifest_key())
            if isinstance(latest, dict):
                protected_hash = latest.get("dataset_hash")

        gens: dict[str, tuple[Any, list[str]]] = {}
        for obj in objects:
            key = obj["Key"]
            name = key[len(prefix) :]
            if not name or name == "latest.manifest.json":
                continue
            gen_id = _generation_id(name)
            if gen_id is None:
                continue
            ts = obj["LastModified"]
            prev_ts, key_list = gens.get(gen_id, (ts, []))
            key_list.append(key)
            gens[gen_id] = (max(prev_ts, ts), key_list)

        ordered = sorted(gens.items(), key=lambda kv: kv[1][0], reverse=True)
        keep_ids = {gen_id for gen_id, _ in ordered[:keep]}
        if protected_hash:
            keep_ids.add(protected_hash)

        to_delete = [
            key
            for gen_id, (_, key_list) in ordered
            if gen_id not in keep_ids
            for key in key_list
        ]
        if not to_delete:
            return []
        for start in range(0, len(to_delete), 1000):
            batch = [{"Key": k} for k in to_delete[start : start + 1000]]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
        return to_delete


def _generation_id(filename: str) -> str | None:
    if filename.startswith("dataset-") and filename.endswith(".tar"):
        return filename[len("dataset-") : -len(".tar")]
    if filename.startswith("dataset-") and filename.endswith(".manifest.json"):
        return filename[len("dataset-") : -len(".manifest.json")]
    if filename.startswith("base_image_cache-") and filename.endswith(".pt"):
        return filename[len("base_image_cache-") : -len(".pt")]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="verify R2 credentials and bucket reachability")
    args = parser.parse_args(argv)

    if args.command == "check":
        store = R2Store()
        store.verify()
        print(
            f"[r2] OK: bucket={store.bucket!r} prefix={store.prefix!r} "
            f"endpoint={config.get_env('R2_ENDPOINT_URL')}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
