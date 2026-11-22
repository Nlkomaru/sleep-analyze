"""Archive the built dataset pool and upload it to Cloudflare R2.

Replaces the legacy ``upload_dataset.sh`` script. Refuses to upload if the
dataset build report failed any hard gate.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import config
import r2_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate local dataset and print target R2 keys without uploading",
    )
    args = parser.parse_args(argv)

    manifest_path = config.MANIFEST_PATH
    dataset_dir = config.DATASET_DIR
    report_path = config.BUILD_DIR / "report.md"

    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found — run build_dataset.py first.")
    if not dataset_dir.is_dir():
        raise SystemExit(f"{dataset_dir} not found — run build_dataset.py first.")
    if not report_path.exists():
        raise SystemExit(f"{report_path} not found — run build_dataset.py first.")

    report_text = report_path.read_text()
    if "hard gate 失敗" in report_text:
        raise SystemExit(
            f"refusing to upload: {report_path} indicates hard gate failure."
        )

    manifest = json.loads(manifest_path.read_text())
    dataset_hash = manifest.get("dataset_hash")
    if not dataset_hash:
        raise SystemExit(f"{manifest_path} is missing 'dataset_hash'.")

    store = r2_store.R2Store(enabled=not args.dry_run)
    tar_key = store.dataset_tar_key(dataset_hash)
    manifest_key = store.dataset_manifest_key(dataset_hash)
    latest_key = store.latest_manifest_key()

    if args.dry_run:
        print(f"[upload] DRY RUN dataset_hash={dataset_hash}")
        print(f"[upload] would upload tar -> {tar_key}")
        print(f"[upload] would upload manifest -> {manifest_key}")
        print(f"[upload] would upload latest -> {latest_key}")
        return 0

    store.verify()
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / f"dataset-{dataset_hash}.tar"
        print(f"[upload] archiving {dataset_dir} -> {tar_path} ...", flush=True)
        r2_store.tar_dataset(dataset_dir, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[upload] uploading tar ({size_mb:.1f} MB) -> {store.uri(tar_key)}", flush=True)
        store.upload(tar_path, tar_key)

    print(f"[upload] uploading manifest -> {store.uri(manifest_key)}", flush=True)
    store.upload(manifest_path, manifest_key)
    print(f"[upload] updating latest -> {store.uri(latest_key)}", flush=True)
    store.upload(manifest_path, latest_key)

    deleted = store.prune_generations("data", keep=config.KEEP_GENERATIONS)
    if deleted:
        print(f"[upload] pruned {len(deleted)} old data object(s)", flush=True)
    print(f"[upload] done: dataset_hash={dataset_hash}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
