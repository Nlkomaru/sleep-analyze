from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="学習済み分類モデルのチェックポイントメタデータを表示します。"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="確認する .pt チェックポイントのパス",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for model_path in args.paths:
        print(f"\n=== {model_path} ===")

        if not model_path.exists():
            print("not found")
            continue

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        print(f"model_name: {checkpoint.get('model_name', '-')}")
        print(f"ensemble:   {checkpoint.get('ensemble', False)}")
        print(f"seed:       {checkpoint.get('seed', '-')}")
        print(f"epochs:     {checkpoint.get('epochs', '-')}")
        print(f"best_epoch: {checkpoint.get('best_epoch', '-')}")
        print(f"best_f1:    {checkpoint.get('best_f1', '-')}")
        print(f"train:      {checkpoint.get('train_dates', [])}")
        print(f"validation: {checkpoint.get('val_dates', [])}")
        print(f"test:       {checkpoint.get('test_dates', [])}")


if __name__ == "__main__":
    main()
