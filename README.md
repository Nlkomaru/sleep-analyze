# sleep-analyze

24 時間の定点カメラ動画から、在床・睡眠状態を判定する睡眠分析アプリケーション。  
画像を `empty` / `exist` / `sleeping` の 3 クラスに分類する。

## ワークフロー

```text
動画からセグメント抽出・ラベリング → labeled/ へ整理・重複除去 → R2 + Google Colab で学習 → アンサンブル化 → 推論
```

| 段階 | コマンド / ノートブック |
| --- | --- |
| ラベリング | `uv run marimo edit src/marimo/label_segments.py` |
| `labeled/` へ移動 | `uv run python src/data/move_output_label_to_labeled.py` |
| 重複除去 | `uv run python src/data/dedup_labeled_images.py near --delete` |
| 学習・再学習（Colab + R2） | [train/README.md](train/README.md) / [train/colab_train.ipynb](train/colab_train.ipynb) |
| 推論（1 枚） | `uv run python src/train/predict_classifier.py` |
| 推論（動画 / Marimo） | `uv run marimo edit src/marimo/detect.py` |
| 推論（日次バッチ / Docker） | `uv run python src/analyze_day.py` |

## ディレクトリ構成

```text
src/
  data/          データ準備（labeled/ への移動・重複除去）
  train/         推論・チェックポイント確認
  marimo/        Marimo ノートブック（ラベリング / 推論 / 閲覧）
  analyze_day.py 日次 mp4 のバッチ推論 CLI
train/           Cloudflare R2 + Google Colab + TensorBoard による学習パイプライン
  src/           データ構築・R2 同期・Sweep 学習・アンサンブル化・採点スクリプト
  colab_train.ipynb  Google Colab 実行用ノートブック
labeled/         ラベル済み画像（empty/ exist/ sleeping/）
```

## セットアップ

Python 3.13 以上と [uv](https://docs.astral.sh/uv/) を使用します。

```bash
uv sync
```

## ラベリングとデータ準備

### セグメント抽出・ラベリング

10 分ごとのセグメント画像を抽出し、Marimo UI 上で `empty` / `exist` / `sleeping` に分類します。

```bash
uv run marimo edit src/marimo/label_segments.py
```

抽出する画角は「抽出設定」のセレクタで選択できます（複数選択可）:

- `full (2304x1296)`: 全画面。日次推論（`analyze_day.py` / `detect.py`）と同じ画角。
- `bed (1518x540)`: ベッド周りの寄りクロップ。

ファイル名には画角タグが付与されます（例: `2026-06-24__001__full.jpg`, `2026-06-24__001__bed.jpg`）。

### `labeled/` への移動

ラベリング結果をクラス別ディレクトリ（`labeled/{empty,exist,sleeping}/`）へ移動します。

```bash
# 変更内容の確認のみ
uv run python src/data/move_output_label_to_labeled.py --dry-run

# 実際に移動
uv run python src/data/move_output_label_to_labeled.py
```

### 重複除去

`dedup_labeled_images.py` に 3 つのサブコマンドを用意しています。`--delete` を外すと削除せずレポートのみ出力します。

```bash
# 連続フレームの近傍重複（知覚的類似度、推奨）
uv run python src/data/dedup_labeled_images.py near --threshold 0.95 --delete

# 完全一致（内容ハッシュ）の重複を一覧表示
uv run python src/data/dedup_labeled_images.py exact

# ファイル名マーカー (*__dup*.jpg) を削除
uv run python src/data/dedup_labeled_images.py marker --delete
```

## 学習・再学習

モデルの学習・定期再学習は **Cloudflare R2 + Google Colab GPU + TensorBoard** のパイプラインで行います。

- ローカルで `labeled/` と修正ラベルからデータセットを構築・検証し、Cloudflare R2 へアップロードします。
- Google Colab 上で [train/colab_train.ipynb](train/colab_train.ipynb) を実行し、Optuna による HPO または固定ハイパラでの Refresh 学習を回します（学習経過は TensorBoard で確認できます）。
- 学習済みメンバーをアンサンブル化し、既存モデルとの比較採点を行って本番用バンドル（`sleep_classifier_ensemble.pt`）を生成します。

詳しいセットアップと運用手順は [train/README.md](train/README.md) を参照してください。

## 推論

### 画像 1 枚の推論

単一モデル・アンサンブルバンドルどちらの `.pt` でも推論できます。

```bash
uv run python src/train/predict_classifier.py path/to/image.jpg --model-path sleep_classifier/sleep_classifier_ensemble.pt
```

### 動画のタイムライン確認（Marimo）

```bash
uv run marimo edit src/marimo/detect.py
```

### 日次バッチ推論（CLI / Docker）

1 日分の mp4 を 3 分ウィンドウで分類し、結果を JSON 出力または取り込み API へ送信します。

```bash
# API へ送信せず JSON を標準出力で確認
uv run python src/analyze_day.py --video path/to/day.mp4 --date 2026-08-01 --dry-run
```

Docker イメージのビルドと実行:

```bash
docker build -t sleep-analyze .
docker run --rm -v /path/to/videos:/videos sleep-analyze --video /videos/2026-08-01.mp4 --date 2026-08-01 --dry-run
```

## 補助ツール

```bash
# ラベル済み画像を一覧・確認する
uv run marimo edit src/marimo/browse_labeled.py

# モデルの test 日付と推論タイムラインを照合する
uv run marimo edit src/marimo/check_classifier_test.py

# 学習済みモデル / アンサンブルに保存された学習・検証・テスト日付を表示する
uv run python src/train/show_classifier_training_data.py path/to/model.pt
```
