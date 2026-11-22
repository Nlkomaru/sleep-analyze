# `train/` — Google Colab + Cloudflare R2 学習・再学習パイプライン

レビューで蓄積された修正ラベルと `labeled/` を統合し、Google Colab GPU + Cloudflare R2 + TensorBoard を使ってモデルを更新・アンサンブル化するパイプラインです。

> 本階層（`train/`）は本番推論コード（`src/analyze_day.py`）から独立しています。チェックポイントのスキーマのみ共通契約を維持しているため、生成された `sleep_classifier_ensemble.pt` をそのまま差し替えられます。

---

## 1. アーキテクチャ

```text
[ローカル環境]
  1. fetch_corrections.py   D1 の修正ラベル + R2 のフレーム画像を取得
  2. build_dataset.py       labeled/ と統合し、bed クロップ生成・split 割当・hard gate 検証
  3. upload_dataset.py      データセット tar と manifest を Cloudflare R2 へアップロード
        │
        ▼ (Cloudflare R2: <R2_TRAIN_PREFIX>/)
[Google Colab (GPU)]
  4. colab_train.ipynb      uv sync → R2 接続確認 → TensorBoard 起動 → sweep.py (hpo / refresh)
        │
        ▼ (R2 へチェックポイント・Optuna journal・TensorBoard ログを逐次同期)
[ローカル環境 / Colab]
  5. finalize_ensemble.py   R2 からメンバーを取得し、貪欲選抜・現本番モデルとの比較採点・bundle 出力
```

---

## 2. 初回セットアップ

### 2.1 ローカル環境（`train/.env`）

`train/.env.example` をコピーして `train/.env` を作成し、各値を設定します（`train/.env` は `.gitignore` 対象です）。

```bash
cp train/.env.example train/.env
```

必須変数:

- `R2_ENDPOINT_URL`: Cloudflare R2 の S3 互換エンドポイント URL
- `R2_ACCESS_KEY_ID`: R2 API トークンの Access Key ID
- `R2_SECRET_ACCESS_KEY`: R2 API トークンの Secret Access Key
- `R2_BUCKET`: 学習データ・成果物を保存する R2 バケット名

任意変数:

- `R2_TRAIN_PREFIX`: バケット内のプレフィックス（既定: `train`）
- `DISCORD_WEBHOOK_URL`: 進捗・完了通知を送る Discord Webhook URL
- `D1_DATABASE_ID`: 修正ラベルを取得する D1 データベースの UUID（`fetch_corrections.py` で必須、`pnpx cf d1 query` を使用）
- `CLOUDFLARE_ACCOUNT_ID`: Cloudflare アカウント ID（`R2_ENDPOINT_URL` から自動抽出できる場合は省略可）
- `PRODUCTION_MODEL_PATH`: 比較採点に使う現本番モデルのパス（既定: `~/data/models/sleep_classifier_ensemble.pt`）

設定後、R2 疎通と Discord 通知を確認します。

```bash
uv run python train/src/r2_store.py check
uv run python train/src/notify_discord.py --self-test
```

### 2.2 Google Colab Secrets

Colab ノートブック（[colab_train.ipynb](colab_train.ipynb)）左側の「シークレット（鍵アイコン）」に以下を登録し、ノートブックからのアクセスを有効化します。

| Secret 名 | 必須 | 説明 |
| --- | --- | --- |
| `R2_ENDPOINT_URL` | 必須 | R2 エンドポイント URL |
| `R2_ACCESS_KEY_ID` | 必須 | R2 Access Key ID |
| `R2_SECRET_ACCESS_KEY` | 必須 | R2 Secret Access Key |
| `R2_BUCKET` | 必須 | R2 バケット名 |
| `R2_TRAIN_PREFIX` | 任意 | バケット内プレフィックス（既定: `train`） |
| `DISCORD_WEBHOOK_URL` | 任意 | Discord Webhook URL |

---

## 3. 定期再学習ワークフロー

### Step 1: 修正ラベルの回収（ローカル）

D1 の修正ラベルと R2 上のフレーム画像を突き合わせてローカルへ取得します。

```bash
# 確認のみ
uv run python train/src/fetch_corrections.py --dry-run

# 取得実行
uv run python train/src/fetch_corrections.py
```

### Step 2: データセット構築と検証（ローカル）

`labeled/` と修正画像を統合し、bed クロップの生成・split 割当・hard gate 検証を実行します。

```bash
uv run python train/src/build_dataset.py
```

`--min-hard-size N` で `test_hard` の最小枚数を変更できます（既定: `80`）。`train/build/report.md` の末尾で gate 通過を確認します。

### Step 3: Cloudflare R2 へアップロード（ローカル）

```bash
# 確認のみ
uv run python train/src/upload_dataset.py --dry-run

# アップロード実行
uv run python train/src/upload_dataset.py
```

### Step 4: Colab で学習 & ローカルでアンサンブル化

1. Google Colab で [colab_train.ipynb](colab_train.ipynb) を開き、GPU ランタイム（T4 等）に接続します。
2. セルを順に実行し、フォームで `MODE`（通常運用は `refresh`、ハイパラ再探索時は `hpo`）と `RUN_NAME` を指定して `sweep.py` を実行します。
3. 学習完了後、ローカル（または Colab の最終セル）でアンサンブル化と比較採点を実行します。

```bash
uv run python train/src/finalize_ensemble.py --run-name <RUN_NAME>
```

単体バンドルのスコアだけを再計測したい場合は `score_bundle.py` を使用します。

```bash
uv run python train/src/score_bundle.py --bundle path/to/sleep_classifier_ensemble.pt
```

---

## 4. 設計メモ

### フラット pool と日付ベースの split

`build_dataset.py` は split ごとに画像をコピーせず、1 枚 1 コピーのフラットな pool（`train/build/dataset/{label}/{date}__{NNN}__{variant}.jpg`）と `train/build/dataset_manifest.json` を生成します。

- 同一日の `full` フレームと `bed` クロップは近重複のため、**日付単位で split** してリークを防ぎます。
- `test_legacy` / `test_full` / `val`（6 月固定分）を固定アンカーとし、新しい修正日のうち `--min-hard-size` に達するまでを `test_hard`、その次の 1 晩を `val`（修正分）、残りを `train` に割り当てます。

### HPO モードと Refresh モード

- `hpo`（`--mode hpo`）: Optuna（TPE + MedianPruner）でハイパーパラメータと `convnext_small` 系バックボーンを探索し、上位 3 config を `best_hparams.json` として R2 に保存します。上位 K=8 のチェックポイントを `topk/` に保持します。
- `refresh`（`--mode refresh`）: `best_hparams.json` の上位 3 config × 6 seed/fold = 18 本を `members/` に学習します。seed ごとに検証 fold をローテーションして多様性を確保し、`finalize_ensemble.py` が `test_full` での貪欲選抜と `test_hard` / `test_legacy` での比較判定を行います。

### Colab 切断からの再開

`sweep.py` は各 trial の完了ごとに Optuna journal（`optuna/journal.log`）と trial メタ情報（`trials/<id>.json`）を R2 へ同期します。Colab のランタイムが途中で切断された場合でも、**同じ `--run-name` で再実行すれば完了済み trial をスキップして続きから再開**できます。

### R2 レイアウト（`<R2_TRAIN_PREFIX>/` 配下）

```text
<R2_TRAIN_PREFIX>/
  best_hparams.json                  hpo が確定した上位 3 config
  data/dataset-<hash>.tar            画像 pool（最新 2 世代を保持）
  data/dataset-<hash>.manifest.json  split 定義マニフェスト
  data/latest.manifest.json          最新データセットのポインタ
  cache/base_image_cache-<hash>.pt   デコード済みベース画像キャッシュ（最新 2 世代を保持）
  runs/<run>/config.json             実行設定
  runs/<run>/optuna/journal.log      Optuna 再開用ジャーナル
  runs/<run>/trials/<id>.json        各 trial のメトリクス
  runs/<run>/topk/<id>/              hpo 上位チェックポイント（K=8）
  runs/<run>/members/<id>/           refresh 学習済みメンバー（18 本）
  runs/<run>/tensorboard/<id>/       TensorBoard イベントログ
  runs/<run>/ensemble/               生成されたアンサンブルバンドル
  report.json                        finalize_ensemble.py の評価レポート
```

### TensorBoard メトリクス

各 trial / メンバーの学習ログは `runs/<run>/tensorboard/<id>/` に出力され、R2 にも同期されます。Colab 上ではノートブックの TensorBoard セル（`%tensorboard`）から loss や `val/macro_f1` などの推移をリアルタイムに確認できます。

### Warm-start（`--init-checkpoint-key`）

同一アーキテクチャの既存チェックポイントから重みを引き継いで学習を始める場合は、`--init-checkpoint-key` に R2 キーを指定します。Colab ノートブックではフォームの `INIT_CHECKPOINT_KEY` に設定します。

### ローカルスモークテスト（`--smoke` / `--no-r2`）

R2 や GPU を使わずにローカルでコード経路だけを通したい場合は、`--no-r2 --smoke` を指定して最小構成で動作確認できます。

```bash
uv run python train/src/sweep.py --mode refresh --run-name smoke --no-r2 --smoke
```
