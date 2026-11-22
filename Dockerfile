# 睡眠分析バッチ用イメージ。CCTV 日次 mp4 を 3 分ウィンドウで分類し、
# 睡眠取り込み API へ送信する (src/analyze_day.py)。
# GPU は不要なので torch は CPU wheel を使う。
FROM python:3.13-slim

# ffmpeg/ffprobe (フレーム抽出) のみ apt から
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch/torchvision は CPU wheel。timm/pillow は通常の PyPI から。
# (requests は使わず urllib 標準ライブラリで POST するため追加不要)
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.7" "torchvision>=0.22" \
    && pip install --no-cache-dir "timm>=1.0.15" "pillow>=11.0.0"

# 推論コードと学習済みモデル (~177MB) を焼き込む
COPY src/ /app/src/
COPY sleep_classifier/sleep_classifier_ensemble.pt /app/sleep_classifier/sleep_classifier_ensemble.pt

# 既定の model-path は cwd 相対 (=/app/sleep_classifier/...) で解決される。
ENTRYPOINT ["python", "/app/src/analyze_day.py"]
