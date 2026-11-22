"""1日分の結合 mp4 を 3 分ウィンドウで empty/exist/sleeping 分類し、

睡眠取り込み API (`/api/record/sleep`) へ POST する
ヘッドレス CLI。marimo 非依存。

フレーム抽出は ffmpeg の `fps=1/<segment_seconds>` フィルタで 1 パス（R2/FUSE
上の mp4 を seek せず順次 1 回だけ読む）。各フレームをアンサンブル
(`sleep_classifier_ensemble.pt`) で softmax 平均分類し、ウィンドウ列を作る。
連続する同ラベルのウィンドウは 1 セグメントに結合してから送信する。

結合 mp4 は絶対時刻を持たないため、各ウィンドウの実時刻は `--anchor`
(既定: 録画日 15:00 JST) を t=0 とした近似。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

JST = timezone(timedelta(hours=9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a daily mp4 into empty/exist/sleeping segments "
        "and POST them to the sleep ingest API."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--date",
        required=True,
        help="録画日ラベル YYYY-MM-DD (取り込み API の upsert キー)",
    )
    parser.add_argument("--camera", default="", help="source.camera に入れるカメラ名")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("sleep_classifier/sleep_classifier_ensemble.pt"),
    )
    parser.add_argument(
        "--model-version",
        default="",
        help="source.version (モデル更新日 or git sha)。未指定なら .pt の mtime",
    )
    parser.add_argument("--segment-seconds", type=int, default=180)
    parser.add_argument(
        "--anchor",
        default="",
        help="ウィンドウ t=0 の実時刻 (ISO8601)。未指定なら <date>T15:00:00+09:00",
    )
    parser.add_argument("--endpoint", default="", help="取り込み API のベース URL")
    parser.add_argument("--user-id", default="", help="クエリ user_id (clerk user id)")
    parser.add_argument("--token", default="", help="クエリ token (JWT)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="ffmpeg 抽出 / torch スレッドの並列数 (~CPU 数)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="POST せず生成 JSON を標準出力するだけ",
    )
    return parser.parse_args()


def probe_duration_seconds(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _extract_chunk(
    video_path: Path,
    out_dir: Path,
    start_seconds: int,
    length_seconds: int,
    segment_seconds: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # 入力前の -ss は高速シーク、-skip_frame nokey は I フレームのみデコード
            "-ss",
            str(start_seconds),
            "-skip_frame",
            "nokey",
            "-i",
            str(video_path),
            "-t",
            str(length_seconds),
            "-an",
            "-vf",
            f"fps=1/{segment_seconds}",
            "-q:v",
            "3",
            str(out_dir / "frame_%05d.jpg"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("frame_*.jpg"))


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    segment_seconds: int,
    duration_seconds: float,
    jobs: int,
) -> list[Path]:
    """fps=1/segment_seconds でフレームを抽出。ウィンドウ i のフレーム列を返す。

    `-skip_frame nokey` で I フレームだけをデコードし、さらにタイムラインを
    `jobs` 個のチャンク(境界は segment_seconds の倍数に揃える)へ分けて並列
    ffmpeg 実行することで ~`jobs` コアを使い切る。fps フィルタが受け取るのは
    I フレームだけになるが、各スロットに最も近いものが 1 枚採られる(数フレーム
    のズレは許容)。チャンクを順に連結すれば ウィンドウ i → frame i の規則的な
    対応は保たれる。
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    total_windows = math.ceil(duration_seconds / segment_seconds)
    jobs = max(1, jobs)
    chunk_windows = math.ceil(total_windows / jobs)

    chunks = []
    for k, start_window in enumerate(range(0, total_windows, chunk_windows)):
        start_seconds = start_window * segment_seconds
        window_count = min(chunk_windows, total_windows - start_window)
        length_seconds = min(
            window_count * segment_seconds,
            math.ceil(duration_seconds) - start_seconds,
        )
        chunks.append((k, start_seconds, length_seconds))

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(
            pool.map(
                lambda c: _extract_chunk(
                    video_path,
                    frames_dir / f"chunk_{c[0]:02d}",
                    c[1],
                    c[2],
                    segment_seconds,
                ),
                chunks,
            )
        )

    frames: list[Path] = []
    for chunk_frames in results:  # results は chunks の順序を保持する
        frames.extend(chunk_frames)
    return frames


def load_ensemble(model_path: Path, device: torch.device):
    # ensemble bundle は "members" に複数 state_dict を持つ。plain checkpoint は
    # "state_dict" 1 つ。どちらも softmax 平均で 1 つの予測にまとめる
    # (predict_classifier.py と同じ流儀)。bundle は非テンソルを含むため
    # weights_only=False (detect.py と同じ)。
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    labels = list(checkpoint["labels"])
    image_size = int(checkpoint["image_size"])
    if checkpoint.get("ensemble"):
        state_dicts = checkpoint["members"]
    else:
        state_dicts = [checkpoint["state_dict"]]
    model = (
        timm.create_model(
            checkpoint["model_name"], pretrained=False, num_classes=len(labels)
        )
        .to(device)
        .eval()
    )
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return model, state_dicts, labels, transform


@torch.no_grad()
def classify_frames(
    frame_paths: list[Path],
    model,
    state_dicts,
    transform,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """全フレームの平均 softmax 確率 [N, num_classes] を返す。"""
    tensors = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
    batch_all = torch.stack(tensors).to(device)
    prob_chunks = []
    for start in range(0, batch_all.shape[0], batch_size):
        batch = batch_all[start : start + batch_size]
        prob_sum = None
        for state_dict in state_dicts:
            model.load_state_dict(state_dict)
            member_probs = torch.softmax(model(batch), dim=1)
            prob_sum = member_probs if prob_sum is None else prob_sum + member_probs
        prob_chunks.append(prob_sum / len(state_dicts))
    return torch.cat(prob_chunks, dim=0).cpu()


def build_segments(
    probs: torch.Tensor,
    labels: list[str],
    anchor: datetime,
    segment_seconds: int,
) -> list[dict]:
    """ウィンドウごとに分類し、連続する同ラベルを 1 セグメントに結合する。"""
    windows = []
    for i in range(probs.shape[0]):
        prob_row = probs[i]
        label = labels[int(prob_row.argmax().item())]
        windows.append(
            {
                "index": i,
                "label": label,
                "confidence": float(prob_row.max().item()),
                "probabilities": {
                    lbl: float(prob_row[j].item()) for j, lbl in enumerate(labels)
                },
            }
        )

    segments: list[dict] = []
    for w in windows:
        last = segments[-1] if segments else None
        if last is not None and last["_label"] == w["label"] and w["index"] == last["_end_index"]:
            # 直前セグメントを延長して確率/確信度を平均に更新
            last["_end_index"] = w["index"] + 1
            last["_confidence_sum"] += w["confidence"]
            for lbl in labels:
                last["_prob_sum"][lbl] += w["probabilities"][lbl]
            last["_count"] += 1
        else:
            segments.append(
                {
                    "_label": w["label"],
                    "_start_index": w["index"],
                    "_end_index": w["index"] + 1,
                    "_confidence_sum": w["confidence"],
                    "_prob_sum": dict(w["probabilities"]),
                    "_count": 1,
                }
            )

    result = []
    for s in segments:
        start = anchor + timedelta(seconds=s["_start_index"] * segment_seconds)
        end = anchor + timedelta(seconds=s["_end_index"] * segment_seconds)
        count = s["_count"]
        result.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "label": s["_label"],
                "confidence": round(s["_confidence_sum"] / count, 6),
                "probabilities": {
                    lbl: round(s["_prob_sum"][lbl] / count, 6) for lbl in labels
                },
            }
        )
    return result


def post_payload(endpoint: str, user_id: str, token: str, payload: dict) -> None:
    url = f"{endpoint}?user_id={urllib.parse.quote(user_id)}&token={urllib.parse.quote(token)}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            # Cloudflare WAF が既定の Python-urllib UA を 1010 で弾くため UA を付ける
            "User-Agent": "sleep-analyze/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
        print(f"ingest ok: HTTP {response.status} {body}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    if args.anchor:
        anchor = datetime.fromisoformat(args.anchor)
    else:
        anchor = datetime.fromisoformat(f"{args.date}T15:00:00").replace(tzinfo=JST)

    if args.model_version:
        version = args.model_version
    else:
        version = datetime.fromtimestamp(
            args.model_path.stat().st_mtime, tz=JST
        ).strftime("%Y-%m-%d")

    torch.set_num_threads(max(1, args.jobs))
    device = torch.device(args.device)
    duration = probe_duration_seconds(args.video)

    frames_dir = Path("/work/frames") if Path("/work").is_dir() else args.video.parent / "_frames"
    frame_paths = extract_frames(
        args.video, frames_dir, args.segment_seconds, duration, args.jobs
    )
    if not frame_paths:
        print(f"no frames extracted from {args.video} (duration={duration}s)", file=sys.stderr)
        return 1

    model, state_dicts, labels, transform = load_ensemble(args.model_path, device)
    probs = classify_frames(
        frame_paths, model, state_dicts, transform, device, args.batch_size
    )
    segments = build_segments(probs, labels, anchor, args.segment_seconds)

    payload = {
        "date": args.date,
        "timezone": "Asia/Tokyo",
        "source": {
            "model": "sleep_classifier_ensemble",
            "version": version,
            "camera": args.camera,
        },
        "segments": segments,
    }

    if args.dry_run or not args.endpoint:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not args.user_id or not args.token:
        print("--user-id and --token are required to POST", file=sys.stderr)
        return 1

    try:
        post_payload(args.endpoint, args.user_id, args.token, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        print(f"ingest failed: HTTP {exc.code} {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"ingest failed: {exc.reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
