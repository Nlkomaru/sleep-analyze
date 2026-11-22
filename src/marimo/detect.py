import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import csv
    import json
    import math
    import subprocess
    from datetime import timedelta
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import timm
    import torch
    from matplotlib.patches import Patch
    from PIL import Image
    from torchvision import transforms

    project_dir = Path(__file__).resolve().parents[2]

    print(project_dir.absolute)
    test_dir = project_dir / "test" / "unlabeled"
    model_dir = project_dir / "sleep_classifier"
    output_dir = project_dir / "output" / "sleep_analysis"
    segment_seconds = 5 * 60
    video_extensions = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
    return (
        Image,
        Patch,
        Path,
        csv,
        json,
        math,
        mo,
        model_dir,
        output_dir,
        plt,
        segment_seconds,
        subprocess,
        test_dir,
        timedelta,
        timm,
        torch,
        transforms,
        video_extensions,
    )


@app.cell
def _(
    Image,
    Patch,
    Path,
    csv,
    json,
    math,
    plt,
    segment_seconds,
    subprocess,
    timedelta,
    timm,
    torch,
    transforms,
):
    def format_duration(seconds: float) -> str:
        return str(timedelta(seconds=round(seconds)))

    def probe_video(video_path: Path) -> dict[str, float | int]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            str(video_path),
        ]
        payload = json.loads(
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        stream = payload["streams"][0]
        return {
            "duration_seconds": float(payload["format"]["duration"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }

    def split_video(
        video_path: Path,
        analysis_dir: Path,
        duration_seconds: float,
    ) -> list[dict[str, object]]:
        segments_dir = analysis_dir / "segments"
        frames_dir = analysis_dir / "frames"
        segments_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)

        segments = []
        for index in range(math.ceil(duration_seconds / segment_seconds)):
            start_seconds = index * segment_seconds
            end_seconds = min((index + 1) * segment_seconds, duration_seconds)
            length_seconds = end_seconds - start_seconds
            segment_path = segments_dir / f"segment_{index + 1:03d}.mp4"
            frame_path = frames_dir / f"segment_{index + 1:03d}.jpg"

            if not segment_path.exists() or segment_path.stat().st_size == 0:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(start_seconds),
                        "-i",
                        str(video_path),
                        "-t",
                        str(length_seconds),
                        "-map",
                        "0:v:0",
                        "-map",
                        "0:a?",
                        "-c",
                        "copy",
                        "-avoid_negative_ts",
                        "make_zero",
                        str(segment_path),
                    ],
                    check=True,
                )

            if not frame_path.exists() or frame_path.stat().st_size == 0:
                center_seconds = start_seconds + length_seconds / 2
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        str(center_seconds),
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "3",
                        str(frame_path),
                    ],
                    check=True,
                )

            segments.append(
                {
                    "index": index + 1,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": length_seconds,
                    "segment_path": segment_path,
                    "frame_path": frame_path,
                }
            )
        return segments

    def load_ensemble(model_paths: list[Path], device: torch.device):
        # Each entry is (display_name, prob_fn) where prob_fn(batch) returns
        # softmax probabilities. A path may be a plain single-model checkpoint
        # (one "state_dict") or an ensemble bundle from finalize_ensemble.py
        # (several "members"). A bundle collapses to ONE entry whose prob_fn
        # averages its members' softmax — i.e. one averaged prediction, not one
        # column per member.
        def make_prob_fn(models: list[object]):
            def prob_fn(batch):
                total = None
                for model in models:
                    probs = torch.softmax(model(batch), dim=1)
                    total = probs if total is None else total + probs
                return total / len(models)

            return prob_fn

        ensemble: list[tuple[str, object]] = []
        expected_labels = None
        expected_image_size = None
        for model_path in model_paths:
            checkpoint = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
            )
            labels = tuple(checkpoint["labels"])
            image_size = int(checkpoint["image_size"])
            if expected_labels is None:
                expected_labels = labels
                expected_image_size = image_size
            if labels != expected_labels or image_size != expected_image_size:
                raise ValueError(
                    f"Model metadata does not match the ensemble: {model_path}"
                )

            if checkpoint.get("ensemble"):
                state_dicts = checkpoint["members"]
            else:
                state_dicts = [checkpoint["state_dict"]]

            models = []
            for state_dict in state_dicts:
                model = timm.create_model(
                    checkpoint["model_name"],
                    pretrained=False,
                    num_classes=len(labels),
                ).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                models.append(model)
            ensemble.append((model_path.stem, make_prob_fn(models)))

        if not ensemble or expected_labels is None or expected_image_size is None:
            raise ValueError("No classifier model was found.")
        return ensemble, expected_labels, expected_image_size

    def classify_segments(
        segments: list[dict[str, object]],
        model_paths: list[Path],
    ) -> tuple[list[dict[str, object]], str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble, labels, image_size = load_ensemble(model_paths, device)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        batch = torch.stack(
            [
                transform(Image.open(segment["frame_path"]).convert("RGB"))
                for segment in segments
            ]
        ).to(device)

        model_probabilities = {}
        with torch.inference_mode():
            for model_name, prob_fn in ensemble:
                model_probabilities[model_name] = prob_fn(batch).cpu()

        results = []
        for segment_index, segment in enumerate(segments):
            per_model = {}
            for model_name, probabilities in model_probabilities.items():
                segment_probabilities = probabilities[segment_index]
                predicted_index = int(segment_probabilities.argmax().item())
                per_model[model_name] = {
                    "predicted_label": labels[predicted_index],
                    "confidence": float(segment_probabilities[predicted_index]),
                    "probabilities": {
                        label: float(probability)
                        for label, probability in zip(
                            labels,
                            segment_probabilities.tolist(),
                        )
                    },
                }
            result = {
                **segment,
                "start": format_duration(float(segment["start_seconds"])),
                "end": format_duration(float(segment["end_seconds"])),
                "model_results": per_model,
            }
            results.append(result)
        return results, str(device)

    def save_results(
        analysis_dir: Path,
        video_path: Path,
        metadata: dict[str, float | int],
        model_paths: list[Path],
        results: list[dict[str, object]],
        device_name: str,
    ) -> tuple[Path, Path]:
        serializable_segments = [
            {
                **result,
                "segment_path": str(result["segment_path"]),
                "frame_path": str(result["frame_path"]),
            }
            for result in results
        ]
        payload = {
            "video_path": str(video_path),
            "duration_seconds": metadata["duration_seconds"],
            "segment_seconds": segment_seconds,
            "device": device_name,
            "models": [str(path) for path in model_paths],
            "segments": serializable_segments,
        }
        json_path = analysis_dir / "analysis.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        csv_path = analysis_dir / "analysis.csv"
        model_names = list(results[0]["model_results"]) if results else []
        labels = (
            list(results[0]["model_results"][model_names[0]]["probabilities"])
            if model_names
            else []
        )
        model_columns = []
        for model_name in model_names:
            model_columns.extend(
                [
                    f"{model_name}__prediction",
                    f"{model_name}__confidence",
                    *[f"{model_name}__probability_{label}" for label in labels],
                ]
            )
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "index",
                    "start",
                    "end",
                    *model_columns,
                    "segment_path",
                    "frame_path",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "index": result["index"],
                        "start": result["start"],
                        "end": result["end"],
                        **{
                            column: value
                            for model_name in model_names
                            for column, value in {
                                f"{model_name}__prediction": result["model_results"][
                                    model_name
                                ]["predicted_label"],
                                f"{model_name}__confidence": result["model_results"][
                                    model_name
                                ]["confidence"],
                                **{
                                    f"{model_name}__probability_{label}": result[
                                        "model_results"
                                    ][model_name]["probabilities"][label]
                                    for label in labels
                                },
                            }.items()
                        },
                        "segment_path": result["segment_path"],
                        "frame_path": result["frame_path"],
                    }
                )
        return json_path, csv_path

    def build_charts(results: list[dict[str, object]]):
        colors = {
            "empty": "#78909c",
            "exist": "#f9a825",
            "sleeping": "#1565c0",
        }
        model_names = list(results[0]["model_results"]) if results else []
        if not model_names:
            figure, axis = plt.subplots(figsize=(10, 2.5))
            axis.text(0.5, 0.5, "No analysis results", ha="center", va="center")
            axis.axis("off")
            return figure

        figure_width = max(12, min(30, len(results) * 0.45))
        figure, axes = plt.subplots(
            len(model_names),
            1,
            figsize=(figure_width, 3.2 * len(model_names)),
            sharex=True,
            squeeze=False,
        )
        x_values = list(range(len(results)))
        tick_step = max(1, math.ceil(len(results) / 16))
        tick_positions = list(range(0, len(results), tick_step))
        if tick_positions[-1] != len(results) - 1:
            tick_positions.append(len(results) - 1)

        for axis, model_name in zip(axes[:, 0], model_names):
            confidence_values = []
            for position, result in enumerate(results):
                model_result = result["model_results"][model_name]
                color = colors.get(
                    str(model_result["predicted_label"]),
                    "#546e7a",
                )
                axis.axvspan(
                    position - 0.5,
                    position + 0.5,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                )
                confidence_values.append(float(model_result["confidence"]))

            axis.plot(
                x_values,
                confidence_values,
                color="#d84315",
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label="confidence",
            )
            axis.set_title(model_name, loc="left", fontsize=11, fontweight="bold")
            axis.set_ylabel("Confidence")
            axis.set_ylim(0, 1.05)
            axis.grid(axis="y", alpha=0.3)

        legend_handles = [
            Patch(facecolor=color, alpha=0.22, label=label)
            for label, color in colors.items()
        ]
        legend_handles.append(axes[0, 0].lines[0])
        axes[0, 0].legend(
            handles=legend_handles,
            loc="lower left",
            ncol=len(legend_handles),
            frameon=False,
        )
        axes[-1, 0].set_xticks(
            tick_positions,
            [str(results[position]["start"]) for position in tick_positions],
            rotation=45,
            ha="right",
        )
        axes[-1, 0].set_xlabel("Segment start")
        axes[-1, 0].set_xlim(-0.5, len(results) - 0.5)
        figure.suptitle(
            "Model-by-model 5-minute classification and confidence",
            fontsize=14,
            fontweight="bold",
        )
        figure.tight_layout()
        return figure

    return (
        build_charts,
        classify_segments,
        format_duration,
        probe_video,
        save_results,
        split_video,
    )


@app.cell
def _(mo, model_dir, test_dir, video_extensions):
    video_paths = (
        sorted(
            path
            for path in test_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in video_extensions
        )
        if test_dir.exists()
        else []
    )

    model_paths = sorted(model_dir.rglob("*.pt")) if model_dir.exists() else []

    video_options = (
        {str(path.relative_to(test_dir)): str(path) for path in video_paths}
        if video_paths
        else {"test/ に動画がありません": ""}
    )
    model_options = {
        str(path.relative_to(model_dir)): str(path) for path in model_paths
    }
    video_select = mo.ui.dropdown(
        options=video_options,
        value=next(iter(video_options)),
        label="分析する動画",
        full_width=True,
    )
    model_select = mo.ui.multiselect(
        options=model_options,
        value=list(model_options),
        label="使用する分類モデル",
        full_width=True,
    )
    analyze_button = mo.ui.run_button(label="5分分割して分析")
    return analyze_button, model_paths, model_select, video_paths, video_select


@app.cell
def _(
    analyze_button,
    mo,
    model_paths,
    model_select,
    video_paths,
    video_select,
):
    mo.vstack(
        [
            mo.md(
                "# Sleep video analysis\n"
                "動画を5分ごとに分割し、各区間の中央フレームを"
                "選択した各モデルで個別に分類して比較します。"
            ),
            video_select,
            model_select,
            analyze_button,
            mo.md(
                f"検出動画: **{len(video_paths)}** / 検出モデル: **{len(model_paths)}**"
            ),
        ]
    )
    return


@app.cell
def _(
    Path,
    analyze_button,
    classify_segments,
    mo,
    model_select,
    output_dir,
    probe_video,
    save_results,
    split_video,
    video_select,
):
    mo.stop(
        not analyze_button.value,
        mo.callout("動画とモデルを指定して分析ボタンを押してください。"),
    )
    mo.stop(
        not video_select.value,
        mo.callout("`test/` 以下に分析対象の動画を追加してください。", kind="warn"),
    )
    mo.stop(
        not model_select.value,
        mo.callout("分類モデルを1つ以上選択してください。", kind="warn"),
    )

    selected_video = Path(video_select.value)
    selected_models = [Path(path) for path in model_select.value]
    metadata = probe_video(selected_video)
    analysis_dir = output_dir / selected_video.stem
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with mo.status.spinner(
        title="動画を5分ごとに分割しています",
        subtitle="既に作成済みの区間は再利用します。",
    ):
        segments = split_video(
            selected_video,
            analysis_dir,
            float(metadata["duration_seconds"]),
        )

    with mo.status.spinner(
        title="睡眠状態を分析しています",
        subtitle=f"{len(segments)}区間 × {len(selected_models)}モデル",
    ):
        results, device_name = classify_segments(segments, selected_models)
        json_path, csv_path = save_results(
            analysis_dir,
            selected_video,
            metadata,
            selected_models,
            results,
            device_name,
        )

    table_rows = [
        {
            "区間": result["index"],
            "開始": result["start"],
            "終了": result["end"],
            **{
                column: value
                for model_name, model_result in result["model_results"].items()
                for column, value in {
                    f"{model_name} 判定": model_result["predicted_label"],
                    f"{model_name} 信頼度": (
                        f"{float(model_result['confidence']):.1%}"
                    ),
                }.items()
            },
        }
        for result in results
    ]
    model_names = list(results[0]["model_results"])
    labels = list(results[0]["model_results"][model_names[0]]["probabilities"])
    summary_lines = []
    for model_name in model_names:
        counts = {
            label: sum(
                result["model_results"][model_name]["predicted_label"] == label
                for result in results
            )
            for label in labels
        }
        count_text = " / ".join(
            f"{label}: **{count}**" for label, count in counts.items()
        )
        summary_lines.append(f"- `{model_name}`: {count_text}")
    summary = "\n".join(summary_lines)
    return (
        analysis_dir,
        csv_path,
        device_name,
        json_path,
        metadata,
        results,
        selected_models,
        selected_video,
        summary,
        table_rows,
    )


@app.cell
def _(
    device_name,
    format_duration,
    metadata,
    mo,
    results,
    selected_models,
    selected_video,
    summary,
):
    mo.md(
        f"""
        ## 分析結果
        - 動画: `{selected_video}`
        - 長さ: **{format_duration(float(metadata['duration_seconds']))}**
        - 区間数: **{len(results)}**（5分単位）
        - モデル数: **{len(selected_models)}**
        - 実行デバイス: **{device_name}**

        ### モデル別の判定数
        {summary}
        """
    )
    return


@app.cell
def _(build_charts, results):
    figure = build_charts(results)
    figure
    return


@app.cell
def _(analysis_dir, csv_path, json_path, mo, table_rows):
    mo.vstack(
        [
            mo.ui.table(table_rows, pagination=True, page_size=20),
            mo.md(
                "### 出力\n"
                f"- 分割動画: `{analysis_dir / 'segments'}`\n"
                f"- 中央フレーム: `{analysis_dir / 'frames'}`\n"
                f"- JSON: `{json_path}`\n"
                f"- CSV: `{csv_path}`"
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
