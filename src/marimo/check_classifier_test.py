import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import html
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import timm
    import torch
    from matplotlib.patches import Patch
    from PIL import Image
    from torchvision import transforms

    project_dir = Path(__file__).resolve().parents[2]
    labeled_dir = project_dir / "labeled"
    model_roots = [project_dir / "sleep_classifier", project_dir / "models"]
    labels = ("empty", "exist", "sleeping")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return (
        Counter,
        Image,
        Patch,
        base64,
        html,
        image_extensions,
        labeled_dir,
        labels,
        mo,
        model_roots,
        plt,
        timm,
        torch,
        transforms,
    )


@app.cell
def _(model_roots):
    model_paths = sorted(
        {path for root in model_roots if root.exists() for path in root.rglob("*.pt")}
    )
    return (model_paths,)


@app.cell
def _(mo, model_paths):
    model_options = {str(path): str(path) for path in model_paths}
    model_select = mo.ui.dropdown(
        options=model_options,
        value=str(model_paths[0]) if model_paths else None,
        label="モデル (.pt)",
        searchable=True,
        full_width=True,
    )
    batch_size = mo.ui.dropdown(
        options={"32": 32, "64": 64, "128": 128, "256": 256},
        value="128",
        label="batch",
    )
    page_size = mo.ui.dropdown(
        options={"24枚": 24, "48枚": 48, "96枚": 96},
        value="24枚",
        label="表示枚数",
    )
    return batch_size, model_select, page_size


@app.cell
def _(batch_size, mo, model_select, page_size):
    mo.vstack(
        [
            mo.md("## controls"),
            mo.hstack(
                [model_select, batch_size, page_size],
                widths=[4, 1, 1],
                align="end",
            ),
        ]
    )
    return


@app.cell
def _(mo, model_paths):
    mo.stop(
        not model_paths,
        mo.callout("`.pt` モデルが見つかりません。", kind="warn"),
    )
    return


@app.cell
def _(Path, model_select, torch):
    model_path = Path(model_select.value)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint_labels = tuple(checkpoint["labels"])
    image_size = int(checkpoint["image_size"])
    fallback_test_dates = [
        "2026-04-12",
        "2026-04-21",
        "2026-05-14",
        "2026-05-18",
        "2026-05-19",
    ]
    test_dates = sorted(checkpoint.get("test_dates") or fallback_test_dates)
    train_dates = sorted(checkpoint.get("train_dates", []))
    val_dates = sorted(checkpoint.get("val_dates", []))
    return (
        checkpoint,
        checkpoint_labels,
        image_size,
        model_path,
        test_dates,
        train_dates,
        val_dates,
    )


@app.cell
def _(
    checkpoint,
    checkpoint_labels,
    image_size,
    mo,
    model_path,
    test_dates,
    train_dates,
    val_dates,
):
    mo.vstack(
        [
            mo.md("# Classifier test date check"),
            mo.md(
                "\n".join(
                    [
                        f"- model: `{model_path}`",
                        f"- architecture: `{checkpoint.get('model_name')}`",
                        f"- labels: `{checkpoint_labels}`",
                        f"- image_size: `{image_size}`",
                        f"- train_dates: {len(train_dates)}",
                        f"- val_dates: {len(val_dates)}",
                        f"- test_dates: {len(test_dates)}",
                    ]
                )
            ),
        ]
    )
    return


@app.cell
def _(image_extensions, labeled_dir, mo, test_dates):
    mo.stop(
        not test_dates,
        mo.callout(
            "この checkpoint には `test_dates` が保存されていません。", kind="warn"
        ),
    )

    test_date_set = set(test_dates)
    test_images = []
    for label_dir in sorted(labeled_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        _label = label_dir.name
        for _image_path in sorted(label_dir.iterdir()):
            if _image_path.suffix.lower() not in image_extensions:
                continue
            _date = _image_path.stem.split("__", 1)[0]
            if _date in test_date_set:
                test_images.append(
                    {
                        "date": _date,
                        "true_label": _label,
                        "path": _image_path,
                    }
                )

    test_images = sorted(
        test_images, key=lambda row: (row["date"], row["true_label"], row["path"].name)
    )
    mo.stop(
        not test_images,
        mo.callout(
            "checkpoint の `test_dates` に一致する labeled 画像がありません。",
            kind="warn",
        ),
    )
    return (test_images,)


@app.cell
def _(
    Image,
    batch_size,
    checkpoint,
    checkpoint_labels,
    image_size,
    test_images,
    timm,
    torch,
    transforms,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=len(checkpoint_labels),
    ).to(device)
    model.eval()

    # An ensemble bundle holds several state_dicts under "members"; a plain
    # checkpoint has a single "state_dict". Both are softmax-averaged uniformly.
    if checkpoint.get("ensemble"):
        state_dicts = checkpoint["members"]
    else:
        state_dicts = [checkpoint["state_dict"]]

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

    results = []
    with torch.inference_mode():
        for _start in range(0, len(test_images), batch_size.value):
            chunk = test_images[_start : _start + batch_size.value]
            batch = torch.stack(
                [transform(Image.open(row["path"]).convert("RGB")) for row in chunk]
            ).to(device)
            prob_sum = None
            for state_dict in state_dicts:
                model.load_state_dict(state_dict)
                member_probs = torch.softmax(model(batch), dim=1)
                prob_sum = member_probs if prob_sum is None else prob_sum + member_probs
            probabilities = (prob_sum / len(state_dicts)).cpu()
            for _row, probs in zip(chunk, probabilities):
                predicted_index = int(probs.argmax().item())
                predicted_label = checkpoint_labels[predicted_index]
                results.append(
                    {
                        **_row,
                        "predicted_label": predicted_label,
                        "correct": predicted_label == _row["true_label"],
                        "confidence": float(probs[predicted_index]),
                        **{
                            f"p_{_label}": float(_prob)
                            for _label, _prob in zip(checkpoint_labels, probs.tolist())
                        },
                    }
                )
    return device, results


@app.cell
def _(Counter, checkpoint_labels, results, test_dates):
    total = len(results)
    correct = sum(row["correct"] for row in results)
    accuracy = correct / total if total else 0.0

    metric_rows = []
    for _label in checkpoint_labels:
        tp = sum(
            row["true_label"] == _label and row["predicted_label"] == _label
            for row in results
        )
        fp = sum(
            row["true_label"] != _label and row["predicted_label"] == _label
            for row in results
        )
        fn = sum(
            row["true_label"] == _label and row["predicted_label"] != _label
            for row in results
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        metric_rows.append(
            {
                "label": _label,
                "support": sum(row["true_label"] == _label for row in results),
                "predicted": sum(row["predicted_label"] == _label for row in results),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = sum(row["f1"] for row in metric_rows) / len(metric_rows)
    by_date_rows = []
    for _date in test_dates:
        rows = [row for row in results if row["date"] == _date]
        true_counts = Counter(row["true_label"] for row in rows)
        pred_counts = Counter(row["predicted_label"] for row in rows)
        by_date_rows.append(
            {
                "date": _date,
                "total": len(rows),
                "accuracy": (
                    sum(row["correct"] for row in rows) / len(rows) if rows else 0.0
                ),
                **{
                    f"true_{_label}": true_counts[_label]
                    for _label in checkpoint_labels
                },
                **{
                    f"pred_{_label}": pred_counts[_label]
                    for _label in checkpoint_labels
                },
            }
        )
    confusion_rows = []
    for _true_label in checkpoint_labels:
        _row = {"true": _true_label}
        for _predicted_label in checkpoint_labels:
            _row[_predicted_label] = sum(
                result["true_label"] == _true_label
                and result["predicted_label"] == _predicted_label
                for result in results
            )
        confusion_rows.append(_row)
    return accuracy, by_date_rows, confusion_rows, macro_f1, metric_rows, total


@app.cell
def _(accuracy, by_date_rows, confusion_rows, device, macro_f1, metric_rows, mo, total):
    def _rounded(rows):
        return [
            {
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in rows
        ]

    mo.vstack(
        [
            mo.md(
                f"device: `{device}` / total: **{total}** / "
                f"accuracy: **{accuracy:.4f}** / macro-F1: **{macro_f1:.4f}**"
            ),
            mo.md("## per-class"),
            mo.ui.table(_rounded(metric_rows), selection=None),
            mo.md("## confusion matrix"),
            mo.ui.table(confusion_rows, selection=None),
            mo.md("## date distribution"),
            mo.ui.table(_rounded(by_date_rows), selection=None),
        ]
    )
    return


@app.cell
def _(Path):
    def capture_order(_path: Path) -> tuple[int, str]:
        _parts = _path.stem.split("__")
        if len(_parts) >= 2 and _parts[1].isdigit():
            return int(_parts[1]), _path.name
        return 10**9, _path.name

    return (capture_order,)


@app.cell
def _(Patch, capture_order, checkpoint_labels, plt, results, test_dates):
    _label_colors = {
        "empty": "#4c78a8",
        "exist": "#f58518",
        "sleeping": "#54a24b",
    }
    _fallback_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    for _index, _label in enumerate(checkpoint_labels):
        _label_colors.setdefault(
            _label, _fallback_colors[_index % len(_fallback_colors)]
        )

    _date_count = max(1, len(test_dates))
    timeline_fig, _axes = plt.subplots(
        _date_count,
        1,
        figsize=(16, max(2.4 * _date_count, 3.0)),
        squeeze=False,
        constrained_layout=True,
    )

    for _axis, _date in zip(_axes[:, 0], test_dates):
        _rows = sorted(
            [result for result in results if result["date"] == _date],
            key=lambda result: capture_order(result["path"]),
        )
        _x_values = list(range(len(_rows)))
        _true_colors = [_label_colors[result["true_label"]] for result in _rows]
        _pred_colors = [_label_colors[result["predicted_label"]] for result in _rows]
        _wrong_x = [
            _x
            for _x, result in zip(_x_values, _rows)
            if result["true_label"] != result["predicted_label"]
        ]

        _axis.scatter(
            _x_values,
            [1] * len(_rows),
            c=_true_colors,
            marker="s",
            s=26,
            linewidths=0,
        )
        _axis.scatter(
            _x_values,
            [0] * len(_rows),
            c=_pred_colors,
            marker="s",
            s=26,
            linewidths=0,
        )
        if _wrong_x:
            _axis.scatter(
                _wrong_x,
                [0] * len(_wrong_x),
                facecolors="none",
                edgecolors="#111111",
                marker="s",
                s=58,
                linewidths=1.1,
            )

        _axis.set_title(f"{_date}  n={len(_rows)}  wrong={len(_wrong_x)}", loc="left")
        _axis.set_yticks([0, 1])
        _axis.set_yticklabels(["predict", "labeled"])
        _axis.set_ylim(-0.7, 1.7)
        _axis.set_xlim(-1, max(1, len(_rows)))
        _axis.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.5)
        _axis.tick_params(axis="x", labelsize=8)

        if _rows:
            _tick_step = max(1, len(_rows) // 12)
            _tick_positions = _x_values[::_tick_step]
            _tick_labels = [
                str(capture_order(_rows[_position]["path"])[0])
                for _position in _tick_positions
            ]
            _axis.set_xticks(_tick_positions)
            _axis.set_xticklabels(_tick_labels, rotation=0)
        _axis.set_xlabel("filename index")

    timeline_fig.legend(
        handles=[
            Patch(facecolor=_label_colors[_label], label=_label)
            for _label in checkpoint_labels
        ]
        + [Patch(facecolor="white", edgecolor="#111111", label="wrong predict")],
        loc="upper right",
    )
    return (timeline_fig,)


@app.cell
def _(mo, timeline_fig):
    mo.vstack(
        [
            mo.md("## labeled / predict timeline"),
            timeline_fig,
        ]
    )
    return


@app.cell
def _(mo):
    only_wrong = mo.ui.checkbox(value=False, label="間違いだけ表示")
    sort_by = mo.ui.dropdown(
        options={
            "日付/ファイル名": "date",
            "低信頼度": "low_confidence",
            "高信頼度": "high_confidence",
        },
        value="日付/ファイル名",
        label="並び順",
    )
    return only_wrong, sort_by


@app.cell
def _(mo, only_wrong, page_size, results, sort_by):
    visible_rows = [
        row for row in results if (not only_wrong.value or not row["correct"])
    ]
    if sort_by.value == "low_confidence":
        visible_rows = sorted(visible_rows, key=lambda row: row["confidence"])
    elif sort_by.value == "high_confidence":
        visible_rows = sorted(
            visible_rows, key=lambda row: row["confidence"], reverse=True
        )
    else:
        visible_rows = sorted(
            visible_rows, key=lambda row: (row["date"], row["path"].name)
        )

    total_pages = max(1, (len(visible_rows) + page_size.value - 1) // page_size.value)
    page = mo.ui.number(
        start=1, stop=total_pages, step=1, value=1, label=f"ページ / {total_pages}"
    )
    return page, visible_rows


@app.cell
def _(mo, only_wrong, page, page_size, sort_by, visible_rows):
    mo.vstack(
        [
            mo.md("## samples"),
            mo.hstack([only_wrong, sort_by, page_size, page], align="end"),
            mo.md(f"表示対象: **{len(visible_rows)}枚**"),
        ]
    )
    return


@app.cell
def _(base64, checkpoint_labels, html, mo, page, page_size, visible_rows):
    _start = (int(page.value) - 1) * page_size.value
    page_rows = visible_rows[_start : _start + page_size.value]

    cards = []
    for _row in page_rows:
        _image_path = _row["path"]
        mime = "image/png" if _image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(_image_path.read_bytes()).decode("ascii")
        status = "ok" if _row["correct"] else "ng"
        probability_text = " ".join(
            f"{_label}:{_row[f'p_{_label}']:.2f}" for _label in checkpoint_labels
        )
        cards.append(
            f"""
            <figure class="check-card {status}">
              <img loading="lazy" src="data:{mime};base64,{encoded}">
              <figcaption>
                <div><strong>{html.escape(_row["true_label"])}</strong> -> <strong>{html.escape(_row["predicted_label"])}</strong> ({_row["confidence"]:.1%})</div>
                <div>{html.escape(_row["date"])} / {html.escape(_image_path.name)}</div>
                <div>{html.escape(probability_text)}</div>
              </figcaption>
            </figure>
            """
        )

    mo.Html(
        f"""
        <style>
          .check-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 12px;
          }}
          .check-card {{
            margin: 0;
            overflow: hidden;
            border: 1px solid var(--sl-color-gray-5);
            border-radius: 8px;
            background: var(--sl-color-neutral-0);
          }}
          .check-card.ng {{
            border-color: #d33;
          }}
          .check-card img {{
            display: block;
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: contain;
            background: #111;
          }}
          .check-card figcaption {{
            display: flex;
            flex-direction: column;
            gap: 3px;
            padding: 8px 10px;
            font-size: 0.78rem;
            line-height: 1.35;
          }}
          .check-card figcaption div {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
        </style>
        <div class="check-grid">
          {"".join(cards)}
        </div>
        """
    )
    return


if __name__ == "__main__":
    app.run()
