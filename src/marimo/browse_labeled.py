import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import base64
    import html
    import math
    from collections import Counter, defaultdict
    from pathlib import Path

    import marimo as mo

    project_dir = Path(__file__).resolve().parents[2]
    labeled_dir = project_dir / "labeled"
    labels = ("empty", "exist", "sleeping")
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return (
        Counter,
        base64,
        defaultdict,
        html,
        image_extensions,
        labeled_dir,
        labels,
        math,
        mo,
    )


@app.cell
def _(defaultdict, image_extensions, labeled_dir, labels):
    images_by_date = defaultdict(list)
    for _label in labels:
        _label_dir = labeled_dir / _label
        if not _label_dir.exists():
            continue
        for _image_path in sorted(_label_dir.iterdir()):
            if _image_path.is_file() and _image_path.suffix.lower() in image_extensions:
                _date = _image_path.stem.split("__", 1)[0]
                images_by_date[_date].append((_label, _image_path))

    dates = sorted(images_by_date, reverse=True)
    return dates, images_by_date


@app.cell
def _(dates, mo):
    date_selector = mo.ui.dropdown(
        options=dates,
        value=dates[0] if dates else None,
        label="日付",
        searchable=True,
        full_width=True,
    )
    page_size = mo.ui.dropdown(
        options={"24枚": 24, "48枚": 48, "96枚": 96},
        value="24枚",
        label="1ページの枚数",
    )
    return date_selector, page_size


@app.cell
def _(Counter, date_selector, images_by_date, labels, math, mo, page_size):
    selected_date = date_selector.value
    selected_images = (
        sorted(images_by_date[selected_date], key=lambda item: item[1].name)
        if selected_date
        else []
    )
    counts = Counter(label for label, _path in selected_images)
    total_pages = max(1, math.ceil(len(selected_images) / page_size.value))
    page = mo.ui.number(
        start=1,
        stop=total_pages,
        step=1,
        value=1,
        label=f"ページ / {total_pages}",
    )
    summary = " / ".join(
        [f"合計 **{len(selected_images)}枚**"]
        + [f"`{label}` **{counts[label]}枚**" for label in labels]
    )
    return page, selected_images, summary


@app.cell
def _(date_selector, mo, page, page_size, summary):
    mo.vstack(
        [
            mo.md("# Labeled画像一覧"),
            mo.hstack(
                [date_selector, page_size, page],
                widths=[3, 1, 1],
                align="end",
            ),
            mo.md(summary),
        ]
    )
    return


@app.cell
def _(base64, html, mo, page, page_size, selected_images):
    start = (int(page.value) - 1) * page_size.value
    visible_images = selected_images[start : start + page_size.value]

    cards = []
    for _label, _image_path in visible_images:
        _mime = "image/png" if _image_path.suffix.lower() == ".png" else "image/jpeg"
        _encoded = base64.b64encode(_image_path.read_bytes()).decode("ascii")
        cards.append(
            f"""
            <figure class="labeled-card">
              <img loading="lazy" src="data:{_mime};base64,{_encoded}">
              <figcaption>
                <strong>{html.escape(_label)}</strong>
                <span title="{html.escape(_image_path.name)}">
                  {html.escape(_image_path.name)}
                </span>
              </figcaption>
            </figure>
            """
        )

    gallery = mo.Html(
        f"""
        <style>
          .labeled-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 14px;
          }}
          .labeled-card {{
            margin: 0;
            overflow: hidden;
            border: 1px solid var(--sl-color-gray-5);
            border-radius: 10px;
            background: var(--sl-color-neutral-0);
          }}
          .labeled-card img {{
            display: block;
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: contain;
            background: #111;
          }}
          .labeled-card figcaption {{
            display: flex;
            flex-direction: column;
            gap: 3px;
            padding: 8px 10px;
            font-size: 0.78rem;
          }}
          .labeled-card figcaption span {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            opacity: 0.75;
          }}
        </style>
        <div class="labeled-grid">
          {"".join(cards)}
        </div>
        """
    )
    mo.stop(
        not selected_images,
        mo.callout("選択した日付の画像がありません。", kind="warn"),
    )
    gallery
    return


if __name__ == "__main__":
    app.run()
