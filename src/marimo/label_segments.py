import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import shutil
    from pathlib import Path

    import marimo as mo

    result_dir = Path("./output-label/result")
    unlabeled_dir = result_dir / "unlabeled"
    label_names = ("empty", "exist", "sleeping")

    video_dir = Path("./test/unlabeled/")
    segment_seconds = 600
    # 画角プリセット。値は (cx, cy, cw, ch) の比率。None はクロップなし(全画面)。
    # "bed" は従来の 1518x540 相当 (= 0.659*2304 x 0.417*1296)。
    crop_presets = {
        "full (2304x1296)": None,
        "bed (1518x540)": (0.315, 0.56, 0.659, 0.417),
    }
    crop_select = mo.ui.multiselect(
        options=list(crop_presets),
        value=["full (2304x1296)", "bed (1518x540)"],
        label="抽出する画角（複数選択するとデータを混在できます）",
    )
    return (
        crop_presets,
        crop_select,
        label_names,
        mo,
        result_dir,
        segment_seconds,
        shutil,
        unlabeled_dir,
        video_dir,
    )


@app.cell
def _(crop_select, mo):
    mo.vstack(
        [
            mo.md(
                "### 抽出設定\n抽出する画角を選んでから下のセルを有効化して実行してください。"
            ),
            crop_select,
        ]
    )
    return


@app.cell
def _(
    crop_presets,
    crop_select,
    mo,
    segment_seconds,
    unlabeled_dir,
    video_dir,
):
    import math
    import subprocess

    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    selected_modes = list(crop_select.value) or list(crop_presets)
    generated = 0
    for vp in sorted(video_dir.glob("*.mp4")):
        dur = float(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(vp),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        w, h = (
            int(x)
            for x in subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    str(vp),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .split("x")
        )
        for mode in selected_modes:
            ratios = crop_presets[mode]
            # ファイル名に付ける画角タグ ("full" / "bed" など)
            tag = mode.split()[0]
            if ratios is None:
                video_filters: list[str] = []
            else:
                cx, cy, cw, ch = ratios
                crop = (
                    f"crop={min(int(w * cw), w - int(w * cx))}:"
                    f"{min(int(h * ch), h - int(h * cy))}:"
                    f"{int(w * cx)}:{int(h * cy)}"
                )
                video_filters = ["-vf", crop]
            for i in range(math.ceil(dur / segment_seconds)):
                fp = unlabeled_dir / f"{vp.stem}__{i + 1:03d}__{tag}.jpg"
                if fp.exists():
                    continue
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(
                            min(
                                i * segment_seconds + segment_seconds // 2,
                                int(dur) - 1,
                            )
                        ),
                        "-i",
                        str(vp),
                        *video_filters,
                        "-frames:v",
                        "1",
                        "-q:v",
                        "4",
                        str(fp),
                    ],
                    check=True,
                )
                generated += 1
    mo.md(f"Generated **{generated}** new previews ({', '.join(selected_modes)})")
    return


@app.cell
def _(mo):
    back_btn = mo.ui.run_button(label="← back")
    empty_btn = mo.ui.run_button(label="empty")
    exist_btn = mo.ui.run_button(label="exist")
    sleeping_btn = mo.ui.run_button(label="sleeping")
    return back_btn, empty_btn, exist_btn, sleeping_btn


@app.cell
def _(back_btn, empty_btn, exist_btn, mo, progress, sleeping_btn):
    # ボタンは画像と別セルにして画像より上に置く。画像の縦横比が full/bed で
    # 変わってもボタンの表示位置が動かず、押しやすさを保つ。
    mo.vstack(
        [
            progress,
            mo.hstack(
                [back_btn, empty_btn, exist_btn, sleeping_btn],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    get_ver, set_ver = mo.state(0)
    get_history, set_history = mo.state([])
    # run_button の内部カウンタ（_value_frontend）用: True/False だけだと set_ver の再実行で同じクリックが何度も処理される
    get_last_clicks, set_last_clicks = mo.state(
        {"back": 0, "empty": 0, "exist": 0, "sleeping": 0}
    )
    return (
        get_history,
        get_last_clicks,
        get_ver,
        set_history,
        set_last_clicks,
        set_ver,
    )


@app.cell
def _(get_ver, label_names, mo, result_dir, unlabeled_dir):
    _ = get_ver()
    # sort せず一度の走査で「先頭1枚」と「残り枚数」だけを得る（クリックごとの再計算を軽く保つ）
    remaining_files = list(unlabeled_dir.glob("*.jpg"))
    remaining_count = len(remaining_files)
    labeled_count = sum(
        len(list((result_dir / label).glob("*.jpg")))
        for label in label_names
        if (result_dir / label).is_dir()
    )
    current_image = min(remaining_files) if remaining_files else None
    progress = mo.md(
        f"**{remaining_count + labeled_count}** total / "
        f"**{labeled_count}** labeled / "
        f"**{remaining_count}** remaining"
    )
    return current_image, progress


@app.cell
def _(current_image, mo):
    import base64
    import io

    from PIL import Image

    mo.stop(current_image is None, mo.md("## All done!"))
    if current_image.exists():
        # 表示用にダウンスケールして埋め込む。元ファイル（学習データ）は無加工。
        # フルサイズ(2304x1296)を毎クリック base64 で送ると重いので幅 768px に縮小する。
        preview = Image.open(current_image).convert("RGB")
        preview.thumbnail((768, 768), Image.BILINEAR)
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=80)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        image = mo.Html(
            f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%">'
        )
    else:
        image = mo.md("preview not found")
    mo.vstack(
        [
            mo.md(f"### `{current_image.stem}`"),
            image,
        ]
    )
    return


@app.cell
def _(
    back_btn,
    current_image,
    empty_btn,
    exist_btn,
    get_history,
    get_last_clicks,
    get_ver,
    result_dir,
    set_history,
    set_last_clicks,
    set_ver,
    shutil,
    sleeping_btn,
    unlabeled_dir,
):
    def _click_id(btn) -> int:
        v = getattr(btn, "_value_frontend", 0)
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    last = dict(get_last_clicks())
    b, e, x, s = (
        _click_id(back_btn),
        _click_id(empty_btn),
        _click_id(exist_btn),
        _click_id(sleeping_btn),
    )

    _clicked = None
    if b > last["back"]:
        last["back"] = b
        set_last_clicks(last)
        _history = list(get_history())
        if _history:
            _stem, _old_label = _history.pop()
            _src = result_dir / _old_label / f"{_stem}.jpg"
            if _src.exists():
                shutil.move(str(_src), str(unlabeled_dir / f"{_stem}.jpg"))
            set_history(_history)
        set_ver(get_ver() + 1)
    elif e > last["empty"]:
        last["empty"] = e
        set_last_clicks(last)
        if current_image is not None:
            _clicked = "empty"
    elif x > last["exist"]:
        last["exist"] = x
        set_last_clicks(last)
        if current_image is not None:
            _clicked = "exist"
    elif s > last["sleeping"]:
        last["sleeping"] = s
        set_last_clicks(last)
        if current_image is not None:
            _clicked = "sleeping"

    if _clicked is not None and current_image is not None:
        _label_dir = result_dir / _clicked
        _label_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current_image), str(_label_dir / current_image.name))
        _history = list(get_history())
        _history.append((current_image.stem, _clicked))
        set_history(_history)
        set_ver(get_ver() + 1)
    return


if __name__ == "__main__":
    app.run()
