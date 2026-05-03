import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import evoc

    return (evoc,)


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    from wigglystuff import ParallelCoordinates
    import marimo as mo
    import polars as pl
    import random
    from copy import deepcopy

    return ParallelCoordinates, Path, deepcopy, mo, np, pl, random


@app.cell
def _(Path, np):
    base_folder = Path("/Users/rikard/Google Drive/Memes")
    image_feature_file = base_folder / "__siglip2_image_features.npz"
    all_image_features = np.load(image_feature_file, allow_pickle=True)["arr_0"].item()
    return all_image_features, base_folder


@app.cell
def _(all_image_features, base_folder, np):
    keys = list(all_image_features.keys())
    values = np.array([all_image_features[k] for k in keys])

    img_paths = [base_folder / path for path in keys]
    values.shape
    return keys, values


@app.cell
def _(evoc, values):
    clusterer = evoc.EVoC()
    clusters = clusterer.fit_predict(values)
    return (clusterer,)


@app.cell
def _(clusterer, pl):
    df = pl.DataFrame(
        {
            f"Layer_{len(clusterer.cluster_layers_) - i}": v
            for i, v in enumerate(clusterer.cluster_layers_)
        }
    )
    return (df,)


@app.cell
def _(ParallelCoordinates, df, mo):
    widget_inner = ParallelCoordinates(df, height=500, color_by="label")
    widget = mo.ui.anywidget(widget_inner)
    widget
    return (widget,)


@app.cell
def _(keys, widget):
    selected_img_paths = [keys[i] for i in widget.selected_indices]
    return (selected_img_paths,)


@app.cell
def _(base_folder, deepcopy, mo, random, selected_img_paths):
    imgs_to_display = deepcopy(selected_img_paths)
    random.shuffle(imgs_to_display)
    imgs_to_display = imgs_to_display[: 3 * 3]
    rows = [imgs_to_display[i : i + 3] for i in range(0, len(imgs_to_display), 3)]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.image(str(base_folder / path), width=300, height=300)
                    for path in row
                ]
            )
            for row in rows
        ]
    )
    return


@app.cell
def _():
    import torch
    from open_clip import create_model_from_pretrained, get_tokenizer

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model, preprocess = create_model_from_pretrained(
        "hf-hub:timm/ViT-SO400M-14-SigLIP2"
    )
    model = model.to(device)
    tokenizer = get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP2")
    return device, model, preprocess, tokenizer, torch


@app.cell
def _(mo):
    search_query = mo.ui.text(
        placeholder="Search for images...",
        label="Search query",
        full_width=True,
    )
    search_query
    return (search_query,)


@app.cell
def _(
    Path,
    device,
    keys,
    mo,
    model,
    np,
    search_query,
    tokenizer,
    torch,
    values,
):
    import base64

    _query = search_query.value
    mo.stop(not _query, mo.md("Enter a search query above."))

    _tokens = tokenizer([_query]).to(device)
    with torch.no_grad(), torch.autocast(device.type):
        _text_features = model.encode_text(_tokens, normalize=True)

    _text_features_np = _text_features.squeeze().cpu().numpy()
    _similarities = values @ _text_features_np

    _top_indices = np.argsort(_similarities)[::-1][:9]

    # Normalise all similarities (full set) to [-1, 1] via min-max
    _s_min, _s_max = _similarities.min(), _similarities.max()

    def _normalise(s):
        if _s_max == _s_min:
            return 0.0
        return float(2.0 * (s - _s_min) / (_s_max - _s_min) - 1.0)

    def _render(path, score):
        try:
            from PIL import Image
            import io

            _img = Image.open(path).convert("RGB")
            _img.thumbnail((300, 300), Image.LANCZOS)
            _buf = io.BytesIO()
            _img.save(_buf, format="JPEG", quality=80)
            _data = base64.b64encode(_buf.getvalue()).decode()
            return mo.Html(
                f'<div style="display:inline-block;width:300px;text-align:center;">'
                f'<img src="data:image/jpeg;base64,{_data}" style="width:300px;height:300px;object-fit:cover;display:block;">'
                f'<div style="background:#222;color:white;padding:4px 8px;font-size:14px;font-weight:bold;">'
                f"{score:.4f}</div></div>"
            )
        except Exception:
            return mo.md(f"_(error loading {Path(path).name})_")

    _items = [_render(keys[i], _normalise(_similarities[i])) for i in _top_indices]
    mo.vstack([mo.hstack(_items[_r * 3 : _r * 3 + 3]) for _r in range(3)])
    return


@app.cell
def _(mo):
    upload_image = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        label="Upload an image to search for similar images",
    )
    upload_image
    return (upload_image,)


@app.cell
def _(
    Path,
    device,
    keys,
    mo,
    model,
    np,
    preprocess,
    torch,
    upload_image,
    values,
):
    import base64 as _base64
    import io as _io
    from PIL import Image as _PILImage

    mo.stop(
        not upload_image.value,
        mo.md("Upload an image above to find similar images."),
    )

    _file = upload_image.value[0]
    _uploaded_img = _PILImage.open(_io.BytesIO(_file.contents)).convert("RGB")
    _img_tensor = preprocess(_uploaded_img).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device.type):
        _img_features = model.encode_image(_img_tensor, normalize=True)

    _img_features_np = _img_features.squeeze().cpu().numpy()
    _img_similarities = values @ _img_features_np

    _top_indices = np.argsort(_img_similarities)[::-1][:9]
    _s_min, _s_max = _img_similarities.min(), _img_similarities.max()

    def _normalise(s):
        if _s_max == _s_min:
            return 0.0
        return float(2.0 * (s - _s_min) / (_s_max - _s_min) - 1.0)

    def _render_img(path, score):
        try:
            _ri = _PILImage.open(path).convert("RGB")
            _ri.thumbnail((300, 300), _PILImage.LANCZOS)
            _buf = _io.BytesIO()
            _ri.save(_buf, format="JPEG", quality=80)
            _data = _base64.b64encode(_buf.getvalue()).decode()
            return mo.Html(
                f'<div style="display:inline-block;width:300px;text-align:center;">'
                f'<img src="data:image/jpeg;base64,{_data}" style="width:300px;height:300px;object-fit:cover;display:block;">'
                f'<div style="background:#222;color:white;padding:4px 8px;font-size:14px;font-weight:bold;">'
                f"{score:.4f}</div></div>"
            )
        except Exception:
            return mo.md(f"_(error loading {Path(path).name})_")

    _items = [
        _render_img(keys[i], _normalise(_img_similarities[i])) for i in _top_indices
    ]
    mo.vstack([mo.hstack(_items[_r * 3 : _r * 3 + 3]) for _r in range(3)])
    return


if __name__ == "__main__":
    app.run()
