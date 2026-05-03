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
    from wigglystuff import ParallelCoordinates
    import marimo as mo
    import polars as pl
    import random
    from copy import deepcopy

    return ParallelCoordinates, Path, deepcopy, mo, pl, random


@app.cell
def _(Path):
    from image_retrieval import FeatureStore

    base_folder = Path("/Users/rikard/Google Drive/Memes")
    store = FeatureStore.load(base_folder / "__siglip2_image_features.npz")
    return base_folder, store


@app.cell
def _(store):
    keys = store.keys
    values = store.values
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

    def _clickable_image(path):
        full_path = base_folder / path
        img = mo.image(str(full_path), width=300, height=300)
        return mo.Html(
            f'<a href="{full_path.as_uri()}" target="_blank" title="Open original" style="display:inline-block;cursor:pointer;">{img.text}</a>'
        )

    mo.vstack([mo.hstack([_clickable_image(path) for path in row]) for row in rows])
    return


@app.cell
def _():
    from image_retrieval import (
        get_device,
        load_siglip2,
        encode_text,
        encode_image,
        normalize_scores,
        rank_by_similarity,
        render_result_card,
    )

    device = get_device()
    model, preprocess, tokenizer = load_siglip2(device=device)
    return (
        device,
        encode_image,
        encode_text,
        model,
        normalize_scores,
        preprocess,
        rank_by_similarity,
        render_result_card,
        tokenizer,
    )


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
    device,
    encode_text,
    keys,
    mo,
    model,
    normalize_scores,
    rank_by_similarity,
    render_result_card,
    search_query,
    tokenizer,
    values,
):
    _query = search_query.value
    mo.stop(not _query, mo.md("Enter a search query above."))

    _text_features_np = encode_text(model, tokenizer, device, _query)
    _top_indices, _similarities = rank_by_similarity(values, _text_features_np)
    _norm_scores = normalize_scores(_similarities)

    _items = [
        mo.Html(render_result_card(keys[i], float(_norm_scores[i])))
        for i in _top_indices
    ]
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
    device,
    encode_image,
    keys,
    mo,
    model,
    normalize_scores,
    preprocess,
    rank_by_similarity,
    render_result_card,
    upload_image,
    values,
):
    import io
    from PIL import Image

    mo.stop(
        not upload_image.value,
        mo.md("Upload an image above to find similar images."),
    )

    _file = upload_image.value[0]
    _uploaded_img = Image.open(io.BytesIO(_file.contents)).convert("RGB")
    _img_features_np = encode_image(model, preprocess, device, _uploaded_img)
    _top_indices, _img_similarities = rank_by_similarity(values, _img_features_np)
    _norm_scores = normalize_scores(_img_similarities)

    _items = [
        mo.Html(render_result_card(keys[i], float(_norm_scores[i])))
        for i in _top_indices
    ]
    mo.vstack([mo.hstack(_items[_r * 3 : _r * 3 + 3]) for _r in range(3)])
    return


if __name__ == "__main__":
    app.run()
