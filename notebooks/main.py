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

    return ParallelCoordinates, Path, mo, np, pl


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
    df = pl.DataFrame({
        f"Layer_{len(clusterer.cluster_layers_) - i}": v for i, v in enumerate(clusterer.cluster_layers_)
    })
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
def _(selected_img_paths):
    imgs_to_display = selected_img_paths[3 * 3]
    return


if __name__ == "__main__":
    app.run()
