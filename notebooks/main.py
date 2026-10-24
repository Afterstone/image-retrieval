import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import evoc
    import marimo as mo
    import polars as pl
    from wigglystuff import ParallelCoordinates

    from image_retrieval.config import load_settings
    from image_retrieval.images import load_image
    from image_retrieval.notebook import NotebookSession
    from image_retrieval.render import render_result_card
    from image_retrieval.search import normalize_scores, search

    settings = load_settings()
    session = NotebookSession(settings)
    return (
        ParallelCoordinates,
        evoc,
        load_image,
        mo,
        normalize_scores,
        pl,
        render_result_card,
        search,
        session,
        settings,
    )


@app.cell
def _(mo, settings):
    profile_selector = mo.ui.dropdown(
        list(settings.profiles), value=settings.default_profile, label="Model profile"
    )
    corpus_selector = mo.ui.dropdown(
        list(settings.corpora), value=settings.default_corpus, label="Collection"
    )
    reload_embeddings = mo.ui.run_button(label="Reload embeddings")
    mo.vstack(
        [
            mo.md(
                "# Local image retrieval\nThe CLI indexes; this notebook only reads committed snapshots."
            ),
            mo.hstack([profile_selector, corpus_selector, reload_embeddings]),
        ]
    )
    return corpus_selector, profile_selector, reload_embeddings


@app.cell
def _(corpus_selector, profile_selector, session, settings):
    session.select(profile_selector.value, corpus_selector.value)
    selected_space = (profile_selector.value, corpus_selector.value)
    _, collection = settings.select(*selected_space)
    return collection, selected_space


@app.cell
def _(mo, reload_embeddings, selected_space, session):
    reload_embeddings.value
    selected_space
    snapshot = None
    snapshot_error = None
    try:
        snapshot = session.reload()
    except (OSError, ValueError) as _exc:
        snapshot_error = str(_exc)
    mo.stop(
        snapshot is None,
        mo.vstack(
            [
                mo.md(f"**Embeddings unavailable:** {snapshot_error}"),
                mo.md(
                    f"Index this space in a terminal, then click **Reload embeddings**:\n\n`{session.indexing_guidance()}`"
                ),
            ]
        ),
    )
    assert snapshot is not None  # mo.stop terminates the cell on a missing snapshot.
    mo.md(
        "**Snapshot: complete**"
        if snapshot.complete
        else "**Snapshot: incomplete — retained paths may be stale. Resume indexing with the CLI.**"
    )
    return (snapshot,)


@app.cell
def _(snapshot):
    keys = snapshot.keys
    values = snapshot.values
    return keys, values


@app.cell
def _(mo, selected_space):
    selected_space
    search_query = mo.ui.text(
        placeholder="Search for images…", label="Text query", full_width=True
    )
    upload_image = mo.ui.file(
        filetypes=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"],
        label="Image query",
    )
    mo.vstack(
        [
            search_query,
            upload_image,
            mo.md(
                "Scores are query-relative min-max values in [-1, 1], **not probabilities**. Do not compare scores across models."
            ),
        ]
    )
    return search_query, upload_image


@app.cell
def _(
    collection,
    mo,
    normalize_scores,
    render_result_card,
    search,
    search_query,
    session,
    snapshot,
):
    mo.stop(not search_query.value, mo.md("Enter a text query above."))
    _indices, _similarities = search(snapshot, session.embedder, search_query.value)
    _scores = normalize_scores(_similarities)
    _cards = [
        mo.Html(
            render_result_card(
                snapshot.keys[i], float(_scores[i]), root=collection.root
            )
        )
        for i in _indices
    ]
    mo.vstack([mo.hstack(_cards[i : i + 3]) for i in range(0, len(_cards), 3)])
    return


@app.cell
def _(
    collection,
    load_image,
    mo,
    normalize_scores,
    render_result_card,
    search,
    session,
    snapshot,
    upload_image,
):
    mo.stop(not upload_image.value, mo.md("Upload an image to find similar images."))
    try:
        _image = load_image(upload_image.value[0].contents).image
    except (OSError, ValueError) as _exc:
        mo.stop(True, mo.md(f"Cannot decode uploaded image: {_exc}"))
    try:
        _indices, _similarities = search(snapshot, session.embedder, _image)
    finally:
        _image.close()
    _scores = normalize_scores(_similarities)
    _cards = [
        mo.Html(
            render_result_card(
                snapshot.keys[i], float(_scores[i]), root=collection.root
            )
        )
        for i in _indices
    ]
    mo.vstack([mo.hstack(_cards[i : i + 3]) for i in range(0, len(_cards), 3)])
    return


@app.cell
def _(evoc, mo, pl, values):
    mo.stop(
        len(values) < 3,
        mo.md("At least three indexed images are needed for clustering."),
    )
    clusterer = evoc.EVoC()
    clusterer.fit_predict(values)
    _layers = clusterer.cluster_layers_
    mo.stop(not _layers, mo.md("No cluster layers were found for this collection."))
    df = pl.DataFrame(
        {f"Layer_{len(_layers) - i}": layer for i, layer in enumerate(_layers)}
    )
    return (df,)


@app.cell
def _(ParallelCoordinates, df, mo):
    widget = mo.ui.anywidget(
        ParallelCoordinates(df, height=500, color_by=df.columns[0])
    )
    widget
    return (widget,)


@app.cell
def _(collection, keys, mo, render_result_card, widget):
    _paths = [keys[i] for i in widget.selected_indices if 0 <= i < len(keys)][:9]
    _cards = [
        mo.Html(render_result_card(path, root=collection.root)) for path in _paths
    ]
    mo.vstack([mo.hstack(_cards[i : i + 3]) for i in range(0, len(_cards), 3)])
    return


if __name__ == "__main__":
    app.run()
