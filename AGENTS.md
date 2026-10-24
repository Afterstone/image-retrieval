# Project workflow

- The CLI owns corpus indexing and snapshot publication. The notebook only reads
  committed snapshots and encodes interactive queries.
- `model.py` defines the embedding seam: homogeneous batches yield ordered,
  finite, nonzero, float32 unit vectors `(N, D)`. Callers do not handle processors.
- Before adapter, profile, prompt or preprocessing changes, read
  [docs/adding-models.md](docs/adding-models.md) and complete its verification steps.
- Keep corpus identity, recipe fingerprint and original-byte content hash separate.
  Validate recipe compatibility even when dimensions match. Preserve legacy caches.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`,
  `uv run pytest`, and `uv run marimo check notebooks/main.py` separately.
  Automated verification is offline; real-model/hardware acceptance is manual.