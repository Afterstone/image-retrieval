# Local image retrieval

Index local image collections, then search by text or an uploaded image and
explore EVoC clusters in a Marimo notebook. Ranking is exact NumPy cosine
similarity. Model spaces are independent: equal dimensions do not make caches
compatible.

## Setup and indexing

Python >=3.14 is required. Install project and development dependencies with
`uv sync`. Copy `.env.example` to `.env` and set `IMAGE_FOLDER` to the collection
directory. Existing `.env` files are not rewritten by this migration.

Run `uv run image-retrieval` to index the default Gemma 2 profile. Select an
alternative with `uv run image-retrieval --profile siglip2 --corpus images`.
The only application flags are `--config`, `--profile`, and `--corpus`, plus help.
Selections override defaults for that invocation; runtime settings stay in YAML.

`profiles.yaml` contains the executable configuration's supported fields:
named pinned model recipes, named collections with stable IDs, cache location,
device (`auto`, `mps`, `cpu`), float32 precision, batch size (1–256), offline mode,
and checkpoint interval (default 500 unique embeddings). Unknown/duplicate keys,
unsafe YAML, mutable model revisions and invalid settings are errors. Relative
paths resolve against the repository root, including when launching elsewhere.
`${NAME}` placeholders are for machine-local paths; process environment overrides
the explicitly read repository `.env`. Importing the package reads neither.

The default profile pins `google/embeddinggemma-2`; the alternative pins
`google/siglip2-so400m-patch14-384`. Both use native dimensions and fixed
adapter-owned preprocessing. Gemma uses the unprefixed image/text similarity
recipe from Google's multimodal example, not its text-document search prompts.
See [preflight evidence](docs/model-preflight.md) for revisions and rationale.

### Hugging Face access and execution

Normal application inference may download pinned weights into the standard Hub
cache. Set `runtime.offline: true` for local files only; it fails if the selected
revision is not cached. Use `hf auth login` or export `HF_TOKEN` for authentication,
outside YAML. If Hugging Face requests license/access acceptance, review and
accept it manually on the model page first; the application cannot accept for you.
The inspected checkpoints are public, ungated Apache-2.0 repositories. Never
commit tokens. No downloaded Python execution (`trust_remote_code=False`) or
model substitution is enabled.

Float32 is the initial CPU/MPS baseline; Gemma fp16 activations are unsafe.
`auto` selects MPS when available, otherwise CPU. Explicit unavailable MPS and
accelerator failures are errors, without CPU retries or OOM batch shrinking.
Disable `PYTORCH_ENABLE_MPS_FALLBACK` when using MPS. Reduce the configured batch
size manually if needed. Weight loading is lazy: unchanged runs need no model
weights when recipe identity can be constructed from pinned settings.

## Notebook

Open with `uv run marimo edit notebooks/main.py`. Choose a model profile and
collection, and use **Reload embeddings** after CLI indexing. It reads committed
snapshots only, never indexes or writes corpus vectors. Switching releases the
previous model and invalidates queries, results and clustering. Missing or
incompatible snapshots show the matching indexing invocation. Incomplete snapshots
are visibly labeled; originals missing from disk produce unavailable-image cards.

Search supports text or a single uploaded image. Both queries and corpus images
use the same adapter and shared Pillow decoding (first frame, EXIF correction,
detached RGB, including WebP). Duplicate content retains separate gallery and
clustering rows. EVoC needs enough data for its clustering parameters; very small
or degenerate collections may not produce meaningful clusters.

Displayed scores are query-relative min-max values in [-1, 1], **not probabilities**
or cross-model quality measurements. Equal/singleton scores display zero.

## Snapshot and resume behavior

Generated caches live at
`.cache/image-retrieval/<corpus-id>/<recipe-id>/features.npz`; source images are
never modified. Each pickle-free NPZ contains schema/completion status, a readable
JSON recipe manifest and fingerprint, relative path/content membership, and unique
SHA-256 hashes aligned to float32 unit vectors. Readers validate before use.
Legacy unidentified NPZ files are left untouched and never automatically migrated.

The recipe conservatively includes resolved CPU/MPS backend and relevant library
versions, so dependency/backend changes can require a separate index. Labels,
collection roots, batch/checkpoint sizes and offline preferences do not affect it.
Moving a collection with the same corpus ID permits reuse after reconciliation.
There is no cross-corpus deduplication or cache garbage collection.

Each run deterministically discovers and hashes all candidates using the same
captured bytes it decodes, encodes only missing unique content, and reconciles
membership. A single OS lock covers load through final publication; saves use
same-directory temporary files and atomic replacement. Notebook readers see whole
committed snapshots without acquiring the writer lock.

During an incomplete scan, old membership remains alongside validated new paths;
unvisited/changed/deleted paths may therefore be stale and are explicitly covered
by the incomplete warning. On completion, membership becomes exactly the readable,
verified current collection: absent and unreadable paths are removed. Previously
computed vectors remain available for within-corpus reuse. Inference/invalid-output
failures stop after saving validated progress. Ctrl-C saves likewise; rerun to
resume. Abrupt process termination can lose work since the last committed
checkpoint but cannot publish a partial NPZ. Corrupt/incompatible stores fail
clearly, without silently discarding them.

Exit statuses: `0` completed successfully; `2` completed with skipped images (also
standard argument errors); `1` config/store/runtime failure; `130` interrupted.
Skipped paths and failure summaries go to the terminal.
Per-image decoding exceptions also append timestamps, corpus/root, failed paths
and tracebacks to `indexing-failures.log` alongside the selected `features.npz`.
The CLI prints the log location at startup. Bad images are skipped and indexing
continues; model, memory-exhaustion and snapshot failures still stop after saving
validated progress. Failure logs append across runs and remain Git-ignored in
the cache directory. Treat these logs as local data: they contain original paths.

## Development verification

Ruff, ty and pytest are project **dev dependencies**, installed by `uv sync`.
Run each gate separately:

- `uv run ruff check .` — lint and import ordering.
- `uv run ruff format --check .` — formatting.
- `uv run ty check` — types, including source, notebook and tests.
- `uv run pytest` — offline contracts and integration behavior.
- `uv run marimo check notebooks/main.py` — reactive notebook structure.

Tests block socket connections, Hub download entry points and pretrained model
loading; they use generated images and lightweight doubles. Dependency resolution
may access package metadata, but tests must never fetch weights or certify hardware.

**Outstanding manual acceptance:** actual pinned model loading, real CPU/MPS
inference and memory, retrieval quality, notebook profile switching, clustering,
and explicit reload. Offline checks are not Mac certification. No Gemma-over-SigLIP
quality gate is required.
