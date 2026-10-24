# Migration plan: generalized Hugging Face embeddings

Recorded: 2026-10-24.

This document is the implementation handoff from the design interview. No migration code has been implemented. Preserve the agreed requirements below; distinguish them from implementation recommendations and preflight facts that still require verification.

## 1. Goal and agreed scope

Replace OpenCLIP inference with a consistent application-owned embedding interface supporting two model families:

- **EmbeddingGemma 2:** default profile, intended model repository `google/embeddinggemma-2`.
- **SigLIP2:** selectable alternative using a Transformers-native checkpoint.

Preserve the current product:

- Local image collections.
- Text-to-image and image-to-image retrieval.
- Existing image clustering and visualization.
- Exact NumPy similarity ranking; no search-scaling redesign.

Text and image embeddings must be comparable **within the selected model's embedding space**. Alignment across different models is neither needed nor supported. Equal dimensions do not establish compatibility.

### Workflow ownership

- **CLI:** generate and persist corpus embeddings.
- **Notebook:** visualize, cluster, select models/collections, and encode interactive queries. Never index the corpus or write corpus vectors.
- One active model in the notebook at a time.
- Explicit embedding reload; no filesystem watcher.
- Automated verification is **offline only**. The user performs manual real-model and hardware verification as needed.

## 2. Existing implementation and migration risks

Reinspect these facts before editing in case the workspace has changed:

- `src/image_retrieval/model.py`: `load_siglip2` loads through OpenCLIP and returns a model/preprocessor/tokenizer tuple.
- `src/image_retrieval/search.py`: text/image encoding exposes OpenCLIP methods. Ranking uses NumPy dot products; `normalize_scores` provides query-relative min-max display scores.
- `src/image_retrieval/features.py`: `FeatureStore` stores a pickled path-to-vector dictionary in NPZ without recipe metadata.
- `src/image_retrieval/indexing.py`: skips known paths, without detecting changed contents or reconciling deletions.
- `src/image_retrieval/config.py`: import-time dotenv/environment globals; importing application modules can require `IMAGE_FOLDER`.
- `src/image_retrieval/device.py`: selects MPS or CPU.
- `notebooks/main.py`: independently loads model and vector store without compatibility validation.
- `pyproject.toml`: Python >=3.14; Transformers already declared, Sentence Transformers absent at inspection.
- `.gitignore` already excludes `.cache`.

The current default downloads Hugging Face-hosted weights through OpenCLIP. This migration changes the inference interface, not merely the download location. New SigLIP weights/preprocessing must not be assumed compatible with old vectors.

## 3. Architecture and embedding contract

Use **Strategy + Adapter** with minimal machinery:

- A small Python `Protocol` defines the embedding interface.
- Family adapters implement the protocol.
- A simple factory/registry constructs the selected adapter from validated settings.
- Dependency injection supplies it to indexing and query handling.
- Avoid a plugin framework, elaborate inheritance, or arbitrary framework keyword passthrough.

The embedding module hides loading, tokenization, image processing, prompting, pooling/projection, precision, device placement, normalization, and identity. Callers must not handle tokenizers or processors.

### Common operation

- Batch-first: homogeneous batches of text strings or decoded Pillow images.
- One vector per input, preserving order.
- Consistent two-dimensional NumPy output of shape `(N, D)`.
- Output is `float32`, finite, nonzero, and unit-normalized.
- Native model dimensions only.
- Invalid output shapes, dimensions, zero vectors, or nonfinite values fail clearly before persistence.
- Support query/corpus intent where required; adapters own fixed documented task recipes and prompts.
- No mixed-modality batches or composite image-plus-caption inputs initially.

Recommended edge behavior: an empty batch returns `(0, D)`. Test the contract through fake implementations and injected lightweight model/processor doubles.

## 4. YAML configuration and CLI

Create tracked `profiles.yaml` with:

- Schema version.
- Named model profiles and default profile (Gemma 2).
- Named collections/corpora and default collection.
- Stable corpus IDs separate from root paths.
- Runtime settings: device, supported precision, batch size, offline mode, checkpoint interval.
- Repo-relative cache root, default `.cache/image-retrieval/`.

YAML owns model recipes. `.env` owns credentials and machine-local paths, referenced through explicit path placeholders such as `${IMAGE_FOLDER}`.

- Resolve relative configured paths against the repository root, not launch working directory.
- Missing placeholders produce clear errors.
- Use safe YAML parsing and strict validation. Reject unknown options, duplicate keys, invalid types/ranges, and unsafe tags.
- Do not expose arbitrary loader/preprocessor kwargs.
- Do not store credentials in YAML or log tokens.
- Replace import-time application globals with explicit configuration loading; library imports should work without `.env`.

Preserve the bare **`image-retrieval`** indexing command. Add only these selection flags, plus standard help:

- `--config`
- `--profile`
- `--corpus`

Selection flags override defaults for that invocation only. Runtime settings remain in YAML. No subcommands, runtime/rebuild flags, search command, wizard, or configuration-writing feature in this migration.

## 5. Image loading and inference policy

### Shared Pillow loader

Use a small common loader for corpus files and uploaded queries:

1. Capture the original file bytes.
2. Decode with Pillow.
3. Use the first frame for animation.
4. Correct EXIF orientation.
5. Convert to a detached RGB Pillow image.

Hash the same captured bytes that are decoded. Keep model-specific resizing and tensor normalization inside the adapter. Support existing formats plus WebP consistently. Preserve Pillow safety defaults; no new image-processing framework.

### Local execution

- Support `auto`, `mps`, and `cpu` through YAML.
- `auto` selects an eligible MPS configuration when available, otherwise CPU.
- Explicit unsupported device selections fail clearly.
- No silent CPU retry after accelerator inference failures.
- Start both adapters with `float32` as the Mac correctness baseline; remove blanket autocast behavior.
- EmbeddingGemma 2 activations do not support `float16`; verify current official guidance during preflight.
- Configurable bounded batches, initially defaulting to one image.
- No automatic OOM retry/batch shrinking.
- Use evaluation/inference mode.

### Hugging Face access

- Standard Hugging Face model cache; initial weight downloads are allowed in normal application use.
- Explicit offline mode uses local model files only.
- Authentication through Hugging Face login or `HF_TOKEN`, outside YAML.
- Explain required access/license steps; acceptance is manual if required.
- Pin immutable model and processor revisions.
- Keep `trust_remote_code=False`. Any model requiring downloaded executable Python needs a separate reviewed decision.
- No silent substitution of another model if the selected model cannot load.

## 6. Identity, caching, and persistence

Keep three identities separate:

| Identity | Purpose |
| --- | --- |
| Corpus ID | Stable configured collection identity, independent of current root location. |
| Recipe ID | Adapter-generated fingerprint of model, preprocessing, and embedding-relevant settings. |
| Content hash | SHA-256 of original image bytes, independent of path. |

Recommended store layout:

`.cache/image-retrieval/<corpus-id>/<recipe-id>/features.npz`

Caches are machine-local, generated, ignored by Git, and located relative to the repo—not in source image directories.

### Recipe identity

Expose a deterministic adapter-owned ID derived from canonical resolved recipe metadata, with a readable manifest containing:

- Exact model and processor revision/configuration.
- Adapter recipe version.
- Shared image-decoding recipe version.
- Model-specific preprocessing and projection/pooling.
- Fixed task prompts/roles.
- Output dimension and normalization.
- Numerical execution settings that affect compatibility.

Profile labels, corpus roots, batch sizes, checkpoint frequency, progress settings, and offline/download preferences must not invalidate embeddings.

**Conservative recommendation, not an independently confirmed requirement:** initially include resolved device backend and relevant inference-library versions in identity until interoperability is verified. This can create separate CPU/MPS stores. If implementing a less conservative policy, document and test its rationale rather than silently assuming numerical interchangeability.

Hash canonical serialized recipe metadata, not a manually assigned label. Validate compatibility on opening, appending, and querying a store—even for equal-dimensional vectors. Query-only recipe changes may conservatively create a new ID initially.

### Store contents and publication

- Single pickle-free NPZ snapshot with numeric/string arrays and embedded JSON manifest.
- Schema version, recipe manifest/fingerprint, and complete/incomplete status.
- Corpus-relative path-to-content-hash membership.
- Unique content hashes and aligned `float32` vector matrix.
- Validate shape, finite/unit vectors, membership alignment, and identity on loading.
- Load with `allow_pickle=False`; no object arrays.
- Atomic replacement using a temporary file in the same directory.
- One CLI writer per store, enforced with a lock.
- Notebook readers observe committed snapshots, not partial writes.
- Preserve unidentified legacy caches; do not automatically load, convert, or delete them.
- Missing store starts empty for CLI; notebook shows indexing guidance.
- Corrupt/incompatible stores fail clearly rather than being silently discarded.

Use content reuse **within a corpus** only. Identical files share computation but retain all paths as separate results and clustering rows. Cross-corpus reuse and general cache garbage collection are deferred.

## 7. Indexing and failure behavior

Each CLI run:

1. Discover candidate images deterministically.
2. Hash all candidates initially; do not rely solely on size/mtime.
3. Reuse vectors for known content under the matching recipe.
4. Encode only missing unique content in bounded batches.
5. Reconcile new, modified, renamed, and deleted paths.
6. Remove absent membership only after a completed scan.

Never delete original images. Moving a collection while keeping its configured identity permits reuse after content reconciliation.

Failure policy:

- Unreadable/corrupt images: skip, report paths, and summarize failures.
- Inference failures or invalid vectors: preserve validated progress and stop.
- Checkpoint every 500 successfully generated unique embeddings by default; configurable in YAML.
- Save on completion and interruption.
- Resume verified committed work.
- Interrupted snapshots remain coherent and are visibly marked incomplete; no deletion reconciliation after incomplete traversal.

Implementation details to make explicit in tests/docs:

- How old membership is retained during an incomplete scan.
- Remove or mark unavailable an existing unreadable item after a completed scan rather than silently presenting an unverified old embedding as current.
- Recommended exit statuses: `0` success, `2` completed with skipped images, `1` runtime/config/store failure, `130` interrupted. Standard argument-parser errors can also use `2`.
- Avoid loading model weights for no-op runs if identity can be resolved safely without doing so; correctness takes priority over this optimization.

## 8. Notebook behavior and scores

- Shared YAML model-profile and collection selectors.
- One active model; release prior model references when switching.
- Invalidate previous queries/results/clustering when the selected space changes.
- Explicit **Reload embeddings** loads a coherent committed snapshot and updates dependent views.
- Missing/incompatible cache: explain the corresponding CLI invocation; never start indexing.
- Show incomplete-snapshot status.
- Encode queries through the same adapter contract and validate compatibility before comparison.
- Resolve collection-relative paths consistently for galleries and result cards.
- Handle missing originals without crashing the whole view.

Preserve exact ranking and existing EVoC/ParallelCoordinates visualization. Keep current relative min-max scores as the default display; raw cosine remains internal. Label relative scores clearly: they are not probabilities and do not measure quality across model families. Test empty/equal/singleton score cases and top-k behavior.

## 9. Implementation sequence and files

### Phase A — Preflight and configuration

- Verify official recipes/revisions and dependency support.
- Update `pyproject.toml`/`uv.lock`: remove `open-clip-torch`; add `sentence-transformers`, PyYAML, direct NumPy, and development pytest. Preserve Python >=3.14; resolve compatible Transformers requirements.
- Implement explicit settings in `src/image_retrieval/config.py` and add `profiles.yaml`.
- Simplify `src/image_retrieval/__init__.py` imports.

### Phase B — Embedding module

Depends on Phase A.

- Replace `load_siglip2` tuple interface in `src/image_retrieval/model.py` with the protocol, factory, metadata, and adapters. A small `embeddings/` package is acceptable if useful.
- Update `src/image_retrieval/device.py`.
- Add `src/image_retrieval/images.py`.
- Remove OpenCLIP-specific encoding responsibilities from `src/image_retrieval/search.py`; preserve reusable ranking/scoring functions.

### Phase C — Persistence

Depends on settled configuration/recipe types; can proceed alongside adapter implementations.

- Evolve `FeatureStore` in `src/image_retrieval/features.py` with content membership, metadata, validation, locking, and atomic publication.

### Phase D — CLI integration

Depends on Phases B and C.

- Update `discover_images`/`encode_and_index` in `src/image_retrieval/indexing.py`.
- Wire selectors and indexing flow in `src/image_retrieval/main.py` and the existing entry point.

### Phase E — Notebook and documentation

Depends on the integrated adapter/store flow.

- Update `notebooks/main.py` for selection, single-model lifecycle, compatible snapshots, and explicit reload.
- Update `src/image_retrieval/render.py` for relative paths and missing files.
- Populate `README.md` and update `.env.example` with safe placeholders. Do not expose existing `.env` secrets.
- Add root `AGENTS.md` and `docs/adding-models.md`.
- Reuse `.gitignore`'s existing `.cache` exclusion; adjust only if needed.

## 10. Agent guidance deliverable

Root `AGENTS.md` should explain the CLI/notebook ownership split and embedding seam, with an explicit pointer to the model-addition guide for adapter, profile, or preprocessing changes.

The guide must require an implementing agent to:

1. Verify supported modalities/retrieval behavior and loading from authoritative sources.
2. Add/reuse the appropriate adapter.
3. Generate a valid pinned YAML profile using the executable schema as the source of truth.
4. Account for preprocessing, prompts, and recipe identity/version changes.
5. Run offline contract/config/cache tests.
6. Update relevant profiles and documentation alongside implementation changes.

Keep instructions maintained and concise. Project-specific skills may be considered later, but are not part of the initial deliverable.

## 11. Verification and acceptance

Add offline tests under `tests/`, covering:

- Strict configuration and path resolution; imports without `.env`.
- Adapter routing, prompt/role handling, batch ordering and output invariants through lightweight doubles.
- Recipe identity stability and invalidation.
- Pillow image canonicalization using generated fixtures.
- Empty/populated store round-trips, malformed data, pickle prohibition, equal-dimensional recipe mismatch, locks, and failed atomic saves.
- New/unchanged/changed/renamed/deleted/duplicate files, bounded batches, checkpoint counts, skips, interruption, and resume.
- CLI selectors and error summaries/statuses.
- Checked query/store compatibility and relative scoring edge cases.

Run `uv run pytest` with tests preventing network/Hub access and pretrained weight loading. Package dependency resolution may access package metadata; the automated test suite must remain offline. Check editor diagnostics and any existing project gates discovered during implementation; do not claim nonexistent gates passed.

Acceptance requires preserving the agreed product workflows and passing offline tests. There is no Gemma-over-SigLIP quality gate.

The user will manually verify real model loading, MPS/CPU behavior, retrieval quality, switching, clustering, and reload. Record these as outstanding until performed. Do not claim Mac certification from offline tests.

## 12. Preflight references and unresolved facts

Official sources consulted during planning:

- https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
- https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- https://huggingface.co/docs/transformers/model_doc/auto

Recheck current primary documentation before implementation. The Google documentation may redirect to its newer site.

Still to establish from documentation/configuration, not guess:

- Exact immutable Gemma2 and SigLIP2 repository revisions.
- Closest suitable Transformers-native SigLIP2 counterpart to the current OpenCLIP checkpoint.
- Dependency versions supporting those recipes on Python >=3.14.
- Correct text-to-image query recipe, pooling/projection, and optional unused-audio-encoder omission for the pinned Gemma configuration.

If these cannot be established, report the blocker rather than substituting another model or enabling remote code silently.

## 13. Explicit exclusions

No new modalities, composite queries, arbitrary Hub-model compatibility, cross-model alignment, hosted inference, CUDA certification, ANN/vector database, cross-corpus deduplication, wizard/config editor, runtime/rebuild CLI flags, prompt tuning, dimension truncation, vector quantization, filesystem watching, automatic legacy-cache deletion, or silent model/device fallback.