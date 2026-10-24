# Adding or changing an embedding recipe

1. **Verify primary sources.** Establish image/text retrieval support, native
   dimensions, correct loading class, pooling/projection, processor behavior,
   numerical precision and task prompts. Record immutable model and processor
   revisions with citations. If an essential fact is unresolved, report the
   blocker rather than substituting a model or enabling remote Python.
2. **Use the embedding seam.** Reuse an appropriate adapter or implement the small
   `Embedder` protocol in `src/image_retrieval/model.py`. Keep all framework loading,
   device placement, tokenization, preprocessing, prompts and normalization inside.
   Preserve homogeneous batches, order, `(N, D)` float32 finite unit vectors and
   `(0, D)` empty output. Keep weights lazy and inference in evaluation mode.
3. **Validate configuration.** Treat `config.py`'s executable strict validation as
   the schema's source of truth. Extend its supported family/checkpoint pairs and
   factory routing explicitly, then add a valid pinned `profiles.yaml` entry.
   Keep credentials outside YAML and retain Python >=3.14 support. Avoid arbitrary
   loader kwargs or implicit Hub-model support.
4. **Version identity.** Account for processor revision/configuration, shared image
   decoding, pooling/projection, fixed prompts/roles, native dimension and execution
   settings. Increment the adapter/shared decoding recipe version when semantics
   change. Hash canonical resolved metadata. Preserve identity independence from
   profile labels, corpus locations, batching/checkpointing and download policy.
   Backend/library-version identity is deliberately conservative until verified.
5. **Prove offline behavior.** Add lightweight model/processor doubles for routing,
   exact prompts and input modalities, bounded batches, dtype, order and invalid
   outputs. Cover configuration acceptance/rejection, recipe stability/invalidation,
   equal-dimensional mismatch, pickle-free caches and checked queries. Run Ruff,
   formatting, ty, pytest and Marimo checks listed in the README. Never download
   or instantiate pretrained weights in the automated suite.
6. **Update the handoff.** Change relevant profiles and documentation in the same
   patch; cite new recipe evidence and describe cache consequences. Record real
   model loading, CPU/MPS execution, quality, switching/clustering/reload as manual
   acceptance until performed. Equal dimensions and passing doubles do not establish
   compatibility or hardware certification.