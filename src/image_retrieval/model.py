"""Application-owned embedding seam and two lazy, fixed-recipe adapters."""

import json
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from typing import Any, Literal, Protocol, Sequence

import numpy as np
from PIL import Image

from .config import Profile, Runtime
from .device import get_device
from .images import DECODING_RECIPE_VERSION

Intent = Literal["query", "corpus"]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recipe_fingerprint(manifest: dict) -> str:
    return sha256(canonical_json(manifest).encode()).hexdigest()


def validate_vectors(
    vectors: Any, rows: int, dimension: int, *, normalize: bool = False
) -> np.ndarray:
    array = np.asarray(vectors)
    if array.shape != (rows, dimension) or array.dtype.kind != "f":
        raise ValueError(
            f"Expected floating embeddings of shape {(rows, dimension)}, got {array.shape}/{array.dtype}"
        )
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain nonfinite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings contain zero vectors")
    if normalize:
        array = (array / norms[:, None]).astype(np.float32)
    elif not np.allclose(norms, 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Embeddings must be unit-normalized")
    return array


class Embedder(Protocol):
    dimension: int

    @property
    def manifest(self) -> dict: ...

    @property
    def recipe_id(self) -> str: ...

    def encode_texts(
        self, texts: Sequence[str], *, intent: Intent = "query"
    ) -> np.ndarray: ...

    def encode_images(
        self, images: Sequence[Image.Image], *, intent: Intent = "corpus"
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class _Adapter:
    """Internal common validation and identity, not an extensible plugin framework."""

    dimension: int

    def __init__(self, profile: Profile, runtime: Runtime, device: str):
        if runtime.precision != "float32" or device not in {"cpu", "mps"}:
            raise ValueError("Adapters require float32 on CPU or MPS")
        self.profile, self.runtime, self.device = profile, runtime, device
        self._model = None
        self._processor = None
        recipe = {
            "profile": asdict(profile),
            "adapter_recipe_version": 1,
            "image_decoding_recipe_version": DECODING_RECIPE_VERSION,
            "dimension": self.dimension,
            "normalization": "l2-float32",
            "execution": {
                "device": device,
                "precision": "float32",
                "autocast": False,
                "attention": "eager",
            },
            "libraries": {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "numpy",
                    "pillow",
                    "torchvision",
                )
            },
        }
        recipe.update(self._recipe())
        self._manifest_json = canonical_json(recipe)

    @property
    def manifest(self) -> dict:
        return json.loads(self._manifest_json)

    @property
    def recipe_id(self) -> str:
        return recipe_fingerprint(self.manifest)

    def _recipe(self) -> dict:
        raise NotImplementedError

    def _check(self, inputs, expected_type, intent):
        if intent not in {"query", "corpus"}:
            raise ValueError("intent must be query or corpus")
        if isinstance(inputs, (str, bytes)) or any(
            not isinstance(item, expected_type) for item in inputs
        ):
            raise TypeError(
                "Expected a homogeneous batch of strings or decoded Pillow images"
            )

    def close(self) -> None:
        self._model = None
        self._processor = None
        import gc

        gc.collect()
        if self.device == "mps":
            import torch

            torch.mps.empty_cache()


class EmbeddingGemma2(_Adapter):
    dimension = 768

    def _recipe(self) -> dict:
        return {
            "processor": "pinned SentenceTransformer processor; vision_soft_tokens_per_image=280",
            "pooling_projection": "pinned Transformer native 512-to-768 projection; mean pooling including prompt; Normalize",
            # Follow Google's concrete multimodal example, not its text-document recipe.
            "prompts": {
                "text_query": "",
                "text_corpus": "",
                "image_query": "",
                "image_corpus": "",
            },
            "task_recipe": "unprefixed image-text similarity (Google multimodal guide)",
            "config_overrides": {"audio_config": None},
            "context_length": 8192,
        }

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.profile.model,
                revision=self.profile.revision,
                device=self.device,
                trust_remote_code=False,
                local_files_only=self.runtime.offline,
                model_kwargs={
                    "torch_dtype": torch.float32,
                    "attn_implementation": "eager",
                },
                config_kwargs={"audio_config": None},
                processor_kwargs={"revision": self.profile.processor_revision},
            )
            self._model.float().eval()
            if self._model.get_embedding_dimension() != self.dimension:
                raise ValueError("Unexpected native EmbeddingGemma dimension")
            self._model.max_seq_length = 8192
        return self._model

    def _encode(self, inputs, prompt: str):
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        import torch

        with torch.inference_mode():
            vectors = self._load().encode(
                list(inputs),
                prompt=prompt,
                batch_size=self.runtime.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return validate_vectors(vectors, len(inputs), self.dimension, normalize=True)

    def encode_texts(self, texts, *, intent: Intent = "query"):
        self._check(texts, str, intent)
        prompt = self.manifest["prompts"][f"text_{intent}"]
        # Explicit modality prevents media-looking user strings becoming URL/path inputs.
        return self._encode([{"text": text} for text in texts], prompt)

    def encode_images(self, images, *, intent: Intent = "corpus"):
        self._check(images, Image.Image, intent)
        return self._encode([{"image": image} for image in images], "")


class SigLIP2(_Adapter):
    dimension = 1152

    def _recipe(self) -> dict:
        return {
            "processor": "pinned AutoProcessor; use_fast=False; 384px RGB",
            "pooling_projection": "native text/vision pooler_output (includes text head)",
            "prompts": {
                "text_query": "",
                "text_corpus": "",
                "image_query": "",
                "image_corpus": "",
            },
            "text": {"padding": "max_length", "max_length": 64, "truncation": True},
        }

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.profile.model,
                revision=self.profile.processor_revision,
                trust_remote_code=False,
                local_files_only=self.runtime.offline,
                use_fast=False,
            )
            self._model = (
                AutoModel.from_pretrained(
                    self.profile.model,
                    revision=self.profile.revision,
                    trust_remote_code=False,
                    local_files_only=self.runtime.offline,
                    dtype=torch.float32,
                    attn_implementation="eager",
                )
                .to(self.device)
                .eval()
            )
        return self._model, self._processor

    def _encode(self, inputs, modality):
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        import torch

        model, processor = self._load()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.runtime.batch_size):
                batch = list(inputs[start : start + self.runtime.batch_size])
                kwargs = {modality: batch, "return_tensors": "pt"}
                if modality == "text":
                    kwargs.update(padding="max_length", max_length=64, truncation=True)
                tensors = processor(**kwargs).to(self.device)
                # Transformers 5.20 returns ModelOutput, not a bare tensor.
                output = (
                    model.get_text_features(**tensors, return_dict=True)
                    if modality == "text"
                    else model.get_image_features(**tensors, return_dict=True)
                )
                vector = output.pooler_output
                batches.append(vector.detach().float().cpu().numpy())
        return validate_vectors(
            np.concatenate(batches), len(inputs), self.dimension, normalize=True
        )

    def encode_texts(self, texts, *, intent: Intent = "query"):
        self._check(texts, str, intent)
        return self._encode(texts, "text")

    def encode_images(self, images, *, intent: Intent = "corpus"):
        self._check(images, Image.Image, intent)
        return self._encode(images, "images")


def create_embedder(profile: Profile, runtime: Runtime) -> Embedder:
    adapters = {"embeddinggemma2": EmbeddingGemma2, "siglip2": SigLIP2}
    if profile.family not in adapters:
        raise ValueError(f"Unsupported embedding family: {profile.family}")
    return adapters[profile.family](profile, runtime, get_device(runtime.device))
