"""Validated pickle-free snapshots and a single-writer publication lock."""

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

from .model import canonical_json, recipe_fingerprint, validate_vectors


class StoreError(ValueError):
    pass


@contextmanager
def writer_lock(path: Path):
    """OS-released advisory lock; persistent lock inode avoids unlink races."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StoreError(f"Another writer holds the store lock: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class FeatureStore:
    def __init__(self, path: Path, corpus_id: str, manifest: dict):
        self.path = path
        self.corpus_id = corpus_id
        self._manifest_json = canonical_json(manifest)
        self.dimension = manifest["dimension"]
        self.membership: dict[str, str] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.complete = False

    @property
    def manifest(self):
        return json.loads(self._manifest_json)

    @property
    def recipe_id(self):
        return recipe_fingerprint(self.manifest)

    def assert_compatible(self, manifest: dict):
        if canonical_json(manifest) != self._manifest_json:
            raise StoreError(
                "Embedding recipe mismatch; index the selected profile/device before querying"
            )

    @classmethod
    def load(
        cls, path: Path, corpus_id: str, manifest: dict, *, missing_ok: bool = False
    ):
        store = cls(path, corpus_id, manifest)
        if not path.exists():
            if missing_ok:
                return store
            raise FileNotFoundError(
                f"No embeddings at {path}; run image-retrieval for this profile/corpus"
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                expected = {"manifest", "paths", "path_hashes", "hashes", "vectors"}
                if (
                    len(archive.files) != len(expected)
                    or set(archive.files) != expected
                ):
                    raise StoreError(
                        "Unidentified/legacy or malformed NPZ; preserved without conversion"
                    )
                arrays = {name: archive[name] for name in expected}
            for name in ("paths", "path_hashes", "hashes"):
                if arrays[name].ndim != 1 or arrays[name].dtype.kind != "U":
                    raise StoreError(f"Invalid string array: {name}")
            if arrays["manifest"].shape != () or arrays["manifest"].dtype.kind != "U":
                raise StoreError("Invalid manifest array")
            metadata = json.loads(str(arrays["manifest"]))
            if set(metadata) != {
                "schema_version",
                "corpus_id",
                "recipe",
                "recipe_id",
                "complete",
            }:
                raise StoreError("Invalid snapshot metadata")
            if (
                type(metadata["schema_version"]) is not int
                or metadata["schema_version"] != 1
            ):
                raise StoreError("Unsupported snapshot schema")
            if (
                metadata["corpus_id"] != corpus_id
                or type(metadata["complete"]) is not bool
            ):
                raise StoreError("Invalid corpus identity or completion status")
            store.assert_compatible(metadata["recipe"])
            if metadata["recipe_id"] != store.recipe_id:
                raise StoreError("Invalid recipe fingerprint")
            paths, path_hashes, hashes = (
                arrays[name].tolist() for name in ("paths", "path_hashes", "hashes")
            )
            if (
                len(paths) != len(path_hashes)
                or len(set(paths)) != len(paths)
                or len(set(hashes)) != len(hashes)
            ):
                raise StoreError("Duplicate or misaligned membership/content arrays")
            matrix = arrays["vectors"]
            if matrix.dtype != np.float32:
                raise StoreError("Stored embeddings must be float32")
            validate_vectors(matrix, len(hashes), store.dimension)
            store.membership = dict(zip(paths, path_hashes, strict=True))
            store.vectors = dict(zip(hashes, matrix, strict=True))
            store.complete = metadata["complete"]
            store.validate()
            return store
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            raise StoreError(f"Cannot load snapshot {path}: {exc}") from exc

    def validate(self):
        if type(self.complete) is not bool:
            raise StoreError("Invalid completion status")
        for path, content_hash in self.membership.items():
            relative = PurePosixPath(path)
            if (
                not path
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != path
                or "\\" in path
            ):
                raise StoreError(
                    "Membership paths must be canonical corpus-relative paths"
                )
            if content_hash not in self.vectors:
                raise StoreError("Membership refers to missing content")
        for content_hash, vector in self.vectors.items():
            if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
                raise StoreError("Invalid SHA-256 content identity")
            if vector.dtype != np.float32:
                raise StoreError("Stored embeddings must be float32")
            validate_vectors(vector[None, :], 1, self.dimension)

    def add(self, content_hash: str, vector: np.ndarray):
        checked = validate_vectors(vector[None, :], 1, self.dimension)
        self.vectors[content_hash] = checked[0].copy()

    def save(self):
        """Caller must hold writer_lock for the entire load/reconcile/save run."""
        self.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        paths, hashes = self.keys, sorted(self.vectors)
        matrix = (
            np.stack([self.vectors[h] for h in hashes])
            if hashes
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        metadata = {
            "schema_version": 1,
            "corpus_id": self.corpus_id,
            "recipe": self.manifest,
            "recipe_id": self.recipe_id,
            "complete": self.complete,
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.path.parent, suffix=".npz", delete=False
            ) as handle:
                temporary = Path(handle.name)
                np.savez(
                    handle,
                    manifest=np.array(canonical_json(metadata)),
                    paths=np.array(paths, dtype=str),
                    path_hashes=np.array(
                        [self.membership[p] for p in paths], dtype=str
                    ),
                    hashes=np.array(hashes, dtype=str),
                    vectors=matrix,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @property
    def keys(self) -> list[str]:
        return sorted(self.membership)

    @property
    def values(self) -> np.ndarray:
        if not self.membership:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.vectors[self.membership[path]] for path in self.keys])
