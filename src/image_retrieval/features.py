from pathlib import Path

import numpy as np


class FeatureStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._features: dict[str, np.ndarray] = {}

    @classmethod
    def load(cls, path: Path) -> "FeatureStore":
        store = cls(path)
        if path.exists():
            store._features = np.load(path, allow_pickle=True)["arr_0"].item()
        return store

    def save(self) -> None:
        np.savez(self._path, self._features)

    def update(self, key: str, vector: np.ndarray) -> None:
        self._features[key] = vector

    def __contains__(self, key: str) -> bool:
        return key in self._features

    @property
    def path(self) -> Path:
        return self._path

    @property
    def keys(self) -> list[str]:
        return list(self._features.keys())

    @property
    def values(self) -> np.ndarray:
        return np.array([self._features[k] for k in self._features])
