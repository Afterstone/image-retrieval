import numpy as np

from .model import Embedder, validate_vectors


def rank_by_similarity(
    feature_matrix: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if feature_matrix.ndim != 2 or query_vector.shape != (feature_matrix.shape[1],):
        raise ValueError("Ranking requires an (N, D) matrix and a (D,) query")
    if not np.isfinite(feature_matrix).all() or not np.isfinite(query_vector).all():
        raise ValueError("Ranking inputs must be finite")
    similarities = feature_matrix @ query_vector
    top_indices = np.argsort(-similarities, kind="stable")[:top_k]
    return top_indices, similarities


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Query-relative [-1, 1] display scores, not probabilities."""
    if scores.size == 0:
        return np.array([], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.zeros_like(scores, dtype=float)
    return 2.0 * (scores - s_min) / (s_max - s_min) - 1.0


def search(store, embedder: Embedder, query, *, top_k: int = 9):
    """Validate the selected space before encoding and exact comparison."""
    from PIL import Image

    store.assert_compatible(embedder.manifest)
    if isinstance(query, str):
        vectors = embedder.encode_texts([query], intent="query")
    elif isinstance(query, Image.Image):
        vectors = embedder.encode_images([query], intent="query")
    else:
        raise TypeError("Query must be text or a decoded Pillow image")
    vectors = validate_vectors(vectors, 1, embedder.dimension)
    return rank_by_similarity(store.values, vectors[0], top_k)
