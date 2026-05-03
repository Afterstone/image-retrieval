import numpy as np
import torch
from PIL import Image

from .config import TOP_K


def encode_text(model, tokenizer, device: torch.device, query: str) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    with torch.no_grad(), torch.autocast(device.type):
        features = model.encode_text(tokens, normalize=True)
    return features.squeeze().cpu().numpy()


def encode_image(
    model, preprocess, device: torch.device, image: Image.Image
) -> np.ndarray:
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device.type):
        features = model.encode_image(tensor, normalize=True)
    return features.squeeze().cpu().numpy()


def rank_by_similarity(
    feature_matrix: np.ndarray,
    query_vector: np.ndarray,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = feature_matrix @ query_vector
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.zeros_like(scores, dtype=float)
    return 2.0 * (scores - s_min) / (s_max - s_min) - 1.0
