from .device import get_device
from .features import FeatureStore
from .indexing import discover_images, encode_and_index
from .model import load_siglip2
from .render import render_result_card
from .search import encode_image, encode_text, normalize_scores, rank_by_similarity

__all__ = [
    "get_device",
    "load_siglip2",
    "FeatureStore",
    "encode_text",
    "encode_image",
    "rank_by_similarity",
    "normalize_scores",
    "render_result_card",
    "discover_images",
    "encode_and_index",
]


def main() -> None:
    from .main import main as _index

    _index()
