"""Canonical image decoding shared by corpus and interactive queries."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

DECODING_RECIPE_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


@dataclass(frozen=True)
class LoadedImage:
    content_hash: str
    image: Image.Image


def load_image(source: Path | bytes) -> LoadedImage:
    captured = source.read_bytes() if isinstance(source, Path) else source
    with Image.open(BytesIO(captured)) as original:
        original.seek(0)
        image = ImageOps.exif_transpose(original).convert("RGB").copy()
    return LoadedImage(sha256(captured).hexdigest(), image)
