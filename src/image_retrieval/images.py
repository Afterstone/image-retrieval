"""Canonical image decoding shared by corpus and interactive queries."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

DECODING_RECIPE_VERSION = 2
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"})


@dataclass(frozen=True)
class LoadedImage:
    content_hash: str
    image: Image.Image
    metadata_warning: str | None = None


def load_image(source: Path | bytes) -> LoadedImage:
    captured = source.read_bytes() if isinstance(source, Path) else source
    metadata_warning = None
    with Image.open(BytesIO(captured)) as original:
        original.seek(0)
        # Pixel decoding failures must not be mistaken for optional metadata failures.
        original.load()
        try:
            original.getexif()
        except MemoryError:
            raise
        except Exception as exc:
            metadata_warning = (
                f"Unreadable EXIF; using unrotated pixels: {type(exc).__name__}: {exc}"
            )
            # Fresh pixel-only image: processors must not reparse the broken EXIF.
            with original.convert("RGB") as rgb:
                image = Image.new("RGB", rgb.size)
                image.paste(rgb)
        else:
            image = ImageOps.exif_transpose(original).convert("RGB").copy()
    return LoadedImage(sha256(captured).hexdigest(), image, metadata_warning)
