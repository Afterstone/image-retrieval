import base64
import io
from html import escape
from pathlib import Path

from PIL import Image

from .images import load_image

THUMBNAIL_SIZE = 300
JPEG_QUALITY = 80


def resolve_original(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes collection root")
    return path


def render_result_card(
    path: str | Path, score: float | None = None, *, root: Path | None = None
) -> str:
    try:
        original = (
            resolve_original(root, str(path))
            if root is not None
            else Path(path).resolve()
        )
        img = load_image(original).image
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
        img.close()
        data = base64.b64encode(buf.getvalue()).decode()
        uri = escape(original.as_uri(), quote=True)
        size = THUMBNAIL_SIZE
        label = "" if score is None else f"Relative score: {score:.4f}"
        return (
            f'<a href="{uri}" target="_blank" style="display:inline-block;text-decoration:none;">'
            f'<div style="display:inline-block;width:{size}px;text-align:center;">'
            f'<img src="data:image/jpeg;base64,{data}" style="width:{size}px;height:{size}px;object-fit:cover;display:block;">'
            f'<div style="background:#222;color:white;padding:4px 8px;font-size:14px;font-weight:bold;">'
            f"{label}</div></div></a>"
        )
    except OSError, ValueError, Image.DecompressionBombError:
        return f"<em>Original unavailable: {escape(Path(path).name)}</em>"
