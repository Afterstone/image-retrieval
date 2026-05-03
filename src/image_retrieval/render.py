import base64
import io
from pathlib import Path

from PIL import Image


def render_result_card(path: str | Path, score: float) -> str:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = base64.b64encode(buf.getvalue()).decode()
        uri = Path(path).as_uri()
        return (
            f'<a href="{uri}" target="_blank" style="display:inline-block;text-decoration:none;">'
            f'<div style="display:inline-block;width:300px;text-align:center;">'
            f'<img src="data:image/jpeg;base64,{data}" style="width:300px;height:300px;object-fit:cover;display:block;">'
            f'<div style="background:#222;color:white;padding:4px 8px;font-size:14px;font-weight:bold;">'
            f"{score:.4f}</div></div></a>"
        )
    except Exception:
        return f"<em>error loading {Path(path).name}</em>"
