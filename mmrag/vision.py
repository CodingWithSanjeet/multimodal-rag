"""Image helpers shared by ingestion and answering."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def image_to_data_uri(image_path: str | Path, max_side: int = 1600) -> str:
    """Base64 JPEG data URI for the Groq vision API (same as notebook cell 6)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
