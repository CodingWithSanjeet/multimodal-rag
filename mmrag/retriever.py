"""Direct Pinecone similarity search (no LangChain)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import ROOT_DIR, load_settings, require_setting

IMAGE_DIR = ROOT_DIR / "assets" / "images"


@dataclass
class RetrievedDoc:
    text: str
    page: int | None
    modality: str
    score: float = 0.0
    image_path: str | None = None  # root-relative, resolved to assets/images/
    metadata: dict = field(default_factory=dict)


def to_int_page(value) -> int | None:
    """Pinecone returns numbers as floats (4.0); show them as 4."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def resolve_image_path(stored: str | None, image_dir: Path = IMAGE_DIR) -> str | None:
    """Map any stored image path (e.g. the notebook's
    'novacore_extracted_images/page_3_image_1.png') to assets/images/<filename>.
    Returns a root-relative path if the file exists, else None."""
    if not stored:
        return None
    name = Path(str(stored).replace("\\", "/")).name
    candidate = image_dir / name
    if not candidate.exists():
        return None
    try:
        return candidate.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return candidate.as_posix()


def match_to_doc(match) -> RetrievedDoc:
    md = dict(match.get("metadata") or {}) if isinstance(match, dict) else dict(match.metadata or {})
    score = match.get("score", 0.0) if isinstance(match, dict) else (match.score or 0.0)
    return RetrievedDoc(
        text=md.get("text", ""),
        page=to_int_page(md.get("page")),
        modality=str(md.get("modality", "text")),
        score=float(score or 0.0),
        image_path=resolve_image_path(md.get("image_path")),
        metadata=md,
    )


@lru_cache(maxsize=1)
def get_index():
    from pinecone import Pinecone

    settings = load_settings()
    pc = Pinecone(api_key=require_setting("PINECONE_API_KEY"))
    return pc.Index(settings.pinecone_index)


def retrieve(question: str, k: int | None = None, namespace: str | None = None) -> list[RetrievedDoc]:
    from .embeddings import embed_query

    settings = load_settings()
    result = get_index().query(
        vector=embed_query(question),
        top_k=k or settings.top_k,
        namespace=namespace or settings.pinecone_namespace,
        include_metadata=True,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches
    return [match_to_doc(m) for m in matches]
