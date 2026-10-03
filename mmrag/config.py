"""Central configuration. Every secret and model name comes from the environment
(or Streamlit secrets when running under Streamlit); nothing is hard-coded."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Keep the fastembed model inside the project so a build-time download
# (render.yaml / Dockerfile) is still there at runtime. fastembed's default is a tmp dir.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ROOT_DIR / ".cache" / "fastembed"))


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from env vars first, then Streamlit secrets (if available)."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st  # optional at import time

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def require_setting(name: str) -> str:
    value = get_setting(name)
    if not value:
        raise RuntimeError(
            f"Missing required setting {name}. Set it as an environment variable "
            "(or in .streamlit/secrets.toml). See .env.example."
        )
    return value


@dataclass(frozen=True)
class Settings:
    text_model: str
    vision_model: str
    embedding_model: str
    embedding_dim: int
    pinecone_index: str
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    top_k: int
    pdf_path: Path
    image_dir: Path
    summaries_path: Path


def load_settings() -> Settings:
    return Settings(
        text_model=get_setting("TEXT_MODEL", "openai/gpt-oss-20b"),
        vision_model=get_setting("VISION_MODEL", "qwen/qwen3.8-27b"),
        embedding_model=get_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_dim=int(get_setting("EMBEDDING_DIM", "384")),
        pinecone_index=get_setting("PINECONE_INDEX_NAME", "novacore-multimodal-rag"),
        pinecone_namespace=get_setting("PINECONE_NAMESPACE", "fy2026-demo"),
        pinecone_cloud=get_setting("PINECONE_CLOUD", "aws"),
        pinecone_region=get_setting("PINECONE_REGION", "us-east-1"),
        top_k=int(get_setting("TOP_K", "5")),
        pdf_path=ROOT_DIR / get_setting("PDF_PATH", "NovaCore_Multimodal_Company_Report_2026.pdf"),
        image_dir=ROOT_DIR / "assets" / "images",
        summaries_path=ROOT_DIR / "data" / "visual_summaries.json",
    )
