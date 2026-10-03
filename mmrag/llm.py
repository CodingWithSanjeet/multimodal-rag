"""Groq client (shared by ingestion and answering)."""

from __future__ import annotations

from functools import lru_cache

from .config import require_setting


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq

    return Groq(api_key=require_setting("GROQ_API_KEY"))
