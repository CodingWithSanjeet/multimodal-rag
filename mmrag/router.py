"""Query routing: if any retrieved doc is a visual whose image exists locally, use the
vision LLM with that image; otherwise use the text LLM."""

from __future__ import annotations

from dataclasses import dataclass

from .retriever import RetrievedDoc


@dataclass
class Route:
    use_vision: bool
    image_paths: list[str]  # unique, in retrieval order


def unique_image_paths(docs: list[RetrievedDoc]) -> list[str]:
    paths: list[str] = []
    for doc in docs:
        if doc.modality == "visual" and doc.image_path and doc.image_path not in paths:
            paths.append(doc.image_path)
    return paths


def route(docs: list[RetrievedDoc]) -> Route:
    paths = unique_image_paths(docs)
    return Route(use_vision=bool(paths), image_paths=paths)
