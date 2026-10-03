"""Answer generation (port of notebook cell 16)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import ROOT_DIR, load_settings
from .retriever import RetrievedDoc, retrieve
from .router import route

TEXT_PROMPT = """You are a helpful assistant answering questions about the NovaCore Systems FY2026 report.

Use ONLY the retrieved context below.
If the answer is not available in the context, say: "I could not find that information in the report."
Mention page numbers when possible.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

VISION_PROMPT = """You are answering questions about the NovaCore Systems FY2026 report.
Use ONLY the retrieved context and attached visuals.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Answer factually using evidence from the text and attached images.
- Mention page numbers.
- If missing, state that it is not found.
"""

MAX_VISION_IMAGES = 1  # stay well within Groq per-request limits (same as the notebook)

_GLYPH_CITATION = re.compile(r"\s*【[^】]*】")
_FLOAT_PAGE = re.compile(r"(\bpages?\s*:?\s*)(\d+)\.0\b", re.IGNORECASE)


def clean_answer(text: str) -> str:
    """Strip gpt-oss style 【...】 citation glyphs and turn 'Page 4.0' into 'Page 4'."""
    text = _GLYPH_CITATION.sub("", text)
    text = text.replace("【", "").replace("】", "")
    text = _FLOAT_PAGE.sub(r"\g<1>\g<2>", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def format_context(docs: list[RetrievedDoc]) -> str:
    parts = []
    for doc in docs:
        page = doc.page if doc.page is not None else "?"
        parts.append(f"[Page: {page} | {doc.modality.upper()}]\n{doc.text}")
    return "\n\n".join(parts)


@dataclass
class RagAnswer:
    question: str
    answer: str
    route: str  # "text" | "vision"
    model: str
    sources: list[dict] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)


def _text_answer(question: str, context: str, model: str) -> str:
    from .llm import get_groq_client

    response = get_groq_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": TEXT_PROMPT.format(context=context, question=question)}],
        temperature=0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content or ""


def _vision_answer(question: str, context: str, image_paths: list[str], model: str) -> str:
    from .llm import get_groq_client
    from .vision import image_to_data_uri

    content: list[dict] = [{"type": "text", "text": VISION_PROMPT.format(context=context, question=question)}]
    for path in image_paths[:MAX_VISION_IMAGES]:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(ROOT_DIR / path)}})
    response = get_groq_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_completion_tokens=900,
    )
    return response.choices[0].message.content or ""


def ask(question: str, k: int | None = None) -> RagAnswer:
    settings = load_settings()
    docs = retrieve(question, k=k)
    context = format_context(docs)
    decision = route(docs)
    if decision.use_vision:
        model = settings.vision_model
        raw = _vision_answer(question, context, decision.image_paths, model)
    else:
        model = settings.text_model
        raw = _text_answer(question, context, model)
    return RagAnswer(
        question=question,
        answer=clean_answer(raw),
        route="vision" if decision.use_vision else "text",
        model=model,
        sources=[
            {
                "page": d.page,
                "modality": d.modality,
                "score": round(d.score, 3),
                "snippet": " ".join(d.text.split())[:160],
            }
            for d in docs
        ],
        image_paths=decision.image_paths,
    )


if __name__ == "__main__":
    import sys

    result = ask(" ".join(sys.argv[1:]) or "What was NovaCore's FY2026 revenue?")
    print(f"[{result.route} / {result.model}]\n{result.answer}\n")
    for s in result.sources:
        print(f"  p{s['page']} {s['modality']:<6} {s['score']}  {s['snippet'][:80]}")
