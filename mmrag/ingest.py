"""Offline ingestion: PDF -> text / Markdown tables / visual summaries -> Pinecone.

Run once from your machine (not from the web app):

    python -m mmrag.ingest --images-only   # PyMuPDF only: write assets/images/, no API calls
    python -m mmrag.ingest --dry-run       # extract + summarize (cached), no Pinecone writes
    python -m mmrag.ingest                 # upsert only if the namespace is empty
    python -m mmrag.ingest --reset         # wipe the namespace and re-upsert everything

Vision summaries are cached in data/visual_summaries.json (keyed by image file name +
sha256), so re-running never repeats a vision call for an unchanged image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pymupdf

from .config import ROOT_DIR, Settings, load_settings

VISUAL_SUMMARY_PROMPT = """
This visual was extracted from page {page_number} of the document "{doc_label}".
Describe the useful business information visible in the visual.

If it is a chart or graph:
- mention important data values and axes
- mention highest and lowest values
- describe the main overall trend

If it is a diagram:
- identify key components or stages
- explain the process flow or relationships

If it is a business image:
- describe key factual details shown

Keep the description concise, structured, and strictly factual.
"""


def rel_path(path: Path) -> str:
    """Root-relative POSIX path stored in Pinecone metadata."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def extract_images(pdf_path: Path, image_dir: Path) -> list[dict]:
    """Write every unique embedded image to image_dir. PyMuPDF only, no API calls.
    File names match the notebook: page_{page}_image_{n}.{ext}."""
    image_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    seen_xrefs: set[int] = set()
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1
            for image_number, info in enumerate(page.get_images(full=True), start=1):
                xref = info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                data = pdf.extract_image(xref)
                path = image_dir / f"page_{page_number}_image_{image_number}.{data['ext']}"
                path.write_bytes(data["image"])
                images.append({"page": page_number, "path": path})
    return images


def extract_text_and_tables(pdf_path: Path) -> list[dict]:
    records: list[dict] = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            page_number = page_index + 1
            text = page.get_text("text").strip()
            if text:
                records.append(
                    {
                        "id": f"p{page_number}-text",
                        "text": text,
                        "metadata": {"page": page_number, "modality": "text"},
                    }
                )
            try:
                tables = page.find_tables().tables
            except Exception as error:  # table detection is best-effort
                print(f"Table extraction warning on page {page_number}: {error}")
                tables = []
            for table_number, table in enumerate(tables, start=1):
                df = table.to_pandas()
                if df.empty:
                    continue
                records.append(
                    {
                        "id": f"p{page_number}-table{table_number}",
                        "text": df.to_markdown(index=False),
                        "metadata": {
                            "page": page_number,
                            "modality": "table",
                            "table_number": table_number,
                        },
                    }
                )
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_visual(image_path: Path, page_number: int, doc_label: str, model: str) -> str:
    from .llm import get_groq_client
    from .vision import image_to_data_uri

    prompt = VISUAL_SUMMARY_PROMPT.format(page_number=page_number, doc_label=doc_label)
    response = get_groq_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


def build_visual_records(
    images: list[dict], settings: Settings, doc_label: str, allow_api: bool = True
) -> list[dict]:
    cache = load_cache(settings.summaries_path)
    records: list[dict] = []
    changed = False
    for image in images:
        path: Path = image["path"]
        digest = _sha256(path)
        entry = cache.get(path.name)
        if entry and entry.get("sha256") == digest:
            summary = entry["summary"]
        elif allow_api:
            print(f"Summarizing {path.name} with {settings.vision_model} ...")
            try:
                summary = summarize_visual(path, image["page"], doc_label, settings.vision_model)
            except Exception as error:
                print(f"Vision summary warning for {path.name}: {error}")
                continue
            cache[path.name] = {
                "sha256": digest,
                "page": image["page"],
                "model": settings.vision_model,
                "summary": summary,
            }
            changed = True
        else:
            print(f"Skipping {path.name}: no cached summary and API calls disabled.")
            continue
        records.append(
            {
                "id": f"p{image['page']}-visual-{path.stem}",
                "text": summary,
                "metadata": {
                    "page": image["page"],
                    "modality": "visual",
                    "image_path": rel_path(path),
                },
            }
        )
    if changed:
        save_cache(settings.summaries_path, cache)
    return records


def build_records(settings: Settings, allow_api: bool = True) -> list[dict]:
    pdf_path = settings.pdf_path
    with pymupdf.open(str(pdf_path)) as pdf:
        doc_label = (pdf.metadata or {}).get("title") or pdf_path.stem
    records = extract_text_and_tables(pdf_path)
    images = extract_images(pdf_path, settings.image_dir)
    records += build_visual_records(images, settings, doc_label, allow_api=allow_api)
    for record in records:
        # "text" is the metadata key LangChain's PineconeVectorStore used, so old and
        # new vectors are read the same way by the retriever.
        record["metadata"]["text"] = record["text"]
        record["metadata"]["source"] = pdf_path.name
    records.sort(key=lambda r: (r["metadata"]["page"], r["id"]))
    return records


def namespace_count(index, namespace: str) -> int:
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {}) if isinstance(stats, dict) else stats.namespaces
    ns = namespaces.get(namespace) if namespaces else None
    if ns is None:
        return 0
    return ns["vector_count"] if isinstance(ns, dict) else ns.vector_count


def ensure_index(pc, settings: Settings):
    from pinecone import ServerlessSpec

    if not pc.has_index(settings.pinecone_index):
        print(f"Creating Pinecone index {settings.pinecone_index} ...")
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(settings.pinecone_index).status["ready"]:
            time.sleep(2)
    return pc.Index(settings.pinecone_index)


def upsert_records(index, records: list[dict], namespace: str, batch_size: int = 50) -> None:
    from .embeddings import embed_documents

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        vectors = embed_documents([r["text"] for r in batch])
        index.upsert(
            vectors=[
                {"id": r["id"], "values": v, "metadata": r["metadata"]}
                for r, v in zip(batch, vectors)
            ],
            namespace=namespace,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-only", action="store_true", help="only extract images (no API calls)")
    parser.add_argument("--dry-run", action="store_true", help="extract + summarize, do not touch Pinecone")
    parser.add_argument("--no-vision", action="store_true", help="never call the vision API; use cached summaries only")
    parser.add_argument("--reset", action="store_true", help="DELETE the namespace and re-upsert everything")
    args = parser.parse_args(argv)

    settings = load_settings()
    if not settings.pdf_path.exists():
        print(f"PDF not found: {settings.pdf_path}", file=sys.stderr)
        return 1

    if args.images_only:
        images = extract_images(settings.pdf_path, settings.image_dir)
        for image in images:
            print(f"page {image['page']}: {rel_path(image['path'])}")
        print(f"{len(images)} images written to {rel_path(settings.image_dir)}/")
        return 0

    records = build_records(settings, allow_api=not args.no_vision)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["metadata"]["modality"]] = counts.get(r["metadata"]["modality"], 0) + 1
    print(f"Built {len(records)} records: {counts}")
    if args.dry_run:
        return 0

    from pinecone import Pinecone

    from .config import require_setting

    pc = Pinecone(api_key=require_setting("PINECONE_API_KEY"))
    index = ensure_index(pc, settings)
    existing = namespace_count(index, settings.pinecone_namespace)
    if existing and not args.reset:
        print(
            f"Namespace '{settings.pinecone_namespace}' already has {existing} vectors; "
            "nothing written. Re-run with --reset to wipe and re-index."
        )
        return 0
    if existing and args.reset:
        print(f"--reset: deleting {existing} vectors in '{settings.pinecone_namespace}' ...")
        index.delete(delete_all=True, namespace=settings.pinecone_namespace)
    upsert_records(index, records, settings.pinecone_namespace)
    print(f"Upserted {len(records)} vectors into {settings.pinecone_index}/{settings.pinecone_namespace}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
