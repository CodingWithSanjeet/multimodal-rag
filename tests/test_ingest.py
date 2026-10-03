from mmrag.config import load_settings
from mmrag.ingest import build_records, extract_text_and_tables


def test_extract_text_and_tables_offline():
    records = extract_text_and_tables(load_settings().pdf_path)
    modalities = [r["metadata"]["modality"] for r in records]
    assert modalities.count("text") == 9
    assert modalities.count("table") == 8
    assert len({r["id"] for r in records}) == len(records)  # deterministic, unique ids


def test_build_records_without_api_uses_cache_only():
    records = build_records(load_settings(), allow_api=False)
    visuals = [r for r in records if r["metadata"]["modality"] == "visual"]
    assert len(visuals) == 6  # page 7 summary is not cached yet
    assert all(r["metadata"]["image_path"].startswith("assets/images/") for r in visuals)
    assert all(r["metadata"]["text"] == r["text"] for r in records)
