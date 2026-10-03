from pathlib import Path

import pytest

from mmrag.answer import clean_answer, format_context
from mmrag.config import ROOT_DIR
from mmrag.ratelimit import DailyCounter, validate_question
from mmrag.retriever import RetrievedDoc, match_to_doc, resolve_image_path, to_int_page
from mmrag.router import route


def doc(modality="text", image_path=None, page=1, text="x"):
    return RetrievedDoc(text=text, page=page, modality=modality, image_path=image_path)


@pytest.mark.parametrize("value,expected", [(4.0, 4), (4, 4), ("7", 7), ("3.0", 3), (None, None), ("", None), ("abc", None)])
def test_to_int_page(value, expected):
    assert to_int_page(value) == expected


def test_resolve_old_notebook_path_maps_to_assets():
    assert resolve_image_path("novacore_extracted_images/page_3_image_1.png") == "assets/images/page_3_image_1.png"
    assert resolve_image_path("C:\\old\\novacore_extracted_images\\page_8_image_1.png") == "assets/images/page_8_image_1.png"
    assert resolve_image_path("assets/images/page_5_image_1.png") == "assets/images/page_5_image_1.png"


def test_resolve_missing_image_returns_none():
    assert resolve_image_path("novacore_extracted_images/page_99_image_1.png") is None
    assert resolve_image_path(None) is None


def test_all_seven_images_present():
    images = sorted(p.name for p in (ROOT_DIR / "assets" / "images").glob("*.png"))
    assert len(images) == 7


def test_match_to_doc_with_legacy_langchain_metadata():
    match = {
        "id": "abc",
        "score": 0.81,
        "metadata": {
            "image_path": "novacore_extracted_images/page_4_image_1.png",
            "modality": "visual",
            "page": 4.0,
            "source": "NovaCore_Multimodal_Company_Report_2026.pdf",
            "text": "Bar chart of growth by region",
        },
    }
    d = match_to_doc(match)
    assert d.page == 4 and d.modality == "visual"
    assert d.image_path == "assets/images/page_4_image_1.png"
    assert d.text.startswith("Bar chart")


def test_router_text_only():
    r = route([doc("text"), doc("table")])
    assert not r.use_vision and r.image_paths == []


def test_router_visual_dedup_and_order():
    a, b = "assets/images/page_3_image_1.png", "assets/images/page_4_image_1.png"
    r = route([doc("table"), doc("visual", b), doc("visual", a), doc("visual", b)])
    assert r.use_vision and r.image_paths == [b, a]


def test_router_visual_without_local_image_falls_back_to_text():
    assert not route([doc("visual", None)]).use_vision


def test_clean_answer_strips_glyphs_and_float_pages():
    raw = "HQ is Singapore【Page: 1.0 | TEXT】. Revenue is on Page 2.0 and pages 4.0."
    assert clean_answer(raw) == "HQ is Singapore. Revenue is on Page 2 and pages 4."


def test_clean_answer_keeps_other_decimals():
    assert clean_answer("Revenue was $37.5M, up 2.0x") == "Revenue was $37.5M, up 2.0x"


def test_format_context():
    assert format_context([doc("table", page=4, text="| a |")]) == "[Page: 4 | TABLE]\n| a |"


def test_validate_question():
    assert validate_question("  ") is not None
    assert validate_question("x" * 501) is not None
    assert validate_question("x" * 500) is None


def test_daily_counter():
    c = DailyCounter(2)
    assert c.try_consume() and c.try_consume()
    assert not c.try_consume()
    assert c.remaining == 0


def test_summary_cache_matches_images():
    import json

    cache = json.loads((ROOT_DIR / "data" / "visual_summaries.json").read_text())
    for name in cache:
        assert (ROOT_DIR / "assets" / "images" / name).exists()
