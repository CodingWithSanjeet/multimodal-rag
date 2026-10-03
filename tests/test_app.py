"""Render the Streamlit app headlessly with the LLM/Pinecone call faked."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from mmrag import answer as answer_mod
from mmrag import embeddings as emb_mod
from mmrag.answer import RagAnswer
from mmrag.config import ROOT_DIR

CALLS = []


def fake_ask(question, k=None):
    CALLS.append(question)
    return RagAnswer(
        question=question,
        answer="Europe grew fastest (31%) on Page 4.",
        route="vision",
        model="qwen/qwen3.8-27b",
        sources=[{"page": 4, "modality": "visual", "score": 0.8, "snippet": "Bar chart"}],
        image_paths=["assets/images/page_4_image_1.png"],
    )


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy")
    monkeypatch.setenv("PINECONE_API_KEY", "dummy")
    monkeypatch.setenv("SESSION_QUESTION_LIMIT", "2")
    monkeypatch.setattr(answer_mod, "ask", fake_ask)
    monkeypatch.setattr(emb_mod, "get_model", lambda: None)
    CALLS.clear()
    at = AppTest.from_file(str(ROOT_DIR / "streamlit_app.py"), default_timeout=60)
    at.run()
    return at


def test_example_button_renders_answer_sources_and_image(app):
    assert not app.exception
    app.button(key="ex1").click().run()
    assert not app.exception
    assert any("Europe grew fastest" in m.value for m in app.markdown)
    assert len(app.dataframe) == 1
    # clicking the same example again is served from st.cache_data
    app.button(key="ex1").click().run()
    assert CALLS.count("Which region had the highest year-over-year revenue growth?") == 1


def test_session_limit(app):
    for q in ["q one?", "q two?", "q three?"]:
        app.text_area(key="question_box").input(q)
        app.button[-1].click().run()  # form submit button is last
    assert any("new questions per session" in w.value for w in app.warning)
    assert CALLS == ["q one?", "q two?"]


def test_missing_keys_stops_app(monkeypatch):
    from mmrag import config

    real = config.get_setting
    monkeypatch.setattr(
        config, "get_setting", lambda n, d=None: None if n.endswith("_API_KEY") else real(n, d)
    )
    at = AppTest.from_file(str(ROOT_DIR / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert any("Missing configuration" in e.value for e in at.error)
