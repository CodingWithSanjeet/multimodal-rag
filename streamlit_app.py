"""Streamlit UI for the NovaCore multimodal RAG demo."""

from __future__ import annotations

import os
from dataclasses import asdict

import streamlit as st

from mmrag.config import ROOT_DIR, get_setting, load_settings
from mmrag.ratelimit import MAX_QUESTION_CHARS, DailyCounter, validate_question

st.set_page_config(page_title="Multimodal RAG · NovaCore FY2026", page_icon="📄", layout="wide")

SESSION_LIMIT = int(get_setting("SESSION_QUESTION_LIMIT", "10"))
DAILY_LIMIT = int(get_setting("DAILY_QUESTION_LIMIT", "200"))

EXAMPLES = [
    ("📄 Text", "What does NovaCore Systems do and where is the company headquartered?"),
    ("📊 Table", "Which region had the highest year-over-year revenue growth?"),
    ("📈 Graph", "According to the revenue growth graph, which quarter had the highest revenue?"),
    ("🔀 Diagram", "According to the supply-chain diagram, where is the critical quality-control point?"),
    ("🥧 Pie chart", "What percentage of the Penang facility electricity came from solar energy?"),
    (
        "🖼️ Hybrid",
        "Which product is listed in the table as having the highest gross margin, "
        "and what details are highlighted about it in the product visual?",
    ),
]
EXAMPLE_QUESTIONS = {q for _, q in EXAMPLES}


@st.cache_resource
def daily_counter() -> DailyCounter:
    return DailyCounter(DAILY_LIMIT)


@st.cache_resource
def answered_examples() -> set[str]:
    return set()


@st.cache_resource(show_spinner="Loading embedding model (first visit after a cold start)...")
def warm_up() -> bool:
    from mmrag.embeddings import get_model

    get_model()
    return True


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=64)
def cached_answer(question: str) -> dict:
    from mmrag.answer import ask

    return asdict(ask(question))


def missing_keys() -> list[str]:
    return [k for k in ("GROQ_API_KEY", "PINECONE_API_KEY") if not get_setting(k)]


def run_question(question: str) -> None:
    error = validate_question(question)
    if error:
        st.warning(error)
        return
    question = question.strip()
    is_cached_example = question in EXAMPLE_QUESTIONS and question in answered_examples()
    if not is_cached_example:
        used = st.session_state.get("questions_used", 0)
        if used >= SESSION_LIMIT:
            st.warning(f"This demo allows {SESSION_LIMIT} new questions per session. The example buttons still work.")
            return
        if not daily_counter().try_consume():
            st.warning("The demo hit its daily question budget. Please try again tomorrow, or use the example buttons.")
            return
        st.session_state["questions_used"] = used + 1
    with st.spinner("Retrieving from Pinecone and asking the LLM..."):
        try:
            result = cached_answer(question)
        except Exception as exc:  # show a friendly error, never the keys
            if not is_cached_example:
                st.session_state["questions_used"] -= 1  # failed calls don't use the session quota
            st.error(f"Something went wrong while answering: {type(exc).__name__}. Please try again.")
            return
    if question in EXAMPLE_QUESTIONS:
        answered_examples().add(question)
    st.session_state["last_result"] = result


def render_result(result: dict) -> None:
    st.subheader("Answer")
    st.markdown(result["answer"])
    st.caption(f"Routed to the **{result['route']}** model: `{result['model']}`")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Retrieved sources**")
        st.dataframe(
            [
                {"page": s["page"], "modality": s["modality"], "score": s["score"], "snippet": s["snippet"]}
                for s in result["sources"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    with right:
        if result["image_paths"]:
            st.markdown("**Retrieved visuals**")
            for i, path in enumerate(result["image_paths"]):
                note = " (sent to the vision model)" if i == 0 else ""
                st.image(str(ROOT_DIR / path), caption=f"{os.path.basename(path)}{note}")
        else:
            st.info("No visuals retrieved; answered from text and tables.")


st.title("📄 Multimodal RAG: text, tables, charts and diagrams")
st.markdown(
    "Ask questions about the synthetic **NovaCore Systems FY2026 report**. Text, Markdown tables and "
    "vision-model summaries of charts/diagrams are embedded with MiniLM (fastembed) into Pinecone; "
    "answers come from Groq (`gpt-oss-20b` for text, a Qwen vision model when a chart is retrieved)."
)
st.caption(
    "⏳ Hosted on a free instance that sleeps when idle: the first load after a while can take 1-2 minutes. "
    "After that, answers take a few seconds."
)

missing = missing_keys()
if missing:
    st.error(
        "Missing configuration: " + ", ".join(missing) + ". Set them as environment variables "
        "or in `.streamlit/secrets.toml` (see `.env.example`)."
    )
    st.stop()

warm_up()

st.markdown("**Try an example:**")
cols = st.columns(3)
for i, (label, question) in enumerate(EXAMPLES):
    if cols[i % 3].button(label, help=question, use_container_width=True, key=f"ex{i}"):
        st.session_state["question_box"] = question
        run_question(question)

with st.form("ask"):
    question = st.text_area(
        "Your question", key="question_box", max_chars=MAX_QUESTION_CHARS, height=90,
        placeholder="e.g. What was NovaCore's FY2026 revenue?",
    )
    if st.form_submit_button("Ask", type="primary"):
        run_question(question)

if "last_result" in st.session_state:
    render_result(st.session_state["last_result"])

# Sidebar last, so the quota shown reflects the question just asked.
with st.sidebar:
    st.header("About")
    pdf = load_settings().pdf_path
    if pdf.exists():
        st.download_button("Download the source PDF", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
    st.markdown(f"New questions left this session: **{max(0, SESSION_LIMIT - st.session_state.get('questions_used', 0))}**")
    st.markdown(f"Max question length: **{MAX_QUESTION_CHARS}** characters")
    st.markdown("[Source code on GitHub](https://github.com/CodingWithSanjeet/multimodal-rag)")
