# Optional: container deploy (Render Docker runtime, Koyeb, Fly.io, Railway...).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mmrag ./mmrag
# Bake the ONNX MiniLM model into the image (no download at cold start).
RUN python -m mmrag.embeddings

COPY streamlit_app.py NovaCore_Multimodal_Company_Report_2026.pdf ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY assets ./assets
COPY data ./data

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8501
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"]
