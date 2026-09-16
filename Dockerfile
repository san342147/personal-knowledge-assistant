FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface HF_HUB_DISABLE_SYMLINKS_WARNING=1
WORKDIR /app
COPY requirements.txt .
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && useradd --create-home --uid 10001 groundeddesk
COPY app/ app/
RUN mkdir -p /app/vectorstore /app/logs /app/.cache && chown -R groundeddesk:groundeddesk /app
USER groundeddesk
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)"
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
