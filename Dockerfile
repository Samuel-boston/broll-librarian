# B-Roll Librarian on a server: the web app and the ingest worker in one container.
# Search embeddings come from the Gemini API rather than a local model, so there is no PyTorch here and
# a small server (2 GB of RAM) is plenty. See docs/HOSTING.md.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[gemini,drive,shots,web]"

RUN useradd --create-home --uid 10001 broll && mkdir -p /data && chown broll /data
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER broll
ENV BROLL_HOME=/data \
    BROLL_WORKSPACE=library \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
