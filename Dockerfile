FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download HF models at build time (avoids runtime download)
ARG HF_TOKEN
RUN HF_TOKEN=$HF_TOKEN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')"
RUN HF_TOKEN=$HF_TOKEN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('clip-ViT-B-32', device='cpu')"

COPY api/ api/
COPY es_retriever.py game_store.py reranker.py elastic_index.py ./
COPY index/data/train-00000-of-00001.parquet index/data/
COPY index/steam_games.index index/metadata.parquet index/

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
