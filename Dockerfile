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

# Download game metadata + FAISS index from HF at build time
RUN mkdir -p index/data && \
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/data/train-00000-of-00001.parquet" \
      -o index/data/train-00000-of-00001.parquet && \
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/datasets/Frmeta/tk-tbi-steam-index-img-only-hnsw/resolve/main/steam_games.index" \
      -o index/steam_games.index && \
    curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/datasets/Frmeta/tk-tbi-steam-index-img-only-hnsw/resolve/main/metadata.parquet" \
      -o index/metadata.parquet

COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
