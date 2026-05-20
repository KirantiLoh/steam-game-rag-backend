"""Elasticsearch indexing: parquet metadata + CLIP vectors."""
import os
import logging
from typing import Any

import faiss
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ES_URL = "http://localhost:9200"
INDEX_NAME = "steam_games"
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("elastic_index")


def _to_list(val: Any) -> list:
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return []


def create_index(es: Elasticsearch):
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "app_id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "english"},
                "short_description": {"type": "text", "analyzer": "english"},
                "description": {"type": "text", "analyzer": "english"},
                "header_image": {"type": "keyword"},
                "screenshots": {"type": "keyword"},
                "price": {"type": "float"},
                "genres": {"type": "keyword"},
                "platforms": {"type": "keyword"},
                "developers": {"type": "keyword"},
                "publishers": {"type": "keyword"},
                "release_date": {"type": "keyword"},
                "metacritic_score": {"type": "integer"},
                "steam_rating": {"type": "integer"},
                "positive_reviews": {"type": "integer"},
                "negative_reviews": {"type": "integer"},
                "image_vector": {
                    "type": "dense_vector",
                    "dims": 512,
                    "index": True,
                    "similarity": "cosine",
                },
            }
        },
    }
    if es.indices.exists(index=INDEX_NAME):
        logger.info("Deleting existing index %s", INDEX_NAME)
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body=mapping)
    logger.info("Index %s created", INDEX_NAME)


def main():
    es = Elasticsearch(ES_URL, request_timeout=120)
    create_index(es)

    # Load FAISS index to extract CLIP vectors
    logger.info("Loading FAISS index...")
    faiss_index = faiss.read_index(os.path.join(BASE_DIR, "index/steam_games.index"))
    n_total = faiss_index.ntotal
    logger.info("Total vectors: %d", n_total)

    # Bulk-extract all vectors from FAISS
    logger.info("Extracting all vectors (this may take a moment)...")
    all_vectors: np.ndarray = faiss_index.reconstruct_n(0, n_total)  # (N, 512)
    logger.info("Vectors shape: %s", all_vectors.shape)

    # Load metadata parquet (app_id / name mapping)
    meta_df = pd.read_parquet(os.path.join(BASE_DIR, "index/metadata.parquet"))

    # Load main parquet for full game data
    logger.info("Loading main parquet...")
    main_df = pd.read_parquet(os.path.join(BASE_DIR, "index/data/train-00000-of-00001.parquet"))
    game_lookup: dict[str, Any] = {}
    for row in main_df.itertuples(index=False):
        game_lookup[str(row.appID)] = row
    logger.info("Loaded %d games from parquet", len(game_lookup))

    # Build bulk actions
    actions: list[dict] = []
    indexed_count = 0
    for i in range(n_total):
        app_id = str(meta_df.iloc[i]["app_id"])
        name = meta_df.iloc[i]["name"]

        vec = all_vectors[i].tolist()

        game_row = game_lookup.get(app_id)
        if game_row is None:
            continue

        platforms = []
        if hasattr(game_row, "windows") and game_row.windows:
            platforms.append("Windows")
        if hasattr(game_row, "mac") and game_row.mac:
            platforms.append("Mac")
        if hasattr(game_row, "linux") and game_row.linux:
            platforms.append("Linux")

        pos = int(game_row.positive) if game_row.positive else 0
        neg = int(game_row.negative) if game_row.negative else 0
        rating = round(pos / (pos + neg) * 100) if (pos + neg) > 0 else 0

        doc = {
            "app_id": app_id,
            "name": name,
            "description": game_row.detailed_description or "",
            "short_description": game_row.short_description or "",
            "header_image": game_row.header_image or "",
            "screenshots": [str(s) for s in (_to_list(game_row.screenshots) if hasattr(game_row, 'screenshots') else []) if s],
            "price": float(game_row.price) if (hasattr(game_row, 'price') and game_row.price) else 0,
            "genres": [str(g) for g in _to_list(game_row.genres) if g] if hasattr(game_row, 'genres') else [],
            "platforms": platforms,
            "developers": [str(d) for d in _to_list(game_row.developers) if d] if hasattr(game_row, 'developers') else [],
            "publishers": [str(p) for p in _to_list(game_row.publishers) if p] if hasattr(game_row, 'publishers') else [],
            "release_date": str(game_row.release_date) if hasattr(game_row, 'release_date') and game_row.release_date else "",
            "metacritic_score": int(game_row.metacritic_score) if (hasattr(game_row, 'metacritic_score') and game_row.metacritic_score) else 0,
            "steam_rating": rating,
            "positive_reviews": pos,
            "negative_reviews": neg,
            "image_vector": vec,
        }

        actions.append({"_index": INDEX_NAME, "_id": app_id, "_source": doc})

        if len(actions) >= BATCH_SIZE:
            successes, errors = helpers.bulk(es, actions, raise_on_error=False)
            indexed_count += successes
            if errors:
                logger.warning("Batch had %d errors (first: %s)", len(errors), errors[0])
            logger.info("Indexed %d / %d", indexed_count, n_total)
            actions = []

    if actions:
        successes, errors = helpers.bulk(es, actions, raise_on_error=False)
        indexed_count += successes
        if errors:
            logger.warning("Final batch had %d errors", len(errors))

    logger.info("Finished! Indexed %d documents", indexed_count)
    logger.info("Refresh & check count...")
    es.indices.refresh(index=INDEX_NAME)
    count = es.count(index=INDEX_NAME)["count"]
    logger.info("ES doc count: %d", count)


if __name__ == "__main__":
    import sys
    main()
