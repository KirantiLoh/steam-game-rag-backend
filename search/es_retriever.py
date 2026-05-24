"""Elasticsearch-based hybrid retriever for Steam games."""
import asyncio
import logging
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import torch
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("steamrec")

INDEX_NAME = "steam_games"
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
RRF_K = 60
TEXT_WEIGHT = 1.5
IMAGE_WEIGHT = 1.0


class ESRetriever:
    def __init__(self, default_k: int = 50):
        self.es = Elasticsearch(ES_URL)
        self.default_k = default_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = SentenceTransformer(
            "clip-ViT-B-32", device=self.device
        )
        self.clip_model.eval()

    # ── helpers ──────────────────────────────────────────────────

    def _build_filter(self, genre: Optional[str] = None, platform: Optional[str] = None,
                      price_min: Optional[float] = None, price_max: Optional[float] = None) -> list:
        clauses = []
        if genre:
            clauses.append({"term": {"genres": genre.lower()}})
        if platform:
            clauses.append({"term": {"platforms": platform}})
        if price_min is not None:
            clauses.append({"range": {"price": {"gte": price_min}}})
        if price_max is not None:
            clauses.append({"range": {"price": {"lte": price_max}}})
        return clauses

    def _es_text_search(self, query: str, size: int, filters: list) -> list:
        must = {"multi_match": {"query": query, "fields": ["name^3", "short_description^2", "description"]}}
        body = {"query": {"bool": {"must": must}}, "size": size}
        if filters:
            body["query"]["bool"]["filter"] = filters
        resp = self.es.search(index=INDEX_NAME, body=body)
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    def _es_vector_search(self, vec: list, size: int, filters: list) -> list:
        knn = {
            "field": "image_vector",
            "query_vector": vec,
            "k": size,
            "num_candidates": size * 2,
        }
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        body = {"knn": knn, "size": size}
        resp = self.es.search(index=INDEX_NAME, body=body)
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    def _rrf_fuse(self, text_ranked: list, vector_ranked: list,
                  limit: int) -> list:
        scores: dict[str, float] = defaultdict(float)
        for rank, (doc_id, _) in enumerate(text_ranked, start=1):
            scores[doc_id] += TEXT_WEIGHT / (RRF_K + rank)
        for rank, (doc_id, _) in enumerate(vector_ranked, start=1):
            scores[doc_id] += IMAGE_WEIGHT / (RRF_K + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    def _encode_text(self, text: str) -> list:
        with torch.no_grad():
            vec = self.clip_model.encode([text], convert_to_tensor=False)
        return vec[0].tolist()

    def _encode_image(self, image) -> list:
        with torch.no_grad():
            vec = self.clip_model.encode(image, convert_to_tensor=False)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return vec[0].tolist()

    # ── public methods ───────────────────────────────────────────

    def search(self, query: str, k: Optional[int] = None,
               genre: Optional[str] = None, platform: Optional[str] = None,
               price_min: Optional[float] = None, price_max: Optional[float] = None) -> list:
        k = k or self.default_k
        filters = self._build_filter(genre, platform, price_min, price_max)

        text_results = self._es_text_search(query, k, filters)
        vec = self._encode_text(query)
        vector_results = self._es_vector_search(vec, k, filters)
        return self._rrf_fuse(text_results, vector_results, k)

    def search_image(self, image, k: Optional[int] = None) -> list:
        k = k or self.default_k
        vec = self._encode_image(image)
        filters = []
        body = {"knn": {"field": "image_vector", "query_vector": vec, "k": k, "num_candidates": k * 2}, "size": k}
        resp = self.es.search(index=INDEX_NAME, body=body)
        return [(hit["_id"], hit["_score"]) for hit in resp["hits"]["hits"]]

    def search_similar(self, app_id: str, k: Optional[int] = None) -> list:
        k = k or self.default_k
        try:
            resp = self.es.get(index=INDEX_NAME, id=app_id, _source_includes=["image_vector"])
        except NotFoundError:
            return []
        src = resp["_source"]
        if "image_vector" not in src:
            return []
        vec = src["image_vector"]
        body = {"knn": {"field": "image_vector", "query_vector": vec, "k": k + 1, "num_candidates": (k + 1) * 2}, "size": k + 1}
        resp = self.es.search(index=INDEX_NAME, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            if hit["_id"] == app_id:
                continue
            results.append((hit["_id"], hit["_score"]))
            if len(results) == k:
                break
        return results

    def get_vector(self, app_id: str) -> Optional[list]:
        try:
            resp = self.es.get(index=INDEX_NAME, id=app_id, _source_includes=["image_vector"])
            return resp["_source"].get("image_vector")
        except NotFoundError:
            return None

    def get_source(self, app_id: str) -> Optional[dict]:
        try:
            resp = self.es.get(index=INDEX_NAME, id=app_id)
            src = resp["_source"]
            del src["image_vector"]
            return src
        except (NotFoundError, KeyError):
            return None
