"""steam_retriever.py - Production-grade hybrid retriever for Steam games."""
import os
import faiss
import torch
import numpy as np
import json
import bm25s
import asyncio
from collections import defaultdict
import heapq
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer

DocID = str

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SteamRetriever:
    def __init__(
        self,
        bm25_index_path: str = "index/steam_games_bm25_index",
        faiss_index_path: str = "index/faiss_index/steam_bge_ivfflat.index",
        metadata_path: str = "index/faiss_index/steam_metadata.json",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        rrf_k: int = 60,
        default_k: int = 10
    ):
        self.bm25 = bm25s.BM25().load(os.path.join(
            BASE_DIR, bm25_index_path), load_corpus=True)

        self.faiss_index = faiss.read_index(
            os.path.join(BASE_DIR, faiss_index_path))

        with open(os.path.join(BASE_DIR, metadata_path), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"dtype": dtype},
            device=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        self.rrf_k = rrf_k
        self.default_k = default_k

    @staticmethod
    def reciprocal_rank_fusion(
        results_list: List[List[DocID]],
        k: int = 60,
        final_k: int = 10
    ) -> List[Tuple[DocID, float]]:
        scores: defaultdict[DocID, float] = defaultdict(float)
        for rank_list in results_list:
            for rank, doc_id in enumerate(rank_list, start=1):
                scores[doc_id] += 1.0 / (k + rank)
        if final_k < len(scores) * 0.3:
            return heapq.nlargest(final_k, scores.items(), key=lambda x: x[1])
        else:
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    def _search_sync(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Core synchronous retrieval logic. Returns results with app_id."""
        with torch.no_grad():
            bm25_docs, _ = self.bm25.retrieve(bm25s.tokenize(query), k=k)

            bm25_list: List[str] = []
            bm25_doc_map: Dict[str, str] = {}
            for doc in bm25_docs[0]:
                if isinstance(doc, dict):
                    doc_text = doc.get("text", str(doc))
                    doc_id = str(doc.get("id", ""))
                    if doc_id and doc_id.isdigit():
                        pos = int(doc_id)
                        if pos < len(self.meta["appIds"]):
                            app_id = self.meta["appIds"][pos]
                            bm25_list.append(app_id)
                            bm25_doc_map[app_id] = doc_text
                else:
                    pass

            query_emb = self.model.encode(
                [query], convert_to_tensor=True).cpu().numpy()
            _, indices = self.faiss_index.search(query_emb, k)

            faiss_list: List[str] = []
            faiss_doc_map: Dict[str, str] = {}
            for idx in indices[0]:
                if idx == -1:
                    break
                idx = int(idx)
                app_id = self.meta["appIds"][idx]
                title = self.meta["titles"][idx]
                desc = self.meta["descriptions"][idx]
                faiss_list.append(app_id)
                faiss_doc_map[app_id] = f"{title}: {desc}"

        rrf_ranked = self.reciprocal_rank_fusion(
            [bm25_list, faiss_list],
            k=self.rrf_k,
            final_k=k
        )

        doc_map = {**bm25_doc_map, **faiss_doc_map}

        return [
            {
                "rank": i + 1,
                "app_id": app_id,
                "document": doc_map.get(app_id, ""),
                "rrf_score": round(score, 4)
            }
            for i, (app_id, score) in enumerate(rrf_ranked)
        ]

    async def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = k or self.default_k
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, k)
