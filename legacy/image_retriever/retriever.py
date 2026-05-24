"""HNSW-based image retriever using CLIP-ViT-B-32 and FAISS."""
import os
import warnings
warnings.filterwarnings("ignore", message=".*slow image processor.*")

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageHNSWRetriever:
    def __init__(
        self,
        index_path: str = "index/steam_games.index",
        metadata_path: str = "index/metadata.parquet",
        model_name: str = "clip-ViT-B-32",
        device: Optional[str] = None,
        default_k: int = 20,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.index = faiss.read_index(os.path.join(BASE_DIR, index_path))
        self.df = pd.read_parquet(os.path.join(BASE_DIR, metadata_path))
        self.app_ids = self.df["app_id"].tolist()
        self.names = self.df["name"].tolist()

        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

        self.default_k = default_k

    def _encode_and_search(self, vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        vec = vec.astype("float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.app_ids):
                continue
            results.append((self.app_ids[idx], float(dist)))
        return results

    def search_text(self, query: str, k: Optional[int] = None) -> List[Tuple[str, float]]:
        k = k or self.default_k
        with torch.no_grad():
            vec = self.model.encode([query], convert_to_tensor=False)
        return self._encode_and_search(vec, k)

    def search_image(self, image, k: Optional[int] = None) -> List[Tuple[str, float]]:
        k = k or self.default_k
        with torch.no_grad():
            vec = self.model.encode(image, convert_to_tensor=False)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return self._encode_and_search(vec, k)

    def search_similar(self, app_id: str, k: Optional[int] = None) -> List[Tuple[str, float]]:
        k = k or self.default_k
        try:
            row_idx = self.app_ids.index(app_id)
        except ValueError:
            return []
        vec = self.index.reconstruct(row_idx)
        return self._encode_and_search(vec, k + 1)[:k]
