"""Unified game metadata store using FronkonGames/steam-games-dataset."""
import os
import numpy as np
import pandas as pd
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _to_list(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return []


class GameStore:
    def __init__(
        self,
        parquet_path: str = "index/data/train-00000-of-00001.parquet",
    ):
        full_path = os.path.join(BASE_DIR, parquet_path)
        df = pd.read_parquet(full_path)

        self._store: dict[str, dict] = {}
        for row in df.itertuples(index=False):
            app_id = str(row.appID)
            platforms = []
            if hasattr(row, 'windows') and row.windows:
                platforms.append("Windows")
            if hasattr(row, 'mac') and row.mac:
                platforms.append("Mac")
            if hasattr(row, 'linux') and row.linux:
                platforms.append("Linux")

            self._store[app_id] = {
                "id": int(app_id),
                "name": row.name or f"Game {app_id}",
                "description": row.detailed_description or "",
                "short_description": row.short_description or "",
                "header_image": row.header_image or f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{app_id}/header.jpg",
                "screenshots": _to_list(row.screenshots),
                "price": float(row.price) if row.price else 0,
                "genres": _to_list(row.genres),
                "developers": _to_list(row.developers),
                "publishers": _to_list(row.publishers),
                "release_date": row.release_date or "",
                "metacritic_score": int(row.metacritic_score) if row.metacritic_score else 0,
                "steam_rating": round(row.positive / (row.positive + row.negative) * 100) if (row.positive + row.negative) > 0 else 0,
                "positive_reviews": int(row.positive) if row.positive else 0,
                "negative_reviews": int(row.negative) if row.negative else 0,
                "platforms": platforms,
            }

    def get_game_by_app_id(self, app_id: str) -> Optional[dict]:
        return self._store.get(app_id)

    def get_game(self, game_id: int) -> Optional[dict]:
        return self._store.get(str(game_id))

    def get_trending(self, n: int = 12) -> list:
        scored = [
            (app_id, g.get("positive_reviews", 0) + g.get("metacritic_score", 0) * 100)
            for app_id, g in self._store.items()
        ]
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:n]
        return [{"game": self._store[app_id], "score": 1.0} for app_id, _ in top]

    def enrich_results(self, results_with_app_id: list) -> list:
        enriched = []
        for app_id, score in results_with_app_id:
            game = self.get_game_by_app_id(app_id)
            if game is not None:
                enriched.append({"game": game, "score": round(score, 4)})
        return enriched