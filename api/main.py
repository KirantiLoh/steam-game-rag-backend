"""Steam Game Recommender — ES-based hybrid search + cross-encoder reranker."""
import asyncio
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from es_retriever import ESRetriever
from game_store import GameStore
from reranker import Reranker

logger = logging.getLogger("steamrec")


class SearchCache:
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - self._timestamps[key] > self.ttl:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return entry

    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)
        while len(self._cache) > self.maxsize:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._timestamps.pop(oldest, None)

    def invalidate(self, prefix: Optional[str] = None):
        if prefix:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
        else:
            self._cache.clear()
            self._timestamps.clear()


search_cache = SearchCache()
retriever: ESRetriever = None
game_store: GameStore = None
reranker: Reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, game_store, reranker
    logger.info("Loading models…")
    loop = asyncio.get_running_loop()

    def _load():
        r = ESRetriever(default_k=50)
        g = GameStore()
        rn = Reranker()
        logger.info("ES doc count: %d", r.es.count(index="steam_games")["count"])
        return r, g, rn

    retriever, game_store, reranker = await loop.run_in_executor(None, _load)
    logger.info("All models loaded")
    yield


app = FastAPI(title="SteamRec API", lifespan=lifespan)


class GameOut(BaseModel):
    id: int
    name: str
    description: str
    short_description: str
    header_image: str
    screenshots: List[str]
    price: float = Field(default=0.0)
    genres: List[str] = Field(default_factory=list)
    developers: List[str] = Field(default_factory=list)
    publishers: List[str] = Field(default_factory=list)
    release_date: str = ""
    metacritic_score: int = 0
    steam_rating: int = 0
    positive_reviews: int = 0
    negative_reviews: int = 0
    platforms: List[str] = Field(default_factory=list)


class SearchResultOut(BaseModel):
    game: GameOut
    score: float


class SearchResponseOut(BaseModel):
    results: List[SearchResultOut]
    query: str
    total: int


def _preprocess_query(raw: str) -> str:
    return raw.strip().lower()


def _enrich(app_id_score_pairs: list) -> list:
    seen = set()
    enriched = []
    for app_id, score in app_id_score_pairs:
        if app_id in seen:
            continue
        seen.add(app_id)
        game = game_store.get_game_by_app_id(app_id)
        if game is not None:
            enriched.append({"game": game, "score": round(float(score), 4)})
    return enriched


# ── Health ──────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Trending ───────────────────────────────────────────────────────


@app.get("/api/trending", response_model=SearchResponseOut)
async def trending(limit: int = Query(12, ge=1, le=50)):
    results = game_store.get_trending(limit)
    return SearchResponseOut(results=results, query="trending", total=len(results))


# ── Search (ES hybrid + optional reranker) ─────────────────────────


@app.get("/api/search", response_model=SearchResponseOut)
async def search(
    q: str = Query("", min_length=0, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    rerank: bool = Query(True, description="Apply cross-encoder reranking"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    platform: Optional[str] = Query(None, description="Filter by platform (Windows, Mac, Linux)"),
    price_min: Optional[float] = Query(None, ge=0, description="Minimum price"),
    price_max: Optional[float] = Query(None, ge=0, description="Maximum price"),
):
    query = _preprocess_query(q)
    if not query:
        return SearchResponseOut(results=[], query=q, total=0)

    try:
        cache_key = f"s|{query}|{limit}|{rerank}|{genre}|{platform}|{price_min}|{price_max}"
        cached = search_cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for query=%s", query)
            return SearchResponseOut(results=_enrich(cached), query=query, total=len(cached))

        fuse_k = limit * 2 if rerank else limit
        loop = asyncio.get_running_loop()
        fused = await loop.run_in_executor(
            None, retriever.search, query, fuse_k, genre, platform, price_min, price_max
        )

        if rerank and fused:
            candidates = []
            for app_id, _ in fused:
                game = game_store.get_game_by_app_id(app_id)
                if game is None:
                    continue
                name = game.get("name", "")
                short_desc = game.get("short_description", "")
                desc = game.get("description", "")
                combined = f"{name}\n\n{short_desc}\n\n{desc}" if short_desc else f"{name}\n\n{desc}"
                candidates.append((app_id, combined))

            reranked = await loop.run_in_executor(
                None, reranker.rerank, query, candidates, limit
            )
            scored = reranked
        else:
            scored = fused

        enriched = _enrich(scored)
        search_cache.set(cache_key, scored)
        return SearchResponseOut(results=enriched, query=query, total=len(enriched))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search service unavailable")


# ── Image Search ────────────────────────────────────────────────────


@app.post("/api/search/image", response_model=SearchResponseOut)
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(20, ge=1, le=100),
):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, retriever.search_image, image, limit)
        enriched = _enrich(results)
        return SearchResponseOut(
            results=enriched, query=f"image search: {file.filename or 'uploaded image'}", total=len(enriched)
        )
    except Exception:
        logger.exception("Image search failed")
        raise HTTPException(status_code=500, detail="Image search failed")


# ── Game Detail ─────────────────────────────────────────────────────


@app.get("/api/games/{game_id}", response_model=GameOut)
async def get_game(game_id: int):
    try:
        game = game_store.get_game(game_id)
        if game is None:
            raise HTTPException(status_code=404, detail="Game not found")
        return game
    except HTTPException:
        raise
    except Exception:
        logger.exception("Game lookup failed")
        raise HTTPException(status_code=500, detail="Game lookup failed")


# ── Similar Games ────────────────────────────────────────────────────


@app.get("/api/games/{game_id}/similar", response_model=SearchResponseOut)
async def similar_games(
    game_id: int,
    limit: int = Query(8, ge=1, le=50),
    rerank: bool = Query(False, description="Apply cross-encoder reranking"),
):
    try:
        app_id = str(game_id)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, retriever.search_similar, app_id, limit * 2 if rerank else limit
        )
        if not results:
            return SearchResponseOut(results=[], query=f"similar to {game_id}", total=0)

        if rerank:
            source_game = game_store.get_game_by_app_id(app_id)
            query_desc = f"{source_game.get('name', '')}\n\n" if source_game else ""

            candidates = []
            for candidate_app_id, _ in results:
                game = game_store.get_game_by_app_id(candidate_app_id)
                if game is None:
                    continue
                name = game.get("name", "")
                short_desc = game.get("short_description", "")
                desc = game.get("description", "")
                combined = f"{name}\n\n{short_desc}\n\n{desc}" if short_desc else f"{name}\n\n{desc}"
                candidates.append((candidate_app_id, combined))

            reranked = await loop.run_in_executor(
                None, reranker.rerank, query_desc, candidates, limit
            )
            enriched = _enrich(reranked)
        else:
            enriched = _enrich(results)

        return SearchResponseOut(
            results=enriched, query=f"similar to {game_id}", total=len(enriched)
        )
    except Exception:
        logger.exception("Similar games failed")
        raise HTTPException(status_code=500, detail="Similar games lookup failed")
