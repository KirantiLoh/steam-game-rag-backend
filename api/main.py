"""Steam Game Recommender API — search, rerank, similar, image search."""
import asyncio
import heapq
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from PIL import Image

from text_retriever.retriever import SteamRetriever
from image_retriever.retriever import ImageHNSWRetriever
from game_store import GameStore
from reranker import Reranker

logger = logging.getLogger("steamrec")

retriever: SteamRetriever = None
image_retriever: ImageHNSWRetriever = None
game_store: GameStore = None
reranker: Reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, image_retriever, game_store, reranker
    logger.info("Loading models and indices...")
    loop = asyncio.get_running_loop()

    def _load():
        t = SteamRetriever(default_k=50)
        i = ImageHNSWRetriever(default_k=50)
        g = GameStore()
        r = Reranker()
        return t, i, g, r

    retriever, image_retriever, game_store, reranker = await loop.run_in_executor(None, _load)
    logger.info("All models loaded successfully")
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


def _fuse_results(
    text_ranked: List[str],
    image_ranked: List[str],
    limit: int,
) -> List:
    scores: dict[str, float] = defaultdict(float)
    for rank_list in [text_ranked, image_ranked]:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] += 1.0 / (60 + rank)
    if limit < len(scores) * 0.3:
        return heapq.nlargest(limit, scores.items(), key=lambda x: x[1])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]


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
async def trending(
    limit: int = Query(12, ge=1, le=50),
):
    results = game_store.get_trending(limit)
    return SearchResponseOut(results=results, query="trending", total=len(results))


# ── Text Search (with RRF + Reranker) ───────────────────────────────

@app.get("/api/search", response_model=SearchResponseOut)
async def search(
    q: str = Query("", min_length=0, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    rerank: bool = Query(True, description="Apply cross-encoder reranking"),
):
    query = q.strip()
    if not query:
        return SearchResponseOut(results=[], query=q, total=0)

    try:
        text_results = await retriever.search(query, k=limit * 2)
        loop = asyncio.get_running_loop()
        image_results = await loop.run_in_executor(
            None, image_retriever.search_text, query, limit * 2
        )

        text_ranked = [r["app_id"] for r in text_results if r.get("app_id")]
        image_ranked = [app_id for app_id, _ in image_results if app_id]

        fused = _fuse_results(text_ranked, image_ranked, limit * 2 if rerank else limit)

        if rerank and fused:
            candidates = []
            for app_id, _ in fused:
                game = game_store.get_game_by_app_id(app_id)
                if game is None:
                    continue
                desc = game.get("description", "") or game.get("short_description", "") or game.get("name", "")
                candidates.append((app_id, desc))

            loop = asyncio.get_running_loop()
            reranked = await loop.run_in_executor(
                None, reranker.rerank, query, candidates, limit
            )
            enriched = _enrich(reranked)
        else:
            enriched = _enrich(fused)

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
        results = await loop.run_in_executor(
            None, image_retriever.search_image, image, limit
        )
        enriched = _enrich(results)
        query = f"image search: {file.filename or 'uploaded image'}"
        return SearchResponseOut(results=enriched, query=query, total=len(enriched))
    except Exception as e:
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
    except Exception as e:
        logger.exception("Game lookup failed")
        raise HTTPException(status_code=500, detail="Game lookup failed")


# ── Similar Games ────────────────────────────────────────────────────

@app.get("/api/games/{game_id}/similar", response_model=SearchResponseOut)
async def similar_games(
    game_id: int,
    limit: int = Query(8, ge=1, le=50),
):
    try:
        app_id = str(game_id)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, image_retriever.search_similar, app_id, limit
        )
        if not results:
            return SearchResponseOut(
                results=[], query=f"similar to {game_id}", total=0
            )
        enriched = _enrich(results)
        return SearchResponseOut(
            results=enriched, query=f"similar to {game_id}", total=len(enriched)
        )
    except Exception as e:
        logger.exception("Similar games failed")
        raise HTTPException(status_code=500, detail="Similar games lookup failed")
