"""Search endpoints for game discovery."""
import asyncio
import logging
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from PIL import Image

from api.models import SearchResponseOut, GameOut

logger = logging.getLogger("steamrec")

router = APIRouter(prefix="/api", tags=["search"])

# Global state (injected from main.py)
retriever = None
game_store = None
reranker = None
search_cache = None


def _preprocess_query(raw: str) -> str:
    """Normalize query string."""
    return raw.strip().lower()


def _enrich(app_id_score_pairs: list) -> list:
    """Enrich app_id/score pairs with full game metadata."""
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


@router.get("/trending", response_model=SearchResponseOut)
async def trending(limit: int = Query(12, ge=1, le=50)):
    """Get trending games based on popularity."""
    results = game_store.get_trending(limit)
    return SearchResponseOut(results=results, query="trending", total=len(results))


@router.get("/search", response_model=SearchResponseOut)
async def search(
    q: str = Query("", min_length=0, max_length=200),
    limit: int = Query(20, ge=1, le=100),
    rerank: bool = Query(True, description="Apply cross-encoder reranking"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    platform: Optional[str] = Query(None, description="Filter by platform (Windows, Mac, Linux)"),
    price_min: Optional[float] = Query(None, ge=0, description="Minimum price"),
    price_max: Optional[float] = Query(None, ge=0, description="Maximum price"),
):
    """
    Hybrid search with BM25 text matching + CLIP vector similarity.

    Optional cross-encoder reranking for improved relevance.
    """
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
            # Batch fetch all games once to avoid N+1 lookups
            app_ids = [app_id for app_id, _ in fused]
            games_cache = {k: v for k, v in game_store.get_games_batch(app_ids).items() if v is not None}

            # Prepare candidates with optimized text formatting
            candidates = []
            for app_id, _ in fused:
                game = games_cache.get(app_id)
                if game is None:
                    continue
                # Use reranker's smart text preparation for optimal token usage
                combined = reranker.prepare_text(
                    game.get("name", ""),
                    game.get("short_description", ""),
                    game.get("description", "")
                )
                candidates.append((app_id, combined))

            reranked = await loop.run_in_executor(
                None, reranker.rerank, query, candidates, limit
            )
            scored = reranked

            # Reuse cached games for enrichment (avoid duplicate lookups)
            seen = set()
            enriched = []
            for app_id, score in scored:
                if app_id in seen or app_id not in games_cache:
                    continue
                seen.add(app_id)
                enriched.append({"game": games_cache[app_id], "score": round(float(score), 4)})
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


@router.post("/search/image", response_model=SearchResponseOut)
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(20, ge=1, le=100),
):
    """Search games by uploading an image (uses CLIP embeddings)."""
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


@router.get("/games/{game_id}", response_model=GameOut)
async def get_game(game_id: int):
    """Get detailed information about a specific game."""
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


@router.get("/games/{game_id}/similar", response_model=SearchResponseOut)
async def similar_games(
    game_id: int,
    limit: int = Query(8, ge=1, le=50),
    rerank: bool = Query(False, description="Apply cross-encoder reranking"),
):
    """Find similar games using vector similarity search."""
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

            # Batch fetch all candidate games to avoid N+1 lookups
            candidate_ids = [candidate_app_id for candidate_app_id, _ in results]
            games_cache = {k: v for k, v in game_store.get_games_batch(candidate_ids).items() if v is not None}

            # Prepare candidates with optimized text formatting
            candidates = []
            for candidate_app_id, _ in results:
                game = games_cache.get(candidate_app_id)
                if game is None:
                    continue
                # Use reranker's smart text preparation
                combined = reranker.prepare_text(
                    game.get("name", ""),
                    game.get("short_description", ""),
                    game.get("description", "")
                )
                candidates.append((candidate_app_id, combined))

            reranked = await loop.run_in_executor(
                None, reranker.rerank, query_desc, candidates, limit
            )

            # Reuse cached games for enrichment
            seen = set()
            enriched = []
            for candidate_app_id, score in reranked:
                if candidate_app_id in seen or candidate_app_id not in games_cache:
                    continue
                seen.add(candidate_app_id)
                enriched.append({"game": games_cache[candidate_app_id], "score": round(float(score), 4)})
        else:
            enriched = _enrich(results)

        return SearchResponseOut(
            results=enriched, query=f"similar to {game_id}", total=len(enriched)
        )
    except Exception:
        logger.exception("Similar games failed")
        raise HTTPException(status_code=500, detail="Similar games lookup failed")
