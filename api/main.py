"""Steam Game Recommender — ES-based hybrid search + cross-encoder reranker."""
import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image

from es_retriever import ESRetriever
from game_store import GameStore
from reranker import Reranker
from rag_service import RAGService
from llm_client import GeminiClient
from session_manager import SessionManager
from rate_limiter import RateLimiter, RateLimitMiddleware
from llm_config import settings

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
retriever: Optional[ESRetriever] = None
game_store: Optional[GameStore] = None
reranker: Optional[Reranker] = None
rag_service: Optional[RAGService] = None
llm_client: Optional[GeminiClient] = None
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, game_store, reranker, rag_service, llm_client, session_manager
    logger.info("Loading models…")
    loop = asyncio.get_running_loop()

    def _load():
        r = ESRetriever(default_k=50)
        g = GameStore()
        rn = Reranker()
        logger.info("ES doc count: %d", r.es.count(index="steam_games")["count"])
        return r, g, rn

    retriever, game_store, reranker = await loop.run_in_executor(None, _load)

    # Initialize RAG and LLM services
    rag_service = RAGService(
        retriever=retriever,
        game_store=game_store,
        reranker=reranker,
        top_k=settings.rag_top_k
    )
    llm_client = GeminiClient()
    session_manager = SessionManager(ttl=settings.session_ttl_seconds)

    logger.info("All models loaded (including Gemini FREE tier)")
    yield


app = FastAPI(title="SteamRec API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware for chat endpoints
rate_limiter = RateLimiter(
    max_requests=settings.max_requests_per_minute,
    window_seconds=60
)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

logger.info(f"Rate limiting enabled: {settings.max_requests_per_minute} requests/minute")


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


# ── Chat Assistant (RAG + Gemini) ──────────────────────────────────


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    genre: Optional[str] = Field(None, description="Filter by genre")
    platform: Optional[str] = Field(None, description="Filter by platform (Windows, Mac, Linux)")
    price_min: Optional[float] = Field(None, ge=0, description="Minimum price")
    price_max: Optional[float] = Field(None, ge=0, description="Maximum price")
    stream: bool = Field(default=True, description="Enable streaming response (SSE)")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    games_retrieved: int


@app.post("/api/chat", response_model=None)
async def chat_assistant(request: ChatRequest):
    """
    Conversational game recommendation assistant with RAG.

    **Streaming Mode (default):**
    Returns Server-Sent Events (SSE) with real-time token streaming.

    **Non-Streaming Mode:**
    Returns complete response as JSON.

    **Features:**
    - Multi-turn conversations with session management
    - RAG-powered recommendations using hybrid search
    - Smart game citations with format: [Game Name](game_id)
    - Filter support: genre, platform, price range

    **Free Tier Limits (Gemini 1.5 Flash):**
    - 15 requests per minute
    - 1 million requests per day
    """
    # Ensure services are initialized
    if not rag_service or not llm_client or not session_manager:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        # Generate or validate session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Get conversation history
        history = session_manager.get_history(session_id)

        # Build filters
        filters = {}
        if request.genre:
            filters["genre"] = request.genre
        if request.platform:
            filters["platform"] = request.platform
        if request.price_min is not None:
            filters["price_min"] = request.price_min
        if request.price_max is not None:
            filters["price_max"] = request.price_max

        # Retrieve RAG context
        games, context = await rag_service.retrieve_context(
            query=request.query,
            filters=filters if filters else None,
            rerank=settings.rag_rerank
        )

        # Add user message to history
        session_manager.add_message(session_id, "user", request.query)

        if request.stream:
            # ── STREAMING RESPONSE (SSE) ──
            async def event_stream():
                """Generate Server-Sent Events stream."""
                accumulated_response = []

                # Send session ID
                yield f"event: session\ndata: {session_id}\n\n"

                # Send metadata
                yield f"event: metadata\ndata: {len(games)}\n\n"

                # Stream LLM tokens
                try:
                    async for token in llm_client.stream_chat(
                        request.query, context, history
                    ):
                        accumulated_response.append(token)
                        # Escape newlines in SSE data
                        token_escaped = token.replace('\n', '\\n')
                        yield f"event: token\ndata: {token_escaped}\n\n"

                    # Save complete response to history
                    full_response = "".join(accumulated_response)
                    session_manager.add_message(session_id, "assistant", full_response)

                    # Send completion event
                    yield f"event: done\ndata: success\n\n"

                except Exception as e:
                    logger.exception("Streaming error")
                    yield f"event: error\ndata: {str(e)}\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                    "Access-Control-Allow-Origin": "*"  # CORS for SSE
                }
            )

        else:
            # ── NON-STREAMING RESPONSE ──
            response_text = await llm_client.get_full_response(
                request.query, context, history
            )

            # Save to history
            session_manager.add_message(session_id, "assistant", response_text)

            return ChatResponse(
                response=response_text,
                session_id=session_id,
                games_retrieved=len(games)
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Chat assistant failed")
        raise HTTPException(
            status_code=500,
            detail="Chat service unavailable. Please try again."
        )


@app.delete("/api/chat/session/{session_id}")
async def clear_chat_session(session_id: str):
    """
    Clear conversation history for a specific session.

    Use this to start a fresh conversation or reset context.
    """
    session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/api/chat/sessions")
async def get_active_sessions():
    """Get count of currently active chat sessions."""
    count = session_manager.get_active_sessions_count()
    return {"active_sessions": count}


@app.get("/api/chat/health")
async def chat_health_check():
    """
    Health check for chat service.

    Returns status and configuration info.
    """
    try:
        assert llm_client is not None
        assert rag_service is not None
        assert session_manager is not None

        return {
            "status": "ok",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "free_tier": True,
            "limits": {
                "requests_per_minute": 15,
                "requests_per_day": 1000000
            },
            "rag_config": {
                "top_k": settings.rag_top_k,
                "rerank": settings.rag_rerank
            }
        }
    except Exception as e:
        logger.exception("Chat health check failed")
        raise HTTPException(
            status_code=503,
            detail="Chat service unavailable"
        )
