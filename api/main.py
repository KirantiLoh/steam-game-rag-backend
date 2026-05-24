"""Steam Game Recommender API — Feature-driven architecture."""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from search.es_retriever import ESRetriever
from search.game_store import GameStore
from search.reranker import Reranker
from search.cache import SearchCache
from chat.rag_service import RAGService
from chat.llm_client import GeminiClient
from chat.session_manager import SessionManager
from chat.rate_limiter import RateLimiter, RateLimitMiddleware
from core.config import settings

# Import route modules
from api.routes import health, search, chat

logger = logging.getLogger("steamrec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("Loading models…")
    loop = asyncio.get_running_loop()

    def _load():
        r = ESRetriever(default_k=50)
        g = GameStore()
        rn = Reranker()
        logger.info("ES doc count: %d", r.es.count(index="steam_games")["count"])
        return r, g, rn

    retriever, game_store, reranker = await loop.run_in_executor(None, _load)

    # Initialize search cache
    search_cache = SearchCache()

    # Initialize RAG and LLM services
    rag_service = RAGService(
        retriever=retriever,
        game_store=game_store,
        reranker=reranker,
        top_k=settings.rag_top_k
    )
    llm_client = GeminiClient()
    session_manager = SessionManager(ttl=settings.session_ttl_seconds)

    # Inject dependencies into route modules
    search.retriever = retriever
    search.game_store = game_store
    search.reranker = reranker
    search.search_cache = search_cache

    chat.rag_service = rag_service
    chat.llm_client = llm_client
    chat.session_manager = session_manager

    logger.info("All models loaded (including Gemini FREE tier)")
    yield


app = FastAPI(title="SteamRec API", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware for chat endpoints
rate_limiter = RateLimiter(
    max_requests=settings.max_requests_per_minute,
    window_seconds=60
)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

logger.info(f"Rate limiting enabled: {settings.max_requests_per_minute} requests/minute")

# Register routers
app.include_router(health.router)
app.include_router(search.router)
app.include_router(chat.router)
