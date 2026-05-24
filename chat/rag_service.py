"""RAG service for game recommendation context preparation."""
import asyncio
import logging
from typing import List, Dict, Optional

from search.es_retriever import ESRetriever
from search.game_store import GameStore
from search.reranker import Reranker

logger = logging.getLogger("steamrec")


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Integrates existing search infrastructure:
    1. ES hybrid search (text + CLIP vectors)
    2. Cross-encoder reranking
    3. Context formatting for LLM
    """

    def __init__(
        self,
        retriever: ESRetriever,
        game_store: GameStore,
        reranker: Reranker,
        top_k: int = 25
    ):
        """
        Args:
            retriever: Elasticsearch hybrid search retriever
            game_store: Game metadata store
            reranker: Cross-encoder reranker
            top_k: Number of top games to include in context
        """
        self.retriever = retriever
        self.game_store = game_store
        self.reranker = reranker
        self.top_k = top_k

    def _format_game_for_context(self, game: Dict) -> str:
        """
        Format a single game for LLM context.

        Includes: name, genres, price, rating, platforms, description
        Optimized for relevance and token efficiency.
        """
        # Extract key fields
        name = game.get("name", "Unknown")
        game_id = game.get("id", "")
        genres = ", ".join(game.get("genres", [])[:3])  # Top 3 genres only
        platforms = ", ".join(game.get("platforms", []))
        price = game.get("price", 0)

        # Rating calculation
        rating = game.get("steam_rating", 0)
        total_reviews = game.get("positive_reviews", 0) + game.get("negative_reviews", 0)

        # Short description (prioritize over detailed)
        description = game.get("short_description", "")[:250]  # Truncate at 250 chars

        # Build formatted string
        parts = [
            f"**{name}** (ID: {game_id})"
        ]

        if genres:
            parts.append(f"Genres: {genres}")

        if price is not None:
            price_str = "Free" if price == 0 else f"${price:.2f}"
            parts.append(f"Price: {price_str}")

        if total_reviews > 0:
            parts.append(f"Rating: {rating}% positive ({total_reviews:,} reviews)")

        if platforms:
            parts.append(f"Platforms: {platforms}")

        if description:
            parts.append(f"Description: {description}")

        return "\n".join(parts)

    def _merge_multimodal_results(
        self,
        text_results: List[tuple],
        image_results: List[tuple],
        limit: int
    ) -> List[tuple]:
        """Merge text and image results using weighted RRF."""
        from collections import defaultdict

        RRF_K = 60
        TEXT_WEIGHT = 1.5  # Prefer text intent
        IMAGE_WEIGHT = 1.0

        scores = defaultdict(float)

        for rank, (app_id, _) in enumerate(text_results, start=1):
            scores[app_id] += TEXT_WEIGHT / (RRF_K + rank)

        for rank, (app_id, _) in enumerate(image_results, start=1):
            scores[app_id] += IMAGE_WEIGHT / (RRF_K + rank)

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        logger.info(
            f"Merged {len(text_results)} text + {len(image_results)} image → {len(merged)} final"
        )

        return merged

    async def retrieve_context(
        self,
        query: str,
        filters: Optional[Dict] = None,
        rerank: bool = True,
        image = None
    ) -> tuple[List[Dict], str]:
        """
        Retrieve and format game context for RAG.

        Args:
            query: User search query
            filters: Optional filters (genre, platform, price_min, price_max)
            rerank: Whether to apply cross-encoder reranking
            image: Optional PIL Image for multimodal search

        Returns:
            tuple: (list of game dicts, formatted context string)
        """
        # Extract filters
        genre = filters.get("genre") if filters else None
        platform = filters.get("platform") if filters else None
        price_min = filters.get("price_min") if filters else None
        price_max = filters.get("price_max") if filters else None

        # Retrieve candidates (2x top_k if reranking)
        retrieve_k = self.top_k * 2 if rerank else self.top_k

        # Run retrieval in executor (blocking operation)
        loop = asyncio.get_event_loop()

        # Detect meta-queries about the image itself
        meta_query_keywords = ["what is this", "identify this", "what game is this", "which game", "name this game", "recognize this"]
        is_meta_query = image is not None and query.strip() and any(keyword in query.lower() for keyword in meta_query_keywords)

        # Branch based on input modality
        if image is not None and query.strip() and not is_meta_query:
            # ── HYBRID: Text + Image ──
            logger.info("Multimodal search: text + image")

            text_candidates = await loop.run_in_executor(
                None,
                lambda: self.retriever.search(
                    query, k=retrieve_k,
                    genre=genre, platform=platform,
                    price_min=price_min, price_max=price_max
                )
            )

            image_candidates = await loop.run_in_executor(
                None,
                lambda: self.retriever.search_image(image, k=retrieve_k)
            )

            candidates = self._merge_multimodal_results(
                text_candidates, image_candidates, retrieve_k
            )

        elif image is not None:
            # ── IMAGE-ONLY (or meta-query about the image) ──
            logger.info("Image-only search" + (" (meta-query detected)" if is_meta_query else ""))
            candidates = await loop.run_in_executor(
                None,
                lambda: self.retriever.search_image(image, k=retrieve_k)
            )

        else:
            # ── TEXT-ONLY (default/backward compatible) ──
            logger.info("Text-only search")
            candidates = await loop.run_in_executor(
                None,
                lambda: self.retriever.search(
                    query, k=retrieve_k,
                    genre=genre, platform=platform,
                    price_min=price_min, price_max=price_max
                )
            )

        if not candidates:
            return [], "No games found matching the criteria."

        # Rerank if enabled
        if rerank and len(candidates) > 1:
            app_ids = [app_id for app_id, _ in candidates]
            games_cache = self.game_store.get_games_batch(app_ids)

            # Prepare candidates for reranking
            rerank_candidates = []
            for app_id, _ in candidates:
                game = games_cache.get(app_id)
                if game:
                    text = self.reranker.prepare_text(
                        game.get("name", ""),
                        game.get("short_description", ""),
                        game.get("description", "")
                    )
                    rerank_candidates.append((app_id, text))

            # Rerank and get top-k
            reranked = await loop.run_in_executor(
                None,
                lambda: self.reranker.rerank(query, rerank_candidates, self.top_k)
            )
            games = [games_cache[app_id] for app_id, _ in reranked if app_id in games_cache]
        else:
            # Just enrich without reranking
            app_ids = [app_id for app_id, _ in candidates[:self.top_k]]
            games = [
                self.game_store.get_game_by_app_id(app_id)
                for app_id in app_ids
            ]
            games = [g for g in games if g is not None]

        # Format context for LLM
        context_parts = []

        # Add image search indicator
        if image is not None:
            context_parts.append("# Image Search Results\n")
            if is_meta_query or not query.strip():
                context_parts.append("The user uploaded a game screenshot/image. The following games are ranked by visual similarity. **The top result (Game 1) is most likely the game shown in the image.** Identify Game 1 as the game from the image and explain what you know about it.\n\n")
            else:
                context_parts.append("The user uploaded an image along with their query. The following games were retrieved based on visual similarity to the uploaded image AND text relevance.\n\n")

        context_parts.extend([
            "# Available Games (Retrieved from Database)\n",
            "Use these games to provide personalized recommendations.\n",
            "Always cite games using the format: **[Game Name](game_id)**\n\n"
        ])

        for i, game in enumerate(games, 1):
            context_parts.append(f"## Game {i}\n")
            context_parts.append(self._format_game_for_context(game))
            context_parts.append("\n\n")

        formatted_context = "".join(context_parts)

        logger.info(
            f"RAG context prepared: {len(games)} games, "
            f"{len(formatted_context)} chars (~{len(formatted_context)//4} tokens)"
        )

        return games, formatted_context
