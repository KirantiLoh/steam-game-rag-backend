"""Chat endpoints for conversational game recommendations."""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.models import ChatRequest, ChatResponse
from core.config import settings

logger = logging.getLogger("steamrec")

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Global state (injected from main.py)
rag_service = None
llm_client = None
session_manager = None


# Extended ChatResponse with metadata
class ChatResponseExtended(BaseModel):
    """Chat response with metadata."""
    response: str
    session_id: str
    games_retrieved: int


@router.post("", response_model=None)
async def chat_assistant(
    query: str = Form(..., min_length=1, max_length=500),
    session_id: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    platform: Optional[str] = Form(None),
    price_min: Optional[float] = Form(None),
    price_max: Optional[float] = Form(None),
    stream: bool = Form(True),
    image: Optional[UploadFile] = File(None),
):
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
        session_id_value = session_id or str(uuid.uuid4())

        # Get conversation history
        history = session_manager.get_history(session_id_value)

        # Process optional image upload
        image_obj = None
        if image is not None:
            try:
                from PIL import Image
                from io import BytesIO
                contents = await image.read()
                image_obj = Image.open(BytesIO(contents)).convert("RGB")
                logger.info(f"Image uploaded: {image.filename}, size={image_obj.size}")
            except Exception as e:
                logger.warning(f"Image processing failed: {e}")
                # Continue without image rather than failing entire request
                image_obj = None

        # Build filters
        filters = {}
        if genre:
            filters["genre"] = genre
        if platform:
            filters["platform"] = platform
        if price_min is not None:
            filters["price_min"] = price_min
        if price_max is not None:
            filters["price_max"] = price_max

        # Retrieve RAG context (with optional image)
        games, context = await rag_service.retrieve_context(
            query=query,
            filters=filters if filters else None,
            rerank=settings.rag_rerank,
            image=image_obj
        )

        # Add user message to history with image indicator
        query_with_meta = query if image_obj is None else f"{query} [+image]"
        session_manager.add_message(session_id_value, "user", query_with_meta)

        if stream:
            # ── STREAMING RESPONSE (SSE) ──
            async def event_stream():
                """Generate Server-Sent Events stream."""
                accumulated_response = []

                # Send session ID
                yield f"event: session\ndata: {session_id_value}\n\n"

                # Send metadata
                yield f"event: metadata\ndata: {len(games)}\n\n"

                # Stream LLM tokens
                try:
                    async for token in llm_client.stream_chat(
                        query, context, history
                    ):
                        accumulated_response.append(token)
                        # Escape newlines in SSE data
                        token_escaped = token.replace('\n', '\\n')
                        yield f"event: token\ndata: {token_escaped}\n\n"

                    # Save complete response to history
                    full_response = "".join(accumulated_response)
                    session_manager.add_message(session_id_value, "assistant", full_response)

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
                query, context, history
            )

            # Save to history
            session_manager.add_message(session_id_value, "assistant", response_text)

            return ChatResponseExtended(
                response=response_text,
                session_id=session_id_value,
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


@router.delete("/session/{session_id}")
async def clear_chat_session(session_id: str):
    """
    Clear conversation history for a specific session.

    Use this to start a fresh conversation or reset context.
    """
    session_manager.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/sessions")
async def get_active_sessions():
    """Get count of currently active chat sessions."""
    count = session_manager.get_active_sessions_count()
    return {"active_sessions": count}


@router.get("/health")
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
