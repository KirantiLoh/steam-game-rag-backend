"""LLM client with streaming support using Google Gemini 1.5 Flash (FREE)."""
import logging
from typing import AsyncGenerator, List, Dict, Optional

from google import genai
from google.genai import types

from core.config import settings

logger = logging.getLogger("steamrec")


SYSTEM_PROMPT = """You are an expert Steam game recommendation assistant with access to a curated database of games.

**Your Role:**
- Provide personalized game recommendations based on user preferences
- Explain WHY each game matches the user's request (gameplay, mechanics, atmosphere, difficulty, etc.)
- Consider genres, price, ratings, and platform availability
- Be conversational, helpful, and enthusiastic about gaming

**Citation Format (IMPORTANT):**
- Always cite games using: **[Game Name](game_id)**
- Example: "I recommend **[Dark Souls III](374320)** for its challenging combat..."
- The game_id is provided in the context above

**Guidelines:**
- Recommend 3-5 games per response unless user asks for more
- For each game, explain what makes it a good fit (2-3 sentences)
- Mention important caveats: difficulty level, price point, mature content, platform exclusivity
- If user asks for clarification about a specific game, provide more details
- Be concise but informative - quality over quantity

**Style:**
- Conversational and natural (avoid bullet points unless requested)
- Enthusiastic but honest about game quality
- Consider user's skill level and preferences from conversation history

Respond as a knowledgeable friend sharing game recommendations."""


class GeminiClient:
    """
    Client for Google Gemini 1.5 Flash with streaming support.

    Features:
    - Free tier: 15 requests/minute, 1M requests/day
    - 1M token context window
    - Fast streaming responses
    """

    def __init__(self):
        """Initialize Gemini client with API key validation."""
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in environment. "
                "Get a free key from https://aistudio.google.com/app/apikey"
            )

        # Configure client
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_id = 'gemini-2.5-flash'

        # Generation configuration
        self.generation_config = types.GenerateContentConfig(
            temperature=settings.temperature,
            max_output_tokens=settings.max_tokens,
            top_p=0.95,
            top_k=40,
            system_instruction=SYSTEM_PROMPT
        )

        logger.info(
            f"Initialized Gemini 2.5 Flash (FREE tier) - "
            f"temp={settings.temperature}, max_tokens={settings.max_tokens}"
        )

    async def stream_chat(
        self,
        user_message: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion with game recommendation context.

        Args:
            user_message: Current user query
            context: RAG context with game information
            history: Previous conversation history

        Yields:
            str: Token chunks as they arrive from Gemini
        """
        # Build contents list
        contents = []

        # Add conversation history
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                ))

        # Add current message with RAG context
        message_with_context = f"""{context}

---

**User Query:** {user_message}"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=message_with_context)]
        ))

        logger.info(
            f"Gemini stream starting: "
            f"{len(history) if history else 0} history messages, "
            f"{len(message_with_context)} chars context"
        )

        try:
            # Stream response
            response = self.client.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=self.generation_config
            )

            # Yield tokens as they arrive
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_msg = str(e)
            logger.exception("Gemini streaming failed")

            # Provide user-friendly error messages
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                yield "\n\n[Rate limit exceeded. Please try again in a minute.]"
            elif "safety" in error_msg.lower():
                yield "\n\n[Response blocked by safety filters. Please rephrase your query.]"
            else:
                yield f"\n\n[Error: Unable to generate response. Please try again.]"

    async def get_full_response(
        self,
        user_message: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Non-streaming version for testing and non-streaming endpoints.

        Args:
            user_message: Current user query
            context: RAG context with game information
            history: Previous conversation history

        Returns:
            str: Complete response text
        """
        # Build contents list
        contents = []

        # Add conversation history
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                ))

        # Add current message
        message_with_context = f"{context}\n\n---\n\n**User Query:** {user_message}"

        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=message_with_context)]
        ))

        # Get response
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.generation_config
        )

        return response.text
