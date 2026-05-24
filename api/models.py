"""Pydantic models for API request/response schemas."""
from typing import List
from pydantic import BaseModel, Field


class GameOut(BaseModel):
    """Game metadata output model."""
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
    """Single search result with game and relevance score."""
    game: GameOut
    score: float


class SearchResponseOut(BaseModel):
    """Search response containing list of results."""
    results: List[SearchResultOut]
    query: str
    total: int


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., min_length=1, max_length=500, description="User query")
    session_id: str = Field(default="", description="Session ID for conversation continuity")
    stream: bool = Field(default=True, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Chat response model (non-streaming)."""
    response: str
    session_id: str
