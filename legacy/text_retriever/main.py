"""Standalone text search endpoint (legacy). Use api/main.py for full app."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .retriever import SteamRetriever

app = FastAPI(title="Steam Text Search")


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class SearchResult(BaseModel):
    rank: int
    app_id: str
    document: str
    rrf_score: float


retriever = SteamRetriever()


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    try:
        return await retriever.search(request.query, request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
