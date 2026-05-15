# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from .retriever import SteamRetriever  # Adjust import path if needed

app = FastAPI(title="Steam RAG Backend")

# 1. Define Request Model (for POST body)


class SearchRequest(BaseModel):
    query: str
    k: int = 10

# 2. Define Response Model (MUST match SteamRetriever output)


class SearchResult(BaseModel):
    rank: int
    document: str      # Contains "Title: Description"
    rrf_score: float   # The fused RRF score


# Initialize Retriever (Load once at startup)
retriever = SteamRetriever()


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Hybrid BM25 + FAISS search with RRF fusion.
    Returns a list of ranked game documents.
    """
    try:
        # Call the async retriever
        results = await retriever.search(request.query, request.k)

        # FastAPI will automatically validate 'results' against 'List[SearchResult]'
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
