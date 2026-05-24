"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}
