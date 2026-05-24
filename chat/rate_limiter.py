"""Rate limiter for Gemini free tier (15 RPM)."""
import time
import logging
from collections import defaultdict, deque
from typing import Optional, Tuple

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("steamrec")


class RateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests per identifier (IP address) within a time window.
    Suitable for Gemini free tier: 15 requests/minute.
    """

    def __init__(self, max_requests: int = 15, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: defaultdict[str, deque] = defaultdict(deque)

    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limit.

        Args:
            identifier: Unique identifier (IP address or session ID)

        Returns:
            tuple: (is_allowed: bool, retry_after_seconds: int or None)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove requests outside the current window
        requests = self._requests[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()

        # Check if limit exceeded
        if len(requests) >= self.max_requests:
            # Calculate retry-after time
            oldest_request = requests[0]
            retry_after = int(oldest_request + self.window_seconds - now) + 1
            logger.warning(
                f"Rate limit exceeded for {identifier}: "
                f"{len(requests)}/{self.max_requests} requests in window"
            )
            return False, retry_after

        # Allow request and record timestamp
        requests.append(now)
        return True, None

    def reset(self, identifier: str) -> None:
        """Reset rate limit for an identifier."""
        self._requests.pop(identifier, None)

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests allowed in current window."""
        now = time.time()
        window_start = now - self.window_seconds

        requests = self._requests[identifier]
        # Clean old requests
        while requests and requests[0] < window_start:
            requests.popleft()

        return max(0, self.max_requests - len(requests))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting chat endpoints.

    Only applies to /api/chat* endpoints to protect Gemini free tier limits.
    """

    def __init__(
        self,
        app,
        rate_limiter: RateLimiter,
        path_prefix: str = "/api/chat"
    ):
        """
        Args:
            app: FastAPI application
            rate_limiter: RateLimiter instance
            path_prefix: URL prefix to apply rate limiting
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.path_prefix = path_prefix

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""

        # Only apply to chat endpoints
        if not request.url.path.startswith(self.path_prefix):
            return await call_next(request)

        # Skip rate limiting for health check
        if request.url.path.endswith("/health"):
            return await call_next(request)

        # Use IP address as identifier
        # In production, consider using authenticated user ID
        identifier = request.client.host

        # Check rate limit
        allowed, retry_after = self.rate_limiter.is_allowed(identifier)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.rate_limiter.max_requests} "
                       f"requests per {self.rate_limiter.window_seconds} seconds. "
                       f"Please try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )

        # Log remaining quota
        remaining = self.rate_limiter.get_remaining(identifier)
        logger.debug(f"Rate limit: {remaining}/{self.rate_limiter.max_requests} remaining for {identifier}")

        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)

        return response
