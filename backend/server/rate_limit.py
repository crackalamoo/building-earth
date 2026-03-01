"""Global sliding-window rate limiter for FastAPI routes."""

from __future__ import annotations

import os
import time
from collections import deque

from fastapi import HTTPException, Request


class RateLimiter:
    """Global (not per-IP) sliding-window rate limiter.

    Usage as a FastAPI dependency::

        limiter = RateLimiter(max_requests=120, window_seconds=60)

        @app.post("/api/chat", dependencies=[Depends(limiter)])
        async def chat(...): ...
    """

    def __init__(self, max_requests: int = 120, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    async def __call__(self, request: Request) -> None:  # noqa: ARG002
        now = time.monotonic()

        # Evict expired timestamps
        while self._timestamps and self._timestamps[0] <= now - self.window_seconds:
            self._timestamps.popleft()

        if len(self._timestamps) >= self.max_requests:
            retry_after = int(self._timestamps[0] - (now - self.window_seconds)) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(retry_after)},
            )

        self._timestamps.append(now)


def create_chat_limiter() -> RateLimiter:
    """Create a rate limiter configured from ``RATE_LIMIT_CHAT`` env var.

    Format: ``<max_requests>/<window_seconds>``, default ``120/60``.
    """
    raw = os.getenv("RATE_LIMIT_CHAT", "120/60")
    max_requests, window_seconds = (int(x) for x in raw.split("/"))
    return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
