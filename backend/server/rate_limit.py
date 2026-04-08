"""Per-IP sliding-window rate limiter for FastAPI routes."""

import os
import time
from collections import deque

from fastapi import HTTPException, Request


def _client_ip(request: Request) -> str:
    """Best-effort real client IP. Behind a reverse proxy (Railway, etc.)
    request.client.host is the proxy's IP, so all users would share one
    bucket. The first entry in X-Forwarded-For is the original client per
    convention; the proxy appends to the list rather than replacing it."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class RateLimiter:
    """Per-IP sliding-window rate limiter.

    Each IP gets its own request counter. Stale entries are cleaned up
    lazily to bound memory usage.

    Usage as a FastAPI dependency::

        limiter = RateLimiter(max_requests=20, window_seconds=60)

        @app.post("/api/chat", dependencies=[Depends(limiter)])
        async def chat(...): ...
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._last_cleanup = time.monotonic()
        self._cleanup_interval = 300.0  # purge stale IPs every 5 min

    async def __call__(self, request: Request) -> None:
        ip = _client_ip(request)
        now = time.monotonic()

        # Lazy cleanup of stale IP buckets
        if now - self._last_cleanup > self._cleanup_interval:
            cutoff = now - self.window_seconds
            stale = [k for k, v in self._buckets.items() if not v or v[-1] < cutoff]
            for k in stale:
                del self._buckets[k]
            self._last_cleanup = now

        timestamps = self._buckets.get(ip)
        if timestamps is None:
            timestamps = deque()
            self._buckets[ip] = timestamps

        # Evict expired timestamps for this IP
        while timestamps and timestamps[0] <= now - self.window_seconds:
            timestamps.popleft()

        if len(timestamps) >= self.max_requests:
            retry_after = int(timestamps[0] - (now - self.window_seconds)) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)


def create_chat_limiter() -> RateLimiter:
    """Create a rate limiter configured from ``RATE_LIMIT_CHAT`` env var.

    Format: ``<max_requests>/<window_seconds>``, default ``120/60``.
    """
    raw = os.getenv("RATE_LIMIT_CHAT", "20/60")
    max_requests, window_seconds = (int(x) for x in raw.split("/"))
    return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
