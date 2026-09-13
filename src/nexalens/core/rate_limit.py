"""
Rate limiting middleware and utilities.

Implements sliding window rate limiting per user/IP using Redis.
"""

import time
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from nexalens.api.auth import get_current_user
from nexalens.core.config import get_settings
from nexalens.core.logging import get_logger
from nexalens.models.schemas import User

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests: int
    window_seconds: int
    key_prefix: str = "ratelimit"


# Default rate limit configs
DEFAULT_LIMITS = {
    "api": RateLimitConfig(requests=100, window_seconds=60),           # 100 req/min
    "query": RateLimitConfig(requests=30, window_seconds=60),          # 30 queries/min
    "auth": RateLimitConfig(requests=10, window_seconds=300),          # 10 login attempts/5min
    "forecast": RateLimitConfig(requests=20, window_seconds=60),       # 20 forecasts/min
    "financial": RateLimitConfig(requests=20, window_seconds=60),      # 20 financial models/min
    "report_schedule": RateLimitConfig(requests=10, window_seconds=60), # 10 schedule creates/min
}


class RateLimiter:
    """Sliding window rate limiter using Redis sorted sets."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[redis.Redis] = None
        self._limits: dict[str, RateLimitConfig] = DEFAULT_LIMITS.copy()

    async def _get_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        return self._redis

    def add_limit(self, name: str, config: RateLimitConfig) -> None:
        """Add or update a rate limit configuration."""
        self._limits[name] = config

    async def check_rate_limit(
        self,
        identifier: str,
        limit_name: str = "api",
    ) -> tuple[bool, int, int]:
        """
        Check if the identifier is within rate limits.
        
        Returns:
            (allowed, remaining, reset_time_seconds)
        """
        config = self._limits.get(limit_name)
        if not config:
            return True, 0, 0

        r = await self._get_redis()
        key = f"{config.key_prefix}:{limit_name}:{identifier}"
        now = time.time()
        window_start = now - config.window_seconds

        # Remove expired entries
        await r.zremrangebyscore(key, 0, window_start)

        # Count current requests
        current_count = await r.zcard(key)

        if current_count >= config.requests:
            # Get oldest entry to calculate reset time
            oldest = await r.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_time = int(oldest[0][1] + config.window_seconds - time.time())
            else:
                reset_time = config.window_seconds
            return False, 0, max(reset_time, 1)

        # Add current request
        await r.zadd(key, {str(time.time()): time.time()})
        await r.expire(key, config.window_seconds + 1)

        remaining = config.requests - current_count - 1
        return True, remaining, config.window_seconds

    async def get_current_usage(self, identifier: str, limit_name: str = "api") -> int:
        """Get current request count for identifier."""
        config = self._limits.get(limit_name)
        if not config:
            return 0

        r = await self._get_redis()
        key = f"{config.key_prefix}:{limit_name}:{identifier}"
        now = time.time()
        window_start = now - self._limits[limit_name].window_seconds

        await r.zremrangebyscore(key, 0, window_start)
        return await r.zcard(key)

    async def reset_limit(self, identifier: str, limit_name: str = "api") -> None:
        """Reset rate limit for an identifier (admin use)."""
        config = self._limits.get(limit_name)
        if not config:
            return

        r = await self._get_redis()
        key = f"{config.key_prefix}:{limit_name}:{identifier}"
        await r.delete(key)

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


# Global rate limiter instance
rate_limiter = RateLimiter()


async def get_client_identifier(request: Request, current_user: User = None) -> str:
    """Get identifier for rate limiting (user ID or IP)."""
    if current_user:
        return f"user:{current_user.id}"
    
    # Get client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


class RateLimitDependency:
    """FastAPI dependency for rate limiting."""

    def __init__(self, limit_name: str = "api"):
        self.limit_name = limit_name

    async def __call__(
        self,
        request: Request,
        current_user: User = Depends(get_current_user),
    ) -> None:
        identifier = await get_client_identifier(request, current_user)
        allowed, remaining, reset_time = await rate_limiter.check_rate_limit(
            identifier, self.limit_name
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": self.limit_name,
                    "retry_after": reset_time,
                },
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + reset_time),
                    "Retry-After": str(reset_time),
                },
            )

        # Add rate limit headers to response
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = int(time.time()) + rate_limiter._limits[self.limit_name].window_seconds


# Pre-configured rate limit dependencies
rate_limit_api = RateLimitDependency("api")
rate_limit_query = RateLimitDependency("query")
rate_limit_auth = RateLimitDependency("auth")
rate_limit_forecast = RateLimitDependency("forecast")
rate_limit_financial = RateLimitDependency("financial")
rate_limit_report_schedule = RateLimitDependency("report_schedule")


async def add_rate_limit_headers(request: Request, response) -> None:
    """Middleware to add rate limit headers to responses."""
    remaining = getattr(request.state, "rate_limit_remaining", None)
    reset = getattr(request.state, "rate_limit_reset", None)
    
    if remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    if reset is not None:
        response.headers["X-RateLimit-Reset"] = str(reset)