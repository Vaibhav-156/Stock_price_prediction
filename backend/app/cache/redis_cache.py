"""
Redis caching layer for computed features and model predictions.
Falls back to a no-op in-memory store when Redis is unavailable.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

try:
    import redis.asyncio as aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class _InMemoryCache:
    """Fallback in-memory TTL cache used when Redis is not available."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[bytes, float]] = {}  # key -> (value, expires_at)

    def get(self, key: str) -> Optional[bytes]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else 0
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisCache:
    """Async Redis cache with typed helpers for DataFrames and dicts.

    Gracefully degrades to an in-memory TTL cache when Redis is unavailable,
    so all API endpoints continue to work without a running Redis instance.
    """

    def __init__(self) -> None:
        self._pool: Optional[Any] = None
        self._fallback = _InMemoryCache()
        self._using_fallback = False

    async def connect(self) -> None:
        if not _REDIS_AVAILABLE:
            logger.warning("redis package not available — using in-memory cache fallback")
            self._using_fallback = True
            return
        if self._pool is None:
            try:
                pool = aioredis.from_url(
                    settings.redis_url,
                    decode_responses=False,
                    max_connections=20,
                    socket_connect_timeout=2,
                )
                # Verify the connection is actually reachable
                await pool.ping()
                self._pool = pool
                self._using_fallback = False
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(
                    f"Redis not reachable ({e}) — using in-memory cache fallback. "
                    "All API endpoints will still work; caching is in-process only."
                )
                self._using_fallback = True

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Generic ops ───────────────────────────────────────

    async def get(self, key: str) -> Optional[bytes]:
        if self._using_fallback or self._pool is None:
            return self._fallback.get(key)
        try:
            return await self._pool.get(key)
        except Exception:
            return self._fallback.get(key)

    async def set(self, key: str, value: bytes, ttl_seconds: int = 3600) -> None:
        if self._using_fallback or self._pool is None:
            self._fallback.set(key, value, ttl_seconds)
            return
        try:
            await self._pool.set(key, value, ex=ttl_seconds)
        except Exception:
            self._fallback.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        if self._using_fallback or self._pool is None:
            self._fallback.delete(key)
            return
        try:
            await self._pool.delete(key)
        except Exception:
            self._fallback.delete(key)

    async def exists(self, key: str) -> bool:
        if self._using_fallback or self._pool is None:
            return self._fallback.exists(key)
        try:
            return bool(await self._pool.exists(key))
        except Exception:
            return self._fallback.exists(key)

    # ── DataFrame cache ───────────────────────────────────

    async def cache_dataframe(
        self, key: str, df: pd.DataFrame, ttl_seconds: int = 3600
    ) -> None:
        payload = df.to_json(orient="split", date_format="iso")
        await self.set(key, payload.encode(), ttl_seconds)

    async def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        raw = await self.get(key)
        if raw is None:
            return None
        return pd.read_json(raw.decode(), orient="split")

    # ── JSON cache ────────────────────────────────────────

    async def cache_json(self, key: str, data: Any, ttl_seconds: int = 3600) -> None:

        class _Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        payload = json.dumps(data, cls=_Encoder)
        await self.set(key, payload.encode(), ttl_seconds)

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode())

    # ── Key helpers ───────────────────────────────────────

    @staticmethod
    def feature_key(symbol: str, as_of: str) -> str:
        return f"features:{symbol}:{as_of}"

    @staticmethod
    def signal_key(symbol: str, as_of: str) -> str:
        return f"signal:{symbol}:{as_of}"

    @staticmethod
    def price_key(symbol: str, start: str, end: str) -> str:
        return f"price:{symbol}:{start}:{end}"


# Singleton
cache = RedisCache()
