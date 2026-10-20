import os
import json
import hashlib
import time
from typing import Optional, Dict, Any

try:
    import redis
except ImportError:
    redis = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class SupportResponseCache:
    """
    Enterprise Cache-Aside Layer for Support Queries.
    Primary: Redis (distributed, TTL-managed).
    Fallback: In-Memory TTL store if Redis daemon is offline.
    Guarantees sub-10ms response times for repeated queries without burning LLM tokens.
    """
    def __init__(self, url: str = REDIS_URL, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.redis_client = None
        self.is_redis_connected = False
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        if redis is not None:
            try:
                # Fast timeout (0.5s) prevents blocking startup if local Redis server is absent
                client = redis.Redis.from_url(url, socket_timeout=0.5, decode_responses=True)
                client.ping()
                self.redis_client = client
                self.is_redis_connected = True
            except Exception:
                self.is_redis_connected = False
                self.redis_client = None

    def _hash_query(self, query: str) -> str:
        """Normalizes and SHA256-hashes query string for consistent indexing."""
        normalized = " ".join(query.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached support response if available and unexpired."""
        key = f"support:cache:{self._hash_query(query)}"

        # 1. Try Redis
        if self.is_redis_connected and self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    data = json.loads(val)
                    data["cache_source"] = "Redis"
                    return data
            except Exception:
                pass

        # 2. Try In-Memory Fallback
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() < entry["expires_at"]:
                data = dict(entry["data"])
                data["cache_source"] = "Memory"
                return data
            else:
                del self._memory_cache[key]

        return None

    def set(self, query: str, data: Dict[str, Any], ttl: Optional[int] = None):
        """Caches response data with TTL in seconds."""
        key = f"support:cache:{self._hash_query(query)}"
        expire_seconds = ttl or self.default_ttl

        # 1. Store in Redis
        if self.is_redis_connected and self.redis_client:
            try:
                self.redis_client.setex(key, expire_seconds, json.dumps(data))
            except Exception:
                pass

        # 2. Store in In-Memory Fallback
        self._memory_cache[key] = {
            "data": data,
            "expires_at": time.time() + expire_seconds
        }

    @property
    def backend_status(self) -> str:
        return "Redis (Online)" if self.is_redis_connected else "In-Memory Fallback (Redis Offline)"

# Global singleton
support_cache = SupportResponseCache()
