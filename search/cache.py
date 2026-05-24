"""Search result caching with TTL."""
import time
from collections import OrderedDict
from typing import Optional


class SearchCache:
    """LRU cache with TTL for search results."""

    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def get(self, key: str):
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - self._timestamps[key] > self.ttl:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return entry

    def set(self, key: str, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._cache.move_to_end(key)
        while len(self._cache) > self.maxsize:
            oldest = next(iter(self._cache))
            self._cache.pop(oldest)
            self._timestamps.pop(oldest, None)

    def invalidate(self, prefix: Optional[str] = None):
        if prefix:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
        else:
            self._cache.clear()
            self._timestamps.clear()
