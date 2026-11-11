import math
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """
    In-memory sliding-window limiter (single process, like the BM25 index).
    `check` returns 0 when the call is allowed, otherwise the number of
    seconds until the next call will be accepted.
    """

    def __init__(self, window_seconds: float = 60.0):
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> int:
        if limit <= 0:
            return 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                return max(1, math.ceil(self.window - (now - hits[0])))
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop idle keys
                for stale in [k for k, v in self._hits.items() if not v]:
                    del self._hits[stale]
            return 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


ask_limiter = RateLimiter()
login_limiter = RateLimiter()
