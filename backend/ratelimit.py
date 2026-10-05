"""In-memory limits on LLM use: a sliding window per client IP, and one global daily budget."""
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


class SlidingWindowLimiter:
    """At most `limit` hits per `window_s` seconds for each key."""

    def __init__(self, limit: int, window_s: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window_s = window_s
        self.clock = clock
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str):
        """Record a hit. Returns (allowed, retry_after_s); a refused hit isn't recorded."""
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window_s - now + 0.999))
            hits.append(now)
            # Drop idle keys so the table can't grow without bound.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class DailyBudget:
    """A global count of LLM calls that resets at midnight UTC."""

    def __init__(self, limit: int, today=_utc_today):
        self.limit = limit
        self.today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        with self._lock:
            day = self.today()
            if day != self._day:
                self._day, self._used = day, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            if self.today() != self._day:
                return self.limit
            return max(0, self.limit - self._used)
