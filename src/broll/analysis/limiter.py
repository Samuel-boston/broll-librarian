"""A client-side cap on how fast we call a provider.

Free Gemini keys allow a handful of requests a minute. Four workers firing at
once burn that in seconds and are answered with 429s and 503s, which look like
the model failing when they are really us knocking too hard. Spacing the calls
out costs nothing on a paid key (leave it at 0) and keeps a free key working.
"""

from __future__ import annotations

import asyncio


class RateLimiter:
    """Allow at most ``per_minute`` acquisitions, evenly spaced. 0 = no limit."""

    def __init__(self, per_minute: float = 0.0):
        self.per_minute = max(0.0, per_minute)
        self.interval = 60.0 / self.per_minute if self.per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> float:
        """Wait for this caller's slot. Returns how long it waited, in seconds."""
        if not self.interval:
            return 0.0
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            wait = max(0.0, self._next_at - now)
            # Reserve the slot inside the lock, sleep outside it, so callers
            # queue in order instead of all waking to the same instant.
            self._next_at = max(now, self._next_at) + self.interval
        if wait:
            await asyncio.sleep(wait)
        return wait
