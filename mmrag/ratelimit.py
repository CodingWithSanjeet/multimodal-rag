"""Tiny in-memory rate limits for the public demo (single instance, resets on restart)."""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone

MAX_QUESTION_CHARS = 500


class DailyCounter:
    """Global per-UTC-day counter shared by all sessions in one process."""

    def __init__(self, limit: int):
        self.limit = limit
        self._day = self._today()
        self._count = 0
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> date:
        return datetime.now(timezone.utc).date()

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._count = today, 0

    def try_consume(self) -> bool:
        with self._lock:
            self._roll()
            if self._count >= self.limit:
                return False
            self._count += 1
            return True

    @property
    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return max(0, self.limit - self._count)


def validate_question(question: str, max_chars: int = MAX_QUESTION_CHARS) -> str | None:
    """Return an error message, or None if the question is acceptable."""
    q = (question or "").strip()
    if not q:
        return "Please enter a question."
    if len(q) > max_chars:
        return f"Questions are limited to {max_chars} characters (yours has {len(q)})."
    return None
