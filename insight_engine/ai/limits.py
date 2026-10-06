"""
Limites de uso da chave do Gemini configurada no servidor.

Quem visita o app publicado usa a cota da chave do dono do app. Dois limites
protegem essa cota:
  - por sessão (navegador): N chamadas por hora;
  - global (todo o servidor): M chamadas por dia.
Quem informa a própria chave na barra lateral não tem limite.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from insight_engine.config import get_setting

DEFAULT_PER_SESSION_PER_HOUR = 20
DEFAULT_GLOBAL_PER_DAY = 300


@dataclass
class SlidingWindowLimiter:
    """Permite até `max_calls` chamadas a cada `window_seconds` (thread-safe)."""

    max_calls: int
    window_seconds: float
    _calls: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def try_acquire(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            while self._calls and now - self._calls[0] >= self.window_seconds:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


def session_limiter() -> SlidingWindowLimiter:
    limit = int(get_setting("SERVER_KEY_CALLS_PER_HOUR") or DEFAULT_PER_SESSION_PER_HOUR)
    return SlidingWindowLimiter(limit, 3600)


def global_limiter() -> SlidingWindowLimiter:
    limit = int(get_setting("SERVER_KEY_CALLS_PER_DAY") or DEFAULT_GLOBAL_PER_DAY)
    return SlidingWindowLimiter(limit, 86400)


LIMIT_MESSAGE = (
    "o limite de uso da chave do servidor foi atingido. Informe a sua própria chave do Gemini na "
    "barra lateral para continuar usando a IA generativa."
)


class BoundedCache(OrderedDict):
    """Dicionário com tamanho máximo: descarta o item usado há mais tempo (LRU)."""

    def __init__(self, max_items: int = 200) -> None:
        super().__init__()
        self.max_items = max_items
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            value = super().__getitem__(key)
            self.move_to_end(key)
            return value

    def __setitem__(self, key, value) -> None:
        with self._lock:
            super().__setitem__(key, value)
            self.move_to_end(key)
            while len(self) > self.max_items:
                self.popitem(last=False)
