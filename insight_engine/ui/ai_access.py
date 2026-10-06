"""
Acesso à IA na interface: qual chave usar, limites de uso e cache.

  - Chave digitada na barra lateral: uso livre (a cota é de quem digitou).
  - Chave do servidor (secrets/.env): limitada por sessão e globalmente,
    para que visitantes do app publicado não esgotem a cota do dono.
  - Relatórios do LLM ficam em cache compartilhado: os mesmos dados geram o
    mesmo relatório sem gastar cota de novo.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from insight_engine.ai.limits import (
    LIMIT_MESSAGE,
    BoundedCache,
    SlidingWindowLimiter,
    global_limiter,
    session_limiter,
)
from insight_engine.ai.providers import create_provider
from insight_engine.ai.providers.base import LLMProvider
from insight_engine.config import get_gemini_api_key

# Chave do `st.session_state` onde a barra lateral guarda a chave digitada pelo usuário
GEMINI_KEY_STATE = "gemini_api_key"
_SESSION_LIMITER_STATE = "_server_key_limiter"


@dataclass(frozen=True)
class AIAccess:
    provider: LLMProvider | None
    # motivo de não haver provedor apesar de haver chave (ex.: pacote ausente)
    provider_error: str | None
    uses_server_key: bool
    # None = sem limite (chave do próprio usuário)
    allow_call: Callable[[], bool] | None
    limit_message: str = LIMIT_MESSAGE


@st.cache_resource
def _global_limiter() -> SlidingWindowLimiter:
    return global_limiter()


@st.cache_resource
def report_cache() -> BoundedCache:
    return BoundedCache(max_items=200)


def ai_access() -> AIAccess:
    user_key = st.session_state.get(GEMINI_KEY_STATE) or None
    api_key = user_key or get_gemini_api_key()
    provider, error = create_provider(api_key)
    uses_server_key = provider is not None and user_key is None

    allow_call = None
    if uses_server_key:
        if _SESSION_LIMITER_STATE not in st.session_state:
            st.session_state[_SESSION_LIMITER_STATE] = session_limiter()
        session = st.session_state[_SESSION_LIMITER_STATE]
        shared = _global_limiter()

        def allow_call() -> bool:
            return session.try_acquire() and shared.try_acquire()

    return AIAccess(provider, error, uses_server_key, allow_call)
