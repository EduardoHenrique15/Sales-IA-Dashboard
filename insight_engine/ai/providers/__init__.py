"""
Provedores de LLM disponíveis e escolha do provedor a partir da chave.
"""

from __future__ import annotations

from insight_engine.ai.providers import gemini
from insight_engine.ai.providers.base import LLMProvider


def create_provider(api_key: str | None) -> tuple[LLMProvider | None, str | None]:
    """Devolve (provedor, motivo). Sem provedor, `motivo` explica por quê (ou é None se não há chave)."""
    if not api_key:
        return None, None
    if not gemini.GENAI_AVAILABLE:
        return None, "o pacote `google-genai` não está instalado."
    return gemini.GeminiProvider(api_key), None
