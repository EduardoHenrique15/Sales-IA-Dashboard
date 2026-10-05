"""
config.py
================================================================
Leitura centralizada de configurações e segredos.

Ordem de prioridade para cada chave:
  1. `st.secrets` (Streamlit Community Cloud / .streamlit/secrets.toml)
  2. Variáveis de ambiente (inclusive as carregadas do arquivo `.env`)
  3. Valor padrão
================================================================
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Apelido que sempre aponta para o modelo Flash mais recente, para o app
# não quebrar quando o Google descontinuar uma versão específica.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"
# Modelo reserva, mais leve, usado quando o principal está sobrecarregado,
# sem cota ou indisponível.
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-flash-lite-latest"


def get_setting(name: str, default: str | None = None) -> str | None:
    """Busca uma configuração em `st.secrets` e, depois, no ambiente."""
    return _from_streamlit_secrets(name) or os.environ.get(name) or default


def _from_streamlit_secrets(name: str) -> str | None:
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:  # noqa: BLE001 - sem secrets.toml ou fora do Streamlit
        pass
    return None


def get_gemini_api_key() -> str | None:
    return get_setting("GEMINI_API_KEY")


def get_gemini_model() -> str:
    return get_setting("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def get_gemini_fallback_model() -> str:
    return get_setting("GEMINI_FALLBACK_MODEL") or DEFAULT_GEMINI_FALLBACK_MODEL


def get_coingecko_api_key() -> str | None:
    return get_setting("COINGECKO_API_KEY")
