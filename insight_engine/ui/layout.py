"""
Áreas da barra lateral compartilhadas entre o `app.py` e as páginas.

O `app.py` cria as áreas em ordem fixa (filtros, depois configurações da IA
e rodapé) antes de rodar a página; as páginas escrevem os filtros na área
reservada, que assim fica sempre no topo. Cada sessão do Streamlit roda em
uma thread própria, então a área é guardada por thread.
"""

from __future__ import annotations

import threading

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_local = threading.local()


def reserve_sidebar_filters() -> None:
    """Reserva o topo da barra lateral para os filtros da página atual."""
    _local.filters = st.sidebar.container()


def sidebar_filters() -> DeltaGenerator:
    """Área de filtros na barra lateral (a barra inteira se nada foi reservado)."""
    return getattr(_local, "filters", None) or st.sidebar
