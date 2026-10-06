"""
Identidade visual: CSS dos componentes, paleta e estilo dos gráficos.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from insight_engine.formatting import PLOTLY_SEPARATORS

# Paleta das análises (versão para fundo escuro da paleta categórica de referência,
# validada para daltonismo: até 3 cores juntas passam em todas as checagens).
SERIES_1 = "#3987e5"  # azul
SERIES_2 = "#d95926"  # laranja
SERIES_3 = "#199e70"  # verde-água
# Polaridade (aumento x redução) e neutro
POSITIVE = "#3987e5"
NEGATIVE = "#e66767"
NEUTRAL = "#6b6b66"
MUTED_LINE = "#8a93b0"

CUSTOM_CSS = """
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1c2333 0%, #161b28 100%);
        border: 1px solid #2a3352;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #9aa4c4;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        /* diminui em telas estreitas para o valor caber em uma linha */
        font-size: clamp(0.95rem, 1.1vw, 1.65rem);
        font-weight: 700;
        color: #f5f7ff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-delta-up { color: #34d399; font-size: 0.85rem; font-weight: 600; }
    .kpi-delta-down { color: #f87171; font-size: 0.85rem; font-weight: 600; }
    .kpi-delta-neutral { color: #9aa4c4; font-size: 0.85rem; font-weight: 600; }

    .report-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .badge-gemini { background: #1e3a2f; color: #34d399; border: 1px solid #2f6f4e; }
    .badge-fallback { background: #2a2440; color: #c4b5fd; border: 1px solid #4c3f7a; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_chart_theme(fig: go.Figure, title: str, height: int, **layout) -> go.Figure:
    """Aplica o estilo padrão do dashboard a um gráfico Plotly (`layout` sobrescreve o padrão)."""
    defaults = dict(
        template="plotly_dark",
        separators=PLOTLY_SEPARATORS,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_layout(title=title, height=height, **{**defaults, **layout})
    return fig
