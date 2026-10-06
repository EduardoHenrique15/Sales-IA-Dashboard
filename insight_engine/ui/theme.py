"""
Identidade visual: paleta dos gráficos (modo claro e escuro), estilo
padrão das figuras Plotly e os poucos ajustes de CSS do app.

As cores da interface (fundo, texto, destaque, fonte) ficam no tema do
Streamlit, em `.streamlit/config.toml`. Aqui ficam só as cores que os
gráficos precisam escolher por conta própria.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st

from insight_engine.formatting import PLOTLY_SEPARATORS


@dataclass(frozen=True)
class ChartPalette:
    """Cores dos gráficos para um modo (claro ou escuro).

    As três séries vêm da paleta categórica de referência, validada para
    daltonismo (até três cores juntas passam em todas as checagens).
    """

    series_1: str  # azul
    series_2: str  # laranja
    series_3: str  # verde-água
    positive: str  # aumento
    negative: str  # redução
    neutral: str  # totais, linhas de referência
    muted_line: str  # série de contexto (ex.: base das anomalias)
    surface: str  # fundo da página (contorno dos marcadores)
    sequential: tuple[str, ...]  # rampa de um só tom (mapas de calor)


LIGHT = ChartPalette(
    series_1="#2a78d6",
    series_2="#eb6834",
    series_3="#1baf7a",
    positive="#2a78d6",
    negative="#e34948",
    neutral="#898781",
    muted_line="#a3a7b3",
    surface="#ffffff",
    sequential=("#eef4fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"),
)
DARK = ChartPalette(
    series_1="#3987e5",
    series_2="#d95926",
    series_3="#199e70",
    positive="#3987e5",
    negative="#e66767",
    neutral="#898781",
    muted_line="#6f7790",
    surface="#0e1117",
    sequential=("#161c28", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
)

# Barra de ferramentas dos gráficos: sem logo do Plotly e sem as ferramentas
# de seleção livre, que confundem quem só quer ler o gráfico.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

CUSTOM_CSS = """
<style>
    /* números alinhados em tabelas e cartões */
    [data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
    /* textos do envio de arquivo em português (o componente só tem textos em inglês) */
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzoneInstructions"] span { font-size: 0; }
    [data-testid="stFileUploaderDropzone"] button p::after { content: "Escolher arquivo"; font-size: 0.875rem; }
    [data-testid="stFileUploaderDropzoneInstructions"] span::after {
        content: "ou arraste aqui · até 20 MB · CSV ou XLSX"; font-size: 0.875rem;
    }
    /* títulos menores no celular */
    @media (max-width: 640px) {
        h1 { font-size: 1.9rem !important; }
        h2 { font-size: 1.4rem !important; }
    }
</style>
"""


def palette() -> ChartPalette:
    """Paleta do tema que o visitante está vendo (escuro quando não dá para saber)."""
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - fora de uma sessão do Streamlit
        kind = None
    return LIGHT if kind == "light" else DARK


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_chart_theme(fig: go.Figure, height: int, title: str | None = None, **layout) -> go.Figure:
    """Estilo padrão das figuras (`layout` sobrescreve o padrão).

    Não define `template`: assim o Streamlit aplica o próprio tema (claro ou
    escuro, com a fonte do app). Os títulos ficam fora do gráfico, no cartão
    que o envolve (ver `components.chart_card`).
    """
    defaults = dict(
        separators=PLOTLY_SEPARATORS,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=40 if title else 8, b=8),
        hoverlabel=dict(namelength=-1),
    )
    if title:
        defaults["title"] = title
    fig.update_layout(height=height, **{**defaults, **layout})
    return fig
