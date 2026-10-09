"""
Filtros da barra lateral, sincronizados com a URL.

Os filtros escolhidos viram parâmetros do endereço da página (ex.:
`?de=2024-07-01&ate=2024-09-30&categorias=Moda`), então um link
compartilhado abre exatamente no mesmo recorte. Valores da URL são
validados contra a base (datas dentro do período, categorias existentes)
antes de chegar aos widgets, para que um link antigo ou editado à mão
nunca quebre a página.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from insight_engine.analytics.periods import COMPARISON_LABELS, SalesFilters
from insight_engine.formatting import format_brl
from insight_engine.ui.layout import sidebar_filters

DEFAULT_PERIOD_DAYS = 90
P_START, P_END, P_CATEGORIES, P_REGIONS, P_COMPARE, P_GOAL = "de", "ate", "categorias", "regioes", "comparar", "meta"
CUSTOM = "custom"
# atalho -> (rótulo, dias até a última data da base; None = histórico inteiro)
PRESETS: dict[str, tuple[str, int | None]] = {
    "30d": ("30 dias", 30),
    "90d": ("90 dias", DEFAULT_PERIOD_DAYS),
    "12m": ("12 meses", 365),
    "tudo": ("Tudo", None),
    CUSTOM: ("Outro", None),
}
COMPARE_KEY = "f_compare"
_COMPARE_SHORT = {"anterior": "Período anterior", "ano_anterior": "Ano anterior"}


def sales_filters(df: pd.DataFrame, dataset_key: str) -> SalesFilters:
    """Período, categorias, regiões e base de comparação (barra lateral + URL).

    Os valores ficam na sessão (`persist_state`), então valem em todas as
    páginas: filtrar "Moda" na visão geral mantém o filtro na previsão.
    """
    area = sidebar_filters()
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    preset_key, range_key, picker_key = (f"f_{name}_{dataset_key}" for name in ("preset", "range", "period"))

    # primeira visita: período da URL (link compartilhado) ou o padrão de 90 dias
    start_url, end_url = _url_date(P_START, min_date, max_date), _url_date(P_END, min_date, max_date)
    if start_url and end_url and start_url <= end_url:
        url_range = (start_url, end_url)
        _seed(preset_key, next((k for k in PRESETS if _preset_range(k, min_date, max_date) == url_range), CUSTOM))
        _seed(range_key, url_range)
    _seed(preset_key, "90d")
    _seed(range_key, _preset_range("90d", min_date, max_date))
    if not _valid_range(st.session_state[range_key], min_date, max_date):
        st.session_state[range_key] = _preset_range("90d", min_date, max_date)

    preset = (
        area.segmented_control(
            "Período",
            options=list(PRESETS),
            format_func=lambda k: PRESETS[k][0],
            key=preset_key,
            required=True,
            persist_state="session",
            width="stretch",
        )
        or "90d"
    )  # `required` garante uma opção; o padrão só satisfaz a checagem de tipos
    if preset == CUSTOM:
        if picker_key not in st.session_state:
            st.session_state[picker_key] = st.session_state[range_key]
        selected = area.date_input("Datas", key=picker_key, min_value=min_date, max_value=max_date, format="DD/MM/YYYY")
        # enquanto o usuário escolhe só a data inicial, o seletor devolve uma única data
        if isinstance(selected, tuple) and len(selected) == 2:
            st.session_state[range_key] = selected
    else:
        st.session_state[range_key] = _preset_range(preset, min_date, max_date)
        st.session_state.pop(picker_key, None)
    start, end = st.session_state[range_key]
    if preset != CUSTOM:
        area.caption(f":material/date_range: {start:%d/%m/%Y} a {end:%d/%m/%Y}")

    categories = segment_filter(df, "category", "Categorias", P_CATEGORIES, dataset_key)
    regions = segment_filter(df, "region", "Regiões", P_REGIONS, dataset_key)

    _seed(
        COMPARE_KEY,
        st.query_params.get(P_COMPARE) if st.query_params.get(P_COMPARE) in COMPARISON_LABELS else "anterior",
    )
    comparison = area.segmented_control(
        "Comparar com",
        options=list(COMPARISON_LABELS),
        format_func=lambda c: _COMPARE_SHORT[c],
        key=COMPARE_KEY,
        required=True,
        persist_state="session",
        width="stretch",
        help="Base usada no crescimento, na análise de variação e no relatório. Comparar com o ano "
        "anterior remove o efeito da sazonalidade (ex.: dezembro x dezembro).",
    )

    _write_url(
        {
            P_START: start.isoformat(),
            P_END: end.isoformat(),
            P_CATEGORIES: categories,
            P_REGIONS: regions,
            P_COMPARE: comparison,
        }
    )
    return SalesFilters(start, end, categories, regions, "ano_anterior" if comparison == "ano_anterior" else "anterior")


def segment_filter(df: pd.DataFrame, column: str, label: str, param: str, dataset_key: str) -> list[str]:
    """Multiselect de categorias/regiões, iniciado pela URL e limitado às opções existentes."""
    options = sorted(df[column].unique())
    key = segment_key(param, dataset_key)
    _seed(key, list(dict.fromkeys(v for v in st.query_params.get_all(param) if v in options)))
    st.session_state[key] = [v for v in st.session_state[key] if v in options]
    values = sidebar_filters().multiselect(
        label, options=options, key=key, placeholder="Todas", persist_state="session"
    )
    _write_url({param: values})
    return values


def segment_key(param: str, dataset_key: str) -> str:
    return f"f_{param}_{dataset_key}"


def select_segment(param: str, dataset_key: str, chart_key: str) -> None:
    """Filtro cruzado: o clique em uma barra vira o filtro de categoria/região.

    Usado como `on_select` do gráfico, roda antes da página ser redesenhada,
    quando ainda é permitido mudar o valor do multiselect.
    """
    points = st.session_state[chart_key].selection.points
    if points:
        st.session_state[segment_key(param, dataset_key)] = [points[0]["y"]]
        # gráfico novo (sem a barra marcada) na próxima execução
        st.session_state[f"_click_{param}"] = st.session_state.get(f"_click_{param}", 0) + 1


def chart_key(param: str) -> str:
    return f"click_{param}_{st.session_state.get(f'_click_{param}', 0)}"


def clear_segments(dataset_key: str) -> None:
    for param in (P_CATEGORIES, P_REGIONS):
        st.session_state[segment_key(param, dataset_key)] = []


def revenue_goal(dataset_key: str) -> float:
    """Meta de receita do período (0 = sem meta), também guardada na URL."""
    area = sidebar_filters()
    key = f"f_goal_{dataset_key}"
    _seed(key, _url_float(P_GOAL))
    goal = area.number_input(
        "Meta de receita do período (R$)",
        key=key,
        min_value=0.0,
        step=50_000.0,
        format="%.0f",
        persist_state="session",
        help="Opcional. Mostra o progresso da receita do período em relação à meta.",
    )
    if goal:
        area.caption(f"Meta: {format_brl(goal, 0)}".replace("$", r"\$"))
    _write_url({P_GOAL: f"{goal:.0f}" if goal else None})
    return float(goal)


# ------------------------------------------------------------------
def _seed(key: str, value) -> None:
    """Define o valor inicial do widget só na primeira vez (depois vale a escolha do usuário)."""
    if key not in st.session_state:
        st.session_state[key] = value


def _preset_range(preset: str, min_date: date, max_date: date) -> tuple[date, date]:
    days = PRESETS.get(preset, PRESETS["90d"])[1]
    if days is None:
        return (min_date, max_date)
    # "30 dias" = os 30 últimos dias, incluindo o último
    return (max(min_date, max_date - timedelta(days=days - 1)), max_date)


def _valid_range(value, min_date: date, max_date: date) -> bool:
    if not isinstance(value, tuple | list) or not value:
        return False
    return all(isinstance(d, date) and min_date <= d <= max_date for d in value)


def _url_date(param: str, min_date: date, max_date: date) -> date | None:
    try:
        value = date.fromisoformat(st.query_params.get(param, ""))
    except ValueError:
        return None
    return value if min_date <= value <= max_date else None


def _url_float(param: str) -> float:
    try:
        value = float(st.query_params.get(param, "0"))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0  # "inf" e "nan" também são float


def _write_url(params: dict) -> None:
    for name, value in params.items():
        if value in (None, [], ""):
            st.query_params.pop(name, None)
        else:
            st.query_params[name] = value
