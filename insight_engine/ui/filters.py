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

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from insight_engine.analytics.periods import COMPARISON_LABELS, SalesFilters

DEFAULT_PERIOD_DAYS = 90
P_START, P_END, P_CATEGORIES, P_REGIONS, P_COMPARE, P_GOAL = "de", "ate", "categorias", "regioes", "comparar", "meta"


def sales_filters(df: pd.DataFrame, dataset_key: str) -> SalesFilters:
    """Período, categorias, regiões e base de comparação (barra lateral + URL)."""
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    default = (max(min_date, max_date - timedelta(days=DEFAULT_PERIOD_DAYS)), max_date)

    period_key = f"f_period_{dataset_key}"
    start_url, end_url = _url_date(P_START, min_date, max_date), _url_date(P_END, min_date, max_date)
    _seed(period_key, (start_url, end_url) if start_url and end_url and start_url <= end_url else default)
    stored = st.session_state[period_key]
    if not _valid_range(stored, min_date, max_date):
        st.session_state[period_key] = default
    selected = st.sidebar.date_input(
        "Período", key=period_key, min_value=min_date, max_value=max_date, format="DD/MM/YYYY"
    )
    # enquanto o usuário escolhe só a data inicial, o seletor devolve uma única data
    start, end = selected if isinstance(selected, tuple) and len(selected) == 2 else default

    categories = segment_filter(df, "category", "Categorias", P_CATEGORIES, dataset_key)
    regions = segment_filter(df, "region", "Regiões", P_REGIONS, dataset_key)

    compare_key = "f_compare"
    _seed(
        compare_key,
        st.query_params.get(P_COMPARE) if st.query_params.get(P_COMPARE) in COMPARISON_LABELS else "anterior",
    )
    comparison = st.sidebar.radio(
        "Comparar com",
        options=list(COMPARISON_LABELS),
        format_func=lambda c: "Período anterior" if c == "anterior" else "Mesmo período do ano anterior",
        key=compare_key,
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
    key = f"f_{param}_{dataset_key}"
    _seed(key, [v for v in st.query_params.get_all(param) if v in options])
    st.session_state[key] = [v for v in st.session_state[key] if v in options]
    values = st.sidebar.multiselect(label, options=options, key=key, placeholder="Todas")
    _write_url({param: values})
    return values


def revenue_goal(dataset_key: str) -> float:
    """Meta de receita do período (0 = sem meta), também guardada na URL."""
    key = f"f_goal_{dataset_key}"
    _seed(key, _url_float(P_GOAL))
    goal = st.sidebar.number_input(
        "Meta de receita do período (R$)",
        key=key,
        min_value=0.0,
        step=50_000.0,
        format="%.0f",
        help="Opcional. Mostra o progresso da receita do período em relação à meta.",
    )
    _write_url({P_GOAL: f"{goal:.0f}" if goal else None})
    return float(goal)


# ------------------------------------------------------------------
def _seed(key: str, value) -> None:
    """Define o valor inicial do widget só na primeira vez (depois vale a escolha do usuário)."""
    if key not in st.session_state:
        st.session_state[key] = value


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
        return max(0.0, float(st.query_params.get(param, "0")))
    except ValueError:
        return 0.0


def _write_url(params: dict) -> None:
    for name, value in params.items():
        if value in (None, [], ""):
            st.query_params.pop(name, None)
        else:
            st.query_params[name] = value
