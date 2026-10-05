"""
Página: Dashboard Executivo de Vendas.
"""

from datetime import timedelta

import streamlit as st

from insight_engine.analytics.kpis import compute_sales_kpis
from insight_engine.analytics.periods import SalesFilters, filter_sales
from insight_engine.analytics.variance import revenue_bridge
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import charts, datasets
from insight_engine.ui.components import escape_currency, kpi_card, report_section

dataset = datasets.active_dataset()
df_all = dataset.df

# ---------- FILTROS (barra lateral) ----------
min_date, max_date = df_all["date"].min().date(), df_all["date"].max().date()
default_start = max(min_date, max_date - timedelta(days=90))

date_range = st.sidebar.date_input(
    "Período",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)
categories = st.sidebar.multiselect("Categorias", options=sorted(df_all["category"].unique()), default=[])
regions = st.sidebar.multiselect("Regiões", options=sorted(df_all["region"].unique()), default=[])

# Enquanto o usuário escolhe só a data inicial, o seletor devolve uma única data.
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, max_date

filters = SalesFilters(start_date, end_date, categories, regions)
df_filtered, df_prev = filter_sales(df_all, filters)
kpis = compute_sales_kpis(df_filtered, previous_df=df_prev)

# ---------- CABEÇALHO ----------
st.title("📊 Dashboard Executivo de Vendas")
st.caption(
    f"Período selecionado: **{filters.period_label}**  •  {format_number(kpis.n_orders)} pedidos analisados"
    f"  •  Base: {dataset.name}"
)

if df_filtered.empty:
    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados. Ajuste o período ou os filtros.")
else:
    # ---------- KPI CARDS ----------
    growth = kpis.revenue_growth_pct
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card(
            "Receita Total",
            format_brl(kpis.total_revenue),
            delta=f"{format_pct(growth, signed=True)} vs período anterior" if growth is not None else None,
            delta_positive=growth >= 0 if growth is not None else None,
        )
    with c2:
        if kpis.total_profit is None or kpis.margin_pct is None:
            kpi_card("Lucro Total", "—", "Custo não informado na base")
        else:
            kpi_card(
                "Lucro Total",
                format_brl(kpis.total_profit),
                f"Margem: {format_pct(kpis.margin_pct)}",
                delta_positive=kpis.margin_pct >= 20,
            )
    with c3:
        kpi_card("Unidades Vendidas", format_number(kpis.total_units))
    with c4:
        kpi_card("Ticket Médio", format_brl(kpis.avg_ticket))
    with c5:
        kpi_card("Categoria Líder", kpis.top_category, format_brl(kpis.top_category_revenue))

    st.write("")

    # ---------- GRÁFICOS ----------
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.plotly_chart(charts.revenue_profit_over_time(df_filtered, show_profit=dataset.has_cost))
    with col_right:
        st.plotly_chart(charts.revenue_by_category(kpis.revenue_by_category))

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(charts.revenue_by_region(kpis.revenue_by_region))
    with col_b:
        st.plotly_chart(charts.top_products(df_filtered))

    # ---------- POR QUE A RECEITA MUDOU? ----------
    st.subheader("🔎 Por que a receita mudou?")
    if df_prev.empty:
        st.info("Não há vendas no período anterior de mesma duração para comparar.")
    else:
        bridge = revenue_bridge(df_filtered, df_prev)
        explanation = (
            f"Comparação com o período anterior de mesma duração: a receita variou "
            f"**{format_brl(bridge.total_change)}**, que se decompõem em três efeitos: "
            f"**volume** {format_brl(bridge.volume_effect)} (mais ou menos unidades vendidas), "
            f"**preço** {format_brl(bridge.price_effect)} (preço médio de cada categoria) e "
            f"**mix** {format_brl(bridge.mix_effect)} (venda migrando entre categorias mais caras ou mais baratas)."
        )
        st.caption(escape_currency(explanation))
        col_bridge, col_segments = st.columns([3, 2])
        with col_bridge:
            st.plotly_chart(charts.revenue_bridge(bridge))
        with col_segments:
            st.plotly_chart(charts.bridge_by_segment(bridge.by_segment))

    with st.expander("🔍 Ver dados brutos filtrados"):
        st.dataframe(df_filtered)

report_section(df_filtered, kpis, filters.period_label, dataset_name="Vendas", context=dataset.name)
