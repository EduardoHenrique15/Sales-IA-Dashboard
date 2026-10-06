"""
Página: Dashboard Executivo de Vendas.
"""

import pandas as pd
import streamlit as st

from insight_engine.ai.context import build_sales_facts
from insight_engine.analytics.kpis import SalesKPIs, compute_sales_kpis
from insight_engine.analytics.periods import apply_segments, filter_sales
from insight_engine.analytics.variance import revenue_bridge
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import charts, datasets, filters
from insight_engine.ui.components import escape_currency, kpi_card, report_section

dataset = datasets.active_dataset()
df_all = dataset.df
dataset_key = datasets.dataset_key(dataset)

# ---------- FILTROS (barra lateral, sincronizados com a URL) ----------
selection = filters.sales_filters(df_all, dataset_key)
goal = filters.revenue_goal(dataset_key)

df_filtered, df_prev = filter_sales(df_all, selection)
kpis = compute_sales_kpis(df_filtered, previous_df=df_prev)
comparison = selection.comparison_label

# ---------- CABEÇALHO ----------
st.title("📊 Dashboard Executivo de Vendas")
st.caption(
    f"Período selecionado: **{selection.period_label}**  •  {format_number(kpis.n_orders)} pedidos analisados"
    f"  •  Base: {dataset.name}  •  🔗 os filtros ficam no endereço da página: copie o link para compartilhar"
)

if df_filtered.empty:
    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados. Ajuste o período ou os filtros.")
else:
    # ---------- KPI CARDS ----------
    growth = kpis.revenue_growth_pct
    short_comparison = "ano anterior" if selection.comparison == "ano_anterior" else "período anterior"
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card(
            "Receita Total",
            format_brl(kpis.total_revenue),
            delta=f"{format_pct(growth, signed=True)} vs {short_comparison}" if growth is not None else None,
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

    # ---------- META ----------
    if goal > 0:
        progress = kpis.total_revenue / goal
        status = (
            f"meta superada em {format_brl(kpis.total_revenue - goal)}"
            if progress >= 1
            else f"faltam {format_brl(goal - kpis.total_revenue)}"
        )
        st.progress(
            min(progress, 1.0),
            text=escape_currency(f"🎯 {format_pct(progress * 100)} da meta de {format_brl(goal)} — {status}"),
        )

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

    # ---------- COMPARAÇÃO E VARIAÇÃO ----------
    st.subheader("🔎 Por que a receita mudou?")
    if df_prev.empty:
        st.info(f"Não há vendas no {comparison} para comparar.")
    else:
        previous_kpis = compute_sales_kpis(df_prev)
        bridge = revenue_bridge(df_filtered, df_prev)
        explanation = (
            f"Comparação com o {comparison}: a receita variou **{format_brl(bridge.total_change)}**, que se "
            f"decompõem em três efeitos: **volume** {format_brl(bridge.volume_effect)} (mais ou menos unidades "
            f"vendidas), **preço** {format_brl(bridge.price_effect)} (preço médio de cada categoria) e **mix** "
            f"{format_brl(bridge.mix_effect)} (venda migrando entre categorias mais caras ou mais baratas)."
        )
        st.caption(escape_currency(explanation))

        def comparison_table(current: SalesKPIs, previous: SalesKPIs) -> pd.DataFrame:
            def row(name, now, before, fmt, pct_points=False):
                if now is None or before is None:
                    return [name, "—", "—", "—"]
                if pct_points:
                    change = f"{format_number(now - before, 1)} p.p."
                else:
                    change = format_pct((now - before) / before * 100, signed=True) if before else "—"
                return [name, fmt(now), fmt(before), change]

            rows = [
                row("Receita", current.total_revenue, previous.total_revenue, format_brl),
                row("Lucro", current.total_profit, previous.total_profit, format_brl),
                row("Margem", current.margin_pct, previous.margin_pct, format_pct, pct_points=True),
                row("Pedidos", current.n_orders, previous.n_orders, format_number),
                row("Unidades", current.total_units, previous.total_units, format_number),
                row("Ticket médio", current.avg_ticket, previous.avg_ticket, format_brl),
            ]
            return pd.DataFrame(rows, columns=["Indicador", "Período atual", "Comparação", "Variação"])

        col_table, col_bridge = st.columns([2, 3])
        with col_table:
            st.markdown(
                f"**Período atual x {comparison}**  \n"
                f"<small>{df_prev['date'].min():%d/%m/%Y} a {df_prev['date'].max():%d/%m/%Y} na comparação</small>",
                unsafe_allow_html=True,
            )
            st.dataframe(comparison_table(kpis, previous_kpis), hide_index=True, width="stretch")
            st.plotly_chart(charts.bridge_by_segment(bridge.by_segment))
        with col_bridge:
            st.plotly_chart(charts.revenue_bridge(bridge))

    with st.expander("🔍 Ver dados brutos filtrados"):
        st.dataframe(df_filtered)

report_section(
    lambda: (
        build_sales_facts(
            history=apply_segments(df_all, selection),
            period=df_filtered,
            previous=df_prev,
            kpis=kpis,
            period_label=selection.period_label,
            has_customers=dataset.has_customers,
            comparison_label=comparison,
        )
        if not df_filtered.empty
        else None
    ),
    period_label=selection.period_label,
    dataset_name="Vendas",
    context=f"{dataset.name}_{selection.comparison}",
)
