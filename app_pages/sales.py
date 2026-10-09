"""
Página: visão geral de vendas.
"""

import pandas as pd
import streamlit as st

from insight_engine.ai.context import build_sales_facts
from insight_engine.analytics.kpis import SalesKPIs, compute_sales_kpis
from insight_engine.analytics.periods import apply_segments, filter_sales
from insight_engine.analytics.variance import revenue_bridge
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import charts, datasets, filters
from insight_engine.ui.components import chart_card, escape_currency, page_header, report_section

# nomes e formatos das colunas na aba "Dados"
DATA_COLUMNS = {
    "date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
    "category": st.column_config.TextColumn("Categoria"),
    "region": st.column_config.TextColumn("Região"),
    "product": st.column_config.TextColumn("Produto"),
    "customer_id": st.column_config.TextColumn("Cliente"),
    "units": st.column_config.NumberColumn("Unidades", format="localized"),
    "unit_price": st.column_config.NumberColumn("Preço unitário (R$)", format="localized"),
    "revenue": st.column_config.NumberColumn("Receita (R$)", format="localized"),
    "cost": st.column_config.NumberColumn("Custo (R$)", format="localized"),
    "profit": st.column_config.NumberColumn("Lucro (R$)", format="localized"),
}

dataset = datasets.active_dataset()
df_all = dataset.df
dataset_key = datasets.dataset_key(dataset)

# ---------- FILTROS (barra lateral, sincronizados com a URL) ----------
selection = filters.sales_filters(df_all, dataset_key)
goal = filters.revenue_goal(dataset_key)

df_filtered, df_prev = filter_sales(df_all, selection)
kpis = compute_sales_kpis(df_filtered, previous_df=df_prev)
previous_kpis = compute_sales_kpis(df_prev) if not df_prev.empty else None
comparison = selection.comparison_label
versus = "vs ano anterior" if selection.comparison == "ano_anterior" else "vs anterior"

# ---------- CABEÇALHO ----------
page_header(
    "Visão geral de vendas",
    f":material/calendar_month: **{selection.period_label}** · {format_number(kpis.n_orders)} pedidos · "
    f"comparação com o {comparison} · Base: {dataset.name}",
    icon=":material/dashboard:",
)

if selection.categories or selection.regions:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption("Filtrando por:", width="content")
        for value in [*selection.categories, *selection.regions]:
            st.badge(value, icon=":material/filter_alt:", color="blue")
        st.button(
            "Limpar filtros",
            icon=":material/filter_alt_off:",
            type="tertiary",
            on_click=filters.clear_segments,
            args=(dataset_key,),
            key="clear_filters",
        )

if df_filtered.empty:
    st.warning(
        "Nenhum dado encontrado para os filtros selecionados. Ajuste o período ou os filtros.",
        icon=":material/search_off:",
    )
else:
    # ---------- KPIs ----------
    # um ponto por dia do calendário (dias sem venda = zero)
    by_day = (
        df_filtered.set_index("date")
        .resample("D")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("order_id", "nunique") if "order_id" in df_filtered.columns else ("revenue", "size"),
        )
    )
    # minigráficos: por dia em períodos de até um mês, por semana nos maiores (menos ruído). Só semanas
    # completas: a semana parcial das pontas pareceria uma queda que não aconteceu.
    if len(by_day) <= 31:
        trend = by_day
    else:
        weekly = by_day.resample("W-MON", label="left", closed="left")
        trend = weekly.sum()[weekly["revenue"].count() == 7]

    def points(values: pd.Series) -> list[float]:
        return [round(float(v), 2) for v in values.fillna(0)]

    def change(now: float | None, before: float | None) -> dict:
        """Variação para o `st.metric` (vazia sem base de comparação; neutra quando arredonda para zero)."""
        if now is None or not before or not has_prev:
            return {}
        pct = (now - before) / abs(before) * 100
        if round(pct, 1) == 0:
            return {"delta": format_pct(0), "delta_color": "off", "delta_arrow": "off"}
        return {"delta": format_pct(pct, signed=True)}

    prev = previous_kpis or SalesKPIs()
    has_prev = previous_kpis is not None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Receita",
        format_brl(kpis.total_revenue, 0),
        **change(kpis.total_revenue, prev.total_revenue),
        delta_description=versus,
        border=True,
        chart_data=points(trend["revenue"]),
        chart_type="area",
        help=f"Valor exato: {format_brl(kpis.total_revenue)}",
    )
    if kpis.total_profit is None or kpis.margin_pct is None:
        c2.metric("Lucro", "—", border=True, help="A base não informa o custo, então o lucro não é calculado.")
    else:
        c2.metric(
            f"Lucro · margem {format_pct(kpis.margin_pct)}",
            format_brl(kpis.total_profit, 0),
            **change(kpis.total_profit, prev.total_profit),
            delta_description=versus,
            border=True,
            chart_data=points(trend["profit"]),
            chart_type="area",
            help=f"Valor exato: {format_brl(kpis.total_profit)}",
        )
    c3.metric(
        "Pedidos",
        format_number(kpis.n_orders),
        **change(kpis.n_orders, prev.n_orders),
        delta_description=versus,
        border=True,
        chart_data=points(trend["orders"]),
        chart_type="bar",
    )
    c4.metric(
        "Ticket médio",
        format_brl(kpis.avg_ticket),
        **change(kpis.avg_ticket, prev.avg_ticket),
        delta_description=versus,
        border=True,
        chart_data=points(trend["revenue"] / trend["orders"].where(trend["orders"] > 0)),
        chart_type="line",
    )

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
            text=escape_currency(f"Meta: {format_pct(progress * 100)} de {format_brl(goal)} atingidos — {status}"),
        )

    # ---------- ABAS ----------
    tab_trend, tab_mix, tab_change, tab_data = st.tabs(
        [
            ":material/show_chart: Evolução",
            ":material/bar_chart: Composição",
            ":material/waterfall_chart: Variação",
            ":material/table_rows: Dados",
        ],
        key="overview_tab",  # mantém a aba aberta depois de um filtro por clique
    )

    with tab_trend:
        granularity = st.segmented_control(
            "Agrupar por",
            options=list(charts.GRANULARITIES),
            format_func=charts.GRANULARITIES.get,
            default="D",
            required=True,
            key="granularity",
            persist_state="session",
        )
        title = "Receita e lucro" if dataset.has_cost else "Receita"
        chart_card(
            f"{title} por {charts.GRANULARITIES[granularity].lower()}",
            charts.revenue_over_time(df_filtered, show_profit=dataset.has_cost, granularity=granularity),
            caption=(
                f"Linha forte: média móvel de {charts.MOVING_AVERAGE_DAYS} dias. Linha clara: valor de cada dia."
                if granularity == "D"
                else "Semanas e meses nas pontas do período podem estar incompletos."
            ),
        )

    with tab_mix:
        st.caption(":material/touch_app: Clique em uma barra de categoria ou região para filtrar o dashboard inteiro.")
        col_cat, col_region = st.columns(2)
        with col_cat:
            key = filters.chart_key(filters.P_CATEGORIES)
            chart_card(
                "Receita por categoria",
                charts.revenue_by_category(kpis.revenue_by_category),
                caption=f"Líder: {kpis.top_category} ({format_brl(kpis.top_category_revenue, 0)})",
                key=key,
                on_select=lambda key=key: filters.select_segment(filters.P_CATEGORIES, dataset_key, key),
            )
        with col_region:
            key = filters.chart_key(filters.P_REGIONS)
            chart_card(
                "Receita por região",
                charts.revenue_by_region(kpis.revenue_by_region),
                caption=f"Líder: {kpis.top_region}",
                key=key,
                on_select=lambda key=key: filters.select_segment(filters.P_REGIONS, dataset_key, key),
            )
        chart_card("Produtos com maior receita", charts.top_products(df_filtered))

    with tab_change:
        if previous_kpis is None:
            st.info(f"Não há vendas no {comparison} para comparar.", icon=":material/info:")
        else:
            bridge = revenue_bridge(df_filtered, df_prev)
            st.markdown(
                escape_currency(
                    f"Em relação ao {comparison}, a receita variou **{format_brl(bridge.total_change)}**. "
                    f"A variação se divide em três efeitos: **volume** {format_brl(bridge.volume_effect)} "
                    f"(mais ou menos unidades vendidas), **preço** {format_brl(bridge.price_effect)} (preço médio "
                    f"de cada categoria) e **mix** {format_brl(bridge.mix_effect)} (venda migrando entre "
                    "categorias mais caras ou mais baratas)."
                )
            )

            def comparison_table(current: SalesKPIs, previous: SalesKPIs) -> pd.DataFrame:
                def row(name, now, before, fmt, pct_points=False):
                    if now is None or before is None:
                        return [name, "—", "—", "—"]
                    if pct_points:
                        variation = f"{format_number(now - before, 1)} p.p."
                    else:
                        variation = format_pct((now - before) / before * 100, signed=True) if before else "—"
                    return [name, fmt(now), fmt(before), variation]

                rows = [
                    row("Receita", current.total_revenue, previous.total_revenue, format_brl),
                    row("Lucro", current.total_profit, previous.total_profit, format_brl),
                    row("Margem", current.margin_pct, previous.margin_pct, format_pct, pct_points=True),
                    row("Pedidos", current.n_orders, previous.n_orders, format_number),
                    row("Unidades", current.total_units, previous.total_units, format_number),
                    row("Ticket médio", current.avg_ticket, previous.avg_ticket, format_brl),
                ]
                return pd.DataFrame(rows, columns=["Indicador", "Período atual", "Comparação", "Variação"])

            col_bridge, col_segment = st.columns(2)
            with col_bridge:
                chart_card(
                    "Da receita anterior à atual",
                    charts.revenue_bridge(bridge),
                    caption="Eixo ampliado em torno da variação (não começa em zero).",
                )
            with col_segment:
                chart_card(
                    "Variação por categoria",
                    charts.bridge_by_segment(bridge.by_segment),
                    caption="Passe o mouse para ver a divisão em volume, preço e mix de cada categoria.",
                )
            with st.container(border=True):
                st.markdown("**Período atual x comparação**")
                st.caption(f"Comparação: {df_prev['date'].min():%d/%m/%Y} a {df_prev['date'].max():%d/%m/%Y}")
                st.dataframe(comparison_table(kpis, previous_kpis), hide_index=True, width="stretch")

    with tab_data:
        columns = [c for c in DATA_COLUMNS if c in df_filtered.columns and df_filtered[c].notna().any()]
        st.caption(f"{format_number(len(df_filtered))} pedidos no período e filtros atuais.")
        st.dataframe(
            df_filtered[columns].sort_values("date", ascending=False),
            column_config={c: DATA_COLUMNS[c] for c in columns},
            hide_index=True,
            width="stretch",
            height=420,
        )
        st.download_button(
            "Baixar estes dados (CSV)",
            data=df_filtered[columns].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
            file_name="vendas_filtradas.csv",
            mime="text/csv",
            icon=":material/download:",
        )

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
