"""
Página: previsão de receita, anomalias e padrões sazonais.
"""

import pandas as pd
import streamlit as st

from insight_engine.analytics.anomalies import daily_series, monthly_seasonality
from insight_engine.analytics.forecasting import BASELINE, InsufficientDataError
from insight_engine.data.sales import PLANTED_ANOMALIES
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import cached, charts, datasets, filters

dataset = datasets.active_dataset()
df_all = dataset.df

# ---------- FILTROS (barra lateral) ----------
dataset_key = datasets.dataset_key(dataset)
categories = filters.segment_filter(df_all, "category", "Categorias", filters.P_CATEGORIES, dataset_key)
regions = filters.segment_filter(df_all, "region", "Regiões", filters.P_REGIONS, dataset_key)
horizon = st.sidebar.radio("Horizonte da previsão", options=[30, 60, 90], format_func=lambda d: f"{d} dias")

df = df_all
if categories:
    df = df[df["category"].isin(categories)]
if regions:
    df = df[df["region"].isin(regions)]

st.title("📈 Previsão e Anomalias")
st.caption(
    f"Histórico de {df['date'].min():%d/%m/%Y} a {df['date'].max():%d/%m/%Y}  •  Base: {dataset.name}"
    if not df.empty
    else f"Base: {dataset.name}"
)
if df.empty:
    st.warning("⚠️ Nenhum dado para os filtros selecionados.")
    st.stop()

revenue = daily_series(df, "revenue")

# ---------- PREVISÃO ----------
st.header("Previsão de receita")
try:
    result = cached.forecast(revenue, horizon)
except InsufficientDataError as exc:
    st.warning(f"⚠️ {exc}")
else:
    total = result.forecast["yhat"].sum()
    scores = result.scores
    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Receita prevista (próximos {horizon} dias)",
        format_brl(total),
        help="Margem de erro típica do total, medida no backtesting.",
    )
    c1.caption(f"Erro típico do total: ± {format_pct(result.total_error_pct)}")
    # texto comum em vez de st.metric: nomes de modelo são longos para o tamanho do valor
    c2.markdown(f"Modelo escolhido  \n**{result.best_model}**")
    c2.caption(f"Erro diário (WAPE): {format_pct(scores.loc[result.best_model, 'wape'])}")
    c3.metric(
        "Ganho sobre o baseline",
        format_pct(scores.loc[result.best_model, "improvement_vs_baseline"]),
        help="Redução do erro em relação a simplesmente repetir a última semana.",
    )
    st.plotly_chart(charts.forecast(result))

    with st.expander("🧪 Como o modelo foi escolhido (backtesting)"):
        st.markdown(
            f"Cada modelo foi treinado só com o passado e testado em **{result.n_folds} janelas de "
            f"{horizon} dias** que ele não tinha visto. Vence o menor **WAPE** (erro absoluto total ÷ "
            "receita total). Os intervalos do gráfico vêm dos erros reais dessas janelas, não de uma "
            "suposição de distribuição normal."
        )
        table = scores.rename(
            columns={
                "wape": "WAPE diário (%)",
                "total_error": f"Erro do total em {horizon} dias (%)",
                "bias": "Viés (%)",
                "rmse": "RMSE (R$)",
                "improvement_vs_baseline": "Ganho vs baseline (%)",
            }
        )
        st.dataframe(table.style.format("{:,.1f}"), width="stretch")
        st.caption(
            "Viés positivo = o modelo subestima a receita. O dia a dia de vendas é ruidoso (poucos pedidos "
            "caros mudam um dia inteiro), por isso o erro do **total** do período é bem menor que o diário. "
            f"O baseline é o modelo '{BASELINE}'."
        )

# ---------- ANOMALIAS ----------
st.header("Anomalias")
metric = st.radio(
    "Detectar anomalias em",
    options=["orders", "revenue"],
    format_func={"orders": "Número de pedidos", "revenue": "Receita"}.get,
    horizontal=True,
    help="O número de pedidos costuma ser o sinal mais confiável para incidentes: a receita oscila muito "
    "com poucos pedidos caros.",
)
series = daily_series(df, metric)
try:
    found = cached.anomalies(series)
except ValueError as exc:
    st.warning(f"⚠️ {exc}")
else:
    label = "Pedidos" if metric == "orders" else "Receita (R$)"
    st.plotly_chart(charts.anomalies(series, found, label))
    st.caption(
        f"{len(found)} dia(s) fora do padrão em {format_number(len(series))} analisados. Método: decomposição "
        "STL em escala logarítmica (tendência + sazonalidade semanal), z-score robusto do resíduo acima de 3,5 "
        "e desvio de pelo menos 20% em relação ao esperado."
    )
    if datasets.is_example(dataset) and not categories and not regions:
        planted = {pd.Timestamp(day) for day, _, _ in PLANTED_ANOMALIES}
        hits = len(planted & set(found.index))
        st.info(
            f"🎯 A base de exemplo tem {len(planted)} anomalias plantadas de propósito "
            f"({', '.join(f'{pd.Timestamp(d):%d/%m/%Y} ({k})' for d, k, _ in PLANTED_ANOMALIES)}). "
            f"Nesta métrica, o detector encontrou **{hits} de {len(planted)}**."
        )
    if not found.empty:
        table = found.assign(
            Data=found.index.strftime("%d/%m/%Y"),
            Observado=found["value"].map(lambda v: format_number(v, 0)),
            Esperado=found["expected"].map(lambda v: format_number(v, 0)),
            Desvio=found["deviation_pct"].map(lambda v: format_pct(v, 0, signed=True)),
            Tipo=found["kind"],
        )[["Data", "Tipo", "Observado", "Esperado", "Desvio"]]
        st.dataframe(table, hide_index=True)

# ---------- PADRÕES SAZONAIS ----------
st.header("Padrões sazonais")
col_week, col_month = st.columns(2)
with col_week:
    try:
        st.plotly_chart(charts.weekday_effect(cached.decomposition(revenue).weekday_profile))
    except ValueError as exc:
        st.warning(f"⚠️ {exc}")
with col_month:
    if len(revenue) >= 365:
        st.plotly_chart(charts.monthly_index(monthly_seasonality(revenue)))
    else:
        st.info("A sazonalidade mensal aparece com pelo menos um ano de histórico.")
