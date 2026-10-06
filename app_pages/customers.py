"""
Página: análise de clientes (RFM + K-Means).
"""

import pandas as pd
import streamlit as st

from insight_engine.analytics.customers import MIN_ACTIONABLE_K, pareto_share
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import cached, charts, datasets, filters

ACTIVE_DAYS = 90

dataset = datasets.active_dataset()
st.title("👥 Análise de Clientes")

if not dataset.has_customers:
    st.info(
        "A base em uso não tem coluna de cliente. Para usar esta página, envie uma planilha com o "
        "identificador do cliente em **Importar dados**."
    )
    st.stop()

df_all = dataset.df
dataset_key = datasets.dataset_key(dataset)
regions = filters.segment_filter(df_all, "region", "Regiões", filters.P_REGIONS, dataset_key)
df = df_all[df_all["region"].isin(regions)] if regions else df_all

try:
    seg = cached.segmentation(df)
except ValueError as exc:
    st.warning(f"⚠️ {exc}")
    st.stop()

customers = seg.customers
st.caption(
    f"Situação em {seg.reference_date - pd.Timedelta(days=1):%d/%m/%Y} (última venda da base)  •  Base: {dataset.name}"
)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", format_number(len(customers)))
active = (customers["recency"] <= ACTIVE_DAYS).mean() * 100
c2.metric(f"Ativos (compraram em {ACTIVE_DAYS} dias)", format_pct(active))
repeat = (customers["frequency"] > 1).mean() * 100
c3.metric("Compraram mais de uma vez", format_pct(repeat))
c4.metric(
    "Receita dos 20% melhores clientes",
    format_pct(pareto_share(customers)),
    help="Concentração da receita (princípio de Pareto).",
)

# ---------- SEGMENTOS RFM ----------
st.header("Segmentos RFM")
st.caption(
    "Cada cliente recebe notas de 1 a 5 (quintis) em **Recência** (dias desde a última compra), "
    "**Frequência** (pedidos) e **Valor** (receita). Regras clássicas de CRM transformam as notas em "
    "segmentos com ação sugerida."
)
st.plotly_chart(charts.segments(seg.segments))
table = seg.segments.assign(
    Segmento=seg.segments.index,
    Clientes=seg.segments["customers"].map(format_number),
    Receita=seg.segments["revenue"].map(format_brl),
    **{
        "% da receita": seg.segments["revenue_pct"].map(format_pct),
        "Dias desde a última compra (média)": seg.segments["avg_recency"].map(format_number),
        "Ação sugerida": seg.segments["action"],
    },
)[["Segmento", "Clientes", "Receita", "% da receita", "Dias desde a última compra (média)", "Ação sugerida"]]
st.table(table.set_index("Segmento"))

export = customers.reset_index()[["customer_id", "segment", "recency", "frequency", "monetary"]]
st.download_button(
    "⬇️ Baixar lista de clientes com segmento (CSV)",
    data=export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
    file_name="clientes_segmentados.csv",
    mime="text/csv",
    help="Pronta para usar em campanhas de CRM/e-mail.",
)

# ---------- K-MEANS ----------
st.header("Grupos encontrados pelo K-Means")
st.caption(
    "Agrupamento não supervisionado com as três métricas RFM (em escala logarítmica e padronizadas). "
    f"O número de grupos é o de maior **coeficiente de silhueta** a partir de k = {MIN_ACTIONABLE_K} "
    "(k = 2 só separaria clientes 'bons' e 'ruins'). A coluna de segmento predominante compara os grupos "
    "com o RFM: quando os dois métodos concordam, a segmentação é mais confiável."
)
clusters = seg.clusters
cluster_table = clusters.assign(
    Clientes=clusters["customers"].map(format_number),
    **{
        "Valor médio (R$)": clusters["avg_monetary"].map(format_brl),
        "Pedidos (média)": clusters["avg_frequency"].map(lambda v: format_number(v, 1)),
        "Dias desde a última compra": clusters["avg_recency"].map(format_number),
        "% da receita": clusters["revenue_pct"].map(format_pct),
        "Segmento RFM predominante": [
            f"{s} ({format_pct(p, 0)})"
            for s, p in zip(clusters["main_segment"], clusters["main_segment_pct"], strict=True)
        ],
    },
)[
    [
        "Clientes",
        "Valor médio (R$)",
        "Pedidos (média)",
        "Dias desde a última compra",
        "% da receita",
        "Segmento RFM predominante",
    ]
]
st.dataframe(cluster_table, width="stretch")

st.caption("Grupo 1 é sempre o de maior valor médio por cliente.")
col_sil, p1, p2, p3 = st.columns(4)
with col_sil:
    st.plotly_chart(charts.silhouette(seg.silhouette_by_k, seg.best_k))
with p1:
    st.plotly_chart(charts.cluster_profile(clusters, "avg_recency", "Dias desde a última compra", "%{y:,.0f} dias"))
with p2:
    st.plotly_chart(charts.cluster_profile(clusters, "avg_frequency", "Pedidos por cliente", "%{y:,.1f} pedidos"))
with p3:
    st.plotly_chart(charts.cluster_profile(clusters, "avg_monetary", "Valor por cliente (R$)", "R$ %{y:,.0f}"))
