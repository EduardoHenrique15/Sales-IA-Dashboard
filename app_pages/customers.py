"""
Página: análise de clientes (RFM + K-Means).
"""

import pandas as pd
import streamlit as st

from insight_engine.analytics.customers import MIN_ACTIONABLE_K, pareto_share
from insight_engine.formatting import format_brl, format_number, format_pct
from insight_engine.ui import cached, charts, datasets, filters
from insight_engine.ui.components import chart_card, page_header

ACTIVE_DAYS = 90

dataset = datasets.active_dataset()

if not dataset.has_customers:
    page_header("Análise de clientes", f"Base: {dataset.name}", icon=":material/groups:")
    st.info(
        "A base em uso não tem coluna de cliente. Para usar esta página, envie uma planilha com o "
        "identificador do cliente em **Importar dados**.",
        icon=":material/person_off:",
    )
    st.page_link("app_pages/upload.py", label="Importar dados", icon=":material/upload_file:")
    st.stop()

df_all = dataset.df
dataset_key = datasets.dataset_key(dataset)
regions = filters.segment_filter(df_all, "region", "Regiões", filters.P_REGIONS, dataset_key)
df = df_all[df_all["region"].isin(regions)] if regions else df_all

try:
    seg = cached.segmentation(df)
except ValueError as exc:
    page_header("Análise de clientes", f"Base: {dataset.name}", icon=":material/groups:")
    st.warning(str(exc), icon=":material/warning:")
    st.stop()

customers = seg.customers
page_header(
    "Análise de clientes",
    f":material/event: Situação em {seg.reference_date - pd.Timedelta(days=1):%d/%m/%Y} (última venda da base) · "
    f"Base: {dataset.name}" + (f" · Regiões: {', '.join(regions)}" if regions else ""),
    icon=":material/groups:",
)

# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", format_number(len(customers)), border=True)
active = (customers["recency"] <= ACTIVE_DAYS).mean() * 100
c2.metric(
    "Ativos",
    format_pct(active),
    border=True,
    help=f"Clientes que compraram nos últimos {ACTIVE_DAYS} dias.",
)
repeat = (customers["frequency"] > 1).mean() * 100
c3.metric("Compraram mais de uma vez", format_pct(repeat), border=True)
c4.metric(
    "Receita dos 20% melhores",
    format_pct(pareto_share(customers)),
    border=True,
    help="Concentração da receita (princípio de Pareto).",
)

# ---------- SEGMENTOS RFM ----------
st.header("Segmentos RFM", icon=":material/workspaces:", anchor=False)
st.caption(
    "Cada cliente recebe notas de 1 a 5 (quintis) em **Recência** (dias desde a última compra), "
    "**Frequência** (pedidos) e **Valor** (receita). Regras clássicas de CRM transformam as notas em "
    "segmentos com ação sugerida."
)
col_segments, col_heatmap = st.columns([3, 2])
with col_segments:
    chart_card(
        "Peso de cada segmento em clientes e em receita",
        charts.segments(seg.segments),
        caption="Segmentos com mais receita do que clientes concentram o valor da base.",
    )
with col_heatmap:
    chart_card(
        "Clientes por recência e frequência",
        charts.rfm_heatmap(customers),
        caption="Canto superior direito: compram muito e há pouco tempo (Campeões).",
    )

summary = seg.segments
st.dataframe(
    pd.DataFrame(
        {
            "Segmento": summary.index,
            "Clientes": summary["customers"].map(format_number).to_numpy(),
            "Receita": summary["revenue"].map(lambda v: format_brl(v, 0)).to_numpy(),
            "% da receita": (summary["revenue_pct"] / 100).to_numpy(),
            "Dias desde a última compra (média)": summary["avg_recency"].map(format_number).to_numpy(),
        }
    ),
    column_config={
        "% da receita": st.column_config.ProgressColumn(
            format="percent", min_value=0.0, max_value=float(max(summary["revenue_pct"].max() / 100, 0.01))
        ),
    },
    hide_index=True,
    width="stretch",
)
with st.expander("Ação sugerida para cada segmento", icon=":material/campaign:"):
    st.markdown("\n".join(f"- **{name}:** {action}" for name, action in summary["action"].items()))

export = customers.reset_index()[["customer_id", "segment", "recency", "frequency", "monetary"]]
st.download_button(
    "Baixar lista de clientes com segmento (CSV)",
    data=export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
    file_name="clientes_segmentados.csv",
    mime="text/csv",
    icon=":material/download:",
    help="Pronta para usar em campanhas de CRM/e-mail.",
)

# ---------- K-MEANS ----------
st.header("Grupos encontrados pelo K-Means", icon=":material/scatter_plot:", anchor=False)
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

col_sil, col_recency = st.columns(2)
with col_sil:
    chart_card(
        f"Silhueta por k (escolhido: {seg.best_k})",
        charts.silhouette(seg.silhouette_by_k, seg.best_k),
        caption="Quanto maior, mais separados os grupos.",
    )
with col_recency:
    chart_card("Dias desde a última compra", charts.cluster_profile(clusters, "avg_recency", "%{y:,.0f} dias"))
col_frequency, col_value = st.columns(2)
with col_frequency:
    chart_card("Pedidos por cliente", charts.cluster_profile(clusters, "avg_frequency", "%{y:,.1f} pedidos"))
with col_value:
    chart_card("Valor por cliente (R$)", charts.cluster_profile(clusters, "avg_monetary", "R$ %{y:,.0f}"))
