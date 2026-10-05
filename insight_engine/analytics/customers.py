"""
Segmentação de clientes: RFM (regras de negócio) + K-Means (não supervisionado).

RFM descreve cada cliente por:
  - Recência: dias desde a última compra (menor = melhor);
  - Frequência: número de pedidos;
  - Valor (Monetary): receita total gerada.

1. Segmentos RFM: cada dimensão vira uma nota de 1 a 5 (quintis) e regras
   clássicas de CRM nomeiam os segmentos ("Campeões", "Em risco"...), cada
   um com uma ação sugerida. É a visão fácil de explicar para o negócio.
2. K-Means: agrupa os clientes pelas três métricas (em log e padronizadas),
   com o número de grupos escolhido pelo coeficiente de silhueta (k >= 3). Mostra a
   estrutura que os dados têm, sem regras definidas a priori; comparar os
   grupos com os segmentos RFM valida as duas abordagens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_CUSTOMERS = 50
K_RANGE = range(2, 7)
# k = 2 costuma ter a maior silhueta, mas só separa "bons" e "ruins", o que é
# pouco acionável. A escolha considera k >= 3; k = 2 aparece só para comparação.
MIN_ACTIONABLE_K = 3
RANDOM_STATE = 42

# segmento -> ação sugerida (na ordem em que as regras são avaliadas)
SEGMENT_ACTIONS = {
    "Campeões": "Recompensar e transformar em promotores (programa de indicação, acesso antecipado).",
    "Clientes fiéis": "Oferecer upgrade e venda cruzada; pedir avaliações.",
    "Potenciais fiéis": "Programa de fidelidade e recomendações personalizadas.",
    "Novos clientes": "Onboarding e incentivo à segunda compra nos primeiros 30 dias.",
    "Precisam de atenção": "Ofertas por tempo limitado baseadas no histórico de compra.",
    "Em risco": "Campanha de reativação personalizada antes que deixem de comprar.",
    "Hibernando": "Reengajar com conteúdo e descontos pontuais; avaliar custo de contato.",
    "Perdidos": "Pesquisa de motivo de saída; reativação só se o custo for baixo.",
}


@dataclass(frozen=True)
class CustomerSegmentation:
    # uma linha por cliente: recency, frequency, monetary, r/f/m_score, segment, cluster
    customers: pd.DataFrame
    # resumo por segmento RFM: clientes, % clientes, receita, % receita, recência/frequência médias, ação
    segments: pd.DataFrame
    # resumo por grupo do K-Means, com o segmento RFM predominante
    clusters: pd.DataFrame
    # silhueta para cada k testado
    silhouette_by_k: pd.Series
    best_k: int
    reference_date: pd.Timestamp


def segment_customers(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> CustomerSegmentation:
    """Calcula RFM, segmentos por regras e grupos do K-Means."""
    rfm = rfm_table(df, reference_date)
    if len(rfm) < MIN_CUSTOMERS:
        raise ValueError(f"São necessários pelo menos {MIN_CUSTOMERS} clientes para a segmentação (há {len(rfm)}).")
    reference = reference_date or df["date"].max() + pd.Timedelta(days=1)

    rfm = _add_scores(rfm)
    rfm["segment"] = _rfm_segment(rfm)
    labels, silhouettes, best_k = _kmeans(rfm)
    rfm["cluster"] = labels

    return CustomerSegmentation(
        customers=rfm,
        segments=_summarize_segments(rfm),
        clusters=_summarize_clusters(rfm),
        silhouette_by_k=silhouettes,
        best_k=best_k,
        reference_date=pd.Timestamp(reference),
    )


def rfm_table(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recência (dias), frequência (pedidos) e valor (receita) por cliente."""
    orders = df.dropna(subset=["customer_id"])
    reference = reference_date or orders["date"].max() + pd.Timedelta(days=1)
    rfm = orders.groupby("customer_id").agg(
        last_purchase=("date", "max"), frequency=("date", "size"), monetary=("revenue", "sum")
    )
    rfm["recency"] = (reference - rfm["last_purchase"]).dt.days
    return rfm[["recency", "frequency", "monetary"]]


def pareto_share(customers: pd.DataFrame, top_fraction: float = 0.2) -> float:
    """Participação na receita dos `top_fraction` clientes que mais compram (em %)."""
    monetary = customers["monetary"].sort_values(ascending=False)
    top_n = max(1, int(np.ceil(len(monetary) * top_fraction)))
    return float(monetary.iloc[:top_n].sum() / monetary.sum() * 100)


def _add_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    # rank(method="first") desempata valores iguais, para que os quintis fiquem bem definidos
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return rfm


def _rfm_segment(rfm: pd.DataFrame) -> pd.Series:
    r, f = rfm["r_score"], rfm["f_score"]
    conditions = [
        (r >= 4) & (f >= 4),
        (r >= 3) & (f >= 4),
        (r >= 4) & (f >= 2),
        (r >= 4) & (f == 1),
        (r == 3),
        (r <= 2) & (f >= 3),
        (r == 2),
        (r == 1),
    ]
    return pd.Series(np.select(conditions, list(SEGMENT_ACTIONS), default="Perdidos"), index=rfm.index)


def _kmeans(rfm: pd.DataFrame) -> tuple[np.ndarray, pd.Series, int]:
    features = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    silhouettes = {}
    labels_by_k = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(features)
        silhouettes[k] = silhouette_score(features, labels, sample_size=min(3000, len(features)), random_state=0)
        labels_by_k[k] = labels
    best_k = max((k for k in silhouettes if k >= MIN_ACTIONABLE_K), key=lambda k: silhouettes[k])

    # numera os grupos do maior para o menor valor médio por cliente (Grupo 1 = mais valioso)
    labels = labels_by_k[best_k]
    order = pd.Series(rfm["monetary"].to_numpy()).groupby(labels).mean().sort_values(ascending=False).index
    renumber = {old: new for new, old in enumerate(order, start=1)}
    return np.array([renumber[label] for label in labels]), pd.Series(silhouettes, name="silhouette"), best_k


def _summarize_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("segment").agg(
        customers=("frequency", "size"),
        revenue=("monetary", "sum"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
    )
    summary["customers_pct"] = summary["customers"] / summary["customers"].sum() * 100
    summary["revenue_pct"] = summary["revenue"] / summary["revenue"].sum() * 100
    summary["action"] = summary.index.map(SEGMENT_ACTIONS)
    ordered = [s for s in SEGMENT_ACTIONS if s in summary.index]
    return summary.loc[ordered]


def _summarize_clusters(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = rfm.groupby("cluster").agg(
        customers=("frequency", "size"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    summary["revenue_pct"] = summary["revenue"] / summary["revenue"].sum() * 100
    dominant = rfm.groupby("cluster")["segment"].agg(lambda s: s.value_counts().index[0])
    share = rfm.groupby("cluster")["segment"].agg(lambda s: s.value_counts(normalize=True).iloc[0] * 100)
    summary["main_segment"] = dominant
    summary["main_segment_pct"] = share
    summary.index = [f"Grupo {i}" for i in summary.index]
    return summary
