import numpy as np
import pandas as pd
import pytest

from insight_engine.analytics.customers import (
    MIN_ACTIONABLE_K,
    SEGMENT_ACTIONS,
    pareto_share,
    rfm_table,
    segment_customers,
)


def test_rfm_calculado_a_mao():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "a", "b"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-10", "2025-01-05"]),
            "revenue": [10.0, 30.0, 5.0],
        }
    )
    rfm = rfm_table(df)  # referência: dia seguinte à última venda (11/01)
    assert rfm.loc["a"].tolist() == [1, 2, 40.0]
    assert rfm.loc["b"].tolist() == [6, 1, 5.0]


def test_ignora_pedidos_sem_cliente():
    df = pd.DataFrame(
        {"customer_id": ["a", None], "date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "revenue": [1.0, 2.0]}
    )
    assert list(rfm_table(df).index) == ["a"]


def test_pareto():
    customers = pd.DataFrame({"monetary": [80.0, 10.0, 5.0, 3.0, 2.0]})
    assert pareto_share(customers, top_fraction=0.2) == pytest.approx(80)


@pytest.fixture(scope="module")
def segmentation(sales_df):
    return segment_customers(sales_df)


def test_todo_cliente_tem_segmento_e_grupo(segmentation, sales_df):
    customers = segmentation.customers
    assert len(customers) == sales_df["customer_id"].nunique()
    assert customers["segment"].isin(SEGMENT_ACTIONS).all()
    assert customers["cluster"].between(1, segmentation.best_k).all()
    assert customers[["r_score", "f_score", "m_score"]].isin(range(1, 6)).all().all()


def test_resumos_somam_100(segmentation):
    assert segmentation.segments["customers_pct"].sum() == pytest.approx(100)
    assert segmentation.segments["revenue_pct"].sum() == pytest.approx(100)
    assert segmentation.clusters["revenue_pct"].sum() == pytest.approx(100)


def test_k_escolhido_pela_silhueta_a_partir_de_3(segmentation):
    candidates = segmentation.silhouette_by_k[segmentation.silhouette_by_k.index >= MIN_ACTIONABLE_K]
    assert segmentation.best_k == candidates.idxmax()


def test_campeoes_compram_mais_e_mais_recentemente_que_perdidos(segmentation):
    segments = segmentation.segments
    assert segments.loc["Campeões", "avg_frequency"] > segments.loc["Perdidos", "avg_frequency"]
    assert segments.loc["Campeões", "avg_recency"] < segments.loc["Perdidos", "avg_recency"]


def test_grupo_1_e_o_de_maior_valor(segmentation):
    assert segmentation.clusters["avg_monetary"].idxmax() == "Grupo 1"


def test_poucos_clientes():
    df = pd.DataFrame(
        {"customer_id": ["a", "b"], "date": pd.to_datetime(["2025-01-01", "2025-01-02"]), "revenue": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="pelo menos"):
        segment_customers(df)


def test_recencia_usa_a_data_de_referencia_e_ignora_a_hora():
    df = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "date": pd.to_datetime(["2025-01-10 23:00", "2025-01-05 08:00"]),
            "revenue": [10.0, 5.0],
        }
    )
    # referência do subconjunto filtrado = dia seguinte ao fim da base inteira, não do subconjunto
    rfm = rfm_table(df, reference_date=pd.Timestamp("2025-01-31"))
    assert rfm["recency"].tolist() == [21, 26]


def test_segmentacao_sem_kmeans():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "customer_id": rng.choice([f"c{i}" for i in range(60)], 400),
            "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 200, 400), unit="D"),
            "revenue": rng.gamma(2, 50, 400),
        }
    )
    result = segment_customers(df, with_clusters=False)
    assert result.best_k == 0
    assert result.customers["segment"].notna().all()
