import pandas as pd
import pytest

from insight_engine.ai.context import build_crypto_facts, build_sales_facts
from insight_engine.analytics.kpis import compute_crypto_kpis, compute_sales_kpis
from insight_engine.analytics.periods import SalesFilters, filter_sales
from tests.helpers import as_date


@pytest.fixture(scope="module")
def q3_2024(sales_df):
    filters = SalesFilters(as_date("2024-07-01"), as_date("2024-09-30"))
    period, previous = filter_sales(sales_df, filters)
    return build_sales_facts(sales_df, period, previous, compute_sales_kpis(period, previous), filters.period_label)


def test_fatos_reunem_todas_as_analises(q3_2024):
    text = "\n".join(q3_2024.to_lines())
    assert "Receita total: R$ 1.742.745,97" in text
    assert "Crescimento da receita vs período anterior de mesma duração: -3,9%" in text
    assert "Receita por categoria: Eletrônicos" in text
    assert "Mann-Kendall" in text
    assert "Moda (-20%)" in text  # categoria em queda
    assert "efeito mix -R$ 52.025,13" in text
    assert "07/08/2024 (queda" in text  # anomalia plantada dentro do período
    assert "Previsão de receita para os 30 dias após 31/12/2025" in text
    assert "Campeões" in text


def test_anomalias_ficam_restritas_ao_periodo(q3_2024):
    assert all(pd.Timestamp("2024-07-01") <= day <= pd.Timestamp("2024-09-30") for day in q3_2024.anomalies.index)


def test_base_sem_custo_nem_cliente(small_sales_df):
    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    facts = build_sales_facts(df, df, df.iloc[0:0], compute_sales_kpis(df), "p", has_customers=False)
    text = "\n".join(facts.to_lines())
    assert "Lucro e margem: não disponíveis" in text
    assert facts.segmentation is None
    assert facts.forecast is None  # histórico curto demais
    assert facts.bridge is None  # sem período anterior


def test_fatos_de_cripto():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "price": range(100, 110), "volume": 1.0})
    facts = build_crypto_facts(df, compute_crypto_kpis(df), "10 dias", "Bitcoin (BTC)")
    text = "\n".join(facts.to_lines())
    assert "Preço atual: US$ 109,00" in text
    assert "Tendência do preço (Mann-Kendall): alta" in text
    assert facts.drawdown_pct == 0
