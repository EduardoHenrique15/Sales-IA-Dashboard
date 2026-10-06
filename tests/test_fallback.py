import pandas as pd
import pytest

from insight_engine.ai import fallback
from insight_engine.ai.context import build_crypto_facts, build_sales_facts
from insight_engine.analytics.kpis import compute_crypto_kpis, compute_sales_kpis
from insight_engine.analytics.periods import SalesFilters, filter_sales
from tests.helpers import as_date


def facts_for(sales_df, start, end):
    filters = SalesFilters(as_date(start), as_date(end))
    period, previous = filter_sales(sales_df, filters)
    return build_sales_facts(sales_df, period, previous, compute_sales_kpis(period, previous), filters.period_label)


@pytest.fixture(scope="module")
def q3_2024_report(sales_df):
    return fallback.sales_report(facts_for(sales_df, "2024-07-01", "2024-09-30"))


def test_relatorio_tem_a_estrutura_completa(q3_2024_report):
    assert "R$ 1.742.745,97" in q3_2024_report.headline
    assert 1 <= len(q3_2024_report.highlights) <= 4
    assert 1 <= len(q3_2024_report.risks) <= 3
    assert 1 <= len(q3_2024_report.actions) <= 5


def test_usa_as_analises_da_parte_4(q3_2024_report):
    risks = " ".join(r.title for r in q3_2024_report.risks)
    actions = " ".join(a.action for a in q3_2024_report.actions)
    assert "Queda em Moda" in risks
    assert "Revisar mix de produtos e campanhas de Moda" in actions
    assert "07/08/2024" in actions  # investigar o dia atípico
    assert any("volume" in h and "mix" in h for h in q3_2024_report.highlights)
    assert any("Mann-Kendall" in h for h in q3_2024_report.highlights)


def test_reativacao_de_clientes_em_risco(q3_2024_report):
    assert any("Em risco" in a.action for a in q3_2024_report.actions)


def test_margem_baixa_vira_risco_e_acao(sales_df):
    df = sales_df[sales_df["date"] >= "2025-10-01"]
    df = df.assign(profit=df["revenue"] * 0.1)
    facts = build_sales_facts(df, df, df.iloc[0:0], compute_sales_kpis(df), "p")
    report = fallback.sales_report(facts)
    assert any(r.title == "Margem abaixo do saudável" for r in report.risks)


def test_sem_custo_nao_fala_de_margem(small_sales_df):
    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    facts = build_sales_facts(df, df, df.iloc[0:0], compute_sales_kpis(df), "p", has_customers=False)
    report = fallback.sales_report(facts)
    assert not any("margem" in h.lower() for h in report.highlights)
    assert "lucro" not in report.highlights[0]


def test_relatorio_de_cripto_com_volatilidade_alta():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=4), "price": [100.0, 120.0, 90.0, 110.0]})
    df["volume"] = 1.0
    report = fallback.crypto_report(build_crypto_facts(df, compute_crypto_kpis(df), "4 dias", "BTC"))
    assert report.risks[0].title == "Volatilidade elevada"
    assert report.actions[0].priority == "alta"


def test_relatorio_sem_dados():
    report = fallback.no_data_report("jan/2025", "Vendas")
    assert "Não há dados de Vendas" in report.headline
