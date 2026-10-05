import pandas as pd

from insight_engine.ai import fallback
from insight_engine.analytics.kpis import compute_crypto_kpis, compute_sales_kpis

SECTIONS = ["## Destaques do Período", "## Diagnóstico de Pontos Críticos / Gargalos", "## Plano de Ação Estratégico"]


def test_relatorio_de_vendas_tem_as_tres_secoes_e_os_numeros(small_sales_df):
    kpis = compute_sales_kpis(small_sales_df)
    report = fallback.sales_report(small_sales_df, kpis, "jan/2025")

    for section in SECTIONS:
        assert section in report
    assert "**R$ 1.100,00**" in report
    assert "**4 pedidos**" in report
    assert "sem dado comparativo" in report


def test_margem_baixa_gera_alerta_e_acao(small_sales_df):
    df = small_sales_df.assign(profit=small_sales_df["revenue"] * 0.1)
    report = fallback.sales_report(df, compute_sales_kpis(df), "p")
    assert "abaixo do ideal (10,0%)" in report
    assert "Reavaliar política de custos" in report


def test_categoria_em_queda_aparece_no_diagnostico_e_no_plano(sales_df):
    df = sales_df[sales_df["date"] >= "2025-01-01"]
    report = fallback.sales_report(df, compute_sales_kpis(df), "2025")
    assert "**Moda**" in report.split("## Diagnóstico")[1]
    assert "Revisar mix de produtos e campanhas de **Moda**" in report


def test_relatorio_de_cripto_com_volatilidade_alta():
    df = pd.DataFrame(
        {"date": pd.date_range("2025-01-01", periods=4), "price": [100.0, 120.0, 90.0, 110.0], "volume": 1.0}
    )
    report = fallback.crypto_report(df, compute_crypto_kpis(df), "4 dias")

    for section in SECTIONS:
        assert section in report
    assert "elevada, exigindo cautela redobrada" in report
    assert "Reforçar disciplina de gestão de risco" in report


def test_relatorio_sem_dados():
    report = fallback.no_data_report("jan/2025", "Vendas")
    assert "Não há dados de **Vendas**" in report
    for section in SECTIONS:
        assert section in report


def test_tendencia_usa_o_teste_de_mann_kendall(sales_df):
    df = sales_df[(sales_df["date"] >= "2025-10-02") & (sales_df["date"] <= "2025-12-31")]
    report = fallback.sales_report(df, compute_sales_kpis(df), "p")
    assert "teste de Mann-Kendall: tendência significativa, p = 0,0" in report
    assert "indica **crescimento**" in report


def test_serie_sem_tendencia_e_descrita_como_estabilidade(small_sales_df):
    report = fallback.sales_report(small_sales_df, compute_sales_kpis(small_sales_df), "p")
    assert "indica **estabilidade**" in report
    assert "tendência não significativa" in report


def test_relatorio_sem_custo(small_sales_df):
    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    report = fallback.sales_report(df, compute_sales_kpis(df), "p")
    assert "lucro e margem indisponíveis" in report
    assert "Margem de lucro:** não disponível" in report
    assert "Reavaliar política de custos" not in report
