from insight_engine.ai.prompts import build_prompt
from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs, compute_sales_kpis


def test_prompt_de_vendas_traz_os_numeros_formatados(small_sales_df):
    previous = small_sales_df.assign(revenue=small_sales_df["revenue"] / 2)
    prompt = build_prompt(compute_sales_kpis(small_sales_df, previous), "jan/2025", "Vendas")

    assert "Receita total: R$ 1.100,00" in prompt
    assert "Margem de lucro: 36,4%" in prompt
    assert "Categoria líder: Eletrônicos (R$ 800,00)" in prompt
    assert "Crescimento da receita vs período anterior: +100,0%" in prompt
    assert '"jan/2025"' in prompt
    assert "não invente" in prompt


def test_prompt_sem_periodo_anterior():
    assert "não disponível" in build_prompt(SalesKPIs(), "p", "Vendas")


def test_prompt_de_cripto():
    kpis = CryptoKPIs(current_price=50_000, period_change_pct=-2.5, volatility_pct=3.21)
    prompt = build_prompt(kpis, "30 dias", "Criptomoedas")

    assert "Preço atual: US$ 50.000,00" in prompt
    assert "Variação no período: -2,50%" in prompt
    assert "Volatilidade diária (desvio padrão dos retornos): 3,21%" in prompt
    assert "Receita" not in prompt
