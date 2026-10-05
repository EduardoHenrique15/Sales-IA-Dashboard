import pandas as pd
import pytest

from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs, compute_crypto_kpis, compute_sales_kpis


class TestSalesKPIs:
    def test_valores_calculados_a_mao(self, small_sales_df):
        kpis = compute_sales_kpis(small_sales_df)

        assert kpis.total_revenue == 1100
        assert kpis.total_profit == 400
        assert kpis.margin_pct == pytest.approx(400 / 1100 * 100)
        assert kpis.total_units == 7
        assert kpis.n_orders == 4
        assert kpis.avg_ticket == 275
        assert kpis.top_category == "Eletrônicos"
        assert kpis.top_category_revenue == 800
        assert kpis.top_region == "Sudeste"
        assert kpis.revenue_by_category.tolist() == [800, 200, 100]  # ordem decrescente

    def test_crescimento_vs_periodo_anterior(self, small_sales_df):
        previous = small_sales_df.assign(revenue=small_sales_df["revenue"] / 2)  # receita de 550
        assert compute_sales_kpis(small_sales_df, previous).revenue_growth_pct == pytest.approx(100)

    @pytest.mark.parametrize("previous", [None, "vazio", "receita_zero"])
    def test_sem_periodo_anterior_comparavel(self, small_sales_df, previous):
        if previous == "vazio":
            previous = small_sales_df.iloc[0:0]
        elif previous == "receita_zero":
            previous = small_sales_df.assign(revenue=0.0)
        assert compute_sales_kpis(small_sales_df, previous).revenue_growth_pct is None

    def test_dados_vazios(self, small_sales_df):
        kpis = compute_sales_kpis(small_sales_df.iloc[0:0])
        assert kpis.total_revenue == 0
        assert kpis.top_category == "N/A"
        assert kpis.revenue_by_category.empty

    def test_e_imutavel(self):
        with pytest.raises(AttributeError):
            SalesKPIs().total_revenue = 1  # type: ignore[misc]


class TestCryptoKPIs:
    def test_valores_calculados_a_mao(self):
        df = pd.DataFrame({"price": [100.0, 110.0, 99.0], "volume": [10.0, 20.0, 30.0]})
        kpis = compute_crypto_kpis(df)

        assert kpis.current_price == 99
        assert kpis.period_change_pct == pytest.approx(-1)
        assert kpis.max_price == 110
        assert kpis.min_price == 99
        assert kpis.avg_volume == 20
        # retornos diários: +10% e -10%; desvio padrão amostral = 14,14%
        assert kpis.volatility_pct == pytest.approx(14.1421, rel=1e-4)

    def test_dados_vazios(self):
        assert compute_crypto_kpis(pd.DataFrame()) == CryptoKPIs()
