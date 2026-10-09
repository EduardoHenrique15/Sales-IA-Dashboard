import pytest

from insight_engine.analytics.kpis import SalesKPIs, compute_sales_kpis


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


def test_sem_custo_lucro_e_margem_ficam_indisponiveis(small_sales_df):
    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    kpis = compute_sales_kpis(df)
    assert kpis.total_profit is None
    assert kpis.margin_pct is None
    assert kpis.total_revenue == 1100  # o restante continua calculado


def test_custo_parcial_tambem_fica_indisponivel(small_sales_df):
    df = small_sales_df.copy()
    df.loc[0, ["cost", "profit"]] = float("nan")
    assert compute_sales_kpis(df).total_profit is None


def test_periodo_sem_vendas_apos_periodo_com_vendas_e_queda_de_100(small_sales_df):
    kpis = compute_sales_kpis(small_sales_df.iloc[0:0], small_sales_df)
    assert kpis.revenue_growth_pct == -100.0
    assert kpis.total_revenue == 0
