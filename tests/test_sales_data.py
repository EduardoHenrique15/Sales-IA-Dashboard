import pandas as pd

from insight_engine.data import sales
from insight_engine.data.schemas import SALES_SCHEMA


def test_base_cobre_o_periodo_fixo(sales_df):
    assert sales_df["date"].min() == pd.Timestamp(sales.SALES_START_DATE)
    assert sales_df["date"].max() == pd.Timestamp(sales.SALES_END_DATE)
    assert sales_df["date"].is_monotonic_increasing


def test_base_e_deterministica(sales_df):
    pd.testing.assert_frame_equal(sales.load_sales_data(), sales_df)


def test_base_respeita_o_esquema(sales_df):
    # order_id é opcional: na base sintética, cada linha já é um pedido
    assert list(sales_df.columns) == [c for c in SALES_SCHEMA.columns if c != "order_id"]
    assert (sales_df[["units", "unit_price", "revenue", "cost"]] >= 0).all().all()
    assert set(sales_df["category"]) == set(sales.CATEGORIES)
    assert set(sales_df["region"]) == set(sales.REGIONS)


def test_lucro_e_receita_menos_custo(sales_df):
    pd.testing.assert_series_equal(
        sales_df["profit"], (sales_df["revenue"] - sales_df["cost"]).round(2), check_names=False
    )


def test_categoria_em_declinio_perde_participacao_no_fim(sales_df):
    """A base tem uma categoria em queda proposital para o diagnóstico da IA."""
    share = sales_df.groupby([sales_df["date"].dt.year, "category"])["revenue"].sum()
    share = share / share.groupby(level=0).transform("sum")
    declining = sales.DECLINING_CATEGORY
    assert share.loc[(2025, declining)] < share.loc[(2023, declining)]


def test_clientes_recorrentes_com_cauda_longa(sales_df):
    orders_per_customer = sales_df.groupby("customer_id").size()
    assert sales_df["customer_id"].notna().all()
    assert 2000 < len(orders_per_customer) < sales.N_CUSTOMERS
    assert orders_per_customer.median() <= 5  # a maioria compra pouco...
    assert orders_per_customer.max() >= 50  # ...e poucos compram muito


def test_cada_cliente_pertence_a_uma_regiao(sales_df):
    assert sales_df.groupby("customer_id")["region"].nunique().max() == 1


def test_anomalias_plantadas_estao_na_base(sales_df):
    orders = sales_df.groupby("date").size()
    typical = orders.median()
    for day, kind, _ in sales.PLANTED_ANOMALIES:
        count = orders[pd.Timestamp(day)]
        assert count > 2 * typical if kind == "pico" else count < typical / 3


def test_periodo_padrao_mantem_os_numeros_conhecidos(sales_df):
    """Clientes e anomalias não alteram os pedidos do último trimestre."""
    q4 = sales_df[(sales_df["date"] >= "2025-10-02") & (sales_df["date"] <= "2025-12-31")]
    assert round(q4["revenue"].sum(), 2) == 2838404.99
