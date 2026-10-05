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
    assert list(sales_df.columns) == list(SALES_SCHEMA.columns)
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
