import pandas as pd

from insight_engine.analytics.periods import SalesFilters, filter_sales, previous_period_df
from tests.helpers import as_date


def test_periodo_anterior_tem_a_mesma_duracao(sales_df):
    previous = previous_period_df(sales_df, as_date("2025-03-01"), as_date("2025-03-31"))
    assert previous["date"].min() == pd.Timestamp("2025-01-29")
    assert previous["date"].max() == pd.Timestamp("2025-02-28")
    assert previous["date"].dt.date.nunique() == 31


def test_filtros_valem_para_os_dois_periodos(sales_df):
    filters = SalesFilters(as_date("2025-06-01"), as_date("2025-06-30"), ["Moda"], ["Sul", "Norte"])
    current, previous = filter_sales(sales_df, filters)

    for df in (current, previous):
        assert set(df["category"]) == {"Moda"}
        assert set(df["region"]) <= {"Sul", "Norte"}
    assert current["date"].min() >= pd.Timestamp("2025-06-01")
    assert current["date"].max() <= pd.Timestamp("2025-06-30")
    assert previous["date"].max() < pd.Timestamp("2025-06-01")


def test_listas_vazias_significam_sem_filtro(sales_df):
    filters = SalesFilters(as_date("2025-01-01"), as_date("2025-12-31"))
    current, _ = filter_sales(sales_df, filters)
    assert set(current["category"]) == set(sales_df["category"])


def test_rotulo_do_periodo():
    assert SalesFilters(as_date("2025-10-02"), as_date("2025-12-31")).period_label == "02/10/2025 a 31/12/2025"
