import pandas as pd

from insight_engine.analytics.periods import (
    SalesFilters,
    comparison_is_complete,
    comparison_window,
    filter_sales,
    previous_period_df,
    window_is_covered,
)
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


def test_comparacao_com_o_mesmo_periodo_do_ano_anterior(sales_df):
    filters = SalesFilters(as_date("2025-12-01"), as_date("2025-12-31"), comparison="ano_anterior")
    current, previous = filter_sales(sales_df, filters)
    assert previous["date"].min() == pd.Timestamp("2024-12-01")
    assert previous["date"].max() == pd.Timestamp("2024-12-31")
    assert filters.comparison_label == "mesmo período do ano anterior"


def test_comparacao_padrao_e_o_periodo_anterior():
    assert SalesFilters(as_date("2025-01-01"), as_date("2025-01-31")).comparison_label == (
        "período anterior de mesma duração"
    )


def test_datas_com_hora_entram_no_dia_inteiro():
    # bases reais (como a Olist) têm data e hora: o último dia do período anterior não pode sumir
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-04 09:00", "2025-01-31 18:30", "2025-02-01 10:00", "2025-02-28 23:59"]),
            "revenue": [1.0, 2.0, 3.0, 4.0],
            "category": "A",
            "region": "Sul",
        }
    )
    current, previous = filter_sales(df, SalesFilters(as_date("2025-02-01"), as_date("2025-02-28")))
    assert current["revenue"].tolist() == [3.0, 4.0]
    assert previous["revenue"].tolist() == [1.0, 2.0]


def test_periodo_anterior_fora_da_base_nao_e_comparado(sales_df):
    # o período anterior começaria antes da base: comparar com um pedaço inflaria o crescimento
    filters = SalesFilters(sales_df["date"].min().date(), as_date("2024-12-31"))
    assert not comparison_is_complete(sales_df, filters)
    _, previous = filter_sales(sales_df, filters)
    assert previous.empty


def test_janela_de_comparacao():
    assert comparison_window(as_date("2025-03-01"), as_date("2025-03-31")) == (
        pd.Timestamp("2025-01-29"),
        pd.Timestamp("2025-02-28"),
    )
    assert comparison_window(as_date("2025-03-01"), as_date("2025-03-31"), "ano_anterior") == (
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-03-31"),
    )


def test_folga_para_base_que_comeca_alguns_dias_depois():
    # base que começa em 05/01 (como a Olist): o ano anterior perde 4 de 233 dias e ainda é comparável
    assert window_is_covered("2017-01-01", "2017-08-21", pd.Timestamp("2017-01-05 10:00"))
    # um ano inteiro que começa 4 meses antes da base, não
    assert not window_is_covered("2016-09-01", "2017-08-31", pd.Timestamp("2017-01-05"))
