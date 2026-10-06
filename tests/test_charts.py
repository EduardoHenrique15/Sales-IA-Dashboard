import pandas as pd

from insight_engine.ui import charts


def test_categorias_em_barras_com_participacao():
    fig = charts.revenue_by_category(pd.Series({"Moda": 30.0, "Beleza": 20.0, "Alimentos": 50.0}))
    assert list(fig.data[0].y) == ["Beleza", "Moda", "Alimentos"]  # do menor para o maior
    assert fig.data[0].text[-1] == "R$ 50 (50,0%)"


def test_barras_de_categoria_nominal_tem_uma_cor_so():
    fig = charts.revenue_by_region(pd.Series({"Sul": 2_500_000.0, "Norte": 900.0}))
    assert isinstance(fig.data[0].marker.color, str)
    assert list(fig.data[0].text) == ["R$ 900", "R$ 2,50 mi"]


def test_datas_do_eixo_em_formato_brasileiro():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=3), "revenue": 1.0, "profit": 0.5})
    fig = charts.revenue_profit_over_time(df)
    formats = [stop.value for stop in fig.layout.xaxis.tickformatstops]
    assert formats == ["%d/%m", "%m/%Y", "%Y"]
    assert "%d/%m/%Y" in fig.data[0].hovertemplate
