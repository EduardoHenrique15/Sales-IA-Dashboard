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
    fig = charts.revenue_over_time(df)
    formats = [stop.value for stop in fig.layout.xaxis.tickformatstops]
    assert formats == ["%d/%m", "%m/%Y", "%Y"]
    assert fig.layout.xaxis.hoverformat == "%d/%m/%Y"


def test_receita_por_dia_tem_media_movel_de_7_dias():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "revenue": range(10), "profit": 0.0})
    fig = charts.revenue_over_time(df, show_profit=False)
    daily, average = fig.data
    assert list(daily.y) == list(range(10)) and daily.showlegend is False
    assert average.y[-1] == sum(range(3, 10)) / 7  # média dos últimos 7 dias
    assert average.name == "Receita"


def test_receita_agrupada_por_mes():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-05", "2025-01-20", "2025-02-03"]),
            "revenue": [10.0, 20.0, 5.0],
            "profit": 1.0,
        }
    )
    fig = charts.revenue_over_time(df, granularity="M")
    assert [trace.name for trace in fig.data] == ["Receita", "Lucro"]
    assert list(fig.data[0].y) == [30.0, 5.0]
    assert fig.layout.xaxis.hoverformat == "%m/%Y"


def test_graficos_nao_fixam_o_template_escuro():
    """Sem template fixo, o Streamlit aplica o tema claro ou escuro do visitante."""
    fig = charts.revenue_by_region(pd.Series({"Sul": 10.0}))
    assert fig.layout.template.layout.paper_bgcolor is None
    assert fig.layout.title.text is None  # o título fica no cartão, fora da figura


def test_paleta_segue_o_tema(monkeypatch):
    from types import SimpleNamespace

    from insight_engine.ui import theme

    monkeypatch.setattr(theme.st, "context", SimpleNamespace(theme=SimpleNamespace(type="light")))
    assert theme.palette() is theme.LIGHT
    fig = charts.revenue_by_region(pd.Series({"Sul": 10.0}))
    assert fig.data[0].marker.color == theme.LIGHT.series_1

    monkeypatch.setattr(theme.st, "context", SimpleNamespace(theme=SimpleNamespace(type="dark")))
    assert theme.palette() is theme.DARK


def test_mapa_de_calor_rfm_conta_clientes_por_nota():
    customers = pd.DataFrame({"r_score": [5, 5, 1], "f_score": [5, 5, 2]})
    fig = charts.rfm_heatmap(customers)
    z = fig.data[0].z
    assert z[4][4] == 2 and z[1][0] == 1  # linhas = frequência, colunas = recência
    assert sum(sum(row) for row in z) == 3
