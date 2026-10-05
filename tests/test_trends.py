import pandas as pd
import pytest

from insight_engine.analytics.trends import Trend, declining_categories, fit_trend


def test_tendencia_de_alta_perfeita():
    trend = fit_trend(pd.Series([10.0, 20.0, 30.0, 40.0]))
    # inclinação de 10/dia * 4 dias sobre a média de 25 = +160% no período
    assert trend.slope_pct == pytest.approx(160)
    assert trend.r2 == pytest.approx(1)
    assert trend.confidence == "alta"


def test_serie_constante_nao_tem_tendencia():
    assert fit_trend(pd.Series([5.0, 5.0, 5.0])).slope_pct == pytest.approx(0)


def test_serie_curta_demais():
    assert fit_trend(pd.Series([1.0])) == Trend(0.0, 0.0)


@pytest.mark.parametrize(("r2", "expected"), [(0.8, "alta"), (0.3, "moderada"), (0.1, "baixa")])
def test_confianca_pelo_r2(r2, expected):
    assert Trend(0.0, r2).confidence == expected


def test_detecta_categoria_em_queda():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-10", "2025-01-10"]),
            "category": ["Moda", "Beleza", "Moda", "Beleza"],
            "revenue": [100.0, 100.0, 50.0, 95.0],  # Moda -50%, Beleza -5%
        }
    )
    assert declining_categories(df) == [("Moda", -50.0)]


def test_sem_segunda_metade_nao_ha_queda():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "category": ["Moda"], "revenue": [10.0]})
    assert declining_categories(df) == []
