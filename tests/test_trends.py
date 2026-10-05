import pandas as pd
import pytest

from insight_engine.analytics.trends import MannKendall, Trend, declining_categories, fit_trend, mann_kendall


def test_tendencia_de_alta_perfeita():
    trend = fit_trend(pd.Series([10.0, 20.0, 30.0, 40.0]))
    # inclinação de 10/dia * 4 dias sobre a média de 25 = +160% no período
    assert trend.slope_pct == pytest.approx(160)
    assert trend.r2 == pytest.approx(1)


def test_serie_constante_nao_tem_tendencia():
    assert fit_trend(pd.Series([5.0, 5.0, 5.0])).slope_pct == pytest.approx(0)


def test_serie_curta_demais():
    assert fit_trend(pd.Series([1.0])) == Trend(0.0, 0.0)


class TestMannKendall:
    def test_alta_significativa(self):
        mk = mann_kendall(pd.Series(range(30), dtype=float))
        assert mk.tau == pytest.approx(1)
        assert mk.significant
        assert mk.direction == "alta"

    def test_queda_significativa_mesmo_com_ruido(self):
        values = [100 - i + (5 if i % 2 else -5) for i in range(40)]
        mk = mann_kendall(pd.Series(values, dtype=float))
        assert mk.direction == "queda"

    def test_serie_sem_tendencia(self):
        values = [10, 12, 9, 11, 10, 12, 9, 11, 10, 12, 9, 11]
        mk = mann_kendall(pd.Series(values, dtype=float))
        assert not mk.significant
        assert mk.direction == "sem tendência"

    @pytest.mark.parametrize("values", [[1.0, 2.0], [5.0] * 10])
    def test_serie_curta_ou_constante(self, values):
        assert mann_kendall(pd.Series(values)) == MannKendall(0.0, 1.0)


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
