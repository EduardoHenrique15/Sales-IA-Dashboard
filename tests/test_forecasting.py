import numpy as np
import pandas as pd
import pytest

from insight_engine.analytics.forecasting import (
    BASELINE,
    MIN_TRAIN_DAYS,
    InsufficientDataError,
    daily_revenue,
    forecast_revenue,
)


def synthetic_series(days: int = 400, seed: int = 0) -> pd.Series:
    """Tendência + sazonalidade semanal + ruído: uma série que os modelos devem prever bem."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=days, freq="D")
    weekly = np.where(index.dayofweek >= 5, -30.0, 10.0)
    values = 200 + np.arange(days) * 0.2 + weekly + rng.normal(0, 5, days)
    return pd.Series(values, index=index)


def test_preenche_dias_sem_venda_com_zero():
    df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-03"]), "revenue": [10.0, 5.0]})
    assert daily_revenue(df).tolist() == [10.0, 0.0, 5.0]


def test_previsao_tem_o_horizonte_e_intervalos_ordenados():
    result = forecast_revenue(synthetic_series(), horizon=30)

    fc = result.forecast
    assert len(fc) == 30
    assert fc.index[0] == synthetic_series().index[-1] + pd.Timedelta(days=1)
    assert (fc["lower_95"] <= fc["lower_80"]).all()
    assert (fc["lower_80"] <= fc["yhat"]).all()
    assert (fc["yhat"] <= fc["upper_80"]).all()
    assert (fc["upper_80"] <= fc["upper_95"]).all()
    assert (fc["lower_95"] >= 0).all()


def test_escolhe_o_modelo_de_menor_erro_e_supera_o_baseline():
    result = forecast_revenue(synthetic_series(), horizon=30)
    assert result.best_model == result.scores["wape"].idxmin()
    assert result.best_model != BASELINE
    assert result.scores.loc[result.best_model, "improvement_vs_baseline"] > 0


def test_erro_do_total_e_menor_que_o_erro_diario(sales_df):
    """Erros diários se compensam no total do período."""
    result = forecast_revenue(daily_revenue(sales_df), horizon=30)
    assert result.total_error_pct < result.scores.loc[result.best_model, "wape"]
    assert set(result.scores.columns) >= {"wape", "rmse", "bias", "total_error", "improvement_vs_baseline"}


def test_numero_de_janelas_se_adapta_ao_historico():
    result = forecast_revenue(synthetic_series(days=MIN_TRAIN_DAYS + 60), horizon=30, max_folds=4)
    assert result.n_folds == 2


def test_historico_insuficiente():
    with pytest.raises(InsufficientDataError, match="pelo menos"):
        forecast_revenue(synthetic_series(days=MIN_TRAIN_DAYS + 10), horizon=30)


def test_erro_do_total_ignora_janelas_sem_venda():
    from insight_engine.analytics.forecasting import _total_error

    assert _total_error(np.array([5.0, 3.0]), np.array([0.0, 30.0])) == pytest.approx(10.0)
    assert np.isnan(_total_error(np.array([5.0]), np.array([0.0])))
