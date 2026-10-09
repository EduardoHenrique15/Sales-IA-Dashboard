"""
Previsão da receita diária com seleção de modelo por backtesting.

Três modelos concorrem:
  - Ingênuo sazonal (baseline): repete a última semana observada;
  - Holt-Winters: suavização exponencial com tendência amortecida e
    sazonalidade semanal;
  - Regressão com calendário: tendência linear + dia da semana + mês
    (captura picos anuais, como novembro/dezembro).

Cada modelo é avaliado em validação temporal (rolling origin): o modelo é
treinado só com o passado e prevê os `horizon` dias seguintes, em várias
janelas. Vence o menor WAPE. Os intervalos de previsão vêm dos erros
reais observados no backtesting (intervalos empíricos), em vez de supor
erros normais.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from insight_engine.analytics.anomalies import daily_series

SEASON = 7  # sazonalidade semanal
MIN_TRAIN_DAYS = 8 * SEASON

BASELINE = "Ingênuo sazonal (baseline)"
HOLT_WINTERS = "Holt-Winters"
CALENDAR = "Regressão com calendário"


class InsufficientDataError(ValueError):
    """Histórico curto demais para treinar e validar os modelos."""


@dataclass(frozen=True)
class ForecastResult:
    history: pd.Series
    # colunas: yhat, lower_80, upper_80, lower_95, upper_95 (índice = datas futuras)
    forecast: pd.DataFrame
    best_model: str
    # métricas do backtesting por modelo: wape, rmse, bias, total_error, improvement_vs_baseline
    scores: pd.DataFrame
    horizon: int
    n_folds: int

    @property
    def total_error_pct(self) -> float:
        """Erro médio do modelo escolhido ao prever o TOTAL do horizonte (em %).

        É a incerteza certa para o total previsto: somar os limites diários
        dos intervalos superestimaria a incerteza do período inteiro.
        """
        return float(self.scores.loc[self.best_model, "total_error"])


def daily_revenue(df: pd.DataFrame) -> pd.Series:
    """Receita por dia, incluindo dias sem venda (receita zero)."""
    return daily_series(df, "revenue")


def forecast_revenue(series: pd.Series, horizon: int = 30, max_folds: int = 4) -> ForecastResult:
    """Escolhe o melhor modelo por backtesting e prevê os próximos `horizon` dias."""
    series = series.astype(float).asfreq("D", fill_value=0.0)
    n_folds = min(max_folds, (len(series) - MIN_TRAIN_DAYS) // horizon)
    if n_folds < 1:
        needed = MIN_TRAIN_DAYS + horizon
        raise InsufficientDataError(
            f"São necessários pelo menos {needed} dias de histórico para prever {horizon} dias "
            f"(a base tem {len(series)})."
        )

    errors = {name: _backtest(series, model, horizon, n_folds) for name, model in MODELS.items()}
    scores = _score(series, errors, horizon, n_folds)
    best = str(scores["wape"].idxmin())

    yhat = np.clip(MODELS[best](series, horizon), 0, None)
    residuals = errors[best]
    future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast = pd.DataFrame({"yhat": yhat}, index=future)
    for level, (low_q, high_q) in {"80": (0.10, 0.90), "95": (0.025, 0.975)}.items():
        forecast[f"lower_{level}"] = np.clip(yhat + np.quantile(residuals, low_q), 0, None)
        forecast[f"upper_{level}"] = yhat + np.quantile(residuals, high_q)

    return ForecastResult(series, forecast, best, scores, horizon, n_folds)


# ------------------------------------------------------------------
# Modelos: recebem o histórico e devolvem `horizon` valores previstos
# ------------------------------------------------------------------
def _seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    last_season = train.to_numpy()[-SEASON:]
    return np.resize(last_season, horizon)


def _holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # avisos de convergência do otimizador
        model = ExponentialSmoothing(
            train.to_numpy(),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=SEASON,
            initialization_method="estimated",
        ).fit()
    return np.asarray(model.forecast(horizon))


def _calendar_regression(train: pd.Series, horizon: int) -> np.ndarray:
    future = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    x_train = _calendar_features(train.index, origin=train.index[0])
    x_future = _calendar_features(future, origin=train.index[0])
    model = Ridge(alpha=1.0).fit(x_train, train.to_numpy())
    return model.predict(x_future)


def _calendar_features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    trend = ((index - origin).days.to_numpy() / 365.0).reshape(-1, 1)
    weekday = np.eye(7)[index.dayofweek]
    month = np.eye(12)[index.month - 1]
    return np.hstack([trend, weekday, month])


MODELS: dict[str, Callable[[pd.Series, int], np.ndarray]] = {
    BASELINE: _seasonal_naive,
    HOLT_WINTERS: _holt_winters,
    CALENDAR: _calendar_regression,
}


# ------------------------------------------------------------------
# Backtesting
# ------------------------------------------------------------------
def _backtest(series: pd.Series, model, horizon: int, n_folds: int) -> np.ndarray:
    """Erros (real − previsto) em `n_folds` janelas consecutivas no fim da série."""
    errors = []
    for fold in range(n_folds, 0, -1):
        cutoff = len(series) - fold * horizon
        train, test = series.iloc[:cutoff], series.iloc[cutoff : cutoff + horizon]
        prediction = np.clip(model(train, horizon), 0, None)
        errors.append(test.to_numpy() - prediction)
    return np.concatenate(errors)


def _total_error(errors_per_fold: np.ndarray, actual_per_fold: np.ndarray) -> float:
    """Erro médio (%) do total de cada janela; janelas sem receita ficam de fora (o % não existe)."""
    valid = actual_per_fold > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(errors_per_fold[valid]) / actual_per_fold[valid]) * 100)


def _score(series: pd.Series, errors: dict[str, np.ndarray], horizon: int, n_folds: int) -> pd.DataFrame:
    actual = series.iloc[-n_folds * horizon :].to_numpy()
    actual_total = np.abs(actual).sum()
    actual_per_fold = actual.reshape(n_folds, horizon).sum(axis=1)
    scores = pd.DataFrame(
        {
            name: {
                # WAPE: erro absoluto total / receita total (robusto a dias com receita zero)
                "wape": np.abs(err).sum() / actual_total * 100,
                "rmse": float(np.sqrt(np.mean(err**2))),
                # viés: positivo = o modelo subestima a receita
                "bias": err.sum() / actual_total * 100,
                # erro ao prever o total de cada janela, em média
                "total_error": _total_error(err.reshape(n_folds, horizon).sum(axis=1), actual_per_fold),
            }
            for name, err in errors.items()
        }
    ).T
    baseline = scores.loc[BASELINE, "wape"]
    scores["improvement_vs_baseline"] = (baseline - scores["wape"]) / baseline * 100
    return scores.sort_values("wape")
