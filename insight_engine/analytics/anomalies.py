"""
Decomposição de séries diárias e detecção de anomalias.

Detecção (`detect_anomalies`):
  1. a série é levada à escala logarítmica: vendas variam de forma
     multiplicativa (um dia bom vende "o dobro", não "R$ 10 mil a mais"),
     e no log essa variação fica simétrica;
  2. STL (Seasonal-Trend decomposition using LOESS) separa tendência,
     sazonalidade semanal e resíduo;
  3. um dia é anômalo quando o resíduo é extremo pelo z-score robusto:
         z = (resíduo − mediana) / (1,4826 × MAD)
     Mediana e MAD não são distorcidos pelas próprias anomalias;
  4. e quando o desvio em relação ao esperado é relevante (materialidade):
     em séries pouco ruidosas, um desvio de 5% pode ser estatisticamente
     raro sem ter importância para o negócio.

A suavização do STL e o limiar foram calibrados na base de exemplo, que
tem anomalias plantadas em datas conhecidas (ver `data.sales`): o
detector encontra todas, com poucos alarmes extras.

Para séries com valores de cauda longa (como a receita, em que poucos
pedidos caros dominam alguns dias), o número de pedidos costuma ser o
sinal mais confiável para incidentes operacionais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

SEASON = 7
MIN_DAYS = 4 * SEASON
DEFAULT_THRESHOLD = 3.5  # limiar usual para z-score robusto (Iglewicz & Hoaglin)
MIN_DEVIATION_PCT = 20.0  # desvio mínimo em relação ao esperado para ser relevante
# Suavizações do STL para detecção: mais rígidas que o padrão, para que
# tendência e sazonalidade não "absorvam" os próprios dias anômalos.
_DETECTION_SMOOTHING = {"seasonal": 31, "trend": 61}

WEEKDAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
MONTHS = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


@dataclass(frozen=True)
class Decomposition:
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series

    @property
    def weekday_profile(self) -> pd.Series:
        """Efeito médio de cada dia da semana, na unidade da série (ex.: R$)."""
        profile = self.seasonal.groupby(self.seasonal.index.dayofweek).mean()
        return profile.rename(index=dict(enumerate(WEEKDAYS)))


def daily_series(df: pd.DataFrame, metric: str = "revenue") -> pd.Series:
    """Série diária de `metric` ("revenue" ou "orders"), com zero nos dias sem venda."""
    grouped = df.groupby(df["date"].dt.normalize())
    daily = grouped["revenue"].sum() if metric == "revenue" else grouped.size().astype(float)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0.0).rename(metric)


def decompose(series: pd.Series) -> Decomposition:
    """Decomposição aditiva (STL robusto, sazonalidade semanal) para visualização."""
    series = _check(series)
    result = STL(series, period=SEASON, robust=True).fit()
    return Decomposition(series, result.trend, result.seasonal, result.resid)


def detect_anomalies(
    series: pd.Series, threshold: float = DEFAULT_THRESHOLD, min_deviation_pct: float = MIN_DEVIATION_PCT
) -> pd.DataFrame:
    """Dias que fogem do padrão, do mais extremo para o menos extremo.

    Colunas: value, expected, deviation_pct, score, kind ("pico" ou "queda").
    """
    series = _check(series)
    log_series = np.log1p(series.clip(lower=0))
    fit = STL(log_series, period=SEASON, robust=True, **_DETECTION_SMOOTHING).fit()

    resid = fit.resid
    mad = float(np.median(np.abs(resid - resid.median())))
    if mad == 0:
        return pd.DataFrame(columns=["value", "expected", "deviation_pct", "score", "kind"])
    score = (resid - resid.median()) / (1.4826 * mad)

    expected = np.expm1(fit.trend + fit.seasonal)
    deviation_pct = (series - expected) / expected.clip(lower=1e-9) * 100
    flagged = (score.abs() > threshold) & (deviation_pct.abs() >= min_deviation_pct)
    anomalies = pd.DataFrame(
        {
            "value": series[flagged],
            "expected": expected[flagged],
            "deviation_pct": deviation_pct[flagged],
            "score": score[flagged],
            "kind": np.where(score[flagged] > 0, "pico", "queda"),
        }
    )
    return anomalies.loc[anomalies["score"].abs().sort_values(ascending=False).index]


def monthly_seasonality(series: pd.Series) -> pd.Series:
    """Índice de sazonalidade mensal: média diária do mês / média do ano (1,0 = típico).

    Cada ano é normalizado pela própria média, para não confundir crescimento
    com sazonalidade. Exige ao menos um ano completo para ser representativo.
    """
    monthly = series.resample("MS").mean()
    by_year = monthly.groupby(monthly.index.year).transform("mean")
    normalized = (monthly / by_year).dropna()
    index = normalized.groupby(normalized.index.month).mean()
    return index.rename(index=lambda m: MONTHS[m - 1])


def _check(series: pd.Series) -> pd.Series:
    series = series.astype(float).asfreq("D", fill_value=0.0)
    if len(series) < MIN_DAYS:
        raise ValueError(f"São necessários pelo menos {MIN_DAYS} dias de histórico (a base tem {len(series)}).")
    return series
