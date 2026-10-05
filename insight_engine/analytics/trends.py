"""
Análises de tendência usadas no diagnóstico do relatório.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class Trend:
    # inclinação da reta, convertida em % de variação ao longo do período
    slope_pct: float
    # qualidade do ajuste (0 a 1)
    r2: float

    @property
    def confidence(self) -> str:
        return "alta" if self.r2 > 0.5 else ("moderada" if self.r2 > 0.2 else "baixa")


def fit_trend(series: pd.Series) -> Trend:
    """Ajusta uma regressão linear simples (posição no tempo -> valor)."""
    if len(series) < 2:
        return Trend(0.0, 0.0)

    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    r2 = model.score(x, y)

    mean_val = y.mean() if y.mean() != 0 else 1
    slope_pct_total = (slope * len(series)) / mean_val * 100
    return Trend(float(slope_pct_total), float(r2))


def declining_categories(df: pd.DataFrame, threshold: float = -0.15) -> list[tuple[str, float]]:
    """Compara a receita da primeira e da segunda metade do período por
    categoria e retorna [(categoria, variação %)] das que caíram mais que
    `threshold` (padrão: queda de 15%)."""
    mid_point = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
    first_half = df[df["date"] <= mid_point]
    second_half = df[df["date"] > mid_point]

    if first_half.empty or second_half.empty:
        return []

    rev_first = first_half.groupby("category")["revenue"].sum()
    rev_second = second_half.groupby("category")["revenue"].sum()

    declining = []
    for cat in rev_first.index:
        v1, v2 = rev_first.get(cat, 0), rev_second.get(cat, 0)
        if v1 > 0 and (v2 - v1) / v1 < threshold:
            declining.append((cat, (v2 - v1) / v1 * 100))
    return declining
