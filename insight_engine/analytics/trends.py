"""
Análises de tendência usadas no diagnóstico do relatório.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class Trend:
    # inclinação da reta, convertida em % de variação ao longo do período
    slope_pct: float
    # qualidade do ajuste (0 a 1)
    r2: float


@dataclass(frozen=True)
class MannKendall:
    """Resultado do teste de tendência de Mann-Kendall."""

    # tau de Kendall entre tempo e valor: de -1 (queda) a +1 (alta)
    tau: float
    p_value: float
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def direction(self) -> str:
        """Direção da tendência: alta, queda ou "sem tendência" (não significativa)."""
        if not self.significant:
            return "sem tendência"
        return "alta" if self.tau > 0 else "queda"


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


def mann_kendall(series: pd.Series, alpha: float = 0.05) -> MannKendall:
    """Teste não paramétrico de tendência monotônica (Mann-Kendall).

    Equivale ao tau de Kendall entre a posição no tempo e o valor. Não supõe
    distribuição normal nem tendência linear, por isso é o teste padrão
    para séries de vendas, que costumam ter outliers e sazonalidade.
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(values) < 4 or np.all(values == values[0]):
        return MannKendall(0.0, 1.0, alpha)
    result = kendalltau(np.arange(len(values)), values)
    return MannKendall(float(result.statistic), float(result.pvalue), alpha)


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
