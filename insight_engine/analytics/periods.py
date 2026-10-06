"""
Recorte da base de vendas por período, categoria e região.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

Comparison = Literal["anterior", "ano_anterior"]
COMPARISON_LABELS: dict[str, str] = {
    "anterior": "período anterior de mesma duração",
    "ano_anterior": "mesmo período do ano anterior",
}


@dataclass(frozen=True)
class SalesFilters:
    start: date
    end: date
    # listas vazias = sem filtro (todas as categorias/regiões)
    categories: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    # base de comparação: período imediatamente anterior ou mesmo período do ano anterior
    comparison: Comparison = "anterior"

    @property
    def period_label(self) -> str:
        return f"{self.start.strftime('%d/%m/%Y')} a {self.end.strftime('%d/%m/%Y')}"

    @property
    def comparison_label(self) -> str:
        return COMPARISON_LABELS[self.comparison]


def filter_sales(df: pd.DataFrame, filters: SalesFilters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (período selecionado, período de comparação), ambos com os
    mesmos filtros de categoria e região."""
    dates = df["date"].dt.date
    current = df.loc[(dates >= filters.start) & (dates <= filters.end)]
    if filters.comparison == "ano_anterior":
        previous = same_period_last_year_df(df, filters.start, filters.end)
    else:
        previous = previous_period_df(df, filters.start, filters.end)
    return apply_segments(current, filters), apply_segments(previous, filters)


def same_period_last_year_df(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Mesmo período, um ano antes (compara sem o efeito da sazonalidade anual)."""
    prev_start = pd.Timestamp(start) - pd.DateOffset(years=1)
    prev_end = pd.Timestamp(end) - pd.DateOffset(years=1)
    return df.loc[(df["date"] >= prev_start) & (df["date"] <= prev_end)]


def previous_period_df(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Retorna o subconjunto do DataFrame correspondente ao período
    imediatamente anterior, de mesma duração — usado para growth %.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    period_len = end_ts - start_ts

    prev_end = start_ts - pd.Timedelta(days=1)
    prev_start = prev_end - period_len

    mask = (df["date"] >= prev_start) & (df["date"] <= prev_end)
    return df.loc[mask]


def apply_segments(df: pd.DataFrame, filters: SalesFilters) -> pd.DataFrame:
    """Aplica só os filtros de categoria e região (sem filtro de data)."""
    if filters.categories:
        df = df[df["category"].isin(filters.categories)]
    if filters.regions:
        df = df[df["region"].isin(filters.regions)]
    return df
