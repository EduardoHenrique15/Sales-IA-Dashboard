"""
Recorte da base de vendas por período, categoria e região.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class SalesFilters:
    start: date
    end: date
    # listas vazias = sem filtro (todas as categorias/regiões)
    categories: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)

    @property
    def period_label(self) -> str:
        return f"{self.start.strftime('%d/%m/%Y')} a {self.end.strftime('%d/%m/%Y')}"


def filter_sales(df: pd.DataFrame, filters: SalesFilters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (período selecionado, período anterior de mesma duração),
    ambos com os mesmos filtros de categoria e região."""
    dates = df["date"].dt.date
    current = df.loc[(dates >= filters.start) & (dates <= filters.end)]
    previous = previous_period_df(df, filters.start, filters.end)
    return _apply_segments(current, filters), _apply_segments(previous, filters)


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


def _apply_segments(df: pd.DataFrame, filters: SalesFilters) -> pd.DataFrame:
    if filters.categories:
        df = df[df["category"].isin(filters.categories)]
    if filters.regions:
        df = df[df["region"].isin(filters.regions)]
    return df
