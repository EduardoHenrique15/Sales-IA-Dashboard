"""
Recorte da base de vendas por período, categoria e região.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd

Comparison = Literal["anterior", "ano_anterior"]
# parte máxima do período de comparação que pode ficar antes do início da base
MAX_MISSING_SHARE = 0.05
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
    mesmos filtros de categoria e região.

    Se a base não cobre o período de comparação inteiro (ex.: comparar 12 meses
    com o ano anterior quando a base começa no meio dele), a comparação volta
    vazia: comparar um ano com alguns meses inflaria o crescimento.
    """
    current = between_days(df, filters.start, filters.end)
    if comparison_is_complete(df, filters):
        previous = between_days(df, *comparison_window(filters.start, filters.end, filters.comparison))
    else:
        previous = df.iloc[0:0]
    return apply_segments(current, filters), apply_segments(previous, filters)


def between_days(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Linhas de `start` a `end`, dias inteiros (inclui as vendas com horário no último dia)."""
    days = df["date"].dt.normalize()
    return df.loc[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def comparison_window(start, end, comparison: str = "anterior") -> tuple[pd.Timestamp, pd.Timestamp]:
    """(início, fim) do período de comparação, em dias inteiros."""
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if comparison == "ano_anterior":
        return start_ts - pd.DateOffset(years=1), end_ts - pd.DateOffset(years=1)
    prev_end = start_ts - pd.Timedelta(days=1)
    return prev_end - (end_ts - start_ts), prev_end


def comparison_is_complete(df: pd.DataFrame, filters: SalesFilters) -> bool:
    """O período de comparação está (praticamente) inteiro dentro da base?"""
    if df.empty:
        return False
    return window_is_covered(*comparison_window(filters.start, filters.end, filters.comparison), df["date"].min())


def window_is_covered(start, end, data_start) -> bool:
    """A janela começa depois do início da base, ou perde no máximo 5% dos dias.

    Comparar com um período que está só em parte na base infla o crescimento (o
    "anterior" parece menor do que foi). A folga de 5% aceita bases que começam
    alguns dias depois do início do ano (ex.: a Olist, com a primeira venda em 05/01/2017).
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    missing_days = (pd.Timestamp(data_start).normalize() - start).days
    return missing_days <= ((end - start).days + 1) * MAX_MISSING_SHARE


def same_period_last_year_df(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Mesmo período, um ano antes (compara sem o efeito da sazonalidade anual)."""
    return between_days(df, *comparison_window(start, end, "ano_anterior"))


def previous_period_df(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Período imediatamente anterior, de mesma duração — usado no crescimento %."""
    return between_days(df, *comparison_window(start, end, "anterior"))


def apply_segments(df: pd.DataFrame, filters: SalesFilters) -> pd.DataFrame:
    """Aplica só os filtros de categoria e região (sem filtro de data)."""
    if filters.categories:
        df = df[df["category"].isin(filters.categories)]
    if filters.regions:
        df = df[df["region"].isin(filters.regions)]
    return df
