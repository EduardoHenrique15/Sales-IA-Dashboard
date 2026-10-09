"""
Cálculo dos indicadores (KPIs), em funções puras.

São a fonte única dos números: os cards da tela e o agente de IA
recebem exatamente os mesmos objetos, garantindo que o relatório em
texto NUNCA contradiga o dashboard visual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


def _empty_series() -> pd.Series:
    return pd.Series(dtype=float)


@dataclass(frozen=True, eq=False)
class SalesKPIs:
    total_revenue: float = 0.0
    # None quando a base não informa o custo (ex.: planilha enviada sem coluna de custo)
    total_profit: float | None = 0.0
    margin_pct: float | None = 0.0
    total_units: int = 0
    n_orders: int = 0
    avg_ticket: float = 0.0
    top_category: str = "N/A"
    top_category_revenue: float = 0.0
    top_region: str = "N/A"
    # None quando não há período anterior comparável
    revenue_growth_pct: float | None = None
    # receita por categoria/região, em ordem decrescente
    revenue_by_category: pd.Series = field(default_factory=_empty_series)
    revenue_by_region: pd.Series = field(default_factory=_empty_series)


def count_orders(df: pd.DataFrame) -> int:
    """Número de pedidos: códigos distintos quando a base os tem, senão uma linha = um pedido."""
    return int(df["order_id"].nunique()) if "order_id" in df.columns else len(df)


def compute_sales_kpis(df: pd.DataFrame, previous_df: pd.DataFrame | None = None) -> SalesKPIs:
    """Calcula os KPIs principais de vendas para o período filtrado.

    `previous_df` (opcional) é o período anterior de mesma duração,
    usado para calcular a variação percentual da receita.
    """
    if df.empty:
        return SalesKPIs()

    total_revenue = float(df["revenue"].sum())
    n_orders = count_orders(df)

    # Com custo ausente em algum pedido, lucro e margem ficam indisponíveis
    # em vez de subestimados.
    total_profit = float(df["profit"].sum()) if df["cost"].notna().all() else None
    margin_pct = None
    if total_profit is not None:
        margin_pct = (total_profit / total_revenue * 100) if total_revenue else 0.0

    revenue_by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    revenue_by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    revenue_growth_pct = None
    if previous_df is not None and not previous_df.empty:
        prev_revenue = float(previous_df["revenue"].sum())
        if prev_revenue > 0:
            revenue_growth_pct = (total_revenue - prev_revenue) / prev_revenue * 100

    return SalesKPIs(
        total_revenue=total_revenue,
        total_profit=total_profit,
        margin_pct=margin_pct,
        total_units=int(df["units"].sum()),
        n_orders=n_orders,
        avg_ticket=total_revenue / n_orders,
        top_category=revenue_by_category.index[0],
        top_category_revenue=float(revenue_by_category.iloc[0]),
        top_region=revenue_by_region.index[0],
        revenue_growth_pct=revenue_growth_pct,
        revenue_by_category=revenue_by_category,
        revenue_by_region=revenue_by_region,
    )
