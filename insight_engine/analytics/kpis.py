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
    total_profit: float = 0.0
    margin_pct: float = 0.0
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


@dataclass(frozen=True)
class CryptoKPIs:
    current_price: float = 0.0
    period_change_pct: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0
    avg_volume: float = 0.0
    volatility_pct: float = 0.0


def compute_sales_kpis(df: pd.DataFrame, previous_df: pd.DataFrame | None = None) -> SalesKPIs:
    """Calcula os KPIs principais de vendas para o período filtrado.

    `previous_df` (opcional) é o período anterior de mesma duração,
    usado para calcular a variação percentual da receita.
    """
    if df.empty:
        return SalesKPIs()

    total_revenue = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())
    n_orders = len(df)

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
        margin_pct=(total_profit / total_revenue * 100) if total_revenue else 0.0,
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


def compute_crypto_kpis(df: pd.DataFrame) -> CryptoKPIs:
    """Calcula KPIs para a série histórica de uma criptomoeda."""
    if df.empty or "price" not in df.columns:
        return CryptoKPIs()

    current_price = float(df["price"].iloc[-1])
    first_price = float(df["price"].iloc[0])
    daily_returns = df["price"].pct_change().dropna()

    return CryptoKPIs(
        current_price=current_price,
        period_change_pct=((current_price - first_price) / first_price * 100) if first_price else 0.0,
        max_price=float(df["price"].max()),
        min_price=float(df["price"].min()),
        avg_volume=float(df["volume"].mean()) if "volume" in df.columns else 0.0,
        volatility_pct=float(daily_returns.std() * 100) if not daily_returns.empty else 0.0,
    )
