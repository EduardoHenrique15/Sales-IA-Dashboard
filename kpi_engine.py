"""
kpi_engine.py
================================================================
Funções puras de cálculo de indicadores (KPIs).

Mantidas separadas de `app.py` para que possam ser:
  - testadas isoladamente,
  - reutilizadas pelo `ai_agent.py` (o agente de IA recebe os
    mesmos números que aparecem na tela, garantindo que o
    relatório em texto NUNCA contradiga o dashboard visual).
================================================================
"""

from __future__ import annotations

import pandas as pd


def compute_sales_kpis(df: pd.DataFrame, previous_df: pd.DataFrame | None = None) -> dict:
    """Calcula os KPIs principais de vendas para o período filtrado.

    `previous_df` (opcional) é o período anterior de mesma duração,
    usado para calcular variação percentual (growth).
    """
    if df.empty:
        return _empty_sales_kpis()

    total_revenue = float(df["revenue"].sum())
    total_profit = float(df["profit"].sum())
    total_units = int(df["units"].sum())
    n_orders = len(df)

    avg_ticket = total_revenue / n_orders if n_orders else 0.0
    margin_pct = (total_profit / total_revenue * 100) if total_revenue else 0.0

    top_category_series = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    top_region_series = df.groupby("region")["revenue"].sum().sort_values(ascending=False)

    top_category = top_category_series.index[0] if not top_category_series.empty else "N/A"
    top_region = top_region_series.index[0] if not top_region_series.empty else "N/A"

    revenue_growth_pct = None
    if previous_df is not None and not previous_df.empty:
        prev_revenue = float(previous_df["revenue"].sum())
        if prev_revenue > 0:
            revenue_growth_pct = (total_revenue - prev_revenue) / prev_revenue * 100

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "margin_pct": margin_pct,
        "total_units": total_units,
        "n_orders": n_orders,
        "avg_ticket": avg_ticket,
        "top_category": top_category,
        "top_category_revenue": float(top_category_series.iloc[0]) if not top_category_series.empty else 0.0,
        "top_region": top_region,
        "revenue_growth_pct": revenue_growth_pct,
        "revenue_by_category": top_category_series,
        "revenue_by_region": top_region_series,
    }


def _empty_sales_kpis() -> dict:
    return {
        "total_revenue": 0.0, "total_profit": 0.0, "margin_pct": 0.0,
        "total_units": 0, "n_orders": 0, "avg_ticket": 0.0,
        "top_category": "N/A", "top_category_revenue": 0.0, "top_region": "N/A",
        "revenue_growth_pct": None,
        "revenue_by_category": pd.Series(dtype=float),
        "revenue_by_region": pd.Series(dtype=float),
    }


def compute_crypto_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs para a série histórica de uma criptomoeda."""
    if df.empty or "price" not in df.columns:
        return {
            "current_price": 0.0, "period_change_pct": 0.0,
            "max_price": 0.0, "min_price": 0.0, "avg_volume": 0.0,
            "volatility_pct": 0.0,
        }

    current_price = float(df["price"].iloc[-1])
    first_price = float(df["price"].iloc[0])
    period_change_pct = ((current_price - first_price) / first_price * 100) if first_price else 0.0

    daily_returns = df["price"].pct_change().dropna()
    volatility_pct = float(daily_returns.std() * 100) if not daily_returns.empty else 0.0

    return {
        "current_price": current_price,
        "period_change_pct": period_change_pct,
        "max_price": float(df["price"].max()),
        "min_price": float(df["price"].min()),
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else 0.0,
        "volatility_pct": volatility_pct,
    }


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
