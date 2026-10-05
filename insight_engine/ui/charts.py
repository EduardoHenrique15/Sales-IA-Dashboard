"""
Construção dos gráficos do dashboard.

Cada função recebe dados já calculados e devolve uma figura Plotly
pronta, com o estilo padrão de `theme.apply_chart_theme`.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from insight_engine.ui.theme import COLOR_PRICE, COLOR_PROFIT, COLOR_REVENUE, apply_chart_theme

_HORIZONTAL_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


# ------------------------------------------------------------------
# Vendas
# ------------------------------------------------------------------
def revenue_profit_over_time(df: pd.DataFrame) -> go.Figure:
    daily = df.groupby(df["date"].dt.date).agg(revenue=("revenue", "sum"), profit=("profit", "sum")).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["revenue"],
            name="Receita",
            mode="lines",
            line=dict(color=COLOR_REVENUE, width=3),
            fill="tozeroy",
            fillcolor="rgba(129,140,248,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["profit"],
            name="Lucro",
            mode="lines",
            line=dict(color=COLOR_PROFIT, width=2, dash="dot"),
        )
    )
    return apply_chart_theme(fig, "Receita e Lucro ao Longo do Tempo", 380, legend=_HORIZONTAL_LEGEND)


def revenue_by_category(revenue: pd.Series) -> go.Figure:
    data = revenue.rename_axis("category").reset_index(name="revenue")
    fig = px.pie(
        data,
        names="category",
        values="revenue",
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Purples_r,
    )
    return apply_chart_theme(fig, "Receita por Categoria", 380, showlegend=True)


def revenue_by_region(revenue: pd.Series) -> go.Figure:
    data = revenue.rename_axis("region").reset_index(name="revenue")
    fig = px.bar(
        data,
        x="region",
        y="revenue",
        color="revenue",
        color_continuous_scale="Purples",
        text_auto=".2s",
    )
    return apply_chart_theme(fig, "Receita por Região", 340, coloraxis_showscale=False)


def top_products(df: pd.DataFrame, n: int = 8) -> go.Figure:
    data = df.groupby("product")["revenue"].sum().sort_values(ascending=True).tail(n).reset_index()
    fig = px.bar(
        data,
        x="revenue",
        y="product",
        orientation="h",
        color="revenue",
        color_continuous_scale="Blues",
        text_auto=".2s",
    )
    return apply_chart_theme(fig, "Top Produtos por Receita", 340, coloraxis_showscale=False)


# ------------------------------------------------------------------
# Criptomoedas
# ------------------------------------------------------------------
def price_over_time(df: pd.DataFrame, coin_name: str, days: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["price"],
            name="Preço (USD)",
            mode="lines",
            line=dict(color=COLOR_PRICE, width=3),
            fill="tozeroy",
            fillcolor="rgba(251,191,36,0.10)",
        )
    )
    return apply_chart_theme(fig, f"Preço de {coin_name} — {days} dias", 400)


def traded_volume(df: pd.DataFrame) -> go.Figure:
    fig = px.bar(df, x="date", y="volume", color_discrete_sequence=[COLOR_REVENUE])
    return apply_chart_theme(fig, "Volume Negociado (USD)", 280)
