"""
Construção dos gráficos do dashboard.

Cada função recebe dados já calculados e devolve uma figura Plotly
pronta, com o estilo padrão de `theme.apply_chart_theme`.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from insight_engine.analytics.forecasting import ForecastResult
from insight_engine.analytics.variance import RevenueBridge
from insight_engine.formatting import format_number
from insight_engine.ui.theme import (
    MUTED_LINE,
    NEGATIVE,
    NEUTRAL,
    POSITIVE,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    apply_chart_theme,
)

_HORIZONTAL_LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)


# ------------------------------------------------------------------
# Vendas
# ------------------------------------------------------------------
def revenue_profit_over_time(df: pd.DataFrame, show_profit: bool = True) -> go.Figure:
    daily = df.groupby(df["date"].dt.normalize()).agg(revenue=("revenue", "sum"), profit=("profit", "sum"))

    fig = go.Figure()
    series = [("revenue", "Receita", SERIES_1)] + ([("profit", "Lucro", SERIES_3)] if show_profit else [])
    for column, name, color in series:
        fig.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily[column],
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate="%{x|%d/%m/%Y}: R$ %{y:,.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    title = "Receita e lucro por dia" if show_profit else "Receita por dia"
    return date_axis(apply_chart_theme(fig, title, 380, legend=_HORIZONTAL_LEGEND, showlegend=show_profit))


def revenue_by_category(revenue: pd.Series) -> go.Figure:
    """Participação por categoria em barras (comparam valores próximos melhor que uma rosca)."""
    data = revenue.sort_values()
    total = data.sum() or 1
    return _labeled_bars(
        data,
        [f"{_short_brl(v)} ({format_number(v / total * 100, 1)}%)" for v in data.values],
        "Receita por categoria",
        380,
        room=2.1,  # coluna estreita e rótulos longos (valor + participação)
    )


def revenue_by_region(revenue: pd.Series) -> go.Figure:
    data = revenue.sort_values()
    return _labeled_bars(data, [_short_brl(v) for v in data.values], "Receita por região", 340)


def top_products(df: pd.DataFrame, n: int = 8) -> go.Figure:
    data = df.groupby("product")["revenue"].sum().sort_values(ascending=True).tail(n)
    return _labeled_bars(data, [_short_brl(v) for v in data.values], f"Top {n} produtos por receita", 340)


def _labeled_bars(data: pd.Series, labels: list[str], title: str, height: int, room: float = 1.45) -> go.Figure:
    """Barras horizontais de uma cor, com o valor escrito ao lado (o eixo numérico fica oculto)."""
    fig = go.Figure(
        go.Bar(
            x=data.values,
            y=data.index,
            orientation="h",
            marker=dict(color=SERIES_1),
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>R$ %{x:,.2f}<extra></extra>",
        )
    )
    # espaço à direita para os rótulos não serem cortados
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, data.max() * room])
    return apply_chart_theme(fig, title, height, showlegend=False)


def date_axis(fig: go.Figure) -> go.Figure:
    """Datas do eixo x em formato numérico brasileiro, que se adapta ao zoom."""
    fig.update_xaxes(
        tickformatstops=[
            dict(dtickrange=[None, 86_400_000 * 40], value="%d/%m"),
            dict(dtickrange=[86_400_000 * 40, 86_400_000 * 300], value="%m/%Y"),
            dict(dtickrange=[86_400_000 * 300, None], value="%Y"),
        ]
    )
    return fig


# ------------------------------------------------------------------
# Análises: variação, previsão, anomalias, sazonalidade e clientes
# ------------------------------------------------------------------
def revenue_bridge(bridge: RevenueBridge) -> go.Figure:
    """Cascata: receita anterior -> efeitos volume/preço/mix -> receita atual."""
    effects = [bridge.volume_effect, bridge.price_effect, bridge.mix_effect]
    fig = go.Figure(
        go.Waterfall(
            x=["Período anterior", "Volume", "Preço", "Mix", "Período atual"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[bridge.previous_revenue, *effects, bridge.current_revenue],
            text=[_short_brl(v) for v in [bridge.previous_revenue, *effects, bridge.current_revenue]],
            textposition="outside",
            connector=dict(line=dict(color=NEUTRAL, width=1)),
            increasing=dict(marker=dict(color=POSITIVE)),
            decreasing=dict(marker=dict(color=NEGATIVE)),
            totals=dict(marker=dict(color=NEUTRAL)),
            hovertemplate="%{x}: R$ %{y:,.2f}<extra></extra>",
        )
    )
    # Eixo ampliado em torno da variação: com o eixo a partir de zero, efeitos de poucos
    # por cento ficariam invisíveis ao lado dos totais. Os valores do eixo ficam visíveis
    # (e o título avisa) para que o recorte seja explícito.
    levels = [bridge.previous_revenue]
    for effect in effects:
        levels.append(levels[-1] + effect)
    low, high = min(levels), max(levels)
    pad = max((high - low) * 0.6, high * 0.02)
    fig.update_yaxes(range=[max(0, low - pad), high + pad], tickformat=",.0f")
    return apply_chart_theme(fig, "Da receita anterior à atual (eixo não começa em zero)", 380, showlegend=False)


def bridge_by_segment(by_segment: pd.DataFrame) -> go.Figure:
    """Contribuição de cada categoria para a variação total (barras divergentes)."""
    data = by_segment.sort_values("total")
    fig = go.Figure(
        go.Bar(
            x=data["total"],
            y=data.index,
            orientation="h",
            marker=dict(color=[POSITIVE if v >= 0 else NEGATIVE for v in data["total"]]),
            text=[_short_brl(v) for v in data["total"]],
            textposition="outside",
            cliponaxis=False,
            customdata=data[["volume", "price", "mix"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Variação: R$ %{x:,.2f}<br>Volume: R$ %{customdata[0]:,.2f}"
                "<br>Preço: R$ %{customdata[1]:,.2f}<br>Mix: R$ %{customdata[2]:,.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color=NEUTRAL, width=1))
    reach = data["total"].abs().max() * 1.6 or 1
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[-reach, reach])
    return apply_chart_theme(fig, "Variação por categoria", 380, showlegend=False)


def forecast(result: ForecastResult, history_days: int = 180) -> go.Figure:
    history = result.history.iloc[-history_days:]
    fc = result.forecast
    band_x = list(fc.index) + list(fc.index[::-1])
    fig = go.Figure()
    for level, opacity in (("95", 0.08), ("80", 0.16)):
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=list(fc[f"upper_{level}"]) + list(fc[f"lower_{level}"][::-1]),
                fill="toself",
                fillcolor=f"rgba(217,89,38,{opacity})",
                line=dict(width=0),
                name=f"Intervalo de {level}%",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history,
            name="Receita realizada",
            mode="lines",
            line=dict(color=SERIES_1, width=2),
            hovertemplate="%{x|%d/%m/%Y}: R$ %{y:,.2f}<extra>Realizado</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc["yhat"],
            name="Previsão",
            mode="lines",
            line=dict(color=SERIES_2, width=2.5),
            hovertemplate="%{x|%d/%m/%Y}: R$ %{y:,.2f}<extra>Previsão</extra>",
        )
    )
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return date_axis(
        apply_chart_theme(fig, "Receita diária: histórico recente e previsão", 420, legend=_HORIZONTAL_LEGEND)
    )


def anomalies(series: pd.Series, found: pd.DataFrame, value_label: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series,
            name=value_label,
            mode="lines",
            line=dict(color=MUTED_LINE, width=1),
            hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f}<extra></extra>",
        )
    )
    for kind, color, symbol in (("pico", POSITIVE, "triangle-up"), ("queda", NEGATIVE, "triangle-down")):
        points = found[found["kind"] == kind]
        fig.add_trace(
            go.Scatter(
                x=points.index,
                y=points["value"],
                name=f"Anomalia: {kind}",
                mode="markers",
                marker=dict(color=color, size=11, symbol=symbol, line=dict(color="#0e1117", width=2)),
                customdata=points[["expected", "deviation_pct"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>Observado: %{y:,.0f}<br>Esperado: %{customdata[0]:,.0f}"
                    "<br>Desvio: %{customdata[1]:+.0f}%<extra></extra>"
                ),
            )
        )
    return date_axis(
        apply_chart_theme(fig, f"{value_label} por dia e anomalias detectadas", 380, legend=_HORIZONTAL_LEGEND)
    )


def weekday_effect(profile: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=profile.index,
            y=profile,
            marker=dict(color=[POSITIVE if v >= 0 else NEGATIVE for v in profile]),
            hovertemplate="%{x}: %{y:+,.0f} em relação à média<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1))
    return apply_chart_theme(fig, "Efeito do dia da semana na receita (R$/dia)", 320, showlegend=False)


def monthly_index(index: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=index.index,
            y=index - 1,
            base=1,
            marker=dict(color=[POSITIVE if v >= 1 else NEGATIVE for v in index]),
            hovertemplate="%{x}: índice %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=1, line=dict(color=NEUTRAL, width=1))
    return apply_chart_theme(fig, "Sazonalidade mensal (1,0 = mês típico)", 320, showlegend=False)


def segments(summary: pd.DataFrame) -> go.Figure:
    data = summary.iloc[::-1]  # primeiro segmento no topo
    fig = go.Figure()
    series = (("customers_pct", "% dos clientes", SERIES_1), ("revenue_pct", "% da receita", SERIES_2))
    for column, name, color in series:
        fig.add_trace(
            go.Bar(
                x=data[column],
                y=data.index,
                orientation="h",
                name=name,
                marker=dict(color=color),
                hovertemplate="<b>%{y}</b><br>" + name + ": %{x:.1f}%<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return apply_chart_theme(fig, "Peso de cada segmento em clientes e em receita", 420, legend=_HORIZONTAL_LEGEND)


def silhouette(scores: pd.Series, best_k: int) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=scores.index,
            y=scores,
            mode="lines+markers",
            line=dict(color=SERIES_1, width=2),
            marker=dict(size=9, color=[SERIES_2 if k == best_k else SERIES_1 for k in scores.index]),
            hovertemplate="k = %{x}: silhueta %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(dtick=1, title="Número de grupos")
    return apply_chart_theme(fig, f"Silhueta por k (escolhido: {best_k})", 280, showlegend=False)


def cluster_profile(clusters: pd.DataFrame, column: str, title: str, hover_format: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=clusters.index,
            y=clusters[column],
            marker=dict(color=SERIES_1),
            hovertemplate="%{x}: " + hover_format + "<extra></extra>",
        )
    )
    return apply_chart_theme(fig, title, 280, showlegend=False)


def _short_brl(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}R$ {format_number(value / 1_000_000, 2)} mi"
    if value >= 1_000:
        return f"{sign}R$ {format_number(value / 1_000, 0)} mil"
    return f"{sign}R$ {format_number(value, 0)}"
