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
from insight_engine.ui.theme import apply_chart_theme, palette

_HORIZONTAL_LEGEND = dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None)
GRANULARITIES = {"D": "Dia", "W": "Semana", "M": "Mês"}
# Rótulo dentro da barra quando cabe e fora quando a barra é curta: assim o texto
# nunca sai da área do gráfico, qualquer que seja a largura da tela.
_BAR_LABELS = dict(
    textposition="auto",
    textangle=0,
    insidetextanchor="end",
    insidetextfont=dict(color="white"),
    cliponaxis=False,
)
MOVING_AVERAGE_DAYS = 7


# ------------------------------------------------------------------
# Vendas
# ------------------------------------------------------------------
def revenue_over_time(df: pd.DataFrame, show_profit: bool = True, granularity: str = "D") -> go.Figure:
    """Receita (e lucro) por dia, semana ou mês.

    Por dia, a série diária fica clara ao fundo e a média móvel de 7 dias em
    destaque: o dia a dia de vendas é ruidoso e a média mostra a tendência.
    """
    p = palette()
    frequency = {"D": "D", "W": "W-MON", "M": "MS"}[granularity]
    # dias/semanas/meses sem venda entram como zero: a média móvel é de 7 dias do
    # calendário (e não das 7 últimas datas com venda) e a linha não "pula" os vazios
    grouped = df.set_index("date")[["revenue", "profit"]].resample(frequency, label="left", closed="left").sum()
    date_format = {"D": "%d/%m/%Y", "W": "semana de %d/%m/%Y", "M": "%m/%Y"}[granularity]

    fig = go.Figure()
    series = [("revenue", "Receita", p.series_1)] + ([("profit", "Lucro", p.series_3)] if show_profit else [])
    for column, name, color in series:
        values = grouped[column]
        if granularity == "D":
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values,
                    name=f"{name} diária",
                    legendgroup=name,
                    showlegend=False,
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.3,
                    hovertemplate="R$ %{y:,.2f}<extra>" + name + " no dia</extra>",
                )
            )
            values = values.rolling(MOVING_AVERAGE_DAYS, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values,
                name=name,
                legendgroup=name,
                mode="lines+markers" if len(values) <= 16 else "lines",
                line=dict(color=color, width=2.5),
                hovertemplate="R$ %{y:,.2f}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(hoverformat=date_format)
    fig.update_yaxes(tickformat=",.0f")
    return date_axis(apply_chart_theme(fig, 400, legend=_HORIZONTAL_LEGEND, margin=dict(l=8, r=8, t=30, b=8)))


def revenue_by_category(revenue: pd.Series) -> go.Figure:
    """Participação por categoria em barras (comparam valores próximos melhor que uma rosca)."""
    data = revenue.sort_values()
    total = data.sum() or 1
    return _labeled_bars(
        data,
        [f"{_short_brl(v)} ({format_number(v / total * 100, 1)}%)" for v in data.values],
        340,
    )


def revenue_by_region(revenue: pd.Series) -> go.Figure:
    data = revenue.sort_values()
    return _labeled_bars(data, [_short_brl(v) for v in data.values], 340)


def top_products(df: pd.DataFrame, n: int = 8) -> go.Figure:
    data = df.groupby("product")["revenue"].sum().sort_values(ascending=True).tail(n)
    return _labeled_bars(data, [_short_brl(v) for v in data.values], 340)


def _labeled_bars(data: pd.Series, labels: list[str], height: int) -> go.Figure:
    """Barras horizontais de uma cor, com o valor escrito (o eixo numérico fica oculto)."""
    fig = go.Figure(
        go.Bar(
            x=data.values,
            y=data.index,
            orientation="h",
            marker=dict(color=palette().series_1),
            text=labels,
            hovertemplate="<b>%{y}</b><br>R$ %{x:,.2f}<extra></extra>",
            **_BAR_LABELS,
        )
    )
    # folga à direita para os rótulos das barras curtas, que ficam do lado de fora
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, data.max() * 1.3])
    fig.update_yaxes(ticksuffix="  ")
    return apply_chart_theme(fig, height, showlegend=False)


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
    p = palette()
    effects = [bridge.volume_effect, bridge.price_effect, bridge.mix_effect]
    fig = go.Figure(
        go.Waterfall(
            x=["Comparação", "Volume", "Preço", "Mix", "Período atual"],
            measure=["absolute", "relative", "relative", "relative", "total"],
            y=[bridge.previous_revenue, *effects, bridge.current_revenue],
            text=[_short_brl(v) for v in [bridge.previous_revenue, *effects, bridge.current_revenue]],
            textposition="outside",
            connector=dict(line=dict(color=p.neutral, width=1)),
            increasing=dict(marker=dict(color=p.positive)),
            decreasing=dict(marker=dict(color=p.negative)),
            totals=dict(marker=dict(color=p.neutral)),
            hovertemplate="%{x}: R$ %{y:,.2f}<extra></extra>",
        )
    )
    # Eixo ampliado em torno da variação: com o eixo a partir de zero, efeitos de poucos
    # por cento ficariam invisíveis ao lado dos totais. Os valores do eixo ficam visíveis
    # (e o subtítulo do cartão avisa) para que o recorte seja explícito.
    levels = [bridge.previous_revenue]
    for effect in effects:
        levels.append(levels[-1] + effect)
    low, high = min(levels), max(levels)
    pad = max((high - low) * 0.6, high * 0.02)
    fig.update_yaxes(range=[max(0, low - pad), high + pad], tickformat=",.0f")
    return apply_chart_theme(fig, 380, showlegend=False)


def bridge_by_segment(by_segment: pd.DataFrame) -> go.Figure:
    """Contribuição de cada categoria para a variação total (barras divergentes)."""
    p = palette()
    data = by_segment.sort_values("total")
    fig = go.Figure(
        go.Bar(
            x=data["total"],
            y=data.index,
            orientation="h",
            marker=dict(color=[p.positive if v >= 0 else p.negative for v in data["total"]]),
            text=[_short_brl(v) for v in data["total"]],
            customdata=data[["volume", "price", "mix"]].to_numpy(),
            hovertemplate=(
                "<b>%{y}</b><br>Variação: R$ %{x:,.2f}<br>Volume: R$ %{customdata[0]:,.2f}"
                "<br>Preço: R$ %{customdata[1]:,.2f}<br>Mix: R$ %{customdata[2]:,.2f}<extra></extra>"
            ),
            **_BAR_LABELS,
        )
    )
    fig.add_vline(x=0, line=dict(color=p.neutral, width=1))
    # Cada lado do zero ocupa só o espaço que os valores pedem (com um mínimo para o rótulo
    # de uma barra curta): com um eixo simétrico, uma queda pequena desperdiçaria metade do gráfico.
    gain, loss = max(data["total"].max(), 0.0), max(-data["total"].min(), 0.0)
    floor = max(gain, loss, 1.0) * 0.3
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False, range=[-max(loss * 1.35, floor), max(gain * 1.35, floor)]
    )
    fig.update_yaxes(ticksuffix="  ")
    return apply_chart_theme(fig, 380, showlegend=False)


def forecast(result: ForecastResult, history_days: int = 180) -> go.Figure:
    p = palette()
    history = result.history.iloc[-history_days:]
    fc = result.forecast
    band_x = list(fc.index) + list(fc.index[::-1])
    fig = go.Figure()
    for level, opacity in (("95", 0.12), ("80", 0.22)):
        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=list(fc[f"upper_{level}"]) + list(fc[f"lower_{level}"][::-1]),
                fill="toself",
                fillcolor=_rgba(p.series_2, opacity),
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
            line=dict(color=p.series_1, width=1.8),
            hovertemplate="R$ %{y:,.2f}<extra>Realizado</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc.index,
            y=fc["yhat"],
            name="Previsão",
            mode="lines",
            line=dict(color=p.series_2, width=2.5),
            hovertemplate="R$ %{y:,.2f}<extra>Previsão</extra>",
        )
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(hoverformat="%d/%m/%Y")
    fig.update_yaxes(tickformat=",.0f")
    return date_axis(apply_chart_theme(fig, 420, legend=_HORIZONTAL_LEGEND, margin=dict(l=8, r=8, t=30, b=8)))


def anomalies(series: pd.Series, found: pd.DataFrame, value_label: str) -> go.Figure:
    p = palette()
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series,
            name=value_label,
            mode="lines",
            line=dict(color=p.muted_line, width=1.2),
            hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f}<extra></extra>",
        )
    )
    for kind, color, symbol in (("pico", p.positive, "triangle-up"), ("queda", p.negative, "triangle-down")):
        points = found[found["kind"] == kind]
        fig.add_trace(
            go.Scatter(
                x=points.index,
                y=points["value"],
                name=f"Anomalia: {kind}",
                mode="markers",
                marker=dict(color=color, size=12, symbol=symbol, line=dict(color=p.surface, width=1.5)),
                customdata=points[["expected", "deviation_pct"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x|%d/%m/%Y}</b><br>Observado: %{y:,.0f}<br>Esperado: %{customdata[0]:,.0f}"
                    "<br>Desvio: %{customdata[1]:+.0f}%<extra></extra>"
                ),
            )
        )
    return date_axis(apply_chart_theme(fig, 380, legend=_HORIZONTAL_LEGEND, margin=dict(l=8, r=8, t=30, b=8)))


def weekday_effect(profile: pd.Series) -> go.Figure:
    p = palette()
    fig = go.Figure(
        go.Bar(
            x=profile.index,
            y=profile,
            marker=dict(color=[p.positive if v >= 0 else p.negative for v in profile]),
            hovertemplate="%{x}: %{y:+,.0f} em relação à média<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=p.neutral, width=1))
    fig.update_yaxes(tickformat="+,.0f")
    return apply_chart_theme(fig, 320, showlegend=False)


def monthly_index(index: pd.Series) -> go.Figure:
    p = palette()
    fig = go.Figure(
        go.Bar(
            x=index.index,
            y=index - 1,
            base=1,
            marker=dict(color=[p.positive if v >= 1 else p.negative for v in index]),
            hovertemplate="%{x}: índice %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=1, line=dict(color=p.neutral, width=1))
    return apply_chart_theme(fig, 320, showlegend=False)


def segments(summary: pd.DataFrame) -> go.Figure:
    p = palette()
    data = summary.iloc[::-1]  # primeiro segmento no topo
    fig = go.Figure()
    series = (("customers_pct", "% dos clientes", p.series_1), ("revenue_pct", "% da receita", p.series_2))
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
    fig.update_xaxes(ticksuffix="%")
    fig.update_yaxes(ticksuffix="  ")
    return apply_chart_theme(fig, 420, legend=_HORIZONTAL_LEGEND, margin=dict(l=8, r=8, t=30, b=8))


def rfm_heatmap(customers: pd.DataFrame) -> go.Figure:
    """Clientes por nota de recência x frequência (rampa de um só tom: mais escuro/claro = mais clientes)."""
    p = palette()
    grid = (
        customers.groupby(["f_score", "r_score"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    )
    steps = len(p.sequential) - 1
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[str(r) for r in grid.columns],
            y=[str(f) for f in grid.index],
            colorscale=[[i / steps, color] for i, color in enumerate(p.sequential)],
            text=grid.to_numpy(),
            texttemplate="%{text}",
            xgap=3,
            ygap=3,
            colorbar=dict(title=dict(text="Clientes"), thickness=10),
            hovertemplate="Recência %{x} · Frequência %{y}<br>%{z} clientes<extra></extra>",
        )
    )
    fig.update_xaxes(title="Recência (5 = comprou há pouco)", showgrid=False, dtick=1)
    fig.update_yaxes(title="Frequência (5 = compra muito)", showgrid=False, dtick=1)
    return apply_chart_theme(fig, 420)


def silhouette(scores: pd.Series, best_k: int) -> go.Figure:
    p = palette()
    fig = go.Figure(
        go.Scatter(
            x=scores.index,
            y=scores,
            mode="lines+markers",
            line=dict(color=p.series_1, width=2),
            marker=dict(size=10, color=[p.series_2 if k == best_k else p.series_1 for k in scores.index]),
            hovertemplate="k = %{x}: silhueta %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(dtick=1, title="Número de grupos (k)")
    return apply_chart_theme(fig, 280, showlegend=False)


def cluster_profile(clusters: pd.DataFrame, column: str, hover_format: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=clusters.index,
            y=clusters[column],
            marker=dict(color=palette().series_1),
            hovertemplate="%{x}: " + hover_format + "<extra></extra>",
        )
    )
    return apply_chart_theme(fig, 280, showlegend=False)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _short_brl(value: float) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000:
        return f"{sign}R$ {format_number(value / 1_000_000, 2)} mi"
    if value >= 1_000:
        return f"{sign}R$ {format_number(value / 1_000, 0)} mil"
    return f"{sign}R$ {format_number(value, 0)}"
