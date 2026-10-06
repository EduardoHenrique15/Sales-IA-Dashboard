"""
Fatos analíticos enviados ao LLM (e usados pelo relatório local).

Tudo é calculado pelas camadas de `analytics` — as mesmas que alimentam a
tela. O LLM recebe só estes fatos, já formatados, e a checagem de números
(`guardrail`) confere o relatório contra exatamente este texto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from insight_engine.analytics.anomalies import daily_series, detect_anomalies
from insight_engine.analytics.customers import CustomerSegmentation, pareto_share, segment_customers
from insight_engine.analytics.forecasting import ForecastResult, InsufficientDataError, forecast_revenue
from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs
from insight_engine.analytics.trends import MannKendall, Trend, declining_categories, fit_trend, mann_kendall
from insight_engine.analytics.variance import RevenueBridge, revenue_bridge
from insight_engine.formatting import format_brl, format_number, format_p_value, format_pct, format_usd

logger = logging.getLogger(__name__)

FORECAST_HORIZON = 30
MAX_ANOMALIES = 5


@dataclass(frozen=True)
class SalesFacts:
    period_label: str
    kpis: SalesKPIs
    trend: Trend
    mann_kendall: MannKendall
    declining: list[tuple[str, float]]
    bridge: RevenueBridge | None
    # anomalias de pedidos dentro do período selecionado
    anomalies: pd.DataFrame
    # previsão a partir do último dia da base (não do período selecionado)
    forecast: ForecastResult | None
    segmentation: CustomerSegmentation | None

    def to_lines(self) -> list[str]:
        k = self.kpis
        lines = [
            f"Período analisado: {self.period_label}",
            f"Receita total: {format_brl(k.total_revenue)}",
        ]
        if k.total_profit is not None and k.margin_pct is not None:
            lines += [f"Lucro total: {format_brl(k.total_profit)}", f"Margem de lucro: {format_pct(k.margin_pct)}"]
        else:
            lines.append("Lucro e margem: não disponíveis (a base não informa o custo)")
        lines += [
            f"Pedidos: {format_number(k.n_orders)}",
            f"Unidades vendidas: {format_number(k.total_units)}",
            f"Ticket médio: {format_brl(k.avg_ticket)}",
            "Crescimento da receita vs período anterior de mesma duração: "
            + (format_pct(k.revenue_growth_pct, signed=True) if k.revenue_growth_pct is not None else "não disponível"),
        ]

        total = k.total_revenue or 1

        def shares(revenue: pd.Series) -> str:
            return "; ".join(f"{name} {format_brl(v)} ({format_pct(v / total * 100)})" for name, v in revenue.items())

        lines.append(f"Receita por categoria: {shares(k.revenue_by_category)}")
        lines.append(f"Receita por região: {shares(k.revenue_by_region)}")

        mk = self.mann_kendall
        direction = {"alta": "alta", "queda": "queda", "sem tendência": "sem tendência significativa"}[mk.direction]
        lines.append(
            f"Tendência da receita diária no período (teste de Mann-Kendall): {direction} "
            f"({format_p_value(mk.p_value)}); reta de regressão equivale a "
            f"{format_pct(self.trend.slope_pct, signed=True)}"
        )
        if self.declining:
            lines.append(
                "Categorias com queda de receita entre a 1ª e a 2ª metade do período: "
                + "; ".join(f"{c} ({format_pct(p, 0)})" for c, p in self.declining)
            )
        else:
            lines.append("Nenhuma categoria com queda maior que 15% entre a 1ª e a 2ª metade do período")

        if self.bridge is not None:
            b = self.bridge
            lines.append(
                f"Variação da receita vs período anterior: {format_brl(b.total_change)}, decomposta em "
                f"efeito volume {format_brl(b.volume_effect)}, efeito preço {format_brl(b.price_effect)} e "
                f"efeito mix {format_brl(b.mix_effect)}"
            )

        if self.anomalies.empty:
            lines.append("Anomalias no número de pedidos no período: nenhuma")
        else:
            items = [
                f"{day:%d/%m/%Y} ({row.kind}, {format_number(row.value)} pedidos vs "
                f"{format_number(row.expected)} esperados, {format_pct(row.deviation_pct, 0, signed=True)})"
                for day, row in self.anomalies.head(MAX_ANOMALIES).iterrows()
            ]
            lines.append("Anomalias no número de pedidos no período: " + "; ".join(items))

        if self.forecast is not None:
            f = self.forecast
            last_day = f.history.index[-1]
            lines.append(
                f"Previsão de receita para os {f.horizon} dias após {last_day:%d/%m/%Y}: "
                f"{format_brl(f.forecast['yhat'].sum())} (erro típico do total no backtesting: "
                f"± {format_pct(f.total_error_pct)}; modelo: {f.best_model})"
            )

        if self.segmentation is not None:
            seg = self.segmentation.segments
            top = ", ".join(
                f"{name}: {format_number(row.customers)} clientes, {format_pct(row.revenue_pct)} da receita"
                for name, row in seg.iterrows()
                if name in ("Campeões", "Em risco", "Perdidos")
            )
            customers = self.segmentation.customers
            lines.append(
                f"Clientes (histórico completo): {format_number(len(customers))}; os 20% que mais compram "
                f"geram {format_pct(pareto_share(customers))} da receita; segmentos RFM: {top}"
            )
        return lines


@dataclass(frozen=True)
class CryptoFacts:
    period_label: str
    coin_name: str
    kpis: CryptoKPIs
    trend: Trend
    mann_kendall: MannKendall

    @property
    def drawdown_pct(self) -> float:
        k = self.kpis
        return (k.current_price - k.max_price) / k.max_price * 100 if k.max_price else 0.0

    def to_lines(self) -> list[str]:
        k, mk = self.kpis, self.mann_kendall
        direction = {"alta": "alta", "queda": "baixa", "sem tendência": "sem tendência significativa"}[mk.direction]
        return [
            f"Ativo: {self.coin_name}; período: {self.period_label}",
            f"Preço atual: {format_usd(k.current_price)}",
            f"Variação no período: {format_pct(k.period_change_pct, 2, signed=True)}",
            f"Máxima do período: {format_usd(k.max_price)}; mínima: {format_usd(k.min_price)}",
            f"Distância do preço atual para a máxima: {format_pct(self.drawdown_pct, 1, signed=True)}",
            f"Volume médio negociado: {format_usd(k.avg_volume, 0)}",
            f"Volatilidade diária (desvio padrão dos retornos): {format_pct(k.volatility_pct, 2)}",
            f"Tendência do preço (Mann-Kendall): {direction} ({format_p_value(mk.p_value)}); "
            f"reta de regressão equivale a {format_pct(self.trend.slope_pct, signed=True)}",
        ]


def build_sales_facts(
    history: pd.DataFrame,
    period: pd.DataFrame,
    previous: pd.DataFrame,
    kpis: SalesKPIs,
    period_label: str,
    has_customers: bool = True,
) -> SalesFacts:
    """Reúne os fatos do período selecionado.

    `history` é a base inteira já com os filtros de categoria/região (sem filtro de
    data): é dela que saem anomalias, previsão e segmentação de clientes.
    """
    daily = period.groupby(period["date"].dt.date)["revenue"].sum()

    anomalies = pd.DataFrame()
    try:
        found = detect_anomalies(daily_series(history, "orders"))
        if not period.empty:
            in_period = (found.index >= period["date"].min()) & (found.index <= period["date"].max())
            anomalies = found[in_period]
    except ValueError:
        logger.info("Histórico curto demais para detectar anomalias")

    forecast = None
    try:
        forecast = forecast_revenue(daily_series(history, "revenue"), FORECAST_HORIZON)
    except InsufficientDataError:
        logger.info("Histórico curto demais para a previsão")

    segmentation = None
    if has_customers:
        try:
            segmentation = segment_customers(history)
        except ValueError:
            logger.info("Clientes insuficientes para a segmentação")

    bridge = revenue_bridge(period, previous) if not period.empty and not previous.empty else None

    return SalesFacts(
        period_label=period_label,
        kpis=kpis,
        trend=fit_trend(daily),
        mann_kendall=mann_kendall(daily),
        declining=declining_categories(period) if not period.empty else [],
        bridge=bridge,
        anomalies=anomalies,
        forecast=forecast,
        segmentation=segmentation,
    )


def build_crypto_facts(df: pd.DataFrame, kpis: CryptoKPIs, period_label: str, coin_name: str) -> CryptoFacts:
    prices = df.set_index("date")["price"] if not df.empty else pd.Series(dtype=float)
    return CryptoFacts(period_label, coin_name, kpis, fit_trend(prices), mann_kendall(prices))
