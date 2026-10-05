"""
Relatório estatístico local (fallback do agente de IA).

Usado quando não há chave do Gemini ou quando a API falha. Gera um
relatório com a mesma estrutura do Gemini a partir de estatística
(regressão linear + teste de tendência de Mann-Kendall) e regras de
negócio, sem nenhum custo ou chamada externa.
"""

from __future__ import annotations

import pandas as pd

from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs
from insight_engine.analytics.trends import MannKendall, declining_categories, fit_trend, mann_kendall
from insight_engine.formatting import format_brl, format_number, format_p_value, format_pct, format_usd

# Margem abaixo desta referência é sinalizada como ponto de atenção
HEALTHY_MARGIN_PCT = 20
# Volatilidade diária (desvio padrão dos retornos) considerada alta
HIGH_VOLATILITY_PCT = 4

_FOOTER = """---
*Relatório gerado automaticamente pelo motor estatístico local (regressão linear, teste de Mann-Kendall \
e regras de negócio), sem uso de API externa de IA generativa nesta execução.*"""


def sales_report(df: pd.DataFrame, kpis: SalesKPIs, period_label: str) -> str:
    daily_revenue = df.groupby(df["date"].dt.date)["revenue"].sum()
    trend = fit_trend(daily_revenue)
    mk = mann_kendall(daily_revenue)
    trend_word = {"alta": "crescimento", "queda": "queda", "sem tendência": "estabilidade"}[mk.direction]

    growth_txt = (
        f"{format_pct(kpis.revenue_growth_pct, signed=True)} em relação ao período anterior"
        if kpis.revenue_growth_pct is not None
        else "sem dado comparativo de período anterior disponível"
    )

    # categoria com pior desempenho relativo (possível gargalo)
    rev_by_cat = kpis.revenue_by_category
    worst_category = rev_by_cat.index[-1] if len(rev_by_cat) > 0 else "N/A"
    best_category = kpis.top_category

    # categorias com queda relevante entre a 1ª e a 2ª metade do período
    declining = declining_categories(df)
    declining_txt = (
        "; ".join(f"**{cat}** ({format_pct(pct, 0)})" for cat, pct in declining)
        if declining
        else "nenhuma categoria com queda relevante (>15%) identificada"
    )

    margin = kpis.margin_pct
    low_margin_flag = margin is not None and margin < HEALTHY_MARGIN_PCT

    action_items = [
        f"Priorizar investimento em **{best_category}**, categoria líder de receita, para sustentar o momentum.",
    ]
    if declining:
        action_items.append(
            f"Revisar mix de produtos e campanhas de **{declining[0][0]}**, que apresentou queda consistente "
            "entre a primeira e a segunda metade do período."
        )
    if mk.direction == "queda":
        action_items.append(
            "Investigar as causas da tendência de queda da receita diária, estatisticamente significativa "
            f"no período ({format_p_value(mk.p_value)})."
        )
    if margin is not None and low_margin_flag:
        action_items.append(
            f"Reavaliar política de custos/precificação: margem atual de {format_pct(margin)} está "
            "abaixo do saudável para o setor (referência: 20-30%)."
        )
    action_items.append(
        f"Direcionar esforços comerciais para a região **{kpis.top_region}**, com maior geração de receita, "
        "e investigar potencial nas regiões com menor participação."
    )
    action_items.append(
        "Estabelecer acompanhamento semanal dos KPIs deste dashboard para antecipar reversões de tendência."
    )
    action_md = _numbered(action_items)

    if margin is None or kpis.total_profit is None:
        profit_txt = "(lucro e margem indisponíveis, pois a base não informa o custo)"
        margin_txt = "não disponível, pois a base não informa o custo dos pedidos."
    else:
        profit_txt = f"com lucro de **{format_brl(kpis.total_profit)}** (margem de {format_pct(margin)})"
        margin_txt = (
            f"abaixo do ideal ({format_pct(margin)}), sinalizando pressão de custos ou descontos agressivos."
            if low_margin_flag
            else f"saudável, em {format_pct(margin)}, indicando controle de custos eficiente."
        )

    return f"""## Destaques do Período

No período analisado ({period_label}), a receita total somou **{format_brl(kpis.total_revenue)}**, \
{profit_txt}. \
Foram registrados **{format_number(kpis.n_orders)} pedidos**, totalizando \
{format_number(kpis.total_units)} unidades vendidas, com ticket médio de **{format_brl(kpis.avg_ticket)}**. \
A série diária de receita indica **{trend_word}** ({_trend_evidence(mk)}); a reta de regressão equivale a \
**{format_pct(trend.slope_pct, signed=True)}** ao longo do período, com variação de {growth_txt}. \
A categoria **{best_category}** lidera em receita, e a região **{kpis.top_region}** é a de maior \
representatividade comercial.

## Diagnóstico de Pontos Críticos / Gargalos

- **Categorias em queda:** {declining_txt}, comparando a primeira e a segunda metade do período selecionado.
- **Categoria de menor receita:** **{worst_category}**, candidata a revisão de estratégia comercial ou \
descontinuação.
- **Margem de lucro:** {margin_txt}
- **Concentração regional:** a receita está fortemente ligada à região {kpis.top_region}, o que representa \
risco de dependência caso o mercado local sofra retração.

## Plano de Ação Estratégico Sugerido

{action_md}

{_FOOTER}"""


def crypto_report(df: pd.DataFrame, kpis: CryptoKPIs, period_label: str) -> str:
    price_series = df.set_index("date")["price"]
    trend = fit_trend(price_series)
    mk = mann_kendall(price_series)
    trend_word = {"alta": "alta", "queda": "baixa", "sem tendência": "lateralização"}[mk.direction]

    volatility_flag = kpis.volatility_pct > HIGH_VOLATILITY_PCT
    drawdown_pct = (kpis.current_price - kpis.max_price) / kpis.max_price * 100 if kpis.max_price else 0

    action_items = [
        (
            "Reforçar disciplina de gestão de risco (stop-loss / dimensionamento de posição) "
            "dado o nível de volatilidade observado."
        )
        if volatility_flag
        else "Manter monitoramento de volatilidade; nível atual está dentro de faixas historicamente administráveis.",
        f"Acompanhar de perto o comportamento em torno da máxima do período ({format_usd(kpis.max_price)}) "
        "como possível resistência técnica.",
        f"Considerar a mínima do período ({format_usd(kpis.min_price)}) como referência de suporte para "
        "decisões de entrada.",
        "Cruzar esta análise de preço com indicadores on-chain e volume para confirmar a força da tendência "
        "antes de decisões relevantes.",
    ]
    volatility_txt = "elevada, exigindo cautela redobrada" if volatility_flag else "dentro de patamares administráveis"
    drawdown_side = "abaixo" if drawdown_pct < 0 else "acima"
    drawdown_txt = "possível correção em curso" if drawdown_pct < -10 else "proximidade de topo histórico recente"
    significance_txt = (
        "tendência estatisticamente significativa"
        if mk.significant
        else "sem tendência estatisticamente significativa, mercado possivelmente em consolidação"
    )

    return f"""## Destaques do Período

No período analisado ({period_label}), o ativo apresentou variação de \
**{format_pct(kpis.period_change_pct, 2, signed=True)}**, encerrando a **{format_usd(kpis.current_price)}**. \
A série de preços indica **{trend_word}** ({_trend_evidence(mk)}); a reta de regressão equivale a \
{format_pct(trend.slope_pct, signed=True)} no período. \
A máxima registrada foi **{format_usd(kpis.max_price)}** e a mínima **{format_usd(kpis.min_price)}**, \
com volume médio negociado de **{format_usd(kpis.avg_volume, 0)}**.

## Diagnóstico de Pontos Críticos / Gargalos

- **Volatilidade diária:** {format_pct(kpis.volatility_pct, 2)} ({volatility_txt}).
- **Distância da máxima:** o preço atual está {format_pct(abs(drawdown_pct))} {drawdown_side} da máxima do período, \
indicando {drawdown_txt}.
- **Força da tendência (Mann-Kendall):** tau = {format_number(mk.tau, 2)}, {format_p_value(mk.p_value)} — \
{significance_txt}.

## Plano de Ação Estratégico Sugerido

{_numbered(action_items)}

{_FOOTER}"""


def no_data_report(period_label: str, dataset_name: str) -> str:
    return f"""## Destaques do Período

Não há dados de **{dataset_name}** disponíveis para o período "{period_label}" com os filtros atuais.

## Diagnóstico de Pontos Críticos / Gargalos

- Ausência total de dados no recorte selecionado impede qualquer diagnóstico quantitativo.

## Plano de Ação Estratégico Sugerido

1. Ajustar o período ou os filtros de categoria/região selecionados.
2. Verificar a integridade da fonte de dados (base enviada ou API externa).
3. Caso o problema persista, enviar novamente o arquivo de dados."""


def _trend_evidence(mk: MannKendall) -> str:
    qualifier = "significativa" if mk.significant else "não significativa"
    return f"teste de Mann-Kendall: tendência {qualifier}, {format_p_value(mk.p_value)}"


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))
