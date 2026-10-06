"""
Relatório estatístico local (fallback do agente de IA).

Usado quando não há chave do Gemini ou quando a API falha. Monta o mesmo
`ExecutiveReport` que o LLM devolveria, a partir dos mesmos fatos
(`context`), com regras de negócio — sem custo e sem chamada externa.
"""

from __future__ import annotations

from insight_engine.ai.context import CryptoFacts, SalesFacts
from insight_engine.ai.report import Action, ExecutiveReport, Risk
from insight_engine.formatting import format_brl, format_number, format_p_value, format_pct, format_usd

# Margem abaixo desta referência é sinalizada como ponto de atenção
HEALTHY_MARGIN_PCT = 20
# Volatilidade diária (desvio padrão dos retornos) considerada alta
HIGH_VOLATILITY_PCT = 4
# Participação da região líder acima da qual há risco de concentração
CONCENTRATION_PCT = 30

FOOTER = (
    "Relatório gerado pelo motor estatístico local (regressão linear, teste de Mann-Kendall, "
    "decomposição volume/preço/mix e regras de negócio), sem IA generativa nesta execução."
)


def sales_report(facts: SalesFacts) -> ExecutiveReport:
    k = facts.kpis
    mk = facts.mann_kendall
    total = k.total_revenue or 1
    region_share = float(k.revenue_by_region.iloc[0]) / total * 100 if len(k.revenue_by_region) else 0.0

    trend_word = {"alta": "crescimento", "queda": "queda", "sem tendência": "estabilidade"}[mk.direction]
    growth = (
        f"{format_pct(k.revenue_growth_pct, signed=True)} vs o período anterior"
        if k.revenue_growth_pct is not None
        else "sem período anterior comparável"
    )
    headline = f"Receita de {format_brl(k.total_revenue)} no período ({growth}), com {trend_word} da receita diária."

    sales_line = (
        f"Receita de {format_brl(k.total_revenue)} em {format_number(k.n_orders)} pedidos "
        f"(ticket médio de {format_brl(k.avg_ticket)})"
    )
    if k.total_profit is not None and k.margin_pct is not None:
        sales_line += f", com lucro de {format_brl(k.total_profit)} e margem de {format_pct(k.margin_pct)}"
    highlights = [
        sales_line + ".",
        f"{k.top_category} lidera com {format_pct(k.top_category_revenue / total * 100)} da receita; "
        f"{k.top_region} é a principal região ({format_pct(region_share)}).",
    ]
    if facts.bridge is not None:
        b = facts.bridge
        highlights.append(
            f"A variação de {format_brl(b.total_change)} vem de volume ({format_brl(b.volume_effect)}), "
            f"preço ({format_brl(b.price_effect)}) e mix ({format_brl(b.mix_effect)})."
        )
    highlights.append(
        f"Tendência diária: {trend_word} (Mann-Kendall, {format_p_value(mk.p_value)}; "
        f"{'estatisticamente significativa' if mk.significant else 'não significativa'})."
    )

    risks: list[Risk] = []
    actions: list[Action] = []

    if facts.declining:
        category, change = facts.declining[0]
        risks.append(
            Risk(
                title=f"Queda em {category}",
                evidence=f"Receita de {category} variou {format_pct(change, 0)} entre a 1ª e a 2ª metade do período.",
                severity="alta" if change < -30 else "média",
            )
        )
        actions.append(
            Action(
                action=f"Revisar mix de produtos e campanhas de {category}",
                rationale=f"Foi a categoria com maior queda no período ({format_pct(change, 0)}).",
                priority="alta",
            )
        )
    if mk.direction == "queda":
        risks.append(
            Risk(
                title="Tendência de queda da receita",
                evidence=f"Queda estatisticamente significativa na receita diária ({format_p_value(mk.p_value)}).",
                severity="alta",
            )
        )
    if (
        facts.bridge is not None
        and facts.bridge.mix_effect < 0
        and abs(facts.bridge.mix_effect) > 0.1 * abs(facts.bridge.total_change or 1)
    ):
        risks.append(
            Risk(
                title="Mix de vendas menos favorável",
                evidence="A migração entre categorias reduziu a receita em "
                f"{format_brl(abs(facts.bridge.mix_effect))}.",
                severity="média",
            )
        )
        actions.append(
            Action(
                action="Reforçar a exposição das categorias de maior preço médio",
                rationale=f"O efeito mix foi de {format_brl(facts.bridge.mix_effect)} no período.",
                priority="média",
            )
        )
    if not facts.anomalies.empty:
        day, row = next(iter(facts.anomalies.iterrows()))
        risks.append(
            Risk(
                title=f"Dia atípico em {day:%d/%m/%Y}",
                evidence=f"{format_number(row.value)} pedidos contra {format_number(row.expected)} esperados "
                f"({format_pct(row.deviation_pct, 0, signed=True)}).",
                severity="alta" if row.kind == "queda" else "baixa",
            )
        )
        if row.kind == "queda":
            actions.append(
                Action(
                    action=f"Investigar a causa da queda de pedidos em {day:%d/%m/%Y}",
                    rationale="Quedas bruscas costumam indicar incidente operacional (site, pagamento, estoque).",
                    priority="alta",
                )
            )
    if k.margin_pct is not None and k.margin_pct < HEALTHY_MARGIN_PCT:
        risks.append(
            Risk(
                title="Margem abaixo do saudável",
                evidence=f"Margem de {format_pct(k.margin_pct)}, abaixo da referência de 20-30%.",
                severity="alta",
            )
        )
        actions.append(
            Action(
                action="Reavaliar política de custos e descontos",
                rationale=f"A margem está em {format_pct(k.margin_pct)}.",
                priority="alta",
            )
        )
    if region_share > CONCENTRATION_PCT:
        risks.append(
            Risk(
                title="Concentração regional",
                evidence=f"{k.top_region} responde por {format_pct(region_share)} da receita.",
                severity="média" if region_share > 50 else "baixa",
            )
        )
    if facts.segmentation is not None:
        segments = facts.segmentation.segments
        if "Em risco" in segments.index:
            at_risk = segments.loc["Em risco"]
            actions.append(
                Action(
                    action="Campanha de reativação para clientes 'Em risco'",
                    rationale=f"São {format_number(at_risk.customers)} clientes que já geraram "
                    f"{format_pct(at_risk.revenue_pct)} da receita e pararam de comprar.",
                    priority="alta" if at_risk.revenue_pct > 15 else "média",
                )
            )
    if facts.forecast is not None:
        f = facts.forecast
        actions.append(
            Action(
                action=f"Planejar estoque e metas com base na previsão de {format_brl(f.forecast['yhat'].sum())}",
                rationale=f"Previsão para os próximos {f.horizon} dias, "
                f"com erro típico de ± {format_pct(f.total_error_pct)}.",
                priority="média",
            )
        )
    actions.append(
        Action(
            action=f"Proteger a liderança de {k.top_category}",
            rationale=f"A categoria concentra {format_pct(k.top_category_revenue / total * 100)} da receita.",
            priority="média",
        )
    )

    if not risks:
        risks.append(
            Risk(
                title="Sem gargalos relevantes",
                evidence="Nenhuma queda de categoria, anomalia ou margem baixa identificada no período.",
                severity="baixa",
            )
        )
    return ExecutiveReport(headline=headline, highlights=highlights[:4], risks=risks[:3], actions=actions[:5])


def crypto_report(facts: CryptoFacts) -> ExecutiveReport:
    k, mk = facts.kpis, facts.mann_kendall
    trend_word = {"alta": "alta", "queda": "baixa", "sem tendência": "lateralização"}[mk.direction]
    high_volatility = k.volatility_pct > HIGH_VOLATILITY_PCT

    risks = [
        Risk(
            title="Volatilidade elevada" if high_volatility else "Volatilidade administrável",
            evidence=f"Desvio padrão diário dos retornos de {format_pct(k.volatility_pct, 2)}.",
            severity="alta" if high_volatility else "baixa",
        ),
        Risk(
            title="Distância da máxima do período",
            evidence=f"Preço {format_pct(facts.drawdown_pct, 1, signed=True)} em relação à máxima de "
            f"{format_usd(k.max_price)}.",
            severity="média" if facts.drawdown_pct < -10 else "baixa",
        ),
    ]
    actions = [
        Action(
            action="Reforçar gestão de risco (stop-loss e tamanho de posição)"
            if high_volatility
            else "Manter o monitoramento da volatilidade",
            rationale=f"Volatilidade diária de {format_pct(k.volatility_pct, 2)}.",
            priority="alta" if high_volatility else "baixa",
        ),
        Action(
            action=f"Usar a máxima ({format_usd(k.max_price)}) e a mínima ({format_usd(k.min_price)}) como referências",
            rationale="Níveis recentes de resistência e suporte.",
            priority="média",
        ),
        Action(
            action="Confirmar a tendência com volume e indicadores on-chain antes de decisões relevantes",
            rationale=f"O teste de Mann-Kendall indica {trend_word} ({format_p_value(mk.p_value)}).",
            priority="média",
        ),
    ]
    return ExecutiveReport(
        headline=f"{facts.coin_name} variou {format_pct(k.period_change_pct, 2, signed=True)} no período e "
        f"encerrou a {format_usd(k.current_price)}.",
        highlights=[
            f"Preço atual de {format_usd(k.current_price)}; máxima de {format_usd(k.max_price)} e mínima de "
            f"{format_usd(k.min_price)}.",
            f"Tendência de {trend_word} (Mann-Kendall, {format_p_value(mk.p_value)}).",
            f"Volume médio negociado de {format_usd(k.avg_volume, 0)}.",
        ],
        risks=risks,
        actions=actions,
    )


def no_data_report(period_label: str, dataset_name: str) -> ExecutiveReport:
    return ExecutiveReport(
        headline=f"Não há dados de {dataset_name} para o período {period_label} com os filtros atuais.",
        highlights=["Nenhum pedido no recorte selecionado."],
        risks=[Risk(title="Sem dados", evidence="O recorte não tem vendas para analisar.", severity="baixa")],
        actions=[
            Action(action="Ajustar o período ou os filtros", rationale="O recorte atual está vazio.", priority="alta"),
            Action(
                action="Verificar a fonte de dados",
                rationale="Confirme se a base enviada cobre o período desejado.",
                priority="média",
            ),
        ],
    )
