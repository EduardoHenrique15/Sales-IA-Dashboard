"""
Prompts enviados ao LLM.

O modelo recebe apenas números já calculados por `analytics.kpis`
(a mesma fonte dos cards da tela) e é instruído a não inventar
valores fora desse bloco.
"""

from __future__ import annotations

from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs
from insight_engine.formatting import format_brl, format_number, format_pct, format_usd


def build_prompt(kpis: SalesKPIs | CryptoKPIs, period_label: str, dataset_name: str) -> str:
    """Monta o prompt do relatório executivo a partir dos KPIs já calculados."""
    if isinstance(kpis, SalesKPIs):
        growth_txt = (
            format_pct(kpis.revenue_growth_pct, signed=True)
            if kpis.revenue_growth_pct is not None
            else "não disponível (sem período anterior comparável)"
        )
        no_cost = "não disponível (a base não informa o custo)"
        profit_txt = format_brl(kpis.total_profit) if kpis.total_profit is not None else no_cost
        margin_txt = format_pct(kpis.margin_pct) if kpis.margin_pct is not None else no_cost
        kpi_block = f"""
- Receita total: {format_brl(kpis.total_revenue)}
- Lucro total: {profit_txt}
- Margem de lucro: {margin_txt}
- Unidades vendidas: {format_number(kpis.total_units)}
- Ticket médio: {format_brl(kpis.avg_ticket)}
- Categoria líder: {kpis.top_category} ({format_brl(kpis.top_category_revenue)})
- Região líder: {kpis.top_region}
- Crescimento da receita vs período anterior: {growth_txt}
""".strip()
    else:
        kpi_block = f"""
- Preço atual: {format_usd(kpis.current_price)}
- Variação no período: {format_pct(kpis.period_change_pct, 2, signed=True)}
- Máxima do período: {format_usd(kpis.max_price)}
- Mínima do período: {format_usd(kpis.min_price)}
- Volume médio negociado: {format_usd(kpis.avg_volume, 0)}
- Volatilidade diária (desvio padrão dos retornos): {format_pct(kpis.volatility_pct, 2)}
""".strip()

    return f"""
Você é um analista de dados sênior escrevendo um relatório executivo direto ao ponto
para a diretoria de uma empresa. Use os dados abaixo (já calculados e verificados)
referentes ao período "{period_label}" (dataset: {dataset_name}).

DADOS:
{kpi_block}

Escreva um relatório em Markdown, em português do Brasil, com EXATAMENTE estas
três seções (use "##" como cabeçalho):

## Destaques do Período
(resuma os KPIs mais importantes e a tendência geral em 3-4 frases objetivas)

## Diagnóstico de Pontos Críticos / Gargalos
(aponte 2-3 riscos, quedas ou ineficiências concretas, com base nos números)

## Plano de Ação Estratégico Sugerido
(liste de 3 a 5 ações práticas e priorizadas, em formato de lista)

Regras: seja direto, use números concretos do bloco de dados acima, não invente
números que não estejam ali, e não escreva nada fora dessas três seções.
""".strip()
