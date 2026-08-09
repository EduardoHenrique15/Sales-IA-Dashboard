"""
ai_agent.py
================================================================
Agente responsável por transformar números em narrativa.

Estratégia de duas camadas (produção-ready):

  1. CAMADA PRIMÁRIA (Google Gemini API)
     Se houver uma API key configurada, o agente monta um prompt
     rico com os KPIs já calculados e pede ao modelo um Relatório
     Executivo estruturado em Markdown.

  2. CAMADA DE FALLBACK (estatística, 100% local, sem custo)
     Se a API key não existir, se a cota estourar (HTTP 429), se
     houver timeout, ou qualquer outro erro de rede, o agente NÃO
     quebra o dashboard: ele gera o mesmo relatório usando
     regressão linear (scikit-learn) sobre a série temporal para
     detectar tendência, além de comparações estatísticas simples
     para achar gargalos. O usuário final não percebe diferença
     na estrutura do relatório — só a "voz" muda.

Isso é exatamente o tipo de resiliência que se espera de um
sistema de IA em produção: nunca deixar a ausência/instabilidade
de uma API terceira quebrar a experiência do usuário.
================================================================
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# A dependência do Gemini é opcional: se o pacote não estiver instalado,
# o agente simplesmente cai direto no fallback estatístico.
try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


class AIAgentError(Exception):
    """Erro conhecido e tratado do agente (usado só para logging interno)."""


# ==================================================================
# FUNÇÃO PÚBLICA PRINCIPAL
# ==================================================================
def generate_executive_summary(
    df: pd.DataFrame,
    kpis: dict,
    period_label: str,
    dataset_name: str = "Vendas",
    api_key: str | None = None,
) -> tuple[str, str]:
    """Gera o Relatório Executivo em Markdown.

    Retorna uma tupla (markdown_text, source), onde `source` é
    "gemini" ou "fallback_estatistico" — usado pelo app para
    exibir um selo indicando qual "motor" gerou o texto.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")

    if df.empty:
        return _no_data_report(period_label, dataset_name), "fallback_estatistico"

    if api_key and _GENAI_AVAILABLE:
        try:
            report = _generate_with_gemini(df, kpis, period_label, dataset_name, api_key)
            return report, "gemini"
        except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha da API
            # Não propaga o erro para a UI: registra e cai no fallback.
            print(f"[ai_agent] Falha na API Gemini, usando fallback. Detalhe: {exc}")

    report = _generate_fallback_report(df, kpis, period_label, dataset_name)
    return report, "fallback_estatistico"


# ==================================================================
# CAMADA 1: GEMINI API
# ==================================================================
def _generate_with_gemini(df, kpis, period_label, dataset_name, api_key, max_retries: int = 2) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = _build_prompt(kpis, period_label, dataset_name)

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
            if not text:
                raise AIAgentError("Resposta vazia da API Gemini")
            return text
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            error_str = str(exc).lower()
            is_rate_limit = "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str
            if is_rate_limit and attempt < max_retries:
                time.sleep(2 ** attempt)  # backoff exponencial
                continue
            raise

    raise last_exception  # pragma: no cover


def _build_prompt(kpis: dict, period_label: str, dataset_name: str) -> str:
    if dataset_name == "Vendas":
        growth_txt = (
            f"{kpis['revenue_growth_pct']:.1f}%" if kpis.get("revenue_growth_pct") is not None
            else "não disponível (sem período anterior comparável)"
        )
        kpi_block = f"""
- Receita total: R$ {kpis['total_revenue']:,.2f}
- Lucro total: R$ {kpis['total_profit']:,.2f}
- Margem de lucro: {kpis['margin_pct']:.1f}%
- Unidades vendidas: {kpis['total_units']}
- Ticket médio: R$ {kpis['avg_ticket']:,.2f}
- Categoria líder: {kpis['top_category']} (R$ {kpis['top_category_revenue']:,.2f})
- Região líder: {kpis['top_region']}
- Crescimento da receita vs período anterior: {growth_txt}
""".strip()
    else:
        kpi_block = f"""
- Preço atual: US$ {kpis['current_price']:,.2f}
- Variação no período: {kpis['period_change_pct']:.2f}%
- Máxima do período: US$ {kpis['max_price']:,.2f}
- Mínima do período: US$ {kpis['min_price']:,.2f}
- Volume médio negociado: US$ {kpis['avg_volume']:,.0f}
- Volatilidade diária (desvio padrão dos retornos): {kpis['volatility_pct']:.2f}%
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


# ==================================================================
# CAMADA 2: FALLBACK ESTATÍSTICO (scikit-learn + regras)
# ==================================================================
def _generate_fallback_report(df: pd.DataFrame, kpis: dict, period_label: str, dataset_name: str) -> str:
    if dataset_name == "Vendas":
        return _fallback_sales_report(df, kpis, period_label)
    return _fallback_crypto_report(df, kpis, period_label)


def _fit_trend(daily_series: pd.Series) -> tuple[float, float]:
    """Ajusta uma regressão linear simples (dia -> valor) e retorna
    (inclinação normalizada em % ao período, R² do ajuste)."""
    if len(daily_series) < 2:
        return 0.0, 0.0

    x = np.arange(len(daily_series)).reshape(-1, 1)
    y = daily_series.values

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    r2 = model.score(x, y)

    mean_val = y.mean() if y.mean() != 0 else 1
    slope_pct_total = (slope * len(daily_series)) / mean_val * 100
    return float(slope_pct_total), float(r2)


def _fallback_sales_report(df: pd.DataFrame, kpis: dict, period_label: str) -> str:
    daily_revenue = df.groupby(df["date"].dt.date)["revenue"].sum()
    trend_pct, r2 = _fit_trend(daily_revenue)

    trend_word = "crescimento" if trend_pct > 1 else ("queda" if trend_pct < -1 else "estabilidade")
    confidence = "alta" if r2 > 0.5 else ("moderada" if r2 > 0.2 else "baixa")

    growth_txt = (
        f"{kpis['revenue_growth_pct']:+.1f}% em relação ao período anterior"
        if kpis.get("revenue_growth_pct") is not None
        else "sem dado comparativo de período anterior disponível"
    )

    # detecção de categoria com pior desempenho relativo (possível gargalo)
    rev_by_cat = kpis["revenue_by_category"]
    worst_category = rev_by_cat.index[-1] if len(rev_by_cat) > 0 else "N/A"
    best_category = kpis["top_category"]

    # compara primeira metade vs segunda metade do período por categoria
    mid_point = df["date"].min() + (df["date"].max() - df["date"].min()) / 2
    first_half = df[df["date"] <= mid_point]
    second_half = df[df["date"] > mid_point]

    declining_categories = []
    if not first_half.empty and not second_half.empty:
        rev_first = first_half.groupby("category")["revenue"].sum()
        rev_second = second_half.groupby("category")["revenue"].sum()
        for cat in rev_first.index:
            v1, v2 = rev_first.get(cat, 0), rev_second.get(cat, 0)
            if v1 > 0 and (v2 - v1) / v1 < -0.15:
                declining_categories.append((cat, (v2 - v1) / v1 * 100))

    declining_txt = (
        "; ".join(f"**{cat}** ({pct:.0f}%)" for cat, pct in declining_categories)
        if declining_categories else "nenhuma categoria com queda relevante (>15%) identificada"
    )

    low_margin_flag = kpis["margin_pct"] < 20
    ticket_flag = kpis["avg_ticket"] < (kpis["total_revenue"] / max(kpis["n_orders"], 1)) * 0.01  # placeholder seguro

    action_items = [
        f"Priorizar investimento em **{best_category}**, categoria líder de receita, para sustentar o momentum.",
    ]
    if declining_categories:
        top_decline_cat = declining_categories[0][0]
        action_items.append(
            f"Revisar mix de produtos e campanhas de **{top_decline_cat}**, que apresentou queda consistente "
            "entre a primeira e a segunda metade do período."
        )
    if low_margin_flag:
        action_items.append(
            f"Reavaliar política de custos/precificação: margem atual de {kpis['margin_pct']:.1f}% está "
            "abaixo do saudável para o setor (referência: 20-30%)."
        )
    action_items.append(
        f"Direcionar esforços comerciais para a região **{kpis['top_region']}**, com maior geração de receita, "
        "e investigar potencial nas regiões com menor participação."
    )
    action_items.append(
        "Estabelecer acompanhamento semanal dos KPIs deste dashboard para antecipar reversões de tendência."
    )

    action_md = "\n".join(f"{i+1}. {item}" for i, item in enumerate(action_items))

    return f"""## Destaques do Período

No período analisado ({period_label}), a receita total somou **R$ {kpis['total_revenue']:,.2f}**, \
com lucro de **R$ {kpis['total_profit']:,.2f}** (margem de {kpis['margin_pct']:.1f}%). \
Foram registrados **{kpis['n_orders']} pedidos**, totalizando {kpis['total_units']} unidades vendidas, \
com ticket médio de **R$ {kpis['avg_ticket']:,.2f}**. A análise de tendência (regressão linear sobre a \
série diária de receita, confiança {confidence}) aponta **{trend_word}** de aproximadamente \
**{trend_pct:+.1f}%** ao longo do período, com variação de {growth_txt}. \
A categoria **{best_category}** lidera em receita, e a região **{kpis['top_region']}** é a de maior \
representatividade comercial.

## Diagnóstico de Pontos Críticos / Gargalos

- **Categorias em queda:** {declining_txt}, comparando a primeira e a segunda metade do período selecionado.
- **Categoria de menor receita:** **{worst_category}**, candidata a revisão de estratégia comercial ou descontinuação.
- **Margem de lucro:** {"abaixo do ideal (" + f"{kpis['margin_pct']:.1f}%" + "), sinalizando pressão de custos ou descontos agressivos." if low_margin_flag else f"saudável, em {kpis['margin_pct']:.1f}%, indicando controle de custos eficiente."}
- **Concentração regional:** a receita está fortemente ligada à região {kpis['top_region']}, o que representa risco de dependência caso o mercado local sofra retração.

## Plano de Ação Estratégico Sugerido

{action_md}

---
*Relatório gerado automaticamente pelo motor estatístico local (regressão linear + regras de negócio), \
sem uso de API externa de IA generativa nesta execução.*"""


def _fallback_crypto_report(df: pd.DataFrame, kpis: dict, period_label: str) -> str:
    price_series = df.set_index("date")["price"]
    trend_pct, r2 = _fit_trend(price_series)
    confidence = "alta" if r2 > 0.5 else ("moderada" if r2 > 0.2 else "baixa")
    trend_word = "alta" if trend_pct > 1 else ("baixa" if trend_pct < -1 else "lateralização")

    volatility_flag = kpis["volatility_pct"] > 4
    drawdown_pct = (kpis["current_price"] - kpis["max_price"]) / kpis["max_price"] * 100 if kpis["max_price"] else 0

    action_items = [
        "Reforçar disciplina de gestão de risco (stop-loss / dimensionamento de posição) dado o nível de volatilidade observado."
        if volatility_flag else
        "Manter monitoramento de volatilidade; nível atual está dentro de faixas historicamente administráveis.",
        f"Acompanhar de perto o comportamento em torno da máxima do período (US$ {kpis['max_price']:,.2f}) como possível resistência técnica.",
        f"Considerar a mínima do período (US$ {kpis['min_price']:,.2f}) como referência de suporte para decisões de entrada.",
        "Cruzar esta análise de preço com indicadores on-chain e volume para confirmar a força da tendência antes de decisões relevantes.",
    ]
    action_md = "\n".join(f"{i+1}. {item}" for i, item in enumerate(action_items))

    return f"""## Destaques do Período

No período analisado ({period_label}), o ativo apresentou variação de **{kpis['period_change_pct']:+.2f}%**, \
encerrando a **US$ {kpis['current_price']:,.2f}**. A regressão linear sobre a série de preços (confiança \
{confidence}) indica tendência de **{trend_word}**, com inclinação equivalente a {trend_pct:+.1f}% no período. \
A máxima registrada foi **US$ {kpis['max_price']:,.2f}** e a mínima **US$ {kpis['min_price']:,.2f}**, com \
volume médio negociado de **US$ {kpis['avg_volume']:,.0f}**.

## Diagnóstico de Pontos Críticos / Gargalos

- **Volatilidade diária:** {kpis['volatility_pct']:.2f}% ({"elevada, exigindo cautela redobrada" if volatility_flag else "dentro de patamares administráveis"}).
- **Distância da máxima:** o preço atual está {abs(drawdown_pct):.1f}% {"abaixo" if drawdown_pct < 0 else "acima"} da máxima do período, indicando {"possível correção em curso" if drawdown_pct < -10 else "proximidade de topo histórico recente"}.
- **Confiança da tendência (R²):** {r2:.2f} — {"tendência bem definida" if r2 > 0.5 else "sinal de tendência fraco, mercado possivelmente em consolidação"}.

## Plano de Ação Estratégico Sugerido

{action_md}

---
*Relatório gerado automaticamente pelo motor estatístico local (regressão linear + regras de negócio), \
sem uso de API externa de IA generativa nesta execução.*"""


def _no_data_report(period_label: str, dataset_name: str) -> str:
    return f"""## Destaques do Período

Não há dados de **{dataset_name}** disponíveis para o período "{period_label}" com os filtros atuais.

## Diagnóstico de Pontos Críticos / Gargalos

- Ausência total de dados no recorte selecionado impede qualquer diagnóstico quantitativo.

## Plano de Ação Estratégico Sugerido

1. Ajustar o período ou os filtros de categoria/região selecionados.
2. Verificar a integridade da fonte de dados (cache local ou API externa).
3. Caso o problema persista, forçar a atualização/regeneração da base de dados."""
