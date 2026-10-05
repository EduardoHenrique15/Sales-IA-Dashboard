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

import logging
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import get_gemini_api_key, get_gemini_fallback_model, get_gemini_model
from formatting import format_brl, format_number, format_pct, format_usd

# A dependência do Gemini é opcional: se o pacote não estiver instalado,
# o agente simplesmente cai direto no fallback estatístico.
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


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
) -> tuple[str, str, str | None]:
    """Gera o Relatório Executivo em Markdown.

    Retorna uma tupla (markdown_text, source, fallback_reason):
      - `source` é "gemini" ou "fallback_estatistico", usado pelo app
        para exibir um selo indicando qual "motor" gerou o texto;
      - `fallback_reason` explica por que o Gemini não foi usado
        quando havia uma chave configurada (None nos demais casos).
    """
    api_key = api_key or get_gemini_api_key()

    if df.empty:
        return _no_data_report(period_label, dataset_name), "fallback_estatistico", None

    fallback_reason = None
    if api_key and _GENAI_AVAILABLE:
        try:
            report = _generate_with_gemini(df, kpis, period_label, dataset_name, api_key)
            return report, "gemini", None
        except Exception as exc:  # noqa: BLE001 - queremos capturar qualquer falha da API
            # Não propaga o erro para a UI: registra e cai no fallback.
            logger.warning("Falha na API Gemini, usando fallback. Detalhe: %s", exc)
            fallback_reason = _describe_gemini_error(exc)
    elif api_key:
        fallback_reason = "o pacote `google-genai` não está instalado."

    report = _generate_fallback_report(df, kpis, period_label, dataset_name)
    return report, "fallback_estatistico", fallback_reason


def _describe_gemini_error(exc: Exception) -> str:
    """Traduz uma falha do Gemini em uma mensagem curta para a interface."""
    code = getattr(exc, "code", None)
    if code == 404:
        models = ", ".join(f"`{m}`" for m in _gemini_models())
        return (
            f"nenhum dos modelos configurados ({models}) está disponível para esta chave. "
            "Ajuste as configurações `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL`."
        )
    if code in (400, 401, 403):
        return "a chave do Gemini é inválida ou não tem permissão de acesso."
    if code == 429:
        return (
            "a cota da API do Gemini foi esgotada (inclusive no modelo reserva). "
            "Tente novamente em alguns minutos."
        )
    if code is not None and code >= 500:
        return (
            "o serviço do Gemini está instável ou sobrecarregado no momento, "
            "inclusive no modelo reserva. Tente novamente em alguns minutos."
        )
    return "erro inesperado na chamada ao Gemini (detalhes no log do servidor)."


# ==================================================================
# CAMADA 1: GEMINI API
# ==================================================================
def _generate_with_gemini(df, kpis, period_label, dataset_name, api_key) -> str:
    """Tenta o modelo principal e, se ele estiver indisponível, sobrecarregado
    ou sem cota, tenta o modelo reserva antes de desistir."""
    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(kpis, period_label, dataset_name)

    last_exc = None
    for model in _gemini_models():
        try:
            return _call_gemini_model(client, model, prompt)
        except genai_errors.APIError as exc:
            if not _can_try_other_model(exc):
                raise
            logger.warning("Modelo %s falhou (%s); tentando o próximo.", model, exc.code)
            last_exc = exc

    raise last_exc


def _gemini_models() -> list[str]:
    models = [get_gemini_model(), get_gemini_fallback_model()]
    return list(dict.fromkeys(m for m in models if m))  # sem repetidos, na ordem


def _is_transient(exc: Exception) -> bool:
    """429 = cota/rate limit; 5xx = instabilidade temporária do serviço."""
    code = getattr(exc, "code", None) or 0
    return code == 429 or code >= 500


def _can_try_other_model(exc: Exception) -> bool:
    # 404 = modelo indisponível para esta chave
    return _is_transient(exc) or getattr(exc, "code", None) == 404


def _call_gemini_model(client, model: str, prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                # O agente não usa ferramentas; desligar a chamada automática
                # de funções (AFC) também evita um aviso da biblioteca.
                config=genai_types.GenerateContentConfig(
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc) and attempt < max_retries:
                time.sleep(2 ** (attempt + 1))  # backoff exponencial: 2s, 4s
                continue
            raise

        text = (response.text or "").strip()
        if not text:
            raise AIAgentError("Resposta vazia da API Gemini")
        return text

    raise AIAgentError("Número máximo de tentativas excedido")  # pragma: no cover


def _build_prompt(kpis: dict, period_label: str, dataset_name: str) -> str:
    if dataset_name == "Vendas":
        growth_txt = (
            format_pct(kpis['revenue_growth_pct'], signed=True) if kpis.get("revenue_growth_pct") is not None
            else "não disponível (sem período anterior comparável)"
        )
        kpi_block = f"""
- Receita total: {format_brl(kpis['total_revenue'])}
- Lucro total: {format_brl(kpis['total_profit'])}
- Margem de lucro: {format_pct(kpis['margin_pct'])}
- Unidades vendidas: {format_number(kpis['total_units'])}
- Ticket médio: {format_brl(kpis['avg_ticket'])}
- Categoria líder: {kpis['top_category']} ({format_brl(kpis['top_category_revenue'])})
- Região líder: {kpis['top_region']}
- Crescimento da receita vs período anterior: {growth_txt}
""".strip()
    else:
        kpi_block = f"""
- Preço atual: {format_usd(kpis['current_price'])}
- Variação no período: {format_pct(kpis['period_change_pct'], 2, signed=True)}
- Máxima do período: {format_usd(kpis['max_price'])}
- Mínima do período: {format_usd(kpis['min_price'])}
- Volume médio negociado: {format_usd(kpis['avg_volume'], 0)}
- Volatilidade diária (desvio padrão dos retornos): {format_pct(kpis['volatility_pct'], 2)}
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
        f"{format_pct(kpis['revenue_growth_pct'], signed=True)} em relação ao período anterior"
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
        "; ".join(f"**{cat}** ({format_pct(pct, 0)})" for cat, pct in declining_categories)
        if declining_categories else "nenhuma categoria com queda relevante (>15%) identificada"
    )

    low_margin_flag = kpis["margin_pct"] < 20

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
            f"Reavaliar política de custos/precificação: margem atual de {format_pct(kpis['margin_pct'])} está "
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

No período analisado ({period_label}), a receita total somou **{format_brl(kpis['total_revenue'])}**, \
com lucro de **{format_brl(kpis['total_profit'])}** (margem de {format_pct(kpis['margin_pct'])}). \
Foram registrados **{format_number(kpis['n_orders'])} pedidos**, totalizando {format_number(kpis['total_units'])} unidades vendidas, \
com ticket médio de **{format_brl(kpis['avg_ticket'])}**. A análise de tendência (regressão linear sobre a \
série diária de receita, confiança {confidence}) aponta **{trend_word}** de aproximadamente \
**{format_pct(trend_pct, signed=True)}** ao longo do período, com variação de {growth_txt}. \
A categoria **{best_category}** lidera em receita, e a região **{kpis['top_region']}** é a de maior \
representatividade comercial.

## Diagnóstico de Pontos Críticos / Gargalos

- **Categorias em queda:** {declining_txt}, comparando a primeira e a segunda metade do período selecionado.
- **Categoria de menor receita:** **{worst_category}**, candidata a revisão de estratégia comercial ou descontinuação.
- **Margem de lucro:** {"abaixo do ideal (" + f"{format_pct(kpis['margin_pct'])}" + "), sinalizando pressão de custos ou descontos agressivos." if low_margin_flag else f"saudável, em {format_pct(kpis['margin_pct'])}, indicando controle de custos eficiente."}
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
        f"Acompanhar de perto o comportamento em torno da máxima do período ({format_usd(kpis['max_price'])}) como possível resistência técnica.",
        f"Considerar a mínima do período ({format_usd(kpis['min_price'])}) como referência de suporte para decisões de entrada.",
        "Cruzar esta análise de preço com indicadores on-chain e volume para confirmar a força da tendência antes de decisões relevantes.",
    ]
    action_md = "\n".join(f"{i+1}. {item}" for i, item in enumerate(action_items))

    return f"""## Destaques do Período

No período analisado ({period_label}), o ativo apresentou variação de **{format_pct(kpis['period_change_pct'], 2, signed=True)}**, \
encerrando a **{format_usd(kpis['current_price'])}**. A regressão linear sobre a série de preços (confiança \
{confidence}) indica tendência de **{trend_word}**, com inclinação equivalente a {format_pct(trend_pct, signed=True)} no período. \
A máxima registrada foi **{format_usd(kpis['max_price'])}** e a mínima **{format_usd(kpis['min_price'])}**, com \
volume médio negociado de **{format_usd(kpis['avg_volume'], 0)}**.

## Diagnóstico de Pontos Críticos / Gargalos

- **Volatilidade diária:** {format_pct(kpis['volatility_pct'], 2)} ({"elevada, exigindo cautela redobrada" if volatility_flag else "dentro de patamares administráveis"}).
- **Distância da máxima:** o preço atual está {format_pct(abs(drawdown_pct))} {"abaixo" if drawdown_pct < 0 else "acima"} da máxima do período, indicando {"possível correção em curso" if drawdown_pct < -10 else "proximidade de topo histórico recente"}.
- **Confiança da tendência (R²):** {format_number(r2, 2)} — {"tendência bem definida" if r2 > 0.5 else "sinal de tendência fraco, mercado possivelmente em consolidação"}.

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
