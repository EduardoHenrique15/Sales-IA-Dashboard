"""
Agente responsável por transformar números em narrativa.

Estratégia de duas camadas:

  1. CAMADA PRIMÁRIA (Google Gemini)
     Com uma chave configurada, o agente envia os KPIs já calculados
     e pede um Relatório Executivo estruturado em Markdown.

  2. CAMADA DE FALLBACK (estatística, 100% local, sem custo)
     Sem chave, ou se a API falhar (cota, instabilidade, modelo
     indisponível), o agente gera o mesmo relatório com regressão
     linear e regras de negócio. A estrutura do relatório é a mesma;
     só a "voz" muda.

A ausência ou instabilidade de uma API de terceiros nunca quebra a
experiência do usuário.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from insight_engine.ai import fallback, gemini
from insight_engine.ai.prompts import build_prompt
from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs
from insight_engine.config import get_gemini_api_key

logger = logging.getLogger(__name__)

SOURCE_GEMINI = "gemini"
SOURCE_FALLBACK = "fallback_estatistico"


@dataclass(frozen=True)
class ReportResult:
    markdown: str
    # SOURCE_GEMINI ou SOURCE_FALLBACK (usado no selo da interface)
    source: str
    # por que o Gemini não foi usado quando havia uma chave configurada
    fallback_reason: str | None = None


def generate_executive_summary(
    df: pd.DataFrame,
    kpis: SalesKPIs | CryptoKPIs,
    period_label: str,
    dataset_name: str = "Vendas",
    api_key: str | None = None,
) -> ReportResult:
    """Gera o Relatório Executivo em Markdown para os dados filtrados."""
    api_key = api_key or get_gemini_api_key()

    if df.empty:
        return ReportResult(fallback.no_data_report(period_label, dataset_name), SOURCE_FALLBACK)

    fallback_reason = None
    if api_key and gemini.GENAI_AVAILABLE:
        try:
            prompt = build_prompt(kpis, period_label, dataset_name)
            return ReportResult(gemini.generate(prompt, api_key), SOURCE_GEMINI)
        except Exception as exc:  # noqa: BLE001 - qualquer falha da API cai no fallback
            logger.warning("Falha na API Gemini, usando fallback. Detalhe: %s", exc)
            fallback_reason = gemini.describe_error(exc)
    elif api_key:
        fallback_reason = "o pacote `google-genai` não está instalado."

    if isinstance(kpis, SalesKPIs):
        report = fallback.sales_report(df, kpis, period_label)
    else:
        report = fallback.crypto_report(df, kpis, period_label)
    return ReportResult(report, SOURCE_FALLBACK, fallback_reason)
