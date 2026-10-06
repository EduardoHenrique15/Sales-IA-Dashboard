"""
Agente responsável por transformar números em narrativa.

Estratégia de duas camadas:

  1. LLM (Gemini, via `providers`): recebe os fatos calculados e devolve um
     Relatório Executivo estruturado; os números citados são conferidos
     contra os fatos (`guardrail`).
  2. Motor estatístico local (`fallback`): sem chave, sem cota ou com a API
     fora do ar, monta o mesmo relatório estruturado a partir dos mesmos
     fatos. A ausência ou instabilidade de uma API de terceiros nunca quebra
     a experiência do usuário.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from insight_engine.ai import fallback
from insight_engine.ai.context import SalesFacts
from insight_engine.ai.guardrail import Verification, verify
from insight_engine.ai.prompts import REPORT_SYSTEM, build_report_prompt
from insight_engine.ai.providers.base import LLMProvider
from insight_engine.ai.report import ExecutiveReport, render_markdown
from insight_engine.config import get_gemini_model

logger = logging.getLogger(__name__)

SOURCE_LLM = "llm"
SOURCE_FALLBACK = "fallback_estatistico"


@dataclass(frozen=True)
class ReportResult:
    report: ExecutiveReport
    markdown: str
    # SOURCE_LLM ou SOURCE_FALLBACK (usado no selo da interface)
    source: str
    provider_name: str | None = None
    # por que o LLM não foi usado quando havia uma chave configurada
    fallback_reason: str | None = None
    # checagem dos números citados pelo LLM (None no relatório local)
    verification: Verification | None = None
    from_cache: bool = False


def generate_executive_summary(
    facts: SalesFacts | None,
    dataset_name: str,
    period_label: str,
    provider: LLMProvider | None = None,
    provider_error: str | None = None,
    cache: MutableMapping[str, ExecutiveReport] | None = None,
    allow_call: Callable[[], bool] | None = None,
    limit_message: str = "",
) -> ReportResult:
    """Gera o Relatório Executivo com o LLM ou, se não for possível, com o motor local.

    `facts` é None quando não há dados no recorte. `cache` guarda relatórios do
    LLM por conteúdo do prompt (mesmos dados = mesmo relatório, sem gastar cota).
    `allow_call` aplica limites de uso; quando devolve False, usa-se o motor local.
    """
    if facts is None:
        return _local(fallback.no_data_report(period_label, dataset_name), None)

    facts_text = "\n".join(f"- {line}" for line in facts.to_lines())
    reason = provider_error

    if provider is not None:
        prompt = build_report_prompt(facts_text, dataset_name)
        key = hashlib.sha256(f"{provider.name}|{get_gemini_model()}|{prompt}".encode()).hexdigest()

        if cache is not None and key in cache:
            report = cache[key]
            return _from_llm(report, provider.name, facts_text, from_cache=True)

        if allow_call is not None and not allow_call():
            reason = limit_message
        else:
            try:
                report = provider.generate_report(prompt, REPORT_SYSTEM)
                if cache is not None:
                    cache[key] = report
                return _from_llm(report, provider.name, facts_text)
            except Exception as exc:  # noqa: BLE001 - qualquer falha da API cai no motor local
                logger.warning("Falha no LLM (%s), usando o motor local. Detalhe: %s", provider.name, exc)
                reason = provider.describe_error(exc)

    return _local(fallback.sales_report(facts), reason)


def _from_llm(report: ExecutiveReport, provider_name: str, facts_text: str, from_cache: bool = False) -> ReportResult:
    markdown = render_markdown(report)
    return ReportResult(
        report=report,
        markdown=markdown,
        source=SOURCE_LLM,
        provider_name=provider_name,
        verification=verify(markdown, facts_text),
        from_cache=from_cache,
    )


def _local(report: ExecutiveReport, reason: str | None) -> ReportResult:
    return ReportResult(
        report=report,
        markdown=render_markdown(report, footer=fallback.FOOTER),
        source=SOURCE_FALLBACK,
        fallback_reason=reason,
    )
