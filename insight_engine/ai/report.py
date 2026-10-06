"""
Estrutura do Relatório Executivo (saída estruturada).

O LLM devolve um JSON com estes campos (validado pelo Pydantic), e o
motor estatístico local monta o mesmo objeto. Assim os dois motores
produzem exatamente a mesma estrutura, exibida pelo mesmo código.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["alta", "média", "baixa"]


class Risk(BaseModel):
    title: str = Field(description="Nome curto do risco ou gargalo.")
    evidence: str = Field(description="Evidência numérica, usando apenas números do bloco de dados.")
    severity: Level = Field(description="Gravidade do risco.")


class Action(BaseModel):
    action: str = Field(description="Ação prática e específica.")
    rationale: str = Field(description="Por que esta ação, ligada a um dado do relatório.")
    priority: Level = Field(description="Prioridade da ação.")


class ExecutiveReport(BaseModel):
    headline: str = Field(description="Uma frase que resume o período para a diretoria.")
    highlights: list[str] = Field(description="3 a 4 destaques do período, com números concretos.")
    risks: list[Risk] = Field(description="2 a 3 pontos críticos ou gargalos.")
    actions: list[Action] = Field(description="3 a 5 ações priorizadas.")


_LEVEL_ICON = {"alta": "🔴", "média": "🟡", "baixa": "🟢"}
_LEVEL_ORDER = {"alta": 0, "média": 1, "baixa": 2}


def render_markdown(report: ExecutiveReport, footer: str | None = None) -> str:
    """Converte o relatório estruturado em Markdown (tela e download)."""
    lines = [f"**{report.headline}**", "", "## Destaques do Período", ""]
    lines += [f"- {item}" for item in report.highlights]

    lines += ["", "## Diagnóstico de Pontos Críticos / Gargalos", ""]
    for risk in sorted(report.risks, key=lambda r: _LEVEL_ORDER[r.severity]):
        lines.append(f"- {_LEVEL_ICON[risk.severity]} **{risk.title}** (gravidade {risk.severity}): {risk.evidence}")

    lines += ["", "## Plano de Ação Estratégico Sugerido", ""]
    actions = sorted(report.actions, key=lambda a: _LEVEL_ORDER[a.priority])
    for i, action in enumerate(actions, start=1):
        lines.append(f"{i}. **{action.action}** (prioridade {action.priority}) — {action.rationale}")

    if footer:
        lines += ["", "---", f"*{footer}*"]
    return "\n".join(lines)
