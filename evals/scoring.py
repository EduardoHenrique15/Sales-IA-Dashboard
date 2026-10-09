"""
Pontuação de uma resposta do chat contra a resposta certa.

Uma pergunta passa quando:
  1. o chat usou uma das consultas certas;
  2. cada número esperado aparece na resposta (aceitando o arredondamento do texto,
     com a mesma regra do verificador do app);
  3. cada termo esperado aparece (sem diferenciar maiúsculas nem acentos);
  4. nenhum número citado deixa de aparecer nos resultados das consultas (sem alucinação);
  5. nenhum texto proibido aparece (ex.: a chave de API);
  6. nas perguntas de robustez (sem resposta nos dados), nenhum número é citado.

Um número com `near` precisa estar no mesmo trecho (linha ou frase) que esse termo:
assim, trocar a receita de Moda pela de Beleza não passa.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

from evals.cases import ROBUSTNESS, Case, Expected
from insight_engine.ai.chat import ChatTurn
from insight_engine.ai.guardrail import _matches, extract_numbers
from insight_engine.formatting import format_number


@dataclass
class CaseResult:
    id: str
    kind: str
    question: str
    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    right_tool: bool = False
    missing_numbers: list[str] = field(default_factory=list)
    missing_texts: list[str] = field(default_factory=list)
    hallucinated: list[str] = field(default_factory=list)
    forbidden_found: list[str] = field(default_factory=list)
    # números citados numa pergunta sem resposta nos dados (robustez)
    unexpected_numbers: list[str] = field(default_factory=list)
    expects_numbers: bool = False
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and self.right_tool
            and not self.missing_numbers
            and not self.missing_texts
            and not self.hallucinated
            and not self.forbidden_found
            and not self.unexpected_numbers
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    @classmethod
    def from_dict(cls, data: dict) -> CaseResult:
        return cls(**{k: v for k, v in data.items() if k != "passed"})


def score(case: Case, expected: Expected, turn: ChatTurn) -> CaseResult:
    called = [call.name for call in turn.tool_calls]
    verification = turn.verification
    result = CaseResult(
        id=case.id,
        kind=case.kind,
        question=case.question,
        answer=turn.text.strip(),
        tools_called=called,
        right_tool=not case.tools or any(name in case.tools for name in called),
        hallucinated=list(verification.unverified) if verification else [],
    )
    return rescore(result, case, expected)


def rescore(result: CaseResult, case: Case, expected: Expected) -> CaseResult:
    """Confere a resposta guardada contra a resposta certa atual.

    Assim, uma resposta certa que mudou (ex.: o método de segmentação) não deixa um
    resultado antigo aprovado por engano. A checagem de alucinação depende dos
    resultados das consultas, que não são guardados, e por isso é mantida.
    """
    answer = result.answer
    normalized = _normalize(answer)
    result.missing_numbers = [
        f"{n.label} ({format_number(n.value, 1)}{'%' if n.percent else ''})"
        for n in expected.numbers
        if not any(
            _matches(c, n.value)
            for part in _passages(answer, n.near)
            for c in extract_numbers(part)
            if c.is_percent == n.percent
        )
    ]
    result.missing_texts = [
        " ou ".join(group) for group in expected.texts if not any(_normalize(t) in normalized for t in group)
    ]
    result.forbidden_found = [text for text in case.forbidden if text.lower() in answer.lower()]
    result.unexpected_numbers = [n.text for n in extract_numbers(answer)] if case.kind == ROBUSTNESS else []
    result.expects_numbers = bool(expected.numbers)
    return result


def _passages(answer: str, near: str | None) -> list[str]:
    """A resposta inteira ou, com `near`, só os trechos (linhas ou frases) que citam o termo."""
    if near is None:
        return [answer]
    # frases terminam em ". " (um ponto colado a dígitos é separador de milhar)
    parts = re.split(r"\n|;|\.\s", answer)
    return [part for part in parts if _normalize(near) in _normalize(part)]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
