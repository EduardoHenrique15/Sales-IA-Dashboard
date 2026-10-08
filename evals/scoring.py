"""
Pontuação de uma resposta do chat contra a resposta certa.

Uma pergunta passa quando:
  1. o chat usou uma das consultas certas;
  2. cada número esperado aparece na resposta (aceitando o arredondamento do texto,
     com a mesma regra do verificador do app);
  3. cada termo esperado aparece (sem diferenciar maiúsculas nem acentos);
  4. nenhum número citado deixa de aparecer nos resultados das consultas (sem alucinação);
  5. nenhum texto proibido aparece (ex.: a chave de API).
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field

from evals.cases import Case, Expected
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
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    @classmethod
    def from_dict(cls, data: dict) -> CaseResult:
        return cls(**{k: v for k, v in data.items() if k != "passed"})


def score(case: Case, expected: Expected, turn: ChatTurn) -> CaseResult:
    answer = turn.text.strip()
    called = [call.name for call in turn.tool_calls]
    cited = extract_numbers(answer)
    normalized = _normalize(answer)

    missing_numbers = [
        f"{n.label} ({format_number(n.value, 1)}{'%' if n.percent else ''})"
        for n in expected.numbers
        if not any(_matches(c, n.value) for c in cited if c.is_percent == n.percent)
    ]
    missing_texts = [
        " ou ".join(group) for group in expected.texts if not any(_normalize(t) in normalized for t in group)
    ]
    verification = turn.verification
    return CaseResult(
        id=case.id,
        kind=case.kind,
        question=case.question,
        answer=answer,
        tools_called=called,
        right_tool=not case.tools or any(name in case.tools for name in called),
        missing_numbers=missing_numbers,
        missing_texts=missing_texts,
        hallucinated=list(verification.unverified) if verification else [],
        forbidden_found=[text for text in case.forbidden if text.lower() in answer.lower()],
        expects_numbers=bool(expected.numbers),
    )


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
