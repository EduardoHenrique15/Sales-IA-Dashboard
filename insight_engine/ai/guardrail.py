"""
Checagem de números (guardrail contra alucinação).

Todo número citado pelo LLM é extraído do texto e procurado nos fatos que
ele recebeu. Um número é "verificado" quando bate com algum fato, aceitando
arredondamento (ex.: "36%" para 35,6%; "R$ 2,84 milhões" para R$ 2.838.404,99).
Números que não aparecem nos fatos — inventados ou calculados pelo modelo —
são listados para o usuário conferir.

Datas, anos e contagens pequenas (ex.: "3 ações", "30 dias") não são
checados: são vocabulário do texto, não métricas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# número no padrão brasileiro (1.234,56 ou 12,5 ou 42), com sinal, R$/US$ e escala opcionais
_NUMBER = re.compile(
    r"(?P<currency>-?\s?(?:R\$|US\$)\s?)?"
    r"(?P<number>[-+]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|[-+]?\d+(?:,\d+)?)"
    r"(?P<scale>\s?(?:mil\b|milh(?:ão|ões)|bilh(?:ão|ões)|mi\b|bi\b))?"
    r"(?P<pct>\s?%)?",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_SCALE = {"mil": 1e3, "mi": 1e6, "milhão": 1e6, "milhões": 1e6, "bi": 1e9, "bilhão": 1e9, "bilhões": 1e9}

# Inteiros pequenos sem unidade (contagens, "top 5") e anos não são métricas
_SMALL_INT_LIMIT = 31
_YEARS = range(1990, 2101)


@dataclass(frozen=True)
class CitedNumber:
    text: str
    value: float
    is_percent: bool
    decimals: int


@dataclass(frozen=True)
class Verification:
    checked: int
    unverified: list[str] = field(default_factory=list)

    @property
    def verified(self) -> int:
        return self.checked - len(self.unverified)

    @property
    def ok(self) -> bool:
        return not self.unverified


def extract_numbers(text: str) -> list[CitedNumber]:
    """Números relevantes citados em `text` (moedas, porcentagens e demais métricas)."""
    text = _DATE.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(text):
        raw = match.group("number")
        value = float(raw.replace(".", "").replace(",", "."))
        decimals = len(raw.split(",")[1]) if "," in raw else 0
        currency = match.group("currency") or ""
        if currency.strip().startswith("-"):
            value = -abs(value)
        scale_text = (match.group("scale") or "").strip().lower()
        if scale_text:
            value *= _SCALE[scale_text]
        is_percent = bool(match.group("pct"))

        plain_integer = not (currency or scale_text or is_percent) and decimals == 0
        if plain_integer and "." not in raw and (abs(value) <= _SMALL_INT_LIMIT or int(abs(value)) in _YEARS):
            continue
        found.append(CitedNumber(match.group(0).strip(), value, is_percent, decimals if not scale_text else 2))
    return found


def verify(answer: str, facts: str) -> Verification:
    """Confere os números de `answer` contra os números presentes em `facts`."""
    known = extract_numbers(facts)
    known_percents = [n.value for n in known if n.is_percent]
    known_values = [n.value for n in known if not n.is_percent]

    cited = extract_numbers(answer)
    unverified = [
        n.text
        for n in cited
        if not any(_matches(n, ref) for ref in (known_percents if n.is_percent else known_values))
        and not (n.is_percent and any(_matches(n, ref) for ref in known_values))
    ]
    return Verification(checked=len(cited), unverified=list(dict.fromkeys(unverified)))


def _matches(cited: CitedNumber, reference: float) -> bool:
    """Aceita o arredondamento usado no texto (ex.: 36 para 35,6) ou 0,5% de diferença."""
    rounding = 0.5 * 10 ** (-cited.decimals)
    for ref in (reference, abs(reference)):
        if abs(cited.value - ref) <= rounding + 1e-9 or abs(cited.value - ref) <= abs(ref) * 0.005:
            return True
        # valor abreviado: "R$ 2,84 milhões" para 2.838.404,99
        if abs(cited.value) >= 1_000 and cited.decimals == 2 and abs(cited.value - ref) <= abs(ref) * 0.006:
            return True
    return False
