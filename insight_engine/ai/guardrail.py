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

# número no padrão brasileiro (1.234,56 ou 12,5 ou 42), com sinal, R$/US$ e escala opcionais.
# O sinal só conta colado ao número ou à moeda: em "- R$ 10" o hífen é marcador de lista.
_NUMBER = re.compile(
    r"(?P<currency>(?:[-−](?=R\$|US\$))?(?:R\$|US\$)\s?)?"
    r"(?P<number>(?:(?<!\w)[-+−])?(?:\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?))"
    r"(?P<scale>\s?(?:mil\b|milh(?:ão|ões)|bilh(?:ão|ões)|mi\b|bi\b))?"
    r"(?P<pct>\s?%)?",
    re.IGNORECASE,
)
_DATE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
_SCALE = {"mil": 1e3, "mi": 1e6, "milhão": 1e6, "milhões": 1e6, "bi": 1e9, "bilhão": 1e9, "bilhões": 1e9}

MAX_SCALED_ROUNDING = 0.05

# Inteiros pequenos sem unidade (contagens, "top 5") e anos não são métricas
_SMALL_INT_LIMIT = 31
_YEARS = range(1990, 2101)


@dataclass(frozen=True)
class CitedNumber:
    text: str
    value: float
    is_percent: bool
    # maior diferença que ainda é arredondamento: 0,5 na última casa escrita ("2,8 milhões" → 50 mil)
    tolerance: float
    # sinal explícito ("+5%", "-R$ 10"): aí "alta" e "queda" não são intercambiáveis
    signed: bool = False


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
        raw = match.group("number").replace("−", "-")
        value = float(raw.replace(".", "").replace(",", "."))
        decimals = len(raw.split(",")[1]) if "," in raw else 0
        currency = match.group("currency") or ""
        signed = raw[0] in "+-"
        if currency[:1] in ("-", "−"):
            value, signed = -abs(value), True
        scale_text = (match.group("scale") or "").strip().lower()
        scale = _SCALE[scale_text] if scale_text else 1.0
        is_percent = bool(match.group("pct"))

        plain_integer = not (currency or scale_text or is_percent) and decimals == 0
        if plain_integer and "." not in raw and (abs(value) <= _SMALL_INT_LIMIT or int(abs(value)) in _YEARS):
            continue
        tolerance = 0.5 * 10 ** (-decimals) * scale
        if scale_text:
            # "R$ 3 milhões" arredondaria qualquer valor de 2,5 a 3,5 milhões: abreviações grosseiras
            # demais deixariam passar números inventados, então o arredondamento vai até 5%
            tolerance = min(tolerance, abs(value * scale) * MAX_SCALED_ROUNDING)
        found.append(CitedNumber(match.group(0).strip(), value * scale, is_percent, tolerance, signed))
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
    """Aceita só o arredondamento usado no texto: "36%" para 35,6; "R$ 2,84 milhões" para 2.838.404,99.

    Sem sinal explícito, o número pode ser o valor absoluto do fato ("queda de 73%" para -73%).
    """
    candidates = (reference,) if cited.signed else (reference, abs(reference))
    return any(abs(cited.value - ref) <= cited.tolerance + 1e-9 for ref in candidates)
