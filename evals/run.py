"""
Avaliação da IA: roda as perguntas de `evals/cases.py` no chat (e alguns
relatórios executivos) com cada modelo e gera `evals/RESULTADOS.md`.

Uso (precisa de GEMINI_API_KEY no .env ou no ambiente):

    python -m evals.run
    python -m evals.run --modelos gemini-2.5-flash gemini-flash-lite-latest
    python -m evals.run --casos receita_2024,ontem --refazer

Cada resposta é salva assim que termina em `evals/resultados/<modelo>.jsonl`:
se a cota acabar no meio, rodar de novo continua de onde parou (use
`--refazer` para começar do zero). Cada modelo é avaliado sozinho, sem o
modelo reserva do app, para que o resultado seja só dele.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from evals.cases import CASES, Case
from evals.scoring import CaseResult, score
from insight_engine.ai.agent import SOURCE_LLM, generate_executive_summary
from insight_engine.ai.chat import ChatTurn, SalesDataTools, ask
from insight_engine.ai.context import build_sales_facts
from insight_engine.analytics.kpis import compute_sales_kpis
from insight_engine.analytics.periods import SalesFilters, apply_segments, filter_sales
from insight_engine.data.sales import load_sales_data
from insight_engine.formatting import format_number, format_pct

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "resultados"
SUMMARY_FILE = ROOT / "RESULTADOS.md"
# apelidos que o Google mantém apontando para a versão atual de cada linha
DEFAULT_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
# motivo de fallback que indica relatório fora do formato (o resto é erro da API)
FORMAT_ERROR = "fora do formato"
# períodos dos relatórios avaliados (início, fim, comparação)
REPORT_PERIODS: list[tuple[str, str, Literal["anterior", "ano_anterior"]]] = [
    ("2025-10-01", "2025-12-31", "anterior"),
    ("2025-10-01", "2025-12-31", "ano_anterior"),
    ("2024-01-01", "2024-12-31", "anterior"),
]


@dataclass
class ReportResult:
    period: str
    valid: bool
    checked: int = 0
    verified: int = 0
    latency_s: float = 0.0
    error: str | None = None


# ------------------------------------------------------------------
# Execução
# ------------------------------------------------------------------
def run_chat(
    provider: Any,
    df: pd.DataFrame,
    cases: Iterable[Case],
    out_file: Path,
    pause_s: float = 0.0,
    redo: bool = False,
) -> list[CaseResult]:
    """Roda as perguntas e grava cada resultado.

    Perguntas já respondidas são puladas; as que deram erro da API (cota, instabilidade)
    são tentadas de novo. Com `redo`, as perguntas pedidas são refeitas de qualquer jeito.
    """
    cases = list(cases)
    ids = {c.id for c in cases}
    saved = load_results(out_file)
    kept = [r for r in saved if r.id not in ids or (not redo and r.error is None)]
    if len(kept) != len(saved):
        _write_lines(out_file, [r.to_dict() for r in kept])
    done = {r.id: r for r in kept}
    tools = SalesDataTools(df)
    results = []
    for case in cases:
        if case.id in done:
            results.append(done[case.id])
            continue
        expected = case.expected(df)
        turn = ChatTurn()
        tokens_before = dict(getattr(provider, "tokens", {}))
        started = time.perf_counter()
        error = None
        try:
            for _ in ask(provider, tools, [], case.question, turn):
                pass
        except Exception as exc:  # noqa: BLE001 - falha da API vira erro registrado, não fim da avaliação
            error = provider.describe_error(exc)
        result = score(case, expected, turn)
        result.latency_s = round(time.perf_counter() - started, 2)
        result.error = error
        tokens = getattr(provider, "tokens", {})
        result.tokens_in = tokens.get("entrada", 0) - tokens_before.get("entrada", 0)
        result.tokens_out = tokens.get("saida", 0) - tokens_before.get("saida", 0)
        _append(out_file, result.to_dict())
        results.append(result)
        status = "ok    " if result.passed else "FALHOU"
        print(f"  {status} {case.id} ({result.latency_s:.1f}s){' — ' + error if error else ''}")
        time.sleep(pause_s)
    return results


def run_reports(provider: Any, df: pd.DataFrame, out_file: Path, pause_s: float = 0.0) -> list[ReportResult]:
    """Gera relatórios executivos e mede validade do formato e números conferidos."""
    results = []
    for start, end, comparison in REPORT_PERIODS:
        selection = SalesFilters(date.fromisoformat(start), date.fromisoformat(end), [], [], comparison)
        current, previous = filter_sales(df, selection)
        facts = build_sales_facts(
            history=apply_segments(df, selection),
            period=current,
            previous=previous,
            kpis=compute_sales_kpis(current, previous_df=previous),
            period_label=selection.period_label,
            comparison_label=selection.comparison_label,
        )
        started = time.perf_counter()
        report = generate_executive_summary(facts, "Vendas", selection.period_label, provider=provider)
        check = report.verification
        result = ReportResult(
            period=f"{selection.period_label} vs {selection.comparison_label}",
            valid=report.source == SOURCE_LLM,
            checked=check.checked if check else 0,
            verified=check.verified if check else 0,
            latency_s=round(time.perf_counter() - started, 2),
            error=None if report.source == SOURCE_LLM else report.fallback_reason,
        )
        results.append(result)
        status = "ok    " if result.valid else "FALHOU"
        print(f"  {status} relatório {result.period} ({result.verified}/{result.checked} números)")
        time.sleep(pause_s)
    _write_lines(out_file, [vars(r) for r in results])
    return results


# ------------------------------------------------------------------
# Resumo em Markdown
# ------------------------------------------------------------------
def write_summary(
    chat: dict[str, list[CaseResult]], reports: dict[str, list[ReportResult]], path: Path | None = None
) -> str:
    path = path or SUMMARY_FILE
    models = list(chat)
    lines = [
        "# Resultados da avaliação da IA",
        "",
        f"Gerado em {date.today():%d/%m/%Y} com `python -m evals.run`, na base de exemplo, "
        f"com {len(CASES)} perguntas ([método](README.md)).",
        "",
        "## Chat com os dados",
        "",
        "| Modelo | Acerto | Números corretos | Sem alucinação | Consulta certa | Latência (mediana) "
        "| Tokens por pergunta |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in models:
        answered = [r for r in chat[model] if r.error is None]
        with_numbers = [r for r in answered if r.expects_numbers]
        lines.append(
            f"| `{model}` | **{_rate(answered, lambda r: r.passed)}** "
            f"| {_rate(with_numbers, lambda r: not r.missing_numbers)} "
            f"| {_rate(answered, lambda r: not r.hallucinated)} "
            f"| {_rate(answered, lambda r: r.right_tool)} "
            f"| {_median(r.latency_s for r in answered)} s "
            f"| {_mean(r.tokens_in + r.tokens_out for r in answered)} |"
        )
    errors = {m: sum(r.error is not None for r in chat[m]) for m in models}
    if any(errors.values()):
        lines += [
            "",
            "Perguntas com erro da API (fora das taxas): " + ", ".join(f"`{m}` {n}" for m, n in errors.items()),
        ]

    lines += ["", "### Acerto por tipo de pergunta", "", "| Tipo | " + " | ".join(f"`{m}`" for m in models) + " |"]
    lines.append("|---" * (len(models) + 1) + "|")
    for kind in dict.fromkeys(c.kind for c in CASES):
        cells = [_rate([r for r in chat[m] if r.kind == kind and r.error is None], lambda r: r.passed) for m in models]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")

    if reports:
        lines += [
            "",
            "## Relatório executivo",
            "",
            "| Modelo | Relatórios no formato certo | Números conferidos | Latência média | Erros da API |",
            "|---|---|---|---|---|",
        ]
        for model, all_items in reports.items():
            # erro da API (cota, modelo indisponível) não mede o modelo: fica de fora das taxas
            items = [r for r in all_items if r.valid or FORMAT_ERROR in (r.error or "")]
            checked = sum(r.checked for r in items)
            verified = sum(r.verified for r in items)
            valid = sum(r.valid for r in items)
            lines.append(
                f"| `{model}` | {valid} de {len(items)} | "
                f"{verified} de {checked} ({format_pct(verified / checked * 100 if checked else 0)}) "
                f"| {_mean(r.latency_s for r in items)} s | {len(all_items) - len(items)} |"
            )

    failures = [(m, r) for m in models for r in chat[m] if not r.passed and r.error is None]
    lines += ["", "## Respostas que não passaram", ""]
    if not failures:
        lines.append("Nenhuma.")
    for model, r in failures:
        problems = (
            [f"faltou {x}" for x in r.missing_numbers + r.missing_texts]
            + [f"número sem fonte: {x}" for x in r.hallucinated]
            + [f"texto proibido: {x}" for x in r.forbidden_found]
            + ([] if r.right_tool else [f"consulta errada ({', '.join(r.tools_called) or 'nenhuma'})"])
        )
        lines.append(f"- `{model}` · **{r.question}** — {'; '.join(problems)}")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _rate(items: list[CaseResult], ok: Callable[[CaseResult], bool]) -> str:
    if not items:
        return "—"
    hits = sum(ok(r) for r in items)
    return f"{format_pct(hits / len(items) * 100, 0)} ({hits}/{len(items)})"


def _median(values: Iterable[float]) -> str:
    values = list(values)
    return format_number(statistics.median(values), 1) if values else "—"


def _mean(values: Iterable[float]) -> str:
    values = list(values)
    return format_number(statistics.mean(values), 0 if values and max(values) > 100 else 1) if values else "—"


# ------------------------------------------------------------------
# Arquivos
# ------------------------------------------------------------------
def results_file(model: str) -> Path:
    return RESULTS_DIR / f"{model}.jsonl"


def load_results(path: Path) -> list[CaseResult]:
    if not path.exists():
        return []
    return [CaseResult.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_lines(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")


# ------------------------------------------------------------------
def main(argv: list[str] | None = None, provider_factory: Callable[[str], Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avaliação do chat e do relatório com IA.")
    parser.add_argument("--modelos", nargs="+", default=DEFAULT_MODELS, help="modelos do Gemini a comparar")
    parser.add_argument("--casos", help="ids das perguntas, separados por vírgula (padrão: todas)")
    parser.add_argument("--pausa", type=float, default=4.0, help="segundos entre chamadas (limite da cota gratuita)")
    parser.add_argument("--sem-relatorio", action="store_true", help="avalia só o chat")
    parser.add_argument("--refazer", action="store_true", help="ignora resultados salvos e refaz tudo")
    args = parser.parse_args(argv)

    if provider_factory is None:
        provider_factory = _gemini_factory()
        if provider_factory is None:
            print("Defina GEMINI_API_KEY no .env ou no ambiente para rodar a avaliação.", file=sys.stderr)
            return 1

    wanted = set(args.casos.split(",")) if args.casos else None
    cases = [c for c in CASES if wanted is None or c.id in wanted]
    df = load_sales_data()

    chat, reports = {}, {}
    for model in args.modelos:
        print(f"\n== {model}: chat ({len(cases)} perguntas)")
        provider = provider_factory(model)
        run_chat(provider, df, cases, results_file(model), args.pausa, args.refazer)
        chat[model] = load_results(results_file(model))  # resumo com todas as perguntas já avaliadas
        if not args.sem_relatorio:
            print(f"== {model}: relatório executivo")
            reports[model] = run_reports(provider, df, RESULTS_DIR / f"{model}_relatorio.jsonl", args.pausa)

    write_summary(chat, reports)
    print(f"\nResumo salvo em {SUMMARY_FILE}")
    return 0


def _gemini_factory() -> Callable[[str], Any] | None:
    from insight_engine.ai.providers.gemini import GENAI_AVAILABLE, GeminiProvider
    from insight_engine.config import get_gemini_api_key

    key = get_gemini_api_key()
    if not key or not GENAI_AVAILABLE:
        return None
    return lambda model: GeminiProvider(key, models=[model])


if __name__ == "__main__":
    sys.exit(main())
