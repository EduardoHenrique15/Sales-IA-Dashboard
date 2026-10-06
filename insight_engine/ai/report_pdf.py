"""
Exportação do Relatório Executivo em PDF (fpdf2).

O PDF é gerado a partir do relatório estruturado (não do Markdown), com
cabeçalho de contexto (período, origem do texto, checagem dos números).
As fontes padrão do PDF cobrem o português (latin-1); símbolos fora dessa
faixa que um LLM pode escrever (travessão, aspas curvas, emojis) são
trocados por equivalentes simples.
"""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

from insight_engine.ai.report import ExecutiveReport

_LEVEL_COLOR = {"alta": (200, 40, 40), "média": (190, 130, 0), "baixa": (30, 140, 60)}
_LEVEL_ORDER = {"alta": 0, "média": 1, "baixa": 2}
_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
    "•": "-",
    "→": "->",
    # espaço que não quebra linha entre o símbolo da moeda e o valor
    "R$ ": "R$\u00a0",
}
_TEXT, _MUTED, _ACCENT = (25, 25, 35), (110, 110, 120), (57, 135, 229)


def render_pdf(
    report: ExecutiveReport,
    title: str,
    period_label: str,
    source_label: str,
    verification_label: str | None = None,
) -> bytes:
    pdf = _ReportPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(0, 9, _safe(title), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(*_MUTED)
    meta = [f"Período: {period_label}", f"Gerado em {datetime.now():%d/%m/%Y %H:%M}", source_label]
    if verification_label:
        meta.append(verification_label)
    pdf.multi_cell(0, 5, _safe("  |  ".join(meta)), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(0, 6.5, _safe(report.headline), align="L", new_x="LMARGIN", new_y="NEXT")

    _section(pdf, "Destaques do Período")
    for item in report.highlights:
        _bullet(pdf, item)

    _section(pdf, "Diagnóstico de Pontos Críticos / Gargalos")
    for risk in sorted(report.risks, key=lambda r: _LEVEL_ORDER[r.severity]):
        _labeled(pdf, f"[{risk.severity.upper()}]", risk.severity, f"{risk.title}: ", risk.evidence)

    _section(pdf, "Plano de Ação Estratégico Sugerido")
    for i, action in enumerate(sorted(report.actions, key=lambda a: _LEVEL_ORDER[a.priority]), start=1):
        _labeled(pdf, f"{i}. [{action.priority.upper()}]", action.priority, f"{action.action}: ", action.rationale)

    return bytes(pdf.output())


class _ReportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4")
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=18)
        self.alias_nb_pages()

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", size=8)
        self.set_text_color(*_MUTED)
        self.cell(0, 6, _safe(f"Insight Engine - Relatório Executivo  |  página {self.page_no()}/{{nb}}"), align="C")


def _section(pdf: FPDF, title: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*_ACCENT)
    pdf.multi_cell(0, 8, _safe(title), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*_ACCENT)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)


def _bullet(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=10.5)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(0, 5.8, _safe(f"- {text}"), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _labeled(pdf: FPDF, label: str, level: str, bold: str, text: str) -> None:
    """Rótulo colorido (gravidade/prioridade, sempre com o texto do nível), título em negrito e explicação."""
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*_LEVEL_COLOR[level])
    pdf.write(5.8, _safe(label + " "))
    pdf.set_text_color(*_TEXT)
    pdf.write(5.8, _safe(bold))
    pdf.set_font("Helvetica", size=10.5)
    pdf.write(5.8, _safe(text))
    pdf.ln(7.5)


def _safe(text: str) -> str:
    for original, replacement in _REPLACEMENTS.items():
        text = text.replace(original, replacement)
    # o que ainda estiver fora do latin-1 (ex.: emojis) é removido
    return text.encode("latin-1", errors="ignore").decode("latin-1")
