from insight_engine.ai.report import Action, ExecutiveReport, Risk
from insight_engine.ai.report_pdf import _safe, render_pdf


def test_gera_um_pdf_valido():
    report = ExecutiveReport(
        headline="Receita de R$ 10,00 — “ótimo” período 🔴",
        highlights=["Destaque com ± 6,8% e acentuação: ação, previsão."],
        risks=[Risk(title="Risco", evidence="Evidência", severity="alta")],
        actions=[Action(action="Ação", rationale="Motivo", priority="baixa")],
    )
    pdf = render_pdf(report, "Relatório Executivo - Vendas", "01/01/2025 a 31/01/2025", "Motor local", "3 de 3")
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_relatorio_longo_quebra_em_varias_paginas():
    report = ExecutiveReport(
        headline="Resumo",
        highlights=["Um destaque bem longo. " * 40] * 10,
        risks=[Risk(title="Risco", evidence="Evidência " * 50, severity="média")] * 5,
        actions=[Action(action="Ação", rationale="Motivo " * 50, priority="alta")] * 5,
    )
    assert render_pdf(report, "Título", "p", "fonte").count(b"/Type /Page\n") >= 2


def test_simbolos_fora_do_latin1_sao_convertidos():
    assert _safe("A — B “c” … 🔴") == 'A - B "c" ... '
    assert _safe("Ação ± 6,8%") == "Ação ± 6,8%"
    assert _safe("R$ 10,00") == "R$ 10,00"  # o valor não se separa do símbolo


def test_sinal_de_menos_tipografico_nao_some():
    assert _safe("Variação de −R$ 70 mil") == "Variação de -R$ 70 mil"
    assert _safe("R$ 10 ≈ 9 €") == "R$ 10 ~ 9 EUR"
