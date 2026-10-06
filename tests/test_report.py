from insight_engine.ai.report import Action, ExecutiveReport, Risk, render_markdown


def sample_report() -> ExecutiveReport:
    return ExecutiveReport(
        headline="Receita de R$ 10,00 no período.",
        highlights=["Destaque 1", "Destaque 2"],
        risks=[
            Risk(title="Risco baixo", evidence="ev1", severity="baixa"),
            Risk(title="Risco alto", evidence="ev2", severity="alta"),
        ],
        actions=[
            Action(action="Ação média", rationale="r1", priority="média"),
            Action(action="Ação alta", rationale="r2", priority="alta"),
        ],
    )


def test_markdown_tem_as_tres_secoes_e_ordena_por_gravidade():
    md = render_markdown(sample_report(), footer="rodapé")

    assert md.startswith("**Receita de R$ 10,00 no período.**")
    for section in ["## Destaques do Período", "## Diagnóstico de Pontos Críticos", "## Plano de Ação"]:
        assert section in md
    assert md.index("Risco alto") < md.index("Risco baixo")
    assert "1. **Ação alta**" in md
    assert md.endswith("*rodapé*")


def test_valida_json_do_llm():
    report = ExecutiveReport.model_validate_json(sample_report().model_dump_json())
    assert report.risks[1].severity == "alta"
