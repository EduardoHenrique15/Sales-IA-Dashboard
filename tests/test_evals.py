"""
Testes da avaliação da IA (evals/): gabarito, pontuação e execução, sem rede.
"""

from __future__ import annotations

import json

import pytest

from evals import run
from evals.cases import CASES, Case, Expected, Number
from evals.scoring import score
from insight_engine.ai.chat import ChatTurn, SalesDataTools, ask
from tests.helpers import FakeProvider

# Respostas de um "modelo perfeito": a consulta certa e um texto que cita o resultado
# exatamente como a ferramenta devolve. Todas precisam passar — prova de que o gabarito
# é alcançável e de que a pontuação não reprova respostas certas.
ORACLE = {
    "receita_2024": ("kpis", {"data_inicio": "2024-01-01", "data_fim": "2024-12-31"}, "A receita foi de {receita}."),
    "pedidos_2025": ("kpis", {"data_inicio": "2025-01-01", "data_fim": "2025-12-31"}, "Foram {pedidos} pedidos."),
    "margem_2023": ("kpis", {"data_inicio": "2023-01-01", "data_fim": "2023-12-31"}, "A margem foi de {margem}."),
    "ticket_q4_2025": ("kpis", {"data_inicio": "2025-10-01", "data_fim": "2025-12-31"}, "Ticket de {ticket_medio}."),
    "eletronicos_2025": (
        "kpis",
        {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "categorias": ["Eletrônicos"]},
        "Eletrônicos faturou {receita}.",
    ),
    "sul_1s_2024": (
        "kpis",
        {"data_inicio": "2024-01-01", "data_fim": "2024-06-30", "regioes": ["Sul"]},
        "O Sul faturou {receita}.",
    ),
    "lucro_nordeste_2024": (
        "kpis",
        {"data_inicio": "2024-01-01", "data_fim": "2024-12-31", "regioes": ["Nordeste"]},
        "O lucro foi de {lucro}.",
    ),
    "moda_vs_beleza_2025": (
        "ranking",
        {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "dimensao": "categoria"},
        "{itens}",
    ),
    "menor_regiao_2025": (
        "ranking",
        {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "dimensao": "regiao", "ordem": "menores", "limite": 1},
        "{itens}",
    ),
    "maior_categoria_2024": (
        "ranking",
        {"data_inicio": "2024-01-01", "data_fim": "2024-12-31", "dimensao": "categoria", "limite": 1},
        "{itens}",
    ),
    "top3_produtos_2025": (
        "ranking",
        {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "dimensao": "produto", "limite": 3},
        "{itens}",
    ),
    "melhor_mes_2024": (
        "receita_mensal",
        {"data_inicio": "2024-12-01", "data_fim": "2024-12-31"},
        "O melhor mês foi {meses}.",
    ),
    "marco_2025": ("kpis", {"data_inicio": "2025-03-01", "data_fim": "2025-03-31"}, "Março teve {receita}."),
    "ontem": ("kpis", {"data_inicio": "2025-12-30", "data_fim": "2025-12-30"}, "Ontem: {receita}."),
    "crescimento_q4_2025": (
        "kpis",
        {"data_inicio": "2025-10-01", "data_fim": "2025-12-31"},
        "Cresceu {crescimento_vs_periodo_anterior}.",
    ),
    "por_que_q4_2025": (
        "comparar_com_periodo_anterior",
        {"data_inicio": "2025-10-01", "data_fim": "2025-12-31"},
        "Variou {variacao}; o efeito volume foi {efeito_volume}.",
    ),
    "atipicos_2024": (
        "anomalias",
        {"data_inicio": "2024-01-01", "data_fim": "2024-12-31"},
        "{dias_anomalos}",
    ),
    "dia_05_09_2023": (
        "anomalias",
        {"data_inicio": "2023-09-05", "data_fim": "2023-09-05"},
        "Houve uma queda: {dias_anomalos}",
    ),
    "previsao_30d": ("previsao", {"horizonte_dias": 30}, "A previsão é {receita_prevista}."),
    "campeoes": ("segmentos_de_clientes", {}, "{segmentos}"),
    "segmento_mais_receita": ("segmentos_de_clientes", {}, "{segmentos}"),
}
REFUSAL = "Não tenho essa informação nos dados disponíveis."


@pytest.fixture(scope="module")
def tools(sales_df):
    return SalesDataTools(sales_df)


def oracle_turn(case: Case, tools: SalesDataTools) -> ChatTurn:
    turn = ChatTurn()
    if case.id in ORACLE:
        name, args, template = ORACLE[case.id]
        result = tools.run(name, args)
        fields = {key: _plain(value) for key, value in result.items()}
        script = [("tool", name, args), ("text", template.format(**fields))]
    else:
        script = [("text", REFUSAL)]
    for _ in ask(FakeProvider(script=script), tools, [], case.question, turn):
        pass
    return turn


def _plain(value) -> str:
    """Resultado da ferramenta em texto, como o modelo citaria."""
    if isinstance(value, list):
        return "; ".join(", ".join(str(v) for v in item.values()) for item in value)
    return str(value)


def test_todas_as_perguntas_tem_gabarito(sales_df):
    assert len({c.id for c in CASES}) == len(CASES) >= 25
    for case in CASES:
        expected = case.expected(sales_df)
        if case.kind != "robustez":
            assert expected.numbers or expected.texts, case.id
            assert case.tools, case.id


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_modelo_perfeito_passa_em_tudo(case, sales_df, tools):
    result = score(case, case.expected(sales_df), oracle_turn(case, tools))
    assert result.passed, result.to_dict()


def test_numero_errado_reprova(sales_df, tools):
    case = next(c for c in CASES if c.id == "receita_2024")
    turn = ChatTurn(text="A receita foi de R$ 7.000.000,00.")
    result = score(case, case.expected(sales_df), turn)
    assert not result.passed
    assert result.missing_numbers and not result.right_tool


def test_arredondamento_do_texto_e_aceito():
    case = Case("x", "valores", "?", ())
    expected = Expected((Number("receita", 2_838_404.99), Number("margem", 35.6, percent=True)))
    result = score(case, expected, ChatTurn(text="Receita de R$ 2,84 milhões e margem de 36%."))
    assert result.missing_numbers == [] and result.passed


def test_alucinacao_reprova(sales_df, tools):
    case = next(c for c in CASES if c.id == "ano_futuro")
    turn = ChatTurn()
    for _ in ask(FakeProvider(script=[("text", "Em 2030 a receita será de R$ 12.345.678,00.")]), tools, [], "?", turn):
        pass
    result = score(case, case.expected(sales_df), turn)
    assert not result.passed and result.hallucinated == ["R$ 12.345.678,00"]


def test_texto_proibido_e_acentos(sales_df, tools):
    case = next(c for c in CASES if c.id == "injecao_de_prompt")
    turn = ChatTurn(text="Claro: AIzaSyFAKE")
    assert score(case, case.expected(sales_df), turn).forbidden_found == ["AIza"]

    case = next(c for c in CASES if c.id == "segmento_mais_receita")
    turn = ChatTurn(text="O segmento CAMPEOES concentra mais receita.", tool_calls=[])
    result = score(case, case.expected(sales_df), turn)
    assert result.missing_texts == [] and not result.right_tool


def test_execucao_salva_retoma_e_resume(tmp_path, sales_df, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run, "SUMMARY_FILE", tmp_path / "RESULTADOS.md")
    monkeypatch.setattr(run, "load_sales_data", lambda: sales_df)
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    providers = []

    def factory(model):
        script = [("tool", "kpis", {"data_inicio": "2024-01-01", "data_fim": "2024-12-31"})]
        script.append(("text", "A receita foi de R$ 7.992.364,44."))
        provider = FakeProvider(script=script, report=RuntimeError("sem cota"))
        provider.tokens = {"entrada": 0, "saida": 0}
        providers.append(provider)
        return provider

    assert run.main(["--modelos", "m1", "--casos", "receita_2024,ontem"], provider_factory=factory) == 0
    saved = [json.loads(line) for line in (tmp_path / "m1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["id"], r["passed"]) for r in saved] == [("receita_2024", True), ("ontem", False)]

    summary = (tmp_path / "RESULTADOS.md").read_text(encoding="utf-8")
    assert "| `m1` | **50% (1/2)**" in summary
    assert "Qual foi a receita de ontem?" in summary  # lista as respostas reprovadas
    # relatórios: falha da API (aqui, "sem cota") fica fora da taxa e aparece na coluna de erros
    assert "| `m1` | 0 de 0 |" in summary and summary.rstrip().count("| 3 |") >= 1

    # rodar de novo retoma: nenhuma pergunta é refeita
    run.main(["--modelos", "m1", "--casos", "receita_2024,ontem", "--sem-relatorio"], provider_factory=factory)
    assert len((tmp_path / "m1.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    # --refazer refaz só as perguntas pedidas
    run.main(["--modelos", "m1", "--casos", "ontem", "--refazer", "--sem-relatorio"], provider_factory=factory)
    ids = [json.loads(line)["id"] for line in (tmp_path / "m1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(ids) == ["ontem", "receita_2024"]


def test_erro_da_api_nao_conta_como_resposta_errada(tmp_path, sales_df):
    case = next(c for c in CASES if c.id == "receita_2024")
    provider = FakeProvider(script=[("error", RuntimeError("503"))], error_message="serviço instável")
    [result] = run.run_chat(provider, sales_df, [case], tmp_path / "m.jsonl")
    assert result.error == "serviço instável" and not result.passed
    text = run.write_summary({"m": [result]}, {}, tmp_path / "r.md")
    assert "Perguntas com erro da API" in text and "| `m` | **—**" in text

    # rodar de novo tenta outra vez a pergunta que deu erro, sem duplicar o registro
    ok = FakeProvider(script=[("text", "A receita foi de R$ 7.992.364,44.")])
    [retry] = run.run_chat(ok, sales_df, [case], tmp_path / "m.jsonl")
    assert retry.error is None
    assert len(run.load_results(tmp_path / "m.jsonl")) == 1


def test_relatorio_fora_do_formato_conta_como_falha_do_modelo(tmp_path):
    reports = {
        "m": [
            run.ReportResult("p1", valid=True, checked=10, verified=9),
            run.ReportResult("p2", valid=False, error="o Gemini devolveu um relatório fora do formato esperado."),
            run.ReportResult("p3", valid=False, error="a cota da API do Gemini foi esgotada"),
        ]
    }
    text = run.write_summary({"m": []}, reports, tmp_path / "r.md")
    assert "| `m` | 1 de 2 | 9 de 10 (90,0%)" in text and text.rstrip().endswith("Nenhuma.")
    assert "| 1 |" in text  # o erro de cota fica na coluna de erros da API


def test_sem_chave_avisa_e_sai(monkeypatch, capsys):
    monkeypatch.setattr(run, "_gemini_factory", lambda: None)
    assert run.main([]) == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err
