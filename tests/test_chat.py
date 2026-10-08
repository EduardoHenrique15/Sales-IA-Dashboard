import pytest

from insight_engine.ai.chat import ChatTurn, SalesDataTools, ask
from insight_engine.ai.providers.base import ChatMessage
from tests.helpers import FakeProvider


@pytest.fixture(scope="module")
def tools(sales_df):
    return SalesDataTools(sales_df)


def test_descreve_a_base_e_as_ferramentas(tools):
    assert "01/01/2023 a 31/12/2025" in tools.dataset_info()
    assert "ontem = 30/12/2025" in tools.dataset_info()  # datas relativas a partir do fim da base
    assert [s.name for s in tools.specs()] == [
        "kpis",
        "ranking",
        "receita_mensal",
        "comparar_com_periodo_anterior",
        "anomalias",
        "previsao",
        "segmentos_de_clientes",
    ]
    categories = tools.specs()[0].parameters["properties"]["categorias"]["items"]["enum"]
    assert "Moda" in categories


def test_kpis_batem_com_o_dashboard(tools):
    result = tools.run("kpis", {"data_inicio": "2025-10-02", "data_fim": "2025-12-31"})
    assert result["receita"] == "R$ 2.838.404,99"
    assert result["crescimento_vs_periodo_anterior"] == "+31,4%"


def test_ranking_das_menores_regioes(tools):
    result = tools.run(
        "ranking",
        {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "dimensao": "regiao", "ordem": "menores", "limite": 1},
    )
    assert result["itens"][0]["nome"] == "Norte"


def test_variacao_e_anomalias_do_terceiro_trimestre_de_2024(tools):
    period = {"data_inicio": "2024-07-01", "data_fim": "2024-09-30"}
    bridge = tools.run("comparar_com_periodo_anterior", period)
    assert bridge["efeito_mix"] == "-R$ 52.025,13"
    assert bridge["variacao_percentual"].endswith("%")  # o modelo não precisa calcular a variação
    anomalies = tools.run("anomalias", period)
    assert [d["data"] for d in anomalies["dias_anomalos"]] == ["07/08/2024"]


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("kpis", {"data_inicio": "2030-01-01", "data_fim": "2030-02-01"}, "fora da base"),
        ("kpis", {"data_inicio": "ontem", "data_fim": "2025-01-01"}, "AAAA-MM-DD"),
        ("kpis", {"data_inicio": "2025-02-01", "data_fim": "2025-01-01"}, "anterior a data_fim"),
        ("ranking", {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "dimensao": "cor"}, "dimensao"),
        ("kpis", {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "categorias": ["Carros"]}, "Carros"),
        ("previsao", {"horizonte_dias": 7}, "30, 60 ou 90"),
        ("kpis", {"data_inicio": "2025-01-01", "data_fim": "2025-12-31", "extra": 1}, "Parâmetros inválidos"),
        ("apagar_tudo", {}, "Ferramenta desconhecida"),
    ],
)
def test_parametros_invalidos_voltam_como_erro_para_o_modelo(tools, name, args, message):
    assert message in tools.run(name, args)["erro"]


def test_base_sem_cliente_nao_oferece_segmentacao(small_sales_df):
    tools = SalesDataTools(small_sales_df, has_cost=False, has_customers=False)
    assert "segmentos_de_clientes" not in [s.name for s in tools.specs()]
    assert "não tem custo" in tools.dataset_info()
    assert "erro" in tools.run("segmentos_de_clientes", {})


def test_pergunta_com_consulta_e_checagem(tools):
    provider = FakeProvider(
        script=[
            ("tool", "kpis", {"data_inicio": "2025-10-02", "data_fim": "2025-12-31"}),
            ("text", "A receita foi de R$ 2.838.404,99 "),
            ("text", "e a meta é R$ 3 milhões."),
        ]
    )
    turn = ChatTurn()
    chunks = list(ask(provider, tools, [ChatMessage("user", "oi"), ChatMessage("assistant", "olá")], "Receita?", turn))

    assert "".join(chunks) == turn.text
    assert turn.tool_calls[0].result["receita"] == "R$ 2.838.404,99"
    assert turn.verification.unverified == ["R$ 3 milhões"]
    assert provider.last_history[-1] == ChatMessage("user", "Receita?")
    assert "31/12/2025" in provider.last_system


def test_pergunta_longa_e_cortada(tools):
    provider = FakeProvider(script=[("text", "ok")])
    list(ask(provider, tools, [], "x" * 2000, ChatTurn()))
    assert len(provider.last_history[-1].text) == 500
