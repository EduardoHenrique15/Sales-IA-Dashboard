import pytest

from insight_engine.ai.guardrail import extract_numbers, verify

FACTS = """
- Receita total: R$ 2.838.404,99
- Margem de lucro: 35,6%
- Pedidos: 2.431
- Crescimento: +31,4%
- Variação: -R$ 70.512,08; efeito mix -R$ 106.971,61
- Anomalias: 07/08/2024 (queda, 4 pedidos vs 15 esperados, -73%)
"""


def test_extrai_moedas_porcentagens_e_numeros():
    values = {(n.value, n.is_percent) for n in extract_numbers(FACTS)}
    assert (2838404.99, False) in values
    assert (35.6, True) in values
    assert (2431.0, False) in values
    assert (-70512.08, False) in values


def test_ignora_datas_anos_e_contagens_pequenas():
    assert extract_numbers("Em 07/08/2024, nos próximos 30 dias de 2026, faça 3 ações.") == []


@pytest.mark.parametrize(
    "citation",
    [
        "R$ 2.838.404,99",  # exato
        "R$ 2,84 milhões",  # abreviado
        "36%",  # arredondado
        "31%",
        "2.431 pedidos",
        "R$ 107 mil",  # valor absoluto de um efeito negativo
        "-73%",
    ],
)
def test_aceita_numeros_dos_fatos_mesmo_arredondados(citation):
    assert verify(f"O relatório cita {citation}.", FACTS).ok


@pytest.mark.parametrize("citation", ["R$ 900 mil", "40%", "R$ 3,5 milhões", "1.200 pedidos"])
def test_aponta_numeros_que_nao_estao_nos_fatos(citation):
    result = verify(f"O relatório cita {citation}.", FACTS)
    assert not result.ok
    assert result.unverified == [citation.replace(" pedidos", "")]


def test_resumo_da_checagem():
    result = verify("Receita de R$ 2,84 milhões, margem de 36% e meta de R$ 900 mil.", FACTS)
    assert (result.checked, result.verified, result.unverified) == (3, 2, ["R$ 900 mil"])


def test_hifen_de_lista_nao_e_sinal_negativo():
    assert verify("- R$ 2.838.404,99 de receita\n- 2.431 pedidos", FACTS).ok


def test_sinal_explicito_trocado_e_apontado():
    # o fato é uma queda de 73%: "+73%" inverte o sentido
    assert not verify("Os pedidos tiveram +73% no dia.", FACTS).ok
    assert not verify("Crescimento de -31,4%.", FACTS).ok
    assert verify("Queda de 73% e crescimento de +31,4%.", FACTS).ok
    assert verify("Variação de −R$ 70.512,08.", FACTS).ok  # sinal de menos tipográfico


@pytest.mark.parametrize(
    ("citation", "ok"),
    [("R$ 2,8 milhões", True), ("R$ 2.838 mil", True), ("R$ 2,9 milhões", False), ("R$ 3 milhões", False)],
)
def test_arredondamento_segue_as_casas_escritas(citation, ok):
    assert verify(f"Receita de {citation}.", FACTS).ok is ok


def test_numero_proximo_mas_nao_arredondado_e_apontado():
    assert not verify("Receita de R$ 2.850.000.", FACTS).ok
