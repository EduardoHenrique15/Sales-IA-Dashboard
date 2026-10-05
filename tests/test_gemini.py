from unittest import mock

import pytest
from google.genai import errors

from insight_engine.ai import gemini


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": f"erro {code}"}})


def ok(text: str = "## Relatório") -> mock.Mock:
    return mock.Mock(text=text)


@pytest.fixture
def client(monkeypatch):
    """Cliente do Gemini simulado; cada teste define as respostas."""
    fake = mock.Mock()
    monkeypatch.setattr(gemini.genai, "Client", mock.Mock(return_value=fake))
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    return fake.models.generate_content


def called_models(generate_content) -> list[str]:
    return [c.kwargs["model"] for c in generate_content.call_args_list]


def test_sucesso_na_primeira_tentativa(client):
    client.side_effect = [ok("texto")]
    assert gemini.generate("prompt", api_key="x") == "texto"
    assert called_models(client) == [gemini.get_gemini_model()]


def test_desliga_a_chamada_automatica_de_funcoes(client):
    client.side_effect = [ok()]
    gemini.generate("prompt", api_key="x")
    assert client.call_args.kwargs["config"].automatic_function_calling.disable is True


def test_erro_temporario_tenta_de_novo_no_mesmo_modelo(client):
    client.side_effect = [api_error(503), ok()]
    gemini.generate("prompt", api_key="x")
    assert called_models(client) == ["gemini-flash-latest"] * 2


def test_modelo_sobrecarregado_passa_para_o_reserva(client):
    client.side_effect = [api_error(503)] * 3 + [ok()]
    gemini.generate("prompt", api_key="x")
    assert called_models(client) == ["gemini-flash-latest"] * 3 + ["gemini-flash-lite-latest"]


def test_modelo_indisponivel_passa_direto_para_o_reserva(client):
    client.side_effect = [api_error(404), ok()]
    gemini.generate("prompt", api_key="x")
    assert called_models(client) == ["gemini-flash-latest", "gemini-flash-lite-latest"]


def test_chave_invalida_desiste_sem_tentar_de_novo(client):
    client.side_effect = [api_error(400)]
    with pytest.raises(errors.APIError):
        gemini.generate("prompt", api_key="x")
    assert client.call_count == 1


def test_todos_os_modelos_falham(client):
    client.side_effect = [api_error(429)] * 6
    with pytest.raises(errors.APIError):
        gemini.generate("prompt", api_key="x")
    assert client.call_count == 6


def test_resposta_vazia(client):
    client.side_effect = [ok("   ")]
    with pytest.raises(gemini.GeminiError):
        gemini.generate("prompt", api_key="x")


def test_modelos_configurados_sem_repeticao(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "m1")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "m1")
    assert gemini.configured_models() == ["m1"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (404, "nenhum dos modelos configurados"),
        (400, "chave do Gemini é inválida"),
        (403, "chave do Gemini é inválida"),
        (429, "cota da API do Gemini foi esgotada"),
        (503, "instável ou sobrecarregado"),
    ],
)
def test_mensagens_de_erro_para_a_interface(code, expected):
    assert expected in gemini.describe_error(api_error(code))


def test_mensagem_para_erro_desconhecido():
    assert "erro inesperado" in gemini.describe_error(RuntimeError("?"))
