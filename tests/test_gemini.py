from types import SimpleNamespace
from unittest import mock

import pytest
from google.genai import errors
from google.genai import types as genai_types
from pydantic import ValidationError

from insight_engine.ai.providers import create_provider, gemini
from insight_engine.ai.providers.base import ChatMessage, TextDelta, ToolCallEvent, ToolSpec
from insight_engine.ai.report import ExecutiveReport
from tests.test_report import sample_report


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": f"erro {code}"}})


@pytest.fixture
def client(monkeypatch):
    """Cliente do Gemini simulado; cada teste define as respostas."""
    fake = mock.Mock()
    monkeypatch.setattr(gemini.genai, "Client", mock.Mock(return_value=fake))
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    return fake.models


def called_models(method) -> list[str]:
    return [c.kwargs["model"] for c in method.call_args_list]


def report_response(report: ExecutiveReport | None = None, text: str | None = None):
    return SimpleNamespace(parsed=report, text=text)


class TestReport:
    def test_saida_estruturada(self, client):
        client.generate_content.side_effect = [report_response(sample_report())]
        report = gemini.GeminiProvider("x").generate_report("prompt", "sistema")

        assert report == sample_report()
        config = client.generate_content.call_args.kwargs["config"]
        assert config.response_schema is ExecutiveReport
        assert config.response_mime_type == "application/json"
        assert config.system_instruction == "sistema"
        assert config.automatic_function_calling.disable is True

    def test_valida_o_json_quando_parsed_nao_vem(self, client):
        client.generate_content.side_effect = [report_response(text=sample_report().model_dump_json())]
        assert gemini.GeminiProvider("x").generate_report("p", "s").headline == sample_report().headline

    def test_json_fora_do_formato(self, client):
        client.generate_content.side_effect = [report_response(text='{"headline": 1}')]
        with pytest.raises(ValidationError):
            gemini.GeminiProvider("x").generate_report("p", "s")

    def test_resposta_vazia(self, client):
        client.generate_content.side_effect = [report_response(text="  ")]
        with pytest.raises(gemini.ProviderError):
            gemini.GeminiProvider("x").generate_report("p", "s")

    def test_erro_temporario_tenta_de_novo_e_depois_usa_o_reserva(self, client):
        client.generate_content.side_effect = [api_error(503)] * 3 + [report_response(sample_report())]
        gemini.GeminiProvider("x").generate_report("p", "s")
        assert called_models(client.generate_content) == ["gemini-flash-latest"] * 3 + ["gemini-flash-lite-latest"]

    def test_modelo_indisponivel_passa_direto_para_o_reserva(self, client):
        client.generate_content.side_effect = [api_error(404), report_response(sample_report())]
        gemini.GeminiProvider("x").generate_report("p", "s")
        assert called_models(client.generate_content) == ["gemini-flash-latest", "gemini-flash-lite-latest"]

    def test_chave_invalida_desiste_sem_tentar_de_novo(self, client):
        client.generate_content.side_effect = [api_error(400)]
        with pytest.raises(errors.APIError):
            gemini.GeminiProvider("x").generate_report("p", "s")
        assert client.generate_content.call_count == 1


def chunk(*parts):
    content = genai_types.Content(role="model", parts=list(parts))
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def call_part(name, **args):
    return genai_types.Part(function_call=genai_types.FunctionCall(name=name, args=args))


class TestChat:
    TOOLS = [ToolSpec("kpis", "KPIs", {"type": "object", "properties": {}})]

    def test_chama_ferramenta_e_depois_responde_em_streaming(self, client):
        client.generate_content_stream.side_effect = [
            iter([chunk(call_part("kpis", data_inicio="2025-01-01"))]),
            iter([chunk(genai_types.Part(text="A receita foi ")), chunk(genai_types.Part(text="R$ 10,00."))]),
        ]
        run_tool = mock.Mock(return_value={"receita": "R$ 10,00"})

        events = list(
            gemini.GeminiProvider("x").chat_stream("sistema", [ChatMessage("user", "Receita?")], self.TOOLS, run_tool)
        )

        assert events == [
            ToolCallEvent("kpis", {"data_inicio": "2025-01-01"}, {"receita": "R$ 10,00"}),
            TextDelta("A receita foi "),
            TextDelta("R$ 10,00."),
        ]
        run_tool.assert_called_once_with("kpis", {"data_inicio": "2025-01-01"})
        # a 2ª chamada leva a pergunta, a chamada da ferramenta e o resultado dela
        second_contents = client.generate_content_stream.call_args_list[1].kwargs["contents"]
        assert [c.role for c in second_contents] == ["user", "model", "user"]
        assert second_contents[2].parts[0].function_response.response == {"receita": "R$ 10,00"}
        config = client.generate_content_stream.call_args.kwargs["config"]
        assert config.tools[0].function_declarations[0].name == "kpis"
        assert config.automatic_function_calling.disable is True

    def test_pensamentos_do_modelo_nao_aparecem_na_resposta(self, client):
        client.generate_content_stream.side_effect = [
            iter([chunk(genai_types.Part(text="raciocínio interno", thought=True), genai_types.Part(text="Oi"))])
        ]
        events = list(gemini.GeminiProvider("x").chat_stream("s", [ChatMessage("user", "?")], self.TOOLS, mock.Mock()))
        assert events == [TextDelta("Oi")]

    def test_limite_de_rodadas_de_ferramentas(self, client):
        client.generate_content_stream.side_effect = lambda **kw: iter([chunk(call_part("kpis"))])
        run_tool = mock.Mock(return_value={})
        events = list(
            gemini.GeminiProvider("x").chat_stream(
                "s", [ChatMessage("user", "?")], self.TOOLS, run_tool, max_tool_rounds=2
            )
        )
        assert run_tool.call_count == 3
        assert "Limite de consultas" in events[-1].text

    def test_erro_antes_do_streaming_usa_o_modelo_reserva(self, client):
        client.generate_content_stream.side_effect = [api_error(404), iter([chunk(genai_types.Part(text="ok"))])]
        events = list(gemini.GeminiProvider("x").chat_stream("s", [ChatMessage("user", "?")], self.TOOLS, mock.Mock()))
        assert events == [TextDelta("ok")]
        assert called_models(client.generate_content_stream) == ["gemini-flash-latest", "gemini-flash-lite-latest"]


class TestModeloEscolhidoETokens:
    """Usados pela avaliação (evals/): um modelo só, sem reserva, e contagem de tokens."""

    def test_relatorio_com_um_modelo_so_nao_usa_o_reserva(self, client):
        client.generate_content.side_effect = [api_error(404)]
        provider = gemini.GeminiProvider("x", models=["so-este"])
        with pytest.raises(errors.APIError):
            provider.generate_report("p", "s")
        assert called_models(client.generate_content) == ["so-este"]

    def test_conta_tokens_e_registra_o_modelo(self, client):
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=30)
        client.generate_content.side_effect = [SimpleNamespace(parsed=sample_report(), text=None, usage_metadata=usage)]
        last = chunk(genai_types.Part(text="ok"))
        last.usage_metadata = SimpleNamespace(prompt_token_count=50, candidates_token_count=5)
        client.generate_content_stream.side_effect = [iter([chunk(genai_types.Part(text="o")), last])]

        provider = gemini.GeminiProvider("x", models=["m1"])
        provider.generate_report("p", "s")
        list(provider.chat_stream("s", [ChatMessage("user", "?")], TestChat.TOOLS, mock.Mock()))

        assert provider.tokens == {"entrada": 170, "saida": 35}
        assert provider.last_model == "m1"


def test_modelos_configurados_sem_repeticao(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "m1")
    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "m1")
    assert gemini.configured_models() == ["m1"]


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (api_error(404), "nenhum dos modelos configurados"),
        (api_error(403), "chave do Gemini é inválida"),
        (errors.APIError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}}), "inválida"),
        (api_error(400), "recusou a requisição"),
        (api_error(429), "cota da API do Gemini foi esgotada"),
        (api_error(503), "instável ou sobrecarregado"),
        (RuntimeError("?"), "erro inesperado"),
    ],
)
def test_mensagens_de_erro_para_a_interface(exc, expected):
    assert expected in gemini.describe_error(exc)


def test_mensagem_para_json_invalido():
    with pytest.raises(ValidationError) as exc:
        ExecutiveReport.model_validate_json("{}")
    assert "fora do formato" in gemini.describe_error(exc.value)


def test_fabrica_de_provedores(monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", mock.Mock())
    assert create_provider(None) == (None, None)
    provider, error = create_provider("chave")
    assert provider.name == "Gemini" and error is None

    monkeypatch.setattr(gemini, "GENAI_AVAILABLE", False)
    assert create_provider("chave") == (None, "o pacote `google-genai` não está instalado.")
