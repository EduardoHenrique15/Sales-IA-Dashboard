"""
Provedor de LLM: Google Gemini (biblioteca `google-genai`).

  - Relatório: saída estruturada (JSON validado contra `ExecutiveReport`).
  - Chat: chamada de funções (function calling) com execução controlada
    pelo app — o modelo pede uma consulta, o app executa e devolve o
    resultado — e resposta em streaming.

Falhas temporárias (cota, instabilidade) têm novas tentativas e, se o
modelo principal continuar indisponível, o modelo reserva é usado.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

from pydantic import ValidationError

from insight_engine.ai.providers.base import (
    ChatEvent,
    ChatMessage,
    ProviderError,
    RunTool,
    TextDelta,
    ToolCallEvent,
    ToolSpec,
)
from insight_engine.ai.report import ExecutiveReport
from insight_engine.config import get_gemini_fallback_model, get_gemini_model

# A dependência do Gemini é opcional: sem o pacote, o app usa só o motor local.
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


class GeminiProvider:
    name = "Gemini"

    def __init__(self, api_key: str, models: list[str] | None = None) -> None:
        self._client = genai.Client(api_key=api_key)
        # Sem `models`, usa o principal e o reserva da configuração. A avaliação (evals/)
        # passa um modelo só, para medir cada modelo sem o reserva interferir.
        self._models = list(models) if models else None
        # modelo que respondeu por último e tokens gastos (usados pela avaliação)
        self.last_model: str | None = None
        self.tokens = {"entrada": 0, "saida": 0}

    # ------------------------------------------------------------------
    # Relatório
    # ------------------------------------------------------------------
    def generate_report(self, prompt: str, system: str) -> ExecutiveReport:
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=ExecutiveReport,
            automatic_function_calling=_no_afc(),
        )

        def call(model: str):
            return self._client.models.generate_content(model=model, contents=prompt, config=config)

        response, model = _with_fallback(call, self._models)
        self.last_model = model
        self._count_tokens(getattr(response, "usage_metadata", None))
        logger.info("Relatório gerado pelo modelo %s", model)
        if isinstance(response.parsed, ExecutiveReport):
            return response.parsed
        text = (response.text or "").strip()
        if not text:
            raise ProviderError("Resposta vazia da API Gemini")
        return ExecutiveReport.model_validate_json(text)

    # ------------------------------------------------------------------
    # Chat com ferramentas
    # ------------------------------------------------------------------
    def chat_stream(
        self,
        system: str,
        history: list[ChatMessage],
        tools: list[ToolSpec],
        run_tool: RunTool,
        max_tool_rounds: int = 5,
    ) -> Iterator[ChatEvent]:
        contents: list[Any] = [
            genai_types.Content(role="user" if m.role == "user" else "model", parts=[genai_types.Part(text=m.text)])
            for m in history
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            tools=[
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=t.name, description=t.description, parameters_json_schema=t.parameters
                        )
                        for t in tools
                    ]
                )
            ],
            # o app executa as ferramentas (e não a biblioteca), para validar e registrar cada chamada
            automatic_function_calling=_no_afc(),
        )

        for _ in range(max_tool_rounds + 1):
            model_parts: list[Any] = []
            calls: list[Any] = []
            usage = None
            for chunk in _stream_with_fallback(self._client, contents, config, self._models, self._set_model):
                usage = getattr(chunk, "usage_metadata", None) or usage  # o último pedaço traz o total
                content = chunk.candidates[0].content if chunk.candidates else None
                for part in content.parts if content and content.parts else []:
                    model_parts.append(part)  # mantém assinaturas de "pensamento" exigidas pelo modelo
                    if part.function_call:
                        calls.append(part.function_call)
                    elif part.text and not part.thought:
                        yield TextDelta(part.text)

            self._count_tokens(usage)
            if not calls:
                return
            contents.append(genai_types.Content(role="model", parts=model_parts))
            responses = []
            for call in calls:
                args = dict(call.args or {})
                result = run_tool(call.name, args)
                yield ToolCallEvent(call.name, args, result)
                responses.append(genai_types.Part.from_function_response(name=call.name, response=result))
            contents.append(genai_types.Content(role="user", parts=responses))

        yield TextDelta("\n\n(Limite de consultas atingido para esta pergunta. Tente uma pergunta mais específica.)")

    # ------------------------------------------------------------------
    def describe_error(self, exc: Exception) -> str:
        return describe_error(exc, self._models)

    def _set_model(self, model: str) -> None:
        self.last_model = model

    def _count_tokens(self, usage: Any) -> None:
        for key, attr in (("entrada", "prompt_token_count"), ("saida", "candidates_token_count")):
            value = getattr(usage, attr, None)
            if isinstance(value, int):
                self.tokens[key] += value


def configured_models() -> list[str]:
    models = [get_gemini_model(), get_gemini_fallback_model()]
    return list(dict.fromkeys(m for m in models if m))  # sem repetidos, na ordem


def describe_error(exc: Exception, models: list[str] | None = None) -> str:
    """Traduz uma falha do Gemini em uma mensagem curta para a interface."""
    if isinstance(exc, ValidationError):
        return "o Gemini devolveu um relatório fora do formato esperado."
    code = getattr(exc, "code", None)
    if code == 404:
        names = ", ".join(f"`{m}`" for m in models or configured_models())
        return (
            f"nenhum dos modelos configurados ({names}) está disponível para esta chave. "
            "Ajuste as configurações `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL`."
        )
    # o Gemini responde 400 tanto para chave inválida ("API key not valid") quanto para pedidos recusados
    if code in (401, 403) or (code == 400 and "api key" in str(exc).lower()):
        return "a chave do Gemini é inválida ou não tem permissão de acesso."
    if code == 400:
        return "o Gemini recusou a requisição (erro 400; detalhes no log do servidor)."
    if code == 429:
        return "a cota da API do Gemini foi esgotada (inclusive no modelo reserva). Tente novamente em alguns minutos."
    if code is not None and code >= 500:
        return (
            "o serviço do Gemini está instável ou sobrecarregado no momento, "
            "inclusive no modelo reserva. Tente novamente em alguns minutos."
        )
    return "erro inesperado na chamada ao Gemini (detalhes no log do servidor)."


# ------------------------------------------------------------------
# Novas tentativas e modelo reserva
# ------------------------------------------------------------------
def _no_afc():
    return genai_types.AutomaticFunctionCallingConfig(disable=True)


def _is_transient(exc: Exception) -> bool:
    """429 = cota/rate limit; 5xx = instabilidade temporária do serviço."""
    code = getattr(exc, "code", None) or 0
    return code == 429 or code >= 500


def _can_try_other_model(exc: Exception) -> bool:
    # 404 = modelo indisponível para esta chave
    return _is_transient(exc) or getattr(exc, "code", None) == 404


def _with_fallback(call: Callable[[str], Any], models: list[str] | None = None) -> tuple[Any, str]:
    """Chama `call(modelo)` com novas tentativas e, se preciso, o modelo reserva."""
    last_exc: Exception = ProviderError("Nenhum modelo do Gemini configurado")
    for model in models or configured_models():
        for attempt in range(MAX_RETRIES + 1):
            try:
                return call(model), model
            except genai_errors.APIError as exc:
                last_exc = exc
                if _is_transient(exc) and attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))  # backoff exponencial: 2s, 4s
                    continue
                if not _can_try_other_model(exc):
                    raise
                logger.warning("Modelo %s falhou (%s); tentando o próximo.", model, exc.code)
                break
    raise last_exc


def _stream_with_fallback(
    client, contents, config, models: list[str] | None = None, on_model: Callable[[str], None] | None = None
) -> Iterator[Any]:
    """Streaming com a mesma política de tentativas, enquanto nada foi enviado ao usuário."""
    last_exc: Exception = ProviderError("Nenhum modelo do Gemini configurado")
    for model in models or configured_models():
        for attempt in range(MAX_RETRIES + 1):
            started = False
            try:
                if on_model:
                    on_model(model)
                for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
                    started = True
                    yield chunk
                return
            except genai_errors.APIError as exc:
                last_exc = exc
                if started:
                    raise
                if _is_transient(exc) and attempt < MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                if not _can_try_other_model(exc):
                    raise
                logger.warning("Modelo %s falhou no chat (%s); tentando o próximo.", model, exc.code)
                break
    raise last_exc
