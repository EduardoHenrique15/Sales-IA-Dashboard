"""
Provedor de LLM: Google Gemini (biblioteca `google-genai`).

Tenta o modelo principal com novas tentativas em falhas temporárias
e, se ele continuar indisponível, sobrecarregado ou sem cota, tenta o
modelo reserva antes de desistir.
"""

from __future__ import annotations

import logging
import time

from insight_engine.config import get_gemini_fallback_model, get_gemini_model

# A dependência do Gemini é opcional: sem o pacote, o agente usa
# diretamente o relatório estatístico.
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Falha conhecida do provedor (ex.: resposta vazia)."""


def generate(prompt: str, api_key: str) -> str:
    """Envia `prompt` ao Gemini e devolve o texto gerado."""
    client = genai.Client(api_key=api_key)

    last_exc = None
    for model in configured_models():
        try:
            text = _call_model(client, model, prompt)
            logger.info("Relatório gerado pelo modelo %s", model)
            return text
        except genai_errors.APIError as exc:
            if not _can_try_other_model(exc):
                raise
            logger.warning("Modelo %s falhou (%s); tentando o próximo.", model, exc.code)
            last_exc = exc

    raise last_exc


def configured_models() -> list[str]:
    models = [get_gemini_model(), get_gemini_fallback_model()]
    return list(dict.fromkeys(m for m in models if m))  # sem repetidos, na ordem


def describe_error(exc: Exception) -> str:
    """Traduz uma falha do Gemini em uma mensagem curta para a interface."""
    code = getattr(exc, "code", None)
    if code == 404:
        models = ", ".join(f"`{m}`" for m in configured_models())
        return (
            f"nenhum dos modelos configurados ({models}) está disponível para esta chave. "
            "Ajuste as configurações `GEMINI_MODEL` / `GEMINI_FALLBACK_MODEL`."
        )
    if code in (400, 401, 403):
        return "a chave do Gemini é inválida ou não tem permissão de acesso."
    if code == 429:
        return (
            "a cota da API do Gemini foi esgotada (inclusive no modelo reserva). "
            "Tente novamente em alguns minutos."
        )
    if code is not None and code >= 500:
        return (
            "o serviço do Gemini está instável ou sobrecarregado no momento, "
            "inclusive no modelo reserva. Tente novamente em alguns minutos."
        )
    return "erro inesperado na chamada ao Gemini (detalhes no log do servidor)."


def _is_transient(exc: Exception) -> bool:
    """429 = cota/rate limit; 5xx = instabilidade temporária do serviço."""
    code = getattr(exc, "code", None) or 0
    return code == 429 or code >= 500


def _can_try_other_model(exc: Exception) -> bool:
    # 404 = modelo indisponível para esta chave
    return _is_transient(exc) or getattr(exc, "code", None) == 404


def _call_model(client, model: str, prompt: str, max_retries: int = 2) -> str:
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                # O agente não usa ferramentas; desligar a chamada automática
                # de funções (AFC) também evita um aviso da biblioteca.
                config=genai_types.GenerateContentConfig(
                    automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc) and attempt < max_retries:
                time.sleep(2 ** (attempt + 1))  # backoff exponencial: 2s, 4s
                continue
            raise

        text = (response.text or "").strip()
        if not text:
            raise GeminiError("Resposta vazia da API Gemini")
        return text

    raise GeminiError("Número máximo de tentativas excedido")  # pragma: no cover
