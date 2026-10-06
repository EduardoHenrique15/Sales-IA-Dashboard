"""
Interface comum dos provedores de LLM.

O restante do app só conhece `LLMProvider`: para trocar o Gemini por outro
modelo (Claude, OpenAI, um modelo local...), basta implementar esta
interface e registrá-la em `providers.create_provider`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from insight_engine.ai.report import ExecutiveReport


class ProviderError(Exception):
    """Falha conhecida do provedor (resposta vazia, fora do formato etc.)."""


@dataclass(frozen=True)
class ToolSpec:
    """Ferramenta que o LLM pode chamar no chat (parâmetros em JSON Schema)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class TextDelta:
    """Trecho da resposta, entregue à medida que o modelo escreve (streaming)."""

    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    """O modelo consultou uma ferramenta (mostrado ao usuário por transparência)."""

    name: str
    args: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)


ChatEvent = TextDelta | ToolCallEvent
RunTool = Callable[[str, dict[str, Any]], dict[str, Any]]


class LLMProvider(Protocol):
    name: str

    def generate_report(self, prompt: str, system: str) -> ExecutiveReport:
        """Gera o Relatório Executivo estruturado."""
        ...

    def chat_stream(
        self,
        system: str,
        history: list[ChatMessage],
        tools: list[ToolSpec],
        run_tool: RunTool,
        max_tool_rounds: int = 5,
    ) -> Iterator[ChatEvent]:
        """Responde à última mensagem do histórico, chamando ferramentas quando preciso."""
        ...

    def describe_error(self, exc: Exception) -> str:
        """Mensagem curta, para a interface, explicando uma falha."""
        ...
