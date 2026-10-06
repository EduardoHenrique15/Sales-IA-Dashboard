"""Utilitários para montar dados de teste."""

from __future__ import annotations

from datetime import date

import pandas as pd

DAY_MS = 86_400_000


def market_chart(prices: list[float], volumes: list[float] | None = None, start_day: int = 20_000) -> dict:
    """Monta uma resposta no formato da CoinGecko `/market_chart`."""
    volumes = volumes if volumes is not None else [1_000.0] * len(prices)
    return {
        "prices": [[(start_day + i) * DAY_MS, p] for i, p in enumerate(prices)],
        "total_volumes": [[(start_day + i) * DAY_MS, v] for i, v in enumerate(volumes)],
    }


def as_date(text: str) -> date:
    return pd.Timestamp(text).date()


class FakeProvider:
    """Provedor de LLM falso, que segue a interface `LLMProvider`.

    `report`: relatório devolvido (ou exceção a levantar).
    `script`: para o chat, lista de passos; cada passo é ("tool", nome, args) ou ("text", texto).
    """

    name = "Fake"

    def __init__(self, report=None, script=None, error_message="falha simulada"):
        self.report = report
        self.script = script or []
        self.error_message = error_message
        self.report_calls = 0
        self.prompts: list[str] = []

    def generate_report(self, prompt, system):
        self.report_calls += 1
        self.prompts.append(prompt)
        if isinstance(self.report, Exception):
            raise self.report
        return self.report

    def chat_stream(self, system, history, tools, run_tool, max_tool_rounds=5):
        from insight_engine.ai.providers.base import TextDelta, ToolCallEvent

        self.last_system, self.last_history, self.last_tools = system, history, tools
        for step in self.script:
            if step[0] == "tool":
                _, name, args = step
                yield ToolCallEvent(name, args, run_tool(name, args))
            elif step[0] == "error":
                raise step[1]
            else:
                yield TextDelta(step[1])

    def describe_error(self, exc):
        return self.error_message
