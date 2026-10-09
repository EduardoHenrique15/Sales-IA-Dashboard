"""
"Converse com seus dados": perguntas em linguagem natural respondidas com
chamadas de funções (function calling).

Segurança: o modelo NÃO gera nem executa código. Ele só escolhe entre as
consultas pré-definidas abaixo e informa os parâmetros; o app valida cada
parâmetro (datas dentro da base, categorias existentes, limites de tamanho)
e executa a consulta. Parâmetros inválidos voltam como mensagem de erro
para o modelo se corrigir.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from insight_engine.ai.guardrail import Verification, verify
from insight_engine.ai.prompts import CHAT_SYSTEM
from insight_engine.ai.providers.base import ChatEvent, ChatMessage, LLMProvider, TextDelta, ToolCallEvent, ToolSpec
from insight_engine.analytics.anomalies import daily_series, detect_anomalies
from insight_engine.analytics.customers import pareto_share, segment_customers
from insight_engine.analytics.forecasting import InsufficientDataError, forecast_revenue
from insight_engine.analytics.kpis import compute_sales_kpis
from insight_engine.analytics.periods import comparison_window, previous_period_df, window_is_covered
from insight_engine.analytics.variance import revenue_bridge
from insight_engine.formatting import format_brl, format_number, format_pct

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
MAX_HISTORY_MESSAGES = 10
MAX_RANKING = 20
MONTH_NAMES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


class ToolInputError(ValueError):
    """Parâmetro inválido enviado pelo modelo; a mensagem volta para ele se corrigir."""


@dataclass
class ChatTurn:
    """Resultado de uma pergunta: texto final, consultas feitas e checagem dos números."""

    text: str = ""
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    verification: Verification | None = None


class SalesDataTools:
    """Consultas seguras sobre uma base de vendas, expostas ao LLM como ferramentas."""

    def __init__(self, df: pd.DataFrame, has_cost: bool = True, has_customers: bool = True) -> None:
        self.df = df
        self.has_cost = has_cost
        self.has_customers = has_customers
        self.min_date = df["date"].min().date()
        self.max_date = df["date"].max().date()
        self.categories = sorted(df["category"].unique())
        self.regions = sorted(df["region"].unique())
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "kpis": self.kpis,
            "ranking": self.ranking,
            "receita_mensal": self.monthly_revenue,
            "comparar_com_periodo_anterior": self.compare_previous,
            "anomalias": self.anomalies,
            "previsao": self.forecast,
        }
        if has_customers:
            self._handlers["segmentos_de_clientes"] = self.customer_segments

    # ------------------------------------------------------------------
    # Descrição para o modelo
    # ------------------------------------------------------------------
    def dataset_info(self) -> str:
        return (
            f"A base tem vendas de {self.min_date:%d/%m/%Y} a {self.max_date:%d/%m/%Y} "
            f"(use datas no formato AAAA-MM-DD). Hoje, para fins de análise, é {self.max_date:%d/%m/%Y}: "
            "datas relativas (ontem, semana passada, mês passado) contam a partir de hoje "
            f"(ex.: ontem = {self.max_date - timedelta(days=1):%d/%m/%Y}). "
            f"Categorias: {', '.join(self.categories)}. Regiões: {', '.join(self.regions)}."
            + ("" if self.has_cost else " A base não tem custo: lucro e margem não estão disponíveis.")
        )

    def specs(self) -> list[ToolSpec]:
        period = {
            "data_inicio": {"type": "string", "description": "Data inicial, AAAA-MM-DD."},
            "data_fim": {"type": "string", "description": "Data final, AAAA-MM-DD."},
        }
        segments = {
            "categorias": {
                "type": "array",
                "items": {"type": "string", "enum": self.categories},
                "description": "Filtrar por categorias (vazio = todas).",
            },
            "regioes": {
                "type": "array",
                "items": {"type": "string", "enum": self.regions},
                "description": "Filtrar por regiões (vazio = todas).",
            },
        }

        def schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
            return {"type": "object", "properties": properties, "required": required}

        specs = [
            ToolSpec(
                "kpis",
                "Receita, lucro, margem, pedidos, unidades, ticket médio e crescimento vs o período anterior "
                "de mesma duração, para um período e filtros.",
                schema({**period, **segments}, ["data_inicio", "data_fim"]),
            ),
            ToolSpec(
                "ranking",
                "Ranking de receita por categoria, região ou produto em um período.",
                schema(
                    {
                        **period,
                        "dimensao": {"type": "string", "enum": ["categoria", "regiao", "produto"]},
                        "ordem": {"type": "string", "enum": ["maiores", "menores"]},
                        "limite": {"type": "integer", "description": f"Quantos itens (1 a {MAX_RANKING})."},
                        **segments,
                    },
                    ["data_inicio", "data_fim", "dimensao"],
                ),
            ),
            ToolSpec(
                "receita_mensal",
                "Receita de cada mês em um período (até 36 meses).",
                schema({**period, **segments}, ["data_inicio", "data_fim"]),
            ),
            ToolSpec(
                "comparar_com_periodo_anterior",
                "Explica a variação da receita vs o período anterior de mesma duração: efeitos volume, preço "
                "e mix, e a variação por categoria.",
                schema({**period, **segments}, ["data_inicio", "data_fim"]),
            ),
            ToolSpec(
                "anomalias",
                "Dias fora do padrão em um período. Use metrica 'pedidos' (padrão), o sinal mais confiável "
                "para incidentes; use 'receita' só se o usuário pedir receita.",
                schema(
                    {**period, "metrica": {"type": "string", "enum": ["pedidos", "receita"]}, **segments},
                    ["data_inicio", "data_fim"],
                ),
            ),
            ToolSpec(
                "previsao",
                "Previsão da receita para os próximos dias após o fim da base, com o erro típico do modelo.",
                schema(
                    {"horizonte_dias": {"type": "integer", "enum": [30, 60, 90]}, **segments},
                    ["horizonte_dias"],
                ),
            ),
        ]
        if self.has_customers:
            specs.append(
                ToolSpec(
                    "segmentos_de_clientes",
                    "Segmentação RFM dos clientes (histórico completo): clientes e receita por segmento.",
                    schema({"regioes": segments["regioes"]}, []),
                )
            )
        return specs

    # ------------------------------------------------------------------
    # Execução
    # ------------------------------------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Executa a ferramenta `name`; erros de parâmetro voltam como {"erro": ...}."""
        handler = self._handlers.get(name)
        if handler is None:
            return {"erro": f"Ferramenta desconhecida: {name}"}
        try:
            return handler(**args)
        except ToolInputError as exc:
            return {"erro": str(exc)}
        except TypeError:  # parâmetro inexistente ou faltando
            return {"erro": f"Parâmetros inválidos para '{name}'. Confira a descrição da ferramenta."}
        except Exception:  # noqa: BLE001 - nunca derrubar o chat por causa de uma consulta
            logger.exception("Falha na ferramenta %s", name)
            return {"erro": "Não foi possível calcular esta consulta."}

    def kpis(self, data_inicio: str, data_fim: str, categorias=None, regioes=None) -> dict[str, Any]:
        start, end = self._period(data_inicio, data_fim)
        df = self._segment(categorias, regioes)
        current, previous = self._slice(df, start, end), self._previous(df, start, end)
        k = compute_sales_kpis(current, previous)
        result = {
            "periodo": f"{start:%d/%m/%Y} a {end:%d/%m/%Y}",
            "receita": format_brl(k.total_revenue),
            "pedidos": format_number(k.n_orders),
            "unidades": format_number(k.total_units),
            "ticket_medio": format_brl(k.avg_ticket),
            "crescimento_vs_periodo_anterior": format_pct(k.revenue_growth_pct, signed=True)
            if k.revenue_growth_pct is not None
            else "sem período anterior comparável",
        }
        if k.total_profit is not None and k.margin_pct is not None:
            result |= {"lucro": format_brl(k.total_profit), "margem": format_pct(k.margin_pct)}
        return result

    def ranking(
        self, data_inicio: str, data_fim: str, dimensao: str, ordem="maiores", limite=5, categorias=None, regioes=None
    ) -> dict[str, Any]:
        column = {"categoria": "category", "regiao": "region", "produto": "product"}.get(dimensao)
        if column is None:
            raise ToolInputError("dimensao deve ser categoria, regiao ou produto.")
        if ordem not in ("maiores", "menores"):
            raise ToolInputError("ordem deve ser 'maiores' ou 'menores'.")
        try:
            limit = max(1, min(int(limite), MAX_RANKING))
        except (TypeError, ValueError) as exc:
            raise ToolInputError(f"limite deve ser um número inteiro de 1 a {MAX_RANKING}.") from exc
        start, end = self._period(data_inicio, data_fim)
        current = self._slice(self._segment(categorias, regioes), start, end)
        revenue = current.groupby(column)["revenue"].sum().sort_values(ascending=(ordem == "menores"))
        total = revenue.sum() or 1
        return {
            "periodo": f"{start:%d/%m/%Y} a {end:%d/%m/%Y}",
            "itens": [
                {"nome": name, "receita": format_brl(value), "participacao": format_pct(value / total * 100)}
                for name, value in revenue.head(limit).items()
            ],
        }

    def monthly_revenue(self, data_inicio: str, data_fim: str, categorias=None, regioes=None) -> dict[str, Any]:
        start, end = self._period(data_inicio, data_fim)
        current = self._slice(self._segment(categorias, regioes), start, end)
        monthly = current.groupby(current["date"].dt.to_period("M"))["revenue"].sum().tail(36)
        return {
            "meses": [
                {"mes": f"{MONTH_NAMES[p.month - 1]}/{p.year}", "receita": format_brl(v)} for p, v in monthly.items()
            ]
        }

    def compare_previous(self, data_inicio: str, data_fim: str, categorias=None, regioes=None) -> dict[str, Any]:
        start, end = self._period(data_inicio, data_fim)
        df = self._segment(categorias, regioes)
        current, previous = self._slice(df, start, end), self._previous(df, start, end)
        if current.empty or previous.empty:
            return {"erro": "Não há vendas no período ou no período anterior para comparar."}
        bridge = revenue_bridge(current, previous)
        return {
            "receita_periodo_anterior": format_brl(bridge.previous_revenue),
            "receita_periodo": format_brl(bridge.current_revenue),
            "variacao": format_brl(bridge.total_change),
            "variacao_percentual": format_pct(bridge.total_change / bridge.previous_revenue * 100, signed=True),
            "efeito_volume": format_brl(bridge.volume_effect),
            "efeito_preco": format_brl(bridge.price_effect),
            "efeito_mix": format_brl(bridge.mix_effect),
            "variacao_por_categoria": {name: format_brl(v) for name, v in bridge.by_segment["total"].items()},
        }

    def anomalies(self, data_inicio: str, data_fim: str, metrica="pedidos", categorias=None, regioes=None):
        if metrica not in ("pedidos", "receita"):
            raise ToolInputError("metrica deve ser 'pedidos' ou 'receita'.")
        start, end = self._period(data_inicio, data_fim)
        series = daily_series(
            self._segment(categorias, regioes), "orders" if metrica == "pedidos" else "revenue", end=self.max_date
        )
        try:
            found = detect_anomalies(series)
        except ValueError as exc:
            raise ToolInputError(str(exc)) from exc
        found = found[(found.index.date >= start) & (found.index.date <= end)]
        fmt = format_number if metrica == "pedidos" else format_brl
        return {
            "metrica": metrica,
            "dias_anomalos": [
                {
                    "data": f"{day:%d/%m/%Y}",
                    "tipo": row.kind,
                    "observado": fmt(row.value),
                    "esperado": fmt(row.expected),
                    "desvio": format_pct(row.deviation_pct, 0, signed=True),
                }
                for day, row in found.head(10).iterrows()
            ],
        }

    def forecast(self, horizonte_dias=30, categorias=None, regioes=None) -> dict[str, Any]:
        horizon = int(horizonte_dias)
        if horizon not in (30, 60, 90):
            raise ToolInputError("horizonte_dias deve ser 30, 60 ou 90.")
        try:
            series = daily_series(self._segment(categorias, regioes), "revenue", end=self.max_date)
            result = forecast_revenue(series, horizon)
        except (InsufficientDataError, ValueError) as exc:
            raise ToolInputError(str(exc)) from exc
        return {
            "a_partir_de": f"{self.max_date:%d/%m/%Y}",
            "horizonte_dias": horizon,
            "receita_prevista": format_brl(result.forecast["yhat"].sum()),
            "erro_tipico_do_total": f"± {format_pct(result.total_error_pct)}",
            "modelo": result.best_model,
        }

    def customer_segments(self, regioes=None) -> dict[str, Any]:
        # só os segmentos RFM: o K-Means seria lento demais para uma resposta de chat
        seg = segment_customers(self._segment(None, regioes), reference_date=self._reference(), with_clusters=False)
        return {
            "clientes": format_number(len(seg.customers)),
            "receita_dos_20pct_melhores_clientes": format_pct(pareto_share(seg.customers)),
            "segmentos": [
                {
                    "segmento": name,
                    "clientes": format_number(row.customers),
                    "participacao_na_receita": format_pct(row.revenue_pct),
                    "acao_sugerida": row.action,
                }
                for name, row in seg.segments.iterrows()
            ],
        }

    # ------------------------------------------------------------------
    # Validação
    # ------------------------------------------------------------------
    def _period(self, start: str, end: str) -> tuple[date, date]:
        try:
            first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
        except ValueError as exc:
            raise ToolInputError("Datas devem estar no formato AAAA-MM-DD.") from exc
        if first > last:
            raise ToolInputError("data_inicio deve ser anterior a data_fim.")
        if last < self.min_date or first > self.max_date:
            raise ToolInputError(
                f"Período fora da base, que vai de {self.min_date:%Y-%m-%d} a {self.max_date:%Y-%m-%d}."
            )
        return max(first, self.min_date), min(last, self.max_date)

    def _segment(self, categories, regions) -> pd.DataFrame:
        df = self.df
        for values, column, valid in ((categories, "category", self.categories), (regions, "region", self.regions)):
            if isinstance(values, str):  # o modelo às vezes manda um texto em vez de uma lista
                values = [values]
            if values:
                unknown = sorted(set(values) - set(valid))
                if unknown:
                    raise ToolInputError(f"Valores inexistentes: {', '.join(unknown)}. Válidos: {', '.join(valid)}.")
                df = df[df[column].isin(values)]
        return df

    @staticmethod
    def _slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        dates = df["date"].dt.date
        return df[(dates >= start) & (dates <= end)]

    def _previous(self, df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        """Período anterior de mesma duração; vazio se ele começa antes da base (comparação parcial)."""
        if not window_is_covered(*comparison_window(start, end), self.min_date):
            return df.iloc[0:0]
        return previous_period_df(df, start, end)

    def _reference(self) -> pd.Timestamp:
        return pd.Timestamp(self.max_date) + pd.Timedelta(days=1)


def ask(
    provider: LLMProvider,
    tools: SalesDataTools,
    history: list[ChatMessage],
    question: str,
    turn: ChatTurn,
) -> Iterator[str]:
    """Responde `question`, emitindo o texto à medida que chega (para streaming na tela).

    `turn` é preenchido durante a execução com o texto completo, as consultas feitas
    e, ao final, a checagem dos números contra os resultados das consultas.
    """
    question = question.strip()[:MAX_QUESTION_CHARS]
    messages = [*history[-MAX_HISTORY_MESSAGES:], ChatMessage("user", question)]
    system = CHAT_SYSTEM.format(dataset_info=tools.dataset_info())

    events: Iterator[ChatEvent] = provider.chat_stream(system, messages, tools.specs(), tools.run)
    for event in events:
        if isinstance(event, ToolCallEvent):
            turn.tool_calls.append(event)
        elif isinstance(event, TextDelta):
            turn.text += event.text
            yield event.text

    facts = "\n".join(json.dumps(call.result, ensure_ascii=False) for call in turn.tool_calls)
    turn.verification = verify(turn.text, facts)
