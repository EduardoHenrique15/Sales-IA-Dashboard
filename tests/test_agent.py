import pytest

from insight_engine.ai import agent
from insight_engine.ai.context import build_sales_facts
from insight_engine.ai.limits import BoundedCache
from insight_engine.ai.report import ExecutiveReport
from insight_engine.analytics.kpis import compute_sales_kpis
from insight_engine.analytics.periods import SalesFilters, filter_sales
from tests.helpers import FakeProvider, as_date


@pytest.fixture(scope="module")
def facts(sales_df):
    filters = SalesFilters(as_date("2025-10-02"), as_date("2025-12-31"))
    period, previous = filter_sales(sales_df, filters)
    return build_sales_facts(sales_df, period, previous, compute_sales_kpis(period, previous), filters.period_label)


def llm_report(*highlights: str) -> ExecutiveReport:
    return ExecutiveReport(headline="Resumo", highlights=list(highlights), risks=[], actions=[])


def generate(facts, **kwargs):
    return agent.generate_executive_summary(facts, "Vendas", "p", **kwargs)


def test_sem_provedor_usa_o_motor_local_sem_aviso(facts):
    result = generate(facts)
    assert result.source == agent.SOURCE_FALLBACK
    assert result.fallback_reason is None
    assert result.verification is None
    assert "## Destaques do Período" in result.markdown


def test_com_provedor_usa_o_llm_e_confere_os_numeros(facts):
    provider = FakeProvider(llm_report("Receita de R$ 2.838.404,99, alta de 31%.", "Meta de R$ 5 milhões."))
    result = generate(facts, provider=provider)

    assert result.source == agent.SOURCE_LLM
    assert result.provider_name == "Fake"
    assert "R$ 2.838.404,99" in provider.prompts[0]  # o prompt leva os fatos
    assert result.verification.unverified == ["R$ 5 milhões"]


def test_falha_do_llm_cai_no_motor_local_com_motivo(facts):
    result = generate(facts, provider=FakeProvider(RuntimeError("caiu"), error_message="o serviço caiu."))
    assert result.source == agent.SOURCE_FALLBACK
    assert result.fallback_reason == "o serviço caiu."


def test_cache_evita_nova_chamada_para_os_mesmos_dados(facts):
    provider, cache = FakeProvider(llm_report("ok")), BoundedCache()
    first = generate(facts, provider=provider, cache=cache)
    second = generate(facts, provider=provider, cache=cache)

    assert provider.report_calls == 1
    assert not first.from_cache and second.from_cache


def test_limite_de_uso_atingido(facts):
    provider = FakeProvider(llm_report("ok"))
    result = generate(facts, provider=provider, allow_call=lambda: False, limit_message="limite atingido.")
    assert provider.report_calls == 0
    assert result.fallback_reason == "limite atingido."


def test_motivo_do_provedor_indisponivel(facts):
    result = generate(facts, provider_error="o pacote `google-genai` não está instalado.")
    assert "google-genai" in result.fallback_reason


def test_sem_dados_nao_chama_o_llm():
    provider = FakeProvider(llm_report("ok"))
    result = agent.generate_executive_summary(None, "Vendas", "jan/2025", provider=provider)
    assert provider.report_calls == 0
    assert "Não há dados" in result.markdown
