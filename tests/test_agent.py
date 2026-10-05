from unittest import mock

import pytest

from insight_engine.ai import agent, gemini
from insight_engine.analytics.kpis import compute_sales_kpis


@pytest.fixture
def kpis(small_sales_df):
    return compute_sales_kpis(small_sales_df)


def test_sem_chave_usa_o_relatorio_local_sem_aviso(small_sales_df, kpis):
    result = agent.generate_executive_summary(small_sales_df, kpis, "p")
    assert result.source == agent.SOURCE_FALLBACK
    assert result.fallback_reason is None
    assert "## Destaques do Período" in result.markdown


def test_com_chave_usa_o_gemini(monkeypatch, small_sales_df, kpis):
    generate = mock.Mock(return_value="## Texto do Gemini")
    monkeypatch.setattr(gemini, "generate", generate)
    monkeypatch.setenv("GEMINI_API_KEY", "chave-do-servidor")

    result = agent.generate_executive_summary(small_sales_df, kpis, "p")

    assert result.source == agent.SOURCE_GEMINI
    assert result.markdown == "## Texto do Gemini"
    assert generate.call_args.args[1] == "chave-do-servidor"
    assert "R$ 1.100,00" in generate.call_args.args[0]  # o prompt leva os KPIs


def test_chave_da_barra_lateral_tem_prioridade(monkeypatch, small_sales_df, kpis):
    generate = mock.Mock(return_value="ok")
    monkeypatch.setattr(gemini, "generate", generate)
    monkeypatch.setenv("GEMINI_API_KEY", "chave-do-servidor")

    agent.generate_executive_summary(small_sales_df, kpis, "p", api_key="chave-digitada")
    assert generate.call_args.args[1] == "chave-digitada"


def test_falha_do_gemini_cai_no_relatorio_local_com_motivo(monkeypatch, small_sales_df, kpis):
    monkeypatch.setattr(gemini, "generate", mock.Mock(side_effect=RuntimeError("falhou")))
    result = agent.generate_executive_summary(small_sales_df, kpis, "p", api_key="x")

    assert result.source == agent.SOURCE_FALLBACK
    assert "erro inesperado" in result.fallback_reason
    assert "## Destaques do Período" in result.markdown


def test_sem_o_pacote_do_gemini(monkeypatch, small_sales_df, kpis):
    monkeypatch.setattr(gemini, "GENAI_AVAILABLE", False)
    result = agent.generate_executive_summary(small_sales_df, kpis, "p", api_key="x")
    assert result.source == agent.SOURCE_FALLBACK
    assert "google-genai" in result.fallback_reason


def test_sem_dados_nao_chama_o_gemini(monkeypatch, small_sales_df, kpis):
    generate = mock.Mock()
    monkeypatch.setattr(gemini, "generate", generate)

    result = agent.generate_executive_summary(small_sales_df.iloc[0:0], kpis, "p", api_key="x")

    generate.assert_not_called()
    assert "Não há dados" in result.markdown
