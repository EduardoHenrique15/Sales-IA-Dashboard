"""
Testes de interface: executam o app de verdade com o AppTest do Streamlit.
"""

import re
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from insight_engine.data.crypto import CryptoDataError, _parse_market_chart
from tests.helpers import market_chart

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 60


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def kpi_values(at: AppTest) -> list[str]:
    cards = [m.value for m in at.markdown if 'class="kpi-value"' in m.value]
    return [re.search(r'kpi-value"[^>]*>(.*?)<', card).group(1) for card in cards]


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=TIMEOUT)


def test_pagina_de_vendas(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "📊 Dashboard Executivo de Vendas"
    assert kpi_values(app)[0] == "R$ 2.838.404,99"  # receita dos últimos 90 dias da base
    assert len(app.get("plotly_chart")) == 6  # 4 da visão geral + 2 da análise de variação
    assert any("Por que a receita mudou" in h.value for h in app.subheader)


def test_filtro_de_categoria_atualiza_os_cards(app):
    app.run()
    app.sidebar.multiselect[0].set_value(["Moda"]).run()

    assert not app.exception
    assert kpi_values(app)[-1] == "Moda"  # categoria líder


def test_relatorio_local_sem_chave(app):
    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert any("Motor estatístico local" in m.value for m in app.markdown)
    assert any("## Destaques do Período" in m.value for m in app.markdown)


def test_pagina_de_cripto(app):
    df = _parse_market_chart(market_chart([100.0, 105.0, 110.0]))
    app.run()
    with mock.patch("insight_engine.ui.cached.load_crypto_data", return_value=df):
        app.switch_page("app_pages/crypto.py").run()

    assert not app.exception
    assert app.title[0].value == "🪙 Dashboard Executivo — Bitcoin (BTC)"
    assert kpi_values(app)[0] == "US$ 110,00"


def test_falha_da_api_de_cripto_mostra_erro_sem_quebrar(app):
    app.run()
    with mock.patch("insight_engine.ui.cached.load_crypto_data", side_effect=CryptoDataError("rate_limit")):
        app.switch_page("app_pages/crypto.py").run()

    assert not app.exception
    assert "limite de requisições" in app.error[0].value


def test_pagina_de_previsao(app):
    app.run()
    app.switch_page("app_pages/forecast.py").run()

    assert not app.exception
    assert [h.value for h in app.header] == ["Previsão de receita", "Anomalias", "Padrões sazonais"]
    assert app.metric[0].value.startswith("R$")
    assert any("encontrou **3 de 3**" in i.value for i in app.info)


def test_pagina_de_clientes(app):
    app.run()
    app.switch_page("app_pages/customers.py").run()

    assert not app.exception
    assert [h.value for h in app.header] == ["Segmentos RFM", "Grupos encontrados pelo K-Means"]
    assert app.metric[0].label == "Clientes"


def test_pagina_de_importacao_sem_arquivo(app):
    app.run()
    app.switch_page("app_pages/upload.py").run()
    assert not app.exception
    assert app.title[0].value == "📤 Importar dados de vendas"


def test_base_enviada_sem_custo_nem_cliente(app, small_sales_df):
    """Com uma base enviada, as páginas usam essa base e se adaptam ao que ela tem."""
    from insight_engine.data.upload import SalesDataset
    from insight_engine.ui.datasets import CHOICE_STATE, UPLOAD_STATE, UPLOADED

    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    app.session_state[UPLOAD_STATE] = SalesDataset(df=df, name="minha_base.csv", has_cost=False, has_customers=False)
    app.session_state[CHOICE_STATE] = UPLOADED
    app.run()

    assert not app.exception
    assert "minha_base.csv" in app.caption[0].value
    assert app.sidebar.radio[0].value == UPLOADED
    assert "—" in kpi_values(app)[1]  # lucro indisponível

    app.switch_page("app_pages/customers.py").run()
    assert not app.exception
    assert "não tem coluna de cliente" in app.info[0].value


def test_relatorio_com_llm_mostra_a_checagem_de_numeros(app):
    from insight_engine.ai.report import ExecutiveReport
    from tests.helpers import FakeProvider

    report = ExecutiveReport(
        headline="Receita de R$ 2.838.404,99.", highlights=["Meta de R$ 9 milhões."], risks=[], actions=[]
    )
    with mock.patch("insight_engine.ui.ai_access.create_provider", return_value=(FakeProvider(report), None)):
        app.run()
        app.button[0].click().run()

    assert not app.exception
    assert any("Gerado por Fake" in m.value for m in app.markdown)
    assert any("não aparecem nos dados" in w.value and "9 milhões" in w.value for w in app.warning)


def test_chat_sem_chave_explica_o_que_falta(app):
    app.run()
    app.switch_page("app_pages/chat.py").run()
    assert not app.exception
    assert "precisa da IA generativa" in app.info[0].value


def test_chat_com_provedor(app):
    from tests.helpers import FakeProvider

    provider = FakeProvider(
        script=[
            ("tool", "kpis", {"data_inicio": "2025-10-02", "data_fim": "2025-12-31"}),
            ("text", "A receita foi de R$ 2.838.404,99."),
        ]
    )
    with mock.patch("insight_engine.ui.ai_access.create_provider", return_value=(provider, None)):
        app.run()
        app.switch_page("app_pages/chat.py").run()
        app.chat_input[0].set_value("Qual a receita do último trimestre?").run()

    assert not app.exception
    assert any("2.838.404,99" in m.value for m in app.chat_message[1].markdown)
    assert any("Consultas feitas (1)" in e.label for e in app.expander)
    assert any("confere com as consultas" in c.value for c in app.caption)
