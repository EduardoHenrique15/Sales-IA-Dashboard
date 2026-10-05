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
    return [re.search(r'kpi-value">(.*?)<', card).group(1) for card in cards]


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=TIMEOUT)


def test_pagina_de_vendas(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "📊 Dashboard Executivo de Vendas"
    assert kpi_values(app)[0] == "R$ 2.838.404,99"  # receita dos últimos 90 dias da base
    assert len(app.get("plotly_chart")) == 4


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
