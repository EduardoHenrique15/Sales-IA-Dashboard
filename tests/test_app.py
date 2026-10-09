"""
Testes de interface: executam o app de verdade com o AppTest do Streamlit.
"""

from datetime import date
from pathlib import Path
from unittest import mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 60


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=TIMEOUT)


def main_captions(at: AppTest) -> list[str]:
    return [c.value for c in at.main.caption]


def test_pagina_de_vendas(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "Visão geral de vendas"
    assert app.metric[0].label == "Receita"
    assert app.metric[0].value == "R$ 2.790.052"  # receita dos últimos 90 dias da base
    assert app.metric[0].proto.delta == "+29,8%"
    assert [m.label for m in app.metric][2:] == ["Pedidos", "Ticket médio"]
    assert len(app.get("plotly_chart")) == 6  # evolução, composição (3) e variação (2)
    assert len(app.tabs) == 4


def test_boas_vindas_aparece_so_na_primeira_visita(app):
    app.run()
    assert [b for b in app.button if b.key == "welcome_start"]
    app.run()
    assert not [b for b in app.button if b.key == "welcome_start"]


def test_filtro_de_categoria_atualiza_os_cards(app):
    app.run()
    app.sidebar.multiselect[0].set_value(["Moda"]).run()

    assert not app.exception
    assert any(c.startswith("Líder: Moda") for c in main_captions(app))
    assert any("Moda" in m.value for m in app.main.markdown if "badge" in m.value)


def test_filtros_valem_em_todas_as_paginas(app):
    app.run()
    app.sidebar.multiselect[0].set_value(["Moda"]).run()
    app.switch_page("app_pages/forecast.py").run()

    assert not app.exception
    assert app.sidebar.multiselect[0].value == ["Moda"]


def test_atalho_de_periodo(app):
    app.run()
    app.sidebar.segmented_control[0].set_value("30d").run()

    assert not app.exception
    assert "02/12/2025 a 31/12/2025" in main_captions(app)[0]
    app.sidebar.segmented_control[0].set_value("custom").run()
    assert app.sidebar.date_input[0].value == (date(2025, 12, 2), date(2025, 12, 31))


def test_agrupar_por_mes(app):
    app.run()
    app.segmented_control(key="granularity").set_value("M").run()

    assert not app.exception
    assert any(m.value == "**Receita e lucro por mês**" for m in app.markdown)


def test_clique_em_barra_vira_filtro():
    def script():
        from types import SimpleNamespace

        import streamlit as st

        from insight_engine.ui import filters

        st.session_state["grafico"] = SimpleNamespace(selection=SimpleNamespace(points=[{"y": "Moda"}]))
        before = filters.chart_key("categorias")
        filters.select_segment("categorias", "abc", "grafico")
        st.write(f"{st.session_state['f_categorias_abc']} {before} {filters.chart_key('categorias')}")

    at = AppTest.from_function(script).run()
    assert at.markdown[0].value == "['Moda'] click_categorias_0 click_categorias_1"


def test_limpar_filtros(app):
    app.query_params["categorias"] = ["Moda"]
    app.run()
    app.button(key="clear_filters").click().run()

    assert not app.exception
    assert app.sidebar.multiselect[0].value == []


def test_relatorio_local_sem_chave(app):
    app.run()
    app.button(key="generate_report").click().run()

    assert not app.exception
    assert any("Motor estatístico local" in m.value for m in app.markdown)
    assert any("Destaques" in m.value for m in app.markdown)
    assert any("Prioridade" in m.value for m in app.markdown)


def test_pagina_de_previsao(app):
    app.run()
    app.switch_page("app_pages/forecast.py").run()

    assert not app.exception
    assert [h.value for h in app.header] == ["Previsão de receita", "Anomalias", "Padrões sazonais"]
    assert app.metric[0].value.startswith("R$")
    assert any("encontrou **3 de 3**" in i.value for i in app.info)

    app.segmented_control(key="horizon").set_value(60).run()
    assert not app.exception
    assert app.metric[0].label == "Receita prevista (60 dias)"


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
    assert app.title[0].value == "Importar dados de vendas"


def test_pagina_sobre(app):
    app.run()
    app.switch_page("app_pages/about.py").run()
    assert not app.exception
    assert app.title[0].value == "Sobre o projeto"
    assert app.get("graphviz_chart")


def test_base_enviada_sem_custo_nem_cliente(app, small_sales_df):
    """Com uma base enviada, as páginas usam essa base e se adaptam ao que ela tem."""
    from insight_engine.data.upload import SalesDataset
    from insight_engine.ui.datasets import CHOICE_STATE, UPLOAD_STATE, UPLOADED

    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"))
    app.session_state[UPLOAD_STATE] = SalesDataset(df=df, name="minha_base.csv", has_cost=False, has_customers=False)
    app.session_state[CHOICE_STATE] = UPLOADED
    app.run()

    assert not app.exception
    assert "minha_base.csv" in main_captions(app)[0]
    assert app.sidebar.radio[0].value == UPLOADED
    assert app.metric[1].value == "—"  # lucro indisponível

    app.switch_page("app_pages/customers.py").run()
    assert not app.exception
    assert "não tem coluna de cliente" in app.info[0].value


def test_base_real_da_olist(app):
    app.run()
    app.sidebar.radio[0].set_value("olist").run()

    assert not app.exception
    assert "Olist" in main_captions(app)[0]
    assert app.metric[1].value == "—"  # a Olist não informa custo
    assert any("Pedidos reais" in c.value for c in app.sidebar.caption)

    app.switch_page("app_pages/customers.py").run()
    assert not app.exception
    assert app.metric[0].value == "94.046"  # clientes únicos, não um por pedido


def test_relatorio_com_llm_mostra_a_checagem_de_numeros(app):
    from insight_engine.ai.report import ExecutiveReport
    from tests.helpers import FakeProvider

    report = ExecutiveReport(
        headline="Receita de R$ 2.790.051,94.", highlights=["Meta de R$ 9 milhões."], risks=[], actions=[]
    )
    with mock.patch("insight_engine.ui.ai_access.create_provider", return_value=(FakeProvider(report), None)):
        app.run()
        app.button(key="generate_report").click().run()

    assert not app.exception
    assert any("Gerado por Fake" in m.value for m in app.markdown)
    assert any("1 de 2 números conferidos" in m.value for m in app.markdown)
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
        assert app.pills[0].options  # sugestões de pergunta na conversa vazia
        app.chat_input[0].set_value("Qual a receita do último trimestre?").run()

    assert not app.exception
    assert any("2.838.404,99" in m.value for message in app.chat_message for m in message.markdown)
    assert any("Consultas feitas (1)" in e.label for e in app.expander)
    assert any("confere com as consultas" in c.value for c in app.caption)


def test_sugestao_do_chat_nao_dispara_de_novo_ao_trocar_de_base(app):
    from tests.helpers import FakeProvider

    provider = FakeProvider(script=[("text", "Resposta.")])
    with mock.patch("insight_engine.ui.ai_access.create_provider", return_value=(provider, None)):
        app.run()
        app.switch_page("app_pages/chat.py").run()
        app.pills[0].set_value(app.pills[0].options[0]).run()
        assert [m.name for m in app.chat_message] == ["assistant", "user", "assistant"]
        app.sidebar.radio[0].set_value("olist").run()

    assert not app.exception
    assert len(app.chat_message) == 1  # só a saudação: nenhuma pergunta foi feita na base nova
    assert any("2018" in option for option in app.pills[0].options)  # exemplos com os anos da Olist


def test_filtros_vem_da_url(app):
    app.query_params["de"] = "2024-07-01"
    app.query_params["ate"] = "2024-09-30"
    app.query_params["categorias"] = ["Moda"]
    app.query_params["meta"] = "200000"
    app.run()

    assert not app.exception
    assert "01/07/2024 a 30/09/2024" in main_captions(app)[0]
    assert app.sidebar.segmented_control[0].value == "custom"  # datas que não são um atalho
    assert any(c.startswith("Líder: Moda") for c in main_captions(app))
    assert "Meta:" in app.get("progress")[0].proto.text


def test_url_invalida_usa_o_padrao(app):
    app.query_params["de"] = "2099-01-01"
    app.query_params["ate"] = "data-ruim"
    app.query_params["categorias"] = ["Carros"]
    app.query_params["comparar"] = "xyz"
    app.query_params["meta"] = "inf"
    app.run()

    assert not app.exception
    assert "03/10/2025 a 31/12/2025" in main_captions(app)[0]
    assert app.metric[0].value == "R$ 2.790.052"
    assert not app.get("progress")  # meta infinita é ignorada


def test_url_com_categoria_repetida(app):
    app.query_params["categorias"] = ["Moda", "Moda"]
    app.run()

    assert not app.exception
    assert app.sidebar.multiselect[0].value == ["Moda"]


def test_periodo_sem_anterior_na_base_nao_e_comparado(app):
    app.query_params["de"] = "2023-01-01"
    app.query_params["ate"] = "2023-03-31"
    app.run()

    assert not app.exception
    assert "sem comparação" in main_captions(app)[0]
    assert not app.metric[0].proto.delta


def test_comparacao_com_o_ano_anterior(app):
    app.run()
    app.sidebar.segmented_control(key="f_compare").set_value("ano_anterior").run()

    assert not app.exception
    assert app.metric[0].proto.delta_description == "vs ano anterior"
    assert "mesmo período do ano anterior" in main_captions(app)[0]
    table = app.dataframe[0].value
    assert list(table["Indicador"]) == ["Receita", "Lucro", "Margem", "Pedidos", "Unidades", "Ticket médio"]


def test_relatorio_oferece_pdf(app):
    app.run()
    app.button(key="generate_report").click().run()
    assert not app.exception
    labels = [b.proto.label for b in app.get("download_button")]
    assert "Baixar PDF" in labels and "Baixar Markdown" in labels
