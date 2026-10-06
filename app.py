"""
Ponto de entrada do Insight Engine.

Este arquivo só configura a aplicação (página, logs, tema, barra
lateral comum) e monta a navegação. Cada página fica em `app_pages/`,
e toda a lógica vive no pacote `insight_engine/`.

Rodar com:
    streamlit run app.py
"""

import streamlit as st

from insight_engine.config import get_gemini_api_key
from insight_engine.logging_setup import setup_logging
from insight_engine.ui import branding
from insight_engine.ui.components import GEMINI_KEY_STATE, sidebar_footer
from insight_engine.ui.layout import reserve_sidebar_filters
from insight_engine.ui.theme import inject_css

WELCOME_STATE = "welcome_seen"

st.set_page_config(
    page_title=f"{branding.APP_NAME} | {branding.TAGLINE}",
    page_icon=branding.LOGO_ICON,
    layout="wide",
    # aberta no computador, recolhida no celular (onde cobriria a página)
    initial_sidebar_state="auto",
)
setup_logging()
inject_css()
st.logo(branding.LOGO, icon_image=branding.LOGO_ICON, size="large")

page = st.navigation(
    {
        "Análises": [
            st.Page("app_pages/sales.py", title="Visão geral", icon=":material/dashboard:", default=True),
            st.Page(
                "app_pages/forecast.py",
                title="Previsão e anomalias",
                icon=":material/trending_up:",
                url_path="previsao",
            ),
            st.Page("app_pages/customers.py", title="Clientes", icon=":material/groups:", url_path="clientes"),
        ],
        "Inteligência artificial": [
            st.Page("app_pages/chat.py", title="Converse com os dados", icon=":material/forum:", url_path="chat"),
        ],
        "Dados e projeto": [
            st.Page("app_pages/upload.py", title="Importar dados", icon=":material/upload_file:", url_path="importar"),
            st.Page("app_pages/about.py", title="Sobre o projeto", icon=":material/info:", url_path="sobre"),
        ],
    }
)


@st.dialog("Bem-vindo ao Insight Engine", icon=":material/waving_hand:", width="medium")
def welcome() -> None:
    st.markdown(
        "Um dashboard executivo de vendas que junta **análise de dados**, **ciência de dados** e **IA "
        "generativa**:\n\n"
        "- :material/dashboard: **Visão geral** com KPIs, variação explicada em volume, preço e mix e metas;\n"
        "- :material/trending_up: **Previsão** escolhida por backtesting e **detecção de anomalias**;\n"
        "- :material/groups: **Segmentação de clientes** com RFM e K-Means;\n"
        "- :material/auto_awesome: **Relatório executivo** e **chat** com IA, com os números conferidos.\n\n"
        "Os dados iniciais são **sintéticos**. Use **Importar dados** para analisar a sua própria planilha."
    )
    if st.button("Explorar o dashboard", type="primary", icon=":material/arrow_forward:", key="welcome_start"):
        st.rerun()


if WELCOME_STATE not in st.session_state:
    st.session_state[WELCOME_STATE] = True
    welcome()

# ---------- BARRA LATERAL COMUM A TODAS AS PÁGINAS ----------
# topo: filtros da página atual (preenchidos pela própria página)
reserve_sidebar_filters()

# depois: configuração da IA e rodapé
with st.sidebar.expander("Inteligência artificial", icon=":material/auto_awesome:"):
    if st.session_state.get(GEMINI_KEY_STATE):
        st.caption(":material/check_circle: Usando a sua chave do Gemini.")
    elif get_gemini_api_key():
        st.caption(":material/check_circle: Gemini ativo com a chave do projeto (com limite de uso).")
    else:
        st.caption(":material/info: Sem chave: o relatório usa o motor estatístico local e o chat fica indisponível.")
    st.text_input(
        "Sua chave do Gemini (opcional)",
        type="password",
        key=GEMINI_KEY_STATE,
        placeholder="Cole a chave aqui",
        help="Gratuita em aistudio.google.com. Fica só nesta sessão do navegador e não é salva.",
    )
sidebar_footer()

page.run()
