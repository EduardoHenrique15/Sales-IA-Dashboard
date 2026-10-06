"""
Ponto de entrada do Insight Engine.

Este arquivo só configura a aplicação (página, logs, tema, barra
lateral comum) e monta a navegação. Cada página fica em `app_pages/`,
e toda a lógica vive no pacote `insight_engine/`.

Rodar com:
    streamlit run app.py
"""

import streamlit as st

from insight_engine.logging_setup import setup_logging
from insight_engine.ui.components import GEMINI_KEY_STATE, footer
from insight_engine.ui.theme import inject_css

st.set_page_config(
    page_title="Insight Engine | Dashboard Executivo com IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
setup_logging()
inject_css()

page = st.navigation(
    {
        "Vendas": [
            st.Page("app_pages/sales.py", title="Visão geral", icon="📦", default=True),
            st.Page("app_pages/forecast.py", title="Previsão e anomalias", icon="📈", url_path="previsao"),
            st.Page("app_pages/customers.py", title="Clientes", icon="👥", url_path="clientes"),
            st.Page("app_pages/chat.py", title="Converse com os dados", icon="💬", url_path="chat"),
            st.Page("app_pages/upload.py", title="Importar dados", icon="📤", url_path="importar"),
        ],
        "Mercado": [
            st.Page("app_pages/crypto.py", title="Criptomoedas", icon="🪙", url_path="criptomoedas"),
        ],
    }
)

# ---------- BARRA LATERAL COMUM A TODAS AS PÁGINAS ----------
st.sidebar.title("📊 Insight Engine")
st.sidebar.caption("Dashboard Executivo com IA")
st.sidebar.text_input(
    "Gemini API Key (opcional)",
    type="password",
    key=GEMINI_KEY_STATE,
    help="Se vazio, usa a chave configurada no servidor (com limite de uso). Sem nenhuma chave, o relatório "
    "é gerado pelo motor estatístico local, sem custo, e o chat fica indisponível.",
)
st.sidebar.divider()

page.run()
footer()
