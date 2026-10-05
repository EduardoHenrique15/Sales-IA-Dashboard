"""
Carregamento de dados com cache do Streamlit.

Evita regerar a base de vendas ou chamar a API a cada interação.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from insight_engine.data.crypto import load_crypto_data
from insight_engine.data.sales import load_sales_data


@st.cache_data(show_spinner="Carregando base de vendas...")
def sales_data() -> pd.DataFrame:
    return load_sales_data()


# Só respostas bem-sucedidas entram no cache: em caso de falha,
# `load_crypto_data` levanta exceção, e o Streamlit não guarda exceções.
@st.cache_data(show_spinner="Consumindo API de cotações em tempo real...", ttl=600)
def crypto_data(coin_name: str, days: int) -> pd.DataFrame:
    return load_crypto_data(coin_name, days)
