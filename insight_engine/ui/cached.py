"""
Carregamentos e análises com cache do Streamlit.

Evita regerar a base de vendas ou refazer modelos a cada
interação. Os resultados dependem só dos argumentos, então o cache é seguro.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from insight_engine.analytics import anomalies as anomalies_mod
from insight_engine.analytics.customers import CustomerSegmentation, segment_customers
from insight_engine.analytics.forecasting import ForecastResult, forecast_revenue
from insight_engine.data.olist import load_olist_sales
from insight_engine.data.sales import load_sales_data


@st.cache_data(show_spinner="Carregando base de vendas...")
def sales_data() -> pd.DataFrame:
    return load_sales_data()


@st.cache_data(show_spinner="Carregando a base da Olist...")
def olist_data() -> pd.DataFrame:
    return load_olist_sales()


@st.cache_data(show_spinner="Treinando e validando os modelos de previsão...", max_entries=32)
def forecast(series: pd.Series, horizon: int) -> ForecastResult:
    return forecast_revenue(series, horizon)


@st.cache_data(show_spinner="Procurando anomalias...", max_entries=32)
def anomalies(series: pd.Series) -> pd.DataFrame:
    return anomalies_mod.detect_anomalies(series)


@st.cache_data(show_spinner=False, max_entries=32)
def decomposition(series: pd.Series) -> anomalies_mod.Decomposition:
    return anomalies_mod.decompose(series)


@st.cache_data(show_spinner="Segmentando clientes...", max_entries=16)
def segmentation(df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> CustomerSegmentation:
    return segment_customers(df, reference_date=reference_date)
