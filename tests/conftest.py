"""
Configuração compartilhada dos testes.

Os testes nunca usam chaves reais nem acessam a internet: as
configurações do `.env`/`secrets.toml` são ignoradas e qualquer
requisição HTTP não simulada falha imediatamente.
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from insight_engine import config
from insight_engine.data.sales import load_sales_data

SETTINGS = [
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODEL",
    "COINGECKO_API_KEY",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Ignora chaves do ambiente local (.env e st.secrets)."""
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_from_streamlit_secrets", lambda name: None)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Bloqueia requisições HTTP reais; cada teste simula o que precisar."""

    def blocked(*args, **kwargs):
        raise RuntimeError("Acesso à rede bloqueado nos testes")

    monkeypatch.setattr(requests, "get", blocked)


@pytest.fixture(scope="session")
def sales_df() -> pd.DataFrame:
    """Base sintética completa (gerada uma vez por sessão de testes)."""
    return load_sales_data()


@pytest.fixture
def small_sales_df() -> pd.DataFrame:
    """Base mínima com valores fáceis de conferir à mão."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"]),
            "category": ["Eletrônicos", "Moda", "Eletrônicos", "Beleza"],
            "region": ["Sudeste", "Sul", "Sudeste", "Norte"],
            "product": ["Notebook Pro", "Jaqueta", "Fone Bluetooth", "Perfume"],
            "units": [1, 2, 3, 1],
            "unit_price": [500.0, 100.0, 100.0, 100.0],
            "revenue": [500.0, 200.0, 300.0, 100.0],
            "cost": [300.0, 150.0, 200.0, 50.0],
            "profit": [200.0, 50.0, 100.0, 50.0],
        }
    )
