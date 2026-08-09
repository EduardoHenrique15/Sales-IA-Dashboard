"""
data_loader.py
================================================================
Camada de acesso a dados do projeto.

Objetivo: isolar TODA a lógica de obtenção de dados (geração
sintética, leitura em cache, consumo de API externa) para que
`app.py` nunca precise saber "de onde" o dado vem — apenas
consome DataFrames já padronizados.

Duas fontes plugáveis:
  1. VENDAS   -> dataset sintético, realista, com sazonalidade,
                 tendência e ruído, cacheado em CSV.
  2. CRIPTO   -> dados reais, consumidos automaticamente via API
                 pública da CoinGecko (sem necessidade de API key).

Para adicionar uma nova fonte, basta criar uma função
`load_<fonte>_data()` que devolva um DataFrame com uma coluna de
data e retornar um dicionário de metadados padronizado.
================================================================
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SALES_CACHE_PATH = os.path.join(DATA_DIR, "sales_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CATEGORIES = ["Eletrônicos", "Moda", "Casa & Decoração", "Alimentos", "Beleza"]
REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]

PRODUCTS_BY_CATEGORY = {
    "Eletrônicos": ["Smartphone X", "Notebook Pro", "Fone Bluetooth", "Smart TV 50pol"],
    "Moda": ["Camiseta Básica", "Calça Jeans", "Tênis Casual", "Jaqueta"],
    "Casa & Decoração": ["Jogo de Panelas", "Luminária LED", "Sofá 2 Lugares", "Tapete"],
    "Alimentos": ["Cesta Orgânica", "Café Especial", "Barra de Cereal", "Azeite Extra"],
    "Beleza": ["Perfume", "Kit Skincare", "Batom Matte", "Shampoo Profissional"],
}

# Fator de sazonalidade / desempenho relativo de cada região (mercado)
REGION_WEIGHT = {
    "Sudeste": 1.35,
    "Sul": 1.05,
    "Nordeste": 0.85,
    "Centro-Oeste": 0.75,
    "Norte": 0.55,
}

# Uma categoria propositalmente "em queda" no fim do período, para o
# agente de IA ter algo relevante a diagnosticar como gargalo.
DECLINING_CATEGORY = "Moda"


# ==================================================================
# 1. FONTE: VENDAS (sintética, com cache em CSV)
# ==================================================================
def _generate_synthetic_sales(start_date: str = "2023-01-01",
                               end_date: str | None = None,
                               seed: int = 42) -> pd.DataFrame:
    """Gera uma base de vendas diária realista, com:
       - tendência de crescimento geral,
       - sazonalidade semanal e mensal,
       - uma categoria em declínio (gargalo proposital),
       - ruído aleatório controlado (reprodutível via seed).
    """
    rng = np.random.default_rng(seed)

    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    total_days = len(dates)

    for i, date in enumerate(dates):
        progress = i / max(total_days - 1, 1)  # 0 -> 1 ao longo do tempo

        # tendência de crescimento suave da empresa como um todo
        growth_factor = 1 + progress * 0.55

        # sazonalidade: fim de semana vende menos (B2C físico) e
        # dezembro tem pico (Black Friday / Natal)
        weekday_factor = 0.75 if date.weekday() >= 5 else 1.0
        month_factor = 1.6 if date.month == 12 else (1.25 if date.month == 11 else 1.0)

        # número de "pedidos" naquele dia
        n_orders = rng.poisson(lam=14 * growth_factor * weekday_factor * month_factor)

        for _ in range(n_orders):
            category = rng.choice(CATEGORIES)
            region = rng.choice(REGIONS, p=_region_probabilities())
            product = rng.choice(PRODUCTS_BY_CATEGORY[category])

            base_price = {
                "Eletrônicos": 1800, "Moda": 140, "Casa & Decoração": 320,
                "Alimentos": 60, "Beleza": 95,
            }[category]

            # categoria em declínio perde força nos últimos 25% do período
            decline_penalty = 1.0
            if category == DECLINING_CATEGORY and progress > 0.75:
                decline_penalty = 1 - (progress - 0.75) * 1.6  # cai forte

            unit_price = max(base_price * rng.normal(1, 0.12), base_price * 0.5)
            units = max(int(rng.normal(3, 1.5) * decline_penalty), 0) or 1
            cost = unit_price * rng.uniform(0.55, 0.72)

            revenue = unit_price * units * REGION_WEIGHT[region] * decline_penalty
            cost_total = cost * units

            rows.append({
                "date": date,
                "category": category,
                "region": region,
                "product": product,
                "units": units,
                "unit_price": round(unit_price, 2),
                "revenue": round(max(revenue, 0), 2),
                "cost": round(max(cost_total, 0), 2),
            })

    df = pd.DataFrame(rows)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _region_probabilities():
    weights = np.array([REGION_WEIGHT[r] for r in REGIONS])
    return weights / weights.sum()


def load_sales_data(force_refresh: bool = False) -> pd.DataFrame:
    """Carrega a base de vendas, usando cache local em CSV.
    Se o arquivo não existir (ou `force_refresh=True`), gera a base
    sintética novamente e salva em disco.
    """
    if os.path.exists(SALES_CACHE_PATH) and not force_refresh:
        df = pd.read_csv(SALES_CACHE_PATH, parse_dates=["date"])
    else:
        df = _generate_synthetic_sales()
        df.to_csv(SALES_CACHE_PATH, index=False)

    # tratamento defensivo de nulos (produção: dados nunca são perfeitos)
    df = df.dropna(subset=["date", "revenue"])
    numeric_cols = ["units", "unit_price", "revenue", "cost", "profit"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df.sort_values("date").reset_index(drop=True)


# ==================================================================
# 2. FONTE: CRIPTOMOEDAS (dados reais via API pública CoinGecko)
# ==================================================================
COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"

COIN_OPTIONS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Solana (SOL)": "solana",
    "BNB": "binancecoin",
}


def load_crypto_data(coin_name: str = "Bitcoin (BTC)", days: int = 180,
                      max_retries: int = 3) -> tuple[pd.DataFrame, str | None]:
    """Busca automaticamente o histórico de preço/volume/market cap de uma
    criptomoeda na API pública da CoinGecko.

    Retorna (DataFrame, erro). Se `erro` não for None, o DataFrame
    devolvido é o último fallback disponível (pode ser vazio).

    Trata explicitamente:
      - HTTP 429 (rate limit) com backoff exponencial,
      - timeouts / falhas de rede,
      - respostas malformadas / dados nulos.
    """
    coin_id = COIN_OPTIONS.get(coin_name, "bitcoin")
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    url = COINGECKO_URL.format(coin_id=coin_id)

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 429:
                wait = 2 ** attempt  # backoff exponencial: 1s, 2s, 4s...
                last_error = "rate_limit"
                time.sleep(wait)
                continue

            resp.raise_for_status()
            payload = resp.json()

            prices = payload.get("prices", [])
            volumes = payload.get("total_volumes", [])
            if not prices:
                return pd.DataFrame(), "empty_response"

            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()

            if volumes:
                vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                df["volume"] = vol_df["volume"]
            else:
                df["volume"] = np.nan

            df = df.dropna(subset=["price"])
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df = df.drop(columns=["timestamp"]).reset_index(drop=True)

            return df, None

        except requests.exceptions.RequestException as exc:
            last_error = f"network_error: {exc}"
            time.sleep(1)
            continue

    return pd.DataFrame(), last_error or "unknown_error"
