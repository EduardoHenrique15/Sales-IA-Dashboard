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
                 tendência e ruído. Gerado em memória a partir de
                 um período e seed fixos, então é sempre idêntico
                 (sem depender de arquivo em disco).
  2. CRIPTO   -> dados reais, consumidos automaticamente via API
                 pública da CoinGecko (chave Demo opcional).

Para adicionar uma nova fonte, basta criar uma função
`load_<fonte>_data()` que devolva um DataFrame com uma coluna de
data e retornar um dicionário de metadados padronizado.
================================================================
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from config import get_coingecko_api_key

# ----------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------
# Período fixo da base sintética: três anos completos, para que os
# números sejam reprodutíveis em qualquer máquina ou deploy.
SALES_START_DATE = "2023-01-01"
SALES_END_DATE = "2025-12-31"
SALES_SEED = 42

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
# 1. FONTE: VENDAS (sintética e determinística)
# ==================================================================
def _generate_synthetic_sales(start_date: str = SALES_START_DATE,
                               end_date: str = SALES_END_DATE,
                               seed: int = SALES_SEED) -> pd.DataFrame:
    """Gera uma base de vendas diária realista, com:
       - tendência de crescimento geral,
       - sazonalidade semanal e mensal,
       - uma categoria em declínio (gargalo proposital),
       - ruído aleatório controlado (reprodutível via seed).
    """
    rng = np.random.default_rng(seed)

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


def load_sales_data() -> pd.DataFrame:
    """Carrega a base de vendas sintética.

    A geração é determinística (período e seed fixos). O cache fica a
    cargo da camada de interface (`st.cache_data`).
    """
    df = _generate_synthetic_sales()

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


class CryptoDataError(Exception):
    """Falha ao obter dados da CoinGecko.

    `kind` identifica o tipo de falha ("rate_limit", "network_error",
    "empty_response" ou "invalid_response") para a interface exibir
    a mensagem adequada.
    """

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


def load_crypto_data(coin_name: str = "Bitcoin (BTC)", days: int = 180,
                      max_retries: int = 3) -> pd.DataFrame:
    """Busca automaticamente o histórico de preço/volume de uma
    criptomoeda na API pública da CoinGecko.

    Retorna um DataFrame com uma linha por dia (colunas `date`,
    `price`, `volume`). Em caso de falha, levanta `CryptoDataError`
    em vez de devolver um resultado vazio, para que falhas não sejam
    guardadas no cache da interface.

    Trata explicitamente:
      - HTTP 429 (rate limit) com backoff exponencial,
      - timeouts / falhas de rede,
      - respostas malformadas / dados nulos.
    """
    coin_id = COIN_OPTIONS.get(coin_name, "bitcoin")
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    url = COINGECKO_URL.format(coin_id=coin_id)

    headers = {}
    api_key = get_coingecko_api_key()
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    last_error = CryptoDataError("network_error", "nenhuma tentativa realizada")
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)

            if resp.status_code == 429:
                last_error = CryptoDataError("rate_limit")
                time.sleep(2 ** attempt)  # backoff exponencial: 1s, 2s, 4s...
                continue

            resp.raise_for_status()
            return _parse_market_chart(resp.json())

        except requests.exceptions.RequestException as exc:
            last_error = CryptoDataError("network_error", str(exc))
            time.sleep(1)
        except ValueError as exc:
            raise CryptoDataError("invalid_response", str(exc)) from exc

    raise last_error


def _parse_market_chart(payload: dict) -> pd.DataFrame:
    """Converte a resposta de `/market_chart` em um DataFrame diário.

    Preço e volume são unidos pelo timestamp (não pela posição na
    lista). A CoinGecko inclui um ponto extra com o horário atual,
    que cai no mesmo dia do último fechamento; nesse caso fica só o
    ponto mais recente do dia.
    """
    prices = payload.get("prices") or []
    if not prices:
        raise CryptoDataError("empty_response")

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    volumes = payload.get("total_volumes") or []
    vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
    df = df.merge(vol_df, on="timestamp", how="left")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["price"])
    if df.empty:
        raise CryptoDataError("empty_response")

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset="date", keep="last")
        .drop(columns=["timestamp"])
    )
    return df[["date", "price", "volume"]].reset_index(drop=True)
