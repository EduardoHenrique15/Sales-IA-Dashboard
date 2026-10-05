"""
Fonte de CRIPTOMOEDAS: dados reais via API pública da CoinGecko.

Funciona sem chave; com a chave Demo gratuita (`COINGECKO_API_KEY`),
o limite de requisições é maior.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from insight_engine.config import get_coingecko_api_key
from insight_engine.data.schemas import CRYPTO_SCHEMA, DataValidationError, validate

logger = logging.getLogger(__name__)

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
    """Busca o histórico diário de preço/volume de uma criptomoeda.

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
                logger.warning("CoinGecko: rate limit (tentativa %d/%d)", attempt + 1, max_retries)
                time.sleep(2 ** attempt)  # backoff exponencial: 1s, 2s, 4s...
                continue

            resp.raise_for_status()
            df = _parse_market_chart(resp.json())
            logger.info("CoinGecko: %s, %d dias carregados", coin_id, len(df))
            return df

        except requests.exceptions.RequestException as exc:
            last_error = CryptoDataError("network_error", str(exc))
            logger.warning("CoinGecko: falha de rede (tentativa %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(1)
        except ValueError as exc:  # JSON malformado
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

    try:
        df = validate(df, CRYPTO_SCHEMA, source="cotações")
    except DataValidationError as exc:
        raise CryptoDataError("invalid_response", str(exc)) from exc
    return df[["date", "price", "volume"]].reset_index(drop=True)
