"""Utilitários para montar dados de teste."""

from __future__ import annotations

from datetime import date

import pandas as pd

DAY_MS = 86_400_000


def market_chart(prices: list[float], volumes: list[float] | None = None, start_day: int = 20_000) -> dict:
    """Monta uma resposta no formato da CoinGecko `/market_chart`."""
    volumes = volumes if volumes is not None else [1_000.0] * len(prices)
    return {
        "prices": [[(start_day + i) * DAY_MS, p] for i, p in enumerate(prices)],
        "total_volumes": [[(start_day + i) * DAY_MS, v] for i, v in enumerate(volumes)],
    }


def as_date(text: str) -> date:
    return pd.Timestamp(text).date()
