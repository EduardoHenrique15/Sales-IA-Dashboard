"""
Fonte de VENDAS: base sintética realista e determinística.

Período e seed são fixos, então a base é sempre idêntica em qualquer
máquina ou deploy, sem depender de arquivo em disco.

Além de tendência e sazonalidade, a base traz:
  - clientes recorrentes, com entrada e saída ao longo do tempo
    (para a segmentação RFM);
  - algumas anomalias plantadas em datas conhecidas (para validar
    o detector de anomalias contra um gabarito).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from insight_engine.data.schemas import SALES_SCHEMA, validate

logger = logging.getLogger(__name__)

# Período fixo da base sintética: três anos completos.
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

BASE_PRICE = {
    "Eletrônicos": 1800,
    "Moda": 140,
    "Casa & Decoração": 320,
    "Alimentos": 60,
    "Beleza": 95,
}

# Fator de desempenho relativo de cada região (mercado)
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

# Clientes: sorteados com um gerador separado, para não alterar os
# pedidos já gerados pela seed principal.
CUSTOMER_SEED = SALES_SEED + 1
N_CUSTOMERS = 4000
REGION_CODE = {"Sudeste": "SE", "Sul": "S", "Nordeste": "NE", "Centro-Oeste": "CO", "Norte": "N"}

# Anomalias plantadas: (data, tipo, fator). "pico" multiplica o número de
# pedidos do dia (campanha); "queda" mantém só uma fração deles (incidente).
PLANTED_ANOMALIES = [
    ("2023-09-05", "queda", 0.15),  # instabilidade no checkout
    ("2024-03-15", "pico", 3.0),  # campanha relâmpago
    ("2024-08-07", "queda", 0.15),  # site fora do ar
]


def load_sales_data() -> pd.DataFrame:
    """Gera, limpa e valida a base de vendas sintética.

    O cache fica a cargo da camada de interface (`st.cache_data`).
    """
    df = _generate_synthetic_sales()
    df = _plant_anomalies(df)
    df = _assign_customers(df)

    # tratamento defensivo de nulos (produção: dados nunca são perfeitos)
    df = df.dropna(subset=["date", "revenue"])
    numeric_cols = ["units", "unit_price", "revenue", "cost", "profit"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    df = validate(df, SALES_SCHEMA, source="vendas")
    logger.info("Base de vendas carregada: %d pedidos", len(df))
    return df.sort_values("date").reset_index(drop=True)


def _generate_synthetic_sales(
    start_date: str = SALES_START_DATE, end_date: str = SALES_END_DATE, seed: int = SALES_SEED
) -> pd.DataFrame:
    """Gera uma base de vendas diária realista, com:
    - tendência de crescimento geral,
    - sazonalidade semanal e mensal,
    - uma categoria em declínio (gargalo proposital),
    - ruído aleatório controlado (reprodutível via seed).
    """
    rng = np.random.default_rng(seed)
    region_probs = _region_probabilities()

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
            region = rng.choice(REGIONS, p=region_probs)
            product = rng.choice(PRODUCTS_BY_CATEGORY[category])
            base_price = BASE_PRICE[category]

            # categoria em declínio perde força nos últimos 25% do período
            decline_penalty = 1.0
            if category == DECLINING_CATEGORY and progress > 0.75:
                decline_penalty = 1 - (progress - 0.75) * 1.6  # cai forte

            unit_price = max(base_price * rng.normal(1, 0.12), base_price * 0.5)
            units = max(int(rng.normal(3, 1.5) * decline_penalty), 0) or 1
            cost = unit_price * rng.uniform(0.55, 0.72)

            revenue = unit_price * units * REGION_WEIGHT[region] * decline_penalty
            cost_total = cost * units

            rows.append(
                {
                    "date": date,
                    "category": category,
                    "region": region,
                    "product": product,
                    "units": units,
                    "unit_price": round(unit_price, 2),
                    "revenue": round(max(revenue, 0), 2),
                    "cost": round(max(cost_total, 0), 2),
                }
            )

    df = pd.DataFrame(rows)
    df["profit"] = (df["revenue"] - df["cost"]).round(2)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _region_probabilities() -> np.ndarray:
    weights = np.array([REGION_WEIGHT[r] for r in REGIONS])
    return weights / weights.sum()


def _plant_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Insere as anomalias de `PLANTED_ANOMALIES` de forma determinística."""
    keep = pd.Series(True, index=df.index)
    extra_orders = []
    for day, kind, factor in PLANTED_ANOMALIES:
        rows = df["date"] == pd.Timestamp(day)
        if kind == "pico":
            extra_orders.extend([df[rows]] * (round(factor) - 1))
        else:
            position = df[rows].groupby("date").cumcount()
            keep[position.index] = position % round(1 / factor) == 0
    df = pd.concat([df[keep], *extra_orders])
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def _assign_customers(df: pd.DataFrame, seed: int = CUSTOMER_SEED) -> pd.DataFrame:
    """Atribui um cliente a cada pedido.

    Cada região tem sua carteira de clientes. Cada cliente tem uma data de
    entrada, um tempo de vida (depois do qual para de comprar) e uma
    propensão de compra com cauda longa: poucos clientes compram muito.
    """
    rng = np.random.default_rng(seed)
    region_probs = dict(zip(REGIONS, _region_probabilities(), strict=True))
    day_index = (df["date"] - df["date"].min()).dt.days.to_numpy()
    total_days = int(day_index.max()) + 1
    regions = df["region"].to_numpy()
    customer = np.empty(len(df), dtype=object)

    for region in REGIONS:
        n = int(N_CUSTOMERS * region_probs[region])
        ids = np.array([f"{REGION_CODE[region]}-{i:05d}" for i in range(n)])
        # 25% da carteira já existe no início; o restante chega ao longo do período
        start = np.where(rng.random(n) < 0.25, 0, rng.integers(0, total_days, n))
        end = start + rng.exponential(scale=500, size=n)
        propensity = rng.lognormal(mean=0.0, sigma=1.0, size=n)

        rows = np.flatnonzero(regions == region)
        rows = rows[np.argsort(day_index[rows], kind="stable")]
        days, first = np.unique(day_index[rows], return_index=True)
        for day, chunk in zip(days, np.split(rows, first[1:]), strict=True):
            active = (start <= day) & (end >= day)
            if not active.any():
                active = start <= day
            weights = propensity[active] / propensity[active].sum()
            customer[chunk] = rng.choice(ids[active], size=len(chunk), p=weights)

    return df.assign(customer_id=customer)
