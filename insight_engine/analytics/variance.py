"""
Análise de variação da receita: efeitos Volume, Preço e Mix (PVM).

Responde "por que a receita mudou em relação ao período anterior?",
decompondo a diferença em três efeitos que somam exatamente a variação:

  Receita = Q × Σ_c (s_c × p_c)
    Q   = unidades totais
    s_c = participação da categoria c nas unidades
    p_c = preço médio da categoria c (receita / unidades)

  Volume = (Q1 − Q0) × Σ s0_c × p0_c      -> vendeu mais ou menos unidades
  Mix    = Q1 × Σ (s1_c − s0_c) × p0_c    -> venda migrou entre categorias
  Preço  = Q1 × Σ s1_c × (p1_c − p0_c)    -> preço médio de cada categoria mudou
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RevenueBridge:
    previous_revenue: float
    current_revenue: float
    volume_effect: float
    price_effect: float
    mix_effect: float
    # contribuição de cada categoria para cada efeito
    by_segment: pd.DataFrame

    @property
    def total_change(self) -> float:
        return self.current_revenue - self.previous_revenue


def revenue_bridge(current: pd.DataFrame, previous: pd.DataFrame, segment: str = "category") -> RevenueBridge:
    """Decompõe a variação de receita entre dois períodos por `segment`."""
    cur = _aggregate(current, segment)
    prev = _aggregate(previous, segment)
    if cur.empty or prev.empty:
        raise ValueError("Os dois períodos precisam ter vendas para a análise de variação.")
    table = cur.join(prev, how="outer", lsuffix="_1", rsuffix="_0").fillna({"revenue_1": 0.0, "revenue_0": 0.0})
    table[["units_1", "units_0"]] = table[["units_1", "units_0"]].fillna(0.0)

    q1, q0 = table["units_1"].sum(), table["units_0"].sum()
    s1, s0 = table["units_1"] / q1, table["units_0"] / q0
    p1 = table["revenue_1"] / table["units_1"]
    p0 = table["revenue_0"] / table["units_0"]
    # Categoria que só existe em um dos períodos: usa o preço do outro, de modo
    # que todo o efeito dela apareça como mix (e não como preço).
    p0 = p0.fillna(p1)
    p1 = p1.fillna(p0)

    by_segment = pd.DataFrame(
        {
            "volume": (q1 - q0) * s0 * p0,
            "mix": q1 * (s1 - s0) * p0,
            "price": q1 * s1 * (p1 - p0),
        }
    )
    by_segment["total"] = table["revenue_1"] - table["revenue_0"]

    return RevenueBridge(
        previous_revenue=float(table["revenue_0"].sum()),
        current_revenue=float(table["revenue_1"].sum()),
        volume_effect=float(by_segment["volume"].sum()),
        price_effect=float(by_segment["price"].sum()),
        mix_effect=float(by_segment["mix"].sum()),
        by_segment=by_segment.sort_values("total"),
    )


def _aggregate(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    # Venda sem quantidade informada (0) conta como 1 unidade, como no preço unitário da
    # importação: descartá-la tiraria a receita dela da ponte, que deixaria de somar a variação.
    units = df["units"].where(df["units"] > 0, (df["revenue"] > 0).astype(int))
    grouped = df.assign(units=units).groupby(segment).agg(revenue=("revenue", "sum"), units=("units", "sum"))
    return grouped[grouped["units"] > 0].astype(float)
