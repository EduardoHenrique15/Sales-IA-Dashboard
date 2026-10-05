"""
Esquemas de validação dos DataFrames (pandera).

Toda fonte de dados passa por aqui antes de chegar aos cálculos:
colunas obrigatórias, tipos e regras de negócio (ex.: receita não
negativa) são conferidos de uma vez, e os erros são reunidos em uma
única mensagem legível.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

SALES_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False),
        "category": pa.Column(str, nullable=False),
        "region": pa.Column(str, nullable=False),
        "product": pa.Column(str, nullable=False),
        "units": pa.Column(int, pa.Check.ge(0)),
        "unit_price": pa.Column(float, pa.Check.ge(0)),
        "revenue": pa.Column(float, pa.Check.ge(0)),
        # custo/lucro podem faltar em bases enviadas pelo usuário
        "cost": pa.Column(float, pa.Check.ge(0), nullable=True),
        "profit": pa.Column(float, nullable=True),
        # opcional: nem toda base identifica o cliente
        "customer_id": pa.Column(str, nullable=True, required=False),
    },
    coerce=True,
    strict="filter",  # colunas extras são descartadas
)

CRYPTO_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]", nullable=False, unique=True),
        "price": pa.Column(float, pa.Check.gt(0)),
        "volume": pa.Column(float, pa.Check.ge(0)),
    },
    coerce=True,
    strict="filter",
)


class DataValidationError(ValueError):
    """Os dados não respeitam o esquema esperado."""


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, source: str) -> pd.DataFrame:
    """Valida `df` contra `schema` e devolve o DataFrame com os tipos corrigidos.

    Levanta `DataValidationError` listando todas as falhas encontradas.
    """
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(f"Dados de {source} inválidos ({_summarize(exc)})") from exc


def _summarize(exc: SchemaErrors) -> str:
    """Resume as falhas do pandera em uma linha, uma vez por coluna/regra."""
    messages = []
    for row in exc.failure_cases.itertuples():
        if row.check == "column_in_dataframe":
            messages.append(f"coluna ausente: {row.failure_case}")
        else:
            messages.append(f"{row.column}: {row.check}")
    return "; ".join(dict.fromkeys(messages))
