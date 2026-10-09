import pytest

from insight_engine.data.schemas import SALES_SCHEMA, DataValidationError, validate


def test_base_valida_passa(small_sales_df):
    validated = validate(small_sales_df, SALES_SCHEMA, source="vendas")
    # customer_id e order_id são opcionais: a base mínima não tem essas colunas
    assert list(validated.columns) == [c for c in SALES_SCHEMA.columns if c not in ("customer_id", "order_id")]


def test_custo_e_cliente_podem_faltar(small_sales_df):
    df = small_sales_df.assign(cost=float("nan"), profit=float("nan"), customer_id=None)
    validated = validate(df, SALES_SCHEMA, source="vendas")
    assert validated["cost"].isna().all()


def test_colunas_extras_sao_descartadas(small_sales_df):
    df = small_sales_df.assign(coluna_extra=1)
    assert "coluna_extra" not in validate(df, SALES_SCHEMA, source="vendas").columns


def test_erros_sao_reunidos_em_uma_mensagem(small_sales_df):
    df = small_sales_df.drop(columns=["region"])
    df.loc[0, "revenue"] = -10

    with pytest.raises(DataValidationError) as exc:
        validate(df, SALES_SCHEMA, source="vendas")

    message = str(exc.value)
    assert "coluna ausente: region" in message
    assert "revenue" in message
