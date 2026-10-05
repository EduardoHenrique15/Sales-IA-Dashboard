import pytest

from insight_engine.data.schemas import CRYPTO_SCHEMA, SALES_SCHEMA, DataValidationError, validate


def test_base_valida_passa(small_sales_df):
    validated = validate(small_sales_df, SALES_SCHEMA, source="vendas")
    assert list(validated.columns) == list(SALES_SCHEMA.columns)


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


def test_cripto_rejeita_preco_nao_positivo_e_datas_repetidas():
    import pandas as pd

    df = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-01"]), "price": [0.0, 10.0], "volume": [1.0, 1.0]}
    )
    with pytest.raises(DataValidationError) as exc:
        validate(df, CRYPTO_SCHEMA, source="cotações")
    assert "price" in str(exc.value)
    assert "date" in str(exc.value)
