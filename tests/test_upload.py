import io

import numpy as np
import pandas as pd
import pytest

from insight_engine.data.upload import (
    NOT_INFORMED,
    UploadError,
    build_sales_dataset,
    guess_mapping,
    parse_dates,
    parse_numbers,
    read_table,
    sample_csv,
)


class TestParseNumbers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("R$ 1.234,56", 1234.56),
            ("1,234.56", 1234.56),
            ("12,5", 12.5),
            ("1.234.567", 1234567.0),
            ("-3,2", -3.2),
            ("1234", 1234.0),
        ],
    )
    def test_formatos_comuns(self, text, expected):
        assert parse_numbers(pd.Series([text])).iloc[0] == pytest.approx(expected)

    def test_ponto_isolado_vira_milhar_em_coluna_com_virgula_decimal(self):
        assert parse_numbers(pd.Series(["1.234", "12,50"])).tolist() == [1234.0, 12.5]

    def test_ponto_isolado_e_decimal_em_coluna_no_padrao_americano(self):
        assert parse_numbers(pd.Series(["1.234", "12.5"])).tolist() == [1.234, 12.5]

    def test_texto_invalido_vira_vazio(self):
        assert np.isnan(parse_numbers(pd.Series(["abc"])).iloc[0])

    def test_coluna_ja_numerica(self):
        assert parse_numbers(pd.Series([10, 2.5])).tolist() == [10.0, 2.5]


def test_datas_brasileiras_e_iso():
    parsed = parse_dates(pd.Series(["31/12/2025", "2025-01-02", "data ruim"]))
    assert parsed.iloc[0] == pd.Timestamp("2025-12-31")
    assert parsed.iloc[1] == pd.Timestamp("2025-01-02")
    assert pd.isna(parsed.iloc[2])


def test_sugere_colunas_pelo_nome():
    columns = ["Data da Venda", "Cód. Cliente", "Estado", "Qtd", "Valor Total", "Observação"]
    mapping = guess_mapping(columns)
    assert mapping["date"] == "Data da Venda"
    assert mapping["revenue"] == "Valor Total"
    assert mapping["region"] == "Estado"
    assert mapping["units"] == "Qtd"
    assert mapping["customer_id"] == "Cód. Cliente"
    assert mapping["cost"] is None


class TestReadTable:
    def test_csv_com_ponto_e_virgula_e_bom(self):
        df = read_table("vendas.csv", sample_csv())
        assert list(df.columns)[0] == "data"
        assert len(df) == 3

    def test_csv_latin1(self):
        content = "data,receita,região\n01/01/2025,10,Sul\n".encode("latin-1")
        assert list(read_table("v.csv", content).columns) == ["data", "receita", "região"]

    def test_excel(self):
        buffer = io.BytesIO()
        pd.DataFrame({"data": pd.to_datetime(["2025-01-01"]), "receita": [10.0]}).to_excel(buffer, index=False)
        assert len(read_table("v.xlsx", buffer.getvalue())) == 1

    def test_formato_nao_suportado(self):
        with pytest.raises(UploadError, match="Formato não suportado"):
            read_table("v.pdf", b"x")


class TestBuildSalesDataset:
    def test_planilha_modelo_completa(self):
        raw = read_table("modelo.csv", sample_csv())
        dataset = build_sales_dataset(raw, guess_mapping(list(raw.columns)), "modelo.csv")

        df = dataset.df
        assert df["revenue"].tolist() == [398.0, 189.9, 2799.0]
        assert df["profit"].tolist() == pytest.approx([158.0, 79.9, 849.0])
        assert df["unit_price"].iloc[0] == 199.0  # 398 / 2 unidades
        assert dataset.has_cost and dataset.has_customers
        assert dataset.notes == []

    def test_so_data_e_receita(self):
        raw = pd.DataFrame({"Data": ["01/02/2025", "02/02/2025"], "Total": ["10,50", "20"]})
        dataset = build_sales_dataset(raw, {"date": "Data", "revenue": "Total"}, "x.csv")

        assert dataset.df["category"].eq(NOT_INFORMED).all()
        assert dataset.df["units"].eq(1).all()
        assert not dataset.has_cost
        assert not dataset.has_customers
        assert len(dataset.notes) == 2  # avisos de custo e de cliente ausentes

    def test_descarta_linhas_invalidas_e_devolucoes(self):
        raw = pd.DataFrame(
            {"data": ["01/01/2025", "x", "02/01/2025", "03/01/2025"], "receita": ["10", "5", "-3", "abc"]}
        )
        dataset = build_sales_dataset(raw, {"date": "data", "revenue": "receita"}, "x.csv")
        assert len(dataset.df) == 1
        assert "3 linha(s) descartada(s)" in dataset.notes[0]

    def test_exige_data_e_receita(self):
        raw = pd.DataFrame({"receita": ["10"]})
        with pytest.raises(UploadError, match="Data do pedido"):
            build_sales_dataset(raw, {"revenue": "receita"}, "x.csv")

    def test_nenhuma_linha_valida(self):
        raw = pd.DataFrame({"data": ["x"], "receita": ["10"]})
        with pytest.raises(UploadError, match="Nenhuma linha válida"):
            build_sales_dataset(raw, {"date": "data", "revenue": "receita"}, "x.csv")

    def test_mesmos_numeros_da_base_original(self, sales_df):
        """Exportar a base em formato brasileiro e reimportar preserva os centavos."""
        sample = sales_df.head(500)
        raw = pd.DataFrame(
            {
                "Data": sample["date"].dt.strftime("%d/%m/%Y"),
                "Valor": sample["revenue"].map(
                    lambda v: f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                ),
            }
        )
        dataset = build_sales_dataset(raw, guess_mapping(list(raw.columns)), "x.csv")
        assert dataset.df["revenue"].sum() == pytest.approx(sample["revenue"].sum())
