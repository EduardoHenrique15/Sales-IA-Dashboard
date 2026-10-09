"""
Importação de bases de vendas enviadas pelo usuário (CSV ou Excel).

Fluxo:
  1. `read_table` lê o arquivo (detecta separador e codificação do CSV);
  2. `guess_mapping` sugere qual coluna corresponde a cada campo;
  3. `build_sales_dataset` converte as colunas escolhidas para o formato
     interno, trata números no padrão brasileiro e valida o resultado.

Só `data` e `receita` são obrigatórias. Sem custo, lucro e margem ficam
indisponíveis; sem cliente, a análise de clientes fica desativada.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from insight_engine.data.schemas import SALES_SCHEMA, validate

NOT_INFORMED = "Não informado"

# campo interno -> rótulo exibido na interface
REQUIRED_FIELDS = {"date": "Data do pedido", "revenue": "Receita (valor do pedido)"}
OPTIONAL_FIELDS = {
    "category": "Categoria",
    "region": "Região",
    "product": "Produto",
    "units": "Quantidade",
    "cost": "Custo",
    "customer_id": "Cliente (ID)",
}

# nomes de coluna comuns para cada campo (sem acento, minúsculos)
_ALIASES = {
    "date": ["data", "date", "dia", "data_pedido", "data_venda", "dt", "order_date"],
    "revenue": ["receita", "faturamento", "valor", "valor_total", "total", "venda", "vendas", "revenue", "amount"],
    "category": ["categoria", "category", "segmento", "linha"],
    "region": ["regiao", "region", "estado", "uf", "cidade", "filial", "loja"],
    "product": ["produto", "product", "item", "sku", "descricao"],
    "units": ["quantidade", "qtd", "qtde", "unidades", "units", "quantity"],
    "cost": ["custo", "cost", "custo_total", "cmv"],
    "customer_id": ["cliente", "customer", "customer_id", "id_cliente", "cod_cliente", "cpf", "email"],
}


class UploadError(ValueError):
    """O arquivo enviado não pôde ser lido ou convertido."""


@dataclass(frozen=True)
class SalesDataset:
    """Base de vendas pronta para análise, com o que ela permite calcular."""

    df: pd.DataFrame
    name: str
    has_cost: bool = True
    has_customers: bool = True
    # avisos sobre linhas descartadas, campos ausentes etc.
    notes: list[str] = field(default_factory=list)


def read_table(file_name: str, content: bytes) -> pd.DataFrame:
    """Lê um CSV (qualquer separador, UTF-8 ou Latin-1) ou uma planilha Excel."""
    name = file_name.lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        if name.endswith((".csv", ".txt")):
            return _read_csv(content)
    except UploadError:
        raise
    except Exception as exc:  # noqa: BLE001 - qualquer falha de leitura vira mensagem amigável
        raise UploadError(f"Não foi possível ler o arquivo: {exc}") from exc
    raise UploadError("Formato não suportado. Envie um arquivo .csv ou .xlsx.")


def guess_mapping(columns: list[str]) -> dict[str, str | None]:
    """Sugere a coluna de cada campo comparando os nomes normalizados."""
    normalized = {col: _normalize(col) for col in columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for field_name in [*REQUIRED_FIELDS, *OPTIONAL_FIELDS]:
        aliases = _ALIASES[field_name]
        match = next((c for c, n in normalized.items() if n in aliases and c not in used), None)
        if match is None:  # nome que contém o apelido (ex.: "valor_liquido")
            match = next(
                (c for c, n in normalized.items() if c not in used and any(a in n.split("_") for a in aliases)),
                None,
            )
        mapping[field_name] = match
        if match:
            used.add(match)
    return mapping


def build_sales_dataset(raw: pd.DataFrame, mapping: dict[str, str | None], name: str) -> SalesDataset:
    """Converte a tabela enviada para o formato interno de vendas."""
    missing = [label for f, label in REQUIRED_FIELDS.items() if not mapping.get(f)]
    if missing:
        raise UploadError(f"Selecione a coluna de: {', '.join(missing)}.")

    def column(field_name: str) -> pd.Series | None:
        col = mapping.get(field_name)
        return raw[col] if col else None

    notes: list[str] = []
    df = pd.DataFrame(index=raw.index)
    df["date"] = parse_dates(column("date"))
    df["revenue"] = parse_numbers(column("revenue"))

    for text_field in ["category", "region", "product"]:
        values = column(text_field)
        df[text_field] = NOT_INFORMED if values is None else values.astype("string").str.strip().fillna(NOT_INFORMED)

    units = column("units")
    df["units"] = 1 if units is None else parse_numbers(units).fillna(1).round().astype(int)

    cost = column("cost")
    df["cost"] = np.nan if cost is None else parse_numbers(cost)

    customer = column("customer_id")
    df["customer_id"] = None if customer is None else customer.astype("string").str.strip()

    # linhas sem data ou receita válidas, e devoluções (receita negativa), são descartadas
    invalid = df["date"].isna() | df["revenue"].isna() | (df["revenue"] < 0)
    if invalid.any():
        notes.append(f"{int(invalid.sum())} linha(s) descartada(s) por data ou receita inválida/negativa.")
    df = df[~invalid]
    if df.empty:
        raise UploadError("Nenhuma linha válida: confira as colunas de data e receita escolhidas.")

    df["cost"] = df["cost"].where(df["cost"] >= 0)
    df["profit"] = df["revenue"] - df["cost"]
    df["unit_price"] = df["revenue"] / df["units"].where(df["units"] > 0, 1)

    has_cost = bool(df["cost"].notna().all())
    if cost is None:
        notes.append("Sem coluna de custo: lucro e margem ficam indisponíveis.")
    elif not has_cost:
        notes.append("Há pedidos sem custo válido: lucro e margem ficam indisponíveis.")
    has_customers = bool(df["customer_id"].notna().any())
    if not has_customers:
        notes.append("Sem coluna de cliente: a análise de clientes (RFM) fica desativada.")

    df = validate(df, SALES_SCHEMA, source="arquivo enviado")
    return SalesDataset(
        df=df.sort_values("date").reset_index(drop=True),
        name=name,
        has_cost=has_cost,
        has_customers=has_customers,
        notes=notes,
    )


def parse_numbers(values: pd.Series) -> pd.Series:
    """Converte textos como "R$ 1.234,56", "1,234.56" ou "12,5" em números."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype(float)
    text = values.astype("string").str.strip()
    # Um ponto isolado ("1.234") é milhar quando a coluna usa vírgula decimal ("12,50"),
    # tem "R$" ou quando todo ponto separa grupos de 3 dígitos ("R$ 1.500", "2.300").
    comma_decimal = bool(text.str.contains(r",\d+\)?\s*$", regex=True).any())
    english = bool(text.str.contains(r",\d{3}\.", regex=True).any())
    dotted = text[text.str.contains(".", regex=False).fillna(False)]
    dots_are_thousands = not dotted.empty and bool(
        dotted.str.fullmatch(r"[^\d.,]*-?\d{1,3}(?:\.\d{3})+[^\d.,]*", na=False).all()
    )
    has_brl = bool(text.str.contains("R$", regex=False).any())
    brazilian = not english and (comma_decimal or has_brl or dots_are_thousands)
    return values.map(lambda v: _parse_number(v, brazilian)).astype(float)


def parse_dates(values: pd.Series) -> pd.Series:
    """Converte datas no formato brasileiro (31/12/2025), ano primeiro (2025-12-31, 2025/12/31) ou de planilhas."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values).dt.tz_localize(None).dt.normalize()
    text = values.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # ano primeiro: com dayfirst, "2025/03/01" viraria 3 de janeiro
    parts = text.str.extract(r"^(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})(?:\D|$)")
    year_first = parts["year"].notna()
    if year_first.any():
        parsed[year_first] = pd.to_datetime(parts[year_first].astype(int), errors="coerce")

    # número serial do Excel (dias desde 30/12/1899), ex.: 45658 = 01/01/2025
    serial = pd.to_numeric(text.where(text.str.fullmatch(r"\d{5}(?:\.\d+)?", na=False)), errors="coerce")
    is_serial = serial.between(20_000, 80_000).fillna(False).astype(bool)  # de 1954 a 2119
    parsed[is_serial] = pd.Timestamp("1899-12-30") + pd.to_timedelta(serial[is_serial].astype(float), unit="D")

    rest = ~year_first & ~is_serial
    parsed[rest] = pd.to_datetime(text[rest], dayfirst=True, errors="coerce", format="mixed")
    return parsed.dt.normalize()


def sample_csv() -> bytes:
    """Planilha-modelo para o usuário baixar e preencher."""
    sample = pd.DataFrame(
        {
            "data": ["01/03/2025", "01/03/2025", "02/03/2025"],
            "cliente": ["C-001", "C-002", "C-001"],
            "categoria": ["Eletrônicos", "Moda", "Eletrônicos"],
            "regiao": ["Sudeste", "Sul", "Sudeste"],
            "produto": ["Fone Bluetooth", "Jaqueta", "Smart TV 50pol"],
            "quantidade": [2, 1, 1],
            "receita": ["398,00", "189,90", "2.799,00"],
            "custo": ["240,00", "110,00", "1.950,00"],
        }
    )
    return sample.to_csv(index=False, sep=";").encode("utf-8-sig")


def _read_csv(content: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            continue
        # sep=None deixa o pandas detectar ; , ou tab. Tudo é lido como texto para
        # preservar formatos como "1.234,56" até a conversão explícita.
        return pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str, skipinitialspace=True)
    raise UploadError("Não foi possível identificar a codificação do arquivo (use UTF-8).")


def _parse_number(value: object, brazilian: bool = False) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, int | float):
        return float(value)
    raw = str(value).strip()
    text = re.sub(r"[^\d,.\-]", "", raw)  # remove "R$", espaços etc.
    if not text:
        return np.nan
    if raw.startswith("(") and raw.endswith(")") and not text.startswith("-"):
        text = "-" + text  # negativo no formato contábil: "(1.234,56)"
    if "," in text and "." in text:
        # o último separador é o decimal: "1.234,56" (BR) ou "1,234.56" (EN)
        thousands = "." if text.rfind(",") > text.rfind(".") else ","
        text = text.replace(thousands, "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    elif text.count(".") > 1 or (brazilian and "." in text):  # "1.234.567" ou "1.234" (BR) = milhares
        text = text.replace(".", "")
    try:
        return float(text)
    except ValueError:
        return np.nan


def _normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
