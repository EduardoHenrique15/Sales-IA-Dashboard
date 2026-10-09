"""
Base real: e-commerce brasileiro da Olist (Kaggle, licença CC BY-NC-SA 4.0).

O dataset público tem ~100 mil pedidos de 2016 a 2018 em várias tabelas.
Este módulo faz duas coisas:

  - `prepare_*`: transforma os CSVs originais nas duas tabelas compactas
    guardadas no repositório (`data/olist/`). Roda uma vez, pelo script
    `scripts/prepare_olist.py`; os CSVs originais não vão para o git.
  - `load_olist_sales`: carrega a tabela de vendas no formato do app.

Decisões de preparação (detalhadas no notebook `notebooks/analise_olist.ipynb`):

  - Janela de 01/01/2017 a 21/08/2018. Antes disso a base tem só ~330 pedidos e
    um buraco de dois meses; depois de 21/08/2018 os pedidos caem para quase
    zero porque a coleta terminou, não porque as vendas caíram.
  - Pedidos cancelados ou indisponíveis ficam de fora.
  - Receita = preço dos itens (o frete não é receita do vendedor).
  - Cliente = `customer_unique_id`. O `customer_id` da Olist muda a cada pedido:
    usá-lo faria todo cliente parecer novo. Os identificadores (32 caracteres)
    viram códigos curtos ("C00001"), que ocupam bem menos espaço.
  - 73 categorias viram 14 grupos (gráficos e filtros legíveis); a categoria
    original vira o "produto" (os produtos da Olist são só códigos).
  - Estados viram as 5 regiões do IBGE.
  - Uma linha por pedido e grupo de categoria (só 0,5% dos pedidos têm itens
    de grupos diferentes). A coluna `order_id` guarda o código do pedido para
    que pedidos e frequência de compra contem pedidos, e não linhas.
  - A Olist não informa custo: lucro e margem ficam indisponíveis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from insight_engine.data.schemas import SALES_SCHEMA, validate

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "olist"
SALES_FILE = DATA_DIR / "vendas.parquet"
ORDERS_FILE = DATA_DIR / "pedidos.parquet"
# pedidos por dia na base original, antes de qualquer corte (mostra por que a janela foi escolhida)
RAW_DAILY_FILE = DATA_DIR / "pedidos_por_dia_original.csv"
DATASET_NAME = "Olist: e-commerce brasileiro (dados reais)"

START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-08-21 23:59:59")
EXCLUDED_STATUS = {"canceled", "unavailable"}
OTHERS = "Outros"

REGIONS = {
    **dict.fromkeys(["AC", "AP", "AM", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

# categoria original -> (grupo, nome legível)
CATEGORIES: dict[str, tuple[str, str]] = {
    "beleza_saude": ("Beleza e saúde", "Beleza e saúde"),
    "perfumaria": ("Beleza e saúde", "Perfumaria"),
    "fraldas_higiene": ("Beleza e saúde", "Fraldas e higiene"),
    "cama_mesa_banho": ("Casa e decoração", "Cama, mesa e banho"),
    "moveis_decoracao": ("Casa e decoração", "Móveis e decoração"),
    "utilidades_domesticas": ("Casa e decoração", "Utilidades domésticas"),
    "casa_conforto": ("Casa e decoração", "Casa e conforto"),
    "casa_conforto_2": ("Casa e decoração", "Casa e conforto 2"),
    "moveis_sala": ("Casa e decoração", "Móveis de sala"),
    "moveis_quarto": ("Casa e decoração", "Móveis de quarto"),
    "moveis_cozinha_area_de_servico_jantar_e_jardim": ("Casa e decoração", "Móveis de cozinha e jardim"),
    "moveis_colchao_e_estofado": ("Casa e decoração", "Colchões e estofados"),
    "moveis_escritorio": ("Casa e decoração", "Móveis de escritório"),
    "la_cuisine": ("Casa e decoração", "La Cuisine"),
    "flores": ("Casa e decoração", "Flores"),
    "artigos_de_natal": ("Casa e decoração", "Artigos de Natal"),
    "artigos_de_festas": ("Casa e decoração", "Artigos de festa"),
    "eletroportateis": ("Eletrodomésticos", "Eletroportáteis"),
    "eletrodomesticos": ("Eletrodomésticos", "Eletrodomésticos"),
    "eletrodomesticos_2": ("Eletrodomésticos", "Eletrodomésticos 2"),
    "portateis_casa_forno_e_cafe": ("Eletrodomésticos", "Fornos e cafeteiras"),
    "portateis_cozinha_e_preparadores_de_alimentos": ("Eletrodomésticos", "Preparadores de alimentos"),
    "climatizacao": ("Eletrodomésticos", "Climatização"),
    "informatica_acessorios": ("Informática e eletrônicos", "Informática e acessórios"),
    "pcs": ("Informática e eletrônicos", "Computadores"),
    "pc_gamer": ("Informática e eletrônicos", "PC gamer"),
    "tablets_impressao_imagem": ("Informática e eletrônicos", "Tablets e impressão"),
    "eletronicos": ("Informática e eletrônicos", "Eletrônicos"),
    "telefonia": ("Informática e eletrônicos", "Telefonia"),
    "telefonia_fixa": ("Informática e eletrônicos", "Telefonia fixa"),
    "consoles_games": ("Informática e eletrônicos", "Consoles e games"),
    "audio": ("Informática e eletrônicos", "Áudio"),
    "cine_foto": ("Informática e eletrônicos", "Cine e foto"),
    "esporte_lazer": ("Esporte e lazer", "Esporte e lazer"),
    "instrumentos_musicais": ("Esporte e lazer", "Instrumentos musicais"),
    "relogios_presentes": ("Relógios e presentes", "Relógios e presentes"),
    "cool_stuff": ("Relógios e presentes", "Cool stuff"),
    "fashion_bolsas_e_acessorios": ("Moda e acessórios", "Bolsas e acessórios"),
    "fashion_calcados": ("Moda e acessórios", "Calçados"),
    "fashion_roupa_masculina": ("Moda e acessórios", "Roupa masculina"),
    "fashion_roupa_feminina": ("Moda e acessórios", "Roupa feminina"),
    "fashion_underwear_e_moda_praia": ("Moda e acessórios", "Underwear e moda praia"),
    "fashion_esporte": ("Moda e acessórios", "Moda esporte"),
    "fashion_roupa_infanto_juvenil": ("Moda e acessórios", "Roupa infantojuvenil"),
    "malas_acessorios": ("Moda e acessórios", "Malas e acessórios"),
    "bebes": ("Bebês e brinquedos", "Bebês"),
    "brinquedos": ("Bebês e brinquedos", "Brinquedos"),
    "automotivo": ("Automotivo", "Automotivo"),
    "ferramentas_jardim": ("Ferramentas e construção", "Ferramentas de jardim"),
    "construcao_ferramentas_construcao": ("Ferramentas e construção", "Ferramentas de construção"),
    "construcao_ferramentas_jardim": ("Ferramentas e construção", "Construção e jardim"),
    "construcao_ferramentas_ferramentas": ("Ferramentas e construção", "Ferramentas"),
    "construcao_ferramentas_iluminacao": ("Ferramentas e construção", "Iluminação"),
    "construcao_ferramentas_seguranca": ("Ferramentas e construção", "Segurança"),
    "casa_construcao": ("Ferramentas e construção", "Casa e construção"),
    "sinalizacao_e_seguranca": ("Ferramentas e construção", "Sinalização e segurança"),
    "agro_industria_e_comercio": ("Ferramentas e construção", "Agro, indústria e comércio"),
    "industria_comercio_e_negocios": ("Ferramentas e construção", "Indústria e negócios"),
    "papelaria": ("Livros, papelaria e mídia", "Papelaria"),
    "livros_tecnicos": ("Livros, papelaria e mídia", "Livros técnicos"),
    "livros_interesse_geral": ("Livros, papelaria e mídia", "Livros de interesse geral"),
    "livros_importados": ("Livros, papelaria e mídia", "Livros importados"),
    "artes": ("Livros, papelaria e mídia", "Artes"),
    "artes_e_artesanato": ("Livros, papelaria e mídia", "Artes e artesanato"),
    "musica": ("Livros, papelaria e mídia", "Música"),
    "cds_dvds_musicais": ("Livros, papelaria e mídia", "CDs e DVDs musicais"),
    "dvds_blu_ray": ("Livros, papelaria e mídia", "DVDs e Blu-ray"),
    "alimentos_bebidas": ("Alimentos e bebidas", "Alimentos e bebidas"),
    "alimentos": ("Alimentos e bebidas", "Alimentos"),
    "bebidas": ("Alimentos e bebidas", "Bebidas"),
    "pet_shop": ("Pet shop", "Pet shop"),
    "market_place": (OTHERS, "Marketplace"),
    "seguros_e_servicos": (OTHERS, "Seguros e serviços"),
}


# ------------------------------------------------------------------
# Preparação (a partir dos CSVs originais)
# ------------------------------------------------------------------
def read_raw(folder: Path) -> dict[str, pd.DataFrame]:
    """Lê os CSVs originais da Olist que a preparação usa."""
    names = ["orders", "order_items", "customers", "products", "order_reviews"]
    tables = {name: pd.read_csv(folder / f"olist_{name}_dataset.csv") for name in names}
    for name in ("orders", "order_reviews"):
        for column in tables[name].columns:
            if column.endswith(("_timestamp", "_date", "_at")):
                tables[name][column] = pd.to_datetime(tables[name][column], errors="coerce")
    return tables


def prepare_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Uma linha por pedido válido na janela, com entrega, avaliação e valores."""
    orders = raw["orders"]
    orders = orders[
        ~orders["order_status"].isin(EXCLUDED_STATUS) & orders["order_purchase_timestamp"].between(START, END)
    ]
    items = (
        raw["order_items"]
        .groupby("order_id")
        .agg(items=("order_item_id", "count"), value=("price", "sum"), freight=("freight_value", "sum"))
    )
    reviews = raw["order_reviews"].groupby("order_id")["review_score"].mean()
    customers = raw["customers"].set_index("customer_id")[["customer_unique_id", "customer_state"]]

    table = (
        orders.join(customers, on="customer_id")
        .join(items, on="order_id", how="inner")  # pedidos sem itens não têm o que analisar
        .join(reviews, on="order_id")
    )
    # códigos curtos no lugar dos identificadores de 32 caracteres (ficam com poucos MB no git)
    codes = {customer: f"C{i:05d}" for i, customer in enumerate(sorted(table["customer_unique_id"].unique()), 1)}
    table = table.assign(
        region=table["customer_state"].map(REGIONS), customer_unique_id=table["customer_unique_id"].map(codes)
    )
    return table.rename(
        columns={
            "customer_unique_id": "customer",
            "customer_state": "state",
            "order_status": "status",
            "order_purchase_timestamp": "purchased_at",
            "order_delivered_customer_date": "delivered_at",
            "order_estimated_delivery_date": "estimated_at",
        }
    )[
        [
            "order_id",
            "customer",
            "state",
            "region",
            "status",
            "purchased_at",
            "delivered_at",
            "estimated_at",
            "items",
            "value",
            "freight",
            "review_score",
        ]
    ].reset_index(drop=True)


def prepare_sales(raw: dict[str, pd.DataFrame], orders: pd.DataFrame) -> pd.DataFrame:
    """Tabela no formato do app: uma linha por pedido e grupo de categoria."""
    products = raw["products"].set_index("product_id")["product_category_name"]
    items = raw["order_items"][raw["order_items"]["order_id"].isin(orders["order_id"])]
    items = items.assign(raw_category=items["product_id"].map(products))
    mapped = items["raw_category"].map(CATEGORIES)
    items = items.assign(
        category=mapped.map(lambda v: v[0] if isinstance(v, tuple) else OTHERS),
        product=mapped.map(lambda v: v[1] if isinstance(v, tuple) else "Sem categoria"),
    )

    # produto da linha = a categoria original de maior receita dentro do grupo
    by_product = items.groupby(["order_id", "category", "product"], as_index=False)["price"].sum()
    main_product = by_product.sort_values("price").drop_duplicates(["order_id", "category"], keep="last")
    lines = items.groupby(["order_id", "category"], as_index=False).agg(
        units=("order_item_id", "count"), revenue=("price", "sum")
    )
    lines = lines.merge(main_product[["order_id", "category", "product"]], on=["order_id", "category"])
    info = orders.set_index("order_id")[["purchased_at", "region", "customer"]]
    lines = lines.join(info, on="order_id")

    codes = {order: f"P{i:06d}" for i, order in enumerate(sorted(orders["order_id"]), 1)}
    sales = pd.DataFrame(
        {
            "order_id": lines["order_id"].map(codes),
            "date": lines["purchased_at"],
            "category": lines["category"],
            "region": lines["region"],
            "product": lines["product"],
            "units": lines["units"],
            "unit_price": (lines["revenue"] / lines["units"]).round(2),
            "revenue": lines["revenue"].round(2),
            "cost": float("nan"),
            "profit": float("nan"),
            "customer_id": lines["customer"],
        }
    )
    return sales.sort_values("date").reset_index(drop=True)


def raw_daily_orders(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pedidos por dia na base original (todos os status e datas)."""
    dates = raw["orders"]["order_purchase_timestamp"].dt.normalize()
    daily = dates.value_counts().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    return daily.rename_axis("data").rename("pedidos").reset_index()


def save(sales: pd.DataFrame, orders: pd.DataFrame, folder: Path = DATA_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    # categóricos e compressão: as tabelas ficam com poucos MB no repositório
    categorical = ["category", "region", "product"]
    sales.astype({c: "category" for c in categorical}).to_parquet(
        folder / SALES_FILE.name, index=False, compression="zstd"
    )
    # o código do pedido só serve para ligar as tabelas durante a preparação
    orders.drop(columns="order_id").astype({c: "category" for c in ["state", "region", "status"]}).to_parquet(
        folder / ORDERS_FILE.name, index=False, compression="zstd"
    )


# ------------------------------------------------------------------
# Uso no app
# ------------------------------------------------------------------
def is_available() -> bool:
    return SALES_FILE.exists()


def load_olist_sales(path: Path = SALES_FILE) -> pd.DataFrame:
    """Carrega e valida a tabela de vendas da Olist (o cache fica na camada de interface)."""
    df = pd.read_parquet(path)
    for column in ("category", "region", "product"):
        df[column] = df[column].astype(str)
    df = validate(df, SALES_SCHEMA, source="Olist")
    logger.info("Base da Olist carregada: %d linhas de pedido", len(df))
    return df.sort_values("date").reset_index(drop=True)


def load_olist_orders(path: Path = ORDERS_FILE) -> pd.DataFrame:
    return pd.read_parquet(path)
