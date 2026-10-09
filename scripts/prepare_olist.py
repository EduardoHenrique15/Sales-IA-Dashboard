"""
Gera `data/olist/vendas.parquet` e `data/olist/pedidos.parquet` a partir dos
CSVs originais da Olist.

Os CSVs não ficam no repositório (são ~60 MB). Para refazer as tabelas:

  1. Baixe o dataset em https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  2. Descompacte em uma pasta (ex.: data/raw/olist/)
  3. python scripts/prepare_olist.py data/raw/olist

As decisões de preparação estão documentadas em `insight_engine/data/olist.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from insight_engine.data import olist  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara as tabelas da Olist usadas pelo app.")
    parser.add_argument("pasta", type=Path, help="pasta com os CSVs originais do Kaggle")
    parser.add_argument("--saida", type=Path, default=olist.DATA_DIR, help="pasta das tabelas geradas")
    args = parser.parse_args(argv)

    raw = olist.read_raw(args.pasta)
    orders = olist.prepare_orders(raw)
    sales = olist.prepare_sales(raw, orders)
    olist.save(sales, orders, args.saida)
    olist.raw_daily_orders(raw).to_csv(args.saida / olist.RAW_DAILY_FILE.name, index=False)

    unmapped = sorted(set(raw["products"]["product_category_name"].dropna()) - set(olist.CATEGORIES))
    print(f"Pedidos: {len(orders):,} | linhas de venda: {len(sales):,} | clientes: {orders['customer'].nunique():,}")
    print(f"Período: {sales['date'].min():%d/%m/%Y} a {sales['date'].max():%d/%m/%Y}")
    if unmapped:
        print(f"Categorias sem grupo (foram para '{olist.OTHERS}'): {', '.join(unmapped)}")
    for path in (args.saida / olist.SALES_FILE.name, args.saida / olist.ORDERS_FILE.name):
        print(f"{path}: {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
