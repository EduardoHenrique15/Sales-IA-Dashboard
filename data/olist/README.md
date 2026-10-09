# Dados da Olist

Tabelas preparadas a partir do dataset público
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
publicado pela Olist no Kaggle sob a licença
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (uso não comercial, com atribuição,
compartilhado pela mesma licença). Os dados são de pedidos reais de 2016 a 2018, anonimizados pela Olist.

| Arquivo | Conteúdo |
|---|---|
| `vendas.parquet` | Formato do app: uma linha por pedido e grupo de categoria (data, categoria, região, produto, unidades, receita, cliente) |
| `pedidos.parquet` | Uma linha por pedido: estado, região, datas de compra, entrega e prazo, valor, frete e nota da avaliação |
| `pedidos_por_dia_original.csv` | Pedidos por dia na base original, antes dos cortes (mostra a escolha da janela de datas) |

Gerados por `python scripts/prepare_olist.py <pasta com os CSVs do Kaggle>`. As decisões de preparação (janela
de datas, cliente único, grupos de categoria, regiões) estão em
[`insight_engine/data/olist.py`](../../insight_engine/data/olist.py) e são analisadas no
[notebook](../../notebooks/analise_olist.ipynb).
