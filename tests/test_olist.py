"""
Testes da base da Olist: preparação a partir dos CSVs e uso no app.
"""

from __future__ import annotations

import pandas as pd
import pytest

from insight_engine.analytics.anomalies import daily_series
from insight_engine.analytics.customers import rfm_table
from insight_engine.analytics.kpis import compute_sales_kpis, count_orders
from insight_engine.data import olist
from scripts import prepare_olist


def raw_tables() -> dict[str, pd.DataFrame]:
    """Mini base no formato original da Olist, com os casos que a preparação trata."""
    ts = pd.Timestamp
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            "order_status": ["delivered", "delivered", "canceled", "delivered", "shipped"],
            "order_purchase_timestamp": [
                ts("2017-03-01 10:00"),
                ts("2017-03-02 11:00"),
                ts("2017-03-03 12:00"),  # cancelado: fica de fora
                ts("2018-09-10 09:00"),  # depois do fim da coleta: fica de fora
                ts("2018-01-15 08:00"),
            ],
            "order_delivered_customer_date": [ts("2017-03-05"), ts("2017-03-20"), pd.NaT, pd.NaT, pd.NaT],
            "order_estimated_delivery_date": [ts("2017-03-10"), ts("2017-03-12"), pd.NaT, pd.NaT, ts("2018-01-30")],
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o1", "o2", "o3", "o4", "o5"],
            "order_item_id": [1, 2, 3, 1, 1, 1, 1],
            "product_id": ["p_cama", "p_moveis", "p_pc", "p_novo", "p_cama", "p_cama", "p_sem"],
            "price": [100.0, 50.0, 30.0, 20.0, 99.0, 99.0, 10.0],
            "freight_value": [10.0, 5.0, 3.0, 2.0, 9.0, 9.0, 1.0],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4", "c5"],
            # c1 e c5 são a mesma pessoa (o customer_id da Olist muda a cada pedido)
            "customer_unique_id": ["u1", "u2", "u3", "u4", "u1"],
            "customer_state": ["SP", "BA", "RS", "AM", "SP"],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p_cama", "p_moveis", "p_pc", "p_novo", "p_sem"],
            "product_category_name": ["cama_mesa_banho", "moveis_decoracao", "pcs", "categoria_nova", None],
        }
    )
    reviews = pd.DataFrame({"order_id": ["o1", "o1", "o2"], "review_score": [5, 3, 1]})
    return {
        "orders": orders,
        "order_items": items,
        "customers": customers,
        "products": products,
        "order_reviews": reviews,
    }


@pytest.fixture
def prepared():
    raw = raw_tables()
    orders = olist.prepare_orders(raw)
    return orders, olist.prepare_sales(raw, orders)


def test_pedidos_validos_na_janela(prepared):
    orders, _ = prepared
    assert sorted(orders["order_id"]) == ["o1", "o2", "o5"]  # sem cancelado e sem pedido depois da coleta
    o1 = orders.set_index("order_id").loc["o1"]
    assert (o1["items"], o1["value"], o1["freight"], o1["review_score"]) == (3, 180.0, 18.0, 4.0)
    assert o1["region"] == "Sudeste"


def test_cliente_unico_e_nao_o_id_do_pedido(prepared):
    orders, sales = prepared
    by_order = orders.set_index("order_id")["customer"]
    assert by_order["o1"] == by_order["o5"] != by_order["o2"]
    assert sales["customer_id"].str.match(r"C\d{5}$").all()


def test_uma_linha_por_pedido_e_grupo_de_categoria(prepared):
    _, sales = prepared
    o1 = sales[sales["order_id"] == sales["order_id"].iloc[0]]
    # o1 tem cama e móveis (mesmo grupo) e um computador (outro grupo): 2 linhas, mesmo código de pedido
    assert sorted(o1["category"]) == ["Casa e decoração", "Informática e eletrônicos"]
    casa = o1.set_index("category").loc["Casa e decoração"]
    assert (casa["units"], casa["revenue"], casa["product"]) == (2, 150.0, "Cama, mesa e banho")
    assert casa["unit_price"] == 75.0


def test_categoria_desconhecida_ou_ausente_vai_para_outros(prepared):
    _, sales = prepared
    others = sales[sales["category"] == olist.OTHERS]
    assert sorted(others["product"]) == ["Sem categoria", "Sem categoria"]


def test_receita_bate_entre_as_tabelas(prepared):
    orders, sales = prepared
    assert sales["revenue"].sum() == pytest.approx(orders["value"].sum())
    assert sales["cost"].isna().all()  # a Olist não informa custo


def test_script_gera_os_arquivos(tmp_path, capsys):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "saida"
    raw_dir.mkdir()
    for name, table in raw_tables().items():
        table.to_csv(raw_dir / f"olist_{name}_dataset.csv", index=False)

    assert prepare_olist.main([str(raw_dir), "--saida", str(out_dir)]) == 0
    sales = olist.load_olist_sales(out_dir / olist.SALES_FILE.name)
    orders = olist.load_olist_orders(out_dir / olist.ORDERS_FILE.name)
    assert len(orders) == 3 and "order_id" not in orders.columns
    daily = pd.read_csv(out_dir / olist.RAW_DAILY_FILE.name)
    assert daily["pedidos"].sum() == 5  # a contagem original inclui cancelados e pedidos fora da janela
    assert count_orders(sales) == 3 and len(sales) == 4
    output = capsys.readouterr().out
    assert "Pedidos: 3" in output and "categoria_nova" in output


# ------------------------------------------------------------------
# Base real guardada no repositório
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def olist_sales():
    return olist.load_olist_sales()


def test_base_real_no_repositorio(olist_sales):
    orders = olist.load_olist_orders()
    assert count_orders(olist_sales) == len(orders) > 95_000
    assert olist_sales["date"].min() >= olist.START and olist_sales["date"].max() <= olist.END
    assert olist_sales["revenue"].sum() == pytest.approx(orders["value"].sum())
    assert set(olist_sales["region"]) == {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}
    assert olist_sales["category"].nunique() == 14


def test_pedidos_e_frequencia_contam_pedidos_e_nao_linhas(olist_sales):
    kpis = compute_sales_kpis(olist_sales)
    assert kpis.n_orders < len(olist_sales)
    assert kpis.avg_ticket == pytest.approx(kpis.total_revenue / kpis.n_orders)
    rfm = rfm_table(olist_sales)
    assert rfm["frequency"].sum() == kpis.n_orders
    assert daily_series(olist_sales, "orders").sum() == kpis.n_orders
