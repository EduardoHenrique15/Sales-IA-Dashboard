import pandas as pd
import pytest

from insight_engine.analytics.periods import SalesFilters, filter_sales
from insight_engine.analytics.variance import revenue_bridge
from tests.helpers import as_date


def frame(rows):
    return pd.DataFrame(rows, columns=["category", "revenue", "units"])


def test_so_volume():
    previous = frame([("A", 100.0, 10)])
    current = frame([("A", 200.0, 20)])  # mesmo preço, o dobro de unidades
    bridge = revenue_bridge(current, previous)
    assert bridge.volume_effect == pytest.approx(100)
    assert bridge.price_effect == pytest.approx(0)
    assert bridge.mix_effect == pytest.approx(0)


def test_so_preco():
    bridge = revenue_bridge(frame([("A", 120.0, 10)]), frame([("A", 100.0, 10)]))
    assert bridge.price_effect == pytest.approx(20)
    assert bridge.volume_effect == pytest.approx(0)


def test_so_mix():
    # mesmas unidades e preços; a venda migra da categoria barata para a cara
    previous = frame([("cara", 100.0, 1), ("barata", 10.0, 1)])
    current = frame([("cara", 200.0, 2), ("barata", 0.0, 0)])
    bridge = revenue_bridge(current, previous)
    assert bridge.mix_effect == pytest.approx(90)
    assert bridge.volume_effect == pytest.approx(0)
    assert bridge.price_effect == pytest.approx(0)


def test_categoria_nova_e_categoria_que_sumiu():
    previous = frame([("A", 80.0, 10), ("C", 40.0, 2)])
    current = frame([("A", 100.0, 10), ("B", 50.0, 5)])
    bridge = revenue_bridge(current, previous)
    assert bridge.by_segment.loc["B", "price"] == pytest.approx(0)  # categoria nova entra como mix
    assert bridge.volume_effect + bridge.price_effect + bridge.mix_effect == pytest.approx(bridge.total_change)


def test_efeitos_somam_a_variacao_na_base_real(sales_df):
    current, previous = filter_sales(sales_df, SalesFilters(as_date("2025-10-02"), as_date("2025-12-31")))
    bridge = revenue_bridge(current, previous)
    assert bridge.current_revenue == pytest.approx(current["revenue"].sum())
    assert bridge.volume_effect + bridge.price_effect + bridge.mix_effect == pytest.approx(bridge.total_change)
    assert bridge.by_segment["total"].sum() == pytest.approx(bridge.total_change)


def test_periodo_vazio():
    with pytest.raises(ValueError):
        revenue_bridge(frame([("A", 1.0, 1)]), frame([]))
