import pytest

from insight_engine.formatting import format_brl, format_number, format_pct, format_usd


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (0, 0, "0"),
        (1234, 0, "1.234"),
        (1234567.891, 2, "1.234.567,89"),
        (-9876.5, 1, "-9.876,5"),
    ],
)
def test_format_number_usa_padrao_brasileiro(value, decimals, expected):
    assert format_number(value, decimals) == expected


def test_format_moedas():
    assert format_brl(2838404.99) == "R$ 2.838.404,99"
    assert format_usd(0.5) == "US$ 0,50"
    assert format_usd(1_000_000, 0) == "US$ 1.000.000"


def test_moeda_negativa_tem_sinal_antes_do_simbolo():
    assert format_brl(-70512.08) == "-R$ 70.512,08"
    assert format_usd(-5) == "-US$ 5,00"


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (12.345, {}, "12,3%"),
        (31.4, {"signed": True}, "+31,4%"),
        (-3.456, {"signed": True}, "-3,5%"),
        (0.03, {"decimals": 2}, "0,03%"),
        (1500, {"decimals": 0}, "1.500%"),
    ],
)
def test_format_pct(value, kwargs, expected):
    assert format_pct(value, **kwargs) == expected
