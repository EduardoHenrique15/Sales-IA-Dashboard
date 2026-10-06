"""
formatting.py
================================================================
Formatação de números no padrão brasileiro (1.234,56).

Usado pela interface e pelo agente de IA, para que valores
apareçam iguais na tela e no relatório.
================================================================
"""

from __future__ import annotations

# Separadores para gráficos Plotly: decimal "," e milhar "."
PLOTLY_SEPARATORS = ",."


def _swap_separators(text: str) -> str:
    """Converte '1,234.56' (padrão en-US) em '1.234,56' (pt-BR)."""
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def format_number(value: float, decimals: int = 0) -> str:
    return _swap_separators(f"{value:,.{decimals}f}")


def format_brl(value: float, decimals: int = 2) -> str:
    """Moeda brasileira; negativos com o sinal antes do símbolo: -R$ 1.234,56."""
    sign = "-" if value < 0 else ""
    return f"{sign}R$ {format_number(abs(value), decimals)}"


def format_usd(value: float, decimals: int = 2) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}US$ {format_number(abs(value), decimals)}"


def format_pct(value: float, decimals: int = 1, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return _swap_separators(f"{value:{sign},.{decimals}f}") + "%"


def format_p_value(p: float) -> str:
    """Formata um p-valor para leitura: "p < 0,001" ou "p = 0,032"."""
    if p < 0.001:
        return "p < 0,001"
    return f"p = {format_number(p, 3)}"
