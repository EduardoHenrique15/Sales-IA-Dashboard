"""
Dados de identidade do projeto exibidos na interface (nome, autor, links).
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Insight Engine"
TAGLINE = "Dashboard executivo de vendas com IA"
VERSION = "0.7.0"
AUTHOR = "Eduardo Henrique"
GITHUB_URL = "https://github.com/EduardoHenrique15/Sales-IA-Dashboard"

_ASSETS = Path(__file__).resolve().parents[2] / "assets"
LOGO = str(_ASSETS / "logo.svg")
LOGO_ICON = str(_ASSETS / "logo_icon.svg")
