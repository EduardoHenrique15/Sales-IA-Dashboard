"""
Configuração de logs da aplicação.
"""

from __future__ import annotations

import logging

from insight_engine.config import get_setting


def setup_logging() -> None:
    """Configura os logs do pacote `insight_engine` (uma única vez).

    O nível pode ser ajustado pela configuração `LOG_LEVEL` (padrão: INFO).
    """
    logger = logging.getLogger("insight_engine")
    if logger.handlers:  # o Streamlit reexecuta o script a cada interação
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    level = getattr(logging, (get_setting("LOG_LEVEL") or "INFO").upper(), None)
    logger.setLevel(level if isinstance(level, int) else logging.INFO)
    logger.propagate = False
