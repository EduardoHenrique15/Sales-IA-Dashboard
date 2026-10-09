"""
O notebook de análise fica no repositório com as saídas (o GitHub mostra os gráficos).
Estes testes garantem que ele foi salvo executado e sem erros e, quando as
dependências do notebook estão instaladas, que ele ainda roda com o código atual.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "analise_olist.ipynb"


def code_cells() -> list[dict]:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return [c for c in cells if c["cell_type"] == "code"]


def test_notebook_salvo_executado_e_sem_erros():
    cells = code_cells()
    assert [c["execution_count"] for c in cells] == list(range(1, len(cells) + 1))  # em ordem, do zero
    outputs = [o for c in cells for o in c["outputs"]]
    assert not [o for o in outputs if o["output_type"] == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 6  # gráficos visíveis no GitHub


def test_notebook_roda_com_o_codigo_atual():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    pytest.importorskip("ipykernel")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(notebook, timeout=600, resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    client.execute()  # levanta CellExecutionError se alguma célula falhar
