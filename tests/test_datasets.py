from types import SimpleNamespace
from unittest import mock

from insight_engine.data.upload import SalesDataset
from insight_engine.ui import datasets


def test_novo_arquivo_com_o_mesmo_nome_limpa_conversa_e_relatorio(small_sales_df):
    dataset = SalesDataset(df=small_sales_df, name="vendas.csv")
    state = {
        "chat_vendas.csv": ["conversa antiga"],
        "report_vendas.csv_abc": "relatório antigo",
        "chat_outra.csv": ["fica"],
        "report_outra.csv_abc": "fica",
    }
    with mock.patch.object(datasets, "st", SimpleNamespace(session_state=state)):
        datasets.set_uploaded(dataset)

    assert set(state) == {"chat_outra.csv", "report_outra.csv_abc", datasets.UPLOAD_STATE, datasets.CHOICE_STATE}
    assert state[datasets.UPLOAD_STATE] is dataset
