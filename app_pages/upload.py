"""
Página: importar uma base de vendas própria (CSV ou Excel).
"""

import io

import pandas as pd
import streamlit as st

from insight_engine.data.schemas import DataValidationError
from insight_engine.data.upload import (
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    UploadError,
    build_sales_dataset,
    guess_mapping,
    read_table,
    sample_csv,
)
from insight_engine.formatting import format_number
from insight_engine.ui import datasets
from insight_engine.ui.components import page_header

NONE_OPTION = "— não tenho esta coluna —"
STEPS = [
    (":material/download:", "1. Prepare a planilha", "Um pedido por linha, em CSV ou Excel. Use o modelo se preferir."),
    (
        ":material/upload_file:",
        "2. Envie o arquivo",
        "Ele fica só nesta sessão do navegador: não é salvo nem compartilhado.",
    ),
    (":material/rule:", "3. Confira as colunas", "O app sugere qual coluna é qual; ajuste e clique em usar."),
]

page_header(
    "Importar dados de vendas",
    "Analise a sua própria base em todas as páginas de vendas. Só **data** e **receita** são obrigatórias; "
    "com custo e cliente, o dashboard calcula também lucro, margem e a segmentação de clientes.",
    icon=":material/upload_file:",
)

for column, (icon, title, text) in zip(st.columns(3), STEPS, strict=True):
    with column, st.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {title}**")
        st.caption(text)

with st.expander("Formato esperado (exemplo)", icon=":material/table_view:"):
    st.dataframe(pd.read_csv(io.BytesIO(sample_csv()), sep=";", dtype=str), hide_index=True, width="stretch")
    st.download_button(
        "Baixar planilha-modelo (CSV)",
        data=sample_csv(),
        file_name="modelo_vendas.csv",
        mime="text/csv",
        icon=":material/download:",
    )

current = datasets.uploaded_dataset()
if current is not None:
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.markdown(
            f":material/check_circle: Base em uso: **{current.name}** ({format_number(len(current.df))} pedidos).",
            width="stretch",
        )
        if st.button("Remover base enviada", icon=":material/delete:", type="tertiary", key="remove_upload"):
            datasets.clear_uploaded()
            st.rerun()

file = st.file_uploader("Arquivo CSV ou Excel", type=["csv", "xlsx"])
if file is None:
    st.stop()

try:
    raw = read_table(file.name, file.getvalue())
except UploadError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

st.caption(f"{format_number(len(raw))} linhas e {len(raw.columns)} colunas lidas. Primeiras linhas:")
st.dataframe(raw.head(10), hide_index=True)

# ---------- MAPEAMENTO DE COLUNAS ----------
st.subheader("Qual coluna é qual?", icon=":material/rule:", anchor=False)
st.caption("As sugestões vêm dos nomes das colunas; confira antes de continuar.")

columns = list(raw.columns)
guesses = guess_mapping(columns)
mapping: dict[str, str | None] = {}
fields = {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}
grid = st.columns(4)
for i, (field, label) in enumerate(fields.items()):
    required = field in REQUIRED_FIELDS
    options = columns if required else [NONE_OPTION, *columns]
    guess = guesses.get(field)
    with grid[i % 4]:
        choice = st.selectbox(
            f"{label}{' *' if required else ''}",
            options=options,
            index=options.index(guess) if guess in options else (None if required else 0),
            placeholder="Selecione...",
            key=f"map_{field}_{file.file_id}",
        )
    mapping[field] = None if choice in (None, NONE_OPTION) else choice

if st.button("Usar estes dados", type="primary", icon=":material/check:", key="use_upload"):
    try:
        dataset = build_sales_dataset(raw, mapping, name=file.name)
    except (UploadError, DataValidationError) as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    datasets.set_uploaded(dataset)
    period = f"{dataset.df['date'].min():%d/%m/%Y} a {dataset.df['date'].max():%d/%m/%Y}"
    st.success(
        f"Base carregada: **{format_number(len(dataset.df))} pedidos** de {period}.", icon=":material/check_circle:"
    )
    for note in dataset.notes:
        st.info(note, icon=":material/info:")
    st.page_link("app_pages/sales.py", label="Ver o dashboard com estes dados", icon=":material/dashboard:")
