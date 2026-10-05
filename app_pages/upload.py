"""
Página: importar uma base de vendas própria (CSV ou Excel).
"""

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

NONE_OPTION = "— não tenho esta coluna —"

st.title("📤 Importar dados de vendas")
st.markdown(
    "Envie uma planilha com **um pedido por linha** para analisar os seus próprios dados em todas as "
    "páginas de vendas. Só **data** e **receita** são obrigatórias; com custo e cliente, o dashboard "
    "calcula também lucro, margem e a segmentação de clientes."
)
st.caption("🔒 O arquivo fica apenas nesta sessão do navegador: não é salvo nem compartilhado.")
st.download_button(
    "⬇️ Baixar planilha-modelo (CSV)",
    data=sample_csv(),
    file_name="modelo_vendas.csv",
    mime="text/csv",
)

current = datasets.uploaded_dataset()
if current is not None:
    st.success(f"Base em uso: **{current.name}** ({format_number(len(current.df))} pedidos).")
    if st.button("Remover base enviada"):
        datasets.clear_uploaded()
        st.rerun()

st.divider()
file = st.file_uploader("Arquivo CSV ou Excel", type=["csv", "xlsx"])
if file is None:
    st.stop()

try:
    raw = read_table(file.name, file.getvalue())
except UploadError as exc:
    st.error(f"❌ {exc}")
    st.stop()

st.caption(f"{format_number(len(raw))} linhas e {len(raw.columns)} colunas lidas. Primeiras linhas:")
st.dataframe(raw.head(10), hide_index=True)

# ---------- MAPEAMENTO DE COLUNAS ----------
st.subheader("Qual coluna é qual?")
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

if st.button("✅ Usar estes dados", type="primary"):
    try:
        dataset = build_sales_dataset(raw, mapping, name=file.name)
    except (UploadError, DataValidationError) as exc:
        st.error(f"❌ {exc}")
        st.stop()

    datasets.set_uploaded(dataset)
    period = f"{dataset.df['date'].min():%d/%m/%Y} a {dataset.df['date'].max():%d/%m/%Y}"
    st.success(f"Base carregada: **{format_number(len(dataset.df))} pedidos** de {period}.")
    for note in dataset.notes:
        st.info(note)
    st.page_link("app_pages/sales.py", label="Ver o dashboard com estes dados", icon="📦")
