"""
Escolha da base de vendas ativa: exemplo (sintética) ou arquivo enviado.

A base enviada fica só na sessão do navegador (`st.session_state`): não é
gravada em disco nem compartilhada com outros usuários.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from insight_engine.data.upload import SalesDataset
from insight_engine.ui import cached

UPLOAD_STATE = "uploaded_dataset"
CHOICE_STATE = "dataset_choice"
EXAMPLE, UPLOADED = "exemplo", "upload"
EXAMPLE_NAME = "Base de exemplo (sintética)"


def example_dataset() -> SalesDataset:
    return SalesDataset(df=cached.sales_data(), name=EXAMPLE_NAME)


def dataset_key(dataset: SalesDataset) -> str:
    """Identificador curto da base, usado para separar o estado dos filtros por base."""
    return hashlib.md5(dataset.name.encode()).hexdigest()[:8]


def is_example(dataset: SalesDataset) -> bool:
    return dataset.name == EXAMPLE_NAME


def set_uploaded(dataset: SalesDataset) -> None:
    st.session_state[UPLOAD_STATE] = dataset
    st.session_state[CHOICE_STATE] = UPLOADED


def clear_uploaded() -> None:
    st.session_state.pop(UPLOAD_STATE, None)
    st.session_state[CHOICE_STATE] = EXAMPLE


def uploaded_dataset() -> SalesDataset | None:
    return st.session_state.get(UPLOAD_STATE)


def active_dataset() -> SalesDataset:
    """Base usada pelas páginas de vendas, com seletor na barra lateral."""
    uploaded = uploaded_dataset()
    if uploaded is None:
        st.sidebar.caption(f"📂 {EXAMPLE_NAME}. Envie a sua em **Importar dados**.")
        return example_dataset()

    names = {EXAMPLE: EXAMPLE_NAME, UPLOADED: f"Meu arquivo: {uploaded.name}"}
    current = st.session_state.get(CHOICE_STATE, UPLOADED)

    # O valor fica guardado fora do widget para sobreviver à troca de página.
    def remember() -> None:
        st.session_state[CHOICE_STATE] = st.session_state["_dataset_choice_widget"]

    choice = st.sidebar.radio(
        "Base de dados",
        options=list(names),
        format_func=names.get,
        index=list(names).index(current),
        key="_dataset_choice_widget",
        on_change=remember,
    )
    return uploaded if choice == UPLOADED else example_dataset()
