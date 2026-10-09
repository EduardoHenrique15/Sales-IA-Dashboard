"""
Escolha da base de vendas ativa: exemplo (sintética) ou arquivo enviado.

A base enviada fica só na sessão do navegador (`st.session_state`): não é
gravada em disco nem compartilhada com outros usuários.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from insight_engine.data import olist
from insight_engine.data.upload import SalesDataset
from insight_engine.ui import cached
from insight_engine.ui.layout import sidebar_filters

UPLOAD_STATE = "uploaded_dataset"
CHOICE_STATE = "dataset_choice"
EXAMPLE, OLIST, UPLOADED = "exemplo", "olist", "upload"
EXAMPLE_NAME = "Base de exemplo (sintética)"
DESCRIPTIONS = {
    EXAMPLE: "Dados sintéticos de 2023 a 2025, com anomalias plantadas de propósito para validar os métodos.",
    OLIST: "Pedidos reais de e-commerce de 2017 e 2018 (Olist, Kaggle, CC BY-NC-SA 4.0). "
    "Sem custo: lucro e margem não são calculados.",
}


def example_dataset() -> SalesDataset:
    return SalesDataset(df=cached.sales_data(), name=EXAMPLE_NAME)


def olist_dataset() -> SalesDataset:
    return SalesDataset(df=cached.olist_data(), name=olist.DATASET_NAME, has_cost=False, has_customers=True)


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
    """Base usada pelas páginas de vendas, escolhida na barra lateral."""
    uploaded = uploaded_dataset()
    names = {EXAMPLE: "Exemplo (sintética)"}
    if olist.is_available():
        names[OLIST] = "Olist: e-commerce real"
    if uploaded is not None:
        names[UPLOADED] = f"Meu arquivo: {uploaded.name}"

    current = st.session_state.get(CHOICE_STATE, UPLOADED if uploaded is not None else EXAMPLE)
    if current not in names:
        current = EXAMPLE

    # O valor fica guardado fora do widget para sobreviver à troca de página.
    def remember() -> None:
        st.session_state[CHOICE_STATE] = st.session_state["_dataset_choice_widget"]

    area = sidebar_filters()
    choice = area.radio(
        "Base de dados",
        options=list(names),
        format_func=names.get,
        index=list(names).index(current),
        key="_dataset_choice_widget",
        on_change=remember,
    )
    if choice == UPLOADED and uploaded is not None:
        return uploaded
    area.caption(DESCRIPTIONS[choice] + ("" if uploaded else " Envie a sua em **Importar dados**."))
    return olist_dataset() if choice == OLIST else example_dataset()
