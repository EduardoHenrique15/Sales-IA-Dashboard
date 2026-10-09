"""
Página: converse com os dados (perguntas em linguagem natural).
"""

import json

import streamlit as st

from insight_engine.ai.chat import MAX_QUESTION_CHARS, ChatTurn, SalesDataTools, ask
from insight_engine.ai.providers.base import ChatMessage
from insight_engine.ui import datasets
from insight_engine.ui.ai_access import ai_access
from insight_engine.ui.components import escape_currency, page_header, verified_message

AVATARS = {"user": ":material/person:", "assistant": ":material/insights:"}


def example_questions(dataset) -> list[str]:
    """Perguntas de exemplo com os anos que existem na base ativa."""
    first, last = dataset.df["date"].min().year, dataset.df["date"].max().year
    previous = last - 1 if first < last else last
    why = (
        "Por que a receita caiu no 3º trimestre de 2024?"  # queda plantada na base de exemplo
        if datasets.is_example(dataset)
        else "Quais categorias mais cresceram no último trimestre?"
    )
    return [
        f"Qual região teve a menor receita em {last}?",
        why,
        f"Houve algum dia atípico em {previous}?",
        "Quanto devemos vender nos próximos 30 dias?",
    ]


dataset = datasets.active_dataset()
examples = example_questions(dataset)
page_header(
    "Converse com os dados",
    "Pergunte em linguagem natural. A IA responde consultando funções de análise pré-definidas e "
    f"seguras — ela não executa código. · Base: {dataset.name}",
    icon=":material/forum:",
)

access = ai_access()
if access.provider is None:
    st.info(
        "O chat precisa da IA generativa. Informe uma chave do Gemini em **Inteligência artificial**, na barra "
        "lateral (gratuita em aistudio.google.com), ou configure `GEMINI_API_KEY` no servidor."
        + (f" Motivo: {access.provider_error}" if access.provider_error else ""),
        icon=":material/key:",
    )
    with st.container(border=True):
        st.markdown("**Exemplos do que você poderá perguntar:**\n" + "\n".join(f"- {q}" for q in examples))
    st.stop()

tools = SalesDataTools(dataset.df, has_cost=dataset.has_cost, has_customers=dataset.has_customers)
state_key = datasets.chat_state_key(dataset)
conversation: list[tuple[ChatMessage, ChatTurn | None]] = st.session_state.setdefault(state_key, [])


def show_details(turn: ChatTurn) -> None:
    if turn.tool_calls:
        with st.expander(f"Consultas feitas ({len(turn.tool_calls)})", icon=":material/build:"):
            for call in turn.tool_calls:
                st.markdown(f"**{call.name}** — parâmetros: `{json.dumps(call.args, ensure_ascii=False)}`")
                st.json(call.result, expanded=False)
    check = turn.verification
    if check is not None and check.checked:
        if check.ok:
            verb = "confere" if check.checked == 1 else "conferem"
            st.caption(f":material/fact_check: {verified_message(check.checked)} na resposta {verb} com as consultas.")
        else:
            st.caption(
                escape_currency(
                    f":material/search: Números não encontrados nas consultas: {', '.join(check.unverified)}"
                )
            )


# ---------- HISTÓRICO ----------
for message, turn in conversation:
    with st.chat_message(message.role, avatar=AVATARS[message.role]):
        st.markdown(escape_currency(message.text))
        if turn is not None:
            show_details(turn)

# ---------- NOVA PERGUNTA ----------
clicked = None
if not conversation:
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(
            "Olá! Posso responder perguntas sobre **receita, lucro, categorias, regiões, clientes, anomalias "
            "e previsão** desta base. Cada número da resposta vem de uma consulta aos dados, que você pode "
            "conferir em **Consultas feitas**."
        )
        # a chave inclui a base: senão a escolha feita em outra base dispararia a pergunta de novo aqui
        clicked = st.pills("Experimente perguntar", examples, key=f"example_{state_key}", label_visibility="visible")
question = st.chat_input("Pergunte sobre as vendas...", max_chars=MAX_QUESTION_CHARS) or clicked

if conversation:
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Limpar conversa", icon=":material/delete_sweep:", type="tertiary", key="clear_chat"):
            st.session_state[state_key] = []
            st.rerun()

if question:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(escape_currency(question))

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        if access.allow_call is not None and not access.allow_call():
            answer = f"Não consegui responder: {access.limit_message}"
            st.warning(answer)
            conversation += [(ChatMessage("user", question), None), (ChatMessage("assistant", answer), None)]
            st.stop()

        turn = ChatTurn()
        history = [message for message, _ in conversation]
        try:
            with st.spinner("Consultando os dados..."):
                stream = ask(access.provider, tools, history, question, turn)
                first = next(stream, "")

            def rest():
                yield first
                yield from stream

            st.write_stream(escape_currency(chunk) for chunk in rest())
        except Exception as exc:  # noqa: BLE001 - falha da API vira mensagem, não erro na tela
            turn.text = f"Não consegui responder: {access.provider.describe_error(exc)}"
            st.error(turn.text)
        show_details(turn)

    conversation += [(ChatMessage("user", question), None), (ChatMessage("assistant", turn.text), turn)]
