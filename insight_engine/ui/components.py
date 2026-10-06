"""
Blocos de interface reutilizados pelas páginas.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from insight_engine.ai.agent import SOURCE_LLM, ReportResult, generate_executive_summary
from insight_engine.ai.context import SalesFacts
from insight_engine.ai.report import ExecutiveReport
from insight_engine.ai.report_pdf import render_pdf
from insight_engine.ui import branding
from insight_engine.ui.ai_access import GEMINI_KEY_STATE, ai_access, report_cache
from insight_engine.ui.theme import PLOTLY_CONFIG

__all__ = ["GEMINI_KEY_STATE", "chart_card", "escape_currency", "page_header", "report_section", "sidebar_footer"]

_LEVEL_COLOR = {"alta": "red", "média": "orange", "baixa": "gray"}
_LEVEL_ORDER = {"alta": 0, "média": 1, "baixa": 2}


def escape_currency(text: str) -> str:
    """Escapa "$" para textos em Markdown do Streamlit.

    O Streamlit interpreta o trecho entre dois "$" como fórmula (LaTeX), então
    um texto com dois valores em reais ("R$ 10 ... R$ 20") viraria uma equação.
    """
    return text.replace("$", r"\$")


def page_header(title: str, subtitle: str, icon: str | None = None) -> None:
    """Título da página com uma linha de contexto logo abaixo."""
    st.title(title, icon=icon, anchor=False)
    st.caption(subtitle)


def chart_card(
    title: str,
    fig: go.Figure,
    *,
    caption: str | None = None,
    key: str | None = None,
    on_select: Callable[[], None] | None = None,
) -> None:
    """Gráfico dentro de um cartão com borda, título e legenda fora da figura.

    Com `on_select`, um clique em uma barra/ponto chama a função (filtro cruzado).
    """
    with st.container(border=True):
        st.markdown(f"**{escape_currency(title)}**")
        if caption:
            st.caption(escape_currency(caption))
        if on_select is None:
            st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key)
        else:
            st.plotly_chart(fig, config=PLOTLY_CONFIG, key=key, on_select=on_select, selection_mode="points")


def report_section(
    build_facts: Callable[[], SalesFacts | None],
    period_label: str,
    dataset_name: str,
    context: str = "",
) -> None:
    """Botão de geração + exibição do Relatório Executivo em cartões.

    `build_facts` só é chamado ao clicar no botão (os fatos incluem previsão e
    segmentação, que levam alguns segundos). O último relatório fica guardado
    por página e por base de dados (`context`).
    """
    state_key = f"report_{dataset_name}_{context}"

    st.space("medium")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("Relatório executivo", icon=":material/summarize:", anchor=False, width="stretch")
        clicked = st.button(
            "Gerar relatório com IA", type="primary", icon=":material/auto_awesome:", key="generate_report"
        )
    st.caption(
        "Escrito por IA a partir dos dados e filtros atuais — a mesma fonte de números do dashboard. "
        "Os números citados pela IA são conferidos automaticamente contra os dados."
    )

    if clicked:
        access = ai_access()
        writer = access.provider.name if access.provider is not None else "motor estatístico local"
        with st.status("Preparando o relatório...", expanded=True) as status:
            st.write(":material/calculate: Calculando indicadores, tendência, previsão e variação")
            facts = build_facts()
            st.write(f":material/edit_note: Redigindo com {writer}")
            st.session_state[state_key] = generate_executive_summary(
                facts=facts,
                dataset_name=dataset_name,
                period_label=period_label,
                provider=access.provider,
                provider_error=access.provider_error,
                cache=report_cache(),
                allow_call=access.allow_call,
                limit_message=access.limit_message,
            )
            st.write(":material/fact_check: Conferindo os números citados")
            status.update(label="Relatório pronto", state="complete", expanded=False)

    result: ReportResult | None = st.session_state.get(state_key)
    if result is None:
        with st.container(border=True):
            st.markdown(
                ":material/lightbulb: Clique em **Gerar relatório com IA** para receber um resumo do período, "
                "os principais riscos e as ações recomendadas, com base nos filtros atuais."
            )
        return

    _report_badges(result)
    if result.fallback_reason:
        st.warning(f"A IA generativa não foi usada porque {result.fallback_reason}", icon=":material/info:")
    _verification_notice(result)
    _report_cards(result.report)
    _report_downloads(result, dataset_name, period_label)


def _report_badges(result: ReportResult) -> None:
    with st.container(horizontal=True, gap="small"):
        if result.source == SOURCE_LLM:
            label = f"Gerado por {result.provider_name}" + (" (do cache)" if result.from_cache else "")
            st.badge(label, icon=":material/auto_awesome:", color="green")
        else:
            st.badge("Motor estatístico local (fallback)", icon=":material/functions:", color="violet")
        check = result.verification
        if check is not None and check.checked:
            text = f"{check.verified} de {check.checked} números conferidos"
            st.badge(text, icon=":material/fact_check:", color="green" if check.ok else "orange")


def _report_cards(report: ExecutiveReport) -> None:
    with st.container(border=True):
        st.caption("Resumo do período")
        st.markdown(f"#### {escape_currency(report.headline)}")

    col_highlights, col_risks = st.columns(2)
    with col_highlights, st.container(border=True, height="stretch"):
        st.markdown("**:material/trending_up: Destaques**")
        st.markdown("\n".join(f"- {escape_currency(item)}" for item in report.highlights))
    with col_risks, st.container(border=True, height="stretch"):
        st.markdown("**:material/warning: Riscos e pontos de atenção**")
        if not report.risks:
            st.caption("Nenhum risco relevante identificado.")
        for risk in sorted(report.risks, key=lambda r: _LEVEL_ORDER[r.severity]):
            st.badge(f"Gravidade {risk.severity}", color=_LEVEL_COLOR[risk.severity])  # type: ignore[arg-type]
            st.markdown(f"**{escape_currency(risk.title)}**  \n{escape_currency(risk.evidence)}")

    st.markdown("**:material/flag: Ações recomendadas**")
    actions = sorted(report.actions, key=lambda a: _LEVEL_ORDER[a.priority])
    for start in range(0, len(actions), 3):
        row = st.columns(3)
        for column, action in zip(row, actions[start : start + 3], strict=False):
            with column, st.container(border=True, height="stretch"):
                st.badge(f"Prioridade {action.priority}", color=_LEVEL_COLOR[action.priority])  # type: ignore[arg-type]
                st.markdown(f"**{escape_currency(action.action)}**")
                st.caption(escape_currency(action.rationale))


def _report_downloads(result: ReportResult, dataset_name: str, period_label: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    source = f"Gerado por {result.provider_name}" if result.source == SOURCE_LLM else "Motor estatístico local"
    check = result.verification
    check_label = f"Números verificados: {check.verified} de {check.checked}" if check and check.checked else None
    with st.container(horizontal=True):
        st.download_button(
            "Baixar PDF",
            data=render_pdf(result.report, f"Relatório Executivo - {dataset_name}", period_label, source, check_label),
            file_name=f"relatorio_executivo_{stamp}.pdf",
            mime="application/pdf",
            icon=":material/picture_as_pdf:",
            type="primary",
        )
        st.download_button(
            "Baixar Markdown",
            data=result.markdown,
            file_name=f"relatorio_executivo_{stamp}.md",
            mime="text/markdown",
            icon=":material/description:",
        )


def verified_message(count: int) -> str:
    """ "O número citado" / "Os 3 números citados" (concordância no singular e no plural)."""
    return "O número citado" if count == 1 else f"Os {count} números citados"


def _verification_notice(result: ReportResult) -> None:
    check = result.verification
    if check is None or check.checked == 0 or check.ok:
        return
    cited = ", ".join(f"`{n}`" for n in check.unverified)
    st.warning(
        escape_currency(
            f"{len(check.unverified)} de {check.checked} números citados pela IA não aparecem nos dados "
            f"calculados: {cited}. Podem ser cálculos próprios do modelo ou erros — confira antes de usar."
        ),
        icon=":material/search:",
    )


def sidebar_footer() -> None:
    st.sidebar.caption(
        f"{branding.APP_NAME} v{branding.VERSION} · por {branding.AUTHOR}  \n"
        f"[Código no GitHub]({branding.GITHUB_URL}) · Streamlit, Plotly, scikit-learn e Gemini"
    )
