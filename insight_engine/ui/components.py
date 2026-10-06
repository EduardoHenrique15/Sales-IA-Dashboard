"""
Blocos de interface reutilizados pelas páginas.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from datetime import datetime

import streamlit as st

from insight_engine.ai.agent import SOURCE_LLM, ReportResult, generate_executive_summary
from insight_engine.ai.context import SalesFacts
from insight_engine.ai.report_pdf import render_pdf
from insight_engine.ui.ai_access import GEMINI_KEY_STATE, ai_access, report_cache

__all__ = ["GEMINI_KEY_STATE", "escape_currency", "footer", "kpi_card", "report_section"]


def escape_currency(text: str) -> str:
    """Escapa "$" para textos em Markdown do Streamlit.

    O Streamlit interpreta o trecho entre dois "$" como fórmula (LaTeX), então
    um texto com dois valores em reais ("R$ 10 ... R$ 20") viraria uma equação.
    """
    return text.replace("$", r"\$")


def kpi_card(label: str, value: str, delta: str | None = None, delta_positive: bool | None = None) -> None:
    # Os textos são escapados porque vêm dos dados (ex.: nome de categoria)
    # e são exibidos como HTML.
    delta_html = ""
    if delta:
        css_class = (
            "kpi-delta-up"
            if delta_positive is True
            else "kpi-delta-down"
            if delta_positive is False
            else "kpi-delta-neutral"
        )
        delta_html = f'<div class="{css_class}">{html.escape(delta)}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value" title="{html.escape(value)}">{html.escape(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_section(
    build_facts: Callable[[], SalesFacts | None],
    period_label: str,
    dataset_name: str,
    context: str = "",
) -> None:
    """Botão de geração + exibição do Relatório Executivo.

    `build_facts` só é chamado ao clicar no botão (os fatos incluem previsão e
    segmentação, que levam alguns segundos). O último relatório fica guardado
    por página e por base de dados (`context`).
    """
    state_key = f"report_{dataset_name}_{context}"

    st.divider()
    st.header("🤖 Relatório Executivo Automático")
    st.caption(
        "Gerado por IA a partir dos dados filtrados acima — mesma fonte de números do dashboard. "
        "Os números citados pela IA são conferidos automaticamente contra os dados."
    )

    if st.button("✨ Gerar Relatório com IA", type="primary"):
        access = ai_access()
        with st.spinner("Analisando dados e redigindo o relatório..."):
            st.session_state[state_key] = generate_executive_summary(
                facts=build_facts(),
                dataset_name=dataset_name,
                period_label=period_label,
                provider=access.provider,
                provider_error=access.provider_error,
                cache=report_cache(),
                allow_call=access.allow_call,
                limit_message=access.limit_message,
            )

    result: ReportResult | None = st.session_state.get(state_key)
    if result is None:
        st.info("Clique no botão acima para gerar o relatório executivo com base nos dados e filtros atuais.")
        return

    if result.source == SOURCE_LLM:
        badge_class, badge_text = "badge-gemini", f"Gerado por {result.provider_name}"
        if result.from_cache:
            badge_text += " (do cache)"
    else:
        badge_class, badge_text = "badge-fallback", "Motor estatístico local (fallback)"
    st.markdown(f'<span class="report-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    if result.fallback_reason:
        st.warning(f"⚠️ A IA generativa não foi usada porque {result.fallback_reason}")
    _verification_notice(result)

    st.markdown(escape_currency(result.markdown))

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    source = f"Gerado por {result.provider_name}" if result.source == SOURCE_LLM else "Motor estatístico local"
    check = result.verification
    check_label = f"Números verificados: {check.verified} de {check.checked}" if check and check.checked else None
    col_pdf, col_md, _ = st.columns([1, 1, 3])
    col_pdf.download_button(
        "⬇️ Baixar PDF",
        data=render_pdf(result.report, f"Relatório Executivo - {dataset_name}", period_label, source, check_label),
        file_name=f"relatorio_executivo_{stamp}.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch",
    )
    col_md.download_button(
        "⬇️ Baixar Markdown",
        data=result.markdown,
        file_name=f"relatorio_executivo_{stamp}.md",
        mime="text/markdown",
        width="stretch",
    )


def verified_message(count: int) -> str:
    """ "O número citado" / "Os 3 números citados" (concordância no singular e no plural)."""
    return "O número citado" if count == 1 else f"Os {count} números citados"


def _verification_notice(result: ReportResult) -> None:
    check = result.verification
    if check is None or check.checked == 0:
        return
    if check.ok:
        st.caption(
            f"✅ {verified_message(check.checked)} pela IA {'confere' if check.checked == 1 else 'conferem'} "
            "com os dados."
        )
    else:
        cited = ", ".join(f"`{n}`" for n in check.unverified)
        st.warning(
            escape_currency(
                f"🔎 {len(check.unverified)} de {check.checked} números citados pela IA não aparecem nos dados "
                f"calculados: {cited}. Podem ser cálculos próprios do modelo ou erros — confira antes de usar."
            )
        )


def footer() -> None:
    st.divider()
    st.caption(
        "Insight Engine • Projeto de portfólio — Streamlit + Pandas + Plotly + IA (Gemini / scikit-learn fallback)"
    )
