"""
Blocos de interface reutilizados pelas páginas.
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from insight_engine.ai.agent import SOURCE_GEMINI, ReportResult, generate_executive_summary
from insight_engine.analytics.kpis import CryptoKPIs, SalesKPIs

# Chave do `st.session_state` onde a barra lateral guarda a chave do Gemini
# digitada pelo usuário.
GEMINI_KEY_STATE = "gemini_api_key"


def kpi_card(label: str, value: str, delta: str | None = None, delta_positive: bool | None = None) -> None:
    # Os textos são escapados porque vêm dos dados (ex.: nome de categoria)
    # e são exibidos como HTML.
    delta_html = ""
    if delta:
        css_class = (
            "kpi-delta-up" if delta_positive is True
            else "kpi-delta-down" if delta_positive is False
            else "kpi-delta-neutral"
        )
        delta_html = f'<div class="{css_class}">{html.escape(delta)}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value">{html.escape(value)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_section(df: pd.DataFrame, kpis: SalesKPIs | CryptoKPIs, period_label: str, dataset_name: str) -> None:
    """Botão de geração + exibição do Relatório Executivo.

    O último relatório fica guardado por página, para que o relatório de
    vendas não apareça na página de criptomoedas e vice-versa.
    """
    state_key = f"report_{dataset_name}"

    st.divider()
    st.header("🤖 Relatório Executivo Automático")
    st.caption("Gerado por IA a partir dos dados filtrados acima — mesma fonte de números do dashboard.")

    if st.button("✨ Gerar Relatório com IA", type="primary"):
        with st.spinner("Analisando dados e redigindo o relatório..."):
            st.session_state[state_key] = generate_executive_summary(
                df=df,
                kpis=kpis,
                period_label=period_label,
                dataset_name=dataset_name,
                api_key=st.session_state.get(GEMINI_KEY_STATE) or None,
            )

    result: ReportResult | None = st.session_state.get(state_key)
    if result is None:
        st.info("Clique no botão acima para gerar o relatório executivo com base nos dados e filtros atuais.")
        return

    if result.source == SOURCE_GEMINI:
        badge_class, badge_text = "badge-gemini", "Gerado por Gemini API"
    else:
        badge_class, badge_text = "badge-fallback", "Motor estatístico local (fallback)"
    st.markdown(f'<span class="report-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    if result.fallback_reason:
        st.warning(f"⚠️ O Gemini não foi usado porque {result.fallback_reason}")

    # O Markdown do Streamlit interpreta texto entre dois "$" como fórmula
    # (LaTeX); escapar o "$" evita que "R$ ... R$" vire uma equação.
    st.markdown(result.markdown.replace("$", r"\$"))

    st.download_button(
        "⬇️ Baixar Relatório (Markdown)",
        data=result.markdown,
        file_name=f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )


def footer() -> None:
    st.divider()
    st.caption("Insight Engine • Projeto de portfólio — Streamlit + Pandas + Plotly + IA (Gemini / scikit-learn fallback)")
