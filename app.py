"""
app.py
================================================================
Ponto de entrada do Dashboard Interativo.

Responsabilidade única deste arquivo: ORQUESTRAR a interface.
Toda a lógica de dados vive em `data_loader.py`, todo cálculo de
métricas em `kpi_engine.py`, e toda geração de texto em
`ai_agent.py`. Isso mantém o app testável e fácil de estender.

Rodar com:
    streamlit run app.py
================================================================
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ai_agent import generate_executive_summary
from data_loader import COIN_OPTIONS, load_crypto_data, load_sales_data
from kpi_engine import compute_crypto_kpis, compute_sales_kpis, previous_period_df

# ==================================================================
# CONFIGURAÇÃO DE PÁGINA E ESTILO
# ==================================================================
st.set_page_config(
    page_title="Insight Engine | Dashboard Executivo com IA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }

    .kpi-card {
        background: linear-gradient(135deg, #1c2333 0%, #161b28 100%);
        border: 1px solid #2a3352;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 8px;
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #9aa4c4;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f5f7ff;
    }
    .kpi-delta-up { color: #34d399; font-size: 0.85rem; font-weight: 600; }
    .kpi-delta-down { color: #f87171; font-size: 0.85rem; font-weight: 600; }
    .kpi-delta-neutral { color: #9aa4c4; font-size: 0.85rem; font-weight: 600; }

    .report-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .badge-gemini { background: #1e3a2f; color: #34d399; border: 1px solid #2f6f4e; }
    .badge-fallback { background: #2a2440; color: #c4b5fd; border: 1px solid #4c3f7a; }

    h1, h2, h3 { color: #f5f7ff; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================================================================
# FUNÇÕES DE CACHE (evita recarregar/reprocessar a cada interação)
# ==================================================================
@st.cache_data(show_spinner="Carregando base de vendas...")
def _cached_sales_data():
    return load_sales_data()


@st.cache_data(show_spinner="Consumindo API de cotações em tempo real...", ttl=600)
def _cached_crypto_data(coin_name: str, days: int):
    return load_crypto_data(coin_name, days)


def kpi_card(label: str, value: str, delta: str | None = None, delta_positive: bool | None = None):
    delta_html = ""
    if delta:
        css_class = (
            "kpi-delta-up" if delta_positive is True
            else "kpi-delta-down" if delta_positive is False
            else "kpi-delta-neutral"
        )
        delta_html = f'<div class="{css_class}">{delta}</div>'

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# SIDEBAR: FONTE DE DADOS + FILTROS
# ==================================================================
st.sidebar.title("📊 Insight Engine")
st.sidebar.caption("Dashboard Executivo com IA")
st.sidebar.divider()

dataset_choice = st.sidebar.radio(
    "Fonte de dados",
    options=["Vendas", "Criptomoedas"],
    help="Vendas: base sintética realista. Criptomoedas: dados reais via API pública CoinGecko.",
)

st.sidebar.divider()

gemini_key_input = st.sidebar.text_input(
    "Gemini API Key (opcional)",
    type="password",
    help="Se vazio, o relatório executivo é gerado pelo motor estatístico local (scikit-learn), sem custo.",
)

st.sidebar.divider()


# ==================================================================
# FLUXO: VENDAS
# ==================================================================
if dataset_choice == "Vendas":
    df_all = _cached_sales_data()

    min_date, max_date = df_all["date"].min().date(), df_all["date"].max().date()
    default_start = max(min_date, max_date - timedelta(days=90))

    date_range = st.sidebar.date_input(
        "Período",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    categories = st.sidebar.multiselect(
        "Categorias", options=sorted(df_all["category"].unique()), default=[]
    )
    regions = st.sidebar.multiselect(
        "Regiões", options=sorted(df_all["region"].unique()), default=[]
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, max_date

    mask = (df_all["date"].dt.date >= start_date) & (df_all["date"].dt.date <= end_date)
    df_filtered = df_all.loc[mask].copy()

    if categories:
        df_filtered = df_filtered[df_filtered["category"].isin(categories)]
    if regions:
        df_filtered = df_filtered[df_filtered["region"].isin(regions)]

    df_prev = previous_period_df(df_all, start_date, end_date)
    if categories:
        df_prev = df_prev[df_prev["category"].isin(categories)]
    if regions:
        df_prev = df_prev[df_prev["region"].isin(regions)]

    kpis = compute_sales_kpis(df_filtered, previous_df=df_prev)
    period_label = f"{start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}"

    # ---------- HEADER ----------
    st.title("📊 Dashboard Executivo de Vendas")
    st.caption(f"Período selecionado: **{period_label}**  •  {kpis['n_orders']} pedidos analisados")

    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados. Ajuste o período ou os filtros.")
    else:
        # ---------- KPI CARDS ----------
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Receita Total", f"R$ {kpis['total_revenue']:,.2f}",
                      delta=f"{kpis['revenue_growth_pct']:+.1f}% vs período anterior" if kpis['revenue_growth_pct'] is not None else None,
                      delta_positive=(kpis['revenue_growth_pct'] or 0) >= 0 if kpis['revenue_growth_pct'] is not None else None)
        with c2:
            kpi_card("Lucro Total", f"R$ {kpis['total_profit']:,.2f}", f"Margem: {kpis['margin_pct']:.1f}%",
                      delta_positive=kpis['margin_pct'] >= 20)
        with c3:
            kpi_card("Unidades Vendidas", f"{kpis['total_units']:,}")
        with c4:
            kpi_card("Ticket Médio", f"R$ {kpis['avg_ticket']:,.2f}")
        with c5:
            kpi_card("Categoria Líder", kpis['top_category'], f"R$ {kpis['top_category_revenue']:,.2f}")

        st.write("")

        # ---------- GRÁFICOS ----------
        col_left, col_right = st.columns([2, 1])

        with col_left:
            daily = df_filtered.groupby(df_filtered["date"].dt.date).agg(
                revenue=("revenue", "sum"), profit=("profit", "sum")
            ).reset_index()

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=daily["date"], y=daily["revenue"], name="Receita",
                mode="lines", line=dict(color="#818cf8", width=3), fill="tozeroy",
                fillcolor="rgba(129,140,248,0.12)",
            ))
            fig_line.add_trace(go.Scatter(
                x=daily["date"], y=daily["profit"], name="Lucro",
                mode="lines", line=dict(color="#34d399", width=2, dash="dot"),
            ))
            fig_line.update_layout(
                title="Receita e Lucro ao Longo do Tempo",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with col_right:
            rev_cat = kpis["revenue_by_category"].reset_index()
            rev_cat.columns = ["category", "revenue"]
            fig_pie = px.pie(
                rev_cat, names="category", values="revenue", hole=0.55,
                color_discrete_sequence=px.colors.sequential.Purples_r,
            )
            fig_pie.update_layout(
                title="Receita por Categoria",
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            rev_region = kpis["revenue_by_region"].reset_index()
            rev_region.columns = ["region", "revenue"]
            fig_bar_region = px.bar(
                rev_region, x="region", y="revenue", color="revenue",
                color_continuous_scale="Purples", text_auto=".2s",
            )
            fig_bar_region.update_layout(
                title="Receita por Região", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=340, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar_region, use_container_width=True)

        with col_b:
            top_products = (
                df_filtered.groupby("product")["revenue"].sum()
                .sort_values(ascending=True).tail(8).reset_index()
            )
            fig_bar_prod = px.bar(
                top_products, x="revenue", y="product", orientation="h",
                color="revenue", color_continuous_scale="Blues", text_auto=".2s",
            )
            fig_bar_prod.update_layout(
                title="Top Produtos por Receita", template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=340, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False,
            )
            st.plotly_chart(fig_bar_prod, use_container_width=True)

        with st.expander("🔍 Ver dados brutos filtrados"):
            st.dataframe(df_filtered, use_container_width=True)

    dataset_name_for_ai = "Vendas"
    df_for_ai = df_filtered
    kpis_for_ai = kpis

# ==================================================================
# FLUXO: CRIPTOMOEDAS
# ==================================================================
else:
    coin_name = st.sidebar.selectbox("Ativo", options=list(COIN_OPTIONS.keys()))
    days = st.sidebar.select_slider("Período (dias)", options=[7, 30, 90, 180, 365], value=90)

    df_crypto, fetch_error = _cached_crypto_data(coin_name, days)

    st.title(f"🪙 Dashboard Executivo — {coin_name}")
    st.caption(f"Últimos {days} dias  •  Fonte: API pública CoinGecko (dados reais, atualização automática)")

    if fetch_error == "rate_limit":
        st.error(
            "⏳ A API pública da CoinGecko atingiu o limite de requisições (HTTP 429) após múltiplas "
            "tentativas com backoff exponencial. Aguarde alguns instantes e recarregue a página."
        )
    elif fetch_error:
        st.error(f"❌ Não foi possível obter os dados da API no momento ({fetch_error}). Tente novamente em instantes.")

    if df_crypto.empty:
        st.warning("⚠️ Nenhum dado disponível para exibir no momento.")
        kpis = compute_crypto_kpis(df_crypto)
    else:
        kpis = compute_crypto_kpis(df_crypto)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Preço Atual", f"US$ {kpis['current_price']:,.2f}",
                      f"{kpis['period_change_pct']:+.2f}% no período",
                      delta_positive=kpis['period_change_pct'] >= 0)
        with c2:
            kpi_card("Máxima do Período", f"US$ {kpis['max_price']:,.2f}")
        with c3:
            kpi_card("Mínima do Período", f"US$ {kpis['min_price']:,.2f}")
        with c4:
            kpi_card("Volatilidade Diária", f"{kpis['volatility_pct']:.2f}%",
                      "Alta" if kpis['volatility_pct'] > 4 else "Moderada",
                      delta_positive=kpis['volatility_pct'] <= 4)

        st.write("")

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df_crypto["date"], y=df_crypto["price"], name="Preço (USD)",
            mode="lines", line=dict(color="#fbbf24", width=3), fill="tozeroy",
            fillcolor="rgba(251,191,36,0.10)",
        ))
        fig_price.update_layout(
            title=f"Preço de {coin_name} — {days} dias", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=400, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        fig_vol = px.bar(
            df_crypto, x="date", y="volume", color_discrete_sequence=["#818cf8"],
        )
        fig_vol.update_layout(
            title="Volume Negociado (USD)", template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            height=280, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

        with st.expander("🔍 Ver dados brutos"):
            st.dataframe(df_crypto, use_container_width=True)

    period_label = f"últimos {days} dias"
    dataset_name_for_ai = "Criptomoedas"
    df_for_ai = df_crypto
    kpis_for_ai = kpis


# ==================================================================
# AGENTE DE IA — RELATÓRIO EXECUTIVO
# ==================================================================
st.divider()
st.header("🤖 Relatório Executivo Automático")
st.caption("Gerado por IA a partir dos dados filtrados acima — mesma fonte de números do dashboard.")

if st.button("✨ Gerar Relatório com IA", type="primary", use_container_width=False):
    with st.spinner("Analisando dados e redigindo o relatório..."):
        report_md, source = generate_executive_summary(
            df=df_for_ai,
            kpis=kpis_for_ai,
            period_label=period_label,
            dataset_name=dataset_name_for_ai,
            api_key=gemini_key_input or None,
        )
    st.session_state["last_report"] = report_md
    st.session_state["last_report_source"] = source

if "last_report" in st.session_state:
    badge_class = "badge-gemini" if st.session_state["last_report_source"] == "gemini" else "badge-fallback"
    badge_text = "Gerado por Gemini API" if st.session_state["last_report_source"] == "gemini" else "Motor estatístico local (fallback)"
    st.markdown(f'<span class="report-badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    st.markdown(st.session_state["last_report"])

    st.download_button(
        "⬇️ Baixar Relatório (Markdown)",
        data=st.session_state["last_report"],
        file_name=f"relatorio_executivo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
    )
else:
    st.info("Clique no botão acima para gerar o relatório executivo com base nos dados e filtros atuais.")

st.divider()
st.caption("Insight Engine • Projeto de portfólio — Streamlit + Pandas + Plotly + IA (Gemini / scikit-learn fallback)")
