"""
Página: Dashboard Executivo de Criptomoedas (dados reais da CoinGecko).
"""

import pandas as pd
import streamlit as st

from insight_engine.ai.context import build_crypto_facts
from insight_engine.analytics.kpis import compute_crypto_kpis
from insight_engine.data.crypto import COIN_OPTIONS, CryptoDataError
from insight_engine.formatting import format_pct, format_usd
from insight_engine.ui import cached, charts
from insight_engine.ui.components import kpi_card, report_section

# Acima desse desvio padrão diário dos retornos, a volatilidade é considerada alta
HIGH_VOLATILITY_PCT = 4

# ---------- FILTROS (barra lateral) ----------
coin_name = st.sidebar.selectbox("Ativo", options=list(COIN_OPTIONS.keys()))
days = st.sidebar.select_slider("Período (dias)", options=[7, 30, 90, 180, 365], value=90)

# ---------- CABEÇALHO ----------
st.title(f"🪙 Dashboard Executivo — {coin_name}")
st.caption(f"Últimos {days} dias  •  Fonte: API pública CoinGecko (dados reais, atualização automática)")

try:
    df_crypto = cached.crypto_data(coin_name, days)
except CryptoDataError as exc:
    df_crypto = pd.DataFrame()
    if exc.kind == "rate_limit":
        st.error(
            "⏳ A API pública da CoinGecko atingiu o limite de requisições (HTTP 429) após múltiplas "
            "tentativas com backoff exponencial. Aguarde alguns instantes e recarregue a página."
        )
    else:
        st.error(f"❌ Não foi possível obter os dados da API no momento ({exc.kind}). Tente novamente em instantes.")

kpis = compute_crypto_kpis(df_crypto)

if df_crypto.empty:
    st.warning("⚠️ Nenhum dado disponível para exibir no momento.")
else:
    # ---------- KPI CARDS ----------
    high_volatility = kpis.volatility_pct > HIGH_VOLATILITY_PCT
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Preço Atual",
            format_usd(kpis.current_price),
            f"{format_pct(kpis.period_change_pct, 2, signed=True)} no período",
            delta_positive=kpis.period_change_pct >= 0,
        )
    with c2:
        kpi_card("Máxima do Período", format_usd(kpis.max_price))
    with c3:
        kpi_card("Mínima do Período", format_usd(kpis.min_price))
    with c4:
        kpi_card(
            "Volatilidade Diária",
            format_pct(kpis.volatility_pct, 2),
            "Alta" if high_volatility else "Moderada",
            delta_positive=not high_volatility,
        )

    st.write("")

    # ---------- GRÁFICOS ----------
    st.plotly_chart(charts.price_over_time(df_crypto, coin_name, days))
    st.plotly_chart(charts.traded_volume(df_crypto))

    with st.expander("🔍 Ver dados brutos"):
        st.dataframe(df_crypto)

period_label = f"últimos {days} dias"
report_section(
    lambda: build_crypto_facts(df_crypto, kpis, period_label, coin_name) if not df_crypto.empty else None,
    period_label=period_label,
    dataset_name="Criptomoedas",
    context=coin_name,
)
