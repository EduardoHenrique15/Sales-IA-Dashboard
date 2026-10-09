"""
Página: sobre o projeto (o que faz, como funciona e com o que foi feito).
"""

import streamlit as st

from insight_engine.ui import branding
from insight_engine.ui.components import page_header

FEATURES = [
    (
        ":material/dashboard:",
        "Visão executiva",
        "KPIs com variação e minigráfico, meta de receita, filtros que viram link compartilhável e filtro "
        "por clique nos gráficos.",
    ),
    (
        ":material/waterfall_chart:",
        "Variação explicada",
        "A mudança da receita decomposta em volume, preço e mix (análise PVM), no total e por categoria.",
    ),
    (
        ":material/trending_up:",
        "Previsão com backtesting",
        "Vários modelos testados em janelas que eles não viram; vence o menor WAPE e os intervalos vêm dos "
        "erros reais.",
    ),
    (
        ":material/crisis_alert:",
        "Detecção de anomalias",
        "Decomposição STL em escala log, z-score robusto (MAD) e um piso de materialidade para evitar alarmes falsos.",
    ),
    (
        ":material/groups:",
        "Segmentação de clientes",
        "Segmentos RFM com ação sugerida, comparados a grupos do K-Means escolhidos pelo coeficiente de silhueta.",
    ),
    (
        ":material/auto_awesome:",
        "IA que mostra o trabalho",
        "Relatório com saída estruturada e chat com chamada de funções; todo número citado é conferido contra "
        "os dados.",
    ),
]

STACK = ["Python", "Streamlit", "pandas", "Plotly", "statsmodels", "scikit-learn", "SciPy", "pandera", "Pydantic"]
STACK_AI = ["Google Gemini", "google-genai", "fpdf2"]
STACK_QUALITY = ["pytest", "ruff", "mypy", "GitHub Actions"]

ARCHITECTURE = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    nodesep=0.35;
    node [shape=box, style="rounded,filled", fillcolor="#3987e5", color="#3987e5", fontcolor="white",
          fontname="Inter, Helvetica, sans-serif", fontsize=13, margin="0.2,0.12"];
    edge [color="#898781", fontcolor="#898781", fontname="Inter, Helvetica, sans-serif", fontsize=11];

    dados [label="Base de vendas\\nvalidada (pandera)"];
    analises [label="Análises\\nKPIs · PVM · previsão\\nanomalias · RFM"];
    painel [label="Dashboard\\n(Streamlit + Plotly)"];
    llm [label="Gemini\\nsaída estruturada\\nchamada de funções"];
    local [label="Motor estatístico\\nlocal", fillcolor="#6b6b66", color="#6b6b66"];
    relatorio [label="Relatório e chat\\ncom números conferidos"];

    dados -> analises;
    analises -> painel;
    analises -> llm [label="fatos calculados"];
    analises -> local [label="sem chave"];
    llm -> relatorio;
    local -> relatorio;
}
"""

page_header("Sobre o projeto", f"{branding.TAGLINE} · por {branding.AUTHOR}", icon=":material/info:")

with st.container(border=True):
    st.markdown(
        f"O **{branding.APP_NAME}** transforma uma planilha de vendas em um painel executivo: indicadores, "
        "variação explicada, previsão, anomalias e segmentação de clientes, com um relatório escrito por IA "
        "em que cada número citado é conferido contra os dados. Funciona com uma base sintética (com anomalias "
        "plantadas para validar os métodos), com 97 mil pedidos reais de e-commerce brasileiro (Olist) ou com a "
        "planilha do usuário. Sem chave de IA, tudo continua funcionando com um motor estatístico local."
    )
    with st.container(horizontal=True):
        st.link_button("Código no GitHub", branding.GITHUB_URL, icon=":material/code:", type="primary")
        st.page_link("app_pages/sales.py", label="Abrir o dashboard", icon=":material/dashboard:")
        st.page_link("app_pages/upload.py", label="Usar meus dados", icon=":material/upload_file:")

st.header("O que ele faz", icon=":material/apps:", anchor=False)
for start in range(0, len(FEATURES), 3):
    for column, (icon, title, text) in zip(st.columns(3), FEATURES[start : start + 3], strict=True):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {title}**")
            st.caption(text)

st.header("Como funciona", icon=":material/account_tree:", anchor=False)
st.caption(
    "Os números vêm sempre do mesmo cálculo: o dashboard, o relatório e o chat leem os mesmos fatos. A IA só "
    "redige e escolhe o que destacar; ela não calcula nem executa código."
)
st.graphviz_chart(ARCHITECTURE, width="stretch")

st.header("Tecnologias", icon=":material/construction:", anchor=False)
for title, items, color in (
    ("Dados e ciência de dados", STACK, "blue"),
    ("IA e relatórios", STACK_AI, "violet"),
    ("Qualidade", STACK_QUALITY, "green"),
):
    st.caption(title)
    with st.container(horizontal=True, gap="small"):
        for item in items:
            st.badge(item, color=color)  # type: ignore[arg-type]

st.header("Qualidade e privacidade", icon=":material/verified:", anchor=False)
col_quality, col_privacy = st.columns(2)
with col_quality, st.container(border=True, height="stretch"):
    st.markdown("**:material/task_alt: Engenharia**")
    st.markdown(
        "- Testes automatizados com cobertura mínima de 90%, incluindo a interface (AppTest)\n"
        "- Integração contínua em Python 3.11 e 3.12, com ruff e mypy\n"
        "- Testes nunca usam chaves reais nem acessam a internet"
    )
with col_privacy, st.container(border=True, height="stretch"):
    st.markdown("**:material/lock: Privacidade**")
    st.markdown(
        "- Arquivos enviados ficam só na sessão do navegador\n"
        "- A chave do Gemini digitada não é salva\n"
        "- Uso da chave do projeto tem limite por sessão e global"
    )

st.caption(f"{branding.APP_NAME} v{branding.VERSION} · {branding.AUTHOR}")
