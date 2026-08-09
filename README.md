# 📊 Insight Engine — Dashboard Executivo com IA

**Transforme planilhas em decisões, em segundos.** Um dashboard interativo que consome dados automaticamente (vendas ou cotações de criptomoedas em tempo real), calcula KPIs de negócio e usa um **Agente de IA** para gerar, sozinho, um relatório executivo em linguagem natural — com diagnóstico de gargalos e plano de ação sugerido.

Construído com **Streamlit + Pandas + Plotly + Google Gemini / scikit-learn**.

---

## 🎯 O problema

Empresas e profissionais perdem horas todos os meses analisando planilhas manualmente para entender tendências de vendas, mercado ou performance financeira. O dado existe — o que falta é **tempo e contexto** para transformá-lo em decisão.

## 💡 A solução

O **Insight Engine** automatiza esse pipeline inteiro:

`Dados brutos → Métricas de negócio → Visualização interativa → Narrativa em linguagem natural`

Em poucos cliques, qualquer pessoa (não só analistas de dados) filtra o período/categoria que interessa e recebe um **relatório executivo pronto**, com destaques, riscos e próximos passos sugeridos — como se tivesse um analista sênior de plantão 24/7.

---

## ✨ Principais funcionalidades

- **KPI Cards dinâmicos** — receita, lucro, margem, ticket médio, crescimento vs. período anterior, tudo recalculado em tempo real conforme os filtros.
- **Gráficos interativos (Plotly)** — série temporal de receita/lucro, distribuição por categoria, ranking por região e top produtos.
- **Duas fontes de dados plugáveis:**
  - 📦 **Vendas** — base sintética realista (sazonalidade, tendência, ruído) gerada e cacheada localmente.
  - 🪙 **Criptomoedas** — dados **reais**, consumidos automaticamente da API pública da CoinGecko (preço, volume, volatilidade).
- **Agente de IA com fallback resiliente** — usa a API do Google Gemini quando disponível; se a chave não existir, a cota estourar (HTTP 429) ou a API falhar, o sistema recorre automaticamente a um **motor estatístico local** (regressão linear via scikit-learn + regras de negócio) para gerar o mesmo relatório, sem quebrar a experiência do usuário.
- **Relatório Executivo estruturado** em Markdown: Destaques do Período, Diagnóstico de Pontos Críticos e Plano de Ação Estratégico — exportável em `.md`.
- **Tratamento de erros de produção** — dados nulos, rate limit de API, timeouts de rede e ausência de dados no filtro são todos tratados sem quebrar a interface.

---

## 🏗️ Arquitetura

```
sales-ai-dashboard/
│
├── app.py              # Orquestração da UI (Streamlit) — filtros, KPI cards, gráficos
├── data_loader.py       # Camada de dados: geração/cache de vendas + consumo da API CoinGecko
├── kpi_engine.py         # Cálculo puro de métricas de negócio (reaproveitado pela IA)
├── ai_agent.py            # Agente de IA: prompt + chamada Gemini + fallback estatístico
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml        # Tema visual dark
└── data/
    └── sales_data.csv     # Cache gerado automaticamente na primeira execução
```

**Por que essa separação importa:** cada módulo tem uma única responsabilidade. `kpi_engine.py` é a **fonte única da verdade** dos números — tanto os cards visuais quanto o prompt de IA consomem exatamente os mesmos valores, garantindo que o texto gerado nunca contradiga o gráfico. Isso facilita testes unitários, manutenção e a troca de qualquer camada (ex: trocar Gemini por outro LLM, ou CSV por um banco de dados) sem tocar no restante do sistema.

### Fluxo de decisão do Agente de IA

```
                 ┌───────────────────────┐
                 │  Existe GEMINI_API_KEY? │
                 └──────────┬────────────┘
                       sim  │   não
              ┌─────────────┘   └──────────────┐
              ▼                                  ▼
   ┌─────────────────────┐          ┌──────────────────────────┐
   │  Chama Gemini API    │          │  Motor estatístico local  │
   │  (com retry/backoff  │          │  (regressão linear +       │
   │   em caso de 429)    │          │   regras de negócio)       │
   └──────────┬───────────┘          └──────────────┬────────────┘
              │  falhou após retries                 │
              └────────────────►  fallback  ◄─────────┘
                                     │
                                     ▼
                    Relatório Executivo em Markdown
                 (mesma estrutura, independente da fonte)
```

---

## 🚀 Como rodar localmente

```bash
# 1. Clone o repositório
git clone https://github.com/EduardoHenrique15/insight-engine.git
cd insight-engine

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Configure sua chave do Gemini para relatórios via IA generativa
cp .env.example .env
# edite o .env e cole sua GEMINI_API_KEY
# sem chave, o app funciona normalmente com o fallback estatístico

# 5. Rode o dashboard
streamlit run app.py
```

O app abre em `http://localhost:8501`. Na primeira execução, a base de vendas sintética é gerada e cacheada automaticamente em `data/sales_data.csv`.

> 🔑 Obtenha uma chave gratuita do Gemini em [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## 📈 Impacto de negócio

| Antes | Com o Insight Engine |
|---|---|
| Horas analisando planilhas manualmente | Insights em segundos, com um clique |
| Relatórios executivos escritos manualmente | Relatório gerado automaticamente por IA |
| Análise depende de um analista disponível | Qualquer stakeholder autoatende via dashboard |
| Risco de não identificar quedas a tempo | Diagnóstico automático de gargalos por categoria/região |
| Decisão baseada em "achismo" | Plano de ação sugerido com base em dados reais |

Esse tipo de automação é o que separa equipes que **reagem** a problemas das que **antecipam** — e é exatamente o tipo de ferramenta que squads de dados, produto e growth usam para escalar decisões sem escalar headcount.

---

## 🧠 Stack técnica

`Python` · `Streamlit` · `Pandas` · `NumPy` · `Plotly` · `scikit-learn` · `Google Gemini API` · `Requests` (integração com API pública CoinGecko)

---

## 🔮 Possíveis evoluções

- Autenticação multiusuário e persistência em banco de dados (PostgreSQL/Supabase).
- Agendamento automático de relatórios por e-mail (diário/semanal).
- Comparação de múltiplos ativos/categorias lado a lado.
- Deploy em Streamlit Community Cloud com link público de demonstração.


