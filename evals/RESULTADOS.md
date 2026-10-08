# Resultados da avaliação da IA

Gerado em 08/10/2026 com `python -m evals.run`, na base de exemplo, com 25 perguntas ([método](README.md)).

## Chat com os dados

| Modelo | Acerto | Números corretos | Sem alucinação | Consulta certa | Latência (mediana) | Tokens por pergunta |
|---|---|---|---|---|---|---|
| `gemini-flash-latest` | **—** | — | — | — | — s | — |
| `gemini-flash-lite-latest` | **100% (23/23)** | 100% (14/14) | 100% (23/23) | 100% (23/23) | 1,8 s | 3.864 |

Perguntas com erro da API (fora das taxas): `gemini-flash-latest` 25, `gemini-flash-lite-latest` 2

### Acerto por tipo de pergunta

| Tipo | `gemini-flash-latest` | `gemini-flash-lite-latest` |
|---|---|---|
| valores | — | 100% (8/8) |
| rankings | — | 100% (2/2) |
| tempo | — | 100% (3/3) |
| variação | — | 100% (2/2) |
| anomalias | — | 100% (1/1) |
| previsão | — | 100% (1/1) |
| clientes | — | 100% (2/2) |
| robustez | — | 100% (4/4) |

## Relatório executivo

| Modelo | Relatórios no formato certo | Números conferidos | Latência média | Erros da API |
|---|---|---|---|---|
| `gemini-flash-latest` | 0 de 0 | 0 de 0 (0,0%) | — s | 3 |
| `gemini-flash-lite-latest` | 3 de 3 | 56 de 56 (100,0%) | 2,1 s | 0 |

## Respostas que não passaram

Nenhuma.
