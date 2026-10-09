# Resultados da avaliação da IA

Gerado em 09/10/2026 com `python -m evals.run`, na base de exemplo, com 25 perguntas ([método](README.md)).

## Chat com os dados

| Modelo | Acerto | Números corretos | Sem alucinação | Consulta certa | Latência (mediana) | Tokens por pergunta |
|---|---|---|---|---|---|---|
| `gemini-flash-lite-latest` | **100% (23/23)** | 100% (14/14) | 100% (23/23) | 100% (23/23) | 1,7 s | 3.864 |

Perguntas com erro da API (fora das taxas): `gemini-flash-lite-latest` 2

### Acerto por tipo de pergunta

| Tipo | `gemini-flash-lite-latest` |
|---|---|
| valores | 100% (8/8) |
| rankings | 100% (2/2) |
| tempo | 100% (3/3) |
| variação | 100% (2/2) |
| anomalias | 100% (1/1) |
| previsão | 100% (1/1) |
| clientes | 100% (2/2) |
| robustez | 100% (4/4) |

## Relatório executivo

| Modelo | Relatórios no formato certo | Números conferidos | Latência média | Erros da API |
|---|---|---|---|---|
| `gemini-flash-lite-latest` | 3 de 3 | 56 de 56 (100,0%) | 2,1 s | 0 |

## Respostas que não passaram

Nenhuma.
