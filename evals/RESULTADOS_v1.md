# Resultados da avaliação da IA

Gerado em 08/10/2026 com `python -m evals.run`, na base de exemplo, com 25 perguntas ([método](README.md)).

## Chat com os dados

| Modelo | Acerto | Números corretos | Sem alucinação | Consulta certa | Latência (mediana) | Tokens por pergunta |
|---|---|---|---|---|---|---|
| `gemini-2.5-flash` | **—** | — | — | — | — s | — |
| `gemini-flash-lite-latest` | **88% (21/24)** | 93% (13/14) | 96% (23/24) | 100% (24/24) | 1,7 s | 3.725 |

Perguntas com erro da API (fora das taxas): `gemini-2.5-flash` 25, `gemini-flash-lite-latest` 1

### Acerto por tipo de pergunta

| Tipo | `gemini-2.5-flash` | `gemini-flash-lite-latest` |
|---|---|---|
| valores | — | 100% (8/8) |
| rankings | — | 100% (2/2) |
| tempo | — | 67% (2/3) |
| variação | — | 50% (1/2) |
| anomalias | — | 50% (1/2) |
| previsão | — | 100% (1/1) |
| clientes | — | 100% (2/2) |
| robustez | — | 100% (4/4) |

## Relatório executivo

| Modelo | Relatórios no formato certo | Números conferidos | Latência média |
|---|---|---|---|
| `gemini-2.5-flash` | 0 de 3 | 0 de 0 (0,0%) | 0,4 s |
| `gemini-flash-lite-latest` | 3 de 3 | 58 de 58 (100,0%) | 2,3 s |

## Respostas que não passaram

- `gemini-flash-lite-latest` · **Qual foi a receita de ontem?** — faltou receita (60.897,4)
- `gemini-flash-lite-latest` · **Por que a receita mudou no 4º trimestre de 2025?** — número sem fonte: +33,25%
- `gemini-flash-lite-latest` · **Houve algum dia atípico em 2024?** — faltou 15/03 ou 15 de março; faltou 07/08 ou 7 de agosto
