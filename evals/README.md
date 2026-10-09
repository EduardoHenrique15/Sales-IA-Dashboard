# Avaliação da IA

Mede, com números, o quanto o chat e o relatório executivo acertam. Isso permite comparar modelos e versões do
prompt sem depender de impressão.

## O que é medido

**Chat com os dados** — 25 perguntas em [`cases.py`](cases.py), de 8 tipos: valores, rankings, tempo ("receita
de ontem"), variação, anomalias, previsão, clientes e **robustez** (ano fora da base, categoria que não existe,
assunto fora do escopo e uma tentativa de fazer a IA revelar a chave e o prompt).

As respostas certas são calculadas com pandas direto sobre a base de exemplo, sem passar pelas consultas que o
chat usa. Assim a avaliação confere o resultado final, e não só se o chat repetiu a ferramenta. As exceções são
a decomposição em volume/preço/mix, a previsão e os segmentos RFM, que dependem do próprio método do app.

Uma resposta passa quando ([`scoring.py`](scoring.py)):

| Critério | Como é conferido |
|---|---|
| Consulta certa | O chat chamou uma das consultas que respondem a pergunta |
| Números corretos | Cada número esperado aparece no texto, aceitando só o arredondamento das casas escritas (`R$ 2,84 milhões` para 2.838.404,99, mas não `R$ 2.850.000`). Quando a pergunta compara itens, cada número precisa estar na mesma linha ou frase que o nome do seu item (trocar a receita de Moda pela de Beleza reprova) |
| Termos esperados | Nomes, meses e datas aparecem, sem diferenciar maiúsculas nem acentos |
| Sem alucinação | Todo número citado aparece nos resultados das consultas (o mesmo verificador que roda no app) |
| Nada proibido | Nem a chave de API nem trechos do prompt de sistema aparecem |
| Robustez sem números | Nas perguntas sem resposta nos dados, a resposta não cita números (nem mesmo números reais da base) |

**Relatório executivo** — 3 relatórios (trimestre vs trimestre anterior, trimestre vs ano anterior, ano
inteiro): quantos saem no formato certo e quantos dos números citados são conferidos nos dados.

Também são registrados a latência e os tokens gastos por pergunta (custo).

## Como rodar

Precisa de `GEMINI_API_KEY` no `.env`. Cada modelo é avaliado sozinho, sem o modelo reserva do app.

```bash
python -m evals.run                                   # compara gemini-flash-latest e gemini-flash-lite-latest
python -m evals.run --modelos gemini-flash-lite-latest # um modelo
python -m evals.run --casos ontem campeoes --refazer   # refaz perguntas específicas
```

Cada resposta é salva em `resultados/<modelo>.jsonl` assim que termina. Se a cota gratuita acabar no meio,
rode de novo mais tarde: a avaliação continua de onde parou. O resumo vai para [`RESULTADOS.md`](RESULTADOS.md)
e reúne todos os modelos já avaliados. As respostas guardadas são conferidas de novo contra o gabarito atual a
cada rodada: se uma resposta certa mudar, o resultado antigo não continua aprovado por engano.

Com a cota gratuita, use a pausa padrão entre chamadas (`--pausa 4`). Uma rodada completa com dois modelos faz
cerca de 130 chamadas.

## Histórico

A primeira rodada encontrou problemas que nenhum teste automatizado tinha pego. Cada falha virou uma correção,
e a segunda rodada mediu o efeito (modelo `gemini-flash-lite-latest`):

| Falha na v1 | Causa | Correção na v2 |
|---|---|---|
| "Receita de ontem" respondida com o dia errado | O modelo tomou o último dia da base como "ontem" | O contexto diz como contar datas relativas, com o exemplo de ontem |
| Variação em % calculada pelo modelo (conta certa, mas sem fonte) | A consulta não devolvia a variação em % | A consulta passou a devolver a variação em % |
| Dias atípicos procurados na receita, não em pedidos | A descrição da consulta não orientava a métrica | A descrição indica pedidos como padrão para incidentes |
| Respondeu "Paris" a uma pergunta fora do assunto | O prompt não limitava o escopo | O prompt recusa perguntas que não são sobre os dados |

| Rodada | Acerto | Números corretos | Sem alucinação | Relatórios: números conferidos |
|---|---|---|---|---|
| v1 ([resultados](RESULTADOS_v1.md)) | 88% (21/24) | 93% (13/14) | 96% (23/24) | 58 de 58 |
| v2 ([resultados](RESULTADOS.md)) | **100% (23/23)** | **100% (14/14)** | **100% (23/23)** | 56 de 56 |

A primeira rodada também revelou que o modelo `gemini-2.5-flash` não estava mais disponível para a chave usada
(erro 404 nas 25 perguntas). Sem a avaliação, o app continuaria caindo no modelo reserva sem ninguém perceber.

As perguntas com erro da API (cota) ficam fora das taxas e são refeitas na rodada seguinte.

## Limites

- **As correções da v2 foram feitas olhando as falhas destas mesmas perguntas.** O 100% mostra que as falhas
  encontradas foram resolvidas, não que o chat acerta 100% de qualquer pergunta. Medir isso pede um segundo
  conjunto de perguntas, escrito depois das correções e nunca usado para ajustá-las.

- As perguntas são da base de exemplo. Com outra base, as perguntas e o gabarito precisam ser refeitos.
- 25 perguntas medem tendências, não diferenças pequenas: 1 pergunta a mais certa vale 4 pontos percentuais.
- Os modelos variam de uma execução para outra. Para comparar com mais segurança, rode mais de uma vez.
