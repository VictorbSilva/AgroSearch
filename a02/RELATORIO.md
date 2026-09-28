# AgroSearch — Relatório Técnico

Laboratório Prático 04 · Tendências em Ciência da Computação · Prof. Me. Ricardo Roberto de Lima. **Equipe:** Victor (entrega individual).

## 1. Objetivo

Protótipo de motor de busca textual para os manuais técnicos da AgroTech Solutions: o técnico digita uma consulta e recebe os trechos mais relevantes, ranqueados por TF-IDF. Tudo em um único arquivo Streamlit (`agrosearch_app.py`), sem scikit-learn nem TfidfVectorizer: índice invertido, TF-IDF e stemmer foram implementados do zero com a biblioteca padrão.

## 2. Arquitetura

| Fase | Implementação |
|---|---|
| 1. Pré-processamento | tokenização (`\w+`), normalização (minúsculas e remoção de acentos via Unicode NFD), remoção de stopwords (lista própria em português) e stemming (stemmer próprio por sufixos). Checkboxes ligam/desligam stopwords e stemming; o vocabulário exibido muda na hora (47 termos sem nenhum filtro, 34 com os dois ligados). |
| 2. Índice invertido | dicionário termo → lista ordenada de IDs, construído a partir dos tokens limpos e exibido com `st.json` (34 termos; ex.: soj → [1, 2, 4]). |
| 3. Busca TF-IDF | a consulta passa pelo mesmo pipeline; para cada documento calcula TF, IDF e TF-IDF de cada termo e soma (TF-IDF acumulado). A tabela é ordenada do maior para o menor score e o vencedor recebe o troféu na coluna de destaque e uma mensagem verde. |

## 3. Fórmulas e decisões

- **TF(t, d)** = ocorrências de t em d / total de termos de d (slide 18 da Aula 3).
- **IDF(t)** = log10(N / df(t)), com N = 5; termo ausente da coleção vale 0. A base 10 reproduz os números dos slides 19 e 20 (log(3/2) ≈ 0,176).
- **TF-IDF acumulado** = soma do TF-IDF dos termos distintos da consulta no documento.
- **Stemmer:** remove o sufixo mais longo de uma lista (ex.: -ação, -mento, -idade, -as, -os, -s) desde que sobre um radical de pelo menos 3 letras; "lagartas" e "Lagartas" viram lagart, "irrigação" vira irrig.

## 4. Exemplo: consulta "irrigação da soja"

Termos após o pipeline: irrig, soj ("da" é stopword). IDF(irrig) = log10(5/2) = 0.3979, IDF(soj) = log10(5/3) = 0.2218.

| Posição | Documento | TF-IDF(irrig) | TF-IDF(soj) | TF-IDF acumulado |
|---|---|---|---|---|
| 1 | Doc 1 | 0.0497 | 0.0277 | 0.0775 |
| 2 | Doc 5 | 0.0568 | 0.0000 | 0.0568 |
| 3 | Doc 2 | 0.0000 | 0.0277 | 0.0277 |
| 4 | Doc 4 | 0.0000 | 0.0277 | 0.0277 |
| 5 | Doc 3 | 0.0000 | 0.0000 | 0.0000 |

Vencedor: **Doc 1**, único que tem os dois termos e o mais curto entre os que citam irrigação.

## 5. Importação de arquivos

A base pode ser ampliada ou substituída pela barra lateral com manuais em .txt, .md, .csv ou .pdf. Cada parágrafo vira um documento: num .txt sem linhas em branco, cada linha; no CSV, a coluna “texto”; no PDF, os parágrafos de cada página. Repetições e trechos com menos de 40 caracteres são ignorados, e cada documento guarda a origem (arquivo e página ou linha). A leitura usa `csv`, `io` e `pypdf` só para extrair o texto: o índice invertido e o TF-IDF continuam calculados do zero sobre a nova base, com N igual ao número de documentos.

## 6. Divisão de tarefas

| Tarefa | Responsável |
|---|---|
| Pipeline de pré-processamento e stemmer | Victor |
| Índice invertido e TF-IDF | Victor |
| Interface Streamlit | Victor |
| Importação de arquivos e relatório | Victor |
