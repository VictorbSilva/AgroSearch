# a03 — AgroSearch: Bônus de Similaridade de Cosseno

**Fonte:** `aulas/Desafio_Lab_AgroSearch.docx`, seção 7 (Desafio Bônus / Nota Extra).

## Enunciado

> Implementar a Similaridade de Cosseno entre o vetor da Query e o vetor TF-IDF dos documentos para lidar melhor com consultas de múltiplas palavras.

## O que foi feito

`agrosearch_app.py` é o app completo do AgroSearch da atividade a02 (pipeline, índice invertido e busca TF-IDF), com uma quarta aba: **⭐ Bônus: Similaridade de Cosseno**. A pasta é autossuficiente, então o app inteiro está aqui.

| Função nova | O que faz |
|---|---|
| `vetor_tfidf(tokens, idfs)` | Vetor TF-IDF esparso no vocabulário da coleção: w(t) = TF(t) · IDF(t). Termos que não existem na coleção ficam de fora. |
| `cosseno(a, b)` | cos(θ) = (a · b) / (‖a‖ · ‖b‖); vale 0 se algum vetor for nulo. |
| `ranquear_cosseno(consulta, documentos, …)` | Monta o vetor da query (TF da query × IDF da coleção) e o de cada documento, e ordena pelo cosseno. |

A aba bônus usa a mesma consulta da aba 3 e mostra três coisas: o vetor TF-IDF da consulta; o ranking por cosseno, do maior para o menor, com a posição que cada documento teria no TF-IDF acumulado; e o documento mais similar em destaque. Tudo segue do zero, sem scikit-learn, como na a02.

Exemplos com consultas de várias palavras:

| Consulta | 1º por cosseno | Cosseno |
|---|---|---|
| controle biológico de lagartas | Doc 2 | 0,601 |
| lagartas na soja e no algodão | Doc 4 | 0,471 |
| irrigação por gotejamento na soja | Doc 5 | 0,441 |
| irrigação da soja | Doc 1 | 0,257 |

## Como rodar

```bash
pip install -r requirements.txt
streamlit run agrosearch_app.py
python -m pytest -q      # 33 testes: os do app da a02 + os do cosseno (tests/test_cosseno.py)
```

## Suposições

- **Vetor da consulta:** o TF é calculado sobre os tokens da própria consulta (ocorrências / total de tokens da consulta), e o IDF é o da coleção de documentos (log10(N/df), o mesmo da a02). Assim, repetir um termo na consulta aumenta o peso dele, o que o TF-IDF acumulado não faz.
- **Termos fora do vocabulário:** um termo que não aparece em nenhum documento não tem dimensão no espaço vetorial e é ignorado. Se nenhum termo da consulta existir, o vetor é nulo e todos os cossenos valem 0; o app avisa.
- **Restrição técnica:** foi mantida a da a02 (sem scikit-learn nem biblioteca de alto nível). O desafio não repete a regra para o bônus, mas ela vale para o AgroSearch como um todo.
- **Resultado nesta base:** com 5 frases curtas, o cosseno e o TF-IDF acumulado costumam concordar no 1º lugar, porque o TF da aula já divide pelo tamanho do documento. O cosseno difere em dois pontos: normaliza pela norma dos vetores, que depende de todos os termos do documento, e respeita o peso dos termos repetidos na consulta. Ele passa a fazer diferença em bases com documentos longos e variados.
