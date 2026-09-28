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

## Importação de arquivos

A base não precisa ficar presa às 5 frases do exemplo. Na barra lateral, **📂 Importar documentos** aceita um ou vários manuais em `.txt`, `.md`, `.csv` ou `.pdf`:

| Formato | Como vira documento |
|---|---|
| `.txt`, `.md` | Cada parágrafo (bloco separado por linha em branco). Sem linhas em branco, cada linha, como a base sugerida. |
| `.csv` | Cada linha da coluna `texto` (ou `conteudo`, `trecho`, `documento`, `text`, `content`), separada por vírgula, ponto e vírgula ou tab. Um CSV de uma coluna só é lido inteiro. |
| `.pdf` | Os parágrafos de cada página (texto extraído com `pypdf`). Como o PDF raramente marca parágrafos com linha em branco, uma linha bem mais curta que as outras (a última do parágrafo) fecha o parágrafo. |

- **Adicionar à base** acrescenta os trechos depois do maior ID atual. **Substituir a base** recomeça a numeração em 1.
- **↩️ Base original** volta aos 5 documentos sugeridos.
- Trechos repetidos e trechos com menos de 40 caracteres (títulos soltos, números de página) são ignorados.
- Cada documento guarda a origem (arquivo e página ou linha), mostrada na base, no pipeline e no documento vencedor.
- Um arquivo que não pode ser lido (formato não aceito, CSV sem coluna de texto, PDF escaneado) aparece com o motivo na barra lateral, e a base não muda.

Pipeline, índice invertido, TF-IDF e ranking por cosseno passam a rodar sobre a base importada, com N igual ao número de documentos. A leitura dos arquivos usa `csv` e `io`, da biblioteca padrão, e `pypdf`, só para extrair o texto. O índice e as fórmulas continuam implementados do zero, como o desafio exige.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run agrosearch_app.py
```

## Suposições

- **Vetor da consulta:** o TF é calculado sobre os tokens da própria consulta (ocorrências / total de tokens da consulta), e o IDF é o da coleção de documentos (log10(N/df), o mesmo da a02). Assim, repetir um termo na consulta aumenta o peso dele, o que o TF-IDF acumulado não faz.
- **Termos fora do vocabulário:** um termo que não aparece em nenhum documento não tem dimensão no espaço vetorial e é ignorado. Se nenhum termo da consulta existir, o vetor é nulo e todos os cossenos valem 0; o app avisa.
- **Restrição técnica:** foi mantida a da a02 (sem scikit-learn nem biblioteca de alto nível). O desafio não repete a regra para o bônus, mas ela vale para o AgroSearch como um todo.
- **Resultado nesta base:** com 5 frases curtas, o cosseno e o TF-IDF acumulado costumam concordar no 1º lugar, porque o TF da aula já divide pelo tamanho do documento. O cosseno difere em dois pontos: normaliza pela norma dos vetores, que depende de todos os termos do documento, e respeita o peso dos termos repetidos na consulta. Ele passa a fazer diferença em bases com documentos longos e variados.
