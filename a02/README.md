# a02 — AgroSearch: Motor de Busca Inteligente

**Fonte:** `aulas/Desafio_Lab_AgroSearch.docx` (Laboratório Prático 04, seções 1 a 6; 0,5 ponto extra na AV1).

## Enunciado (resumo)

A AgroTech Solutions quer um protótipo de motor de busca textual para seus manuais técnicos: o técnico digita uma consulta e recebe os trechos mais relevantes, ranqueados. As regras são:

- **Entrega:** um único arquivo `.py` em Streamlit e um relatório em PDF de no máximo 2 páginas.
- **Proibido:** scikit-learn, `TfidfVectorizer` ou outra biblioteca de alto nível. O índice invertido e o TF-IDF devem ser implementados do zero.

O app integra três fases:

1. **Pipeline de pré-processamento** com as 4 etapas. Checkboxes ligam e desligam stemming e stopwords, e o vocabulário muda na tela.
2. **Índice invertido** (Termo → [IDs de Docs]), exibido com `st.json` ou `st.dataframe`.
3. **Busca TF-IDF:** para a query digitada, calcula TF, IDF e TF-IDF e mostra uma tabela ordenada do maior para o menor TF-IDF acumulado, destacando o documento vencedor.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `agrosearch_app.py` | O app, num único arquivo. Usa a biblioteca padrão, `pandas` e `streamlit`, e `pypdf` para ler os PDFs importados. |
| `relatorio.pdf` | Relatório técnico em 2 páginas: arquitetura, fórmulas, exemplo e divisão de tarefas. |
| `RELATORIO.md`, `gerar_relatorio.py`, `relatorio_pdf.py` | Fonte do relatório, com os números calculados pelo próprio app, e o gerador do PDF. |

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

Pipeline, índice invertido e TF-IDF passam a rodar sobre a base importada, com N igual ao número de documentos. A leitura dos arquivos usa `csv` e `io`, da biblioteca padrão, e `pypdf`, só para extrair o texto. O índice e as fórmulas continuam implementados do zero, como o desafio exige.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run agrosearch_app.py
python gerar_relatorio.py      # regera RELATORIO.md e relatorio.pdf
```

## Suposições

- **Base de documentos:** o app abre com os 5 documentos do “hardcode sugerido” da seção 5. A importação de arquivos permite trocar ou ampliar essa base com manuais reais.
- **Fórmulas:** o desafio não fixa as fórmulas. Usei TF(t, d) = ocorrências / total de termos de d e IDF(t) = log10(N / df(t)), como nos slides 18 a 20 da Aula 3. A base 10 é a que reproduz o exemplo do slide 20 (log(3/2) ≈ 0,176). Termo ausente da coleção tem IDF 0.
- **TF-IDF acumulado:** é a soma do TF-IDF dos termos distintos da consulta em cada documento. Empates são desempatados pelo menor ID.
- **Stemming:** como o desafio proíbe bibliotecas de alto nível, o stemmer é próprio. Ele remove o sufixo mais longo de uma lista de sufixos do português, desde que sobre um radical de pelo menos 3 letras. É um stemmer rudimentar, no espírito da `pratica02.py`, e não o RSLP completo.
- **Stopwords:** a lista em português é própria. A remoção de stopwords roda depois da normalização, então a lista não tem acentos.
- **Checkboxes:** ficam na barra lateral e valem para as três fases. Mudar os filtros refaz o vocabulário, o índice invertido e o ranking.
- **Equipe:** o desafio é para grupos de 2 ou 3 alunos. Esta entrega foi feita individualmente, por isso a divisão de tarefas do relatório tem um único responsável. Se o trabalho for em grupo, atualize a seção 6 de `gerar_relatorio.py` e rode o script de novo.
- **Bônus:** o desafio bônus (similaridade de cosseno) está na atividade a03.
