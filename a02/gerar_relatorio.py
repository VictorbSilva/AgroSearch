"""Gera RELATORIO.md e relatorio.pdf (máx. 2 páginas) com os números calculados pelo próprio app.

Uso: python gerar_relatorio.py
"""

from datetime import datetime, timezone
from pathlib import Path

from agrosearch_app import (
    DOCUMENTOS,
    TAMANHO_MINIMO,
    indice_invertido,
    preprocessar,
    ranquear,
    vocabulario,
)
from relatorio_pdf import gerar_pdf

PASTA = Path(__file__).resolve().parent
CONSULTA = "irrigação da soja"


def montar_markdown() -> str:
    r = ranquear(CONSULTA, DOCUMENTOS)
    indice = indice_invertido({d: preprocessar(t) for d, t in DOCUMENTOS.items()})
    linhas_rank = "\n".join(
        f"| {i} | Doc {x['doc']} | " + " | ".join(f"{x['tfidf'][t]:.4f}" for t in r["termos"]) + f" | {x['score']:.4f} |"
        for i, x in enumerate(r["linhas"], 1)
    )
    cab_termos = " | ".join(f"TF-IDF({t})" for t in r["termos"])
    sep_termos = "|".join("---" for _ in r["termos"])
    idfs = ", ".join(f"IDF({t}) = log10(5/{r['df'][t]}) = {r['idf'][t]:.4f}" for t in r["termos"])
    return f"""# AgroSearch — Relatório Técnico

Laboratório Prático 04 · Tendências em Ciência da Computação · Prof. Me. Ricardo Roberto de Lima. **Equipe:** Victor (entrega individual).

## 1. Objetivo

Protótipo de motor de busca textual para os manuais técnicos da AgroTech Solutions: o técnico digita uma consulta e recebe os trechos mais relevantes, ranqueados por TF-IDF. Tudo em um único arquivo Streamlit (`agrosearch_app.py`), sem scikit-learn nem TfidfVectorizer: índice invertido, TF-IDF e stemmer foram implementados do zero com a biblioteca padrão.

## 2. Arquitetura

| Fase | Implementação |
|---|---|
| 1. Pré-processamento | tokenização (`\\w+`), normalização (minúsculas e remoção de acentos via Unicode NFD), remoção de stopwords (lista própria em português) e stemming (stemmer próprio por sufixos). Checkboxes ligam/desligam stopwords e stemming; o vocabulário exibido muda na hora ({len(vocabulario(DOCUMENTOS, False, False))} termos sem nenhum filtro, {len(vocabulario(DOCUMENTOS))} com os dois ligados). |
| 2. Índice invertido | dicionário termo → lista ordenada de IDs, construído a partir dos tokens limpos e exibido com `st.json` ({len(indice)} termos; ex.: soj → {indice['soj']}). |
| 3. Busca TF-IDF | a consulta passa pelo mesmo pipeline; para cada documento calcula TF, IDF e TF-IDF de cada termo e soma (TF-IDF acumulado). A tabela é ordenada do maior para o menor score e o vencedor recebe o troféu na coluna de destaque e uma mensagem verde. |

## 3. Fórmulas e decisões

- **TF(t, d)** = ocorrências de t em d / total de termos de d (slide 18 da Aula 3).
- **IDF(t)** = log10(N / df(t)), com N = 5; termo ausente da coleção vale 0. A base 10 reproduz os números dos slides 19 e 20 (log(3/2) ≈ 0,176).
- **TF-IDF acumulado** = soma do TF-IDF dos termos distintos da consulta no documento.
- **Stemmer:** remove o sufixo mais longo de uma lista (ex.: -ação, -mento, -idade, -as, -os, -s) desde que sobre um radical de pelo menos 3 letras; "lagartas" e "Lagartas" viram lagart, "irrigação" vira irrig.

## 4. Exemplo: consulta "{CONSULTA}"

Termos após o pipeline: {", ".join(r["termos"])} ("da" é stopword). {idfs}.

| Posição | Documento | {cab_termos} | TF-IDF acumulado |
|---|---|{sep_termos}|---|
{linhas_rank}

Vencedor: **Doc {r['linhas'][0]['doc']}**, único que tem os dois termos e o mais curto entre os que citam irrigação.

## 5. Importação de arquivos

A base pode ser ampliada ou substituída pela barra lateral com manuais em .txt, .md, .csv ou .pdf. Cada parágrafo vira um documento: num .txt sem linhas em branco, cada linha; no CSV, a coluna “texto”; no PDF, os parágrafos de cada página. Repetições e trechos com menos de {TAMANHO_MINIMO} caracteres são ignorados, e cada documento guarda a origem (arquivo e página ou linha). A leitura usa `csv`, `io` e `pypdf` só para extrair o texto: o índice invertido e o TF-IDF continuam calculados do zero sobre a nova base, com N igual ao número de documentos.

## 6. Divisão de tarefas

| Tarefa | Responsável |
|---|---|
| Pipeline de pré-processamento e stemmer | Victor |
| Índice invertido e TF-IDF | Victor |
| Interface Streamlit | Victor |
| Importação de arquivos e relatório | Victor |
"""


def main() -> int:
    md = PASTA / "RELATORIO.md"
    md.write_text(montar_markdown(), encoding="utf-8")
    paginas = gerar_pdf(md, PASTA / "relatorio.pdf", data=datetime(2026, 9, 28, tzinfo=timezone.utc))
    print(f"relatorio.pdf gerado com {paginas} página(s)")
    return paginas


if __name__ == "__main__":
    main()
