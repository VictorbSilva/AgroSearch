# AgroSearch — Motor de Busca Inteligente

Laboratório Prático 04 da disciplina **Tendências em Ciência da Computação** (Prof. Me. Ricardo Roberto de Lima), valendo 0,5 ponto extra na AV1.

Protótipo de busca textual para os manuais técnicos da AgroTech Solutions. O técnico digita uma consulta e recebe os trechos mais relevantes, ranqueados. O pipeline de pré-processamento, o índice invertido, o TF-IDF e a similaridade de cosseno foram implementados do zero, sem scikit-learn, como o desafio exige.

| Pasta | Entrega | Conteúdo |
|---|---|---|
| [`a02/`](a02/) | Desafio principal | App Streamlit num único arquivo (`agrosearch_app.py`) com as 3 fases: pipeline de pré-processamento com checkboxes de stopwords e stemming, índice invertido e ranking por TF-IDF acumulado. Inclui o relatório técnico `relatorio.pdf` (2 páginas). |
| [`a03/`](a03/) | Desafio bônus | O mesmo app com uma 4ª aba que ranqueia os documentos pela similaridade de cosseno entre o vetor TF-IDF da consulta e o de cada documento. |

Cada pasta é independente e tem o próprio `README.md` (enunciado, como rodar e suposições), `requirements.txt` e testes. As duas entregas foram testadas e aprovadas numa auditoria independente antes da publicação.

## Como rodar

```bash
cd a03                      # ou a02
pip install -r requirements.txt
streamlit run agrosearch_app.py
python -m pytest -q
```
