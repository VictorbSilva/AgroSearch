import ast
import json
import math
from pathlib import Path

import pytest

from agrosearch_app import (
    DOCUMENTOS,
    etapas,
    idf,
    indice_invertido,
    preprocessar,
    ranquear,
    stem,
    tf,
    vocabulario,
)

PASTA = Path(__file__).resolve().parents[1]
APP = PASTA / "agrosearch_app.py"


# ------------------------------------------------------------------ Fase 1: pipeline
def test_pipeline_tem_as_4_etapas_na_ordem():
    passos = etapas("A Irrigação da soja!")
    assert list(passos) == ["1. Tokenização", "2. Normalização", "3. Remoção de stopwords", "4. Stemming"]
    assert passos["1. Tokenização"] == ["A", "Irrigação", "da", "soja"]  # pontuação descartada
    assert passos["2. Normalização"] == ["a", "irrigacao", "da", "soja"]  # minúsculas, sem acento
    assert passos["3. Remoção de stopwords"] == ["irrigacao", "soja"]
    assert passos["4. Stemming"] == ["irrig", "soj"]


def test_normalizacao_remove_acentos_e_caixa():
    assert preprocessar("ÁGUA Orgânico nitrogênio", usar_stopwords=False, usar_stemming=False) == [
        "agua", "organico", "nitrogenio"
    ]


@pytest.mark.parametrize("palavras", [("lagarta", "lagartas"), ("irrigação", "irrigações"), ("soja", "sojas")])
def test_stemming_junta_variacoes(palavras):
    radicais = {preprocessar(p)[0] for p in palavras}
    assert len(radicais) == 1


def test_stemming_mantem_radical_minimo():
    assert stem("sol") == "sol"
    assert len(stem("agua")) >= 3


def test_checkboxes_mudam_o_vocabulario():
    completo = vocabulario(DOCUMENTOS, usar_stopwords=False, usar_stemming=False)
    sem_stop = vocabulario(DOCUMENTOS, usar_stopwords=True, usar_stemming=False)
    com_stem = vocabulario(DOCUMENTOS, usar_stopwords=True, usar_stemming=True)
    assert len(sem_stop) < len(completo)
    assert "de" in completo and "de" not in sem_stop
    assert "irrigacao" in sem_stop and "irrig" in com_stem and "irrigacao" not in com_stem


# ------------------------------------------------------------------ Fase 2: índice invertido
def test_indice_invertido_termo_para_ids():
    docs = {d: preprocessar(t) for d, t in DOCUMENTOS.items()}
    indice = indice_invertido(docs)
    assert indice["soj"] == [1, 2, 4]
    assert indice["lagart"] == [2, 4]
    assert indice["irrig"] == [1, 5]
    for termo, ids in indice.items():  # cada posting corresponde a documentos que têm o termo
        assert ids == sorted(set(ids))
        assert all(termo in docs[d] for d in ids)


def test_indice_usa_tokens_limpos():
    indice = indice_invertido({d: preprocessar(t) for d, t in DOCUMENTOS.items()})
    assert "a" not in indice and "Irrigação" not in indice


# ------------------------------------------------------------------ Fase 3: TF-IDF
def test_tf_e_idf_seguem_as_formulas_dos_slides():
    tokens = ["gato", "viu", "gato", "telhado"]
    assert tf("gato", tokens) == 0.5
    indice = {"gato": [1, 3], "cachorro": [2, 3]}
    assert math.isclose(idf("gato", indice, 3), math.log10(3 / 2))  # ≈ 0,176, como no slide 20
    assert idf("inexistente", indice, 3) == 0.0


def test_ranking_calcula_tfidf_acumulado_corretamente():
    r = ranquear("irrigação da soja", DOCUMENTOS)
    assert r["termos"] == ["irrig", "soj"]
    doc1 = next(linha for linha in r["linhas"] if linha["doc"] == 1)
    n1 = len(preprocessar(DOCUMENTOS[1]))  # 8 termos
    esperado = (1 / n1) * math.log10(5 / 2) + (1 / n1) * math.log10(5 / 3)
    assert math.isclose(doc1["score"], esperado)
    assert math.isclose(r["idf"]["irrig"], math.log10(5 / 2))


def test_ranking_ordenado_do_maior_para_o_menor_com_vencedor():
    r = ranquear("irrigação da soja", DOCUMENTOS)
    scores = [linha["score"] for linha in r["linhas"]]
    assert scores == sorted(scores, reverse=True)
    assert r["linhas"][0]["doc"] == 1


def test_consulta_sem_termos_conhecidos_da_zero():
    r = ranquear("tratores elétricos", DOCUMENTOS)
    assert all(linha["score"] == 0 for linha in r["linhas"])


# ------------------------------------------------------------------ restrição técnica
def test_nao_usa_bibliotecas_de_alto_nivel():
    arvore = ast.parse(APP.read_text(encoding="utf-8"))
    modulos = {n.names[0].name.split(".")[0] for n in ast.walk(arvore) if isinstance(n, ast.Import)}
    modulos |= {n.module.split(".")[0] for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom) and n.module}
    assert modulos <= {"math", "re", "unicodedata", "collections", "pandas", "streamlit"}
    nomes = {n.id for n in ast.walk(arvore) if isinstance(n, ast.Name)}
    nomes |= {n.attr for n in ast.walk(arvore) if isinstance(n, ast.Attribute)}
    assert not nomes & {"TfidfVectorizer", "CountVectorizer", "BM25Okapi"}


# ------------------------------------------------------------------ interface Streamlit
def _app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=30).run()


def test_app_integra_as_3_fases_sem_erro():
    at = _app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["1️⃣ Pipeline de Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca TF-IDF"]


def test_app_checkboxes_mudam_vocabulario_exibido():
    at = _app()
    com_stop = at.metric[0].value
    at.checkbox(key="stopwords").uncheck().run()
    sem_stop = at.metric[0].value
    assert int(sem_stop) > int(com_stop)
    at.checkbox(key="stemming").uncheck().run()
    assert any("irrigacao" in m.value for m in at.tabs[0].markdown)


def test_app_exibe_indice_com_st_json():
    at = _app()
    indice = json.loads(at.tabs[1].json[0].value)
    assert indice["soj"] == [1, 2, 4]
    assert indice["lagart"] == [2, 4]


def test_app_busca_mostra_tabela_ordenada_e_vencedor():
    at = _app()
    at.text_input(key="consulta").set_value("lagartas na soja").run()
    fase3 = at.tabs[2]
    ranking = fase3.dataframe[1].value
    assert list(ranking["TF-IDF acumulado"]) == sorted(ranking["TF-IDF acumulado"], reverse=True)
    assert ranking.iloc[0]["🏆"] == "🏆"
    assert "Documento vencedor" in fase3.success[0].value


# ------------------------------------------------------------------ relatório
def test_relatorio_pdf_ate_2_paginas_com_divisao_de_tarefas():
    from pypdf import PdfReader

    pdf = PASTA / "relatorio.pdf"
    assert pdf.exists()
    leitor = PdfReader(pdf)
    assert len(leitor.pages) <= 2
    texto = " ".join(p.extract_text() for p in leitor.pages)
    assert "Divisão de tarefas" in texto
