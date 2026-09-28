import math
from pathlib import Path

import pytest

from agrosearch_app import (
    DOCUMENTOS,
    cosseno,
    idf,
    indice_invertido,
    preprocessar,
    ranquear_cosseno,
    tf,
    vetor_tfidf,
)

APP = Path(__file__).resolve().parents[1] / "agrosearch_app.py"


def _idfs():
    indice = indice_invertido({d: preprocessar(t) for d, t in DOCUMENTOS.items()})
    return {t: idf(t, indice, len(DOCUMENTOS)) for t in indice}


# ------------------------------------------------------------------ critério 1: query como vetor TF-IDF no vocabulário
def test_vetor_da_query_usa_tf_da_query_e_idf_da_colecao():
    idfs = _idfs()
    tokens = preprocessar("irrigação da soja")  # ["irrig", "soj"]
    vetor = vetor_tfidf(tokens, idfs)
    assert set(vetor) == {"irrig", "soj"}
    assert math.isclose(vetor["irrig"], 0.5 * math.log10(5 / 2))
    assert math.isclose(vetor["soj"], 0.5 * math.log10(5 / 3))


def test_termos_fora_do_vocabulario_ficam_fora_do_vetor():
    vetor = vetor_tfidf(preprocessar("soja trator"), _idfs())
    assert set(vetor) == {"soj"}  # "trator" não existe na coleção


def test_consulta_repetindo_termo_muda_o_peso_no_vetor():
    idfs = _idfs()
    simples = vetor_tfidf(preprocessar("soja irrigação"), idfs)
    enfatizada = vetor_tfidf(preprocessar("soja soja soja irrigação"), idfs)
    assert enfatizada["soj"] / enfatizada["irrig"] > simples["soj"] / simples["irrig"]


# ------------------------------------------------------------------ critério 2: cosseno e ranking
def test_cosseno_segue_a_formula():
    a, b = {"x": 1.0, "y": 2.0}, {"x": 2.0, "z": 1.0}
    assert math.isclose(cosseno(a, b), 2 / (math.sqrt(5) * math.sqrt(5)))
    assert cosseno(a, {}) == 0.0
    assert math.isclose(cosseno(a, a), 1.0)


def test_cosseno_calculado_a_mao_para_o_doc_1():
    idfs = _idfs()
    q = vetor_tfidf(preprocessar("irrigação da soja"), idfs)
    tokens_d1 = preprocessar(DOCUMENTOS[1])
    d1 = {t: tf(t, tokens_d1) * idfs[t] for t in set(tokens_d1)}
    esperado = sum(q[t] * d1.get(t, 0) for t in q) / (
        math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d1.values()))
    )
    obtido = next(x["cosseno"] for x in ranquear_cosseno("irrigação da soja", DOCUMENTOS)["linhas"] if x["doc"] == 1)
    assert math.isclose(obtido, esperado)


def test_ranking_ordenado_por_cosseno_decrescente_e_entre_0_e_1():
    linhas = ranquear_cosseno("lagartas na soja e no algodão", DOCUMENTOS)["linhas"]
    valores = [x["cosseno"] for x in linhas]
    assert valores == sorted(valores, reverse=True)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in valores)


def test_documento_identico_a_consulta_tem_cosseno_1():
    linhas = ranquear_cosseno(DOCUMENTOS[3], DOCUMENTOS)["linhas"]
    assert linhas[0]["doc"] == 3
    assert math.isclose(linhas[0]["cosseno"], 1.0)


# ------------------------------------------------------------------ critério 3: consultas de múltiplas palavras
@pytest.mark.parametrize(
    "consulta,esperado",
    [
        ("controle biológico de lagartas", 2),
        ("lagartas na soja e no algodão", 4),
        ("irrigação por gotejamento na soja", 5),
        ("adubação verde para o milho", 3),
    ],
)
def test_consultas_de_varias_palavras_trazem_o_documento_certo(consulta, esperado):
    assert ranquear_cosseno(consulta, DOCUMENTOS)["linhas"][0]["doc"] == esperado


def test_consulta_sem_termos_conhecidos_da_vetor_nulo():
    r = ranquear_cosseno("trator elétrico", DOCUMENTOS)
    assert r["vetor_consulta"] == {}
    assert all(x["cosseno"] == 0 for x in r["linhas"])


def test_cosseno_respeita_os_filtros_do_pipeline():
    sem_stem = ranquear_cosseno("lagartas", DOCUMENTOS, usar_stemming=False)
    assert set(sem_stem["vetor_consulta"]) == {"lagartas"}
    assert {x["doc"] for x in sem_stem["linhas"] if x["cosseno"] > 0} == {2, 4}


# ------------------------------------------------------------------ interface
def test_app_mostra_ranking_por_cosseno_na_aba_bonus():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.text_input(key="consulta").set_value("controle biológico de lagartas").run()
    assert not at.exception
    aba = at.tabs[3]
    assert "Cosseno" in aba.label
    ranking = aba.dataframe[1].value
    assert list(ranking["Cosseno"]) == sorted(ranking["Cosseno"], reverse=True)
    assert ranking.iloc[0]["Documento"] == "Doc 2"
    assert "Doc 2" in aba.success[0].value


def test_app_avisa_quando_a_consulta_nao_tem_termos_do_vocabulario():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.text_input(key="consulta").set_value("trator elétrico").run()
    assert "vetor da consulta é nulo" in at.tabs[3].warning[0].value
