"""AgroSearch — Motor de Busca Inteligente (Laboratório Prático 04).

App Streamlit num único arquivo com os 3 pilares do desafio:
  Fase 1: pipeline de pré-processamento (tokenização, normalização, stopwords, stemming),
          com checkboxes para ligar/desligar stopwords e stemming;
  Fase 2: índice invertido (termo -> [IDs de docs]) construído do zero;
  Fase 3: busca e ranqueamento por TF-IDF calculado do zero.
  Bônus:  similaridade de cosseno entre o vetor TF-IDF da query e o vetor TF-IDF de cada
          documento, para lidar melhor com consultas de múltiplas palavras.

Restrição do desafio: nada de scikit-learn/TfidfVectorizer ou biblioteca de alto nível;
o índice invertido, o TF-IDF, os vetores, o cosseno e o stemmer são implementados aqui,
só com a biblioteca padrão.

Rodar:  streamlit run agrosearch_app.py
"""

import math
import re
import unicodedata

# ----------------------------------------------------------------------------- base (hardcode sugerido)
DOCUMENTOS = {
    1: "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    2: "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    3: "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    4: "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    5: "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
}

# Stopwords do português, já sem acento (a normalização vem antes da remoção).
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos", "e", "em",
    "no", "na", "nos", "nas", "para", "pra", "por", "pela", "pelo", "pelas", "pelos", "com", "sem",
    "que", "se", "ao", "aos", "ou", "mais", "muito", "como", "seu", "sua", "seus", "suas", "ele",
    "ela", "eles", "elas", "este", "esta", "isso", "isto", "ja", "nao", "entre", "sobre", "ate",
    "durante", "ser", "foi", "sao", "tem", "ha",
}

# Stemmer rudimentar próprio (sufixos do português, do mais longo para o mais curto).
SUFIXOS = sorted(
    [
        "amente", "mente", "acoes", "acao", "icoes", "icao", "ancias", "ancia", "encias", "encia",
        "idades", "idade", "mentos", "mento", "adoras", "adores", "adora", "ador", "osos", "osas",
        "oso", "osa", "ivos", "ivas", "ivo", "iva", "aveis", "avel", "iveis", "ivel", "istas",
        "ista", "ando", "endo", "indo", "ados", "adas", "ado", "ada", "idos", "idas", "ido", "ida",
        "as", "os", "es", "a", "o", "e", "s",
    ],
    key=len,
    reverse=True,
)
RADICAL_MINIMO = 3


# ----------------------------------------------------------------------------- Fase 1: pipeline
def tokenizar(texto: str) -> list[str]:
    """Etapa 1 — divide o texto em palavras, descartando pontuação."""
    return re.findall(r"\w+", texto)


def normalizar(tokens: list[str]) -> list[str]:
    """Etapa 2 — minúsculas e remoção de acentos ("Irrigação" -> "irrigacao")."""
    return [unicodedata.normalize("NFD", t).encode("ascii", "ignore").decode("ascii").lower() for t in tokens]


def remover_stopwords(tokens: list[str]) -> list[str]:
    """Etapa 3 — tira palavras muito frequentes e de pouco poder discriminativo."""
    return [t for t in tokens if t not in STOPWORDS]


def stem(palavra: str) -> str:
    """Remove o sufixo mais longo que deixe um radical com pelo menos 3 letras."""
    for sufixo in SUFIXOS:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= RADICAL_MINIMO:
            return palavra[: -len(sufixo)]
    return palavra


def aplicar_stemming(tokens: list[str]) -> list[str]:
    """Etapa 4 — reduz cada palavra ao radical."""
    return [stem(t) for t in tokens]


def etapas(texto: str, usar_stopwords: bool = True, usar_stemming: bool = True) -> dict[str, list[str]]:
    """Resultado de cada etapa do pipeline, na ordem."""
    passos = {"1. Tokenização": tokenizar(texto)}
    passos["2. Normalização"] = normalizar(passos["1. Tokenização"])
    atual = passos["2. Normalização"]
    passos["3. Remoção de stopwords"] = remover_stopwords(atual) if usar_stopwords else atual
    atual = passos["3. Remoção de stopwords"]
    passos["4. Stemming"] = aplicar_stemming(atual) if usar_stemming else atual
    return passos


def preprocessar(texto: str, usar_stopwords: bool = True, usar_stemming: bool = True) -> list[str]:
    return etapas(texto, usar_stopwords, usar_stemming)["4. Stemming"]


def vocabulario(documentos: dict, usar_stopwords: bool = True, usar_stemming: bool = True) -> list[str]:
    termos = set()
    for texto in documentos.values():
        termos.update(preprocessar(texto, usar_stopwords, usar_stemming))
    return sorted(termos)


# ----------------------------------------------------------------------------- Fase 2: índice invertido
def indice_invertido(docs_tokens: dict[int, list[str]]) -> dict[str, list[int]]:
    """Termo -> lista ordenada dos IDs dos documentos em que ele aparece."""
    indice: dict[str, list[int]] = {}
    for doc_id in sorted(docs_tokens):
        for termo in docs_tokens[doc_id]:
            postings = indice.setdefault(termo, [])
            if not postings or postings[-1] != doc_id:
                postings.append(doc_id)
    return dict(sorted(indice.items()))


# ----------------------------------------------------------------------------- Fase 3: TF-IDF
def tf(termo: str, tokens: list[str]) -> float:
    """TF(t, d) = ocorrências de t em d / total de termos de d."""
    return tokens.count(termo) / len(tokens) if tokens else 0.0


def idf(termo: str, indice: dict[str, list[int]], n_docs: int) -> float:
    """IDF(t) = log10(N / df(t)); termo ausente da coleção vale 0."""
    df = len(indice.get(termo, []))
    return math.log10(n_docs / df) if df else 0.0


def ranquear(consulta: str, documentos: dict, usar_stopwords: bool = True, usar_stemming: bool = True) -> dict:
    """Calcula TF, IDF e TF-IDF de cada termo da consulta em cada documento.

    O score do documento é o TF-IDF acumulado: a soma do TF-IDF dos termos (distintos) da consulta.
    Retorna {"termos", "idf", "linhas"}, com as linhas ordenadas do maior para o menor score.
    """
    docs_tokens = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in documentos.items()}
    indice = indice_invertido(docs_tokens)
    termos = list(dict.fromkeys(preprocessar(consulta, usar_stopwords, usar_stemming)))
    idfs = {t: idf(t, indice, len(documentos)) for t in termos}
    linhas = []
    for doc_id, tokens in docs_tokens.items():
        linha = {"doc": doc_id, "tf": {}, "tfidf": {}}
        for t in termos:
            linha["tf"][t] = tf(t, tokens)
            linha["tfidf"][t] = linha["tf"][t] * idfs[t]
        linha["score"] = sum(linha["tfidf"].values())
        linhas.append(linha)
    linhas.sort(key=lambda x: (-x["score"], x["doc"]))
    return {"termos": termos, "idf": idfs, "df": {t: len(indice.get(t, [])) for t in termos}, "linhas": linhas}


# ----------------------------------------------------------------------------- Bônus: similaridade de cosseno
def vetor_tfidf(tokens: list[str], idfs: dict[str, float]) -> dict[str, float]:
    """Vetor TF-IDF esparso no vocabulário da coleção: w(t) = TF(t) · IDF(t).

    Termos fora do vocabulário (sem IDF na coleção) ficam de fora: não existem no espaço vetorial.
    """
    return {t: tf(t, tokens) * idfs[t] for t in dict.fromkeys(tokens) if idfs.get(t, 0.0) > 0}


def cosseno(a: dict[str, float], b: dict[str, float]) -> float:
    """cos(θ) = (a · b) / (‖a‖ · ‖b‖); vale 0 se algum vetor for nulo."""
    produto = sum(peso * b.get(termo, 0.0) for termo, peso in a.items())
    norma = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return produto / norma if norma else 0.0


def ranquear_cosseno(consulta: str, documentos: dict, usar_stopwords: bool = True, usar_stemming: bool = True) -> dict:
    """Ranqueia os documentos pelo cosseno entre o vetor TF-IDF da query e o de cada documento.

    O IDF é sempre o da coleção de documentos; o TF da query é calculado sobre os tokens da própria query.
    Retorna {"vetor_consulta", "linhas"}, com as linhas ordenadas do maior para o menor cosseno.
    """
    docs_tokens = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in documentos.items()}
    indice = indice_invertido(docs_tokens)
    idfs = {t: idf(t, indice, len(documentos)) for t in indice}
    vetor_q = vetor_tfidf(preprocessar(consulta, usar_stopwords, usar_stemming), idfs)
    acumulado = {x["doc"]: x["score"] for x in ranquear(consulta, documentos, usar_stopwords, usar_stemming)["linhas"]}
    linhas = [
        {"doc": d, "cosseno": cosseno(vetor_q, vetor_tfidf(tokens, idfs)), "tfidf_acumulado": acumulado[d]}
        for d, tokens in docs_tokens.items()
    ]
    linhas.sort(key=lambda x: (-x["cosseno"], x["doc"]))
    return {"vetor_consulta": vetor_q, "linhas": linhas}


# ----------------------------------------------------------------------------- interface
def main():
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
    st.title("🌱 AgroSearch — Motor de Busca Inteligente")
    st.caption("Protótipo de busca textual nos manuais técnicos da AgroTech Solutions: "
               "pré-processamento, índice invertido e TF-IDF implementados do zero.")

    st.sidebar.header("⚙️ Pré-processamento")
    usar_stopwords = st.sidebar.checkbox("Remover stopwords", value=True, key="stopwords")
    usar_stemming = st.sidebar.checkbox("Aplicar stemming", value=True, key="stemming")

    with st.expander("📚 Base de documentos"):
        st.dataframe(pd.DataFrame({"ID": list(DOCUMENTOS), "Texto": list(DOCUMENTOS.values())}), hide_index=True)

    fase1, fase2, fase3, bonus = st.tabs([
        "1️⃣ Pipeline de Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca TF-IDF", "⭐ Bônus: Similaridade de Cosseno",
    ])

    with fase1:
        doc_id = st.selectbox("Documento para inspecionar", list(DOCUMENTOS), format_func=lambda d: f"Doc {d}", key="doc")
        for nome, tokens in etapas(DOCUMENTOS[doc_id], usar_stopwords, usar_stemming).items():
            with st.expander(nome, expanded=True):
                st.write(tokens)
        vocab = vocabulario(DOCUMENTOS, usar_stopwords, usar_stemming)
        st.metric("Termos no vocabulário", len(vocab))
        st.write(", ".join(vocab))
        st.caption("Ligue/desligue stopwords e stemming na barra lateral para ver o vocabulário mudar.")

    with fase2:
        docs_tokens = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in DOCUMENTOS.items()}
        indice = indice_invertido(docs_tokens)
        st.markdown(f"**{len(indice)} termos** mapeados para os documentos em que aparecem (Termo → [IDs de Docs]).")
        st.json(indice)

    with fase3:
        consulta = st.text_input("Digite sua consulta:", value="irrigação da soja", key="consulta")
        r = ranquear(consulta, DOCUMENTOS, usar_stopwords, usar_stemming)
        if not r["termos"]:
            st.warning("A consulta não tem termos depois do pré-processamento.")
        else:
            st.markdown("**IDF dos termos da consulta** — IDF(t) = log10(N / df(t)), N = 5")
            st.dataframe(pd.DataFrame([
                {"Termo": t, "df(t)": r["df"][t], "IDF(t)": round(r["idf"][t], 4)} for t in r["termos"]
            ]), hide_index=True)

            vencedor = r["linhas"][0] if r["linhas"][0]["score"] > 0 else None
            tabela = []
            for linha in r["linhas"]:
                registro = {"🏆": "🏆" if vencedor and linha["doc"] == vencedor["doc"] else "", "Documento": f"Doc {linha['doc']}"}
                for t in r["termos"]:
                    registro[f"TF({t})"] = round(linha["tf"][t], 4)
                    registro[f"TF-IDF({t})"] = round(linha["tfidf"][t], 4)
                registro["TF-IDF acumulado"] = round(linha["score"], 4)
                tabela.append(registro)
            st.markdown("**Ranking (maior → menor TF-IDF acumulado)** — TF(t, d) = ocorrências / total de termos de d")
            st.dataframe(pd.DataFrame(tabela), hide_index=True)
            if vencedor:
                st.success(f"🏆 Documento vencedor: Doc {vencedor['doc']} (TF-IDF acumulado = {vencedor['score']:.4f}) — "
                           f"{DOCUMENTOS[vencedor['doc']]}")
            else:
                st.warning("Nenhum documento contém os termos da consulta.")

    with bonus:
        st.markdown(
            "A consulta vira um **vetor TF-IDF** no mesmo vocabulário dos documentos (TF da consulta × IDF da "
            "coleção), e cada documento é ranqueado pela **similaridade de cosseno** com ela: "
            "cos(θ) = (q · d) / (‖q‖ · ‖d‖). Diferente do TF-IDF acumulado, o cosseno normaliza pelo tamanho dos "
            "vetores e premia o documento cujo perfil de termos se parece com o da consulta inteira, o que ajuda "
            "em consultas de várias palavras."
        )
        st.caption(f"Consulta (a mesma da aba 3): “{consulta}”")
        rc = ranquear_cosseno(consulta, DOCUMENTOS, usar_stopwords, usar_stemming)
        if not rc["vetor_consulta"]:
            st.warning("Nenhum termo da consulta existe no vocabulário dos documentos: o vetor da consulta é nulo.")
        else:
            st.markdown("**Vetor TF-IDF da consulta**")
            st.dataframe(pd.DataFrame([{"Termo": t, "Peso TF-IDF": round(p, 4)} for t, p in rc["vetor_consulta"].items()]),
                         hide_index=True)
            posicao_acumulado = {x["doc"]: i for i, x in enumerate(sorted(rc["linhas"], key=lambda x: (-x["tfidf_acumulado"], x["doc"])), 1)}
            melhor = rc["linhas"][0]
            st.markdown("**Ranking por similaridade de cosseno (maior → menor)**")
            st.dataframe(pd.DataFrame([
                {"🏆": "🏆" if x["doc"] == melhor["doc"] and x["cosseno"] > 0 else "", "Posição": i, "Documento": f"Doc {x['doc']}",
                 "Cosseno": round(x["cosseno"], 4), "TF-IDF acumulado": round(x["tfidf_acumulado"], 4),
                 "Posição no TF-IDF acumulado": posicao_acumulado[x["doc"]]}
                for i, x in enumerate(rc["linhas"], 1)
            ]), hide_index=True)
            if melhor["cosseno"] > 0:
                st.success(f"🏆 Mais similar à consulta: Doc {melhor['doc']} (cosseno = {melhor['cosseno']:.4f}) — "
                           f"{DOCUMENTOS[melhor['doc']]}")


if __name__ == "__main__":
    main()
