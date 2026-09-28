"""AgroSearch — Motor de Busca Inteligente (Laboratório Prático 04).

App Streamlit num único arquivo com os 3 pilares do desafio:
  Fase 1: pipeline de pré-processamento (tokenização, normalização, stopwords, stemming),
          com checkboxes para ligar/desligar stopwords e stemming;
  Fase 2: índice invertido (termo -> [IDs de docs]) construído do zero;
  Fase 3: busca e ranqueamento por TF-IDF calculado do zero.

Restrição do desafio: nada de scikit-learn/TfidfVectorizer ou biblioteca de alto nível;
o índice invertido, o TF-IDF e o stemmer são implementados aqui, só com a biblioteca padrão.

Importação: a barra lateral aceita manuais em .txt, .md, .csv e .pdf; cada parágrafo vira um documento
e a base importada substitui ou amplia os 5 documentos sugeridos. A leitura dos arquivos usa csv/io
(biblioteca padrão) e pypdf, só para extrair o texto.

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


# ----------------------------------------------------------------------------- importação de arquivos
# Ler os arquivos usa a biblioteca padrão (csv, io) e o pypdf, só para extrair o texto. O índice invertido
# e o TF-IDF continuam calculados do zero, agora sobre os trechos importados.
TIPOS_ACEITOS = ["txt", "md", "csv", "pdf"]
TAMANHO_MINIMO = 40  # trechos mais curtos (títulos soltos, números de página) não viram documento
COLUNAS_TEXTO = {"texto", "conteudo", "trecho", "documento", "text", "content"}
ORIGEM_PADRAO = "base sugerida (seção 5)"


def decodificar(conteudo: bytes) -> str:
    """UTF-8 (com ou sem BOM); se falhar, Latin-1, comum em arquivos salvos no Windows."""
    try:
        return conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        return conteudo.decode("latin-1")


def paragrafos_por_linha_curta(linhas: list[str]) -> list[str]:
    """Junta em parágrafos as linhas de um texto sem linhas em branco, como o extraído de um PDF.

    A última linha de um parágrafo não chega à margem: uma linha com menos de 80% da maior largura fecha
    o parágrafo (e separa os títulos, que depois caem pelo tamanho mínimo).
    """
    linhas = [re.sub(r"\s+", " ", linha).strip() for linha in linhas if linha.strip()]
    largura = max((len(linha) for linha in linhas), default=0)
    paragrafos, atual = [], []
    for linha in linhas:
        atual.append(linha)
        if len(linha) < 0.8 * largura:
            paragrafos.append(" ".join(atual))
            atual = []
    return paragrafos + ([" ".join(atual)] if atual else [])


def dividir_em_trechos(texto: str, linha_por_trecho: bool = True) -> list[str]:
    """Um trecho por parágrafo (blocos separados por linha em branco).

    Sem linhas em branco: com linha_por_trecho=True, cada linha vira um trecho, como na base sugerida
    (uma frase por documento); com False (PDF), os parágrafos saem de paragrafos_por_linha_curta.
    Quebras de linha dentro de um parágrafo viram espaço, e trechos com menos de TAMANHO_MINIMO
    caracteres são descartados.
    """
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    blocos = [b for b in re.split(r"\n\s*\n", texto) if b.strip()]
    if len(blocos) == 1:
        linhas = blocos[0].split("\n")
        blocos = linhas if linha_por_trecho else paragrafos_por_linha_curta(linhas)
    trechos = [re.sub(r"\s+", " ", b).strip() for b in blocos]
    return [t for t in trechos if len(t) >= TAMANHO_MINIMO]


def ler_csv(texto: str) -> list[tuple[int, str]]:
    """(número da linha, texto) da coluna de conteúdo do CSV, separado por vírgula, ponto e vírgula ou tab.

    A coluna é a que tiver cabeçalho texto, conteúdo, trecho, documento, text ou content. Um CSV de uma
    coluna só, sem esse cabeçalho, é lido inteiro, uma linha por documento.
    """
    import csv
    import io

    try:
        dialeto = csv.Sniffer().sniff(texto[:2048], delimiters=",;\t")
    except csv.Error:
        dialeto = csv.excel
    linhas = [linha for linha in csv.reader(io.StringIO(texto), dialeto)]
    if not linhas:
        return []
    cabecalho = [normalizar([c.strip()])[0] if c.strip() else "" for c in linhas[0]]
    coluna = next((i for i, nome in enumerate(cabecalho) if nome in COLUNAS_TEXTO), None)
    inicio = 1
    if coluna is None:
        if max(len(linha) for linha in linhas) > 1:
            raise ValueError("o CSV tem várias colunas e nenhuma se chama texto, conteudo ou trecho")
        coluna, inicio = 0, 0
    return [(n, re.sub(r"\s+", " ", linha[coluna]).strip()) for n, linha in enumerate(linhas[inicio:], inicio + 1)
            if len(linha) > coluna and len(linha[coluna].strip()) >= TAMANHO_MINIMO]


def ler_pdf(conteudo: bytes) -> list[tuple[int, str]]:
    """(página, texto) de cada página do PDF, na ordem."""
    import io

    from pypdf import PdfReader

    return [(n, pagina.extract_text() or "") for n, pagina in enumerate(PdfReader(io.BytesIO(conteudo)).pages, 1)]


def extrair_trechos(nome: str, conteudo: bytes) -> list[tuple[str, str]]:
    """Lê um arquivo importado e devolve os trechos que viram documentos, cada um com a sua origem."""
    extensao = nome.rsplit(".", 1)[-1].lower() if "." in nome else ""
    if extensao in ("txt", "md"):
        return [(nome, t) for t in dividir_em_trechos(decodificar(conteudo))]
    if extensao == "csv":
        return [(f"{nome}, linha {n}", t) for n, t in ler_csv(decodificar(conteudo))]
    if extensao == "pdf":
        return [(f"{nome}, p. {n}", t) for n, texto in ler_pdf(conteudo) for t in dividir_em_trechos(texto, linha_por_trecho=False)]
    raise ValueError(f"formato .{extensao or '?'} não aceito (use {', '.join('.' + t for t in TIPOS_ACEITOS)})")


def importar(arquivos: list[tuple[str, bytes]], base: dict[int, str], origens: dict[int, str],
             substituir: bool = False) -> tuple[dict[int, str], dict[int, str], list[tuple[str, int, str]]]:
    """Acrescenta à base (ou põe no lugar dela) os trechos dos arquivos: devolve (base, origens, relatório).

    Os IDs continuam inteiros e sequenciais: a partir de 1 ao substituir, ou depois do maior ID atual ao
    adicionar. Trechos repetidos (já na base ou no mesmo lote) são ignorados. O relatório traz, por arquivo,
    quantos trechos entraram ou o motivo da recusa. Se a base ficaria vazia, nada muda.
    """
    nova_base = {} if substituir else dict(base)
    novas_origens = {} if substituir else dict(origens)
    vistos = set(nova_base.values())
    proximo = max(nova_base, default=0) + 1
    relatorio = []
    for nome, conteudo in arquivos:
        try:
            trechos = extrair_trechos(nome, conteudo)
        except Exception as erro:  # formato não aceito, CSV sem coluna de texto, PDF corrompido…
            relatorio.append((nome, 0, f"não foi possível ler ({erro})"))
            continue
        novos = 0
        for origem, trecho in trechos:
            if trecho not in vistos:
                vistos.add(trecho)
                nova_base[proximo], novas_origens[proximo] = trecho, origem
                proximo += 1
                novos += 1
        relatorio.append((nome, novos, "" if novos else "nenhum trecho novo com texto (arquivo vazio, PDF escaneado ou só repetições)"))
    if not nova_base:
        return dict(base), dict(origens), relatorio
    return nova_base, novas_origens, relatorio


def resumir(texto: str, limite: int = 300) -> str:
    return texto if len(texto) <= limite else texto[:limite].rsplit(" ", 1)[0] + "…"


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

    st.sidebar.header("📂 Importar documentos")
    if "base" not in st.session_state:
        st.session_state["base"] = dict(DOCUMENTOS)
        st.session_state["origens"] = {d: ORIGEM_PADRAO for d in DOCUMENTOS}
    arquivos = st.sidebar.file_uploader(
        "Manuais em .txt, .md, .csv ou .pdf", type=TIPOS_ACEITOS, accept_multiple_files=True, key="arquivos",
        help="Cada parágrafo vira um documento (num .txt sem linhas em branco, cada linha). No CSV, a coluna "
             "“texto”; no PDF, os parágrafos de cada página.")
    modo = st.sidebar.radio("Ao importar", ["Adicionar à base", "Substituir a base"], key="modo_importacao")
    col_importar, col_restaurar = st.sidebar.columns(2)
    if col_importar.button("📥 Importar", key="importar", disabled=not arquivos, width="stretch"):
        base, origens, relatorio = importar([(a.name, a.getvalue()) for a in arquivos], st.session_state["base"],
                                            st.session_state["origens"], substituir=modo == "Substituir a base")
        st.session_state.update(base=base, origens=origens, relatorio_importacao=relatorio)
    if col_restaurar.button("↩️ Base original", key="restaurar", width="stretch"):
        st.session_state.update(base=dict(DOCUMENTOS), origens={d: ORIGEM_PADRAO for d in DOCUMENTOS},
                                relatorio_importacao=[])
    for nome, novos, motivo in st.session_state.get("relatorio_importacao", []):
        if novos:
            st.sidebar.success(f"{nome}: {novos} trecho(s) importado(s).")
        else:
            st.sidebar.error(f"{nome}: {motivo}.")
    documentos, origens = st.session_state["base"], st.session_state["origens"]
    st.sidebar.caption(f"Base atual: **{len(documentos)} documentos** "
                       f"({sum(o != ORIGEM_PADRAO for o in origens.values())} importados).")
    if st.session_state.get("doc") not in documentos:
        st.session_state["doc"] = next(iter(documentos))

    with st.expander(f"📚 Base de documentos ({len(documentos)})"):
        st.dataframe(pd.DataFrame({"ID": list(documentos), "Origem": [origens[d] for d in documentos],
                                   "Texto": list(documentos.values())}), hide_index=True, width="stretch")

    fase1, fase2, fase3 = st.tabs(["1️⃣ Pipeline de Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca TF-IDF"])

    with fase1:
        doc_id = st.selectbox("Documento para inspecionar", list(documentos), format_func=lambda d: f"Doc {d} · {origens[d]}", key="doc")
        for nome, tokens in etapas(documentos[doc_id], usar_stopwords, usar_stemming).items():
            with st.expander(nome, expanded=True):
                st.write(tokens)
        vocab = vocabulario(documentos, usar_stopwords, usar_stemming)
        st.metric("Termos no vocabulário", len(vocab))
        st.write(", ".join(vocab))
        st.caption("Ligue/desligue stopwords e stemming na barra lateral para ver o vocabulário mudar.")

    with fase2:
        docs_tokens = {d: preprocessar(t, usar_stopwords, usar_stemming) for d, t in documentos.items()}
        indice = indice_invertido(docs_tokens)
        st.markdown(f"**{len(indice)} termos** mapeados para os documentos em que aparecem (Termo → [IDs de Docs]).")
        st.json(indice)

    with fase3:
        consulta = st.text_input("Digite sua consulta:", value="irrigação da soja", key="consulta")
        r = ranquear(consulta, documentos, usar_stopwords, usar_stemming)
        if not r["termos"]:
            st.warning("A consulta não tem termos depois do pré-processamento.")
            return
        st.markdown(f"**IDF dos termos da consulta** — IDF(t) = log10(N / df(t)), N = {len(documentos)}")
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
                       f"{resumir(documentos[vencedor['doc']])} ({origens[vencedor['doc']]})")
        else:
            st.warning("Nenhum documento contém os termos da consulta.")


if __name__ == "__main__":
    main()
