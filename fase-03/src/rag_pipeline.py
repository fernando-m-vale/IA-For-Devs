"""
Pipeline RAG (Retrieval-Augmented Generation) do assistente médico.

Fluxo de uma pergunta:
1. Busca híbrida: a pergunta é ranqueada contra os protocolos internos por
   (a) similaridade semântica (embeddings + ChromaDB), que captura o tema, e
   (b) BM25, que captura termos clínicos exatos ("compressão medular").
   Os dois rankings são combinados por Reciprocal Rank Fusion (RRF).
2. Guardrail de recuperação: se nenhum protocolo atingir o limiar mínimo de
   relevância semântica, o pipeline responde "sem protocolo aplicável" SEM
   chamar a LLM.
3. Os k protocolos escolhidos são montados no MESMO formato usado no
   fine-tuning v2 (system prompt + "Contexto ... Pergunta: ...").
4. A LLM (modelo fine-tunado `medassist`, servido pelo Ollama) responde a
   partir do contexto, citando a fonte.
5. A resposta é devolvida com os protocolos usados, suas posições em cada
   busca e o trecho do protocolo (explainability).
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi

from .data_augmentation import CODIGO_RE, SYSTEM_PROMPT_RAG, format_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data" / "synthetic"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "protocolos_internos"

MODELO_PADRAO = "medassist"
K_PADRAO = 2               # mesmo formato do treino: protocolo correto + 1 distrator
LIMIAR_RELEVANCIA = 0.47   # calibrado em conjunto separado (ver outputs/rag_eval/calibracao_limiar.csv)
RRF_K = 60                 # constante padrão do Reciprocal Rank Fusion

# Variante A (padrão): system prompt idêntico ao do fine-tuning.
# Variante B: + instrução de completude. Mantida para reprodutibilidade do
# teste A/B; não trouxe ganho de completude (ver diário de bordo).
INSTRUCAO_COMPLETUDE = (
    " Inclua todos os critérios e condutas do protocolo que sejam relevantes"
    " para a pergunta."
)
PROMPTS = {
    "A": SYSTEM_PROMPT_RAG,
    "B": SYSTEM_PROMPT_RAG + INSTRUCAO_COMPLETUDE,
}

RESPOSTA_SEM_PROTOCOLO = (
    "Não encontrei, na base de protocolos internos, informação relevante para "
    "esta pergunta. Prefiro não responder sem respaldo em um protocolo interno; "
    "a conduta deve ser definida pelo médico responsável.\n\n"
    "Fonte: nenhum protocolo aplicável"
)

STOPWORDS = set(
    "a o as os de da do das dos e em no na nos nas um uma uns umas para por com "
    "que se ao aos a e ou como qual quais quando onde ser deve devem".split()
)
NUMERO_RE = re.compile(r"\d+(?:[.,]\d+)?")

# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class RespostaRAG:
    """Resposta do assistente com os dados de rastreabilidade."""
    pergunta: str
    resposta: str
    fontes_recuperadas: list[dict] = field(default_factory=list)
    bloqueado_por_limiar: bool = False
    modelo: str = MODELO_PADRAO
    variante_prompt: str = "A"
    busca: str = "hibrida"
    segundos: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sem acentos, sem pontuação e sem stopwords (para o BM25)."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in STOPWORDS]


def extrair_numeros(texto: str) -> set[str]:
    """
    Números do texto (doses, limiares, prazos), ignorando os códigos de
    protocolo (HOSP-2024-...) e as linhas 'Fonte:'.
    """
    texto = CODIGO_RE.sub("", texto)
    texto = "\n".join(
        linha for linha in texto.splitlines()
        if not linha.strip().lower().startswith("fonte")
    )
    return {n.replace(",", ".") for n in NUMERO_RE.findall(texto)}


def numeros_sem_suporte(resposta: str, textos_contexto: list[str]) -> set[str]:
    """
    Verificação numérica: números da resposta que não aparecem em nenhum dos
    protocolos fornecidos como contexto. Um resultado não vazio indica conteúdo
    provavelmente inventado (base do guardrail da Etapa 6).
    """
    suporte = set().union(*(extrair_numeros(t) for t in textos_contexto)) if textos_contexto else set()
    return extrair_numeros(resposta) - suporte

# ---------------------------------------------------------------------------
# Base vetorial
# ---------------------------------------------------------------------------

def carregar_protocolos(path: Path = DATA_DIR / "protocolos.json") -> list[Document]:
    """
    Converte cada protocolo em um Document do LangChain.

    Sem divisão em trechos (chunking): cada protocolo tem um único parágrafo,
    e dividir separaria critérios que só fazem sentido juntos.
    """
    protocolos = json.loads(path.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=f"{p['titulo']}\n{p['conteudo']}",
            metadata={
                "id": p["id"],
                "fonte": p["fonte"],
                "titulo": p["titulo"],
                "conteudo": p["conteudo"],
                "categoria": p["categoria"],
            },
        )
        for p in protocolos
    ]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Usa a cópia local do modelo de embeddings; baixa só se ainda não existir."""
    kwargs = {"model_name": EMBEDDING_MODEL, "encode_kwargs": {"normalize_embeddings": True}}
    try:
        return HuggingFaceEmbeddings(**kwargs, model_kwargs={"local_files_only": True})
    except Exception:
        return HuggingFaceEmbeddings(**kwargs)


def _abrir_colecao(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(rebuild: bool = False) -> Chroma:
    """
    Abre (ou cria) a coleção persistida no ChromaDB. Se estiver vazia ou se
    rebuild=True, indexa os protocolos de protocolos.json.
    Use rebuild=True sempre que o texto dos protocolos mudar.
    """
    embeddings = get_embeddings()
    store = _abrir_colecao(embeddings)

    if rebuild:
        store.delete_collection()
        store = _abrir_colecao(embeddings)

    if not store.get(limit=1)["ids"]:
        docs = carregar_protocolos()
        store.add_documents(docs, ids=[d.metadata["id"] for d in docs])
        logger.info(f"[rag] {len(docs)} protocolos indexados em {VECTORSTORE_DIR}")

    return store


def recuperar(store: Chroma, pergunta: str, k: int = K_PADRAO) -> list[tuple[Document, float]]:
    """Busca apenas semântica: k protocolos com score de relevância (0 a 1)."""
    return store.similarity_search_with_relevance_scores(pergunta, k=k)

# ---------------------------------------------------------------------------
# Busca híbrida (semântica + BM25 com Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

class BuscaHibrida:
    """
    Combina o ranking semântico (ChromaDB) com o ranking lexical (BM25).

    Reciprocal Rank Fusion: score(d) = Σ 1 / (RRF_K + posição de d em cada busca).
    Usa apenas as posições, não os scores brutos, porque as escalas das duas
    buscas não são comparáveis. Protocolos sem nenhum termo em comum com a
    pergunta (BM25 = 0) não recebem contribuição lexical.
    """

    def __init__(self, store: Chroma, docs: list[Document] | None = None):
        self.store = store
        self.docs = docs or carregar_protocolos()
        self.por_id = {d.metadata["id"]: d for d in self.docs}
        self.bm25 = BM25Okapi([tokenizar(d.page_content) for d in self.docs])

    def buscar(self, pergunta: str, k: int = K_PADRAO, modo: str = "hibrida") -> list[dict]:
        """
        Retorna os k protocolos escolhidos, cada um com:
        doc, score_semantico, pos_semantica, pos_bm25, score_rrf.
        """
        semanticos = self.store.similarity_search_with_relevance_scores(
            pergunta, k=len(self.docs)
        )
        info = {
            d.metadata["id"]: {
                "doc": self.por_id[d.metadata["id"]],
                "score_semantico": float(score),
                "pos_semantica": pos,
                "pos_bm25": None,
            }
            for pos, (d, score) in enumerate(semanticos, 1)
        }

        scores_bm25 = self.bm25.get_scores(tokenizar(pergunta))
        ranking_bm25 = sorted(
            [(d.metadata["id"], s) for d, s in zip(self.docs, scores_bm25) if s > 0],
            key=lambda x: -x[1],
        )
        for pos, (doc_id, _) in enumerate(ranking_bm25, 1):
            info[doc_id]["pos_bm25"] = pos

        for item in info.values():
            rrf = 1 / (RRF_K + item["pos_semantica"])
            if modo == "hibrida" and item["pos_bm25"] is not None:
                rrf += 1 / (RRF_K + item["pos_bm25"])
            item["score_rrf"] = rrf

        escolhidos = sorted(info.values(), key=lambda x: (-x["score_rrf"], x["pos_semantica"]))
        return escolhidos[:k]

    def melhor_score_semantico(self, pergunta: str) -> float:
        return float(recuperar(self.store, pergunta, k=1)[0][1])

# ---------------------------------------------------------------------------
# Montagem do prompt
# ---------------------------------------------------------------------------

def montar_mensagem_usuario(pergunta: str, docs: list[Document]) -> str:
    """
    Monta a mensagem do usuário no MESMO formato do dataset de fine-tuning v2
    (ver rag_example em data_augmentation.py). Diferenças de formato reduzem
    o comportamento aprendido no treino.
    """
    contexto = [
        {"fonte": d.metadata["fonte"], "titulo": d.metadata["titulo"],
         "conteudo": d.metadata["conteudo"]}
        for d in docs
    ]
    return (
        "Contexto (protocolos internos recuperados):\n"
        f"{format_context(contexto)}\n\nPergunta: {pergunta}"
    )

# ---------------------------------------------------------------------------
# Assistente
# ---------------------------------------------------------------------------

class AssistenteRAG:
    """Assistente médico: busca híbrida + LLM via Ollama."""

    def __init__(
        self,
        modelo: str = MODELO_PADRAO,
        variante_prompt: str = "A",
        k: int = K_PADRAO,
        limiar: float = LIMIAR_RELEVANCIA,
        temperatura: float = 0.3,
        busca: str = "hibrida",
        store: Chroma | None = None,
    ):
        if variante_prompt not in PROMPTS:
            raise ValueError(f"Variante '{variante_prompt}' inválida. Opções: {list(PROMPTS)}")
        if busca not in ("hibrida", "semantica"):
            raise ValueError("busca deve ser 'hibrida' ou 'semantica'")
        self.modelo = modelo
        self.variante_prompt = variante_prompt
        self.k = k
        self.limiar = limiar
        self.busca = busca
        self.buscador = BuscaHibrida(store or build_vectorstore())
        self.llm = ChatOllama(model=modelo, temperature=temperatura, num_ctx=4096)

    def responder(self, pergunta: str) -> RespostaRAG:
        inicio = time.time()
        escolhidos = self.buscador.buscar(pergunta, self.k, modo=self.busca)
        melhor_score = self.buscador.melhor_score_semantico(pergunta)

        fontes = [
            {
                "codigo": CODIGO_RE.search(e["doc"].metadata["fonte"]).group(),
                "id": e["doc"].metadata["id"],
                "titulo": e["doc"].metadata["titulo"],
                "score": round(e["score_semantico"], 3),
                "pos_semantica": e["pos_semantica"],
                "pos_bm25": e["pos_bm25"],
                "trecho": e["doc"].metadata["conteudo"],
            }
            for e in escolhidos
        ]

        # Guardrail de recuperação: não chama a LLM sem protocolo relevante
        if melhor_score < self.limiar:
            logger.info(
                f"[rag] Bloqueado por limiar ({melhor_score:.3f} < {self.limiar}) | "
                f"pergunta={pergunta!r}"
            )
            return RespostaRAG(
                pergunta=pergunta,
                resposta=RESPOSTA_SEM_PROTOCOLO,
                fontes_recuperadas=fontes,
                bloqueado_por_limiar=True,
                modelo=self.modelo,
                variante_prompt=self.variante_prompt,
                busca=self.busca,
                segundos=round(time.time() - inicio, 2),
            )

        mensagens = [
            SystemMessage(content=PROMPTS[self.variante_prompt]),
            HumanMessage(content=montar_mensagem_usuario(pergunta, [e["doc"] for e in escolhidos])),
        ]
        resposta = self.llm.invoke(mensagens).content.strip()
        segundos = round(time.time() - inicio, 2)

        logger.info(
            f"[rag] modelo={self.modelo} variante={self.variante_prompt} busca={self.busca} "
            f"fontes={[(f['codigo'], f['score'], f['pos_bm25']) for f in fontes]} tempo={segundos}s"
        )
        return RespostaRAG(
            pergunta=pergunta,
            resposta=resposta,
            fontes_recuperadas=fontes,
            modelo=self.modelo,
            variante_prompt=self.variante_prompt,
            busca=self.busca,
            segundos=segundos,
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for ruidoso in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    if "--rebuild" in sys.argv:
        build_vectorstore(rebuild=True)

    assistente = AssistenteRAG()
    pergunta = "O que caracteriza uma emergência de compressão medular em oncologia?"
    r = assistente.responder(pergunta)
    print("\nPERGUNTA:", r.pergunta)
    print("\nRESPOSTA:\n", r.resposta)
    print("\nFONTES RECUPERADAS:")
    for f in r.fontes_recuperadas:
        print(f"  {f['codigo']} | semântico={f['score']} (pos {f['pos_semantica']}) "
              f"| BM25 pos {f['pos_bm25']} | {f['titulo']}")
    print(f"\nTempo: {r.segundos}s")
