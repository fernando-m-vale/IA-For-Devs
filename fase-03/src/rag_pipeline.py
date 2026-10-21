"""
Pipeline RAG (Retrieval-Augmented Generation) do assistente médico.

Fluxo de uma pergunta:
1. A pergunta é convertida em embedding e o ChromaDB devolve os k protocolos
   internos mais próximos semanticamente, com um score de relevância (0 a 1).
2. Guardrail de recuperação: se nenhum protocolo atingir o limiar mínimo de
   relevância, o pipeline responde "sem protocolo aplicável" SEM chamar a LLM.
3. Os protocolos recuperados são montados no MESMO formato usado no
   fine-tuning v2 (system prompt + "Contexto ... Pergunta: ...").
4. A LLM (modelo fine-tunado `medassist`, servido pelo Ollama) responde a
   partir do contexto, citando a fonte.
5. A resposta é devolvida junto com os protocolos usados e seus scores
   (explainability).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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
LIMIAR_RELEVANCIA = 0.30   # provisório: será calibrado com src.evaluate_rag --calibrar

# Variante A: system prompt idêntico ao do fine-tuning.
# Variante B: acrescenta uma instrução de completude (teste A/B).
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
    segundos: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

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
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
    """Retorna os k protocolos mais relevantes com score de relevância (0 a 1)."""
    return store.similarity_search_with_relevance_scores(pergunta, k=k)


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
    """Assistente médico: recuperação no ChromaDB + LLM via Ollama."""

    def __init__(
        self,
        modelo: str = MODELO_PADRAO,
        variante_prompt: str = "A",
        k: int = K_PADRAO,
        limiar: float = LIMIAR_RELEVANCIA,
        temperatura: float = 0.3,
        store: Chroma | None = None,
    ):
        if variante_prompt not in PROMPTS:
            raise ValueError(f"Variante '{variante_prompt}' inválida. Opções: {list(PROMPTS)}")
        self.modelo = modelo
        self.variante_prompt = variante_prompt
        self.k = k
        self.limiar = limiar
        self.store = store or build_vectorstore()
        self.llm = ChatOllama(model=modelo, temperature=temperatura, num_ctx=4096)

    def responder(self, pergunta: str) -> RespostaRAG:
        inicio = time.time()
        resultados = recuperar(self.store, pergunta, self.k)

        fontes = [
            {
                "codigo": CODIGO_RE.search(d.metadata["fonte"]).group(),
                "id": d.metadata["id"],
                "titulo": d.metadata["titulo"],
                "score": round(float(score), 3),
            }
            for d, score in resultados
        ]
        melhor_score = max((f["score"] for f in fontes), default=0.0)

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
                segundos=round(time.time() - inicio, 2),
            )

        mensagens = [
            SystemMessage(content=PROMPTS[self.variante_prompt]),
            HumanMessage(content=montar_mensagem_usuario(pergunta, [d for d, _ in resultados])),
        ]
        resposta = self.llm.invoke(mensagens).content.strip()
        segundos = round(time.time() - inicio, 2)

        logger.info(
            f"[rag] modelo={self.modelo} variante={self.variante_prompt} "
            f"fontes={[(f['codigo'], f['score']) for f in fontes]} tempo={segundos}s"
        )
        return RespostaRAG(
            pergunta=pergunta,
            resposta=resposta,
            fontes_recuperadas=fontes,
            modelo=self.modelo,
            variante_prompt=self.variante_prompt,
            segundos=segundos,
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if "--rebuild" in sys.argv:
        build_vectorstore(rebuild=True)

    assistente = AssistenteRAG()
    pergunta = "Qual a conduta para um nódulo mamário classificado como BI-RADS 4?"
    r = assistente.responder(pergunta)
    print("\nPERGUNTA:", r.pergunta)
    print("\nRESPOSTA:\n", r.resposta)
    print("\nFONTES RECUPERADAS:")
    for f in r.fontes_recuperadas:
        print(f"  {f['codigo']} | score={f['score']} | {f['titulo']}")
    print(f"\nTempo: {r.segundos}s")
