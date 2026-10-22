"""
Diagnóstico da recuperação híbrida (sem LLM e sem alterar rag_pipeline.py).

1. Detalha as falhas da busca híbrida atual no conjunto curado: tokens BM25
   (com e sem stopwords de domínio), top-4 semântico, top-4 BM25, top-k final
   e se a âncora semântica foi acionada.
2. Ablação: compara hit@1/hit@k em três variantes da busca híbrida:
     a) só stopwords de domínio (configuração atual do pipeline)
     b) só âncora semântica (sem stopwords de domínio)
     c) ambas
   A âncora (forçar o 1º semântico no top-k) não existe mais no pipeline;
   foi descartada com base nesta ablação e só é reproduzida aqui.

Uso:
  poetry run python -m src.diagnostico_recuperacao
"""

from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd
from rank_bm25 import BM25Okapi

from .rag_pipeline import (
    CODIGO_RE,
    DATA_DIR,
    K_PADRAO,
    RRF_K,
    STOPWORDS,
    STOPWORDS_DOMINIO,
    BuscaHibrida,
    build_vectorstore,
    carregar_protocolos,
    recuperar,
)

VARIANTES = {
    "a) só stopwords de domínio (atual)": {"stopwords_dominio": True, "ancora": False},
    "b) só âncora": {"stopwords_dominio": False, "ancora": True},
    "c) ambas": {"stopwords_dominio": True, "ancora": True},
}
CONFIG_ATUAL = VARIANTES["a) só stopwords de domínio (atual)"]


def tokenizar(texto: str, stopwords: set[str]) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", texto) if t not in stopwords]


def codigo(doc) -> str:
    return CODIGO_RE.search(doc.metadata["fonte"]).group()


class BuscaVariante:
    """Reimplementação de BuscaHibrida.buscar com as duas correções ligáveis."""

    def __init__(self, store, docs, stopwords_dominio: bool, ancora: bool):
        self.store = store
        self.docs = docs
        self.ancora = ancora
        self.stopwords = STOPWORDS | STOPWORDS_DOMINIO if stopwords_dominio else STOPWORDS
        self.bm25 = BM25Okapi([tokenizar(d.page_content, self.stopwords) for d in docs])

    def detalhar(self, pergunta: str, k: int = K_PADRAO, modo: str = "hibrida") -> dict:
        semanticos = recuperar(self.store, pergunta, k=len(self.docs))
        info = {
            d.metadata["id"]: {"doc": d, "score_semantico": s, "pos_semantica": pos,
                               "pos_bm25": None, "score_bm25": 0.0, "ancora_semantica": False}
            for pos, (d, s) in enumerate(semanticos, 1)
        }
        tokens = tokenizar(pergunta, self.stopwords)
        scores = self.bm25.get_scores(tokens)
        ranking_bm25 = sorted(
            [(d.metadata["id"], s) for d, s in zip(self.docs, scores) if s > 0], key=lambda x: -x[1]
        )
        for pos, (doc_id, s) in enumerate(ranking_bm25, 1):
            info[doc_id]["pos_bm25"] = pos
            info[doc_id]["score_bm25"] = float(s)

        for item in info.values():
            rrf = 1 / (RRF_K + item["pos_semantica"])
            if modo == "hibrida" and item["pos_bm25"] is not None:
                rrf += 1 / (RRF_K + item["pos_bm25"])
            item["score_rrf"] = rrf

        escolhidos = sorted(info.values(), key=lambda x: (-x["score_rrf"], x["pos_semantica"]))[:k]
        if modo == "hibrida" and self.ancora:
            top = next(i for i in info.values() if i["pos_semantica"] == 1)
            if top not in escolhidos:
                top["ancora_semantica"] = True
                escolhidos[-1] = top

        return {
            "tokens": tokens,
            "top_semantico": [(codigo(d), round(s, 3)) for d, s in semanticos[:4]],
            "top_bm25": [(codigo(info[i]["doc"]), round(float(s), 3)) for i, s in ranking_bm25[:4]],
            "escolhidos": escolhidos,
        }

    def buscar_ids(self, pergunta: str, k: int, modo: str) -> list[str]:
        return [e["doc"].metadata["id"] for e in self.detalhar(pergunta, k, modo)["escolhidos"]]


def conjuntos() -> dict[str, list[tuple[str, str]]]:
    gold = json.loads((DATA_DIR / "perguntas_medicas.json").read_text(encoding="utf-8"))
    curadas = json.loads((DATA_DIR / "qa_curado.json").read_text(encoding="utf-8"))
    return {
        "avaliacao (10)": [(g["pergunta"], g["fonte"]) for g in gold],
        f"curadas ({len(curadas)})": [(c["pergunta"], c["fonte_id"]) for c in curadas],
    }


def diagnosticar_falhas(store, docs) -> None:
    atual = BuscaVariante(store, docs, **CONFIG_ATUAL)
    sem_dominio = BuscaVariante(store, docs, stopwords_dominio=False, ancora=False)
    referencia = BuscaHibrida(store, docs)

    curadas = list(conjuntos().values())[1]
    falhas = []
    for pergunta, esperado in curadas:
        ids = atual.buscar_ids(pergunta, K_PADRAO, "hibrida")
        ids_pipeline = [e["doc"].metadata["id"] for e in referencia.buscar(pergunta, K_PADRAO)]
        assert ids == ids_pipeline, f"Variante atual divergiu do pipeline: {pergunta!r}"
        if esperado not in ids:
            falhas.append((pergunta, esperado))

    print(f"=== Falhas da busca híbrida atual nas curadas: {len(falhas)} ===")
    for n, (pergunta, esperado) in enumerate(falhas, 1):
        d = atual.detalhar(pergunta)
        print(f"\n[{n}] {pergunta}")
        print(f"    esperado: {esperado}")
        print(f"    tokens BM25 antes das stopwords de domínio: {sem_dominio.detalhar(pergunta)['tokens']}")
        print(f"    tokens BM25 depois:                         {d['tokens']}")
        print(f"    top-4 semântico: {d['top_semantico']}")
        print(f"    top-4 BM25:      {d['top_bm25']}")
        finais = [
            (codigo(e["doc"]), e["doc"].metadata["id"], f"sem pos {e['pos_semantica']}",
             f"bm25 pos {e['pos_bm25']}", f"rrf {e['score_rrf']:.4f}")
            for e in d["escolhidos"]
        ]
        print(f"    top-{K_PADRAO} final:     {finais}")
        print(f"    âncora acionada: {any(e['ancora_semantica'] for e in d['escolhidos'])}")


def ablacao(store, docs, k: int = K_PADRAO) -> None:
    linhas = []
    for variante, flags in VARIANTES.items():
        busca = BuscaVariante(store, docs, **flags)
        for nome, itens in conjuntos().items():
            for modo in ("semantica", "hibrida"):
                hit1 = hitk = 0
                for pergunta, esperado in itens:
                    ids = busca.buscar_ids(pergunta, k, modo)
                    hit1 += ids[0] == esperado
                    hitk += esperado in ids
                linhas.append({"variante": variante, "conjunto": nome, "busca": modo,
                               "hit@1": round(hit1 / len(itens), 3),
                               f"hit@{k}": round(hitk / len(itens), 3)})
    print("\n=== Ablação (a semântica não depende das flags; repetida como referência) ===")
    print(pd.DataFrame(linhas).to_string(index=False))


if __name__ == "__main__":
    import logging
    for ruidoso in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    store = build_vectorstore()
    docs = carregar_protocolos()
    diagnosticar_falhas(store, docs)
    ablacao(store, docs)
