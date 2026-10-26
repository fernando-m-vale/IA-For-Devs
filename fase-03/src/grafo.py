"""
Grafo LangGraph do assistente médico (Etapa 5).

MODO PERGUNTA
  extrair_ids -> rotear -> {agente_b | painel | agente_c | rag} -> verificar_saida -> FIM
  Roteamento (determinístico, nesta ordem):
    a) a pergunta cita um ID P000          -> agente_b (ferramentas por paciente)
    b) vocabulário de painel               -> painel (template, SEM LLM)
    c) vocabulário de estatística          -> agente_c (text-to-SQL protegido)
    d) caso contrário                      -> rag (protocolos internos)
  Se casar painel e estatística, vence o painel: é o caminho sem LLM, e o
  text-to-SQL errou justamente perguntas de painel na Etapa 4.

MODO REVISÃO (um paciente)
  carregar_paciente -> avaliar_pendencias -> buscar_protocolos -> sugerir_conduta
  -> verificar_saida -> gerar_alertas -> aprovacao_medica (interrupt) -> registrar_decisao
  Os protocolos vêm do vínculo paciente_protocolos (sem busca semântica); os
  alertas são montados por template a partir das pendências ATRASADO/CRITICO.
  Paciente inexistente encerra cedo com mensagem fixa.

verificar_saida é, por enquanto, um nó de passagem: ponto de encaixe dos
guardrails de saída da Etapa 6.

Uso:
  poetry run python -m src.grafo --pergunta "Quais pacientes estão com exames atrasados?"
  poetry run python -m src.grafo --revisar P007 --aprovar --comentario "ok"
"""

from __future__ import annotations

import argparse
import json
import logging
import operator
import re
import time
import uuid
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import prontuarios
from .config import DATA_REFERENCIA
from .data_augmentation import CODIGO_RE, SYSTEM_PROMPT_RAG, format_context
from .regras_clinicas import gera_alerta

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[1] / "data" / "synthetic"

ID_RE = re.compile(r"\bP\d{3}\b")
# \b no início de cada termo; também no fim quando o termo é palavra completa
# (com plural opcional). Radicais (atrasad-, crític-) ficam abertos no fim.
VOCAB_PAINEL = re.compile(
    r"\batrasad|\balertas?\b|\bpend[eê]ncias?\b|\bcr[ií]tic|\bseguimento vencido\b", re.IGNORECASE)
VOCAB_ESTATISTICA = re.compile(
    r"\bquant[oa]s\b|\bm[eé]dias?\b|\btotal\b|\bpercentual\b|\bpropor[cç][aã]o\b", re.IGNORECASE)

MODELO_CONDUTA = "medassist"
RODAPE_REVISAO = "Sugestão de apoio à decisão — requer validação médica."
DECISOES = ("aprovado", "rejeitado")


# ---------------------------------------------------------------------------
# Estado e auditoria
# ---------------------------------------------------------------------------

class Estado(TypedDict, total=False):
    modo: Literal["pergunta", "revisao"]
    pergunta: str
    paciente_ids: list[str]
    rota: str
    dados_paciente: dict
    pendencias: list[dict]
    protocolos: list[dict]           # {codigo, titulo, texto}
    conduta: str                     # texto bruto do LLM (modo revisão)
    resposta: str
    fontes: list[str]
    alertas: list[dict]
    aprovacao: dict
    auditoria: Annotated[list[dict], operator.add]   # cada nó acrescenta um evento


def no(nome: str):
    """
    Envolve um nó: mede o tempo e acrescenta à auditoria o evento
    {no, inicio, segundos, resumo}. O nó devolve o resumo na chave '_resumo'.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(estado: Estado) -> dict:
            inicio, t0 = datetime.now().isoformat(timespec="seconds"), time.time()
            saida = dict(fn(estado) or {})
            evento = {"no": nome, "inicio": inicio, "segundos": round(time.time() - t0, 3),
                      "resumo": saida.pop("_resumo", "")}
            logger.info(f"[grafo] {nome} | {evento['segundos']}s | {evento['resumo']}")
            return {**saida, "auditoria": [evento]}
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Dependências pesadas, criadas só quando usadas (os testes não carregam LLM)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _agente_b():
    from .agente_prontuarios import AgenteProntuarios
    return AgenteProntuarios()


@lru_cache(maxsize=1)
def _agente_c():
    from .agente_sql import AgenteSQL
    return AgenteSQL()


@lru_cache(maxsize=1)
def _rag():
    from .rag_pipeline import AssistenteRAG
    return AssistenteRAG()


@lru_cache(maxsize=1)
def _llm_conduta():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODELO_CONDUTA, temperature=0.3, num_ctx=4096)


def gerar_conduta(mensagens: list) -> str:
    """Única chamada de LLM do modo revisão (substituída por mock nos testes)."""
    return _llm_conduta().invoke(mensagens).content.strip()


@lru_cache(maxsize=1)
def _protocolos_por_codigo() -> dict[str, dict]:
    protocolos = json.loads((DATA_DIR / "protocolos.json").read_text(encoding="utf-8"))
    return {CODIGO_RE.search(p["fonte"]).group(): p for p in protocolos}


# ---------------------------------------------------------------------------
# Modo pergunta
# ---------------------------------------------------------------------------

def escolher_rota(pergunta: str, paciente_ids: list[str]) -> tuple[str, str]:
    """Rota e motivo, na ordem de precedência documentada no módulo."""
    if paciente_ids:
        return "agente_b", f"pergunta cita paciente(s) {', '.join(paciente_ids)}"
    painel, estatistica = VOCAB_PAINEL.search(pergunta), VOCAB_ESTATISTICA.search(pergunta)
    if painel:
        extra = f"; também casou estatística ('{estatistica.group()}'), painel tem precedência" \
            if estatistica else ""
        return "painel", f"vocabulário de painel ('{painel.group()}'){extra}"
    if estatistica:
        return "agente_c", f"vocabulário de estatística ('{estatistica.group()}')"
    return "rag", "sem ID nem vocabulário de painel/estatística: pergunta sobre protocolos"


@no("extrair_ids")
def extrair_ids(estado: Estado) -> dict:
    ids = sorted(set(ID_RE.findall(estado["pergunta"])))
    return {"paciente_ids": ids, "_resumo": f"IDs: {ids or 'nenhum'}"}


@no("rotear")
def rotear(estado: Estado) -> dict:
    rota, motivo = escolher_rota(estado["pergunta"], estado["paciente_ids"])
    return {"rota": rota, "_resumo": f"rota={rota} ({motivo})"}


@no("agente_b")
def no_agente_b(estado: Estado) -> dict:
    r = _agente_b().responder(estado["pergunta"])
    fontes = [f"{c['ferramenta']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r.chamadas]
    return {"resposta": r.resposta, "fontes": fontes,
            "_resumo": f"caminho={r.caminho}, {len(r.chamadas)} chamada(s)"}


def resposta_painel(pergunta: str) -> tuple[str, list[str]]:
    """Painel por template a partir de pacientes_com_alerta(): nenhum texto vem de LLM."""
    pendencias = prontuarios.pacientes_com_alerta()["pendencias"]
    quer_atrasado = re.search(r"atrasad|seguimento vencido", pergunta, re.IGNORECASE)
    quer_critico = re.search(r"cr[ií]tic", pergunta, re.IGNORECASE)
    if quer_atrasado and not quer_critico:
        selecao, rotulo = [p for p in pendencias if p["nivel"] == "ATRASADO"], "atrasadas"
    elif quer_critico and not quer_atrasado:
        selecao, rotulo = [p for p in pendencias if p["nivel"] == "CRITICO"], "críticas"
    else:
        selecao, rotulo = pendencias, "com alerta (atrasadas ou críticas)"

    pacientes = sorted({p["paciente_id"] for p in selecao})
    if not selecao:
        return f"Nenhuma pendência {rotulo} na data de referência {DATA_REFERENCIA}.", []
    linhas = [f"Pendências {rotulo} na data de referência {DATA_REFERENCIA}: "
              f"{len(selecao)} pendência(s) em {len(pacientes)} paciente(s)."]
    linhas += [f"- {p['paciente_id']} | {p['nivel']} | {p['exame']} ({p['status']}, previsto "
               f"para {p['data_prevista']}) — {p['justificativa']}" for p in selecao]
    fontes = sorted({p["protocolo_codigo"] for p in selecao if p["protocolo_codigo"]})
    return "\n".join(linhas), fontes


@no("painel")
def no_painel(estado: Estado) -> dict:
    resposta, fontes = resposta_painel(estado["pergunta"])
    return {"resposta": resposta, "fontes": fontes, "_resumo": f"template, fontes={fontes}"}


@no("agente_c")
def no_agente_c(estado: Estado) -> dict:
    r = _agente_c().responder(estado["pergunta"])
    return {"resposta": r.resposta, "fontes": [r.sql_executado] if r.sql_executado else [],
            "_resumo": f"caminho={r.caminho}, n_base={r.n_base}"}


@no("rag")
def no_rag(estado: Estado) -> dict:
    r = _rag().responder(estado["pergunta"])
    fontes = [f["codigo"] for f in r.fontes_recuperadas]
    return {"resposta": r.resposta, "fontes": fontes,
            "_resumo": f"bloqueado_por_limiar={r.bloqueado_por_limiar}, fontes={fontes}"}


# ---------------------------------------------------------------------------
# Modo revisão
# ---------------------------------------------------------------------------

@no("carregar_paciente")
def carregar_paciente(estado: Estado) -> dict:
    pid = estado["paciente_ids"][0]
    dados = prontuarios.obter_paciente(pid)
    if "erro" in dados:
        return {"dados_paciente": dados,
                "resposta": f"O paciente {pid} não foi encontrado no cadastro.\n\n{RODAPE_REVISAO}",
                "_resumo": f"{pid}: {dados['erro']} (encerra cedo)"}
    return {"dados_paciente": dados, "_resumo": f"{pid}: {dados['diagnostico']}"}


@no("avaliar_pendencias")
def avaliar_pendencias(estado: Estado) -> dict:
    pendencias = prontuarios.verificar_pendencias(estado["dados_paciente"]["id"])["pendencias"]
    niveis = [p["nivel"] for p in pendencias]
    return {"pendencias": pendencias, "_resumo": f"{len(pendencias)} pendência(s): {niveis}"}


@no("buscar_protocolos")
def buscar_protocolos(estado: Estado) -> dict:
    por_codigo = _protocolos_por_codigo()
    protocolos = [
        {"codigo": v["codigo"], "titulo": por_codigo[v["codigo"]]["titulo"],
         "texto": por_codigo[v["codigo"]]["conteudo"]}
        for v in estado["dados_paciente"]["protocolos"]
    ]
    return {"protocolos": protocolos, "_resumo": f"vinculados: {[p['codigo'] for p in protocolos]}"}


def montar_pergunta_revisao(dados: dict, pendencias: list[dict]) -> str:
    if pendencias:
        lista = "; ".join(f"{p['exame']} ({p['status']}, previsto para {p['data_prevista']}, "
                          f"nível {p['nivel']})" for p in pendencias)
    else:
        lista = "nenhuma"
    return (f"Paciente {dados['id']}, {dados['idade']} anos, sexo {dados['sexo']}. "
            f"Diagnóstico: {dados['diagnostico']}. Estádio: {dados['estadio'] or 'não informado'}. "
            f"Status do tratamento: {dados['status_tratamento']}. Pendências de exames: {lista}. "
            "Qual a conduta sugerida segundo os protocolos internos? Cite a fonte.")


def mensagens_conduta(dados: dict, pendencias: list[dict], protocolos: list[dict]) -> list:
    """Mesmo formato do fine-tuning v2: system prompt RAG + 'Contexto ... Pergunta: ...'."""
    por_codigo = _protocolos_por_codigo()
    contexto = [{"fonte": por_codigo[p["codigo"]]["fonte"], "titulo": p["titulo"], "conteudo": p["texto"]}
                for p in protocolos]
    usuario = ("Contexto (protocolos internos recuperados):\n"
               f"{format_context(contexto)}\n\nPergunta: {montar_pergunta_revisao(dados, pendencias)}")
    return [SystemMessage(content=SYSTEM_PROMPT_RAG), HumanMessage(content=usuario)]


@no("sugerir_conduta")
def sugerir_conduta(estado: Estado) -> dict:
    conduta = gerar_conduta(mensagens_conduta(estado["dados_paciente"], estado["pendencias"],
                                              estado["protocolos"]))
    fontes = sorted(set(CODIGO_RE.findall(conduta)))
    return {"conduta": conduta, "resposta": conduta, "fontes": fontes,
            "_resumo": f"fontes citadas: {fontes}"}


@no("verificar_saida")
def verificar_saida(estado: Estado) -> dict:
    # Ponto de encaixe dos guardrails de saída (Etapa 6). Por ora, passagem.
    return {"_resumo": "passagem (guardrails na Etapa 6)"}


@no("gerar_alertas")
def gerar_alertas(estado: Estado) -> dict:
    alertas = [p for p in estado["pendencias"] if gera_alerta(p["nivel"])]
    linhas = []
    for a in alertas:
        linha = (f"- [{a['nivel']}] {a['exame']} ({a['status']}, previsto para {a['data_prevista']}): "
                 f"{a['justificativa']}")
        if a["trecho_protocolo"]:
            linha += f' Protocolo {a["protocolo_codigo"]}: "{a["trecho_protocolo"]}"'
        linhas.append(linha)
    bloco = "Alertas:\n" + "\n".join(linhas) if alertas else "Alertas: nenhum."
    resposta = f"Conduta sugerida:\n{estado['conduta']}\n\n{bloco}\n\n{RODAPE_REVISAO}"
    return {"alertas": alertas, "resposta": resposta,
            "_resumo": f"{len(alertas)} alerta(s): {[a['nivel'] for a in alertas]}"}


@no("aprovacao_medica")
def aprovacao_medica(estado: Estado) -> dict:
    decisao = interrupt({"paciente_id": estado["dados_paciente"]["id"],
                         "resposta": estado["resposta"], "alertas": estado["alertas"]})
    if decisao.get("decisao") not in DECISOES:
        raise ValueError(f"decisão inválida: {decisao!r}; use {DECISOES}")
    return {"aprovacao": decisao, "_resumo": f"retomado com decisão {decisao['decisao']}"}


@no("registrar_decisao")
def registrar_decisao(estado: Estado) -> dict:
    d = estado["aprovacao"]
    return {"_resumo": f"{estado['dados_paciente']['id']}: {d['decisao']}"
                       f" | comentário: {d.get('comentario') or '-'}"}


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------

def construir_grafo(checkpointer=None):
    g = StateGraph(Estado)
    for nome, fn in [("extrair_ids", extrair_ids), ("rotear", rotear), ("agente_b", no_agente_b),
                     ("painel", no_painel), ("agente_c", no_agente_c), ("rag", no_rag),
                     ("carregar_paciente", carregar_paciente), ("avaliar_pendencias", avaliar_pendencias),
                     ("buscar_protocolos", buscar_protocolos), ("sugerir_conduta", sugerir_conduta),
                     ("verificar_saida", verificar_saida), ("gerar_alertas", gerar_alertas),
                     ("aprovacao_medica", aprovacao_medica), ("registrar_decisao", registrar_decisao)]:
        g.add_node(nome, fn)

    g.add_conditional_edges(START, lambda e: e["modo"],
                            {"pergunta": "extrair_ids", "revisao": "carregar_paciente"})
    # Modo pergunta
    g.add_edge("extrair_ids", "rotear")
    g.add_conditional_edges("rotear", lambda e: e["rota"],
                            {r: r for r in ("agente_b", "painel", "agente_c", "rag")})
    for rota in ("agente_b", "painel", "agente_c", "rag"):
        g.add_edge(rota, "verificar_saida")
    # Modo revisão
    g.add_conditional_edges("carregar_paciente",
                            lambda e: "fim" if "erro" in e["dados_paciente"] else "seguir",
                            {"fim": END, "seguir": "avaliar_pendencias"})
    g.add_edge("avaliar_pendencias", "buscar_protocolos")
    g.add_edge("buscar_protocolos", "sugerir_conduta")
    g.add_edge("sugerir_conduta", "verificar_saida")
    g.add_conditional_edges("verificar_saida", lambda e: e["modo"],
                            {"pergunta": END, "revisao": "gerar_alertas"})
    g.add_edge("gerar_alertas", "aprovacao_medica")
    g.add_edge("aprovacao_medica", "registrar_decisao")
    g.add_edge("registrar_decisao", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def nova_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def perguntar(grafo, pergunta: str) -> dict:
    return grafo.invoke({"modo": "pergunta", "pergunta": pergunta, "auditoria": []}, nova_config())


def iniciar_revisao(grafo, paciente_id: str) -> tuple[dict, dict]:
    """Roda a revisão até o interrupt (ou até o fim, se o paciente não existir)."""
    config = nova_config()
    estado = grafo.invoke({"modo": "revisao", "paciente_ids": [paciente_id], "auditoria": []}, config)
    return estado, config


def decidir(grafo, config: dict, decisao: str, comentario: str = "") -> dict:
    return grafo.invoke(Command(resume={"decisao": decisao, "comentario": comentario}), config)


def aguardando_aprovacao(grafo, config: dict) -> bool:
    return "aprovacao_medica" in grafo.get_state(config).next


def exportar_diagrama(grafo, destino_dir, png: bool = False) -> list[str]:
    """
    Grava o diagrama Mermaid (local). O PNG é opcional porque draw_mermaid_png()
    envia o diagrama ao serviço externo mermaid.ink para renderizar.
    """
    destino_dir = Path(destino_dir)
    gerados = []
    (destino_dir / "grafo.mmd").write_text(grafo.get_graph().draw_mermaid(), encoding="utf-8")
    gerados.append(str(destino_dir / "grafo.mmd"))
    if png:
        (destino_dir / "grafo.png").write_bytes(grafo.get_graph().draw_mermaid_png())
        gerados.append(str(destino_dir / "grafo.png"))
    return gerados


def _imprimir(estado: dict) -> None:
    print("\nRESPOSTA:\n" + (estado.get("resposta") or ""))
    print("\nCAMINHO:")
    for ev in estado.get("auditoria", []):
        print(f"  {ev['no']:<20} {ev['segundos']:>7.2f}s  {ev['resumo']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--pergunta")
    grupo.add_argument("--revisar", metavar="P000")
    decisao = parser.add_mutually_exclusive_group()
    decisao.add_argument("--aprovar", action="store_true")
    decisao.add_argument("--rejeitar", action="store_true")
    parser.add_argument("--comentario", default="")
    parser.add_argument("--diagrama", action="store_true", help="exporta reports/grafo.mmd")
    parser.add_argument("--png", action="store_true",
                        help="com --diagrama, também gera o PNG (envia o diagrama ao mermaid.ink)")
    args = parser.parse_args()

    grafo = construir_grafo()
    if args.diagrama:
        print("Diagrama:", exportar_diagrama(grafo, Path(__file__).parents[1] / "reports", args.png))

    if args.pergunta:
        _imprimir(perguntar(grafo, args.pergunta))
    else:
        estado, config = iniciar_revisao(grafo, args.revisar)
        if aguardando_aprovacao(grafo, config):
            _imprimir(estado)
            escolha = "aprovado" if args.aprovar else "rejeitado" if args.rejeitar else None
            if escolha is None:
                escolha = input("\nDecisão médica [aprovado/rejeitado]: ").strip().lower()
                args.comentario = args.comentario or input("Comentário: ").strip()
            estado = decidir(grafo, config, escolha, args.comentario)
            print(f"\nDecisão registrada: {estado['aprovacao']}")
        _imprimir(estado)
