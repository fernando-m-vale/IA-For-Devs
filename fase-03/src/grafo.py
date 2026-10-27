"""
Grafo LangGraph do assistente médico (Etapas 5 e 6).

MODO PERGUNTA
  extrair_ids -> entrada -> rotear -> {agente_b | painel | agente_c | rag}
  -> verificar_saida -> [reparar -> verificar_saida] -> [recurso_seguro] -> finalizar
  entrada: guardrail determinístico (automedicação em 1ª pessoa -> resposta fixa;
  padrões de injeção -> evento de segurança, sem bloquear).
  Roteamento (determinístico, nesta ordem):
    a) a pergunta cita um ID P000          -> agente_b (ferramentas por paciente)
    b) vocabulário de painel               -> painel (template, SEM LLM)
    c) vocabulário de estatística          -> agente_c (text-to-SQL protegido)
    d) caso contrário                      -> rag (protocolos internos)
  Se casar painel e estatística, vence o painel: é o caminho sem LLM, e o
  text-to-SQL errou justamente perguntas de painel na Etapa 4.

MODO REVISÃO (um paciente)
  carregar_paciente -> avaliar_pendencias -> buscar_protocolos -> sugerir_conduta
  -> verificar_saida -> [reparar -> verificar_saida] -> [recurso_seguro]
  -> gerar_alertas -> aprovacao_medica (interrupt) -> registrar_decisao -> finalizar
  Os protocolos vêm do vínculo paciente_protocolos (sem busca semântica); os
  alertas são montados por template a partir das pendências ATRASADO/CRITICO.
  Paciente inexistente encerra cedo com mensagem fixa.

verificar_saida (Etapa 6):
  rag e revisão: verificação determinística (fonte inventada, número sem
                 suporte; status na revisão) + juiz de fidelidade;
  agente_b:      só status sem suporte (determinístico);
  painel, agente_c, bloqueios e respostas fixas: sem verificação (texto montado
                 pelo código, sem LLM).
  Problema DETERMINÍSTICO: 1 reparo (o modelo regera com a lista de problemas); se
  falhar de novo, recurso_seguro (resposta montada pelo código, sem texto do modelo).
  O juiz é SINALIZADOR: não bloqueia nem dispara reparo. O que ele não confirmar
  vira aviso no topo da resposta (ou da conduta) e sinalizado_pelo_juiz=True na
  auditoria; na revisão, o médico vê o aviso antes de aprovar.

Auditoria (src/auditoria.py): no modo revisão, gerar_alertas grava o registro
"pendente_aprovacao" e finalizar grava o "decisao" (mesmo id); nos demais casos,
finalizar grava um registro "execucao". No modo pergunta, finalizar também
acrescenta o bloco "Por que esta resposta" (no modo revisão ele vem antes do rodapé).

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

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import auditoria, prontuarios
from .config import DATA_REFERENCIA
from .data_augmentation import CODIGO_RE, SYSTEM_PROMPT_RAG, format_context
from .guardrails import (MODELO_JUIZ, aviso_juiz, eventos_seguranca_entrada, filtro_entrada,
                         juiz_fidelidade, reprovado, status_das_chamadas, verificar_deterministico)
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
MODELO_AGENTE_B = "llama3.2:3b"
MAX_REPAROS = 1
NUM_PREDICT_CONDUTA = 600  # sugerir_conduta e reparos
JUIZ_MODO_RAG = "completo"
JUIZ_MODO_REVISAO = "completo"
RODAPE_REVISAO = "Sugestão de apoio à decisão — requer validação médica."
CONDUTA_NAO_GERADA = "Conduta não gerada — falhou na verificação."
DECISOES = ("aprovado", "rejeitado")


# ---------------------------------------------------------------------------
# Estado e auditoria
# ---------------------------------------------------------------------------

class Estado(TypedDict, total=False):
    id_execucao: str                 # mesmo id em todos os registros de auditoria da execução
    modo: Literal["pergunta", "revisao"]
    pergunta: str
    paciente_ids: list[str]
    rota: str
    motivo_rota: str
    bloqueio: dict | None
    dados_paciente: dict
    pendencias: list[dict]
    protocolos: list[dict]           # {codigo, titulo, texto}
    conduta: str                     # texto do LLM (modo revisão)
    resposta: str
    fontes: list[str]
    fontes_detalhe: list[dict]       # {codigo, titulo, trecho} para o "Por que esta resposta"
    alertas: list[dict]
    aprovacao: dict
    verificacao: dict                # contexto do que verificar na saída
    chamadas_b: list[dict]           # retorno das ferramentas do agente B
    resposta_c: dict                 # resultado completo do agente C (SQL, linhas, supressão)
    rag_doc_ids: list[str]
    rag_bloqueado_limiar: bool
    status_verificacao: str          # ok | reparar | recurso_seguro
    juiz: dict | None
    sinalizado_pelo_juiz: bool       # o juiz não confirmou afirmações (aviso, sem bloqueio)
    reparos: int
    verificacoes: Annotated[list[dict], operator.add]
    guardrails: Annotated[list[str], operator.add]
    eventos_seguranca: Annotated[list[dict], operator.add]
    modelos: Annotated[list[str], operator.add]
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


@lru_cache(maxsize=2)
def _llm(modelo: str):
    from langchain_ollama import ChatOllama
    temperatura = 0.3 if modelo == MODELO_CONDUTA else 0
    # Teto de tokens por precaução (ver guardrails.NUM_PREDICT_JUIZ); a maior resposta
    # final medida tem ~300 tokens.
    return ChatOllama(model=modelo, temperature=temperatura, num_ctx=4096, num_predict=NUM_PREDICT_CONDUTA)


def gerar_conduta(mensagens: list) -> str:
    """Chamada de LLM do modo revisão (geração e reparo; substituída por mock nos testes)."""
    return _llm(MODELO_CONDUTA).invoke(mensagens).content.strip()


def regerar(modelo: str, mensagens: list) -> str:
    """Reparo das rotas rag e agente_b."""
    return _llm(modelo).invoke(mensagens).content.strip()


@lru_cache(maxsize=1)
def _protocolos_por_codigo() -> dict[str, dict]:
    protocolos = json.loads((DATA_DIR / "protocolos.json").read_text(encoding="utf-8"))
    return {CODIGO_RE.search(p["fonte"]).group(): p for p in protocolos}


def _trecho(texto: str, n: int = 180) -> str:
    return texto if len(texto) <= n else texto[:n].rsplit(" ", 1)[0] + "…"


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


@no("entrada")
def entrada(estado: Estado) -> dict:
    bloqueio = filtro_entrada(estado["pergunta"])
    eventos = eventos_seguranca_entrada(estado["pergunta"])
    saida = {"bloqueio": bloqueio, "eventos_seguranca": eventos}
    if eventos:
        logger.warning(f"[seguranca] padrão de injeção na entrada: {[e['trecho'] for e in eventos]}")
    if bloqueio:
        saida.update(resposta=bloqueio["resposta_fixa"], rota="bloqueio_entrada",
                     motivo_rota=bloqueio["motivo"], guardrails=["entrada_automedicacao"])
        saida["_resumo"] = f"BLOQUEADO: {bloqueio['motivo']}"
    else:
        saida["_resumo"] = f"liberado; eventos de segurança: {[e['trecho'] for e in eventos] or 'nenhum'}"
    return saida


@no("rotear")
def rotear(estado: Estado) -> dict:
    rota, motivo = escolher_rota(estado["pergunta"], estado["paciente_ids"])
    return {"rota": rota, "motivo_rota": motivo, "_resumo": f"rota={rota} ({motivo})"}


@no("agente_b")
def no_agente_b(estado: Estado) -> dict:
    r = _agente_b().responder(estado["pergunta"])
    fontes = [f"{c['ferramenta']}({json.dumps(c['args'], ensure_ascii=False)})" for c in r.chamadas]
    verificacao = {"ativa": r.caminho == "llm" and bool(r.chamadas), "juiz": False, "numeros": False,
                   "status": sorted(status_das_chamadas(r.chamadas)), "fontes_texto": [],
                   "pergunta": estado["pergunta"]}
    return {"resposta": r.resposta, "fontes": fontes, "chamadas_b": r.chamadas,
            "modelos": [MODELO_AGENTE_B], "verificacao": verificacao,
            "fontes_detalhe": [{"codigo": f, "titulo": "ferramenta de prontuário", "trecho": ""}
                               for f in fontes],
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
    por_codigo = _protocolos_por_codigo()
    detalhe = [{"codigo": c, "titulo": por_codigo[c]["titulo"], "trecho": ""} for c in fontes]
    return {"resposta": resposta, "fontes": fontes, "fontes_detalhe": detalhe,
            "verificacao": {"ativa": False}, "_resumo": f"template, fontes={fontes}"}


@no("agente_c")
def no_agente_c(estado: Estado) -> dict:
    from .agente_sql import MODELO_SQL
    r = _agente_c().responder(estado["pergunta"])
    fontes = [r.sql_executado] if r.sql_executado else []
    return {"resposta": r.resposta, "fontes": fontes, "modelos": [MODELO_SQL], "resposta_c": r.to_dict(),
            "fontes_detalhe": [{"codigo": "SQL", "titulo": s, "trecho": ""} for s in fontes],
            "verificacao": {"ativa": False}, "_resumo": f"caminho={r.caminho}, n_base={r.n_base}"}


@no("rag")
def no_rag(estado: Estado) -> dict:
    rag = _rag()
    r = rag.responder(estado["pergunta"])
    fontes = [f["codigo"] for f in r.fontes_recuperadas]
    verificacao = {"ativa": not r.bloqueado_por_limiar, "juiz": True, "juiz_modo": JUIZ_MODO_RAG,
                   "numeros": True, "codigos": fontes, "fontes_texto": [f"[{f['codigo']}] {f['trecho']}"
                                                       for f in r.fontes_recuperadas],
                   "pergunta": estado["pergunta"]}
    return {"resposta": r.resposta, "fontes": fontes, "modelos": [rag.modelo],
            "rag_doc_ids": [f["id"] for f in r.fontes_recuperadas], "verificacao": verificacao,
            "rag_bloqueado_limiar": r.bloqueado_por_limiar,
            "fontes_detalhe": [{"codigo": f["codigo"], "titulo": f["titulo"], "trecho": _trecho(f["trecho"])}
                               for f in r.fontes_recuperadas],
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


def dados_para_verificacao(dados: dict, pendencias: list[dict]) -> dict:
    """Dados da paciente que o juiz e a checagem numérica podem usar como suporte."""
    return {
        "id": dados["id"], "idade": dados["idade"], "sexo": dados["sexo"],
        "diagnostico": dados["diagnostico"], "estadio": dados["estadio"],
        "status_tratamento": dados["status_tratamento"], "ultima_consulta": dados["ultima_consulta"],
        "pendencias": [{"exame": p["exame"], "status": p["status"], "data_prevista": p["data_prevista"],
                        "nivel": p["nivel"]} for p in pendencias],
    }


@no("sugerir_conduta")
def sugerir_conduta(estado: Estado) -> dict:
    dados, pendencias, protocolos = estado["dados_paciente"], estado["pendencias"], estado["protocolos"]
    conduta = gerar_conduta(mensagens_conduta(dados, pendencias, protocolos))
    fontes = sorted(set(CODIGO_RE.findall(conduta)))
    exames = prontuarios.listar_exames(dados["id"])["exames"]
    verificacao = {"ativa": True, "juiz": True, "juiz_modo": JUIZ_MODO_REVISAO, "numeros": True,
                   "codigos": [p["codigo"] for p in protocolos],
                   "fontes_texto": [f"[{p['codigo']}] {p['texto']}" for p in protocolos],
                   "dados": dados_para_verificacao(dados, pendencias),
                   "status": sorted({e["status"] for e in exames}),
                   "pergunta": montar_pergunta_revisao(dados, pendencias)}
    return {"conduta": conduta, "resposta": conduta, "fontes": fontes, "modelos": [MODELO_CONDUTA],
            "verificacao": verificacao, "reparos": 0,
            "fontes_detalhe": [{"codigo": p["codigo"], "titulo": p["titulo"], "trecho": _trecho(p["texto"])}
                               for p in protocolos],
            "_resumo": f"fontes citadas: {fontes}"}


# ---------------------------------------------------------------------------
# Verificação de saída, reparo e recurso seguro (Etapa 6)
# ---------------------------------------------------------------------------

def _texto_verificado(estado: Estado) -> str:
    return estado["conduta"] if estado["modo"] == "revisao" else estado["resposta"]


@no("verificar_saida")
def verificar_saida(estado: Estado) -> dict:
    ctx = estado.get("verificacao") or {}
    if not ctx.get("ativa"):
        return {"status_verificacao": "ok",
                "_resumo": "sem verificação: resposta montada pelo código ou fixa"}

    texto, reparos = _texto_verificado(estado), estado.get("reparos", 0)
    status = set(ctx["status"]) if ctx.get("status") is not None else None
    problemas = verificar_deterministico(
        texto, ctx.get("fontes_texto", []), ctx.get("dados"), ctx.get("pergunta", ""),
        set(ctx["codigos"]) if ctx.get("codigos") is not None else None,
        status_retornados=status, verificar_numeros=ctx.get("numeros", True))
    juiz, modelos = None, []
    if ctx.get("juiz"):
        juiz = juiz_fidelidade(texto, ctx.get("fontes_texto", []), ctx.get("dados"),
                               modo=ctx.get("juiz_modo", "completo"))
        modelos = [MODELO_JUIZ]

    # Só os problemas determinísticos bloqueiam (reparo -> recurso seguro).
    # O juiz é sinalizador: o que ele não confirmar vira aviso no topo da resposta.
    if not problemas:
        decisao = "ok"
    elif reparos < MAX_REPAROS:
        decisao = "reparar"
    else:
        decisao = "recurso_seguro"

    saida = {"status_verificacao": decisao, "juiz": juiz, "modelos": modelos,
             "guardrails": [p["tipo"] for p in problemas],
             "verificacoes": [{"tentativa": reparos + 1, "texto": texto, "problemas": problemas,
                               "veredito_juiz": juiz["veredito"] if juiz else None,
                               "sinalizacoes_juiz": juiz["afirmacoes_sem_suporte"] if juiz else [],
                              "juiz_execucao": {k: juiz.get(k) for k in ("tokens_gerados", "truncado",
                                                                         "timeout", "motivo", "segundos")}
                              if juiz else None}]}
    sinalizado = bool(juiz) and reprovado(juiz["veredito"]) and decisao == "ok"
    if sinalizado:
        chave = "conduta" if estado["modo"] == "revisao" else "resposta"
        aviso = aviso_juiz(juiz, objeto=chave)
        saida.update({chave: f"{aviso}\n\n{texto}", "sinalizado_pelo_juiz": True,
                      "guardrails": saida["guardrails"] + ["juiz_sinalizou"]})
        if estado["modo"] == "revisao":
            saida["resposta"] = saida["conduta"]
    saida["_resumo"] = (f"tentativa {reparos + 1}: {[p['tipo'] for p in problemas] or 'sem problemas'}"
                        f"{' | juiz=' + juiz['veredito'] if juiz else ''}"
                        f"{' (sinalizado, sem bloqueio)' if sinalizado else ''} -> {decisao}")
    return saida


def _feedback(problemas: list[dict]) -> str:
    descricoes = {
        "fonte_inventada": "cita protocolo(s) que não estão no contexto: {}",
        "numero_sem_suporte": "contém número(s) que não aparecem no contexto nem nos dados: {}",
        "status_sem_suporte": "atribui status de exame que não foi retornado: {}",
        "juiz_infiel": "afirmações sem suporte nos protocolos ou nos dados da paciente: {}",
        "juiz_indeterminado": "a verificação de fidelidade não pôde confirmar a resposta {}",
    }
    linhas = []
    for p in problemas:
        if p["tipo"] == "inconsistente_com_perfil":
            linhas.append(f"- A paciente é {p['detalhe']['subtipo']}. Terapia anti-HER2 e o monitoramento "
                          "cardíaco associado não se aplicam; remova essas recomendações. Protocolo "
                          f"HOSP-2024-ONCO-004: \"{p['detalhe']['trecho_protocolo']}\"")
        else:
            linhas.append(f"- {descricoes.get(p['tipo'], p['tipo'] + ': {}').format(p['detalhe'])}")
    return ("A resposta anterior falhou na verificação:\n" + "\n".join(linhas) +
            "\nReescreva a resposta corrigindo exatamente esses problemas. Use somente o que está "
            "no contexto e nos dados fornecidos; não acrescente informações novas.")


@no("reparar")
def reparar(estado: Estado) -> dict:
    problemas = estado["verificacoes"][-1]["problemas"]
    feedback = HumanMessage(content=_feedback(problemas))
    anterior = AIMessage(content=_texto_verificado(estado))

    if estado["modo"] == "revisao":
        msgs = mensagens_conduta(estado["dados_paciente"], estado["pendencias"], estado["protocolos"])
        conduta = gerar_conduta(msgs + [anterior, feedback])
        saida = {"conduta": conduta, "resposta": conduta, "fontes": sorted(set(CODIGO_RE.findall(conduta)))}
    elif estado["rota"] == "rag":
        from .rag_pipeline import PROMPTS, montar_mensagem_usuario
        rag = _rag()
        docs = [rag.buscador.por_id[i] for i in estado["rag_doc_ids"]]
        msgs = [SystemMessage(content=PROMPTS[rag.variante_prompt]),
                HumanMessage(content=montar_mensagem_usuario(estado["pergunta"], docs))]
        saida = {"resposta": regerar(rag.modelo, msgs + [anterior, feedback])}
    else:  # agente_b: reescreve a partir do retorno das ferramentas, sem novas chamadas
        from .agente_prontuarios import SYSTEM_PROMPT_AGENTE
        dados = [{"ferramenta": c["ferramenta"], "resultado": c["resultado"]}
                 for c in estado["chamadas_b"] if not c.get("rejeitada")]
        msgs = [SystemMessage(content=SYSTEM_PROMPT_AGENTE), HumanMessage(content=estado["pergunta"]),
                HumanMessage(content="Dados retornados pelas ferramentas:\n"
                                     + json.dumps(dados, ensure_ascii=False))]
        saida = {"resposta": regerar(MODELO_AGENTE_B, msgs + [anterior, feedback])}

    return {**saida, "reparos": estado.get("reparos", 0) + 1,
            "_resumo": f"regerado com feedback: {[p['tipo'] for p in problemas]}"}


def template_b(chamadas: list[dict]) -> str:
    linhas = ["Resposta montada diretamente dos dados do prontuário (a resposta do modelo não "
              "passou na verificação):"]
    for c in chamadas:
        if c.get("rejeitada"):
            continue
        r = c["resultado"]
        if "erro" in r:
            linhas.append(f"- {c['ferramenta']}: {r['erro']}")
        elif c["ferramenta"] == "obter_paciente":
            linhas.append(f"- {r['id']}: {r['idade']} anos; diagnóstico: {r['diagnostico']}; estádio: "
                          f"{r['estadio'] or 'não informado'}; status: {r['status_tratamento']}; "
                          f"última consulta: {r['ultima_consulta']}")
        elif c["ferramenta"] == "listar_exames":
            linhas += [f"- {r['paciente_id']}: {e['exame']} — {e['status']}, previsto para "
                       f"{e['data_prevista'] or 'sem data'}" + (f"; resultado: {e['resultado']}"
                                                                if e["resultado"] else "")
                       for e in r["exames"]] or [f"- {r['paciente_id']}: nenhum exame"]
        elif c["ferramenta"] == "verificar_pendencias":
            linhas += [f"- {r['paciente_id']}: [{p['nivel']}] {p['exame']} — {p['status']}, previsto "
                       f"para {p['data_prevista']}: {p['justificativa']}"
                       for p in r["pendencias"]] or [f"- {r['paciente_id']}: nenhuma pendência"]
    return "\n".join(linhas)


@no("recurso_seguro")
def recurso_seguro(estado: Estado) -> dict:
    if estado["modo"] == "revisao":
        return {"conduta": CONDUTA_NAO_GERADA, "resposta": CONDUTA_NAO_GERADA, "fontes": [],
                "guardrails": ["recurso_seguro"], "_resumo": "conduta omitida; seguem só alertas e trechos"}
    if estado["rota"] == "rag":
        detalhe = (estado.get("fontes_detalhe") or [None])[0]
        por_codigo = _protocolos_por_codigo()
        texto = ("Não foi possível gerar uma resposta que passasse na verificação de fidelidade. "
                 "Consulte diretamente o trecho do protocolo mais relevante:")
        if detalhe:
            texto += f"\n[{detalhe['codigo']}] {detalhe['titulo']}: {por_codigo[detalhe['codigo']]['conteudo']}"
        return {"resposta": texto, "guardrails": ["recurso_seguro"], "_resumo": "aviso + trecho do protocolo"}
    return {"resposta": template_b(estado.get("chamadas_b", [])), "guardrails": ["recurso_seguro"],
            "_resumo": "resposta por template a partir do retorno das ferramentas"}


# ---------------------------------------------------------------------------
# Alertas, aprovação, explicação e auditoria
# ---------------------------------------------------------------------------

def bloco_porque(estado: Estado) -> str:
    linhas = ["Por que esta resposta:",
              f"- Rota: {estado.get('rota') or estado.get('modo')}"
              + (f" ({estado['motivo_rota']})" if estado.get("motivo_rota") else "")]
    if estado.get("fontes_detalhe"):
        linhas.append("- Fontes:")
        linhas += [f"  • {f['codigo']} — {f['titulo']}" + (f": \"{f['trecho']}\"" if f["trecho"] else "")
                   for f in estado["fontes_detalhe"]]
    verificacoes = estado.get("verificacoes", [])
    if verificacoes:
        linhas.append("- Verificações: " + "; ".join(
            f"tentativa {v['tentativa']}: {[p['tipo'] for p in v['problemas']] or 'ok'}" for v in verificacoes))
        if estado.get("juiz"):
            j = estado["juiz"]
            linha = f"- Juiz de fidelidade (experimental, não exaustivo): {j['veredito']}"
            if estado.get("sinalizado_pelo_juiz"):
                linha += " — sinalizador, não bloqueia; dúvidas levantadas: " + (
                    "; ".join(f"«{a}»" for a in j["afirmacoes_sem_suporte"]) or "verificação inconclusiva")
            linhas.append(linha)
    else:
        linhas.append("- Verificações: não aplicável (texto montado pelo código ou resposta fixa)")
    linhas.append(f"- Reparos: {estado.get('reparos', 0)}")
    if "recurso_seguro" in estado.get("guardrails", []):
        linhas.append("- Recurso seguro acionado: o texto do modelo foi descartado")
    return "\n".join(linhas)


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
    resposta = (f"Conduta sugerida:\n{estado['conduta']}\n\n{bloco}\n\n{bloco_porque(estado)}"
                f"\n\n{RODAPE_REVISAO}")
    # Registro gravado aqui, e não em aprovacao_medica: na retomada o LangGraph
    # reexecuta o nó do interrupt, o que duplicaria o registro.
    auditoria.registrar_execucao({**estado, "alertas": alertas, "resposta": resposta},
                                 tipo="pendente_aprovacao")
    return {"alertas": alertas, "resposta": resposta,
            "_resumo": f"{len(alertas)} alerta(s): {[a['nivel'] for a in alertas]}; "
                       "auditoria: pendente_aprovacao"}


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


@no("finalizar")
def finalizar(estado: Estado) -> dict:
    saida = {}
    if estado["modo"] == "pergunta":
        saida["resposta"] = f"{estado.get('resposta', '')}\n\n{bloco_porque(estado)}"
    tipo = "decisao" if estado.get("aprovacao") else "execucao"
    registro = auditoria.registrar_execucao({**estado, **saida}, tipo=tipo)
    return {**saida, "_resumo": f"auditoria: {tipo} (id {registro['id']})"}


# ---------------------------------------------------------------------------
# Montagem do grafo
# ---------------------------------------------------------------------------

def _apos_verificacao(estado: Estado) -> str:
    if estado["status_verificacao"] in ("reparar", "recurso_seguro"):
        return estado["status_verificacao"]
    return "gerar_alertas" if estado["modo"] == "revisao" else "finalizar"


def construir_grafo(checkpointer=None):
    g = StateGraph(Estado)
    for nome, fn in [("extrair_ids", extrair_ids), ("entrada", entrada), ("rotear", rotear),
                     ("agente_b", no_agente_b), ("painel", no_painel), ("agente_c", no_agente_c),
                     ("rag", no_rag), ("carregar_paciente", carregar_paciente),
                     ("avaliar_pendencias", avaliar_pendencias), ("buscar_protocolos", buscar_protocolos),
                     ("sugerir_conduta", sugerir_conduta), ("verificar_saida", verificar_saida),
                     ("reparar", reparar), ("recurso_seguro", recurso_seguro),
                     ("gerar_alertas", gerar_alertas), ("aprovacao_medica", aprovacao_medica),
                     ("registrar_decisao", registrar_decisao), ("finalizar", finalizar)]:
        g.add_node(nome, fn)

    g.add_conditional_edges(START, lambda e: e["modo"],
                            {"pergunta": "extrair_ids", "revisao": "carregar_paciente"})
    # Modo pergunta
    g.add_edge("extrair_ids", "entrada")
    g.add_conditional_edges("entrada", lambda e: "bloqueado" if e.get("bloqueio") else "liberado",
                            {"bloqueado": "finalizar", "liberado": "rotear"})
    g.add_conditional_edges("rotear", lambda e: e["rota"],
                            {r: r for r in ("agente_b", "painel", "agente_c", "rag")})
    for rota in ("agente_b", "painel", "agente_c", "rag"):
        g.add_edge(rota, "verificar_saida")
    # Modo revisão
    g.add_conditional_edges("carregar_paciente",
                            lambda e: "fim" if "erro" in e["dados_paciente"] else "seguir",
                            {"fim": "finalizar", "seguir": "avaliar_pendencias"})
    g.add_edge("avaliar_pendencias", "buscar_protocolos")
    g.add_edge("buscar_protocolos", "sugerir_conduta")
    g.add_edge("sugerir_conduta", "verificar_saida")
    # Verificação, reparo e recurso seguro (comuns aos dois modos)
    g.add_conditional_edges("verificar_saida", _apos_verificacao,
                            {"reparar": "reparar", "recurso_seguro": "recurso_seguro",
                             "gerar_alertas": "gerar_alertas", "finalizar": "finalizar"})
    g.add_edge("reparar", "verificar_saida")
    g.add_conditional_edges("recurso_seguro", lambda e: "gerar_alertas" if e["modo"] == "revisao" else "finalizar",
                            {"gerar_alertas": "gerar_alertas", "finalizar": "finalizar"})
    g.add_edge("gerar_alertas", "aprovacao_medica")
    g.add_edge("aprovacao_medica", "registrar_decisao")
    g.add_edge("registrar_decisao", "finalizar")
    g.add_edge("finalizar", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def nova_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def perguntar(grafo, pergunta: str) -> dict:
    return grafo.invoke({"id_execucao": auditoria.novo_id(), "modo": "pergunta", "pergunta": pergunta,
                         "auditoria": []}, nova_config())


def iniciar_revisao(grafo, paciente_id: str) -> tuple[dict, dict]:
    """Roda a revisão até o interrupt (ou até o fim, se o paciente não existir)."""
    config = nova_config()
    estado = grafo.invoke({"id_execucao": auditoria.novo_id(), "modo": "revisao",
                           "paciente_ids": [paciente_id], "auditoria": []}, config)
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
