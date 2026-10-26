"""
Guardrails do assistente (Etapa 6), em camadas:

1. Entrada (determinística): bloqueia pedidos de automedicação/prescrição em 1ª
   pessoa com resposta fixa; detecta padrões de injeção (não bloqueia, registra
   evento de segurança).
2. Saída determinística (barata): fonte inventada, número sem suporte e status
   de exame sem suporte, reusando as verificações das etapas anteriores.
3. Saída semântica: juiz de fidelidade (llama3.2:3b, temperatura 0, JSON) que
   confere afirmações clínicas contra os protocolos e os dados da paciente. O
   veredito é derivado só das afirmações; JSON inválido -> "indeterminado".
   No grafo o juiz é um SINALIZADOR: não bloqueia nem dispara reparo; o que ele
   não confirmar vira aviso no topo da resposta (ver src/grafo.py). Motivo: na
   avaliação contra respostas rotuladas, o juiz de 3B reprovou a maioria das
   respostas fiéis (ver outputs/juiz_eval/).
   Modos: "completo" (todas as afirmações) e "paciente" (só afirmações sobre a
   paciente, contra os dados dela).
"""

from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache

from .data_augmentation import CODIGO_RE
from .evaluate_agente import _status_retornados, normalizar, termos_status_afirmados
from .rag_pipeline import numeros_sem_suporte

logger = logging.getLogger(__name__)

MODELO_JUIZ = "llama3.2:3b"

AUTOMEDICACAO_RE = re.compile(
    r"\b(devo tomar|posso tomar|quanto (eu )?tomo|me receite|qual dose (eu )?devo)\b", re.IGNORECASE)
INJECAO_RE = re.compile(r"\b(ignore (as|todas as) instruç|desconsidere|system prompt)", re.IGNORECASE)
RESPOSTA_AUTOMEDICACAO = ("Não posso orientar uso de medicamentos para você. Procure seu médico ou, "
                          "em urgência, o pronto atendimento.")

VEREDITOS_REPROVADOS = ("infiel", "indeterminado")


def aviso_juiz(juiz: dict) -> str:
    """Aviso colocado no topo da resposta quando o juiz não confirma afirmações."""
    if juiz["afirmacoes_sem_suporte"]:
        itens = "; ".join(f"«{a}»" for a in juiz["afirmacoes_sem_suporte"])
        return f"⚠ Verificação automática não confirmou: {itens}. Confira no protocolo antes de usar."
    return "⚠ Verificação automática não pôde ser concluída. Confira no protocolo antes de usar."


# ---------------------------------------------------------------------------
# 1. Entrada
# ---------------------------------------------------------------------------

def filtro_entrada(pergunta: str) -> dict | None:
    """Bloqueio da entrada: None ou {motivo, resposta_fixa}."""
    if m := AUTOMEDICACAO_RE.search(pergunta):
        return {"motivo": f"pedido de automedicação/prescrição em 1ª pessoa ('{m.group()}')",
                "resposta_fixa": RESPOSTA_AUTOMEDICACAO}
    return None


def eventos_seguranca_entrada(pergunta: str) -> list[dict]:
    """Padrões de injeção: não bloqueiam, mas viram evento de segurança."""
    return [{"tipo": "possivel_injecao", "trecho": m.group()} for m in INJECAO_RE.finditer(pergunta)]


# ---------------------------------------------------------------------------
# 2. Saída determinística
# ---------------------------------------------------------------------------

def verificar_deterministico(
    resposta: str,
    fontes_texto: list[str],
    dados_paciente: dict | None,
    pergunta: str,
    codigos_permitidos: set[str] | None,
    status_retornados: set[str] | None = None,
    verificar_numeros: bool = True,
) -> list[dict]:
    """
    Problemas encontrados ({tipo, detalhe}):
      fonte_inventada     código citado fora de codigos_permitidos
      numero_sem_suporte  número ausente dos textos-fonte, dos dados da paciente e da pergunta
      status_sem_suporte  status de exame afirmado e não devolvido pelas ferramentas
                          (só quando status_retornados é informado)
    """
    problemas = []
    if codigos_permitidos is not None:
        if inventados := sorted(set(CODIGO_RE.findall(resposta)) - set(codigos_permitidos)):
            problemas.append({"tipo": "fonte_inventada", "detalhe": inventados})
    if verificar_numeros:
        suporte = list(fontes_texto) + [pergunta]
        if dados_paciente:
            suporte.append(json.dumps(dados_paciente, ensure_ascii=False))
        if sem := sorted(numeros_sem_suporte(resposta, suporte)):
            problemas.append({"tipo": "numero_sem_suporte", "detalhe": sem})
    if status_retornados is not None:
        retornados = {normalizar(s)[:-1] for s in status_retornados}
        if sem := [t for t in termos_status_afirmados(normalizar(resposta)) if t not in retornados]:
            problemas.append({"tipo": "status_sem_suporte", "detalhe": sem})
    return problemas


def status_das_chamadas(chamadas: list[dict]) -> set[str]:
    return _status_retornados([c["resultado"] for c in chamadas if not c.get("rejeitada")])


# ---------------------------------------------------------------------------
# 3. Juiz de fidelidade
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_JUIZ = """Você é um verificador de fidelidade de respostas clínicas.
Liste cada afirmação clínica da RESPOSTA e verifique se ela tem suporte nos PROTOCOLOS e nos DADOS DA PACIENTE.

Regras:
1. Tem suporte a afirmação dita nos PROTOCOLOS (ou consequência direta do que está escrito) ou o fato sobre a paciente presente nos DADOS DA PACIENTE.
2. Condições do protocolo (ex.: "em HER2 positivo", "em pacientes abaixo de 35 anos", "em hormonioterapia") só se aplicam à paciente se os DADOS DA PACIENTE satisfizerem a condição. Aplicar à paciente uma conduta cuja condição ela não satisfaz é SEM SUPORTE.
3. Afirmações sobre a paciente (doenças, diagnóstico, tratamento, status) ausentes dos DADOS DA PACIENTE são SEM SUPORTE.
4. Números, doses, faixas e prazos devem corresponder ao protocolo, inclusive à categoria a que se referem.
5. Omitir informação NÃO é falta de fidelidade. Ignore a linha "Fonte:" e frases de cautela (ex.: "consulte o médico responsável").

Responda APENAS com JSON no formato:
{"afirmacoes": [{"texto": "...", "suporte": "sim" ou "nao", "motivo": "..."}], "veredito": "fiel" ou "infiel"}
O veredito é "infiel" se qualquer afirmação tiver suporte "nao"."""


@lru_cache(maxsize=1)
def _llm_juiz():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=MODELO_JUIZ, temperature=0, format="json")


def _mensagem_juiz(resposta: str, protocolos_texto: list[str], dados_paciente: dict | None) -> str:
    protocolos = "\n\n".join(protocolos_texto) or "(nenhum)"
    dados = (json.dumps(dados_paciente, ensure_ascii=False, indent=1) if dados_paciente
             else "não se aplica (pergunta geral, sem paciente específica)")
    return f"PROTOCOLOS:\n{protocolos}\n\nDADOS DA PACIENTE:\n{dados}\n\nRESPOSTA:\n{resposta}"


SYSTEM_PROMPT_JUIZ_PACIENTE = """Você é um verificador de fidelidade de respostas clínicas sobre UMA paciente.
Liste APENAS as afirmações que a RESPOSTA faz SOBRE A PACIENTE e verifique cada uma contra os DADOS DA PACIENTE.

São afirmações sobre a paciente: diagnóstico, doenças, receptores/subtipo (ex.: HER2 positivo, triplo-negativo), estádio, status do tratamento, exames e pendências.
Também conta como afirmação sobre a paciente uma conduta que pressupõe uma característica dela (ex.: "adicionar trastuzumabe por ser HER2 positivo", "por estar em hormonioterapia"): a característica precisa estar nos DADOS DA PACIENTE.

Regras:
1. Tem suporte o que está nos DADOS DA PACIENTE ou é consequência direta deles (ex.: "carcinoma triplo-negativo" implica HER2 negativo).
2. Característica da paciente ausente ou contrária aos DADOS DA PACIENTE é SEM SUPORTE.
3. NÃO avalie afirmações gerais sobre o protocolo, doses ou critérios que não falem da paciente; deixe-as fora da lista.
4. Omitir informação NÃO é falta de fidelidade. Ignore a linha "Fonte:".

Responda APENAS com JSON no formato:
{"afirmacoes": [{"texto": "...", "suporte": "sim" ou "nao", "motivo": "..."}], "veredito": "fiel" ou "infiel"}"""

PROMPTS_JUIZ = {"completo": SYSTEM_PROMPT_JUIZ, "paciente": SYSTEM_PROMPT_JUIZ_PACIENTE}


def interpretar_juiz(conteudo: str) -> dict:
    """
    JSON do juiz -> {veredito, afirmacoes, afirmacoes_sem_suporte, veredito_modelo}.
    O veredito é derivado SÓ das afirmações (infiel se houver ao menos uma sem
    suporte); o campo "veredito" escrito pelo juiz é registrado, mas ignorado.
    JSON inválido ou sem lista de afirmações -> indeterminado.
    """
    try:
        dados = json.loads(conteudo)
        afirmacoes = dados["afirmacoes"]
        if not isinstance(afirmacoes, list):
            raise ValueError("afirmacoes não é lista")
        sem_suporte = [str(a.get("texto", "")) for a in afirmacoes
                       if str(a.get("suporte", "")).strip().lower() in ("nao", "não", "false", "no")]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
        return {"veredito": "indeterminado", "afirmacoes": [], "afirmacoes_sem_suporte": [],
                "veredito_modelo": None, "erro": f"{type(e).__name__}: {e}"}
    return {"veredito": "infiel" if sem_suporte else "fiel", "afirmacoes": afirmacoes,
            "afirmacoes_sem_suporte": sem_suporte,
            "veredito_modelo": str(dados.get("veredito", "")).strip().lower() or None}


def juiz_fidelidade(resposta: str, protocolos_texto: list[str], dados_paciente: dict | None = None,
                    modo: str = "completo") -> dict:
    """modo 'completo': todas as afirmações; 'paciente': só as que falam da paciente."""
    inicio = time.time()
    from langchain_core.messages import HumanMessage, SystemMessage
    saida = _llm_juiz().invoke([SystemMessage(content=PROMPTS_JUIZ[modo]),
                                HumanMessage(content=_mensagem_juiz(resposta, protocolos_texto,
                                                                    dados_paciente))])
    resultado = interpretar_juiz(saida.content or "")
    resultado["modo"] = modo
    resultado["segundos"] = round(time.time() - inicio, 2)
    logger.info(f"[juiz:{modo}] veredito={resultado['veredito']} | "
                f"sem suporte={resultado['afirmacoes_sem_suporte']} | {resultado['segundos']}s")
    return resultado


def reprovado(veredito: str) -> bool:
    return veredito in VEREDITOS_REPROVADOS
