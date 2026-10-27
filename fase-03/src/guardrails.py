"""
Guardrails do assistente (Etapa 6), em camadas:

1. Entrada (determinística): bloqueia pedidos de automedicação/prescrição em 1ª
   pessoa com resposta fixa; detecta padrões de injeção (não bloqueia, registra
   evento de segurança).
2. Saída determinística (barata): fonte inventada, número sem suporte, status
   de exame sem suporte (reusando as verificações das etapas anteriores) e uma
   única regra de consistência com o perfil da paciente (anti-HER2; ver abaixo).
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
# Com format="json", o llama3.2:3b às vezes entra em laço gerando espaços até o teto
# padrão do Ollama (10 x num_ctx = 40.960 tokens, >2 h na CPU; P004 em 26/10/2026).
# O maior JSON legítimo do juiz medido tem ~490 tokens, por isso o teto é 800.
NUM_PREDICT_JUIZ = 800
TIMEOUT_JUIZ_S = 180

AUTOMEDICACAO_RE = re.compile(
    r"\b(devo tomar|posso tomar|quanto (eu )?tomo|me receite|qual dose (eu )?devo)\b", re.IGNORECASE)
INJECAO_RE = re.compile(r"\b(ignore (as|todas as) instruç|desconsidere|system prompt)", re.IGNORECASE)
RESPOSTA_AUTOMEDICACAO = ("Não posso orientar uso de medicamentos para você. Procure seu médico ou, "
                          "em urgência, o pronto atendimento.")

VEREDITOS_REPROVADOS = ("infiel", "indeterminado")


def aviso_juiz(juiz: dict, objeto: str = "conduta") -> str:
    """
    Aviso no topo da resposta quando o juiz não confirma afirmações. Deixa claro que a
    verificação é experimental e NÃO exaustiva: no P004, o juiz levantou dúvida sobre uma
    frase correta e deixou passar o erro real.
    """
    ressalva = f"Ela não é exaustiva: o restante da {objeto} também precisa ser conferido no protocolo."
    if juiz["afirmacoes_sem_suporte"]:
        itens = "; ".join(f"«{a}»" for a in juiz["afirmacoes_sem_suporte"])
        return f"⚠ Verificação automática experimental levantou dúvida sobre: {itens}. {ressalva}"
    return f"⚠ Verificação automática experimental não pôde ser concluída. {ressalva}"


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

# Regra de consistência com o perfil da paciente (anti-HER2).
# É a ÚNICA regra de perfil, por escolha: foi o erro clínico real que nem as checagens
# de forma nem o juiz pegaram (P004, triplo-negativa, recebeu monitoramento cardíaco de
# terapia anti-HER2). Generalizar regras de perfil exigiria uma ontologia clínica
# (subtipos, contraindicações, condições de cada conduta), fora do escopo do projeto.
TERMOS_ANTI_HER2_RE = re.compile(
    r"\b(trastuzumab\w*|pertuzumab\w*|anti-?her2|ecocardiograma\w*|função cardíaca)\b", re.IGNORECASE)
PERFIL_HER2_NEGATIVO = [
    (re.compile(r"triplo[- ]negativ", re.IGNORECASE), "triplo-negativa"),
    (re.compile(r"her2[- ]?negativ|her2\s*-(?!\w)|her2\s*0\b", re.IGNORECASE), "HER2 negativa"),
]
NEGACOES = {"nao", "sem", "nunca"}
JANELA_NEGACAO_PERFIL = 3
# Trechos literais do HOSP-2024-ONCO-004 que restringem a terapia anti-HER2 (testados contra protocolos.json).
TRECHOS_ONCO004_ANTI_HER2 = [
    "Em HER2 positivo, adicionar trastuzumabe e pertuzumabe.",
    "Em terapia anti-HER2 (trastuzumabe), monitorar a função cardíaca com ecocardiograma antes do "
    "início e a cada 3 meses durante o tratamento.",
]


def subtipo_her2_negativo(dados_paciente: dict | None) -> str | None:
    diagnostico = (dados_paciente or {}).get("diagnostico") or ""
    return next((rotulo for padrao, rotulo in PERFIL_HER2_NEGATIVO if padrao.search(diagnostico)), None)


def termos_anti_her2_afirmados(texto: str) -> list[str]:
    """Termos de terapia anti-HER2/monitoramento cardíaco que não estão negados ('não indicado trastuzumabe')."""
    termos = []
    for m in TERMOS_ANTI_HER2_RE.finditer(texto):
        anteriores = [normalizar(w).strip(".,;:!?\"'()") for w in texto[:m.start()].split()[-JANELA_NEGACAO_PERFIL:]]
        if not NEGACOES & set(anteriores):
            termos.append(m.group())
    return list(dict.fromkeys(termos))


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
      fonte_inventada          código citado fora de codigos_permitidos
      numero_sem_suporte       número ausente dos textos-fonte, dos dados da paciente e da pergunta
      status_sem_suporte       status de exame afirmado e não devolvido pelas ferramentas
                               (só quando status_retornados é informado)
      inconsistente_com_perfil terapia anti-HER2 ou monitoramento cardíaco recomendados a paciente
                               triplo-negativa ou HER2 negativa (só com dados_paciente, isto é,
                               no modo revisão)
    """
    problemas = []
    if subtipo := subtipo_her2_negativo(dados_paciente):
        if termos := termos_anti_her2_afirmados(resposta):
            problemas.append({"tipo": "inconsistente_com_perfil",
                              "detalhe": {"subtipo": subtipo, "termos": termos,
                                          "trecho_protocolo": " ".join(TRECHOS_ONCO004_ANTI_HER2)}})
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
    return ChatOllama(model=MODELO_JUIZ, temperature=0, format="json", num_ctx=4096,
                      num_predict=NUM_PREDICT_JUIZ, client_kwargs={"timeout": TIMEOUT_JUIZ_S})


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
    """
    modo 'completo': todas as afirmações; 'paciente': só as que falam da paciente.
    Registra tokens gerados e se a saída foi truncada (teto NUM_PREDICT_JUIZ) ou
    estourou o timeout; nos dois casos o veredito é indeterminado.
    """
    import httpx
    from langchain_core.messages import HumanMessage, SystemMessage
    inicio = time.time()
    try:
        saida = _llm_juiz().invoke([SystemMessage(content=PROMPTS_JUIZ[modo]),
                                    HumanMessage(content=_mensagem_juiz(resposta, protocolos_texto,
                                                                        dados_paciente))])
    except httpx.TimeoutException:
        resultado = {"veredito": "indeterminado", "afirmacoes": [], "afirmacoes_sem_suporte": [],
                     "veredito_modelo": None, "motivo": "timeout", "tokens_gerados": None,
                     "truncado": False, "timeout": True}
    else:
        metadados = saida.response_metadata or {}
        resultado = interpretar_juiz(saida.content or "")
        resultado["tokens_gerados"] = metadados.get("eval_count")
        resultado["truncado"] = metadados.get("done_reason") == "length"
        resultado["timeout"] = False
        if resultado["truncado"]:
            resultado.update(veredito="indeterminado", motivo="truncado")
    resultado["modo"] = modo
    resultado["segundos"] = round(time.time() - inicio, 2)
    logger.info(f"[juiz:{modo}] veredito={resultado['veredito']} | "
                f"sem suporte={resultado['afirmacoes_sem_suporte']} | {resultado['segundos']}s")
    return resultado


def reprovado(veredito: str) -> bool:
    return veredito in VEREDITOS_REPROVADOS
