"""
Log de auditoria do grafo, em logs/auditoria/AAAA-MM-DD.jsonl (um arquivo por dia).

ATENÇÃO: este log é DADO SENSÍVEL. Contém conteúdo clínico pseudonimizado
(perguntas, respostas, condutas, diagnósticos e pendências por código de
paciente) e por isso fica fora do git (logs/ no .gitignore). Em produção exigiria
controle de acesso, criptografia em repouso e política de retenção e descarte.

Nunca registra nome de paciente: campos de nome são removidos e qualquer
"Paciente X" restante no texto (inclusive em pergunta e resposta) é mascarado.

Tipos de registro (todos com o mesmo "id" dentro de uma execução):
  execucao            modo pergunta e revisão encerrada cedo (paciente inexistente)
  pendente_aprovacao  revisão parada no interrupt: conduta, alertas, verificações (com o
                      texto de cada tentativa, mascarado) e veredito
  decisao             retomada da revisão: decisão médica e comentário
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

DIRETORIO_LOGS = Path(__file__).parents[1] / "logs" / "auditoria"
CAMPOS_NOME = {"nome", "nome_anonimizado", "nome_completo"}
NOME_RE = re.compile(r"\bPaciente [A-Z]\b")
TIPOS = ("execucao", "pendente_aprovacao", "decisao")


def _sem_nomes(obj):
    if isinstance(obj, dict):
        return {k: _sem_nomes(v) for k, v in obj.items() if k not in CAMPOS_NOME}
    if isinstance(obj, list):
        return [_sem_nomes(v) for v in obj]
    if isinstance(obj, str):
        return NOME_RE.sub("[nome removido]", obj)
    return obj


def novo_id() -> str:
    return str(uuid.uuid4())


def montar_registro(estado: dict, tipo: str = "execucao") -> dict:
    if tipo not in TIPOS:
        raise ValueError(f"tipo de registro inválido: {tipo!r}")
    base = {
        "id": estado.get("id_execucao") or novo_id(),
        "tipo": tipo,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modo": estado.get("modo"),
        "paciente_ids": estado.get("paciente_ids", []),
    }
    if tipo == "decisao":
        registro = {**base, "decisao_medica": estado.get("aprovacao"),
                    "eventos": [e for e in estado.get("auditoria", [])
                                if e["no"] in ("aprovacao_medica", "registrar_decisao")]}
    else:
        juiz = estado.get("juiz") or {}
        registro = {
            **base,
            "rota": estado.get("rota"),
            "pergunta": estado.get("pergunta"),
            "resposta": estado.get("resposta"),
            "eventos": estado.get("auditoria", []),
            "modelos": sorted(set(estado.get("modelos", []))),
            "fontes": estado.get("fontes", []),
            "guardrails_disparados": estado.get("guardrails", []),
            "verificacoes": estado.get("verificacoes", []),
            "veredito_juiz": juiz.get("veredito"),
            "sinalizado_pelo_juiz": bool(estado.get("sinalizado_pelo_juiz")),
            "reparos": estado.get("reparos", 0),
            "eventos_seguranca": estado.get("eventos_seguranca", []),
            "decisao_medica": estado.get("aprovacao"),
        }
        if tipo == "pendente_aprovacao":
            registro["conduta"] = estado.get("conduta")
            registro["alertas"] = [{"exame": a["exame"], "nivel": a["nivel"], "status": a["status"],
                                    "data_prevista": a["data_prevista"], "protocolo_codigo": a["protocolo_codigo"]}
                                   for a in estado.get("alertas", [])]
    return _sem_nomes(registro)


def registrar_execucao(estado: dict, tipo: str = "execucao", diretorio: Path | None = None) -> dict:
    """Acrescenta o registro ao JSONL do dia e devolve o registro."""
    diretorio = Path(diretorio or DIRETORIO_LOGS)
    diretorio.mkdir(parents=True, exist_ok=True)
    registro = montar_registro(estado, tipo)
    with open(diretorio / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro
