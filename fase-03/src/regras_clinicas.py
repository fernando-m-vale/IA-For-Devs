"""
Regras clínicas de criticidade das pendências de exames.

Cada regra liga um protocolo e um tipo de exame a um trecho LITERAL do
protocolo que justifica a criticidade. validar_regras() roda na importação e
impede que uma regra cite um texto que o protocolo não contém.

Classificação de uma pendência (exame com status em STATUS_PENDENTES):
  ATRASADO: data_prevista < data_ref (tem precedência sobre as demais)
  CRITICO:  existe regra com o mesmo protocolo_codigo do exame, padrao_exame
            casando com o nome do exame e, se a regra tiver padrao_diagnostico,
            esse padrão casando com o diagnóstico do paciente
  ROTINA:   nenhum dos casos anteriores
Geram alerta: ATRASADO e CRITICO.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .config import DATA_REFERENCIA, STATUS_PENDENTES

DATA_DIR = Path(__file__).parents[1] / "data" / "synthetic"
DB_PATH = DATA_DIR / "prontuarios.db"
CODIGO_RE = re.compile(r"HOSP-\d{4}-[A-Z]+-\d{3}")

ATRASADO, CRITICO, ROTINA = "ATRASADO", "CRITICO", "ROTINA"
NIVEIS_ALERTA = (ATRASADO, CRITICO)

REGRAS_CRITICIDADE = [
    {
        "protocolo_codigo": "HOSP-2024-RAD-001",
        "padrao_exame": r"bi[óo]psia|core biopsy",
        "criticidade": "critica",
        "trecho_protocolo": (
            "Em todas as subcategorias de BI-RADS 4, a biópsia está indicada "
            "para esclarecimento histológico."
        ),
        "justificativa": "Biópsia em lesão BI-RADS 4: esclarecimento histológico de lesão suspeita.",
    },
    {
        "protocolo_codigo": "HOSP-2024-ONCO-004",
        "padrao_exame": r"avalia[çc][ãa]o de resposta",
        "criticidade": "critica",
        "trecho_protocolo": "Avaliação de resposta por imagem a cada 2 ciclos.",
        "justificativa": "Avaliação de resposta na quimioterapia neoadjuvante orienta a continuidade do esquema.",
    },
    {
        "protocolo_codigo": "HOSP-2024-ONCO-004",
        "padrao_exame": r"ecocardiograma",
        "criticidade": "critica",
        "trecho_protocolo": (
            "Em terapia anti-HER2 (trastuzumabe), monitorar a função cardíaca com "
            "ecocardiograma antes do início e a cada 3 meses durante o tratamento."
        ),
        "justificativa": "Monitorização de cardiotoxicidade durante terapia anti-HER2.",
    },
    {
        "protocolo_codigo": "HOSP-2024-RAD-001",
        "padrao_exame": r"\b(usg|ultrassonografia|mamografia)\b",
        "padrao_diagnostico": r"BI-RADS 3",
        "criticidade": "critica",
        "trecho_protocolo": (
            "BI-RADS 3: provavelmente benigno (<2% de malignidade), seguimento em 6 meses."
        ),
        "justificativa": "Seguimento de imagem em 6 meses de lesão BI-RADS 3.",
    },
]


def validar_regras(protocolos_path: Path = DATA_DIR / "protocolos.json") -> None:
    """Falha se o protocolo de uma regra não existir ou não contiver o trecho literal."""
    protocolos = json.loads(protocolos_path.read_text(encoding="utf-8"))
    conteudo = {CODIGO_RE.search(p["fonte"]).group(): p["conteudo"] for p in protocolos}
    erros = []
    for r in REGRAS_CRITICIDADE:
        texto = conteudo.get(r["protocolo_codigo"])
        if texto is None:
            erros.append(f"{r['protocolo_codigo']}: protocolo inexistente")
        elif r["trecho_protocolo"] not in texto:
            erros.append(f"{r['protocolo_codigo']}: trecho não encontrado: {r['trecho_protocolo']!r}")
    if erros:
        raise ValueError("Regras clínicas inválidas:\n  " + "\n  ".join(erros))


validar_regras()

_PADROES = [
    (r,
     re.compile(r["padrao_exame"], re.IGNORECASE),
     re.compile(r["padrao_diagnostico"], re.IGNORECASE) if r.get("padrao_diagnostico") else None)
    for r in REGRAS_CRITICIDADE
]


def regra_aplicavel(protocolo_codigo: str | None, exame: str, diagnostico: str) -> dict | None:
    if protocolo_codigo is None:
        return None
    return next(
        (r for r, padrao_exame, padrao_diag in _PADROES
         if r["protocolo_codigo"] == protocolo_codigo
         and padrao_exame.search(exame)
         and (padrao_diag is None or padrao_diag.search(diagnostico))),
        None,
    )


def gera_alerta(nivel: str) -> bool:
    return nivel in NIVEIS_ALERTA


def classificar_pendencias(
    paciente_id: str,
    data_ref: str = DATA_REFERENCIA,
    conn: sqlite3.Connection | None = None,
) -> list[dict]:
    """Classifica os exames pendentes do paciente em ATRASADO, CRITICO ou ROTINA."""
    proprio = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    try:
        marcadores = ",".join("?" * len(STATUS_PENDENTES))
        linhas = conn.execute(
            f"""SELECT e.id, e.exame, e.status, e.data_prevista, e.protocolo_codigo, p.diagnostico
                FROM exames e JOIN pacientes p ON p.id = e.paciente_id
                WHERE e.paciente_id = ? AND e.status IN ({marcadores}) ORDER BY e.id""",
            (paciente_id, *STATUS_PENDENTES),
        ).fetchall()
    finally:
        if proprio:
            conn.close()

    pendencias = []
    for exame_id, exame, status, data_prevista, protocolo, diagnostico in linhas:
        regra = regra_aplicavel(protocolo, exame, diagnostico)
        if data_prevista is not None and data_prevista < data_ref:
            nivel = ATRASADO
            justificativa = f"Previsto para {data_prevista}, antes da data de referência {data_ref}."
            if regra:
                justificativa += " " + regra["justificativa"]
        elif regra:
            nivel = CRITICO
            justificativa = regra["justificativa"]
        else:
            nivel = ROTINA
            justificativa = "Pendente, sem regra de criticidade aplicável."
        pendencias.append({
            "exame_id": exame_id,
            "exame": exame,
            "status": status,
            "data_prevista": data_prevista,
            "nivel": nivel,
            "protocolo_codigo": protocolo,
            "trecho_protocolo": regra["trecho_protocolo"] if regra else None,
            "justificativa": justificativa,
        })
    return pendencias
