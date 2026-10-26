"""
Avaliação do modo revisão do grafo com o LLM real (1 repetição, 12 pacientes).

A revisão roda até o interrupt de aprovação médica e para ali: a avaliação não
simula a decisão do médico. Por paciente:
  segundos            soma dos tempos dos nós até o interrupt
  alertas             nível e exame de cada alerta gerado (template)
  conduta             texto do medassist
  fontes_ok           as fontes citadas na conduta estão entre os protocolos vinculados
  numeros_sem_suporte números da conduta ausentes do texto dos protocolos vinculados
  numeros_fora_dos_dados  dos anteriores, os que também não estão nos dados do
                      paciente enviados no prompt (idade, datas de exame)

Uso:
  poetry run python -m src.evaluate_grafo
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import grafo
from .rag_pipeline import numeros_sem_suporte

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parents[1] / "outputs" / "grafo_eval"
PACIENTES = [f"P{i:03d}" for i in range(1, 13)]


def avaliar() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    g = grafo.construir_grafo()
    linhas = []
    for pid in PACIENTES:
        estado, config = grafo.iniciar_revisao(g, pid)
        textos = [p["texto"] for p in estado["protocolos"]]
        vinculados = [p["codigo"] for p in estado["protocolos"]]
        sem_suporte = sorted(numeros_sem_suporte(estado["conduta"], textos))
        dados_prompt = grafo.montar_pergunta_revisao(estado["dados_paciente"], estado["pendencias"])
        fora_dos_dados = sorted(numeros_sem_suporte(estado["conduta"], textos + [dados_prompt]))
        linha = {
            "paciente_id": pid,
            "aguardando_aprovacao": grafo.aguardando_aprovacao(g, config),
            "segundos": round(sum(e["segundos"] for e in estado["auditoria"]), 2),
            "alertas": [{"nivel": a["nivel"], "exame": a["exame"]} for a in estado["alertas"]],
            "protocolos_vinculados": vinculados,
            "fontes_citadas": estado["fontes"],
            "cita_fonte": bool(estado["fontes"]),
            "fontes_ok": set(estado["fontes"]) <= set(vinculados),
            "numeros_sem_suporte": sem_suporte,
            "numeros_fora_dos_dados": fora_dos_dados,
            "conduta": estado["conduta"],
            "resposta": estado["resposta"],
            "caminho": estado["auditoria"],
        }
        linhas.append(linha)
        logger.info(f"[eval-grafo] {pid} | {linha['segundos']}s | alertas={len(linha['alertas'])} "
                    f"| fontes={linha['fontes_citadas']} | sem_suporte={sem_suporte}")

    (OUTPUT_DIR / "revisao.json").write_text(
        json.dumps({"modelo": grafo.MODELO_CONDUTA, "repeticoes": 1, "pacientes": linhas},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[eval-grafo] salvo em {OUTPUT_DIR / 'revisao.json'}")
    return linhas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    for ruidoso in ("httpx", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    avaliar()
