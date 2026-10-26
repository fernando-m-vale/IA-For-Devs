"""
Avaliação do juiz de fidelidade contra o conjunto rotulado à mão
(tests/fixtures/juiz_rotulado.json), 1 repetição, temperatura 0.

"indeterminado" conta como reprovação.
Métricas:
  recall_infiel           infiéis reprovadas / infiéis
  taxa_bloqueio_indevido  fiéis reprovadas / fiéis
  indeterminados          respostas do juiz em JSON inválido ou sem lista de afirmações
  segundos                média por item

Uso:
  poetry run python -m src.evaluate_juiz                                  # juiz completo, 24 itens
  poetry run python -m src.evaluate_juiz --modo paciente --apenas revisao # 12 itens de revisão
  poetry run python -m src.evaluate_juiz --reavaliar outputs/juiz_eval/avaliacao_juiz.json  # sem LLM
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .guardrails import MODELO_JUIZ, interpretar_juiz, juiz_fidelidade, reprovado

logger = logging.getLogger(__name__)

BASE = Path(__file__).parents[1]
FIXTURE = BASE / "tests" / "fixtures" / "juiz_rotulado.json"
OUTPUT_DIR = BASE / "outputs" / "juiz_eval"


def carregar_itens(apenas: str | None = None) -> list[dict]:
    itens = json.loads(FIXTURE.read_text(encoding="utf-8"))["itens"]
    return [i for i in itens if apenas is None or i["id"].startswith(apenas)]


def resumir(resultados: list[dict], modo: str) -> dict:
    infieis = [r for r in resultados if r["rotulo"] == "infiel"]
    fieis = [r for r in resultados if r["rotulo"] == "fiel"]
    matriz = {"infiel_reprovado": sum(r["reprovou"] for r in infieis),
              "infiel_aprovado": sum(not r["reprovou"] for r in infieis),
              "fiel_reprovado": sum(r["reprovou"] for r in fieis),
              "fiel_aprovado": sum(not r["reprovou"] for r in fieis)}
    return {
        "modelo": MODELO_JUIZ, "modo": modo, "n": len(resultados), "matriz": matriz,
        "recall_infiel": round(matriz["infiel_reprovado"] / len(infieis), 3) if infieis else None,
        "taxa_bloqueio_indevido": round(matriz["fiel_reprovado"] / len(fieis), 3) if fieis else None,
        "indeterminados": sum(r["veredito"] == "indeterminado" for r in resultados),
        "segundos_medio": round(sum(r["segundos"] for r in resultados) / len(resultados), 2),
    }


def imprimir(resumo: dict, resultados: list[dict]) -> None:
    m = resumo["matriz"]
    print(f"\n{json.dumps(resumo, ensure_ascii=False, indent=1)}")
    print("\nMatriz de confusão (linhas = rótulo humano, colunas = juiz):")
    print("              reprovou  aprovou")
    print(f"  infiel      {m['infiel_reprovado']:>8}  {m['infiel_aprovado']:>7}")
    print(f"  fiel        {m['fiel_reprovado']:>8}  {m['fiel_aprovado']:>7}")
    print("\nErros do juiz:")
    for r in resultados:
        if r["reprovou"] != (r["rotulo"] == "infiel"):
            tipo = "FALSO NEGATIVO (infiel aprovada)" if r["rotulo"] == "infiel" else "FALSO POSITIVO (fiel reprovada)"
            print(f"- {r['id']} | {tipo} | veredito={r['veredito']} | rótulo: {r['justificativa']}")
            for a in r["afirmacoes"]:
                print(f"    [{a.get('suporte')}] {a.get('texto')} — {a.get('motivo')}")


def avaliar(modo: str = "completo", apenas: str | None = None, saida: str | None = None) -> dict:
    resultados = []
    for item in carregar_itens(apenas):
        j = juiz_fidelidade(item["resposta"], item["protocolos_texto"], item["dados_paciente"], modo=modo)
        resultados.append({"id": item["id"], "rotulo": item["rotulo"], "justificativa": item["justificativa"],
                           "resposta": item["resposta"], "veredito": j["veredito"],
                           "veredito_modelo": j["veredito_modelo"], "reprovou": reprovado(j["veredito"]),
                           "afirmacoes": j["afirmacoes"], "afirmacoes_sem_suporte": j["afirmacoes_sem_suporte"],
                           "segundos": j["segundos"]})
        logger.info(f"[eval-juiz] {item['id']} | rótulo={item['rotulo']} | veredito={j['veredito']}")
    resumo = resumir(resultados, modo)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nome = saida or ("avaliacao_juiz.json" if modo == "completo" and apenas is None
                     else f"avaliacao_juiz_{modo}{'_' + apenas if apenas else ''}.json")
    (OUTPUT_DIR / nome).write_text(json.dumps({"resumo": resumo, "resultados": resultados},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    imprimir(resumo, resultados)
    return resumo


def reavaliar(arquivo: Path, apenas: str | None = None) -> tuple[dict, list[dict]]:
    """Recalcula vereditos a partir das afirmações salvas (regra atual), sem LLM."""
    dados = json.loads(Path(arquivo).read_text(encoding="utf-8"))
    resultados = []
    for r in dados["resultados"]:
        if apenas and not r["id"].startswith(apenas):
            continue
        j = interpretar_juiz(json.dumps({"afirmacoes": r["afirmacoes"]}) if r["veredito"] != "indeterminado"
                             else "")
        resultados.append({**r, "veredito": j["veredito"], "reprovou": reprovado(j["veredito"]),
                           "afirmacoes_sem_suporte": j["afirmacoes_sem_suporte"]})
    return resumir(resultados, dados["resumo"].get("modo", "completo")), resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s",
                        datefmt="%H:%M:%S")
    for ruidoso in ("httpx", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--modo", default="completo", choices=("completo", "paciente"))
    parser.add_argument("--apenas", help="prefixo dos ids (ex.: revisao, rag)")
    parser.add_argument("--reavaliar", metavar="ARQUIVO.json")
    args = parser.parse_args()
    if args.reavaliar:
        resumo, resultados = reavaliar(Path(args.reavaliar), args.apenas)
        imprimir(resumo, resultados)
    else:
        avaliar(args.modo, args.apenas)
