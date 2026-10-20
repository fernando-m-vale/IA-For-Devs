"""
Aumento sintético e curadoria do dataset de fine-tuning.

Fluxo:
1. Um modelo "professor" (LLaMA 3.1 8B via Ollama) gera pares pergunta/resposta
   a partir de cada protocolo interno.
2. Os pares brutos passam por curadoria: anonimização, filtro de tamanho,
   filtro de segurança, deduplicação exata e semântica, e remoção de
   vazamento para o conjunto de avaliação.
3. Os 10 pares escritos manualmente (perguntas_medicas.json) NÃO entram no
   treino: formam o conjunto de avaliação (held-out).
4. Saída em formato chat (JSONL), pronto para fine-tuning com Unsloth.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parents[1] / "data" / "synthetic"
TEACHER_MODEL = "llama3.1"
PAIRS_PER_CALL = 8
ROUNDS_PER_PROTOCOL = 2

ROUND_FOCUS = {
    1: "condutas e casos clínicos curtos (uma situação de paciente e qual a conduta segundo o protocolo, explicando o motivo)",
    2: "definições, critérios, interpretação de achados e situações de exceção ou dúvida previstas no protocolo",
}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM_PROMPT_TEACHER = """Você é um médico especialista criando material de treinamento para um assistente virtual hospitalar.
Gere perguntas que médicos fariam no dia a dia e respostas baseadas EXCLUSIVAMENTE no protocolo fornecido.

Regras para as respostas:
- Português do Brasil, linguagem técnica adequada a médicos.
- Cada resposta deve ter de 2 a 4 frases: a orientação principal E o motivo clínico ou critério do protocolo que a justifica.
- Nunca invente informação que não esteja no protocolo.
- São orientações para o médico, nunca prescrições diretas ao paciente.

Regras para as perguntas:
- Cada pergunta deve abordar um aspecto DIFERENTE do protocolo.
- Não gere perguntas que diferem apenas pela idade, sexo ou outro detalhe trivial do paciente.
- Em temas de saúde mamária, refira-se a "a paciente".

Exemplo de formato (tema fictício, use apenas como modelo de estilo, não copie o conteúdo):
{"pergunta": "Paciente com pressão arterial de 150x95 mmHg em consulta isolada já deve iniciar anti-hipertensivo?",
 "resposta": "Não necessariamente. O diagnóstico de hipertensão exige confirmação com medidas em pelo menos duas consultas ou por monitorização ambulatorial, pois uma medida isolada pode refletir o efeito do avental branco. A decisão de iniciar tratamento depende dessa confirmação e da estratificação de risco cardiovascular."}"""

FINETUNE_SYSTEM_PROMPT = (
    "Você é um assistente de apoio à decisão clínica do hospital. "
    "Responda com base nos protocolos internos, cite a fonte e nunca prescreva "
    "diretamente: a decisão final é sempre do médico."
)

# ---------------------------------------------------------------------------
# Anonimização e segurança
# ---------------------------------------------------------------------------

ANON_PATTERNS = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"), "[TELEFONE]"),
    (re.compile(r"\b(?:Sr|Sra|Dr|Dra)\.\s+[A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+)*"), "[NOME]"),
]

UNSAFE_PATTERNS = re.compile(
    r"\b(prescrevo|receito|você deve tomar|tome\s|automedica)", re.IGNORECASE
)


def anonymize_text(text: str) -> tuple[str, int]:
    """Substitui dados pessoais identificáveis. Retorna (texto, nº de substituições)."""
    total = 0
    for pattern, replacement in ANON_PATTERNS:
        text, n = pattern.subn(replacement, text)
        total += n
    return text, total


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ---------------------------------------------------------------------------
# Geração (modelo professor)
# ---------------------------------------------------------------------------

def build_generation_prompt(protocolo: dict, n: int, rodada: int) -> str:
    return f"""Protocolo: {protocolo['titulo']} ({protocolo['fonte']})

Conteúdo:
{protocolo['conteudo']}

Gere {n} pares de pergunta e resposta distintos sobre este protocolo.
Foco desta rodada: {ROUND_FOCUS[rodada]}.

Responda APENAS com JSON no formato:
{{"pares": [{{"pergunta": "...", "resposta": "..."}}]}}"""


def generate_raw_pairs(
    protocolos: list[dict],
    output_path: Path,
    model: str = TEACHER_MODEL,
    n: int = PAIRS_PER_CALL,
    rounds: int = ROUNDS_PER_PROTOCOL,
) -> list[dict]:
    """
    Gera pares brutos para cada protocolo, em várias rodadas com focos
    diferentes. Retomável: combinações (protocolo, rodada) já presentes
    em output_path são puladas, e o progresso é salvo após cada chamada.
    """
    pairs: list[dict] = []
    if output_path.exists():
        pairs = json.loads(output_path.read_text(encoding="utf-8"))
    done = {(p["fonte_id"], p.get("rodada", 1)) for p in pairs}

    for prot in protocolos:
        for rodada in range(1, rounds + 1):
            if (prot["id"], rodada) in done:
                logger.info(f"[geracao] {prot['id']} rodada {rodada} já processada, pulando")
                continue

            logger.info(f"[geracao] {prot['id']} rodada {rodada}/{rounds} ({prot['titulo']})")
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TEACHER},
                    {"role": "user", "content": build_generation_prompt(prot, n, rodada)},
                ],
                format="json",
                options={"temperature": 0.7, "num_ctx": 4096},
            )

            try:
                novos = json.loads(response["message"]["content"]).get("pares", [])
            except json.JSONDecodeError:
                logger.warning(
                    f"[geracao] JSON inválido em {prot['id']} rodada {rodada}, "
                    "será refeito na próxima execução"
                )
                continue

            validos = 0
            for par in novos:
                if isinstance(par, dict) and par.get("pergunta") and par.get("resposta"):
                    pairs.append({
                        "pergunta": str(par["pergunta"]).strip(),
                        "resposta": str(par["resposta"]).strip(),
                        "fonte_id": prot["id"],
                        "fonte": prot["fonte"],
                        "categoria": prot["categoria"],
                        "rodada": rodada,
                    })
                    validos += 1

            output_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[geracao] {prot['id']} rodada {rodada}: {validos} pares (total: {len(pairs)})")

    return pairs

# ---------------------------------------------------------------------------
# Curadoria
# ---------------------------------------------------------------------------

def curate(
    pairs: list[dict],
    eval_questions: list[str],
    min_question_chars: int = 15,
    min_answer_chars: int = 100,
    max_answer_chars: int = 1500,
    dedup_threshold: float = 0.92,
    leakage_threshold: float = 0.85,
) -> tuple[list[dict], dict]:
    """Aplica os 5 filtros de curadoria e retorna (pares_curados, estatísticas)."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    stats: Counter = Counter(total_bruto=len(pairs))

    # Filtros 1-3: anonimização, tamanho, segurança
    candidates = []
    for par in pairs:
        pergunta, n1 = anonymize_text(_normalize(par["pergunta"]))
        resposta, n2 = anonymize_text(_normalize(par["resposta"]))
        stats["substituicoes_anonimizacao"] += n1 + n2

        if len(pergunta) < min_question_chars or not (
            min_answer_chars <= len(resposta) <= max_answer_chars
        ):
            stats["removido_tamanho"] += 1
            continue
        if UNSAFE_PATTERNS.search(resposta):
            stats["removido_seguranca"] += 1
            continue
        candidates.append({**par, "pergunta": pergunta, "resposta": resposta})

    # Filtro 4a: duplicatas exatas
    seen, unique = set(), []
    for c in candidates:
        key = c["pergunta"].lower()
        if key in seen:
            stats["removido_duplicata_exata"] += 1
            continue
        seen.add(key)
        unique.append(c)

    if not unique:
        stats["total_curado"] = 0
        return [], dict(stats)

    # Filtros 4b e 5: duplicatas semânticas e vazamento para avaliação
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    q_emb = encoder.encode([u["pergunta"] for u in unique], normalize_embeddings=True)
    eval_emb = encoder.encode(eval_questions, normalize_embeddings=True)

    kept, kept_emb = [], []
    for item, emb in zip(unique, q_emb):
        if float(np.max(eval_emb @ emb)) >= leakage_threshold:
            stats["removido_vazamento_avaliacao"] += 1
            continue
        if kept_emb and float(np.max(np.array(kept_emb) @ emb)) >= dedup_threshold:
            stats["removido_duplicata_semantica"] += 1
            continue
        kept.append(item)
        kept_emb.append(emb)

    stats["total_curado"] = len(kept)
    stats["por_protocolo"] = dict(Counter(k["fonte_id"] for k in kept))
    return kept, dict(stats)

# ---------------------------------------------------------------------------
# Formatação e orquestração
# ---------------------------------------------------------------------------

def apply_human_review(pairs: list[dict], review_path: Path) -> tuple[list[dict], int]:
    """Filtro 6: remove pares rejeitados na revisão humana contra os protocolos."""
    if not review_path.exists():
        logger.warning("[curadoria] revisao_humana.json não encontrado, filtro 6 não aplicado")
        return pairs, 0
    review = json.loads(review_path.read_text(encoding="utf-8"))
    rejeitadas = {_normalize(r["pergunta"]).lower() for r in review["rejeitados"]}
    kept = [p for p in pairs if _normalize(p["pergunta"]).lower() not in rejeitadas]
    return kept, len(pairs) - len(kept)


def to_chat_example(pergunta: str, resposta: str, fonte: str) -> dict:
    """Formato chat usado pelo Unsloth. A resposta termina citando a fonte."""
    return {
        "messages": [
            {"role": "system", "content": FINETUNE_SYSTEM_PROMPT},
            {"role": "user", "content": pergunta},
            {"role": "assistant", "content": f"{resposta}\n\nFonte: {fonte}"},
        ]
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_finetuning_dataset(base_dir: Path = BASE_DIR, model: str = TEACHER_MODEL) -> dict:
    protocolos = json.loads((base_dir / "protocolos.json").read_text(encoding="utf-8"))
    gold = json.loads((base_dir / "perguntas_medicas.json").read_text(encoding="utf-8"))
    fonte_por_id = {p["id"]: p["fonte"] for p in protocolos}

    raw = generate_raw_pairs(protocolos, base_dir / "qa_bruto_v2.json", model=model)
    curated, stats = curate(raw, [g["pergunta"] for g in gold])

    curated, n_rev = apply_human_review(curated, base_dir / "revisao_humana.json")
    stats["removido_revisao_humana"] = n_rev
    stats["total_curado"] = len(curated)
    stats["por_protocolo"] = dict(Counter(c["fonte_id"] for c in curated))

    seguranca_path = base_dir / "exemplos_seguranca.json"
    seguranca = []
    if seguranca_path.exists():
        seguranca = json.loads(seguranca_path.read_text(encoding="utf-8"))
    stats["exemplos_seguranca_manuais"] = len(seguranca)

    train = [to_chat_example(c["pergunta"], c["resposta"], c["fonte"]) for c in curated]
    train += [to_chat_example(s["pergunta"], s["resposta"], s["fonte"]) for s in seguranca]

    stats["total_treino"] = len(train)
    evaluation = [
        to_chat_example(g["pergunta"], g["resposta"], fonte_por_id.get(g["fonte"], g["fonte"]))
        for g in gold
    ]

    write_jsonl(train, base_dir / "finetune_train.jsonl")
    write_jsonl(evaluation, base_dir / "finetune_eval.jsonl")
    (base_dir / "relatorio_curadoria.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"[curadoria] Treino: {len(train)} | Avaliação: {len(evaluation)}")
    logger.info(f"[curadoria] Estatísticas: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    build_finetuning_dataset()
