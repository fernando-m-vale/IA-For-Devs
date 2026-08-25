"""
Módulo de integração com LLM local (Ollama + LLaMA 3.1) para gerar as explicações em linguagem natural dos diagnósticos produzidos pelos modelos de Machine Learning. 
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import ollama

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.1"

SYSTEM_PROMPT = """Você é um assistente de apoio ao diagnóstico médico especializado em oncologia.
Seu papel é ajudar profissionais de saúde a interpretar resultados de modelos de
Machine Learning para diagnóstico de câncer de mama.

Regras importantes:
- Sempre comunique resultados de forma clara, objetiva e em português.
- Nunca emita diagnósticos definitivos — você é uma ferramenta de APOIO à decisão médica.
- Sempre reforce que o médico deve ter a palavra final.
- Use terminologia médica adequada, mas explique termos técnicos quando necessário.
- Seja conciso: respostas entre 150 e 300 palavras, salvo quando solicitado diferente.
- Nunca invente informações que não estejam nos dados fornecidos."""


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisInput:
    """
    Dados de entrada de um diagnóstico para interpretação pela LLM.

    Attributes
    ----------
    patient_id : str
        Identificador anônimo do paciente.
    prediction : str
        Predição do modelo: 'Maligno' ou 'Benigno'.
    confidence : float
        Confiança do modelo na predição (recall do modelo, 0.0 a 1.0).
    model_name : str
        Nome do modelo que gerou o diagnóstico.
    top_features : list[dict]
        Lista das features mais relevantes com nome e valor SHAP.
        Ex: [{"feature": "area3", "shap_value": 0.85, "patient_value": 1200.5}]
    optimized_by_ag : bool
        Se os hiperparâmetros do modelo foram otimizados pelo AG.
    best_hyperparams : dict
        Hiperparâmetros usados pelo modelo.
    """
    patient_id: str
    prediction: str
    confidence: float
    model_name: str
    top_features: list[dict] = field(default_factory=list)
    optimized_by_ag: bool = False
    best_hyperparams: dict = field(default_factory=dict)


@dataclass
class InterpretationResult:
    """Resultado da interpretação gerada pela LLM."""
    patient_id: str
    prompt_used: str
    explanation: str
    actionable_insights: str
    quality_score: Optional[float] = None
    quality_justification: Optional[str] = None
    model_used: str = DEFAULT_MODEL
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_explanation_prompt(diagnosis: DiagnosisInput) -> str:
    """
    Constrói o prompt principal para explicação do diagnóstico.

    Técnica de prompt engineering: structured input + role prompting +
    output format specification.
    """
    features_text = ""
    if diagnosis.top_features:
        features_text = "\n\nCaracterísticas mais relevantes para esta predição:"
        for i, feat in enumerate(diagnosis.top_features[:5], 1):
            direction = "↑ aumenta risco" if feat["shap_value"] > 0 else "↓ reduz risco"
            features_text += (
                f"\n  {i}. {feat['feature']}: valor={feat.get('patient_value', 'N/A'):.2f} "
                f"| impacto SHAP={feat['shap_value']:+.4f} ({direction})"
            )

    ag_text = ""
    if diagnosis.optimized_by_ag:
        ag_text = (
            f"\n\nObs: Este modelo teve seus hiperparâmetros otimizados por "
            f"Algoritmo Genético. Hiperparâmetros utilizados: "
            f"{json.dumps(diagnosis.best_hyperparams, ensure_ascii=False)}"
        )

    prompt = f"""Analise o seguinte resultado de diagnóstico assistido por IA e gere uma
explicação clara para o profissional de saúde responsável.

=== RESULTADO DO DIAGNÓSTICO ===
Paciente ID: {diagnosis.patient_id}
Predição do modelo: {diagnosis.prediction}
Modelo utilizado: {diagnosis.model_name}
Confiança do modelo (recall em validação cruzada): {diagnosis.confidence:.1%}{features_text}{ag_text}

=== INSTRUÇÕES ===
Gere uma explicação estruturada com:
1. Resumo do resultado (1 parágrafo)
2. Interpretação das características mais relevantes (o que cada uma indica clinicamente)
3. Limitações e recomendação (sempre reforce que o médico decide)

Responda em português, de forma clara e profissional."""

    return prompt


def build_insights_prompt(diagnosis: DiagnosisInput) -> str:
    """
    Constrói o prompt para geração de insights acionáveis.

    Técnica: chain-of-thought prompting — pede que a LLM raciocine
    passo a passo sobre as implicações clínicas dos dados.
    """
    features_json = json.dumps(diagnosis.top_features[:5], ensure_ascii=False, indent=2)

    prompt = f"""Com base nas características mais relevantes identificadas pelo modelo de ML
para o diagnóstico de {'câncer de mama maligno' if diagnosis.prediction == 'Maligno' else 'tumor benigno'}
do paciente {diagnosis.patient_id}, gere insights acionáveis para o médico responsável.

Características identificadas (com valores SHAP):
{features_json}

Raciocine passo a passo:
1. O que cada característica indica biologicamente sobre o tumor?
2. Quais características merecem atenção prioritária do médico?
3. Que exames complementares poderiam ser indicados com base nestes achados?
4. Qual é a recomendação geral de conduta (sempre reforçando que é uma sugestão de apoio)?

Responda em português, de forma objetiva e acionável."""

    return prompt


def build_quality_evaluation_prompt(
    diagnosis: DiagnosisInput,
    explanation: str,
) -> str:
    """
    Constrói o prompt para auto-avaliação da qualidade da interpretação gerada.

    Técnica: self-evaluation prompting — a LLM avalia sua própria resposta
    anterior segundo critérios definidos.
    """
    prompt = f"""Avalie a qualidade da seguinte explicação médica gerada por IA,
considerando os dados originais do diagnóstico.

=== DADOS ORIGINAIS ===
Predição: {diagnosis.prediction}
Confiança: {diagnosis.confidence:.1%}
Features relevantes: {json.dumps([f['feature'] for f in diagnosis.top_features[:5]], ensure_ascii=False)}

=== EXPLICAÇÃO GERADA ===
{explanation}

=== CRITÉRIOS DE AVALIAÇÃO ===
Avalie de 0 a 10 cada critério:
1. Precisão técnica: as informações estão corretas e condizem com os dados?
2. Clareza: um médico conseguiria entender facilmente?
3. Segurança: a explicação reforça adequadamente o papel de apoio da IA?
4. Utilidade clínica: as informações são acionáveis na prática?

Responda EXATAMENTE neste formato JSON:
{{
  "scores": {{
    "precisao_tecnica": <0-10>,
    "clareza": <0-10>,
    "seguranca": <0-10>,
    "utilidade_clinica": <0-10>
  }},
  "media": <media dos 4 scores>,
  "pontos_fortes": "<1-2 frases>",
  "pontos_de_melhoria": "<1-2 frases>"
}}"""

    return prompt


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def generate_explanation(
    diagnosis: DiagnosisInput,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> InterpretationResult:
    """
    Gera explicação completa do diagnóstico via LLM.

    Usa temperatura baixa (0.3) para respostas mais consistentes e menos
    criativas — adequado para contexto médico onde precisão importa mais
    que criatividade.

    Parameters
    ----------
    diagnosis : DiagnosisInput
        Dados do diagnóstico a interpretar.
    model : str
        Modelo Ollama a usar.
    temperature : float
        Temperatura da geração (0=determinístico, 1=criativo).

    Returns
    -------
    InterpretationResult com explicação e insights gerados.
    """
    start = time.time()
    prompt = build_explanation_prompt(diagnosis)

    logger.info(f"[LLM] Gerando explicação | paciente={diagnosis.patient_id} | modelo={model}")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": temperature},
    )
    explanation = response["message"]["content"]

    # Gerar insights acionáveis em chamada separada
    insights_prompt = build_insights_prompt(diagnosis)
    insights_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": insights_prompt},
        ],
        options={"temperature": temperature},
    )
    actionable_insights = insights_response["message"]["content"]

    elapsed = round(time.time() - start, 2)
    logger.info(f"[LLM] Explicação gerada em {elapsed}s")

    return InterpretationResult(
        patient_id=diagnosis.patient_id,
        prompt_used=prompt,
        explanation=explanation,
        actionable_insights=actionable_insights,
        model_used=model,
        elapsed_seconds=elapsed,
    )


def evaluate_quality(
    result: InterpretationResult,
    diagnosis: DiagnosisInput,
    model: str = DEFAULT_MODEL,
) -> InterpretationResult:
    """
    Avalia a qualidade da interpretação gerada usando self-evaluation prompting.

    Atualiza result.quality_score e result.quality_justification in-place.

    Returns
    -------
    O mesmo InterpretationResult com os campos de qualidade preenchidos.
    """
    logger.info(f"[LLM] Avaliando qualidade | paciente={diagnosis.patient_id}")

    eval_prompt = build_quality_evaluation_prompt(diagnosis, result.explanation)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Você é um avaliador especializado em qualidade de explicações médicas geradas por IA. Responda sempre em JSON válido."},
            {"role": "user",   "content": eval_prompt},
        ],
        options={"temperature": 0.1},  # temperatura mínima para JSON consistente
    )

    raw = response["message"]["content"]

    try:
        # Extrair JSON da resposta (a LLM pode adicionar texto antes/depois)
        start_idx = raw.find("{")
        end_idx = raw.rfind("}") + 1
        json_str = raw[start_idx:end_idx]
        eval_data = json.loads(json_str)

        result.quality_score = float(eval_data.get("media", 0))
        result.quality_justification = (
            f"Pontos fortes: {eval_data.get('pontos_fortes', 'N/A')} | "
            f"Melhorias: {eval_data.get('pontos_de_melhoria', 'N/A')}"
        )
        scores = eval_data.get("scores", {})
        logger.info(
            f"[LLM] Qualidade: média={result.quality_score:.1f} | "
            f"scores={scores}"
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[LLM] Não foi possível parsear avaliação de qualidade: {e}")
        result.quality_score = None
        result.quality_justification = f"Falha no parse: {raw[:200]}"

    return result


def interpret_diagnosis(
    diagnosis: DiagnosisInput,
    model: str = DEFAULT_MODEL,
    evaluate: bool = True,
) -> InterpretationResult:
    """
    Pipeline completo: gera explicação + insights + avalia qualidade.

    Função principal a ser chamada pelo notebook.

    Parameters
    ----------
    diagnosis : DiagnosisInput
        Dados do diagnóstico.
    model : str
        Modelo Ollama.
    evaluate : bool
        Se True, executa a auto-avaliação de qualidade.

    Returns
    -------
    InterpretationResult completo.
    """
    result = generate_explanation(diagnosis, model=model)

    if evaluate:
        result = evaluate_quality(result, diagnosis, model=model)

    return result