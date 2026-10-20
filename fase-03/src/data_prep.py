"""
Módulo de preparação de dados sintéticos médicos.

Gera três artefatos para o assistente médico:
1. protocolos.json   — protocolos clínicos para o pipeline RAG (ChromaDB)
2. perguntas_medicas.json — pares Q&A para fine-tuning (formato MedQuAD)
3. prontuarios.db    — banco SQLite com pacientes sintéticos para o SQL Agent

Todos os dados são completamente sintéticos e anonimizados por construção
(não derivam de nenhum dado real de paciente).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Protocolos médicos sintéticos (para ChromaDB / RAG)
# ---------------------------------------------------------------------------

PROTOCOLOS = [
    {
        "id": "PROT-001",
        "titulo": "Triagem de Nódulo Mamário Suspeito",
        "categoria": "oncologia",
        "conteudo": (
            "Diante de nódulo mamário detectado em exame clínico ou de imagem, "
            "o protocolo recomenda: (1) classificar o nódulo pelo sistema BI-RADS; "
            "(2) solicitar ultrassonografia mamária bilateral para nódulos em pacientes "
            "abaixo de 35 anos, ou mamografia digital para pacientes acima de 35 anos; "
            "(3) encaminhar para biópsia por agulha grossa (core biopsy) em lesões "
            "BI-RADS 4 ou 5; (4) BI-RADS 3 requer seguimento em 6 meses; "
            "(5) BI-RADS 1 e 2 são achados benignos — seguimento de rotina anual."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ONCO-001",
    },
    {
        "id": "PROT-002",
        "titulo": "Interpretação de Laudos de Biópsia Mamária",
        "categoria": "oncologia",
        "conteudo": (
            "O laudo de biópsia mamária deve ser interpretado em conjunto com a "
            "clínica e a imagem. Carcinoma ductal in situ (CDIS): lesão não invasiva, "
            "tratamento cirúrgico conservador geralmente indicado. Carcinoma invasivo "
            "sem tipo especial (NST): tipo mais comum, prognóstico baseado em grau "
            "histológico, status de receptores hormonais (RE, RP) e HER2. Carcinoma "
            "lobular invasivo: frequentemente multifocal, requer avaliação bilateral. "
            "Resultado discordante (imagem suspeita com histologia benigna): repetir "
            "biópsia em local diferente, pois a amostra pode não ter representado a lesão."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ONCO-002",
    },
    {
        "id": "PROT-003",
        "titulo": "Estadiamento de Câncer de Mama — Sistema TNM",
        "categoria": "oncologia",
        "conteudo": (
            "O estadiamento TNM classifica o câncer de mama em: T (tumor primário): "
            "T1 ≤2cm, T2 2–5cm, T3 >5cm, T4 invasão de parede torácica/pele. "
            "N (linfonodos regionais): N0 sem metástase, N1 1–3 linfonodos axilares, "
            "N2 4–9 linfonodos, N3 ≥10 linfonodos ou linfonodo infraclavicular. "
            "M (metástase a distância): M0 ausente, M1 presente. Estágio I: T1N0M0. "
            "Estágio II: T2N0M0 ou T1-2N1M0. Estágio III: T3-4 ou N2-3. "
            "Estágio IV: qualquer T, qualquer N, M1."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ONCO-003",
    },
    {
        "id": "PROT-004",
        "titulo": "Indicações de Quimioterapia Neoadjuvante",
        "categoria": "oncologia",
        "conteudo": (
            "A quimioterapia neoadjuvante (antes da cirurgia) está indicada em: "
            "(1) tumores localmente avançados (estágio III); (2) pacientes com "
            "intenção de cirurgia conservadora em tumores >3cm; (3) subtipos "
            "biologicamente agressivos (triplo-negativo e HER2 positivo) mesmo "
            "em estágios iniciais. O esquema preferencial é AC-T (adriamicina + "
            "ciclofosfamida seguido de taxano). Em HER2 positivo, adicionar "
            "trastuzumabe e pertuzumabe. Avaliação de resposta por imagem a cada "
            "2 ciclos."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ONCO-004",
    },
    {
        "id": "PROT-005",
        "titulo": "Seguimento Pós-Tratamento de Câncer de Mama",
        "categoria": "oncologia",
        "conteudo": (
            "Após término do tratamento primário, o seguimento recomendado é: "
            "consultas a cada 3–4 meses nos primeiros 2 anos, semestrais do 3º "
            "ao 5º ano, e anuais a partir do 6º ano. Exames de imagem: mamografia "
            "anual da mama residual/contralateral. Não há evidência de benefício "
            "para rastreamento rotineiro de metástases com PET-CT ou cintilografia "
            "óssea em pacientes assintomáticas. Avaliação de densidade mineral óssea "
            "a cada 2 anos em pacientes em hormonioterapia com inibidor de aromatase."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ONCO-005",
    },
    {
        "id": "PROT-006",
        "titulo": "Protocolo de Triagem para Diabetes Mellitus Tipo 2",
        "categoria": "endocrinologia",
        "conteudo": (
            "Triagem para DM2 recomendada em adultos ≥45 anos, ou mais jovens com "
            "IMC ≥25 e pelo menos um fator de risco adicional: familiar de 1º grau "
            "com DM, hipertensão, dislipidemia, síndrome dos ovários policísticos, "
            "histórico de diabetes gestacional. Exames: glicemia de jejum ≥126 mg/dL "
            "em duas ocasiões confirma DM2. HbA1c ≥6,5% confirma DM2. Glicemia de "
            "jejum 100–125 mg/dL caracteriza pré-diabetes — intervenção de estilo "
            "de vida recomendada, reavaliação em 6–12 meses."
        ),
        "fonte": "Protocolo Interno HOSP-2024-ENDO-001",
    },
    {
        "id": "PROT-007",
        "titulo": "Manejo de Dor Oncológica — Escada Analgésica OMS",
        "categoria": "cuidados_paliativos",
        "conteudo": (
            "A escada analgésica da OMS orienta o manejo da dor em 3 degraus: "
            "Degrau 1 (dor leve, EVA 1–3): analgésicos não opioides — paracetamol "
            "500–1000mg a cada 6h ou AINEs (ibuprofeno 400–600mg a cada 8h). "
            "Degrau 2 (dor moderada, EVA 4–6): opioides fracos — codeína 30–60mg "
            "a cada 4–6h associado a não opioide. Degrau 3 (dor intensa, EVA 7–10): "
            "opioides fortes — morfina de liberação imediata 5–10mg a cada 4h, "
            "titular conforme resposta. Adjuvantes (antidepressivos, anticonvulsivantes) "
            "em qualquer degrau para dor neuropática."
        ),
        "fonte": "Protocolo Interno HOSP-2024-PAL-001",
    },
    {
        "id": "PROT-008",
        "titulo": "Critérios de Internação em Emergência Oncológica",
        "categoria": "emergencia",
        "conteudo": (
            "Emergências oncológicas que requerem internação imediata: (1) Neutropenia "
            "febril: temperatura >38,3°C uma vez ou >38°C por 1h, com neutrófilos "
            "<500/mm³ — antibioticoterapia empírica em até 1h; (2) Síndrome de "
            "compressão medular: dor lombar com déficit neurológico — corticoterapia "
            "imediata e radioterapia de urgência; (3) Hipercalcemia maligna: cálcio "
            ">12 mg/dL sintomático — hidratação vigorosa e bisfosfonato IV; "
            "(4) Tamponamento cardíaco: dispneia + hipotensão + turgência jugular "
            "em paciente oncológico — pericardiocentese de urgência."
        ),
        "fonte": "Protocolo Interno HOSP-2024-EMG-001",
    },
    {
        "id": "PROT-009",
        "titulo": "Uso do Sistema BI-RADS na Mamografia",
        "categoria": "radiologia",
        "conteudo": (
            "O sistema BI-RADS (Breast Imaging-Reporting and Data System) padroniza "
            "laudos mamográficos: BI-RADS 0: avaliação incompleta, exame adicional "
            "necessário. BI-RADS 1: negativo, sem achados. BI-RADS 2: achado benigno "
            "definitivo. BI-RADS 3: provavelmente benigno (<2% de malignidade), "
            "seguimento em 6 meses. BI-RADS 4A: baixa suspeita (>2% e ≤10%). "
            "BI-RADS 4B: suspeita intermediária (>10% e ≤50%). BI-RADS 4C: alta "
            "suspeita (>50% e <95%). BI-RADS 5: altamente sugestivo de malignidade "
            "(≥95%), biópsia obrigatória. BI-RADS 6: malignidade confirmada por "
            "biópsia prévia."
        ),
        "fonte": "Protocolo Interno HOSP-2024-RAD-001",
    },
    {
        "id": "PROT-010",
        "titulo": "Comunicação de Diagnóstico Oncológico ao Paciente",
        "categoria": "humanizacao",
        "conteudo": (
            "A comunicação de diagnóstico oncológico deve seguir o protocolo SPIKES: "
            "S (Setting): ambiente privado e confortável, familiar presente se o "
            "paciente desejar; P (Perception): avaliar o que o paciente já sabe; "
            "I (Invitation): verificar quanto o paciente deseja saber; K (Knowledge): "
            "transmitir a informação em linguagem acessível, evitando jargões; "
            "E (Emotions): reconhecer e validar emoções, oferecer silêncio; "
            "S (Strategy): apresentar o plano de tratamento de forma clara, "
            "ressaltar que não está sozinho. Nunca comunicar diagnóstico por telefone "
            "ou através de terceiros sem consentimento do paciente."
        ),
        "fonte": "Protocolo Interno HOSP-2024-HUM-001",
    },
]

# ---------------------------------------------------------------------------
# Pares pergunta/resposta (formato fine-tuning — inspirado no MedQuAD)
# ---------------------------------------------------------------------------

PERGUNTAS_MEDICAS = [
    {
        "id": "QA-001",
        "pergunta": "O que é o sistema BI-RADS e como interpretar a categoria 4?",
        "resposta": (
            "O BI-RADS é um sistema padronizado para classificar achados mamográficos. "
            "A categoria 4 indica lesão suspeita e é subdividida em: 4A (baixa suspeita, "
            "risco de 2–10%), 4B (suspeita intermediária, 10–50%) e 4C (alta suspeita, "
            "50–95%). Todas as lesões BI-RADS 4 requerem biópsia para esclarecimento "
            "histológico. O resultado da biópsia deve ser correlacionado com a imagem "
            "para avaliar concordância."
        ),
        "categoria": "radiologia",
        "fonte": "PROT-009",
    },
    {
        "id": "QA-002",
        "pergunta": "Quais são os critérios para indicar quimioterapia neoadjuvante?",
        "resposta": (
            "A quimioterapia neoadjuvante está indicada em tumores localmente avançados "
            "(estágio III), quando há intenção de cirurgia conservadora em tumores "
            "maiores que 3cm, e em subtipos biologicamente agressivos como triplo-negativo "
            "e HER2 positivo, mesmo em estágios iniciais. O benefício é maior nesses "
            "subtipos, pois a resposta patológica completa correlaciona-se com melhor "
            "prognóstico."
        ),
        "categoria": "oncologia",
        "fonte": "PROT-004",
    },
    {
        "id": "QA-003",
        "pergunta": "Como manejar neutropenia febril em paciente oncológico?",
        "resposta": (
            "Neutropenia febril é uma emergência oncológica. Definida por temperatura "
            ">38,3°C ou >38°C por 1h, com neutrófilos <500/mm³. A conduta é: iniciar "
            "antibioticoterapia empírica de amplo espectro em até 1 hora após a admissão, "
            "coletar hemoculturas antes dos antibióticos, internar para monitorização. "
            "Esquema empírico de escolha: piperacilina-tazobactam ou cefepima IV. "
            "Adicionar vancomicina se suspeita de infecção por gram-positivo ou "
            "instabilidade hemodinâmica."
        ),
        "categoria": "emergencia",
        "fonte": "PROT-008",
    },
    {
        "id": "QA-004",
        "pergunta": "Qual o esquema de seguimento após tratamento de câncer de mama?",
        "resposta": (
            "O seguimento pós-tratamento de câncer de mama inclui consultas a cada "
            "3–4 meses nos primeiros 2 anos, semestrais do 3º ao 5º ano, e anuais "
            "a partir do 6º ano. A mamografia anual está indicada para a mama residual "
            "e contralateral. Não há benefício comprovado para rastreamento rotineiro "
            "de metástases em pacientes assintomáticas."
        ),
        "categoria": "oncologia",
        "fonte": "PROT-005",
    },
    {
        "id": "QA-005",
        "pergunta": "Como diagnosticar diabetes mellitus tipo 2?",
        "resposta": (
            "O diagnóstico de DM2 é confirmado por qualquer um dos seguintes critérios: "
            "glicemia de jejum ≥126 mg/dL em duas ocasiões distintas, HbA1c ≥6,5% "
            "confirmada em segunda coleta, glicemia 2h após sobrecarga oral de glicose "
            "≥200 mg/dL, ou glicemia aleatória ≥200 mg/dL na presença de sintomas "
            "clássicos (poliúria, polidipsia, perda de peso inexplicada)."
        ),
        "categoria": "endocrinologia",
        "fonte": "PROT-006",
    },
    {
        "id": "QA-006",
        "pergunta": "Quando usar morfina para controle de dor oncológica?",
        "resposta": (
            "A morfina está indicada no 3º degrau da escada analgésica da OMS, para "
            "dor intensa (EVA 7–10) que não respondeu a opioides fracos ou em situações "
            "de dor severa ab initio. A dose inicial recomendada é morfina de liberação "
            "imediata 5–10mg a cada 4h, com titulação baseada na resposta. Para dor "
            "irruptiva, prescrever dose de resgate de 1/6 da dose total diária. "
            "Monitorar efeitos adversos: constipação (tratar profilaticamente), "
            "náuseas e sonolência."
        ),
        "categoria": "cuidados_paliativos",
        "fonte": "PROT-007",
    },
    {
        "id": "QA-007",
        "pergunta": "O que é carcinoma ductal in situ e como é tratado?",
        "resposta": (
            "Carcinoma ductal in situ (CDIS) é uma lesão não invasiva, confinada aos "
            "ductos mamários, sem invasão da membrana basal. Representa estágio 0 do "
            "câncer de mama. O tratamento é geralmente cirúrgico conservador "
            "(tumorectomia) com margens livres, seguido de radioterapia adjuvante. "
            "Mastectomia pode ser indicada em lesões extensas ou multicêntricas. "
            "Em casos RE positivos, hormonioterapia adjuvante reduz o risco de "
            "recorrência e de carcinoma invasivo contralateral."
        ),
        "categoria": "oncologia",
        "fonte": "PROT-002",
    },
    {
        "id": "QA-008",
        "pergunta": "Como comunicar um diagnóstico de câncer ao paciente?",
        "resposta": (
            "A comunicação de diagnóstico oncológico deve seguir o protocolo SPIKES: "
            "preparar o ambiente (privado, com familiar se desejado), avaliar o que o "
            "paciente já sabe, verificar quanto ele deseja saber, transmitir a informação "
            "em linguagem acessível, reconhecer e validar as emoções, e apresentar o "
            "plano de tratamento. Nunca comunicar diagnóstico por telefone ou via "
            "terceiros sem consentimento."
        ),
        "categoria": "humanizacao",
        "fonte": "PROT-010",
    },
    {
        "id": "QA-009",
        "pergunta": "Quais exames solicitar na triagem de nódulo mamário?",
        "resposta": (
            "Para nódulo mamário suspeito, a escolha do exame depende da idade: "
            "ultrassonografia mamária bilateral para pacientes abaixo de 35 anos "
            "(mama densa, menor sensibilidade da mamografia); mamografia digital para "
            "pacientes acima de 35 anos. Em casos de alta suspeição clínica independente "
            "da idade, ambos os exames podem ser solicitados simultaneamente. "
            "A classificação BI-RADS orienta a conduta subsequente."
        ),
        "categoria": "oncologia",
        "fonte": "PROT-001",
    },
    {
        "id": "QA-010",
        "pergunta": "O que caracteriza uma emergência de compressão medular em oncologia?",
        "resposta": (
            "Compressão medular maligna é caracterizada por dor lombar progressiva "
            "associada a déficit neurológico (fraqueza de membros inferiores, "
            "alteração de sensibilidade, disfunção vesical ou intestinal) em paciente "
            "oncológico. É uma emergência — corticoterapia (dexametasona) deve ser "
            "iniciada imediatamente, seguida de radioterapia de urgência ou cirurgia "
            "descompressiva. O prognóstico neurológico depende diretamente da "
            "velocidade de intervenção."
        ),
        "categoria": "emergencia",
        "fonte": "PROT-008",
    },
]

# ---------------------------------------------------------------------------
# Prontuários sintéticos (SQLite)
# ---------------------------------------------------------------------------

PACIENTES_SINTETICOS = [
    {"id": "P001", "nome_anonimizado": "Paciente A", "idade": 52, "sexo": "F",
     "diagnostico": "Carcinoma ductal invasivo NST", "estadio": "IIA",
     "status_tratamento": "em_tratamento", "ultima_consulta": "2026-09-15",
     "exames_pendentes": "HbA1c, hemograma completo",
     "alerta": None},
    {"id": "P002", "nome_anonimizado": "Paciente B", "idade": 45, "sexo": "F",
     "diagnostico": "Nódulo mamário BI-RADS 4B", "estadio": None,
     "status_tratamento": "aguardando_biópsia", "ultima_consulta": "2026-10-01",
     "exames_pendentes": "core biopsy agendada 2026-10-25",
     "alerta": "BIÓPSIA PENDENTE — seguimento obrigatório"},
    {"id": "P003", "nome_anonimizado": "Paciente C", "idade": 67, "sexo": "F",
     "diagnostico": "Carcinoma ductal in situ (CDIS)", "estadio": "0",
     "status_tratamento": "pós_cirurgia", "ultima_consulta": "2026-08-20",
     "exames_pendentes": "mamografia controle",
     "alerta": None},
    {"id": "P004", "nome_anonimizado": "Paciente D", "idade": 38, "sexo": "F",
     "diagnostico": "Carcinoma triplo-negativo", "estadio": "IIIA",
     "status_tratamento": "quimioterapia_neoadjuvante", "ultima_consulta": "2026-10-10",
     "exames_pendentes": "PET-CT avaliação resposta",
     "alerta": "AVALIAÇÃO DE RESPOSTA PENDENTE — 2 ciclos completados"},
    {"id": "P005", "nome_anonimizado": "Paciente E", "idade": 71, "sexo": "F",
     "diagnostico": "Carcinoma lobular invasivo", "estadio": "IIB",
     "status_tratamento": "hormonioterapia", "ultima_consulta": "2026-07-30",
     "exames_pendentes": "densitometria óssea",
     "alerta": None},
    {"id": "P006", "nome_anonimizado": "Paciente F", "idade": 55, "sexo": "F",
     "diagnostico": "Seguimento pós-tratamento — remissão completa", "estadio": "I",
     "status_tratamento": "seguimento", "ultima_consulta": "2026-09-05",
     "exames_pendentes": "mamografia anual",
     "alerta": None},
    {"id": "P007", "nome_anonimizado": "Paciente G", "idade": 48, "sexo": "F",
     "diagnostico": "Nódulo BI-RADS 3 — provavelmente benigno", "estadio": None,
     "status_tratamento": "seguimento_imagem", "ultima_consulta": "2026-04-15",
     "exames_pendentes": "USG mamária controle (prazo: outubro 2026)",
     "alerta": "SEGUIMENTO EM ATRASO — USG deveria ter sido realizada"},
    {"id": "P008", "nome_anonimizado": "Paciente H", "idade": 60, "sexo": "F",
     "diagnostico": "Carcinoma ductal invasivo HER2+", "estadio": "IIB",
     "status_tratamento": "em_tratamento", "ultima_consulta": "2026-10-12",
     "exames_pendentes": "ecocardiograma (toxicidade por trastuzumabe)",
     "alerta": "ECOCARDIOGRAMA OBRIGATÓRIO — terapia anti-HER2 em andamento"},
]


# ---------------------------------------------------------------------------
# Funções de geração
# ---------------------------------------------------------------------------

def generate_protocols(output_dir: Path) -> None:
    """Salva os protocolos médicos sintéticos em JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocolos.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(PROTOCOLOS, f, ensure_ascii=False, indent=2)
    print(f"[data_prep] {len(PROTOCOLOS)} protocolos salvos em {path}")


def generate_qa_pairs(output_dir: Path) -> None:
    """Salva os pares Q&A para fine-tuning em JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "perguntas_medicas.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(PERGUNTAS_MEDICAS, f, ensure_ascii=False, indent=2)
    print(f"[data_prep] {len(PERGUNTAS_MEDICAS)} pares Q&A salvos em {path}")


def generate_prontuarios_db(output_dir: Path) -> None:
    """Cria o banco SQLite com prontuários sintéticos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "prontuarios.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id TEXT PRIMARY KEY,
            nome_anonimizado TEXT,
            idade INTEGER,
            sexo TEXT,
            diagnostico TEXT,
            estadio TEXT,
            status_tratamento TEXT,
            ultima_consulta TEXT,
            exames_pendentes TEXT,
            alerta TEXT
        )
    """)

    cursor.executemany(
        """INSERT OR REPLACE INTO pacientes VALUES
           (:id, :nome_anonimizado, :idade, :sexo, :diagnostico, :estadio,
            :status_tratamento, :ultima_consulta, :exames_pendentes, :alerta)""",
        PACIENTES_SINTETICOS,
    )

    conn.commit()
    conn.close()
    print(f"[data_prep] {len(PACIENTES_SINTETICOS)} pacientes salvos em {db_path}")


def generate_all(base_dir: Path | None = None) -> None:
    """Gera todos os artefatos sintéticos."""
    if base_dir is None:
        base_dir = Path(__file__).parents[1] / "data" / "synthetic"

    generate_protocols(base_dir)
    generate_qa_pairs(base_dir)
    generate_prontuarios_db(base_dir)
    print("[data_prep] Todos os artefatos gerados com sucesso.")


if __name__ == "__main__":
    generate_all()
