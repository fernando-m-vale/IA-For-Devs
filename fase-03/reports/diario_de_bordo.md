# Diário de Bordo — Tech Challenge Fase 3 (Fine-tuning + LangChain + LangGraph)

> Rascunho vivo do relatório técnico final. Cada decisão e achado relevante
> é registrado aqui na ordem em que aconteceu.

## 1. Escopo e decisões iniciais

### 1.1 Restrições do ambiente
- Máquina local apenas com CPU (16 GB RAM) e prazo inferior a 2 semanas.
- Fine-tuning real de LLM exige GPU: a etapa de treino roda no **Google Colab
  (GPU T4, 15 GB)**; inferência e pipeline do assistente rodam localmente via
  **Ollama**.

### 1.2 Arquitetura planejada
| Requisito do enunciado | Implementação |
|---|---|
| Fine-tuning de LLM | QLoRA com Unsloth sobre LLaMA 3.2 3B Instruct (Colab) |
| Assistente com LangChain | RAG (ChromaDB + embeddings multilíngues) + modelo fine-tunado via Ollama |
| Consulta a base estruturada | SQLite com prontuários sintéticos + agente SQL |
| Fluxos automatizados | LangGraph (triagem → exames pendentes → conduta → alertas) |
| Segurança e explainability | Guardrails, logging de auditoria, citação da fonte |

### 1.3 Escolha do modelo base
- **LLaMA 3.2 3B** em vez do 3.1 8B usado na Fase 2: treino viável no Colab
  gratuito e inferência mais rápida na CPU local (na Fase 2 o 8B levava de
  4 a 8 minutos por resposta, o que inviabilizaria um fluxo LangGraph com
  várias chamadas).

### 1.4 Ambiente
- Poetry com `requires-python = ">=3.10,<4.0"` (sem limite superior, o
  Poetry tentava validar compatibilidade com uma hipotética Python 4.0 e a
  resolução falhava) e `package-mode = false`.

## 2. Dataset sintético e curadoria

### 2.1 Dados base (`src/data_prep.py`)
- 10 protocolos clínicos sintéticos (oncologia mamária, endocrinologia,
  dor, emergência, radiologia, comunicação), 10 pares pergunta/resposta
  escritos manualmente e 8 prontuários sintéticos em SQLite.
- Dados sintéticos em português em vez de PubMedQA/MedQuAD (sugeridos no
  enunciado): os datasets sugeridos estão em inglês e são grandes demais
  para demonstração local; dados próprios mantêm a narrativa do hospital
  das Fases 1 e 2 e permitem demonstrar a etapa de curadoria.
- **Os 10 pares manuais nunca entram no treino:** formam o conjunto de
  avaliação (held-out).

### 2.2 Aumento sintético (`src/data_augmentation.py`)
- Modelo "professor" LLaMA 3.1 8B (Ollama local) gera pares a partir de cada
  protocolo para o modelo "aluno" de 3B (abordagem próxima de destilação).
- Script retomável: o progresso é salvo após cada chamada (validado na
  prática quando o VS Code travou durante a geração).

### 2.3 Pipeline de curadoria (6 filtros)
1. Anonimização (CPF, e-mail, telefone, nomes com tratamento)
2. Tamanho mínimo/máximo
3. Segurança (linguagem de prescrição direta)
4. Duplicatas exatas e semânticas (embeddings multilíngues, limiar 0,92)
5. Vazamento para o conjunto de avaliação (similaridade ≥ 0,85)
6. **Revisão humana contra o texto do protocolo**

### 2.4 Iteração de prompt engineering do gerador
| | v1 | v2 |
|---|---|---|
| Pares por chamada | 20 | 8 (2 rodadas com focos diferentes) |
| Instrução de formato | nenhuma | 2 a 4 frases com justificativa + exemplo few-shot |
| Brutos | 187 | 151 |
| Removidos por tamanho | 137 (73%) | 68 (45%) |
| Vazamento removido | 6 | 0 |
| Curados (filtros 1–5) | 38 | 67 |

- **Diagnóstico da v1:** respostas telegráficas ("Solicitar mamografia
  digital.") e falsa variedade (mesma pergunta trocando só a idade). Com 20
  pares por chamada o modelo encurta as respostas. Baixar o limite de
  tamanho foi descartado: ensinaria respostas rasas.
- PROT-003 e PROT-004 ficaram com zero pares na v1 (todas as respostas
  caíram no filtro de tamanho); na v2, os 10 protocolos foram representados.

### 2.5 Revisão humana (filtro 6)
- Leitura dos 67 pares contra os protocolos revelou **19 respostas com erro
  factual, contradição, informação inventada ou conclusão insegura** que
  passaram por todos os filtros automáticos. Exemplos: T3 classificado como
  Estágio II; IMC 22 tratado como "≥25"; recomendação de não internar
  paciente oncológico com dispneia e hipotensão.
- **Erro na fonte:** o protocolo PROT-002 descrevia imagem suspeita com
  histologia benigna como resultado "concordante" (o correto é discordante).
  O modelo professor reproduziu o erro; o protocolo foi corrigido.
- **Lição:** filtros automáticos avaliam forma, não correção. Em domínio
  médico, a revisão humana não é opcional.
- Registro auditável: `revisao_humana.json` (pergunta + motivo).

### 2.6 Exemplos manuais de segurança
- 8 exemplos escritos manualmente: pedidos de prescrição direta, confirmação
  de diagnóstico sem biópsia, perguntas fora da base de protocolos e
  tentativa de fazer o modelo ignorar as regras.

### 2.7 Dataset final v1
- 48 pares curados + 8 de segurança = **56 exemplos de treino**; 10 de avaliação.

## 3. Fine-tuning com QLoRA

### 3.1 Configuração (comum às duas versões)
- Unsloth, LLaMA 3.2 3B Instruct em 4 bits (QLoRA), LoRA r=16, alpha=16,
  7 módulos de projeção; **24,3 M parâmetros treináveis (0,75% do total)**.
- 3 épocas, lr 2e-4, batch efetivo 8, otimizador adamw_8bit.
- Perda calculada apenas nas respostas do assistente (`train_on_responses_only`).
- Avaliação antes/depois no mesmo notebook (LoRA inicializado com B=0
  reproduz exatamente o modelo base).

### 3.2 Fine-tuning v1 — formato pergunta → resposta
- Treino de 60 s; perda de 2,7 para ~1,1, sem colapso (sem memorização).

| Métrica | Base | Fine-tunado v1 |
|---|---|---|
| Cita fonte | 30%* | 100% |
| Fonte correta | 0% | 10% |
| Recusa adversarial | 75% | 0%** |
| Palavras por resposta | 162 | 61 |

\* O base "citava" fontes externas inventadas (NCCN, ASCO).
\*\* Parte é falha da regex ("Não, pois..." não reconhecido), mas houve
regressão real.

- **Achados:**
  - O modelo aprendeu o **formato** da citação, não a associação com o
    conteúdo: citou `HOSP-2024-ONCO-001` (o protocolo mais frequente no
    treino) em 100% das respostas. Citação sempre presente e sempre igual
    gera falsa sensação de rastreabilidade.
  - **Regressão de segurança:** inventou um "protocolo TACTOS-THROMBOSIS"
    para infarto (sem protocolo na base) e produziu alucinações confiantes
    (compressão medular "levando a anemia", HbA1c em mmol/L). Fenômeno
    conhecido: fine-tuning com poucos dados pode degradar o alinhamento de
    segurança do modelo base.
- **Causa:** o treino ensinou a responder de memória, mas no sistema final o
  modelo receberá o protocolo via RAG. Treinou-se uma tarefa diferente da que
  ele executará.

### 3.3 Fine-tuning v2 — formato RAG (retrieval-augmented fine-tuning)
- Cada exemplo contém os protocolos recuperados como contexto:

| Tipo | Contexto | Resposta ensinada | Qtd |
|---|---|---|---|
| Positivo | Protocolo correto + 1 distrator, ordem aleatória | Resposta curada + fonte | 48 |
| Negativo | 2 protocolos de outra área clínica | Recusa por falta de contexto | 20 |
| Segurança | Protocolo relevante (ou nenhum) | Exemplos manuais | 8 |

- O distrator obriga o modelo a escolher a fonte correta; negativos usam
  apenas protocolos de **outra área clínica** (para não ensinar recusa a
  perguntas que o contexto de fato responde). Proporção de exemplos de
  recusa: 37% (v1: 14%).
- Avaliação com o mesmo contexto para base e fine-tunado: 10 com contexto,
  5 sem contexto (protocolos de outra área), 4 adversariais. Nova métrica:
  **fonte inventada** (código citado que não estava no contexto).
- Treino: 30 passos; perda de 1,4 para ~0,2–0,8 (início mais baixo que a v1
  porque o contexto torna a resposta mais previsível).

| Métrica | Base (com contexto) | Fine-tunado v2 |
|---|---|---|
| Cita fonte | 70% | 100% |
| Fonte correta | 100%* | 100% |
| Recusa sem contexto | 20%** | 100% (80% na revisão manual) |
| Recusa adversarial | 75% | 100% |
| Fonte inventada | 10,5% | 0% |
| Palavras por resposta | 81 | 34 |

\* Inflado: a métrica aceita o código em qualquer parte do texto (em um caso o
base citou o protocolo errado no campo "Fonte" e o certo no corpo).
\*\* A regex não reconheceu um caso parcial ("consultar um médico").

- **Revisão manual das respostas do fine-tunado v2:**
  - Ganhos claros: nenhuma fonte inventada (o base inventou
    "HOSP-2024-PEM-001" e um "Manejo de Infecções" inexistente); recusou
    informar dose de codeína a paciente (o base informou a dose); recusou
    infarto citando a ausência de protocolo (a v1 inventou um protocolo).
  - Falso positivo da métrica: na pergunta sobre comunicação de diagnóstico
    sem contexto, o modelo respondeu de memória e só rotulou
    "nenhum protocolo aplicável" na fonte. Recusa real: 4/5.
  - Motivo de recusa genérico: em dois casos adversariais, recusou dizendo
    que "o protocolo não cobre", quando o motivo correto é não prescrever a
    pacientes. O comportamento é seguro, a justificativa é imprecisa.
  - Sem excesso de recusa: respondeu 10/10 perguntas com contexto válido.
  - **Perda de completude:** respostas 58% mais curtas que o base, herdando
    a brevidade dos pares de treino. Ex.: compressão medular respondida só
    com "dor lombar com déficit neurológico", omitindo a conduta; BI-RADS 4
    descrito apenas pela faixa da categoria 4B.

### 3.4 Conclusão sobre o fine-tuning
- O fine-tuning ensinou **comportamento** (ancorar no contexto, citar a fonte
  correta, recusar sem respaldo), e não conhecimento factual, que é papel do
  RAG. Fine-tuning e RAG são complementares no desenho final.
- Limitações: dataset pequeno (76 exemplos), avaliação com 19 casos,
  respostas menos completas que as do modelo base, templates de recusa
  reproduzidos quase literalmente.
- Mitigação planejada: completude será reforçada no prompt do pipeline RAG
  e guardrails em código complementarão o comportamento do modelo.

## 4. Próximos passos

- [x] Etapa 0: setup da Fase 3
- [x] Etapa 1: dataset sintético e curadoria
- [x] Etapa 2: fine-tuning QLoRA (v1 e v2)
- [ ] Etapa 2b: registrar o modelo fine-tunado no Ollama local
- [ ] Etapa 3: pipeline RAG com LangChain
- [ ] Etapa 4: agente SQL para prontuários
- [ ] Etapa 5: fluxo LangGraph
- [ ] Etapa 6: segurança, logging e explainability
- [ ] Etapa 7: documentação e relatório técnico
- [ ] Etapa 8: vídeo de demonstração
