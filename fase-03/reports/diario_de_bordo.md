Tech Challenge Fase 3 (Fine-tuning + LangChain + LangGraph)

Escopo e decisões iniciais

1.1 Restrições do ambiente
- Máquina local apenas com CPU (16 GB RAM) e prazo inferior a 2 semanas.
- Fine-tuning real de LLM exige GPU: a etapa de treino roda no Google Colab
  (GPU T4, 15 GB); inferência e pipeline do assistente rodam localmente via
  Ollama.

1.2 Arquitetura planejada
| Requisito do enunciado | Implementação |
|---|---|
| Fine-tuning de LLM | QLoRA com Unsloth sobre LLaMA 3.2 3B Instruct (Colab) |
| Assistente com LangChain | RAG (ChromaDB + embeddings multilíngues) + modelo fine-tunado via Ollama |
| Consulta a base estruturada | SQLite com prontuários sintéticos + agente SQL |
| Fluxos automatizados | LangGraph (triagem → exames pendentes → conduta → alertas) |
| Segurança e explainability | Guardrails, logging de auditoria, citação da fonte |

1.3 Escolha do modelo base
- LLaMA 3.2 3B em vez do 3.1 8B usado na Fase 2: treino viável no Colab
  gratuito e inferência mais rápida na CPU local (na Fase 2 o 8B levava de
  4 a 8 minutos por resposta, o que inviabilizaria um fluxo LangGraph com
  várias chamadas).

1.4 Ambiente
- Poetry com `requires-python = ">=3.10,<4.0"` (sem limite superior, o
  Poetry tentava validar compatibilidade com uma hipotética Python 4.0 e a
  resolução falhava) e `package-mode = false`.

2. Dataset sintético e curadoria

2.1 Dados base (`src/data_prep.py`)
- 10 protocolos clínicos sintéticos (oncologia mamária, endocrinologia,
  dor, emergência, radiologia, comunicação), 10 pares pergunta/resposta
  escritos manualmente e 8 prontuários sintéticos em SQLite.
- Dados sintéticos em português em vez de PubMedQA/MedQuAD (sugeridos no
  enunciado): os datasets sugeridos estão em inglês e são grandes demais
  para demonstração local; dados próprios mantêm a narrativa do hospital
  das Fases 1 e 2 e permitem demonstrar a etapa de curadoria.
- Os 10 pares manuais nunca entram no treino: formam o conjunto de
  avaliação (held-out).

2.2 Aumento sintético (`src/data_augmentation.py`)
- Modelo "professor" LLaMA 3.1 8B (Ollama local) gera pares a partir de cada
  protocolo para o modelo "aluno" de 3B (abordagem próxima de destilação).
- Script retomável: o progresso é salvo após cada chamada.

2.3 Pipeline de curadoria (6 filtros)
1. Anonimização (CPF, e-mail, telefone, nomes com tratamento)
2. Tamanho mínimo/máximo
3. Segurança (linguagem de prescrição direta)
4. Duplicatas exatas e semânticas (embeddings multilíngues, limiar 0,92)
5. Vazamento para o conjunto de avaliação (similaridade ≥ 0,85)
6. Revisão humana contra o texto do protocolo

2.4 Iteração de prompt engineering do gerador
| | v1 | v2 |
|---|---|---|
| Pares por chamada | 20 | 8 (2 rodadas com focos diferentes) |
| Instrução de formato | nenhuma | 2 a 4 frases com justificativa + exemplo few-shot |
| Brutos | 187 | 151 |
| Removidos por tamanho | 137 (73%) | 68 (45%) |
| Vazamento removido | 6 | 0 |
| Curados (filtros 1–5) | 38 | 67 |

- Diagnóstico da v1: respostas telegráficas ("Solicitar mamografia
  digital.") e falsa variedade (mesma pergunta trocando só a idade). Com 20
  pares por chamada o modelo encurta as respostas. Baixar o limite de
  tamanho foi descartado: ensinaria respostas rasas.
- PROT-003 e PROT-004 ficaram com zero pares na v1 (todas as respostas
  caíram no filtro de tamanho); na v2, os 10 protocolos foram representados.

2.5 Revisão humana (filtro 6)
- Leitura dos 67 pares contra os protocolos revelou 19 respostas com erro
  factual, contradição, informação inventada ou conclusão insegura que
  passaram por todos os filtros automáticos. Exemplos: T3 classificado como
  Estágio II; IMC 22 tratado como "≥25"; recomendação de não internar
  paciente oncológico com dispneia e hipotensão.
- Erro na fonte: o protocolo PROT-002 descrevia imagem suspeita com
  histologia benigna como resultado "concordante" (o correto é discordante).
  O modelo professor reproduziu o erro; o protocolo foi corrigido.
- Lição: filtros automáticos avaliam forma, não correção. Em domínio
  médico, a revisão humana não é opcional.
- Registro auditável: `revisao_humana.json` (pergunta + motivo).

2.6 Exemplos manuais de segurança
- 8 exemplos escritos manualmente: pedidos de prescrição direta, confirmação
  de diagnóstico sem biópsia, perguntas fora da base de protocolos e
  tentativa de fazer o modelo ignorar as regras.

2.7 Dataset final v1
- 48 pares curados + 8 de segurança = 56 exemplos de treino; 10 de avaliação.

3. Fine-tuning com QLoRA

3.1 Configuração (comum às duas versões)
- Unsloth, LLaMA 3.2 3B Instruct em 4 bits (QLoRA), LoRA r=16, alpha=16,
  7 módulos de projeção; 24,3 M parâmetros treináveis (0,75% do total).
- 3 épocas, lr 2e-4, batch efetivo 8, otimizador adamw_8bit.
- Perda calculada apenas nas respostas do assistente (`train_on_responses_only`).
- Avaliação antes/depois no mesmo notebook (LoRA inicializado com B=0
  reproduz exatamente o modelo base).

3.2 Fine-tuning v1 — formato pergunta → resposta
- Treino de 60 s; perda de 2,7 para ~1,1, sem colapso (sem memorização).

| Métrica | Base | Fine-tunado v1 |
|---|---|---|
| Cita fonte | 30%* | 100% |
| Fonte correta | 0% | 10% |
| Recusa adversarial | 75% | 0% |
| Palavras por resposta | 162 | 61 |

\* O base "citava" fontes externas inventadas (NCCN, ASCO).
\*\* Parte é falha da regex ("Não, pois..." não reconhecido), mas houve
regressão real.

- Outros pontos
  - O modelo aprendeu o formato da citação, não a associação com o
    conteúdo: citou `HOSP-2024-ONCO-001` (o protocolo mais frequente no
    treino) em 100% das respostas. Citação sempre presente e sempre igual
    gera falsa sensação de rastreabilidade.
  - Regressão de segurança: inventou um "protocolo TACTOS-THROMBOSIS"
    para infarto (sem protocolo na base) e produziu alucinações confiantes
    (compressão medular "levando a anemia", HbA1c em mmol/L). Fenômeno
    conhecido: fine-tuning com poucos dados pode degradar o alinhamento de
    segurança do modelo base.
- Causa: o treino ensinou a responder de memória, mas no sistema final o
  modelo receberá o protocolo via RAG. Treinou-se uma tarefa diferente da que
  ele executará.

3.3 Fine-tuning v2 — formato RAG (retrieval-augmented fine-tuning)
- Cada exemplo contém os protocolos recuperados como contexto:

| Tipo | Contexto | Resposta ensinada | Qtd |
|---|---|---|---|
| Positivo | Protocolo correto + 1 distrator, ordem aleatória | Resposta curada + fonte | 48 |
| Negativo | 2 protocolos de outra área clínica | Recusa por falta de contexto | 20 |
| Segurança | Protocolo relevante (ou nenhum) | Exemplos manuais | 8 |

- O distrator obriga o modelo a escolher a fonte correta; negativos usam
  apenas protocolos de outra área clínica (para não ensinar recusa a
  perguntas que o contexto de fato responde). Proporção de exemplos de
  recusa: 37% (v1: 14%).
- Avaliação com o mesmo contexto para base e fine-tunado: 10 com contexto,
  5 sem contexto (protocolos de outra área), 4 adversariais. Nova métrica:
  fonte inventada (código citado que não estava no contexto).
- Treino: 30 passos; perda de 1,4 para ~0,2–0,8 (início mais baixo que a v1
  porque o contexto torna a resposta mais previsível).

| Métrica | Base (com contexto) | Fine-tunado v2 |
|---|---|---|
| Cita fonte | 70% | 100% |
| Fonte correta | 100%* | 100% |
| Recusa sem contexto | 20% | 100% (80% na revisão manual) |
| Recusa adversarial | 75% | 100% |
| Fonte inventada | 10,5% | 0% |
| Palavras por resposta | 81 | 34 |

\* Inflado: a métrica aceita o código em qualquer parte do texto (em um caso o
base citou o protocolo errado no campo "Fonte" e o certo no corpo).
\*\* A regex não reconheceu um caso parcial ("consultar um médico").

- Revisão manual das respostas do fine-tunado v2:
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
  - Perda de completude: respostas 58% mais curtas que o base, herdando
    a brevidade dos pares de treino. Ex.: compressão medular respondida só
    com "dor lombar com déficit neurológico", omitindo a conduta; BI-RADS 4
    descrito apenas pela faixa da categoria 4B.

3.4 Conclusão sobre o fine-tuning
- O fine-tuning ensinou comportamento (ancorar no contexto, citar a fonte
  correta, recusar sem respaldo), e não conhecimento factual, que é papel do
  RAG. Fine-tuning e RAG são complementares no desenho final.
- Limitações: dataset pequeno (76 exemplos), avaliação com 19 casos,
  respostas menos completas que as do modelo base, templates de recusa
  reproduzidos quase literalmente.
- Mitigação planejada: completude será reforçada no prompt do pipeline RAG
  e guardrails em código complementarão o comportamento do modelo.

4. Pipeline RAG com LangChain (Etapa 3)

4.1 Desenho
- Base de conhecimento: os 10 protocolos internos, um documento por
  protocolo (sem chunking: cada protocolo tem um parágrafo, e cortá-lo
  separaria critério e conduta).
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, roda em
  CPU, multilíngue). Vector store: ChromaDB persistente com distância de
  cosseno.
- Geração: `ChatOllama` com o `medassist` (fine-tunado v2), temperatura 0,3,
  k=2 protocolos no contexto. O prompt do pipeline é idêntico ao do
  treino, porque o fine-tuning v2 aprendeu exatamente esse formato.
- Rastreabilidade: cada resposta devolve as fontes recuperadas (código,
  score, posição em cada busca e trecho), o modelo, a variante de prompt e o
  tempo. É a base da explicabilidade da Etapa 6.

4.2 Limiar de relevância (guardrail de entrada)
- Se o melhor score semântico ficar abaixo do limiar, a pergunta é recusada
  antes de chegar ao LLM.
- Calibração num conjunto separado (10 perguntas fora do escopo próprias +
  perguntas legítimas), nunca no conjunto de avaliação, para não ajustar o
  limiar olhando a prova. Critério: o maior limiar que bloqueia no máximo 5%
  das perguntas legítimas. Resultado: 0,47.
- Limitação: embeddings medem proximidade de tema, não cobertura. "Sepse
  em paciente oncológico" e "hipoglicemia em diabético" passam do limiar porque
  se parecem com protocolos existentes, sem que eles respondam à pergunta. Na
  avaliação, "vacinas em idosos" foi bloqueada com 0,470, sem margem nenhuma.
  O limiar é uma barreira grosseira e não pode ser a única defesa.
  Decisão: não reajustá-lo com base na avaliação (seria overfitting).

4.3 Avaliação v1
- Conjunto: 10 perguntas com protocolo (gold), 8 fora do escopo (5 distantes
  e 3 próximas: hipoglicemia, sepse oncológica, náuseas por quimio) e 4
  adversariais. Configurações: base (`llama3.2:3b`) × fine-tunado, prompts A e B.

| Métrica (v1, 1 repetição) | base_A | ft_A | ft_B |
|---|---|---|---|
| Recuperação correta | 0,90 | 0,90 | 0,90 |
| Cita fonte | 0,80 | 1,00 | 1,00 |
| Fonte correta | 0,70 | 0,90 | 0,90 |
| Recusa fora do escopo | 0,50 | 0,875 | 0,875 |
| Recusa adversarial | 0,50 | 0,50 | 0,75 |
| Fonte inventada | 9,1% | 0% | 4,5% |
| Palavras por resposta | 64 | 36 | 32 |

- Busca falhou na compressão medular: a busca semântica não trouxe o
  EMG-001 (o protocolo fala em "dor lombar com déficit neurológico", não em
  "compressão medular").
- As métricas tinham falhas: a cobertura comparava com respostas de
  referência que tinham informação fora dos protocolos (premiava invenção); a
  regex de recusa não reconhecia as recusas do base; e `fonte_correta`
  aceitava o código em qualquer parte do texto.
- Variância: uma segunda execução do mesmo código deu números diferentes.
  A partir daí, toda avaliação passou a ter 3 repetições e métricas de
  recusa consistente (recusou nas 3).
- Prompt B não mostrou ganho consistente; mantido o "A", idêntico ao treino.

4.4 Busca híbrida (semântica + BM25 com RRF)
- BM25 (`rank_bm25`) captura termos exatos que o embedding dilui. Os dois
  rankings são combinados por Reciprocal Rank Fusion (RRF, k=60), que usa
  as posições e não os scores (as escalas são incomparáveis). O limiar
  continua usando o score semântico.
- Tokenização do BM25: minúsculas, sem acentos e sem stopwords.

| Recuperação (hit@1 / hit@2) | Semântica | Híbrida |
|---|---|---|
| Avaliação (10) | 0,900 / 0,900 | 1,000 / 1,000 |
| Curadas (48) | 0,646 / 0,875 | 0,792 / 1,000 |

4.5 Avaliação v2
- Novo problema encontrado: na hipoglicemia, o ENDO-001 era o 1º semântico (0,613),
  mas o BM25 não pontuou "diabético" ≠ "diabetes", e a palavra genérica
  "paciente" deu pontos ao EMG-001 e ao HUM-001. O RRF tirou o ENDO-001 do
  contexto, e o ft_A inventou um tratamento diferente em cada repetição.
- Duas correções testadas: (a) stopwords de domínio no BM25 (paciente,
  tratamento, conduta, protocolo…); (b) "âncora semântica" (o 1º semântico
  sempre entra no top-k).
- A ablação refutou a âncora:

| Variante | Curadas hit@1 / hit@2 |
|---|---|
| a) só stopwords | 0,792 / 1,000 |
| b) só âncora | 0,833 / 0,938 |
| c) ambas | 0,792 / 0,938 |

  Nas 3 perguntas de BI-RADS, o 1º semântico (errado) venceu por 0,01, e o BM25
  tinha o RAD-001 com folga. A âncora expulsou o protocolo certo. Ela é forte
  justamente quando a semântica é fraca: um defeito de concepção. Com k=2, a
  métrica relevante é o hit@2. Decisão: só stopwords. Elas corrigem a
  hipoglicemia (sem match genérico, prevalece a ordem semântica). A âncora com
  margem foi descartada, porque exigiria calibrar um parâmetro novo em 4 perguntas.
- Lição: a primeira correção foi aceita sem comparar com a versão anterior
  da própria híbrida; a piora só apareceu na comparação seguinte. Cada mudança
  agora é medida contra a versão imediatamente anterior, e duas mudanças
  simultâneas exigem ablação.

4.6 Métricas corrigidas e reavaliação sem LLM
- `cita_fonte`: aceita "Fonte:" em qualquer posição (antes só no início da linha).
- Recusa: (i) aceita "não está listado/disponível/contemplado" (falso
  negativo do base no infarto); (ii) não conta como recusa se o corpo tiver
  número sem suporte ou uma virada do tipo "no entanto, posso fornecer" (o base
  "recusou" a receita de quimio e entregou doses inventadas).
- Pedido de receita de quimio: medido por dose segura (nenhuma dose sem
  suporte), e não por recusa. Responder o esquema do protocolo sem doses é
  aceitável.
- Modo `--reavaliar`: recalcula as métricas a partir das respostas salvas,
  sem chamar o LLM. Permitiu separar o efeito da régua nova do efeito da
  busca nova.

4.7 Resultado final da Etapa 3 (busca híbrida + stopwords, 3 repetições)

| Métrica | base_A | ft_A |
|---|---|---|
| Recuperação correta | 1,000 | 1,000 |
| Cita fonte / fonte correta | 0,800 / 0,800 | 1,000 / 1,000 |
| Cobertura numérica | 0,850 | 0,740 |
| Palavras por resposta | 61,5 | 32,0 |
| Recusa fora do escopo (consistente) | 0,750 (0,625) | 0,875 (0,750) |
| Recusa adversarial (consistente) | 0,889 (0,667) | 0,778 (0,667) |
| Dose segura (receita de quimio) | 0,667 | 1,000 |
| Fonte inventada | 6,1% | 1,5% |
| Número sem suporte | 7,6% | 1,5% |
| Segundos por resposta | 16,2 | 11,7 |

- A avaliação completa rodou com a âncora. Só uma pergunta com protocolo
  recupera documentos diferentes sem ela (compressão medular: muda o
  distrator). Essa pergunta foi reavaliada à parte com a busca final
  (3 repetições × 2 modelos), e as métricas ficaram idênticas.
- O fine-tuning se justifica pela segurança, não pela cobertura: o
  ft_A inventa 4× menos fontes e 5× menos números. O base cobre mais números
  porque escreve o dobro.
- Codeína: o ft_A informou a dose do protocolo a um pedido de
  automedicação em 2 de 3 repetições (na v2 recusou 3/3, com as mesmas fontes).
  É variação de amostragem, e sem as repetições não teríamos visto.

4.8 Falhas percebidas
- Hipoglicemia (ft_A): "a preferência é administrar insulina, pois é mais
  rápida e eficaz". Insulina agrava a hipoglicemia. A resposta não tem
  número e cita uma fonte que estava no contexto, então passou por todas as
  verificações. Em outra repetição, "50–100 mg de glucosa" (dose errada).
- Sepse oncológica: os dois modelos responderam sem protocolo em todas as
  execuções; o base inventou "HOSP-2024-INFE-001".
- BI-RADS 4 (ft_A): três faixas de probabilidade diferentes nas 3
  repetições. Os números existem no protocolo, mas foram associados à
  subcategoria errada, e a checagem numérica não pega isso.
- Conclusão: as defesas atuais verificam a forma (há fonte? a fonte
  existe? os números existem no texto?), e nenhuma verifica o conteúdo.

4.9 Consequência para o desenho da segurança
Defesa em camadas, em que cada uma cobre o que a anterior não pega:
1. Entrada: limiar de relevância e filtro de pedidos de automedicação ou
   prescrição.
2. Saída, determinística (barata): fonte inventada e número sem suporte,
   com bloqueio.
3. Saída, semântica: um LLM juiz (`llama3.1:8b`, diferente do modelo
   avaliado) verifica se cada afirmação clínica está sustentada pelo
   protocolo recuperado (*faithfulness*). Esta camada mira os casos
   insulina e sepse.
4. Humano no circuito: toda resposta é apoio à decisão e é validada pelo médico.

5. Agente para os prontuários (Etapa 4)

5.1 Normalização do banco
- O banco original tinha uma única tabela, com `exames_pendentes` em texto
  livre e um `alerta` escrito à mão (a resposta já estava no dado). Não havia
  como separar "solicitado", "agendado" e "atrasado" com SQL.
- Novo schema: `pacientes`, `exames` (um exame por linha, com status e
  datas) e `paciente_protocolos` (N:N com `motivo`, porque um paciente pode
  ter mais de um protocolo: o P001 é oncológico e tem rastreio de diabetes).
  Integridade no banco: `NOT NULL`, `CHECK` nos status, FKs ativas e validação
  de que todo código de protocolo existe na base de conhecimento.
- Coluna `alerta` removida: o alerta passa a ser calculado, não copiado.
- `DATA_REFERENCIA = 2026-10-20` fixa (em `config.py`): sem ela, "atrasado"
  mudaria conforme o dia de execução (mesmo princípio da semente fixa).
- 4 pacientes de casos-limite (P009–P012): sem pendências, com resultado
  registrado, com vários protocolos e com prazo exatamente na data de
  referência (P012: não está atrasado, porque a regra usa `<`).

5.2 Regras clínicas com o trecho do protocolo validado
- Primeira tentativa: as regras de pendência e atraso não reproduziram os
  alertas antigos. O alerta antigo misturava duas coisas: a pendência (um
  fato) e a criticidade (um julgamento clínico que não estava em lugar nenhum).
- Opções consideradas: uma coluna de prioridade preenchida à mão (a resposta
  voltaria a estar no banco), o LLM julgar (não determinístico) ou regras
  explícitas que citam o protocolo. Escolhida a terceira.
- `regras_clinicas.py`: 4 regras (biópsia em BI-RADS 4, avaliação de resposta
  na neoadjuvância, ecocardiograma em anti-HER2, seguimento de BI-RADS 3). Cada
  regra cita um trecho do protocolo, e a importação falha se o trecho não
  existir literalmente no texto.
- Níveis: ATRASADO (pendente com `data_prevista < referência`) > CRITICO
  (casa uma regra) > ROTINA (sem alerta). Os 4 alertas antigos são reproduzidos
  por regra, e os pacientes de rotina ficam sem alerta.
- Lacuna na base: o PROT-004 não falava em ecocardiograma. Completado com o
  monitoramento cardíaco em anti-HER2 (prática padrão). Efeitos colaterais
  medidos sem LLM: hit@1 das curadas 0,792 → 0,812, por acaso (o BM25 penaliza
  documento longo), e métricas das respostas salvas idênticas. Os datasets de
  fine-tuning e as respostas da avaliação do RAG usaram o texto anterior.
- Bug corrigido antes do commit: a regra 4 casava só protocolo e exame,
  então uma USG de paciente BI-RADS 4 seria tratada como seguimento de BI-RADS 3.
  Foi incluído o campo `padrao_diagnostico`, com teste unitário para o caso
  positivo e para o negativo.

5.3 Três abordagens consideradas
| Abordagem | Decisão |
|---|---|
| A. Text-to-SQL livre (`create_sql_agent`) | Descartada: o modelo de 3B escreve SQL ruim, há risco de consulta indevida e é difícil de auditar |
| B. Ferramentas parametrizadas (SQL fixo) | Base: segura, auditável e alinhada ao fluxo da Etapa 5 |
| C. Text-to-SQL com guardrails | Experimento comparativo |

- Modelo: `llama3.2:3b` base escolhe as ferramentas (o `medassist` foi
  treinado só no formato RAG); temperatura 0.

5.4 Opção B: ferramentas parametrizadas
- Garantias de arquitetura, não de prompt:
  - conexão SQLite somente leitura (`mode=ro`);
  - ID validado por regex (barra `P001' OR '1'='1`);
  - nenhuma ferramenta devolve o nome do paciente (minimização de dados,
    LGPD art. 6º, III): o modelo não revela o que nunca recebeu.
- Ferramentas: `obter_paciente`, `listar_exames`, `verificar_pendencias`
  e `pacientes_com_alerta` (painel).
- v1 (15 casos × 3 repetições): escolha de ferramenta e argumentos 100%
  corretos quando a pergunta cita um paciente. Falhas:
  - painel sem ID: o modelo "chutou" o P001 e afirmou atrasos falsos (6/6);
  - agregação: acessou P001–P003 (15 chamadas) sem que ninguém pedisse e
    inventou "média de 52 anos" (a única paciente em hormonioterapia tem 71);
  - escreveu uma chamada de ferramenta como texto (JSON cru para o usuário);
  - na redação, trocou "solicitado" por "agendado".
- v2, tirando do modelo as decisões em que ele erra:
  - roteamento estrutural pelo ID extraído da pergunta: com ID, só as
    ferramentas de paciente, e cada chamada só é aceita para os IDs citados;
    sem ID, só o painel. O agente só acessa pacientes que o médico citou.
  - limite de 4 chamadas por iteração;
  - curto-circuitos determinísticos: paciente inexistente gera resposta fixa
    ("não foi encontrado no cadastro"; "não há informações" foi considerado
    ambíguo, porque sugere paciente existente sem dados); chamada de ferramenta
    em texto é substituída por "não é possível responder com as ferramentas";
  - status explícito em todas as saídas, inclusive no painel.

| Métrica B (avaliador final) | v1 | v2 |
|---|---|---|
| Ferramenta / argumentos corretos | 0,769 / 0,818 | 1,000 / 1,000 |
| Fatos do gabarito presentes | 0,778 | 1,000 |
| Painel sem falha | 0,000 | 1,000 |
| Fora do alcance (recusa) | 0,000 | 1,000 |
| Status sem suporte | — | 0,091 |
| Fato ou nome inventado | 0 | 0 |
| Sem falha (geral) | 0,556 | 0,933 |

- Erro que restou: "agendado para 2026-10-15" num exame "solicitado"
  (P007, 3/3). O modelo lê `data_prevista` e escreve a forma natural em
  português; a instrução no prompt não resolveu. Fica como evidência para o
  guardrail de saída da Etapa 6 (detectar, gerar de novo com o erro apontado e,
  se persistir, usar template).
- Agregados em B (4 perguntas novas): recusou 2, e respondeu 2 com
  contagens erradas ("5 exames pendentes", que na verdade são 12; o 5 era o
  número de pendências com alerta). A checagem numérica pegou o "5", mas não o
  "4", que existe em "BI-RADS 4".

5.5 O avaliador também foi testado
- Duas rodadas seguidas revelaram falhas nas próprias métricas (data por
  extenso, negação "não foi realizado", "solicitado em <data>" lido como
  status). As respostas reais rotuladas pela análise humana viraram um teste
  de regressão do avaliador (`tests/fixtures`), que cresce a cada rodada.
- Detalhes de desenho: números da pergunta contam como fonte permitida (o
  "50" de "mais de 50 anos"); chamadas rejeitadas não contam como fonte de
  fatos.
- Limitação documentada com `@unittest.expectedFailure`: a checagem
  verifica se o número existe nos resultados, não a quem foi atribuído (o "52"
  era a idade do P001). O rótulo humano não foi alterado para o teste
  passar. É a mesma classe de erro do BI-RADS no RAG, e o alvo do juiz de
  fidelidade.
- Critérios por resultado, não por caminho: na injeção e no fora do
  alcance, consultar uma ferramenta inofensiva não é falha. Falha é vazar,
  inventar ou exceder a autorização.

5.6 Opção C: text-to-SQL protegido
- Guardrails no banco e no código, não no prompt:
  - `mode=ro` e `set_authorizer` do SQLite: só leitura nas 3 tabelas, com a
    coluna `nome_anonimizado` negada (controle de acesso por coluna, imune a
    injeção de prompt);
  - uma única instrução, `LIMIT`, timeout (`set_progress_handler`);
  - só agregados (COUNT/AVG/SUM/MIN/MAX), sem `GROUP BY` por ID;
  - supressão de células pequenas (k=3): agregados de grupos pequenos
    reidentificam pessoas. A "média de idade em hormonioterapia" é a idade da
    única paciente (n=1). O gabarito dessa pergunta é a supressão, e não "71".
  - validações acrescentadas após a primeira rodada: rótulo do `GROUP BY` no
    `SELECT`, rejeição de filtro por paciente, literal categórico inexistente
    devolvido ao modelo com os valores válidos, e fim da correção quando o SQL
    se repete.

| C (19 casos × 3) | v1 | v2 |
|---|---|---|
| Sem falha (geral) | 0,684 | 0,789 |
| ID inexistente | 0,500 | 1,000 |
| Supressão correta | 0,333 | 0,667 |
| Segundos por resposta | 7,7 | 9,5 (B: ~30) |

- Privacidade: 100% em todas as rodadas. Nenhum nome, nenhum dado
  individual, todas as injeções contidas. As camadas determinísticas funcionaram.
- Correção: o ponto fraco. O modelo gera SQL válido que responde a outra pergunta:
  - "Pacientes com exames atrasados?" → `data_prevista < data_solicitacao AND
    status = 'realizado'` → "0". O P007 está atrasado. É um falso negativo
    tranquilizador, o erro mais perigoso do projeto, porque faz o médico deixar de agir;
  - "Pendências que geram alerta?": o modelo inventou um critério de alerta
    por status de tratamento.
- Causa: "atrasado" e "alerta" são regras que existem em um único lugar
  (`regras_clinicas.py`). O C obriga o LLM a recriá-las em SQL, e ele cria uma
  versão diferente. Nenhuma validação de forma pega isso.

5.7 Decisão: arquitetura híbrida
| Pergunta | Caminho |
|---|---|
| Com ID de paciente | B (ferramentas, acesso só ao paciente citado) |
| Sem ID, sobre alertas, atrasos ou pendências | B (painel): conceitos derivados das regras nunca são recalculados pelo LLM |
| Sem ID, contagem ou média | C (estatística protegida) |

O roteamento entra como nó do grafo LangGraph (Etapa 5). Em caso de dúvida
entre painel e estatística, a pergunta vai para o painel (caminho seguro).

6. Fluxos com LangGraph

6.1 Desenho: um grafo, dois modos
- Modo pergunta (o assistente): `entrada → extrair_ids → roteador →
  {agente_b | painel | agente_c | rag} → verificar_saida → finalizar`.
- Modo revisão (o fluxo automatizado do enunciado): `carregar_paciente →
  avaliar_pendencias → buscar_protocolos → sugerir_conduta → verificar_saida →
  gerar_alertas → aprovacao_medica (interrupt) → registrar_decisao`.
- Princípio aplicado: o LLM só entra onde há linguagem a produzir. Na
  revisão, dados, pendências, protocolos e alertas são determinísticos. O único
  passo com LLM é `sugerir_conduta` (`medassist`).
- Protocolos pelo vínculo, não por busca: o paciente já tem os protocolos em
  `paciente_protocolos`. Buscar por similaridade trocaria uma informação certa
  por uma estimativa.
- Painel por template: levava 70 s quando o LLM reescrevia a lista; passou a
  levar 0,01 s.
- Auditoria no estado: cada nó acrescenta um evento (nó, início, duração,
  resumo). É a base do log e da explicabilidade.

6.2 Roteador determinístico
| Ordem | Condição | Rota |
|---|---|---|
| 1 | ID de paciente na pergunta | agente B |
| 2 | vocabulário das regras (atrasado, alerta, pendência, crítico, seguimento vencido) | painel |
| 3 | vocabulário estatístico (quantos, média, total, proporção) | agente C |
| 4 | demais | RAG |

- Se casar painel e estatística, vence o painel (caminho seguro): "Quantos
  pacientes têm exames atrasados?" é contado pela regra, e não pelo SQL gerado.
- Falso positivo corrigido: sem limite de palavra, "totalmente" mandava
  uma pergunta de protocolo para o agente C. Passou a usar `\b`, com teste de regressão.
- Teste sem LLM: 27 perguntas com a rota esperada.

6.3 Humano no circuito
- `interrupt` do LangGraph com `MemorySaver`: o grafo pausa antes de registrar
  a conduta, e o médico aprova ou rejeita com um comentário.
- Pegadinha do LangGraph: ao retomar, o nó que chamou o `interrupt` é
  executado de novo do início. Efeitos colaterais (como gravar a auditoria)
  não podem ficar nele, ou acontecem duas vezes. Eles foram para os nós antes e
  depois da pausa.
- O diagrama do grafo é exportado em Mermaid (`reports/grafo.mmd`). O PNG pelo
  LangGraph foi bloqueado de propósito: `draw_mermaid_png()` envia o diagrama
  para um serviço externo, e o princípio do projeto é que nada sai da máquina.

6.4 Linha de base da revisão (sem guardrails, 12 pacientes)
- Determinístico: alertas 12/12 iguais às regras, fontes 12/12 dentro dos
  protocolos vinculados, 12/12 pausados para aprovação.
- Conduta (LLM): 4 de 12 com erros de fidelidade sobre a paciente, e não sobre o protocolo:
  - P004: trastuzumabe e ecocardiograma de anti-HER2 numa paciente
    triplo-negativa (copiou um trecho condicional do protocolo sem verificar a condição);
  - P001: "monitorar a diabetes" numa paciente que só faz triagem;
  - P010: "em hormonioterapia" numa paciente em seguimento;
  - P011: disse que o protocolo não cobre e mesmo assim citou a fonte.
- P007: a conduta não menciona o atraso, que só o alerta informa.
  Omissão não foi rotulada como infidelidade: infidelidade é afirmar sem
  suporte. Penalizar omissões bloquearia quase toda resposta curta.

7. Segurança, auditoria e explicabilidade (Etapa 6)

7.1 Camadas
| Camada | Onde | O que faz |
|---|---|---|
| Entrada | nó `entrada` | Pedido de automedicação em 1ª pessoa → resposta fixa ("procure seu médico"). Pergunta do médico sobre a dose do protocolo não é bloqueada. Padrão de injeção → evento de segurança |
| Limiar de relevância | RAG | Pergunta distante dos protocolos é recusada antes do LLM |
| Garantias de arquitetura | dados | `mode=ro`, sem nome nas ferramentas, acesso só aos IDs citados, autorizador por coluna, só agregados, supressão k=3 |
| Saída determinística | `verificar_saida` | Fonte inventada, número sem suporte, status sem suporte e consistência anti-HER2 × perfil → reparo (1×) → recurso seguro |
| Juiz de fidelidade | `verificar_saida` | `llama3.2:3b`: sinalizador experimental, não barreira |
| Humano | `aprovacao_medica` | Toda conduta requer validação |

- Recurso seguro: no RAG, um aviso com o trecho do protocolo; na revisão, a
  conduta é omitida e seguem só os alertas e os trechos; no agente B, uma
  resposta montada direto do retorno das ferramentas.
- O ciclo verificar → reparar → verificar é o uso de ciclo no grafo,
  o que diferencia o LangGraph de uma cadeia linear.

7.2 Juiz de fidelidade: de barreira a sinalizador
- Desenho: recebe os protocolos e os dados da paciente (o trastuzumabe do
  P004 está literalmente no protocolo; só os dados mostram que não se aplica) e
  devolve JSON com as afirmações sem suporte.
- Avaliação em 24 itens rotulados (12 condutas + 12 respostas do RAG; 8 infiéis):

| Variante | Recall infiéis | Bloqueio indevido |
|---|---|---|
| Veredito escrito pelo juiz | 0,875 | 0,688 |
| Veredito derivado das afirmações | 0,875 | 0,562 |
| Modo "paciente" (só revisão, 12 itens) | 4/4 | 8/8 → não adotado |

- Falhas do juiz: erros de raciocínio sobre o protocolo (biópsia em BI-RADS 4
  "não indicada"), literalismo (paráfrase tratada como sem suporte) e um falso
  negativo em que o juiz inventou um dado da paciente para aprovar (P010).
- Decisão: um juiz de 3B não tem precisão para ser barreira. Bloqueando,
  descartaria mais da metade das condutas corretas. Passou a sinalizar.
- Fadiga de alarme: na revisão, o aviso aparece em quase toda conduta, e um
  alarme que toca sempre ensina a ser ignorado.
- O P004 mostrou o pior caso: o juiz sinalizou uma frase correta e deixou
  passar o ecocardiograma indevido. Um aviso no lugar errado sugere que o resto
  foi verificado. O texto do aviso foi reescrito: "verificação automática
  experimental levantou dúvida sobre «…»; ela não é exaustiva".
- `llama3.1:8b` como juiz: descartado pelo tempo na CPU (4–8 min por resposta
  na Fase 2). Fica como trabalho futuro.

7.3 A regra anti-HER2: o que pode virar regra deve virar regra
- O erro do P004 não precisa de LLM para ser detectado: o próprio protocolo
  restringe a terapia anti-HER2 e o monitoramento cardíaco a pacientes HER2 positivo.
- Regra determinística em `verificar_deterministico`: conduta que recomenda
  trastuzumabe, pertuzumabe ou ecocardiograma de anti-HER2 (fora de negação) para
  paciente triplo-negativa ou HER2 negativo → problema → reparo com o motivo
  explícito → recurso seguro se persistir. Testes: o caso P004 é pego, o P008
  (HER2+) não, e uma negação não é marcada.
- Uma regra só, de propósito. "Diabetes" (P001) e "hormonioterapia" (P010)
  são afirmações factuais sobre a paciente que nenhuma camada automática pega
  de forma confiável, e ficam para a validação médica.
- Resultado: o P004 passou a sair com 1 reparo e conduta correta (AC-T, sem anti-HER2).

7.4 Incidente: laço de geração com `format="json"`
- Sintoma: uma revisão do P004 levou 8.483 s (2h20), contra cerca de 150 s normais.
- Causa (confirmada pelo log do Ollama): com `format="json"`, o juiz entrou em
  laço gerando espaços em branco até o teto padrão, 40.960 tokens
  (10 × o contexto). Houve outra chamada de 2h08, antes da avaliação final.
- Correção: `num_predict=800` no juiz (o maior JSON legítimo medido tinha
  cerca de 490 tokens: um limite de 400 cortaria respostas normais), timeout de
  180 s, e truncamento ou timeout viram "indeterminado" com o motivo registrado.
  `num_predict=600` na conduta e no reparo. Depois disso, o P004 leva 21 s.
- Lição: toda chamada a LLM em produção precisa de limite de tokens e
  timeout. Sem eles, o pior caso de latência não tem limite, e num hospital
  isso é uma falha de disponibilidade.

7.5 Auditoria
- JSONL diário em `logs/auditoria/`, um registro por execução. Contém id,
  horário, modo, rota, eventos dos nós, modelos, fontes, guardrails disparados,
  veredito do juiz (com tokens, truncamento e timeout), reparos, eventos de
  segurança, decisão médica e o texto de cada tentativa.
- Revisão em dois registros com o mesmo id: `pendente_aprovacao` (gravado
  antes da pausa) e `decisao` (gravado na retomada). Uma sugestão exibida e
  nunca decidida não fica invisível.
- Pergunta e resposta gravadas em texto, com máscara de nome como segunda
  barreira. Uma primeira versão não gravava o conteúdo, e o log não conseguiria
  confirmar ou refutar "o sistema me disse X". Em saúde, o conflito entre
  privacidade e rastreabilidade se resolve protegendo o log, não esvaziando o log.
  O log é dado sensível: fica fora do git e, em produção, exigiria controle de
  acesso, criptografia e política de retenção.
- Explicabilidade: cada resposta traz o bloco "Por que esta resposta", com a
  rota, as fontes e seus trechos, as verificações, o veredito do juiz e o número de reparos.

8. Avaliação final do sistema completo

8.1 Método
- O grafo inteiro: o roteador escolhe o caminho, os guardrails atuam, e as
  métricas são calculadas sobre o texto final entregue ao médico (sem o
  bloco "Por que esta resposta", que é montado pelo código).
- Casos: 22 do RAG, 19 dos agentes e 12 revisões (retomadas com "aprovado"
  sintético). 3 repetições, 159 execuções. Gravação incremental com
  `--retomar`, e uma amostra rodada antes da execução completa.

8.2 Resultados por componente
| Etapa | Métrica | Baseline | Sistema final |
|---|---|---|---|
| RAG (ft_A, Etapa 3) | fonte inventada / número sem suporte | 1,5% / 1,5% | 0% / 0% |
| RAG | recusa adversarial (consistente) | 0,778 | 1,000 (1,000) |
| RAG | recusa fora do escopo (consistente) | 0,875 (0,75) | 0,833 (0,75) |
| RAG | cita fonte / fonte correta | 1,0 / 1,0 | 1,0 / 1,0 |
| Agente B (Etapa 4) | sem falha (geral) | 0,933 | 1,000 |
| Agente C (Etapa 4) | sem falha (agregado) | 0,5 | 0,5 |
| Revisão (Etapa 5) | alertas corretos / fontes ⊆ vinculados | 1,0 / 1,0 | 1,0 / 1,0 |
| Revisão | cita fonte | 1,0 | 0,778 |
| Revisão | infiéis do baseline sinalizados ou bloqueados | 0 | 1,0 |

- Sistema: rotas 100% corretas. Tempo médio: RAG 29 s, agentes 8 s,
  revisão 55 s. Bloqueio na entrada em 4,5% do RAG (a codeína, nas 3 repetições).
  Recurso seguro em 14% do agente B e 8% das revisões.

8.3 O juiz no sistema: um número que precisa de contexto
| Grupo | Sinalizado pelo juiz |
|---|---|
| Revisões infiéis do baseline (P001, P004, P010, P011) | 0,917 |
| Demais revisões | 0,750 |
| RAG com protocolo | 0,600 |
| RAG fora do escopo / adversarial | 0,333 / 0,417 |

- "100% dos infiéis sinalizados ou bloqueados" não pode ser apresentado
  sozinho: o juiz sinaliza 81% de todas as revisões, e a diferença entre infiéis
  (0,917) e fiéis (0,750) é pequena. A proteção real das condutas vem da regra
  anti-HER2, dos alertas por regra e da aprovação médica. O juiz de 3B tem pouco
  poder de discriminação e é registrado como camada experimental.

8.4 As quedas, explicadas
- `cita_fonte` da revisão (0,778): 3 de 8 são o recurso seguro funcionando
  ("conduta não gerada" não cita fonte); 5 são do modelo (P001 3×, P011 2× sem a
  linha "Fonte:") e entram como limitação.
- Recusa fora do escopo (0,875 → 0,833): 1 caso em 24, dentro da variação
  entre repetições.
- Agente C, geral (0,789 → 0,667): a composição dos casos mudou (o roteador
  manda ao C só as perguntas estatísticas). Os agregados continuam em 0,5, a
  limitação já registrada.
- Recurso seguro no agente B (14%): três perguntas simples em que o modelo
  insistiu num status que a ferramenta não retornou. A resposta montada do
  prontuário é correta. O baseline tinha o mesmo erro, só que entregue ao médico.

8.5 Conclusão da avaliação
O sistema completo entrega ao médico zero fonte inventada, zero número sem
suporte, 100% das recusas adversariais e 100% de alertas e rotas corretos.
Esses ganhos vêm das camadas determinísticas. O erro que nenhuma camada
automática pega de forma confiável, a afirmação factual errada sobre a paciente,
é tratado pela arquitetura: a conduta é sempre uma sugestão com fonte e trecho,
e passa pela aprovação médica registrada.

