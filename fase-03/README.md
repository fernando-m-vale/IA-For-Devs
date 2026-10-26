# Tech Challenge Fase 3 — Assistente médico com LLM fine-tunada, RAG e LangGraph

> **Aviso:** todos os dados deste projeto (protocolos, pacientes, exames, perguntas)
> são **100% sintéticos**, criados para fins acadêmicos. O sistema **não é um
> dispositivo médico e não deve ser usado em decisões clínicas reais**.

## 1. Visão geral

Assistente de apoio à decisão clínica para um hospital fictício. Ele:

- responde perguntas sobre **protocolos internos** com RAG (busca híbrida semântica + BM25)
  e um **LLaMA 3.2 3B fine-tunado** com QLoRA (`medassist`), sempre citando a fonte;
- consulta **prontuários** (SQLite) por dois caminhos: ferramentas parametrizadas por
  paciente (agente B) e text-to-SQL protegido só para estatísticas agregadas (agente C);
- monta um **painel de pendências** (exames atrasados ou críticos) por regras clínicas
  determinísticas ligadas a trechos literais dos protocolos;
- executa uma **revisão automatizada de paciente** em um fluxo LangGraph que pausa
  para **aprovação médica** (interrupt) antes de registrar a decisão;
- aplica **guardrails em camadas** (entrada, verificação determinística com reparo e
  recurso seguro, juiz de fidelidade como sinalizador) e grava **auditoria** de cada execução.

O histórico das decisões técnicas, com números de cada iteração, está em
[`reports/diario_de_bordo.md`](reports/diario_de_bordo.md).

## 2. Arquitetura

```
pergunta ──► extrair_ids ──► entrada (guardrail) ──► rotear ─┬─► agente_b  (ID P000 na pergunta)
                                   │                          ├─► painel    (atrasado/alerta/pendência/crítico — sem LLM)
                                   │ bloqueio                 ├─► agente_c  (quantos/média/total — text-to-SQL agregado)
                                   ▼                          └─► rag       (demais: protocolos)
                               finalizar ◄── verificar_saida ◄──┘   (reparar ↺, recurso_seguro)

revisão P000 ─► carregar_paciente ─► avaliar_pendencias ─► buscar_protocolos ─► sugerir_conduta (medassist)
              ─► verificar_saida ─► gerar_alertas ─► aprovacao_medica (interrupt) ─► registrar_decisao ─► finalizar
```

Diagrama completo gerado pelo próprio grafo: [`reports/grafo.mmd`](reports/grafo.mmd)
(Mermaid; abre no GitHub ou em qualquer visualizador Mermaid).

| Componente | Implementação |
|---|---|
| LLM de resposta | `medassist` = LLaMA 3.2 3B Instruct + QLoRA (Unsloth), GGUF Q4_K_M no Ollama |
| Agentes, juiz e baseline | `llama3.2:3b` no Ollama |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (CPU) |
| Vector store | ChromaDB persistente (cosseno) + BM25 (`rank_bm25`), fusão por RRF |
| Orquestração | LangGraph (`StateGraph`, `MemorySaver`, `interrupt`) |
| Prontuários | SQLite normalizado: `pacientes`, `exames`, `paciente_protocolos` |

## 3. Pré-requisitos

| Item | Versão / observação |
|---|---|
| Python | `>=3.10,<4.0` (testado com 3.13.13) |
| Poetry | 2.x (testado com 2.2.1) |
| Ollama | testado com 0.40.5 |
| Modelo `llama3.2:3b` | `ollama pull llama3.2:3b` |
| Modelo `medassist` | GGUF fine-tunado (~2 GB) + [`models/Modelfile`](models/Modelfile). **TODO: link do Hugging Face para o GGUF** |
| Modelo `llama3.1` (8B) | opcional: só para regerar o dataset sintético (modelo "professor") |
| Hardware usado | notebook só com CPU, 16 GB de RAM (Windows 11); fine-tuning no Google Colab (GPU T4) |

O modelo de embeddings é baixado do Hugging Face na primeira execução e depois usado
da cópia local.

## 4. Instalação

```bash
git clone https://github.com/fernando-m-vale/IA-For-Devs.git
cd IA-For-Devs/fase-03

# 1. Dependências (ambiente virtual em .venv/)
poetry config virtualenvs.in-project true   # opcional
poetry install

# 2. Dados sintéticos: protocolos.json, perguntas_medicas.json e prontuarios.db
poetry run python src/data_prep.py

# 3. Modelos no Ollama
ollama pull llama3.2:3b
#    baixe o GGUF do medassist para models/ (TODO: link do Hugging Face) e registre:
cd models && ollama create medassist -f Modelfile && cd ..

# 4. Índice vetorial dos protocolos (recria data/vectorstore/ e responde uma pergunta de exemplo)
poetry run python -m src.rag_pipeline --rebuild

# 5. Testes (sem LLM, ~5 s)
poetry run python -m unittest discover -s tests
```

O banco `prontuarios.db`, o índice `data/vectorstore/` e o GGUF não vão para o git:
são gerados (ou baixados) nos passos acima.

## 5. Uso

Toda interação passa pela CLI do grafo: `poetry run python -m src.grafo --help`.

```bash
# Pergunta sobre protocolo -> rota rag
poetry run python -m src.grafo --pergunta "O que caracteriza uma emergência de compressão medular em oncologia?"

# Pergunta sobre um paciente -> rota agente_b (só acessa os IDs citados)
poetry run python -m src.grafo --pergunta "O paciente P007 tem alguma pendência de exame atrasada?"

# Estatística -> rota agente_c (só agregados; grupos com menos de 3 registros são suprimidos)
poetry run python -m src.grafo --pergunta "Quantos exames pendentes (solicitados ou agendados) existem no total?"

# Painel -> rota painel (template, sem LLM)
poetry run python -m src.grafo --pergunta "Quais pacientes estão com exames atrasados?"

# Revisão de paciente: pausa para aprovação médica e registra a decisão
poetry run python -m src.grafo --revisar P008 --aprovar --comentario "conferido"
poetry run python -m src.grafo --revisar P002 --rejeitar --comentario "aguardar laudo"
poetry run python -m src.grafo --revisar P007        # sem flag: pergunta a decisão no terminal

# Exportar o diagrama (reports/grafo.mmd). --png também gera PNG, mas envia o
# diagrama ao serviço externo mermaid.ink para renderizar.
poetry run python -m src.grafo --diagrama --pergunta "Quais pacientes estão com exames atrasados?"
```

Saída real do painel (sem LLM):

```
Pendências atrasadas na data de referência 2026-10-20: 1 pendência(s) em 1 paciente(s).
- P007 | ATRASADO | USG mamária de controle (solicitado, previsto para 2026-10-15) — Previsto para 2026-10-15, antes da data de referência 2026-10-20. Seguimento de imagem em 6 meses de lesão BI-RADS 3.

Por que esta resposta:
- Rota: painel (vocabulário de painel ('atrasad'))
- Fontes:
  • HOSP-2024-RAD-001 — Uso do Sistema BI-RADS na Mamografia
- Verificações: não aplicável (texto montado pelo código ou resposta fixa)
- Reparos: 0

CAMINHO:
  extrair_ids             0.00s  IDs: nenhum
  entrada                 0.00s  liberado; eventos de segurança: nenhum
  rotear                  0.00s  rota=painel (vocabulário de painel ('atrasad'))
  painel                  0.02s  template, fontes=['HOSP-2024-RAD-001']
  verificar_saida         0.00s  sem verificação: resposta montada pelo código ou fixa
  finalizar               0.01s  auditoria: execucao (id dd2ae25c-…)
```

Pedido de automedicação bloqueado na entrada (sem LLM):

```bash
poetry run python -m src.grafo --pergunta "Qual dose de codeína eu devo tomar para dor nas costas?"
# Não posso orientar uso de medicamentos para você. Procure seu médico ou, em urgência, o pronto atendimento.
```

A **data de referência** das pendências é fixa (`DATA_REFERENCIA = "2026-10-20"` em
`src/config.py`) para os resultados serem reproduzíveis.

### Log de auditoria

Cada execução grava um registro JSON em `logs/auditoria/AAAA-MM-DD.jsonl`:
`execucao` (perguntas e revisões encerradas cedo), `pendente_aprovacao` (revisão parada
no interrupt, com conduta, alertas e o texto de cada tentativa verificada) e `decisao`
(retomada, com a decisão e o comentário). Os dois registros de uma revisão têm o mesmo `id`.

```bash
poetry run python -c "import glob, json; [print(r['timestamp'], r['tipo'], r['id'][:8], r.get('rota'), r.get('veredito_juiz')) for f in sorted(glob.glob('logs/auditoria/*.jsonl')) for r in map(json.loads, open(f, encoding='utf-8'))]"
```

## 6. Fine-tuning (reprodução)

1. **Datasets** (já versionados em `data/synthetic/`; regerar é opcional):
   - `poetry run python src/data_prep.py` — protocolos, perguntas de avaliação e prontuários;
   - `poetry run python -m src.data_augmentation` — o `llama3.1` gera pares a partir dos
     protocolos e a curadoria (6 filtros, incluindo revisão humana registrada em
     `revisao_humana.json`) produz `finetune_train.jsonl` / `finetune_eval.jsonl`. O script é
     retomável e não regera o que já está em `qa_bruto_v2.json`;
   - `poetry run python -m src.data_augmentation --rag` — formato RAG (v2, usado no sistema):
     `finetune_train_rag.jsonl` / `finetune_eval_rag.jsonl`.
2. **Treino no Google Colab** (Ambiente de execução → T4 GPU), na ordem:
   - [`notebooks/01_finetuning_qlora_v1.ipynb`](notebooks/01_finetuning_qlora_v1.ipynb) —
     v1, formato pergunta → resposta (histórico; mantido com as saídas);
   - [`notebooks/01_finetuning_qlora_v2_rag.ipynb`](notebooks/01_finetuning_qlora_v2_rag.ipynb) —
     **v2, formato RAG (o modelo usado)**. Faça upload de `finetune_train_rag.jsonl` e
     `finetune_eval_rag.jsonl`; o notebook treina (QLoRA, r=16, 3 épocas), avalia antes/depois
     e exporta para o Google Drive (`/MyDrive/fiap_fase3`): o GGUF `Q4_K_M`, o `Modelfile`,
     `curva_perda_v2.png` e `avaliacao_finetuning_v2.json`.
3. **Registro local:** copie o GGUF para `models/` e use o
   [`models/Modelfile`](models/Modelfile) **do repositório** (não o exportado): ele fixa o system
   prompt do treino, `temperature 0.3` e `num_ctx 4096` (o contexto RAG com dois protocolos
   passa do padrão de 2048). Depois, `ollama create medassist -f Modelfile`.

Resultados do fine-tuning: [`outputs/finetuning_v1/`](outputs/finetuning_v1/) e
[`outputs/finetuning_v2/`](outputs/finetuning_v2/).

## 7. Avaliações e testes

Todos os comandos rodam a partir de `fase-03/` com `poetry run python -m …`. Tempos medidos
em CPU (16 GB); variam com a máquina.

| Avaliação | Comando | LLM | Tempo aprox. | Resultados |
|---|---|---|---|---|
| RAG (base × fine-tunado, 3 rep.) | `src.evaluate_rag` | sim | 30–60 min | `outputs/rag_eval/` |
| Calibração do limiar | `src.evaluate_rag --calibrar` | não | ~2 min | `outputs/rag_eval/calibracao_limiar.csv` |
| Recuperação semântica × híbrida | `src.evaluate_rag --recuperacao` | não | ~2 min | `outputs/rag_eval/comparacao_recuperacao.csv` |
| Ablação da busca | `src.diagnostico_recuperacao` | não | ~2 min | saída no terminal |
| Agente B (ferramentas, 19 casos) | `src.evaluate_agente --repeticoes 3` | sim | 30–45 min | `outputs/agente_eval/` |
| Agente C (text-to-SQL, 19 casos) | `src.evaluate_agente_sql --repeticoes 3` | sim | 10–15 min | `outputs/agente_sql_eval/` |
| Juiz de fidelidade (24 itens rotulados) | `src.evaluate_juiz` | sim | 15–20 min | `outputs/juiz_eval/` |
| Revisão dos 12 pacientes pelo grafo | `src.evaluate_grafo` | sim | 5–15 min | `outputs/grafo_eval/` |
| Sistema completo — amostra | `src.evaluate_final --amostra` | sim | 10–15 min | `outputs/final_eval/amostra.jsonl` |
| Sistema completo (53 casos × 3) | `src.evaluate_final --repeticoes 3` | sim | ~3 h (estimativa); `--retomar` continua de onde parou | `outputs/final_eval/` |

- **Os resultados brutos (`outputs/**/*.json`, ~1 MB) são versionados**: com eles, todas as
  tabelas do relatório podem ser recalculadas **sem LLM**, com `--reavaliar ARQUIVO.json`
  (em `evaluate_rag`, `evaluate_agente` e `evaluate_juiz`) e `evaluate_final --so-resumo`.
  Isso também permite mudar uma métrica e medir o efeito nas respostas já geradas, sem
  rodar os modelos de novo.
- O log de auditoria (`logs/`) **não** é versionado (dado sensível).
- **Testes** (sem LLM; LLM, juiz e auditoria são substituídos por mocks ou diretório temporário):

```bash
poetry run python -m unittest discover -s tests
```

Incluem testes de regressão do avaliador com respostas reais rotuladas à mão
(`tests/fixtures/`), dos guardrails, do roteador, da revisão com interrupt e da auditoria.

## 8. Estrutura

```
fase-03/
├── data/synthetic/           dados sintéticos (protocolos, Q&A, datasets de fine-tuning, revisão humana)
├── models/Modelfile          registro do medassist no Ollama (o .gguf fica fora do git)
├── notebooks/                fine-tuning no Colab: v1 (pergunta → resposta) e v2 (formato RAG)
├── outputs/                  resultados das avaliações (JSONs brutos e resumos .csv) e do fine-tuning
├── reports/                  diário de bordo e diagrama do grafo (grafo.mmd)
├── logs/auditoria/           log de auditoria JSONL (gerado; fora do git — dado sensível)
├── tests/                    testes unitários e fixtures rotuladas
└── src/
    ├── config.py                 DATA_REFERENCIA, status pendentes e K_MIN da supressão
    ├── data_prep.py              gera protocolos, perguntas de avaliação e o banco de prontuários
    ├── data_augmentation.py      aumento sintético com modelo professor, curadoria e datasets de fine-tuning
    ├── rag_pipeline.py           índice ChromaDB, busca híbrida (BM25 + semântica, RRF) e AssistenteRAG
    ├── regras_clinicas.py        regras de criticidade com trecho de protocolo validado; classifica pendências
    ├── prontuarios.py            acesso somente leitura ao banco (ferramentas do agente B, sem nomes)
    ├── agente_prontuarios.py     agente B: tool calling com roteamento por ID e curto-circuitos
    ├── agente_sql.py             agente C: text-to-SQL agregado com autorizador, validações e supressão k<3
    ├── guardrails.py             filtro de entrada, verificação determinística e juiz de fidelidade
    ├── auditoria.py              registros JSONL por execução, sem nomes
    ├── grafo.py                  grafo LangGraph (modos pergunta e revisão) e CLI
    ├── evaluate_rag.py           avaliação do RAG, calibração do limiar e comparação de recuperação
    ├── diagnostico_recuperacao.py ablação da busca híbrida (stopwords × âncora semântica)
    ├── evaluate_agente.py        avaliação do agente B (e casos agregados)
    ├── evaluate_agente_sql.py    avaliação do agente C com gabarito por SQL
    ├── evaluate_juiz.py          avaliação do juiz contra o conjunto rotulado
    ├── evaluate_grafo.py         revisão real dos 12 pacientes pelo grafo
    └── evaluate_final.py         avaliação do sistema completo e comparação com os baselines
```

## 9. Segurança

Defesa em camadas — cada uma cobre o que a anterior não pega:

1. **Entrada:** pedidos de automedicação/prescrição em 1ª pessoa recebem resposta fixa sem
   chegar a nenhum modelo; padrões de injeção de prompt viram evento de segurança.
2. **Roteamento estrutural:** pergunta com ID só acessa as ferramentas por paciente e **só os
   IDs citados** (outro ID é rejeitado sem executar e registrado); pergunta sem ID só acessa o
   painel; no máximo 4 chamadas de ferramenta por iteração.
3. **Dados:**
   - conexão SQLite **somente leitura** (`mode=ro`);
   - as ferramentas **não devolvem nomes** (o modelo não pode revelar o que não recebe);
   - text-to-SQL só aceita **consultas agregadas** validadas (uma instrução, sem comentários,
     sem filtro por paciente, rótulo do GROUP BY no SELECT, valores categóricos existentes);
   - **autorizador por coluna** (`set_authorizer`) nega `nome_anonimizado`, `sqlite_master` e
     pragmas; limite de tempo por consulta e de 50 linhas;
   - **supressão de células pequenas**: resultados com 1 ou 2 registros (`K_MIN = 3`) não são divulgados.
4. **Recuperação:** limiar de relevância (0,47, calibrado em conjunto separado) recusa perguntas
   sem protocolo antes de chamar o LLM.
5. **Saída:** verificação determinística (fonte inventada, número sem suporte, status sem
   suporte) com 1 reparo e, se falhar de novo, **recurso seguro** (texto do modelo descartado,
   resposta montada pelo código); **juiz de fidelidade** como sinalizador: o que ele não
   confirma vira aviso no topo da resposta, sem bloquear.
6. **Humano no circuito:** a revisão de paciente pausa para aprovação médica, e toda saída de
   revisão termina com "Sugestão de apoio à decisão — requer validação médica."
7. **Auditoria:** um registro por execução (rota, nós e tempos, fontes, guardrails, veredito do
   juiz, reparos, eventos de segurança, decisão médica), sem nomes de paciente (campos
   removidos e "Paciente X" mascarado).

> `logs/` é **dado sensível**: contém conteúdo clínico pseudonimizado. Fica fora do git; em
> produção exigiria controle de acesso, criptografia em repouso e política de retenção.

## 10. Limitações

- Dados sintéticos e pequenos (10 protocolos, 12 pacientes): os números não generalizam.
- Modelos de 3B em CPU: respostas lentas (10–60 s) e não determinísticas mesmo com temperatura 0.
- O fine-tuning ensinou comportamento (citar fonte, recusar), não conhecimento; as respostas ficaram mais curtas que as do modelo base.
- O juiz de fidelidade de 3B reprova muitas respostas fiéis; por isso só sinaliza, não bloqueia.
- As verificações determinísticas checam forma (fontes, números, status), não atribuição (o número certo do paciente errado passa).
- O text-to-SQL pode gerar consulta válida que responde outra pergunta (ex.: contagem de atrasados = 0).
- O limiar de relevância mede proximidade de tema, não cobertura: perguntas vizinhas aos protocolos passam.
- A supressão conta registros, não pacientes distintos, e MIN/MAX podem revelar o valor de um indivíduo.
- O roteador é por vocabulário: perguntas fora dos padrões caem no RAG.
- A aprovação médica usa `MemorySaver`: revisões pendentes se perdem ao encerrar o processo.
- Avaliações com poucos casos (12 a 24 por conjunto) e 1 a 3 repetições.
