# Diagramas do sistema

Fontes Mermaid dos diagramas do relatório técnico (renderizam direto no GitHub).
As versões em PNG estão em `reports/diagramas/`.

## Figura 1 — Arquitetura geral

Laranja: etapas com LLM · verde: processamento determinístico · azul: guardrails · amarelo: humano · cinza: dados.

```mermaid
flowchart TB
    subgraph OFF[" 1 · PREPARAÇÃO E FINE-TUNING — Google Colab, GPU T4 "]
        direction LR
        D1[(Protocolos internos<br/>10, sintéticos)] --> D2[Geração professor-aluno<br/>Llama 3.1 8B]
        D2 --> D3[Curadoria em 6 filtros<br/>+ revisão humana]
        D3 --> D4[Dataset formato RAG<br/>76 treino / 19 avaliação]
        D4 --> D5[QLoRA · Unsloth<br/>Llama 3.2 3B Instruct]
        D5 --> D6[GGUF q4_k_m<br/>Hugging Face Hub]
    end

    subgraph RUN[" 2 · ASSISTENTE — execução 100% local, CPU "]
        direction TB
        U([Médico]) --> G[Grafo LangGraph<br/>roteamento determinístico]
        G --> RAG[RAG LangChain<br/>ChromaDB + BM25 · RRF]
        G --> TB[Agente B<br/>ferramentas parametrizadas]
        G --> TC[Agente C<br/>text-to-SQL protegido]
        G --> RV[Revisão de paciente<br/>regras clínicas]
        RAG --> M1[[medassist<br/>fine-tunado · Ollama]]
        RV --> M1
        TB --> M2[[llama3.2:3b<br/>ferramentas · SQL · juiz]]
        TC --> M2
        TB --> DB[(Prontuários SQLite<br/>somente leitura)]
        TC --> DB
        RV --> DB
        RAG --> KB[(Base de protocolos<br/>embeddings MiniLM)]
        G --> GR{{Guardrails<br/>entrada · saída · juiz}}
        GR --> HM[/Aprovação médica/]
        G --> LOG[(Auditoria JSONL)]
    end

    D6 -. ollama create .-> M1
    D1 -. indexação .-> KB

    classDef llm fill:#FDE9D9,stroke:#C55A11,color:#000
    classDef data fill:#EDEDED,stroke:#595959,color:#000
    classDef guard fill:#DDEBF7,stroke:#1F4E78,color:#000
    classDef human fill:#FFF2CC,stroke:#BF8F00,color:#000
    classDef proc fill:#E2EFDA,stroke:#548235,color:#000
    class M1,M2,D2,D5 llm
    class D1,D4,D6,DB,KB,LOG data
    class GR guard
    class U,HM human
    class D3,G,RAG,TB,TC,RV proc
    style OFF fill:#FAFAFA,stroke:#999
    style RUN fill:#FAFAFA,stroke:#999
```

## Figura 2 — Fluxo do grafo LangGraph

Gerado a partir do grafo real (`src/grafo.py --diagrama` exporta a versão crua em `reports/grafo.mmd`); esta versão acrescenta os rótulos das arestas condicionais.

```mermaid
flowchart TD
    S([Início]) -->|modo pergunta| EX[extrair_ids]
    S -->|modo revisão| CP[carregar_paciente]

    subgraph P[" MODO PERGUNTA — assistente "]
        EX --> EN{{entrada<br/>filtro de automedicação<br/>e injeção}}
        EN -->|liberado| RT{rotear}
        RT -->|ID de paciente| AB[agente_b<br/>ferramentas parametrizadas]
        RT -->|vocabulário das regras| PN[painel<br/>template, sem LLM]
        RT -->|contagem / média| AC[agente_c<br/>SQL protegido]
        RT -->|demais| RG[rag<br/>protocolos + medassist]
    end

    subgraph R[" MODO REVISÃO — fluxo automatizado "]
        CP -->|paciente existe| AP[avaliar_pendencias<br/>regras clínicas]
        AP --> BP[buscar_protocolos<br/>pelo vínculo]
        BP --> SC[sugerir_conduta<br/>medassist]
    end

    AB --> VS
    AC --> VS
    PN --> VS
    RG --> VS
    SC --> VS

    VS{{verificar_saida<br/>determinística + juiz}}
    VS -->|problema| RP[reparar<br/>1 nova tentativa]
    RP --> VS
    VS -->|persistiu| RS[recurso_seguro]
    VS -->|ok · pergunta| FI
    VS -->|ok · revisão| GA
    RS -->|pergunta| FI
    RS -->|revisão| GA

    GA[gerar_alertas<br/>auditoria: pendente_aprovacao] --> AM[/aprovacao_medica<br/>interrupt — pausa/]
    AM --> RD[registrar_decisao<br/>auditoria: decisao]
    RD --> FI

    EN -->|bloqueado| FI
    CP -->|não encontrado| FI
    FI[finalizar<br/>resposta + 'Por que esta resposta'] --> E([Fim])

    classDef llm fill:#FDE9D9,stroke:#C55A11,color:#000
    classDef det fill:#E2EFDA,stroke:#548235,color:#000
    classDef guard fill:#DDEBF7,stroke:#1F4E78,color:#000
    classDef human fill:#FFF2CC,stroke:#BF8F00,color:#000
    class RG,SC,AB,AC,RP llm
    class EX,RT,PN,CP,AP,BP,GA,RD,FI det
    class EN,VS,RS guard
    class AM human
    style P fill:#FAFAFA,stroke:#999
    style R fill:#FAFAFA,stroke:#999
```
