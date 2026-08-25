# Tech Challenge Fase 2 — Otimização de Modelos via AG + LLM

> Pós-Graduação em IA para Devs — FIAP | Fase 2: Algoritmos Genéticos e LLM

## Sobre o projeto

Evolução do sistema de diagnóstico de câncer de mama desenvolvido na Fase 1,
com dois novos componentes:

1. **Algoritmo Genético** para otimização automática de hiperparâmetros dos
   modelos de ML (Regressão Logística, Random Forest, KNN e SVM).
2. **Integração com LLM local** (LLaMA 3.1 8B via Ollama) para geração de
   explicações em linguagem natural dos diagnósticos, tornando os resultados
   interpretáveis para profissionais de saúde.

## Resultados principais

| Modelo | Baseline (Fase 1) | AG Otimizado | Ganho |
|---|---|---|---|
| Logistic Regression | 0.9440 | 0.9577 | +0.0137 |
| Random Forest | 0.9340 | 0.9435 | +0.0095 |
| KNN | 0.9250 | 0.9385 | +0.0135 |
| SVM | 0.9576 | 0.9717 | +0.0141 |

**Métrica:** Recall médio (5-fold Cross-Validation estratificado)

**Qualidade das explicações LLM:** 8.5–8.8/10 (auto-avaliação via
self-evaluation prompting)

## Como executar

### Pré-requisitos
- Python 3.10+
- Poetry (`pip install poetry`)
- Ollama instalado ([ollama.com/download](https://ollama.com/download))

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/fernando-m-vale/IA-For-Devs.git
cd IA-For-Devs/fase-02

# 2. Instalar dependências via Poetry
poetry install

# 3. Baixar o modelo LLM (necessário apenas uma vez, ~5 GB)
ollama pull llama3.1

# 4. Executar os notebooks na ordem
poetry run jupyter notebook notebooks/
```

Execute na seguinte ordem:
1. `01_genetic_algorithm.ipynb` — otimização via AG (pode demorar,
   especialmente o Exp 3 com Random Forest ~20 min)
2. `02_llm_integration.ipynb` — integração com LLM (requer Ollama rodando)

O dataset é baixado automaticamente via `ucimlrepo` na primeira execução.

## Algoritmo Genético

**Operadores implementados:**
- Seleção por torneio (tournament_size=3)
- Cruzamento uniforme
- Mutação por gene
- Elitismo (n=2)

**Experimentos realizados:**

| Experimento | População | Gerações | Mutação | Resultado |
|---|---|---|---|---|
| 1 — Conservador | 10 | 10 | 0.10 | Bom para KNN |
| 2 — Balanceado | 20 | 20 | 0.20 | Melhor em 3/4 modelos |
| 3 — Exploratório | 30 | 30 | 0.30 | Mais custoso, ganho marginal |

## Integração com LLM

**Modelo:** LLaMA 3.1 8B (Ollama local — sem API key, sem internet)

**Técnicas de prompt engineering:**
- Role prompting
- Structured input
- Output format specification
- Chain-of-thought
- Self-evaluation prompting

## Dataset

- **Nome:** Breast Cancer Wisconsin (Diagnostic)
- **Fonte:** [UCI ML Repository, id=17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Acesso:** automático via `ucimlrepo` na primeira execução

## Vídeo de demonstração

🔗 [https://www.youtube.com/watch?v=5fyzxE77Jc8](https://www.youtube.com/watch?v=5fyzxE77Jc8)

## Autor

**Fernando Marques do Vale**
Pós-Graduação em IA para Devs — FIAP
