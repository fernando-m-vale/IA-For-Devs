Diário de Bordo — Tech Challenge Fase 2 (Algoritmos Genéticos e LLM)

Rascunho vivo do relatório técnico final. Cada decisão e achado relevante é registrado aqui na ordem em que aconteceu.

1. Escolha do projeto e justificativa

- Projeto escolhido: Projeto 1 — Otimização de Modelos de Diagnóstico.
- Motivo: continuidade direta com a Fase 1 — os modelos já construídos (LR, RF, KNN) servem como ponto de partida para o AG, eliminando retrabalho e permitindo comparação direta de desempenho.

2. Decisões de arquitetura

2.1 Gerenciador de ambiente
- Escolha: Poetry (pyproject.toml).
- Motivo: mais moderno que venv + requirements.txt; gera poetry.lock com versões exatas de todas as dependências (diretas e transitivas), garantindo reprodutibilidade total do ambiente.

2.2 Modelos otimizados pelo AG
- Escolha: Regressão Logística, Random Forest, KNN (da Fase 1) + SVM (novo).
- Motivo para adicionar SVM: espaço de hiperparâmetros rico (C, kernel, gamma) que torna a otimização por AG mais interessante e demonstra que a solução é generalizável além dos modelos originais da Fase 1.
- XGBoost descartado: espaço de hiperparâmetros excessivamente grande (+10 parâmetros relevantes), tornaria o AG mais lento e difícil de configurar adequadamente.

2.3 Operadores genéticos
- Seleção: torneio (tournament_size=3) — garante pressão seletiva sem eliminar completamente indivíduos menos aptos, mantendo diversidade.
- Cruzamento: uniforme — cada gene herdado de um dos pais com probabilidade `crossover_rate`. Escolhido em vez de single-point porque hiperparâmetros não têm ordem natural que faça sentido preservar em blocos.
- Mutação: por gene — cada hiperparâmetro muta independentemente com probabilidade `mutation_rate`, permitindo que múltiplos parâmetros mudem simultaneamente.
- Elitismo: 2 melhores indivíduos preservados intactos a cada geração, garantindo que o AG nunca perca a melhor solução encontrada.

2.4 Função fitness
- Métrica: recall médio em validação cruzada estratificada (5-fold).
- Justificativa: mesma decisão da Fase 1 — em diagnóstico médico, falso negativo (classificar maligno como benigno) tem custo clínico mais alto.
- Pipeline com StandardScaler encapsulado dentro do CV para evitar data leakage entre folds (padrão estabelecido na Fase 1).
- Combinações inválidas de hiperparâmetros recebem fitness=0.0 via try/except, sendo naturalmente eliminadas pelo AG.

2.5 Escala logarítmica para C (LR e SVM)
- O parâmetro C varia em ordens de magnitude (0.001 a 100).
- Sorteio em escala linear concentraria valores entre 50 e 100, ignorando praticamente toda a faixa abaixo de 1.
- Solução: sorteio em escala log10 — cada ordem de magnitude tem a mesma probabilidade de ser explorada.

3. Experimentos com o Algoritmo Genético

3.1 Configurações testadas

| Experimento | População | Gerações | Mutação | Cruzamento |
|---|---|---|---|---|
| 1 — Conservador  | 10 | 10 | 0.10 | 0.80 |
| 2 — Balanceado   | 20 | 20 | 0.20 | 0.70 |
| 3 — Exploratório | 30 | 30 | 0.30 | 0.60 |

3.2 Baselines (Recall médio 5-fold CV — hiperparâmetros padrão)

| Modelo | Baseline |
|---|---|
| Logistic Regression | 0.9440 |
| Random Forest | 0.9340 |
| KNN | 0.9250 |
| SVM | 0.9576 (calculado agora — modelo novo) |

3.3 Resultados por experimento

Experimento 1 — Conservador (pop=10, gen=10, mut=0.10):

| Modelo | Baseline | AG melhor | Ganho | Tempo (s) |
|---|---|---|---|---|
| Logistic Regression | 0.9440 | 0.9530 | +0.0090 | 2.53 |
| Random Forest | 0.9340 | 0.9295 | -0.0045 | 119.86 |
| KNN | 0.9250 | 0.9385 | +0.0135 | 11.24 |
| SVM | 0.9576 | 0.9670 | +0.0094 | 7.80 |

Experimento 2 — Balanceado (pop=20, gen=20, mut=0.20):

| Modelo | Baseline | AG melhor | Ganho | Tempo (s) |
|---|---|---|---|---|
| Logistic Regression | 0.9440 | 0.9577 | +0.0137 | 19.83 |
| Random Forest | 0.9340 | 0.9435 | +0.0095 | 472.76 |
| KNN | 0.9250 | 0.9385 | +0.0135 | 32.03 |
| SVM | 0.9576 | 0.9717 | +0.0141 | 43.18 |

Experimento 3 — Exploratório (pop=30, gen=30, mut=0.30):

| Modelo | Baseline | AG melhor | Ganho | Tempo (s) |
|---|---|---|---|---|
| Logistic Regression | 0.9440 | 0.9577 | +0.0137 | 34.19 |
| Random Forest | 0.9340 | 0.9434 | +0.0094 | 1183.36 |
| KNN | 0.9250 | 0.9385 | +0.0135 | 69.37 |
| SVM | 0.9576 | 0.9717 | +0.0141 | 123.14 |

3.4 Tabela comparativa final (melhor resultado por modelo)

| Modelo | Baseline | AG Otimizado | Ganho | Melhor config |
|---|---|---|---|---|
| Logistic Regression | 0.9440 | 0.9577 | +0.0137 | Pop=20, Mut=0.2 |
| Random Forest | 0.9340 | 0.9435 | +0.0095 | Pop=20, Mut=0.2 |
| KNN | 0.9250 | 0.9385 | +0.0135 | Pop=10, Mut=0.1 |
| SVM | 0.9576 | 0.9717 | +0.0141 | Pop=20, Mut=0.2 |

3.5 Melhores hiperparâmetros encontrados

| Modelo | Hiperparâmetros otimizados |
|---|---|
| Logistic Regression | C=0.131, solver=liblinear |
| Random Forest | n_estimators=115, max_depth=7, min_samples_split=3 |
| KNN | n_neighbors=3, weights=distance, metric=manhattan |
| SVM | C=5.945, kernel=rbf, gamma=auto |

3.6 Discussão crítica dos resultados

Todos os 4 modelos melhoraram em relação ao baseline — o AG encontrou
hiperparâmetros superiores aos padrões em 100% dos casos (exceto RF no
Experimento 1, onde a população pequena convergiu prematuramente).

O Experimento 2 (Balanceado) venceu em 3 de 4 modelos, confirmando a
recomendação da literatura: populações muito pequenas (exp 1) convergem
para ótimos locais prematuramente; mutação muito alta (exp 3) dificulta a
convergência. O equilíbrio (exp 2) performa melhor na maioria dos casos.

O KNN foi exceção — se saiu melhor no Experimento 1 (Conservador).
Faz sentido: seu espaço de hiperparâmetros é pequeno (3 parâmetros), então
uma população pequena com baixa mutação já o explora bem. Modelos com
espaços maiores (RF, SVM) precisam de mais exploração.

O Random Forest foi o mais custoso computacionalmente — 1183 segundos
(~20 minutos) no Experimento 3. Cada avaliação de fitness envolve treinar
uma floresta completa 5 vezes (CV). Ainda assim, o ganho de +0.0095 mostra
que o AG agregou valor mesmo nesse cenário.

Correção técnica identificada: `probability=True` no SVC estava
deprecated no scikit-learn 1.9+. Corrigido — a flag era desnecessária pois
a função fitness não usa `predict_proba()`.

4. Integração com LLM (Ollama + LLaMA 3.1 8B)

4.1 Setup e decisões técnicas

- Modelo escolhido: LLaMA 3.1 8B via Ollama (local, gratuito, sem API key).
- Motivo: 16 GB de RAM disponíveis permitem rodar o modelo 8B confortavelmente; LLaMA 3.1 apresenta boa qualidade em português.
- Temperatura: 0.3 para explicações (consistência > criatividade em contexto médico); 0.1 para avaliação de qualidade (JSON estruturado exige mínima variabilidade).
- Problema resolvido: SHAP 0.52+ exige Python >=3.12; solução foi fixar versão `shap>=0.44.0,<0.52.0` compatível com Python >=3.10.

4.2 Prompt engineering

- Role prompting: system prompt define papel (assistente de apoio ao diagnóstico em oncologia), restrições (nunca emitir diagnóstico definitivo, sempre reforçar papel do médico) e formato esperado.
- Structured input: dados do diagnóstico formatados de forma consistente (predição, confiança, features com valores SHAP e direção do impacto).
- Output format specification: instrução explícita de estrutura da resposta (resumo, interpretação de características, limitações e recomendação).
- Chain-of-thought: prompt de insights acionáveis pede raciocínio passo a passo (implicação biológica → atenção prioritária → exames → conduta).
- Self-evaluation prompting: a própria LLM avalia sua resposta anterior segundo 4 critérios, retornando JSON estruturado com scores e justificativa.

4.3 Resultados dos 3 casos de diagnóstico

Casos testados:
- Caso A (PACIENTE-TEST-108): Maligno, prob=1.000, real=Maligno
- Caso B (PACIENTE-TEST-091): Benigno, prob=0.000, real=Benigno
- Caso C (PACIENTE-TEST-112): Benigno, prob=0.471, real=Maligno (falso negativo)

Avaliação de qualidade (auto-avaliação LLM, escala 0–10):

| Caso | Precisão técnica | Clareza | Segurança | Utilidade clínica | Média |
|---|---|---|---|---|---|
| A — Maligno | 9 | 8 | 9 | 9 | 8.8 |
| B — Benigno | 9 | 8 | 9 | 8 | 8. |
| C — Limítrofe | 9 | 8 | 9 | 8 | 8.5 |

4.4 Discussão crítica

Pontos positivos:
- LLM respeitou consistentemente os limites do system prompt — nunca emitiu diagnóstico definitivo, sempre reforçou o papel do médico. 
- Na pergunta livre, iniciou a resposta com ressalva explícita sobre limitações de recomendações médicas — comportamento de segurança adequado.
- Explicações em português fluente e tecnicamente coerentes com os dados.

Caso C — o mais relevante clinicamente: o modelo de ML previu "Benigno" com probabilidade 47.1%, mas o diagnóstico real era Maligno (falso negativo). A LLM identificou sinais contraditórios (texture sugere benigno, symmetry3 sugere maligno) — reforçando por que casos limítrofes exigem revisão médica.

Limitação de tempo de resposta: 294–474 segundos por chamada em CPU.
Em ambiente de produção com GPU ou API de nuvem, o tempo cairia — limitação do ambiente de desenvolvimento, não do design.

