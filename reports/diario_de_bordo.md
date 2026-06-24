# Diário de Bordo — Tech Challenge Fase 1 (IA Foundations)

> Este documento é o rascunho vivo do relatório técnico final. Cada decisão e
> descoberta relevante é registrada aqui, na ordem em que aconteceu, para que
> a versão final seja apenas uma revisão e formatação deste conteúdo — não uma
> reescrita do zero.

## 1. Escolha do desafio e dataset

- **Desafio escolhido:** B — Rede de hospitais focada em saúde da mulher.
- **Motivo:** tecnicamente mais leve que o Desafio A (não exige split de
  validação separado, Dockerfile é opcional), e o dataset principal sugerido
  é o mesmo nos dois desafios.
- **Dataset escolhido:** Breast Cancer Wisconsin (Diagnostic), via UCI
  Machine Learning Repository (id=17), acessado pela biblioteca `ucimlrepo`.
- **Motivo da fonte:** é a fonte primária do dataset (Kaggle e scikit-learn
  apenas replicam essa origem). Usar a fonte oficial via API mostra acesso
  aos metadados completos e permite citação acadêmica formal no relatório.
- **Extra (CNN com mamografias):** decisão em aberto, será avaliada conforme
  o tempo disponível ao final das etapas obrigatórias.

## 2. Estrutura do projeto

Estrutura completa com notebooks (exploração e narrativa) + `src/` (funções
reutilizáveis), em vez de notebooks monolíticos. Justificativa: separa a
lógica testável da exploração visual, e demonstra organização de projeto.

```
notebooks/  → 01_eda, 02_preprocessing, 03_modeling, 04_evaluation_shap
src/        → data_loader.py, preprocessing.py, models.py, evaluate.py
data/       → raw/ (dataset original) e processed/ (dataset tratado)
reports/    → relatório técnico final e figuras
```

## 3. Etapa 1 — Análise Exploratória de Dados (EDA)

### 3.1 Carregamento e estrutura

- Função `load_breast_cancer_data()` em `src/data_loader.py` busca o dataset
  via `ucimlrepo`, combina features e target em um único DataFrame, e salva
  uma cópia local em `data/raw/breast_cancer_wisconsin.csv`.
- **Bug encontrado e corrigido:** `y.values` retornava array 2D (porque
  `ucimlrepo` retorna o target como DataFrame, não Series), causando
  `ValueError` ao atribuir à coluna `diagnosis`. Corrigido com
  `y.iloc[:, 0].values`, extraindo a Series antes de converter para array.
- **Resultado:** 569 linhas, 31 colunas (30 features numéricas + coluna
  `diagnosis`).
- **Valores ausentes:** nenhum. Todas as 31 colunas têm 569 valores não-nulos.
  Dispensa etapa de imputação no pré-processamento.

### 3.2 Estrutura das features

As 30 features numéricas correspondem a 10 características-base do núcleo
celular (raio, textura, perímetro, área, suavidade, compacidade, concavidade,
pontos de concavidade, simetria, dimensão fractal), cada uma medida de 3
formas: média (sufixo 1), erro padrão (sufixo 2) e "pior caso" (sufixo 3).
Isso antecipa alta correlação esperada entre variantes da mesma característica
(ex: `radius1` e `radius3`), a ser confirmada formalmente na Etapa 2.

### 3.3 Balanceamento das classes

| Diagnóstico | Contagem | Percentual |
|---|---|---|
| Benigno (B) | 357 | 62.7% |
| Maligno (M) | 212 | 37.3% |

**Discussão:** desbalanceamento leve, não severo (não exige técnicas como
SMOTE). Ainda assim, é relevante para a escolha de métrica de avaliação: um
classificador trivial que sempre prevista "Benigno" atingiria 62.7% de
accuracy sem capturar nenhum caso real de câncer. Isso justifica priorizar
**recall** (capacidade de detectar os casos malignos) como métrica mais
relevante clinicamente, complementar à accuracy — falso negativo (classificar
maligno como benigno) é o erro de maior risco neste contexto.

### 3.4 Estatísticas descritivas

- Grande disparidade de escala entre features (ex: `area1` varia de 143.5 a
  2501, enquanto `smoothness1` varia de 0.05 a 0.16). Isso é esperado, dado
  que as unidades físicas são diferentes, mas implica a necessidade de
  **padronização/normalização** na Etapa 2, já que alguns algoritmos (KNN,
  Regressão Logística) são sensíveis à escala das variáveis.
- `concavity1` e `concave_points1` têm valor mínimo igual a zero — alguns
  tumores não apresentam concavidade detectável nessa medida.

### 3.5 Distribuições por diagnóstico

Comparando histogramas de features-chave, separados por diagnóstico:

- **Separação forte:** `radius1`, `perimeter1`, `area1` — tumores malignos
  tendem a ser fisicamente maiores, com picos de distribuição bem distintos
  entre as classes.
- **Separação forte (padrão diferente):** `concavity1` — benignos concentram-se
  perto de zero; malignos têm distribuição mais espalhada e deslocada para
  valores mais altos.
- **Separação moderada:** `texture1` — discrimina, mas com sobreposição
  maior na região intermediária.
- **Separação fraca:** `smoothness1` — distribuições quase totalmente
  sobrepostas entre as classes; discrimina pouco isoladamente.

**Nota para Etapa 5 (SHAP):** vale conferir se a importância de features
atribuída pelos modelos confirma esse padrão visual (espera-se que
`smoothness1` tenha baixa importância relativa, e `area1`/`concavity1` alta).

## 4. Próximos passos

- [ ] Etapa 2: análise de correlação formal + decisão de padronização
- [ ] Etapa 3: escolha dos algoritmos de classificação
- [ ] Etapa 4: treinamento, métricas e justificativa da métrica principal
- [ ] Etapa 5: feature importance e SHAP
- [ ] Decisão final sobre o extra de CNN
- [ ] Etapa 7: README, Dockerfile (se aplicável), relatório final
- [ ] Etapa 8: vídeo de demonstração
