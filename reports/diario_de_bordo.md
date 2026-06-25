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

## 4. Etapa 2 — Pré-processamento

### 4.1 Análise de correlação

**Multicolinearidade entre features:** confirmada a suspeita levantada na
EDA. O heatmap de correlação mostra blocos fortes entre variantes da mesma
característica geométrica — `radius`, `perimeter` e `area` (nas três versões:
mean, se, worst) apresentam correlação próxima de 0.99 entre si, o que é
esperado matematicamente (raio, perímetro e área de uma estrutura aproximadamente
circular são dependentes entre si). Padrão semelhante ocorre entre
`concavity` e `concave_points`.

**Decisão:** manter todas as 30 features no dataset, em vez de remover
manualmente as redundantes ou aplicar PCA. Justificativa: preserva a
interpretabilidade exigida pelo desafio (feature importance e SHAP na Etapa 5),
e a multicolinearidade afeta principalmente a estabilidade dos coeficientes
de Regressão Logística — não a performance preditiva, e não afeta modelos
baseados em árvores. Essa limitação será mencionada na discussão crítica do
relatório final.

**Correlação com o diagnóstico:** ranking das features mais correlacionadas
com o diagnóstico maligno confirma e refina os achados visuais da EDA:

| Posição | Feature | Correlação |
|---|---|---|
| 1º | `concave_points3` | 0.79 |
| 2º | `perimeter3` | 0.78 |
| 3º | `concave_points1` | 0.78 |
| 4º | `radius3` | 0.78 |
| ~7º-8º | `area1`, `perimeter1`, `radius1` | 0.71–0.74 |
| ~21º | `smoothness1` | 0.36 |
| Próximo de 0 | `symmetry2`, `texture2`, `fractal_dimension1`, `smoothness2` | -0.07 a 0.08 |

**Achado relevante:** as features de "erro padrão" (sufixo 2, exceto
`radius2`/`perimeter2`/`area2`) têm correlação próxima de zero com o
diagnóstico — a variabilidade da medição não discrimina bem entre as classes;
o que importa é o valor médio e o "pior caso" de cada característica.

### 4.2 Separação treino/teste

- Split 80/20 (`test_size=0.2`), com `stratify=y` para preservar a proporção
  de classes em ambos os conjuntos, e `random_state=42` para reprodutibilidade.
- Resultado: 455 amostras de treino, 114 de teste.
- Proporção de Maligno: 37.4% no treino vs 36.8% no teste — próximas entre si
  e do valor original do dataset completo (37.3%), confirmando que a
  estratificação funcionou corretamente.

### 4.3 Padronização (StandardScaler)

- **Motivo:** grande disparidade de escala entre features (ex: `area1` em
  centenas/milhares vs `smoothness1` entre 0 e 1) prejudicaria algoritmos
  sensíveis a distância/magnitude, como KNN e Regressão Logística.
- **Prevenção de data leakage:** o scaler foi ajustado (`fit_transform`)
  apenas com os dados de treino, e aplicado (`transform`, sem reajuste) aos
  dados de teste — evitando que estatísticas do conjunto de teste influenciem
  o treinamento.
- **Resultado:** após padronização, todas as features do treino apresentam
  média ≈ 0 e desvio padrão ≈ 1, confirmando a transformação correta.

### 4.4 Escolha dos algoritmos de classificação (Etapa 3)

Foram escolhidos três algoritmos de classificação, representando abordagens
distintas de Machine Learning, para enriquecer a comparação de desempenho
e a discussão crítica do relatório:

- **Regressão Logística:** modelo linear/estatístico clássico, calcula uma
  combinação linear das features e converte em probabilidade de diagnóstico
  maligno. Vantagem: alta interpretabilidade direta dos coeficientes.
  Limitação conhecida: sensível à multicolinearidade identificada na Etapa 2.
- **Random Forest:** ensemble de múltiplas Árvores de Decisão. Vantagem:
  captura relações não-lineares entre features e é robusto à
  multicolinearidade; tende a ter ótima performance "fora da caixa".
  Limitação: menor interpretabilidade direta (compensada pelo SHAP na Etapa 5).
- **KNN (K-Nearest Neighbors, k=5):** classifica cada amostra pela maioria
  entre seus 5 vizinhos mais próximos no espaço de features. Vantagem:
  conceito intuitivo de explicar. Depende da padronização já realizada
  (Etapa 2), pois é sensível à escala das variáveis.

Implementação centralizada em `src/models.py`, com uma função `get_model()`
que retorna a instância do modelo solicitado (lança `ValueError` para nomes
inválidos) e `train_model()` que treina diretamente com os dados de treino
fornecidos, usando `get_model()` internamente.

## 5. Etapa 4 — Avaliação dos modelos

### 5.1 Avaliação em split único (treino/teste 80/20)

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 |
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 |
| KNN | 0.956 | 0.974 | 0.905 | 0.938 |

Os três modelos apresentaram desempenho elevado. No split único, Random
Forest se destacou com Precision de 100% (nenhum falso positivo no conjunto
de teste) e a maior Accuracy/F1, empatando com Logistic Regression em Recall.

**Ressalva metodológica:** um Precision de 100% em um conjunto de teste de
apenas 114 amostras é um resultado a ser interpretado com cautela — pode
refletir generalização real ou favorecimento parcial pela composição
específica desse split. Isso motivou a realização de validação cruzada
(5.2) antes de eleger um modelo como superior.

### 5.2 Validação cruzada (5-fold Stratified Cross-Validation)

Para obter uma estimativa mais robusta de generalização, foi aplicada
validação cruzada estratificada com 5 folds (`StratifiedKFold`,
`shuffle=True`, `random_state=42`), com métrica `recall`. O pré-processamento
(padronização) foi encapsulado em um `Pipeline` do scikit-learn, garantindo
que o `StandardScaler` seja reajustado a cada fold — evitando data leakage
entre folds.

| Modelo | Recall médio (5-fold) | Desvio padrão |
|---|---|---|
| **Logistic Regression** | **0.944** | 0.052 |
| Random Forest | 0.934 | 0.054 |
| KNN | 0.925 | 0.046 |

**Discussão:** a validação cruzada reverteu parcialmente o quadro observado
no split único. A Regressão Logística apresentou o maior recall médio
(94.4%), superando o Random Forest (93.4%), o que confirma a suspeita de que
o resultado "perfeito" deste último no split único (Precision = 100%) era
parcialmente favorecido pela composição específica daquele split, não um
indicador definitivo de superioridade. Os desvios padrão são comparáveis
entre os três modelos (0.046–0.054), sem destaque de instabilidade. O KNN
apresentou o menor desvio padrão (mais consistente entre folds), porém com a
menor média de recall.

**Conclusão metodológica:** este resultado ilustra a importância de não
confiar em métricas de um único split de treino/teste para eleger um modelo
— validação cruzada oferece uma estimativa mais confiável do desempenho
esperado em dados novos.

### 5.3 Matriz de confusão (split único, 114 amostras de teste: 72 Benignos, 42 Malignos)

| Modelo | Falsos Negativos (Maligno → Benigno) | Falsos Positivos (Benigno → Maligno) |
|---|---|---|
| Logistic Regression | 3 | 1 |
| Random Forest | 3 | 0 |
| KNN | 4 | 1 |

**Discussão:** os números absolutos confirmam os recalls observados — 3
falsos negativos equivalem a 39/42 (92.9%) para Logistic Regression e Random
Forest, e 4 falsos negativos equivalem a 38/42 (90.5%) para o KNN.

Em termos práticos, cada falso negativo representa um caso em que um
paciente com câncer maligno real receberia, do modelo isoladamente, um
resultado classificado como benigno. Isso reforça concretamente por que o
modelo deve atuar como ferramenta de apoio à decisão clínica, nunca como
substituto do julgamento médico — especialmente considerando que o modelo
não tem acesso a sintomas clínicos, histórico familiar ou outros fatores de
risco que o(a) médico(a) avalia em conjunto.

O Random Forest apresentou zero falsos positivos neste split específico,
mas — à luz da validação cruzada (5.2), que mostrou recall médio inferior ao
da Regressão Logística — esse resultado é interpretado como característica
da composição deste split, não como propriedade estável e generalizável do
modelo.

### 5.4 Decisão sobre o modelo a interpretar com SHAP

Com base na validação cruzada (5.2), que indicou a Regressão Logística como
o modelo mais robusto (maior recall médio, com desvio padrão comparável aos
demais), optou-se por concentrar a análise de interpretabilidade via SHAP
(Etapa 6) apenas neste modelo, em vez de replicar a análise para os três
algoritmos.

## 6. Próximos passos

- [x] Etapa 1: EDA
- [x] Etapa 2: análise de correlação + split treino/teste + padronização
- [x] Etapa 3: escolha dos algoritmos de classificação (Regressão
  Logística, Random Forest, KNN)
- [x] Etapa 4: treinamento, métricas e justificativa da métrica principal
- [x] Etapa 5: feature importance e SHAP (Logistic Regression + Random
  Forest, com investigação de divergência por multicolinearidade)
- [x] Decisão final sobre o extra de CNN (decidido: não realizar — ver
  seção 8 para justificativa)
- [ ] Etapa 7: README, Dockerfile (se aplicável), relatório final
- [ ] Etapa 8: vídeo de demonstração

## 7. Etapa 5 — Interpretabilidade (SHAP)

### 7.1 Primeira tentativa e divergência identificada

A análise SHAP foi inicialmente realizada apenas para a Regressão Logística
(modelo escolhido como mais robusto na validação cruzada — ver 5.2), usando
`shap.LinearExplainer`. O ranking de importância resultante apontou
`texture3` e `radius2` como as duas features mais influentes no modelo —
resultado inesperado, já que `radius2` havia sido identificada na Etapa 2
como tendo correlação próxima de zero com o diagnóstico, e `texture3` nunca
apareceu entre as features mais correlacionadas.

Essa divergência foi tratada como um sinal de alerta a ser investigado, não
como um achado a aceitar diretamente.

### 7.2 Investigação da causa raiz

Três verificações confirmaram a hipótese de multicolinearidade distorcendo
os coeficientes da Regressão Logística:

1. **Coeficientes do modelo:** `texture3` (1.43) e `radius2` (1.23) são, de
   fato, os dois coeficientes de maior magnitude absoluta do modelo —
   confirmando que o SHAP estava descrevendo corretamente o comportamento
   do modelo, e o problema estava no treinamento, não na ferramenta.
2. **Correlação entre variantes da mesma característica:**
   `radius1` vs `radius3`: 0.97; `radius1` vs `radius2`: 0.68;
   `texture1` vs `texture3`: 0.91 — confirmando alta multicolinearidade
   exatamente nas features que "herdaram" peso desproporcional.
3. **Comparação com SHAP do Random Forest** (`TreeExplainer`, robusto a
   multicolinearidade por natureza — decide por divisões binárias, não por
   combinação linear de pesos): o ranking do Random Forest elegeu como
   top-5 `area3`, `concave_points3`, `radius3`, `concave_points1` e
   `perimeter3` — exatamente as features já identificadas como mais
   correlacionadas com o diagnóstico na Etapa 2. Já `texture3` e `radius2`
   caíram para as posições 13ª e 14ª no ranking do Random Forest.

**Nota técnica:** a subamostragem padrão do `LinearExplainer`
(`max_samples=100` de 455 disponíveis) foi corrigida para usar o conjunto de
treino completo, via `shap.maskers.Independent(X_train.values, max_samples=455)`,
eliminando variabilidade adicional do cálculo.

### 7.3 Conclusão sobre interpretabilidade

A divergência confirma uma limitação prevista desde a Etapa 2: a
multicolinearidade entre features afeta a estabilidade dos coeficientes da
Regressão Logística, fazendo com que features correlacionadas (mas não
necessariamente as mais preditivas) recebam peso desproporcional.

**Decisão adotada:** a Regressão Logística é mantida como modelo preditivo
principal do projeto, por apresentar o maior recall médio em validação
cruzada (Etapa 5.2) — multicolinearidade não compromete a capacidade
preditiva do modelo, apenas a interpretabilidade direta de seus coeficientes
individuais. Para fins de interpretação clínica de quais características
mais influenciam o diagnóstico, adota-se o ranking de importância do
**Random Forest** como referência — por ser robusto à multicolinearidade e
concordar com a análise de correlação simples (Etapa 2) e com os padrões
visuais identificados na EDA (Etapa 1).

**Conclusão clínica (Random Forest + correlação, convergentes):** tamanho
(`radius`, `area`, `perimeter`) e irregularidade de contorno (`concavity`,
`concave_points`) do núcleo celular, especialmente nas medições "pior caso"
(sufixo 3), são os indicadores mais fortes de malignidade no dataset.

**Nota metodológica geral:** este episódio ilustra por que validar resultados
de interpretabilidade contra múltiplas fontes de evidência (correlação, EDA
visual, e mais de um modelo) é importante — um único método de
interpretabilidade, aplicado a um modelo sensível a uma limitação conhecida
dos dados, pode levar a conclusões equivocadas se aceito sem verificação.

## 8. Decisão final sobre o extra de Visão Computacional (CNN)

**Decisão:** não realizar o extra de CNN com mamografias (CBIS-DDSM).

**Justificativa:**

1. O enunciado caracteriza o extra como compensatório, não cumulativo:
   "não é obrigatório, mas pode aumentar sua nota caso não atinja a
   pontuação máxima" — destinado a recuperar pontuação eventualmente perdida
   na parte obrigatória, não a somar acima de uma entrega já completa.
2. A análise tabular obrigatória foi concluída com profundidade acima do
   mínimo esperado: EDA completa, pré-processamento com prevenção de data
   leakage, três modelos comparados, validação cruzada, matriz de confusão,
   e uma investigação crítica da causa raiz de uma divergência na análise
   SHAP (multicolinearidade), incluindo comparação entre dois explicadores.
3. Restrições práticas de ambiente (treinamento em CPU, sem GPU disponível)
   tornariam o ciclo de desenvolvimento da CNN lento e arriscado dentro do
   prazo remanescente, com alto risco de uma entrega rasa ou inacabada
   comprometer a percepção de qualidade do projeto como um todo.

O tempo que seria investido neste extra foi redirecionado para a finalização
cuidadosa do relatório técnico e do vídeo de demonstração.
