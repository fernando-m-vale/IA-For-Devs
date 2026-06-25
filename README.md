# Diagnóstico de Câncer de Mama com Machine Learning

> Tech Challenge — Fase 1 (IA Foundations) | Pós-Tech IA para Devs — FIAP

## Projeto

Este projeto foi desenvolvido como Tech Challenge da Fase 1 do curso de
Pós-Graduação em IA para Devs (FIAP), referente ao Desafio B: uma rede
de hospitais e centros de saúde especializados em saúde da mulher busca
implementar um sistema inteligente de suporte ao diagnóstico, capaz de
ajudar profissionais de saúde na identificação precoce de condições que
afetam a segurança e a saúde feminina.

O projeto constrói uma solução de Machine Learning para classificação de
diagnóstico de câncer de mama (maligno ou benigno) a partir de dados
clínicos tabulares, aplicando fundamentos de análise exploratória,
pré-processamento, modelagem preditiva, avaliação crítica e
interpretabilidade (SHAP).

## Principais Resultados

| Item | Resultado |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository (id=17) |
| Modelos avaliados | Regressão Logística, Random Forest, KNN (k=5) |
| Modelo selecionado | Regressão Logística (maior recall médio em validação cruzada) |
| Recall médio (5-fold CV) | 94.4% (vs. 93.4% Random Forest, 92.5% KNN) |
| Métrica priorizada | Recall — minimizar falsos negativos (casos malignos não detectados) |
| Interpretabilidade | SHAP (LinearExplainer + TreeExplainer), com investigação de uma
divergência causada por multicolinearidade entre features |

Mais detalhes em [`reports/relatorio_tecnico.md`](reports/relatorio_tecnico.md)
e no histórico completo de decisões em
[`reports/diario_de_bordo.md`](reports/diario_de_bordo.md).

## Estrutura do repositório

```
IA-For-Devs/
├── data/
│   ├── raw/                       # dataset original (gerado automaticamente)
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb                # análise exploratória de dados
│   ├── 02_preprocessing.ipynb       # correlação, split treino/teste, padronização
│   ├── 03_modeling.ipynb            # treinamento dos 3 modelos
│   └── 04_evaluation_shap.ipynb     # métricas, validação cruzada, SHAP
├── src/
│   ├── data_loader.py               # carregamento do dataset (UCI via ucimlrepo)
│   ├── preprocessing.py             # split + padronização reutilizáveis
│   ├── models.py                    # definição e treino dos classificadores
│   └── evaluate.py                  # cálculo de métricas de avaliação
├── outputs/
│   └── models/                      # modelos treinados serializados (.pkl)
├── reports/
│   ├── diario_de_bordo.md           # registro cronológico de decisões e achados
│   └── relatorio_tecnico.md         # relatório técnico consolidado (entregável)
├── requirements.txt
└── README.md
```

## Como executar

### Pré-requisitos
- Python 3.10+
- pip

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/fernando-m-vale/IA-For-Devs.git
cd IA-For-Devs

# 2. Criar e ativar um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Executar os notebooks, na ordem, com Jupyter
jupyter notebook notebooks/
```

Execute os notebooks na seguinte ordem, pois cada um depende de artefatos
gerados pelo anterior:

1. `01_eda.ipynb` — exploração inicial dos dados
2. `02_preprocessing.ipynb` — correlação e preparação dos dados
3. `03_modeling.ipynb` — treinamento dos modelos (salva os modelos em
   `outputs/models/`)
4. `04_evaluation_shap.ipynb` — avaliação final e interpretabilidade

O dataset é obtido automaticamente via API do UCI Machine Learning
Repository na primeira execução do `01_eda.ipynb` — não é necessário
download manual.

## Dataset

- Nome: Breast Cancer Wisconsin (Diagnostic)
- Fonte: [UCI Machine Learning Repository, id=17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Acesso: biblioteca `ucimlrepo`, encapsulada em `src/data_loader.py`
- Tamanho: 569 amostras, 30 features numéricas + diagnóstico (M/B)
- Características: medidas extraídas de biópsias por aspiração com
  agulha fina (FNA), descrevendo propriedades do núcleo celular (raio,
  textura, perímetro, área, suavidade, compacidade, concavidade, pontos de
  concavidade, simetria, dimensão fractal), cada uma em três variantes
  (média, erro padrão, "pior caso").

## Metodologia resumida

1. EDA: balanceamento de classes (62.7% Benigno / 37.3% Maligno),
   distribuições por diagnóstico, identificação de features mais
   discriminativas visualmente.
2. Pré-processamento: análise de correlação (identificação de
   multicolinearidade esperada entre variantes da mesma característica),
   separação treino/teste estratificada (80/20), padronização
   (`StandardScaler`) com prevenção de data leakage.
3. Modelagem: três classificadores (Regressão Logística, Random Forest,
   KNN), avaliados por accuracy, precision, recall e F1-score.
4. Validação: validação cruzada estratificada (5-fold) para confirmar
   a robustez dos resultados além de um único split de treino/teste.
5. Interpretabilidade: SHAP aplicado à Regressão Logística e ao Random
   Forest, com investigação da causa de uma divergência entre os dois
   modelos (multicolinearidade distorcendo coeficientes lineares).

Discussão completa, incluindo justificativas de cada decisão técnica, no
[relatório técnico](reports/relatorio_tecnico.md).

## Limitações e considerações éticas

- O modelo deve ser utilizado exclusivamente como ferramenta de apoio
  à decisão clínica — o(a) médico(a) deve manter sempre a palavra final no
  diagnóstico, considerando informações (sintomas, histórico familiar,
  outros exames) que o modelo não tem acesso.
- A Regressão Logística apresenta multicolinearidade entre features, o que
  compromete a confiabilidade de seus coeficientes individuais como medida
  de importância — por isso, a interpretação de quais características mais
  influenciam o diagnóstico utiliza o ranking de importância do Random
  Forest como referência, validado contra a análise de correlação.
- O dataset utilizado, apesar de amplamente validado na literatura
  acadêmica, tem origem em um único centro de pesquisa, o que pode limitar
  a generalização para populações com características demográficas
  distintas.

## Vídeo de demonstração

🔗 *[link a ser adicionado após a gravação]*

## Autor

Fernando Marques do Vale, rm375623
Pós-Graduação em IA para Devs — FIAP
