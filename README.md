# IA-For-Devs
Respositório para desenvolvimento dos projetos da Pós Graduação IA para Devs da FIAP

---

## Tech Challenge — Fase 2: Diagnóstico de Câncer de Mama

Classificação de diagnóstico (Maligno / Benigno) usando o dataset **Breast Cancer Wisconsin** da UCI.

### Estrutura do projeto

```
IA-For-Devs/
├── data/
│   ├── raw/            # Dataset original baixado via ucimlrepo (não versionado)
│   └── processed/      # Features processadas prontas para modelagem
├── notebooks/
│   ├── 01_eda.ipynb              # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb    # Limpeza e normalização
│   ├── 03_modeling.ipynb         # Treinamento e seleção de modelos
│   └── 04_evaluation_shap.ipynb  # Métricas finais e explicabilidade com SHAP
├── src/
│   ├── data_loader.py    # Download e leitura do dataset
│   ├── preprocessing.py  # Pipeline de pré-processamento
│   ├── models.py         # Definição e treino dos modelos
│   └── evaluate.py       # Métricas e relatórios
├── reports/figures/  # Gráficos exportados
├── outputs/          # Modelos serializados e artefatos de saída
└── requirements.txt
```

### Instalação

```bash
pip install -r requirements.txt
```

### Dataset

Baixado automaticamente via `ucimlrepo` (ID 17 — Breast Cancer Wisconsin Diagnostic).
