# data_loader.py — carrega o dataset Breast Cancer Wisconsin via ucimlrepo e salva em data/raw/

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_breast_cancer_data(
    save_path: str | Path = Path(__file__).resolve().parents[1] / "data" / "raw" / "breast_cancer_wisconsin.csv",
) -> pd.DataFrame:
    """Baixa o dataset Breast Cancer Wisconsin (Diagnostic) do UCI ML Repository.

    Fonte: UCI Machine Learning Repository, dataset id=17.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    O target 'diagnosis' assume dois valores:
        M — Maligno
        B — Benigno

    Args:
        save_path: Caminho onde o CSV combinado (features + target) será salvo.
                   Por padrão, grava em data/raw/breast_cancer_wisconsin.csv
                   relativo à raiz do projeto.

    Returns:
        DataFrame com as 30 features numéricas e a coluna 'diagnosis'.
    """
    dataset = fetch_ucirepo(id=17)

    X: pd.DataFrame = dataset.data.features
    y: pd.DataFrame = dataset.data.targets

    df = X.copy()
    df["diagnosis"] = y.values

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    return df
