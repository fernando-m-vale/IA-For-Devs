"""
Módulo de função fitness para o Algoritmo Genético.

A função fitness avalia um conjunto de hiperparâmetros treinando o modelo
correspondente com validação cruzada estratificada (5-fold) e retornando
o recall médio — métrica prioritária para o diagnóstico de câncer de mama,
onde falsos negativos têm custo clínico mais alto.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# Espaço de busca de hiperparâmetros para cada modelo.
# Cada entrada define os genes possíveis do cromossomo daquele modelo.
SEARCH_SPACE = {
    "logistic_regression": {
        "C":        {"type": "float",  "low": 0.001, "high": 100.0, "log": True},
        "solver":   {"type": "choice", "options": ["lbfgs", "liblinear"]},
        "max_iter": {"type": "int",    "low": 100,   "high": 2000},
    },
    "random_forest": {
        "n_estimators":      {"type": "int", "low": 10,  "high": 300},
        "max_depth":         {"type": "int", "low": 3,   "high": 20},
        "min_samples_split": {"type": "int", "low": 2,   "high": 10},
        "min_samples_leaf":  {"type": "int", "low": 1,   "high": 10},
    },
    "knn": {
        "n_neighbors": {"type": "int",    "low": 1,  "high": 30},
        "weights":     {"type": "choice", "options": ["uniform", "distance"]},
        "metric":      {"type": "choice", "options": ["euclidean", "manhattan"]},
    },
    "svm": {
        "C":      {"type": "float",  "low": 0.01, "high": 100.0, "log": True},
        "kernel": {"type": "choice", "options": ["rbf", "poly", "sigmoid"]},
        "gamma":  {"type": "choice", "options": ["scale", "auto"]},
    },
}


def sample_individual(model_name: str, rng: np.random.Generator) -> dict:
    """
    Gera um indivíduo aleatório (cromossomo) para o modelo especificado,
    amostrando cada gene dentro do espaço de busca definido em SEARCH_SPACE.

    Parameters
    ----------
    model_name : str
        Nome do modelo — chave em SEARCH_SPACE.
    rng : np.random.Generator
        Gerador de números aleatórios para reprodutibilidade.

    Returns
    -------
    dict com um valor para cada hiperparâmetro do modelo.
    """
    space = SEARCH_SPACE[model_name]
    individual = {}
    for param, config in space.items():
        if config["type"] == "int":
            individual[param] = int(rng.integers(config["low"], config["high"] + 1))
        elif config["type"] == "float":
            if config.get("log", False):
                log_low = np.log10(config["low"])
                log_high = np.log10(config["high"])
                individual[param] = float(10 ** rng.uniform(log_low, log_high))
            else:
                individual[param] = float(rng.uniform(config["low"], config["high"]))
        elif config["type"] == "choice":
            individual[param] = config["options"][
                int(rng.integers(0, len(config["options"])))
            ]
    return individual


def build_model(model_name: str, hyperparams: dict):
    """
    Instancia o modelo scikit-learn com os hiperparâmetros fornecidos.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=42, **hyperparams)
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=42, **hyperparams)
    elif model_name == "knn":
        return KNeighborsClassifier(**hyperparams)
    elif model_name == "svm":
        return SVC(random_state=42, probability=True, **hyperparams)
    else:
        raise ValueError(f"Modelo '{model_name}' não reconhecido.")


def evaluate_fitness(
    model_name: str,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    metric: str = "recall",
) -> float:
    """
    Avalia o fitness de um indivíduo treinando o modelo com os hiperparâmetros
    fornecidos via validação cruzada estratificada.

    O modelo é sempre embrulhado em um Pipeline com StandardScaler para
    evitar data leakage entre folds — a padronização é refeita a cada fold,
    usando apenas os dados de treino daquele fold.

    Parameters
    ----------
    model_name : str
        Nome do modelo a avaliar.
    hyperparams : dict
        Hiperparâmetros do indivíduo (cromossomo).
    X : np.ndarray
        Features do dataset completo (não pré-escalado).
    y : np.ndarray
        Target binário (1 = Maligno, 0 = Benigno).
    cv_folds : int
        Número de folds da validação cruzada.
    metric : str
        Métrica de avaliação — padrão "recall".

    Returns
    -------
    float: recall médio nos cv_folds folds. Retorna 0.0 em caso de erro
    (combinação de hiperparâmetros inválida para o solver/kernel escolhido).
    """
    try:
        model = build_model(model_name, hyperparams)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
        return float(scores.mean())
    except Exception:
        # Combinações inválidas (ex: solver incompatível com penalty)
        # recebem fitness zero para serem naturalmente eliminadas pelo AG.
        return 0.0
