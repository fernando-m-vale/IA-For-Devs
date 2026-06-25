"""
Módulo de modelagem: define os classificadores utilizados no projeto e uma
função utilitária para treiná-los de forma padronizada.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_model(model_name: str, random_state: int = 42):
    """
    Retorna uma instância não treinada do modelo solicitado.

    Parameters
    ----------
    model_name : str
        Um de: 'logistic_regression', 'random_forest', 'knn'.
    random_state : int
        Semente de aleatoriedade, para reprodutibilidade (ignorada pelo KNN,
        que não possui componente aleatório).

    Returns
    -------
    Instância do classificador do scikit-learn, ainda não treinada.
    """
    models = {
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }

    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' não reconhecido. Opções: {list(models.keys())}"
        )

    return models[model_name]


def train_model(model_name: str, X_train, y_train, random_state: int = 42):
    """
    Instancia e treina um modelo com os dados de treino fornecidos.

    Returns
    -------
    Modelo treinado (fitted).
    """
    model = get_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)
    return model
