"""
Módulo de avaliação: cálculo e apresentação de métricas de classificação.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name: str = "modelo") -> dict:
    """
    Calcula as principais métricas de classificação para um modelo treinado.

    Returns
    -------
    dict com accuracy, precision, recall, f1 e a matriz de confusão.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    return metrics


def evaluate_all_models(models_dict: dict, X_test, y_test) -> pd.DataFrame:
    """
    Avalia múltiplos modelos treinados e retorna um DataFrame comparativo,
    ordenado por recall (métrica priorizada neste contexto clínico).
    """
    results = [evaluate_model(model, X_test, y_test, name)
               for name, model in models_dict.items()]
    df_results = pd.DataFrame(results).set_index("modelo")
    return df_results.sort_values("recall", ascending=False)
