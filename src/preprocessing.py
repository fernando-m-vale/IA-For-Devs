"""
Módulo de pré-processamento: separação treino/teste e padronização,
reutilizável entre notebooks.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(df: pd.DataFrame, target_col: str = "diagnosis",
                     test_size: float = 0.2, random_state: int = 42):
    """
    Separa o DataFrame em treino/teste (estratificado) e aplica padronização
    (StandardScaler), ajustando o scaler apenas com os dados de treino para
    evitar data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset completo, incluindo a coluna alvo.
    target_col : str
        Nome da coluna alvo. Espera-se valores 'M' (maligno) e 'B' (benigno).
    test_size : float
        Proporção dos dados destinada ao conjunto de teste.
    random_state : int
        Semente de aleatoriedade, para reprodutibilidade.

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
        Features padronizadas, com nomes de colunas preservados.
    y_train, y_test : pd.Series
        Target binário (1 = Maligno, 0 = Benigno).
    scaler : StandardScaler
        Scaler já ajustado, para uso posterior (ex: SHAP, novos dados).
    """
    X = df.drop(columns=[target_col])
    y = (df[target_col] == "M").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
