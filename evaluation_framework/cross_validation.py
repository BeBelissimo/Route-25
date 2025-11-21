import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import is_classifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)


# =========================
# Função para cross-validation
# =========================

def cross_validate_model(model, X, y, n_splits=5, metrics=["rmse", "mae", "r2"], report_dir=None, model_name="model"):
    """
    Executa k-fold cross-validation e calcula métricas.
    """

    # Converter para NumPy se necessário
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        # Ajuste para classes começarem em 0
        y_train_adj = y_train - 1
        y_test_adj = y_test - 1


        model.fit(X_train, y_train_adj)
        y_pred = model.predict(X_test)
    
        if is_classifier(model):

            classes = np.unique(y_test_adj)
            class_mapping = {cls: idx for idx, cls in enumerate(classes)}
            y_test_mapped = np.array([class_mapping[c] for c in y_test_adj])
            y_pred_mapped = np.array([class_mapping[c] for c in y_pred])

            
        # Ajustar predict_proba para colunas na ordem do mapeamento
            y_proba = model.predict_proba(X_test)
            # Se necessário, reordenar colunas
            if y_proba.shape[1] != len(classes):
                raise ValueError("Número de colunas em predict_proba não corresponde ao número de classes.")


            fold_metrics = {
                "Fold": fold + 1,
                "Accuracy": accuracy_score(y_test_mapped, y_pred_mapped),
                "F1": f1_score(y_test_mapped, y_pred_mapped, average='macro'),
                "ROC_AUC": roc_auc_score(
                    y_test_mapped,
                    y_proba,
                    multi_class='ovr'
                )
            }
        else:
            fold_metrics = {
                "Fold": fold + 1,
                "MAE": mean_absolute_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R2": r2_score(y_test, y_pred)
            }
        results.append(fold_metrics)

    df_results = pd.DataFrame(results)
    df_results.loc["Mean"] = df_results.mean(numeric_only=True)

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        df_results.to_csv(os.path.join(report_dir, f"{model_name}_crossval.csv"), index=False)

    print(f"✅ Cross-validation concluída para {model_name}.")
    return df_results
