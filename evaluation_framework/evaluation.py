
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize

# =========================
# Função para avaliação de classificação
# =========================
def evaluate_classification(model_path, preprocessor_path, X_test, y_test, report_dir, model_name):
    """
    Avalia um modelo de classificação e salva métricas e gráficos.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Carregar modelo e pré-processador
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path) if preprocessor_path else None

    # Pré-processar dados
    X_test_preprocessed = preprocessor.transform(X_test) if preprocessor else X_test

    # Predições
    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)

    # Métricas
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average='macro'),
        "ROC_AUC": roc_auc_score(y_test, y_proba, multi_class='ovr')
    }

    # Salvar métricas
    pd.DataFrame([metrics]).to_csv(os.path.join(report_dir, f"{model_name}_metrics.csv"), index=False)

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(os.path.join(report_dir, f"{model_name}_confusion.png"))
    plt.close()

    # Curva ROC
    classes = np.unique(y_test)
    y_true_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {cls} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(report_dir, f"{model_name}_roc.png"))
    plt.close()

    print(f"✅ Avaliação concluída para {model_name}. Métricas salvas em {report_dir}")
    return metrics

# =========================
# Função para avaliação de regressão
# =========================
def evaluate_regression(model_path, scaler_path, X_test, y_test, report_dir, model_name):
    """
    Avalia um modelo de regressão e salva métricas e gráficos.
    """
    os.makedirs(report_dir, exist_ok=True)

    # Carregar modelo e scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None

    # Pré-processar dados
    X_test_scaled = scaler.transform(X_test) if scaler else X_test

    # Predições
    y_pred = model.predict(X_test_scaled)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # Salvar métricas
    pd.DataFrame([metrics]).to_csv(os.path.join(report_dir, f"{model_name}_metrics.csv"), index=False)

    # Gráfico Pred vs Real
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Predicted vs Actual - {model_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(os.path.join(report_dir, f"{model_name}_pred_vs_actual.png"))
    plt.close()

    # Distribuição dos erros
    errors = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(errors, bins=30, kde=True)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Error')
    plt.savefig(os.path.join(report_dir, f"{model_name}_error_distribution.png"))
    plt.close()

    print(f"✅ Avaliação concluída para {model_name}. Métricas salvas em {report_dir}")
    return metrics
