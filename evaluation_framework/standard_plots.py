# 3) Generate standard plots (residuals/ROC/PR, confusion matrix)
import os
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)




# =========================
# Funções para plots padrão
# =========================
def plot_roc_curve(y_true, y_proba, model_name, report_dir):
    """
    Plota a curva ROC para modelos de classificação.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid()

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    plt.savefig(os.path.join(report_dir, f'roc_curve_{model_name}.png'))
    plt.close()




def plot_confusion_matrix(y_true, y_pred, model_name, report_dir):
    """
    Plota a matriz de confusão para modelos de classificação.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    plt.savefig(os.path.join(report_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()




def plot_residuals(y_true, y_pred, model_name, report_dir):
    """
    Plota os resíduos para modelos de regressão.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='r', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals Plot - {model_name}')
    plt.grid()

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    plt.savefig(os.path.join(report_dir, f'residuals_plot_{model_name}.png'))
    plt.close()




def plot_precision_recall_curve(y_true, y_proba, model_name, report_dir):
    """ Plota a curva Precision-Recall para modelos de classificação.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid()

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    plt.savefig(os.path.join(report_dir, f'precision_recall_curve_{model_name}.png'))
    plt.close()


def plot_all_standard_plots(model_type, y_true, y_pred, y_proba, model_name, report_dir):
    """
    Gera todos os plots padrão com base no tipo de modelo.
    """
    if model_type == 'classification':
        plot_roc_curve(y_true, y_proba, model_name, report_dir)
        plot_confusion_matrix(y_true, y_pred, model_name, report_dir)
        plot_precision_recall_curve(y_true, y_proba, model_name, report_dir)
    elif model_type == 'regression':
        plot_residuals(y_true, y_pred, model_name, report_dir)
    else:
        raise ValueError("model_type deve ser 'classification' ou 'regression'.")
    