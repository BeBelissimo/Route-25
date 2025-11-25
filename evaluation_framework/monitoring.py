
import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from datetime import datetime

# =========================
# Drift Checks (PSI & KS)
# =========================
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()), buckets+1)
    expected_counts = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Evita divisão por zero
    expected_counts = np.where(expected_counts == 0, 1e-6, expected_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    # Ignora bins com zero
    mask = (expected_counts > 0) & (actual_counts > 0)
    if not np.any(mask):
        return 0.0

    psi = np.sum((expected_counts[mask] - actual_counts[mask]) * np.log(expected_counts[mask] / actual_counts[mask]))
    return psi

def calculate_ks(expected, actual):
    ks_stat, p_value = ks_2samp(expected, actual)
    return ks_stat, p_value

def generate_drift_report(reference_df, current_df, report_dir, model_name):
    os.makedirs(report_dir, exist_ok=True)
    results = []
    for col in reference_df.columns:
        psi = calculate_psi(reference_df[col], current_df[col])
        ks_stat, p_value = calculate_ks(reference_df[col], current_df[col])
        results.append({
            'feature': col,
            'PSI': round(psi, 4),
            'KS_stat': round(ks_stat, 4),
            'KS_p_value': round(p_value, 4),
            'drift_flag': psi > 0.25 or ks_stat > 0.1
        })
    df_report = pd.DataFrame(results)
    csv_path = os.path.join(report_dir, f"drift_report_{model_name}.csv")
    html_path = os.path.join(report_dir, f"drift_report_{model_name}.html")
    df_report.to_csv(csv_path, index=False)
    df_report.to_html(html_path)
    return df_report

# =========================
# Tracking Experiments
# =========================
def track_experiment_csv(report_dir, model_name, metrics):
    os.makedirs(report_dir, exist_ok=True)
    csv_path = os.path.join(report_dir, "experiment_tracking.csv")

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        **metrics
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(csv_path, index=False)
    return csv_path

# MLflow opcional
try:
    import mlflow
    def track_experiment_mlflow(model_name, metrics, params=None):
        mlflow.start_run(run_name=model_name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)
        mlflow.end_run()
except ImportError:
    def track_experiment_mlflow(*args, **kwargs):
        print("⚠ MLflow não está instalado. Tracking via MLflow ignorado.")

# =========================
# Alert Thresholds
# =========================
def check_alerts(baseline_metrics, current_metrics, threshold=0.10):
    alerts = {}
    for metric, baseline_value in baseline_metrics.items():
        if metric in current_metrics:
            drop = (baseline_value - current_metrics[metric]) / baseline_value
            alerts[metric] = drop > threshold
    return alerts
