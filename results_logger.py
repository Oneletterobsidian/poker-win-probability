"""
results_logger.py
Saves model evaluation metrics to results.csv automatically.
Import this in every model script.
"""

import pandas as pd
import os
from datetime import datetime

RESULTS_PATH = "results.csv"

def log_results(model_name: str, metrics: dict):
    """
    Append model results to results.csv.

    Args:
        model_name: e.g. "Linear Regression"
        metrics:    dict with keys: mse, mae, r2, ms_per_sample
    """
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model":          model_name,
        "mse":            round(metrics["mse"], 5),
        "mae":            round(metrics["mae"], 5),
        "r2":             round(metrics["r2"], 4),
        "ms_per_sample":  round(metrics["ms_per_sample"], 4),
    }

    # Append if file exists, create if not
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults logged to '{RESULTS_PATH}'")
    print(df.to_string(index=False))