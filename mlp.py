"""
mlp.py
Multi-Layer Perceptron regression model for poker win probability estimation.
"""

import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from results_logger import log_results

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "poker_dataset.csv"
MODEL_PATH  = "mlp.pkl"
TEST_SIZE   = 0.2
RANDOM_SEED = 42

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(path: str):
    df = pd.read_csv(path)
    X  = df.drop(columns=["win_probability"]).values
    y  = df["win_probability"].values
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

# ── Evaluate model ────────────────────────────────────────────────────────────
def evaluate(model, scaler, X_test: np.ndarray, y_test: np.ndarray):
    X_scaled = scaler.transform(X_test)

    start  = time.perf_counter()
    y_pred = model.predict(X_scaled)
    elapsed = time.perf_counter() - start

    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    ms_per_sample = (elapsed / len(y_test)) * 1000

    print("\n── MLP Results ─────────────────────────────────")
    print(f"  MSE              : {mse:.5f}")
    print(f"  MAE              : {mae:.5f}")
    print(f"  R²               : {r2:.4f}")
    print(f"  Inference time   : {ms_per_sample:.4f} ms/sample")
    print("────────────────────────────────────────────────")

    return {"mse": mse, "mae": mae, "r2": r2, "ms_per_sample": ms_per_sample}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load & split
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # 2. Scale features — MLP is sensitive to feature scale like Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. Train
    print("\nTraining MLP (this may take a minute)...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
        activation='relu',                  # ReLU activation
        solver='adam',                      # adaptive optimizer
        learning_rate_init=0.001,
        max_iter=500,                       # max training epochs
        early_stopping=True,               # stop if validation loss plateaus
        validation_fraction=0.1,
        random_state=RANDOM_SEED
    )
    model.fit(X_train_scaled, y_train)
    print(f"Done. (trained for {model.n_iter_} epochs)")

    # 4. Evaluate
    metrics = evaluate(model, scaler, X_test, y_test)

    # 5. Log results
    log_results("MLP", metrics)

    # 6. Save model + scaler
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")

    return metrics

if __name__ == "__main__":
    main()
