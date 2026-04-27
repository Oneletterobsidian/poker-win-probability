"""
gradient_boosting.py
Gradient Boosting regression model for poker win probability estimation.
"""

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from results_logger import log_results

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "poker_dataset.csv"
MODEL_PATH  = "gradient_boosting.pkl"
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
def evaluate(model, X_test: np.ndarray, y_test: np.ndarray):
    start  = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - start

    mse  = mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    ms_per_sample = (elapsed / len(y_test)) * 1000

    print("\n── Gradient Boosting Results ───────────────────")
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

    # 2. Train
    # Note: Gradient Boosting also does not require feature scaling
    print("\nTraining Gradient Boosting (this may take a minute)...")
    model = GradientBoostingRegressor(
        n_estimators=200,       # number of boosting stages
        learning_rate=0.05,     # shrinkage — smaller = more robust, needs more trees
        max_depth=4,            # shallow trees reduce overfitting
        min_samples_leaf=2,
        subsample=0.8,          # stochastic GB: use 80% of data per tree
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    print("Done.")

    # 3. Evaluate
    metrics = evaluate(model, X_test, y_test)

    # 4. Log results
    log_results("Gradient Boosting", metrics)

    # 5. Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")

    return metrics

if __name__ == "__main__":
    main()
