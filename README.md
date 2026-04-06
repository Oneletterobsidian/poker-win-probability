# Poker Win Probability Estimator

A machine learning system that estimates Texas Hold'em win probability using Monte Carlo simulation as training labels.

Instead of running expensive simulations at inference time, trained models provide fast predictions based on encoded card features.

---

## Project Structure

| File | Description |
|------|-------------|
| `mc_simulation.py` | Monte Carlo win probability estimator |
| `feature_encoding.py` | 135-dim feature vector encoder |
| `dataset_generator.py` | Synthetic dataset generator |
| `linear_regression.py` | Baseline linear regression model |
| `results_logger.py` | Auto-saves model metrics to results.csv |

---

## Quickstart

Install dependencies:
```bash
pip install treys scikit-learn pandas numpy tqdm joblib
```

Generate dataset:
```bash
python dataset_generator.py
```

Train baseline model:
```bash
python linear_regression.py
```

---

## Feature Vector (135 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| Hole cards (One-Hot) | 34 | 2 cards × 17 (13 rank + 4 suit) |
| Board cards (One-Hot) | 85 | 5 cards × 17, zero-padded |
| Handcrafted features | 15 | Flush, straight, pair, connectivity, etc. |
| Num opponents | 1 | Normalised to [0, 1] |

---

## Models Compared

| Model | MSE | MAE | R² | Inference (ms/sample) |
|-------|-----|-----|----|-----------------------|
| Linear Regression | 0.02793 | 0.12752 | 0.522 | 0.0053 |
| Random Forest | - | - | - | - |
| Gradient Boosting | - | - | - | - |
| MLP | - | - | - | - |
