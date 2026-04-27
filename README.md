# Poker Win Probability Estimator

A supervised machine learning system that estimates **Texas Hold'em win probability** using Monte Carlo simulation as training labels.

Instead of running expensive simulations at inference time, trained models provide fast predictions based on encoded card features — achieving speedups of **10,000×** or more over MC simulation.

---

## Project Overview

Texas Hold'em is an imperfect information game. Given a player's hole cards and the public board cards, this project formulates win probability estimation as a **regression problem**.

A dataset of 50,000 labeled game states was generated via Monte Carlo simulation, and four regression models were trained and compared.

**Course**: EECE5644 — Introduction to Machine Learning and Pattern Recognition
**Institution**: Northeastern University, Spring 2026

---

## Project Structure

| File | Description |
|------|-------------|
| `mc_simulation.py` | Monte Carlo win probability estimator (1,000 simulations per state) |
| `feature_encoding.py` | 135-dimensional feature vector encoder |
| `dataset_generator.py` | Synthetic dataset generator (50,000 game states) |
| `linear_regression.py` | Baseline linear regression model |
| `results_logger.py` | Auto-saves model metrics to `results.csv` |

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

Each game state is encoded into a 135-dimensional feature vector:

| Component | Dims | Description |
|-----------|------|-------------|
| Hole cards (One-Hot) | 34 | 2 cards × 17 (13 rank + 4 suit) |
| Board cards (One-Hot) | 85 | 5 cards × 17, zero-padded for earlier stages |
| Handcrafted features | 15 | Flush draw, straight draw, pair, connectivity, ace presence, etc. |
| Num opponents | 1 | Normalized to [0, 1] |
| **Total** | **135** | |

Game stages covered: **Preflop, Flop, Turn, River**. Opponents sampled uniformly from 1–8.

---

## Dataset Statistics

Labels (win probability) were generated using 1,000 Monte Carlo simulations per state:

| Statistic | Value |
|-----------|-------|
| Count | 50,000 |
| Mean | 0.217 |
| Std | 0.226 |
| Min | 0.000 |
| 25th percentile | 0.052 |
| Median | 0.138 |
| 75th percentile | 0.311 |
| Max | 1.000 |

---

## Results

### Model Comparison (50,000 samples)

All models trained on 80% / evaluated on 20% split.

| Model | MSE | MAE | R² | Inference (ms/sample) |
|-------|-----|-----|----|-----------------------|
| Linear Regression | 0.02131 | 0.10243 | 0.5757 | 0.0001 |
| Random Forest | 0.01362 | **0.06591** | 0.7288 | 0.0039 |
| Gradient Boosting | 0.01443 | 0.07285 | 0.7127 | 0.0032 |
| **MLP** | **0.01323** | 0.06853 | **0.7367** | 0.0010 |

- **MLP** achieved the best R² (0.737) and MSE (0.01323)
- **Random Forest** achieved the best MAE (0.066)
- **Linear Regression** was fastest at inference but significantly weaker due to the non-linear nature of card interactions
- All models run in under 0.01 ms/sample — orders of magnitude faster than MC simulation (~hundreds of ms)

### Effect of Dataset Size

| Dataset | Model | MSE | MAE | R² |
|---------|-------|-----|-----|----|
| 500 | Linear Regression | 0.02793 | 0.12752 | 0.522 |
| 500 | Random Forest | 0.01658 | 0.09531 | 0.716 |
| 500 | Gradient Boosting | 0.01695 | 0.09214 | 0.710 |
| 500 | **MLP** | 0.07341 | 0.21152 | **-0.257** |
| 50,000 | Linear Regression | 0.02131 | 0.10243 | 0.576 |
| 50,000 | Random Forest | 0.01362 | 0.06591 | 0.729 |
| 50,000 | Gradient Boosting | 0.01443 | 0.07285 | 0.713 |
| 50,000 | **MLP** | 0.01323 | 0.06853 | **0.737** |

Key finding: **MLP is highly sensitive to dataset size** — with only 500 samples it performs worse than a naive mean predictor (R² = -0.257), but scales to the best overall model at 50,000 samples. Tree-based models generalize well even with limited data.

---

## Discussion

**Why not linear regression?** Win probability depends on card *combinations* (flush draws, straight draws, pairs), not individual features. These interactions are non-linear and invisible to a linear model.

**Why does R² plateau around 0.73–0.74?** Two factors: (1) MC labels computed from 1,000 simulations carry inherent noise, setting an accuracy ceiling; (2) the 135-dim feature vector may not fully capture complex multi-way card interactions.

---

## Future Work

- Increase `N_SIMULATIONS` from 1,000 → 5,000 to reduce label noise
- Add richer handcrafted features (number of outs, pot odds)
- Experiment with XGBoost or deeper MLP architectures
- Generalization experiments across different opponent counts
- Calibration analysis of predicted probabilities

---

## References

- Tantia, K., & Trieu, N. *Poker Bot: A Reinforced Learning Neural Network*. Harvey Mudd College. https://www.cs.hmc.edu/~ktantia/poker.html
- Zhang, X. C., & Li, Y. (2020). *A Texas Hold'em Decision Model Based on Reinforcement Learning*. CCDC 2020. https://doi.org/10.1109/CCDC49329.2020.9164345
