"""
dataset_generator.py
Combines MC simulation + feature encoding to generate a labeled dataset.
Output: CSV with 135 feature columns + 1 label column (win_probability)
"""

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from treys import Card, Deck, Evaluator

from feature_encoding import build_feature_vector
from mc_simulation import estimate_win_probability

# ── Config ────────────────────────────────────────────────────────────────────
N_SAMPLES      = 500   # total game states to generate
N_SIMULATIONS  = 200    # MC samples per state (higher = less label noise)
MAX_OPPONENTS  = 8        # randomly sample 1–8 opponents per state
OUTPUT_PATH    = "poker_dataset.csv"
RANDOM_SEED    = 42

STAGES = {
    'preflop': 0,
    'flop':    3,
    'turn':    4,
    'river':   5,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Random game state sampler ─────────────────────────────────────────────────
def sample_game_state() -> tuple:
    """
    Randomly deal a valid game state.
    Returns (hole_cards, board_cards, num_opponents, stage_name)
    """
    deck = Deck()
    deck.shuffle()

    # Deal 2 hole cards
    hole_cards = deck.draw(2)

    # Randomly pick a game stage
    stage = random.choice(list(STAGES.keys()))
    n_board = STAGES[stage]

    # Deal board cards
    board_cards = deck.draw(n_board) if n_board > 0 else []

    # Random number of opponents
    num_opponents = random.randint(1, MAX_OPPONENTS)

    return hole_cards, board_cards, num_opponents, stage


# ── Main generation loop ──────────────────────────────────────────────────────
def generate_dataset(n_samples: int, n_simulations: int, output_path: str):
    records = []

    print(f"Generating {n_samples} samples ({n_simulations} MC sims each)...")

    for _ in tqdm(range(n_samples)):
        hole_cards, board_cards, num_opponents, stage = sample_game_state()

        # ── Label: MC estimated win probability ──
        win_prob = estimate_win_probability(
            hole_cards, board_cards, num_opponents, n_simulations
        )

        # ── Features: 135-dim vector ──
        features = build_feature_vector(hole_cards, board_cards, num_opponents)

        records.append((*features, win_prob))

    # ── Save to CSV ───────────────────────────────────────────────────────────
    n_features = 135
    col_names  = [f"f_{i}" for i in range(n_features)] + ["win_probability"]
    df = pd.DataFrame(records, columns=col_names)
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to '{output_path}'")
    print(f"Shape: {df.shape}")
    print(f"\nLabel statistics:")
    print(df["win_probability"].describe().round(3))

    return df


# ── Quick dataset preview ─────────────────────────────────────────────────────
def preview_dataset(df: pd.DataFrame, n: int = 5):
    print(f"\nFirst {n} rows (last 5 features + label):")
    print(df.iloc[:n, -6:].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(N_SAMPLES, N_SIMULATIONS, OUTPUT_PATH)
    preview_dataset(df)
