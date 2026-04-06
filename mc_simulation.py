from treys import Card, Deck, Evaluator
import random

evaluator = Evaluator()

def estimate_win_probability(hole_cards: list, board_cards: list, num_opponents: int, n_simulations: int = 1000) -> float:
    """
    Estimate win probability using Monte Carlo simulation.

    Args:
        hole_cards:    List of 2 treys Card ints, e.g. [Card.new('As'), Card.new('Kh')]
        board_cards:   List of 0, 3, 4, or 5 treys Card ints (Preflop/Flop/Turn/River)
        num_opponents: Number of opponents (1–8)
        n_simulations: Number of MC samples (higher = more accurate, slower)

    Returns:
        Estimated win probability as a float in [0, 1]
    """
    wins = 0

    # All known cards — we won't redeal these
    known_cards = hole_cards + board_cards

    for _ in range(n_simulations):
        # ── 1. Build a fresh shuffled deck excluding known cards ──
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in known_cards]
        random.shuffle(deck.cards)

        # ── 2. Deal hole cards to each opponent ──
        opponent_hands = []
        remaining = deck.cards[:]
        for _ in range(num_opponents):
            opp_hand = remaining[:2]
            remaining = remaining[2:]
            opponent_hands.append(opp_hand)

        # ── 3. Complete the board to 5 cards ──
        cards_needed = 5 - len(board_cards)
        simulated_board = board_cards + remaining[:cards_needed]

        # ── 4. Evaluate hand strengths (lower score = better in treys) ──
        my_score = evaluator.evaluate(simulated_board, hole_cards)
        opponent_scores = [evaluator.evaluate(simulated_board, opp) for opp in opponent_hands]

        # ── 5. Win if our score is strictly better than ALL opponents ──
        if my_score < min(opponent_scores):
            wins += 1

    return wins / n_simulations


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Scenario: As Kh vs 1 opponent, no board yet (Preflop)
    hole  = [Card.new('As'), Card.new('Kh')]
    board = []  # Preflop — no community cards yet

    prob = estimate_win_probability(hole, board, num_opponents=1, n_simulations=5000)
    print(f"As Kh vs 1 opponent (Preflop) — estimated win prob: {prob:.3f}")
    # Expected: ~0.67 (AKs is a strong hand)
