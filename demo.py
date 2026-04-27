import os
import sys
import numpy as np
import joblib
from treys import Card

from feature_encoding import build_feature_vector

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mlp.pkl")

try:
    checkpoint = joblib.load(MODEL_PATH)
    model  = checkpoint["model"]
    scaler = checkpoint["scaler"]
    print("✓ MLP model loaded.\n")
except FileNotFoundError:
    print(f"Error: {MODEL_PATH} not found. Make sure mlp.pkl is in the same folder.")
    sys.exit(1)

RANK_MAP = {
    '2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8',
    '9':'9','t':'T','T':'T','j':'J','J':'J','q':'Q','Q':'Q',
    'k':'K','K':'K','a':'A','A':'A'
}
SUIT_MAP = {'s':'s','h':'h','d':'d','c':'c'}

def parse_card(token: str):
    token = token.strip()
    if len(token) != 2:
        raise ValueError(f"Invalid card: '{token}'. Use format like As, Kh, 2d.")
    rank = RANK_MAP.get(token[0])
    suit = SUIT_MAP.get(token[1].lower())
    if not rank or not suit:
        raise ValueError(f"Invalid card: '{token}'. Ranks: 2-9 T J Q K A, Suits: s h d c")
    return Card.new(rank + suit)

def parse_cards(s: str):
    tokens = [t for t in s.replace(',', ' ').split() if t]
    return [parse_card(t) for t in tokens]

def predict(hole_cards, board_cards, num_opponents):
    fv = build_feature_vector(hole_cards, board_cards, num_opponents)
    fv_scaled = scaler.transform(fv.reshape(1, -1))
    prob = float(model.predict(fv_scaled)[0])
    return max(0.0, min(1.0, prob))

def bar(p, width=30):
    filled = int(p * width)
    return '█' * filled + '░' * (width - filled)

def show_result(hole_cards, board_cards, num_opponents, prob):
    stage = {0:'Preflop', 3:'Flop', 4:'Turn', 5:'River'}.get(len(board_cards), '?')
    hole_str  = ' '.join(Card.int_to_str(c) for c in hole_cards)
    board_str = ' '.join(Card.int_to_str(c) for c in board_cards) if board_cards else '(none)'
    pct = prob * 100

    print("\n" + "─" * 45)
    print(f"  Stage      : {stage}")
    print(f"  Hole cards : {hole_str}")
    print(f"  Board      : {board_str}")
    print(f"  Opponents  : {num_opponents}")
    print(f"  {'─'*40}")
    print(f"  Win prob   : {pct:.1f}%")
    print(f"  {bar(prob)}  {pct:.1f}%")

    if pct >= 60:
        verdict = "Strong hand"
    elif pct >= 40:
        verdict = "Marginal"
    else:
        verdict = "Underdog"
    print(f"  Verdict    : {verdict}")
    print("─" * 45 + "\n")

def main():
    print("=" * 45)
    print("  Texas Hold'em Win Probability — MLP Demo")
    print("=" * 45)
    print("  Card format: rank + suit")
    print("  Ranks: 2-9  T  J  Q  K  A")
    print("  Suits: s(spade) h(heart) d(diamond) c(club)")
    print("  Example: As Kh  →  Ace of spades, King of hearts")
    print("  Type 'quit' to exit, 'example' for a demo hand.\n")

    while True:
        try:
            # Hole cards
            raw = input("Hole cards (2 cards, e.g. 'As Kh'): ").strip()
            if raw.lower() == 'quit':
                break
            if raw.lower() == 'example':
                raw = 'As Ks'
                print(f"  → Using: {raw}")

            hole_cards = parse_cards(raw)
            if len(hole_cards) != 2:
                print("  Need exactly 2 hole cards.\n")
                continue

            # Board cards
            raw_board = input("Board cards (0/3/4/5 cards, or press Enter for preflop): ").strip()
            if raw_board.lower() == 'quit':
                break

            board_cards = parse_cards(raw_board) if raw_board else []
            if len(board_cards) not in (0, 3, 4, 5):
                print("  Board must have 0 (preflop), 3 (flop), 4 (turn), or 5 (river) cards.\n")
                continue

            # Check duplicates
            all_ids = [c for c in hole_cards + board_cards]
            if len(all_ids) != len(set(all_ids)):
                print("  Duplicate cards detected. Please re-enter.\n")
                continue

            # Opponents
            raw_opp = input("Number of opponents (1-8, default 2): ").strip()
            if raw_opp.lower() == 'quit':
                break
            num_opponents = int(raw_opp) if raw_opp else 2
            if not (1 <= num_opponents <= 8):
                print("  Opponents must be between 1 and 8.\n")
                continue

            prob = predict(hole_cards, board_cards, num_opponents)
            show_result(hole_cards, board_cards, num_opponents, prob)

            again = input("Try another hand? (Enter to continue, 'quit' to exit): ").strip()
            if again.lower() == 'quit':
                break
            print()

        except ValueError as e:
            print(f"  Input error: {e}\n")
        except KeyboardInterrupt:
            break

    print("\nGood luck at the table!")

if __name__ == "__main__":
    main()