import numpy as np
from treys import Card

# ── Constants ────────────────────────────────────────────────────────────────
RANKS = list(range(2, 15))          # 2–14 (14 = Ace)
SUITS = [8, 4, 2, 1]               # treys internal suit ints: s, h, d, c
STAGES = ['preflop', 'flop', 'turn', 'river']

# ── Low-level helpers ────────────────────────────────────────────────────────
def get_rank(card: int) -> int:
    """Return rank as int 2–14."""
    return Card.get_rank_int(card) + 2   # treys: 0=2 ... 12=A → shift to 2–14

def get_suit(card: int) -> int:
    """Return treys suit int (8/4/2/1)."""
    return Card.get_suit_int(card)

# ── One-Hot encoding ─────────────────────────────────────────────────────────
def one_hot_rank(card: int) -> np.ndarray:
    """13-dim one-hot for rank (index 0=2, index 12=Ace)."""
    vec = np.zeros(13)
    vec[get_rank(card) - 2] = 1
    return vec

def one_hot_suit(card: int) -> np.ndarray:
    """4-dim one-hot for suit (s/h/d/c)."""
    vec = np.zeros(4)
    vec[SUITS.index(get_suit(card))] = 1
    return vec

def encode_card(card: int) -> np.ndarray:
    """17-dim one-hot for a single card (13 rank + 4 suit)."""
    return np.concatenate([one_hot_rank(card), one_hot_suit(card)])

def encode_cards(cards: list, max_cards: int) -> np.ndarray:
    """
    Encode a variable-length card list into a fixed-size vector.
    Missing cards (e.g. no board yet) are zero-padded.
    max_cards × 17 dims total.
    """
    vecs = [encode_card(c) for c in cards]
    while len(vecs) < max_cards:
        vecs.append(np.zeros(17))       # pad missing cards with zeros
    return np.concatenate(vecs)

# ── Hand-crafted features ────────────────────────────────────────────────────
def handcrafted_features(hole_cards: list, board_cards: list) -> np.ndarray:
    """
    Compute interpretable features that are hard for a model to derive
    from raw one-hot encodings alone.

    Returns a 1-D numpy array.
    """
    all_cards  = hole_cards + board_cards
    ranks      = [get_rank(c) for c in all_cards]
    suits      = [get_suit(c) for c in all_cards]
    hole_ranks = [get_rank(c) for c in hole_cards]
    hole_suits = [get_suit(c) for c in hole_cards]

    # ── Hole-card features ───────────────────────────────────────────────────
    is_suited      = float(hole_suits[0] == hole_suits[1])
    rank_gap       = abs(hole_ranks[0] - hole_ranks[1])       # 0 = pair, 1 = connected
    is_pair        = float(rank_gap == 0)
    is_connected   = float(rank_gap == 1)
    has_ace        = float(14 in hole_ranks)
    high_card      = max(hole_ranks) / 14.0                   # normalised 0–1

    # ── Board features ───────────────────────────────────────────────────────
    suit_counts    = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values()) if suits else 0
    is_flush       = float(max_suit_count >= 5)
    flush_draw     = float(max_suit_count == 4)               # one card away

    rank_set       = sorted(set(ranks))
    is_straight    = float(
        len(rank_set) >= 5 and
        any(rank_set[i+4] - rank_set[i] == 4 for i in range(len(rank_set) - 4))
    )

    rank_counts    = {r: ranks.count(r) for r in set(ranks)}
    has_board_pair = float(any(v >= 2 for v in rank_counts.values()))
    has_trips      = float(any(v >= 3 for v in rank_counts.values()))

    # ── Game stage ───────────────────────────────────────────────────────────
    n_board = len(board_cards)
    stage_vec = np.zeros(4)                                   # preflop/flop/turn/river
    stage_idx = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_board, 0)
    stage_vec[stage_idx] = 1.0

    scalar_feats = np.array([
        is_suited, rank_gap / 12.0, is_pair, is_connected,
        has_ace, high_card,
        is_flush, flush_draw, is_straight,
        has_board_pair, has_trips,
    ])

    return np.concatenate([scalar_feats, stage_vec])          # 11 + 4 = 15 dims

# ── Master feature vector ────────────────────────────────────────────────────
def build_feature_vector(hole_cards: list, board_cards: list, num_opponents: int) -> np.ndarray:
    """
    Combine all features into one flat vector.

    Dimensions:
      2 hole cards   × 17 =  34   (one-hot)
      5 board cards  × 17 =  85   (one-hot, zero-padded if < 5)
      handcrafted         =  15
      num_opponents       =   1   (normalised)
      ─────────────────────────
      TOTAL               = 135
    """
    hole_enc  = encode_cards(hole_cards, max_cards=2)         # 34 dims
    board_enc = encode_cards(board_cards, max_cards=5)        # 85 dims
    hc_feats  = handcrafted_features(hole_cards, board_cards) # 15 dims
    opp_feat  = np.array([num_opponents / 8.0])               #  1 dim (normalised)

    return np.concatenate([hole_enc, board_enc, hc_feats, opp_feat])  # 135 dims


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hole  = [Card.new('As'), Card.new('Ks')]
    board = [Card.new('Qs'), Card.new('Js'), Card.new('2h')]  # Flop

    fv = build_feature_vector(hole, board, num_opponents=2)
    print(f"Feature vector shape : {fv.shape}")               # (135,)
    print(f"First 34 dims (hole) : {fv[:34]}")
    print(f"Handcrafted features : {fv[119:134]}")
    print(f"Opponent feature     : {fv[134]:.3f}")            # 2/8 = 0.25
