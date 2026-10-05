"""Tile-level tokenizer for 10×10 Boxoban levels.

Sequence layout (103 tokens): `<bos> <dK>|<uncond> tile×100 <eos>`, tiles in
row-major order with no row separators (see docs/design-decisions.md, D4).
"""

import numpy as np

SIZE = 10
N_TILES = SIZE * SIZE
N_BUCKETS = 10

TILES = "# @$.*+"  # wall, floor, player, box, goal, box on goal, player on goal
SPECIALS = ["<pad>", "<bos>", "<eos>", "<uncond>"] + [f"<d{k}>" for k in range(N_BUCKETS)]
VOCAB = list(TILES) + SPECIALS
TOKEN_ID = {tok: i for i, tok in enumerate(VOCAB)}

PAD, BOS, EOS, UNCOND = (TOKEN_ID[t] for t in ["<pad>", "<bos>", "<eos>", "<uncond>"])
BUCKET0 = TOKEN_ID["<d0>"]
SEQ_LEN = 2 + N_TILES + 1
PREFIX_LEN = 2  # <bos> + condition; the model predicts from position PREFIX_LEN on

# Byte -> token id lookup for fast batch encoding; 255 marks a non-tile byte.
_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _ch in enumerate(TILES):
    _LUT[ord(_ch)] = _i


def condition_token(bucket: int | None) -> int:
    if bucket is None:
        return UNCOND
    if not 0 <= bucket < N_BUCKETS:
        raise ValueError(f"bucket must be in 0..{N_BUCKETS - 1}, got {bucket}")
    return BUCKET0 + bucket


def is_valid(text: str) -> bool:
    """Structurally valid 10×10 level: known tiles, one player, boxes == goals ≥ 1."""
    rows = text.split("\n")
    if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
        return False
    if any(ch not in TILES for ch in text.replace("\n", "")):
        return False
    players = text.count("@") + text.count("+")
    boxes = text.count("$") + text.count("*")
    goals = text.count(".") + text.count("*") + text.count("+")
    return players == 1 and boxes >= 1 and boxes == goals


def encode(text: str, bucket: int | None = None) -> list[int]:
    """Encode a 10×10 level (rows joined by "\\n") with an optional difficulty bucket."""
    rows = text.split("\n")
    if len(rows) != SIZE or any(len(r) != SIZE for r in rows):
        raise ValueError("level must be exactly 10 rows of 10 tiles")
    try:
        tiles = [TOKEN_ID[ch] for ch in "".join(rows)]
    except KeyError as e:
        raise ValueError(f"unknown tile {e.args[0]!r}") from None
    return [BOS, condition_token(bucket), *tiles, EOS]


def encode_batch(texts: list[str], buckets: np.ndarray | None = None) -> np.ndarray:
    """Vectorized `encode` for many levels: returns uint8 array of shape (n, SEQ_LEN)."""
    flat = "".join(texts).replace("\n", "").encode("ascii")
    tiles = _LUT[np.frombuffer(flat, dtype=np.uint8)]
    if tiles.size != len(texts) * N_TILES or (tiles == 255).any():
        raise ValueError("every level must be 10×10 using only tile characters")
    out = np.empty((len(texts), SEQ_LEN), dtype=np.uint8)
    out[:, 0] = BOS
    out[:, 1] = UNCOND if buckets is None else BUCKET0 + np.asarray(buckets)
    out[:, PREFIX_LEN : PREFIX_LEN + N_TILES] = tiles.reshape(-1, N_TILES)
    out[:, -1] = EOS
    return out


def decode(ids) -> str | None:
    """Turn generated ids back into level text.

    Accepts a full sequence (with prefix) or just the generated part. Reads tiles
    up to `<eos>`; returns None unless exactly 100 tile tokens precede it (or the
    sequence ends after 100 tiles).
    """
    ids = [int(i) for i in ids]
    if len(ids) >= PREFIX_LEN and ids[0] == BOS:
        ids = ids[PREFIX_LEN:]
    if EOS in ids:
        ids = ids[: ids.index(EOS)]
    if len(ids) != N_TILES or any(i >= len(TILES) for i in ids):
        return None
    chars = "".join(TILES[i] for i in ids)
    return "\n".join(chars[r * SIZE : (r + 1) * SIZE] for r in range(SIZE))
