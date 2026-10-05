"""The 8 symmetries of a square grid (rotations and reflections).

Sokoban rules are invariant under all of them, so a transformed level has the
same solvability and optimal move count. Used for training augmentation and
for symmetry-aware novelty/deduplication.
"""


def _rows(text: str) -> list[str]:
    return text.split("\n")


def _rot90(rows: list[str]) -> list[str]:
    """Rotate clockwise."""
    return ["".join(row[c] for row in reversed(rows)) for c in range(len(rows[0]))]


def _flip(rows: list[str]) -> list[str]:
    """Mirror left-right."""
    return [row[::-1] for row in rows]


def transform(text: str, k: int) -> str:
    """Apply symmetry `k` in 0..7: rotate clockwise `k % 4` times, then mirror if `k >= 4`."""
    if not 0 <= k < 8:
        raise ValueError(f"symmetry index must be in 0..7, got {k}")
    rows = _rows(text)
    if len({len(r) for r in rows}) != 1:
        raise ValueError("level rows must all have the same width")
    for _ in range(k % 4):
        rows = _rot90(rows)
    if k >= 4:
        rows = _flip(rows)
    return "\n".join(rows)


def orbit(text: str) -> list[str]:
    """All 8 transforms of a level (with repeats if the level is symmetric)."""
    return [transform(text, k) for k in range(8)]


def canonical(text: str) -> str:
    """A representative that is identical for every level in the same symmetry orbit."""
    return min(orbit(text))
