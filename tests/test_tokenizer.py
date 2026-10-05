import json
import random
from pathlib import Path

import boxwright_solver as solver
import numpy as np
import pytest

from boxwright import tokenizer as tok

FIXTURE = Path(__file__).parent / "fixtures" / "boxoban_medium_valid_sample.json"
LEVELS = [c["text"] for c in json.loads(FIXTURE.read_text())]


def test_round_trip():
    for text in LEVELS:
        ids = tok.encode(text, bucket=3)
        assert len(ids) == tok.SEQ_LEN
        assert ids[:2] == [tok.BOS, tok.BUCKET0 + 3] and ids[-1] == tok.EOS
        assert tok.decode(ids) == text
        assert tok.decode(ids[tok.PREFIX_LEN :]) == text


def test_batch_matches_single():
    buckets = np.arange(len(LEVELS)) % tok.N_BUCKETS
    batch = tok.encode_batch(LEVELS, buckets)
    assert batch.shape == (len(LEVELS), tok.SEQ_LEN)
    for row, text, b in zip(batch, LEVELS, buckets):
        assert row.tolist() == tok.encode(text, int(b))
    assert (tok.encode_batch(LEVELS[:2])[:, 1] == tok.UNCOND).all()


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        tok.encode("#" * 10)
    with pytest.raises(ValueError):
        tok.encode(LEVELS[0].replace("#", "x", 1))
    with pytest.raises(ValueError):
        tok.encode(LEVELS[0], bucket=10)
    with pytest.raises(ValueError):
        tok.encode_batch([LEVELS[0][:-1]])


def test_decode_rejects_malformed():
    ids = tok.encode(LEVELS[0])
    assert tok.decode(ids[:-5]) is None, "too few tiles"
    assert tok.decode(ids[:50] + [tok.EOS]) is None, "early eos"
    assert tok.decode(ids[:10] + [tok.BUCKET0] + ids[11:]) is None, "special token in grid"
    assert tok.decode(ids[:-1]) == LEVELS[0], "missing eos is fine after 100 tiles"


def test_validity_agrees_with_solver_parser():
    rng = random.Random(0)
    weights = [30, 50, 3, 4, 4, 1, 1]  # walls, floor, then rare player/box/goal tiles
    agree = 0
    for i in range(3000):
        if i % 2:  # random grid
            text = "\n".join("".join(rng.choices(tok.TILES, weights, k=10)) for _ in range(10))
        else:  # real level with 1–2 random tile changes, near the valid/invalid boundary
            cells = list(rng.choice(LEVELS).replace("\n", ""))
            for _ in range(rng.randint(1, 2)):
                cells[rng.randrange(100)] = rng.choice(tok.TILES)
            text = "\n".join("".join(cells[r * 10 : (r + 1) * 10]) for r in range(10))
        valid = tok.is_valid(text)
        assert valid == (solver.solve(text, max_nodes=1).status != "invalid"), text
        agree += valid
    assert agree > 50, "fuzzing should produce a reasonable number of valid levels"
    assert all(tok.is_valid(t) for t in LEVELS)
