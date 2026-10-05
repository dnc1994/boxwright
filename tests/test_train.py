import json
from pathlib import Path

import torch

from boxwright import tokenizer as tok
from boxwright.data.symmetry import orbit
from boxwright.train_pretrain import augment, lr_factor

FIXTURE = Path(__file__).parent / "fixtures" / "boxoban_medium_valid_sample.json"


def test_augment_yields_symmetric_images_only():
    texts = [c["text"] for c in json.loads(FIXTURE.read_text())]
    seqs = torch.from_numpy(tok.encode_batch(texts * 8))
    out = augment(seqs, torch.Generator().manual_seed(0))
    assert (out[:, : tok.PREFIX_LEN] == seqs[:, : tok.PREFIX_LEN]).all()
    assert (out[:, -1] == tok.EOS).all()
    seen = set()
    for row, text in zip(out, texts * 8):
        image = tok.decode(row)
        assert image in orbit(text)
        seen.add(orbit(text).index(image))
    assert len(seen) == 8, "all 8 symmetries should appear"


def test_lr_schedule():
    f = [lr_factor(s, 100, 10, 0.2) for s in range(101)]
    assert f[0] == 0.1 and f[9] == 1.0
    assert all(x == 1.0 for x in f[10:80])
    assert f[90] == 0.5 and f[100] == 0.0
