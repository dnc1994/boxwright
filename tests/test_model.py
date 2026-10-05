import pytest
import torch

from boxwright import tokenizer as tok
from boxwright.model import PRESETS, LevelModel, ModelConfig

TINY = dict(d_model=64, n_layers=2, n_heads=2, d_ff=128)


@pytest.fixture(params=["1d", "2d"])
def model(request):
    torch.manual_seed(0)
    m = LevelModel(ModelConfig(**TINY, rope=request.param)).eval()
    # Non-zero residual outputs so the test exercises every layer.
    for b in m.blocks:
        torch.nn.init.normal_(b.attn.proj.weight, std=0.02)
        torch.nn.init.normal_(b.mlp.down.weight, std=0.02)
    return m


def random_seqs(n=3):
    seqs = torch.randint(0, len(tok.TILES), (n, tok.SEQ_LEN))
    seqs[:, 0], seqs[:, 1], seqs[:, -1] = tok.BOS, tok.UNCOND, tok.EOS
    return seqs


def test_preset_sizes():
    sizes = {name: LevelModel(cfg).num_params() / 1e6 for name, cfg in PRESETS.items()}
    assert 4 < sizes["6m"] < 8 and 15 < sizes["20m"] < 25 and 40 < sizes["50m"] < 60, sizes


def test_causal(model):
    seqs = random_seqs()
    logits = model(seqs)
    changed = seqs.clone()
    changed[:, 60:] = (changed[:, 60:] + 1) % len(tok.TILES)
    assert torch.allclose(model(changed)[:, :60], logits[:, :60], atol=1e-5)


def test_kv_cache_matches_full_forward(model):
    seqs = random_seqs()
    full = model(seqs)
    caches = [{} for _ in model.blocks]
    steps = [model(seqs[:, :2], caches)]
    for t in range(2, tok.SEQ_LEN):
        steps.append(model(seqs[:, t : t + 1], caches, start=t))
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


def test_loss_and_generate(model):
    loss = model.loss(random_seqs())
    assert loss.ndim == 0 and torch.isfinite(loss)
    conds = torch.tensor([tok.BUCKET0, tok.BUCKET0 + 9])
    g = torch.Generator().manual_seed(0)
    for scale in [1.0, 2.0]:
        out = model.generate(conds, cfg_scale=scale, generator=g)
        assert out.shape == (2, tok.SEQ_LEN)
        assert (out[:, 1] == conds).all() and (out[:, -1] == tok.EOS).all()
        assert all(tok.decode(row) is not None for row in out)


def test_loss_does_not_modify_int64_input(model):
    seqs = random_seqs().long()
    before = seqs.clone()
    model.loss(seqs).backward()
    assert torch.equal(seqs, before)
