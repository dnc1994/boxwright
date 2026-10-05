"""Decoder-only transformer for level generation (docs/design-decisions.md, D5–D6).

Pre-norm blocks with RMSNorm, QK-norm, SwiGLU, no biases, untied embeddings.

Positional signal is rotary, in one of two modes:
- "1d": standard RoPE over the sequence index.
- "2d": axial RoPE over grid coordinates (half of each head's dims rotate by row,
  half by column). Queries are rotated by the coordinates of the cell they are
  about to predict and keys by the coordinates of the cell they hold, so a score
  depends on (target cell − key cell). "The tile directly above the one I'm
  placing" is then the same offset everywhere in the grid, including at row wraps.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from boxwright import tokenizer as tok


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    d_ff: int = 1024
    vocab_size: int = len(tok.VOCAB)
    seq_len: int = tok.SEQ_LEN
    rope: str = "2d"  # "1d" or "2d"
    rope_base: float = 100.0  # positions span ≤ 103 (1d) or ≤ 12 (2d), so a small base suffices

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


PRESETS = {
    "6m": ModelConfig(d_model=256, n_layers=6, n_heads=4, d_ff=768),
    "20m": ModelConfig(d_model=384, n_layers=12, n_heads=6, d_ff=1024),
    "50m": ModelConfig(d_model=512, n_layers=16, n_heads=8, d_ff=1408),
}


def _positions(cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Per sequence index: (query positions, key positions), each (seq_len, n_axes)."""
    n = cfg.seq_len
    if cfg.rope == "1d":
        idx = torch.arange(n).unsqueeze(1)
        return idx, idx
    if cfg.rope != "2d":
        raise ValueError(f"unknown rope mode {cfg.rope!r}")
    # The cell held by sequence index j: prefix tokens sit at a virtual cell (-1, -1),
    # tile k at (k // 10, k % 10), <eos> at (10, 0).
    cells = [(-1, -1)] * tok.PREFIX_LEN
    cells += [(k // tok.SIZE, k % tok.SIZE) for k in range(tok.N_TILES)]
    cells += [(tok.SIZE, 0)]
    key = torch.tensor(cells[:n])
    # Index j predicts token j + 1, so its query uses the next cell's coordinates.
    query = torch.tensor((cells[1:] + [(tok.SIZE, 1)])[:n])
    return query, key


def _rotary_tables(pos: torch.Tensor, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq, head_dim // 2), splitting dims evenly across axes."""
    n_axes = pos.shape[1]
    pairs = head_dim // 2
    per_axis = pairs // n_axes
    if per_axis * n_axes != pairs:
        raise ValueError(f"head_dim {head_dim} not divisible across {n_axes} rotary axes")
    inv_freq = base ** (-torch.arange(per_axis) / per_axis)
    angles = torch.cat([pos[:, a : a + 1].float() * inv_freq for a in range(n_axes)], dim=1)
    return angles.cos(), angles.sin()


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to x of shape (B, H, T, D) using (T, D/2) tables."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim)
        self.k_norm = nn.RMSNorm(cfg.head_dim)

    def forward(self, x, rope, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        (q_cos, q_sin), (k_cos, k_sin) = rope
        q, k = _rotate(q, q_cos, q_sin), _rotate(k, k_cos, k_sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # With a cache, new queries attend to all cached keys; causal only within the new chunk.
        causal = T > 1
        if causal and k.shape[2] != T:
            raise ValueError("multi-token chunks with an existing cache are not supported")
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, rope, cache=None):
        x = x + self.attn(self.attn_norm(x), rope, cache)
        return x + self.mlp(self.mlp_norm(x))


class LevelModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        q_pos, k_pos = _positions(cfg)
        for name, pos in [("q", q_pos), ("k", k_pos)]:
            cos, sin = _rotary_tables(pos, cfg.head_dim, cfg.rope_base)
            self.register_buffer(f"{name}_cos", cos, persistent=False)
            self.register_buffer(f"{name}_sin", sin, persistent=False)
        self.apply(self._init)
        for block in self.blocks:  # residual-branch outputs start at zero
            nn.init.zeros_(block.attn.proj.weight)
            nn.init.zeros_(block.mlp.down.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=m.in_features**-0.5)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=1.0)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, caches: list[dict] | None = None, start: int = 0) -> torch.Tensor:
        """Logits for `idx` (B, T) occupying sequence positions start..start+T-1."""
        T = idx.shape[1]
        sl = slice(start, start + T)
        rope = ((self.q_cos[sl], self.q_sin[sl]), (self.k_cos[sl], self.k_sin[sl]))
        x = self.embed(idx.long())
        for i, block in enumerate(self.blocks):
            x = block(x, rope, None if caches is None else caches[i])
        return self.head(self.norm(x)).float()

    def loss(self, seqs: torch.Tensor) -> torch.Tensor:
        """Next-token loss on the grid and <eos> (the prefix is given, not predicted)."""
        logits = self(seqs[:, :-1])
        targets = seqs[:, 1:].long().clone()  # .long() is a no-op view for int64 input
        targets[:, : tok.PREFIX_LEN - 1] = -100
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def generate(
        self,
        conditions: torch.Tensor,
        temperature: float = 1.0,
        cfg_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample grids for condition token ids (B,), returning full sequences (B, SEQ_LEN).

        Only tile tokens are allowed during the 100 grid steps, and <eos> is forced
        at the end, so every output decodes to a 10×10 grid (it may still be
        structurally invalid, e.g. two players). `cfg_scale` > 1 applies
        classifier-free guidance against the <uncond> prefix.
        """
        B, device = conditions.shape[0], conditions.device
        guided = cfg_scale != 1.0
        cond = conditions.to(device)
        if guided:
            cond = torch.cat([cond, torch.full_like(cond, tok.UNCOND)])
        seqs = torch.stack([torch.full_like(cond, tok.BOS), cond], dim=1)
        caches = [{} for _ in self.blocks]
        logits = self(seqs, caches)[:, -1]
        out = [seqs[:B]]
        n_tiles = len(tok.TILES)
        for step in range(tok.N_TILES):
            step_logits = logits[:, :n_tiles]
            if guided:
                c, u = step_logits[:B], step_logits[B:]
                step_logits = u + cfg_scale * (c - u)
            probs = F.softmax(step_logits / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            out.append(nxt)
            feed = torch.cat([nxt, nxt]) if guided else nxt
            logits = self(feed, caches, start=tok.PREFIX_LEN + step)[:, -1]
        out.append(torch.full((B, 1), tok.EOS, device=device, dtype=out[0].dtype))
        return torch.cat(out, dim=1)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
