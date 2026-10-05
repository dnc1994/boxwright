"""Pretrain a LevelModel on Boxoban (docs/design-decisions.md, D6–D7).

    uv run python -m boxwright.train_pretrain --preset 6m --steps 2000 --out checkpoints/dev

Muon for hidden weight matrices and AdamW for everything else, on a
warmup-stable-decay schedule. Each batch gets a random grid symmetry per level
and has its difficulty condition dropped to <uncond> with probability
`--cond-dropout`. Periodic evals report validation loss plus solver-checked
sample metrics; logs go to stdout, `<out>/log.jsonl`, and optionally W&B.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from boxwright import tokenizer as tok
from boxwright.data.novelty import NoveltyIndex
from boxwright.eval import metrics
from boxwright.model import PRESETS, LevelModel

# Overridable so cloud runs can read from a mounted volume.
DATA = Path(os.environ.get("BOXWRIGHT_DATA", Path(__file__).resolve().parents[1] / "data" / "processed"))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(split: str, limit: int | None = None, fraction: float = 1.0, seed: int = 0) -> torch.Tensor:
    table = pq.read_table(DATA / f"{split}.parquet", columns=["text", "bucket"])
    n = len(table)
    keep = np.arange(n)
    if fraction < 1.0:
        keep = np.sort(np.random.default_rng(seed).choice(n, int(n * fraction), replace=False))
    if limit is not None:
        keep = keep[:limit]
    table = table.take(keep)
    seqs = tok.encode_batch(table["text"].to_pylist(), table["bucket"].to_numpy())
    return torch.from_numpy(seqs)


def augment(seqs: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply an independent random grid symmetry (of 8) to each sequence."""
    seqs = seqs.clone()
    grids = seqs[:, tok.PREFIX_LEN : tok.PREFIX_LEN + tok.N_TILES].view(-1, tok.SIZE, tok.SIZE)
    ks = torch.randint(0, 8, (len(seqs),), generator=gen, device="cpu").to(seqs.device)
    for k in range(1, 8):
        mask = ks == k
        if mask.any():
            g = torch.rot90(grids[mask], k % 4, dims=(1, 2))
            grids[mask] = g.flip(2) if k >= 4 else g
    return seqs


def lr_factor(step: int, total: int, warmup: int, decay_frac: float) -> float:
    """Warmup-stable-decay: linear warmup, flat, then linear decay to zero."""
    decay_start = int(total * (1 - decay_frac))
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - decay_start))


def build_optimizers(model: LevelModel, lr: float, weight_decay: float, kind: str):
    hidden = [p for n, p in model.blocks.named_parameters() if p.ndim == 2]
    other = [p for p in model.parameters() if not any(p is h for h in hidden)]
    if kind == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)]
    return [
        # match_rms_adamw scales Muon updates to AdamW-like RMS, so one lr serves both.
        torch.optim.Muon(hidden, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW(other, lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
    ]


@torch.no_grad()
def evaluate(model, val, args, device, edges, novelty, gen) -> dict:
    model.eval()
    losses = []
    for i in range(0, len(val), args.batch_size):
        batch = val[i : i + args.batch_size].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            losses.append(model.loss(batch).item())
    out = {"val_loss": float(np.mean(losses))}

    targets = [b for b in range(tok.N_BUCKETS) for _ in range(args.samples_per_bucket)]
    conds = torch.tensor([tok.condition_token(b) for b in targets], device=device)
    seqs = model.generate(conds, temperature=1.0, generator=gen)
    texts = [tok.decode(s) for s in seqs.cpu()]
    out |= {f"sample_{k}": v for k, v in metrics.summarize(metrics.score(texts, targets, edges, novelty)).items()}
    model.train()
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="6m", choices=PRESETS)
    ap.add_argument("--rope", default="2d", choices=["1d", "2d"])
    ap.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--cond-dropout", type=float, default=0.1)
    ap.add_argument("--data-fraction", type=float, default=1.0)
    ap.add_argument("--train-limit", type=int, default=None, help="cap training levels (for quick runs)")
    ap.add_argument("--val-size", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--samples-per-bucket", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cfg = PRESETS[args.preset]
    cfg.rope = args.rope
    model = LevelModel(cfg).to(device)
    print(f"model {args.preset} ({model.num_params() / 1e6:.1f}M params, rope={cfg.rope}) on {device}")

    train = load_split("train", args.train_limit, args.data_fraction, args.seed).to(device)
    val = load_split("val", args.val_size)
    edges = json.loads((DATA / "buckets.json").read_text())["edges"]
    novelty = NoveltyIndex.from_parquet(DATA / "train.parquet")
    print(f"train {len(train):,} levels, val {len(val):,}")

    optimizers = build_optimizers(model, args.lr, args.weight_decay, args.optimizer)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    step_fn = torch.compile(model.loss) if args.compile and device.type == "cuda" else model.loss
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    sample_gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    args.out.mkdir(parents=True, exist_ok=True)
    run_config = {"args": {k: str(v) for k, v in vars(args).items()}, "model": asdict(cfg)}
    (args.out / "config.json").write_text(json.dumps(run_config, indent=2))
    log = (args.out / "log.jsonl").open("a")
    if args.wandb:
        import wandb

        wandb.init(project="boxwright", name=args.out.name, config=run_config)

    def emit(record: dict) -> None:
        log.write(json.dumps(record) + "\n")
        log.flush()
        if args.wandb:
            wandb.log(record, step=record["step"])

    model.train()
    start, tokens = time.perf_counter(), 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            record = {"step": step, **evaluate(model, val, args, device, edges, novelty, sample_gen)}
            elapsed = time.perf_counter() - start
            record["tokens_per_sec"] = tokens / elapsed if step else 0.0
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            emit(record)
            ckpt = {"model": model.state_dict(), "config": asdict(cfg), "step": step}
            torch.save(ckpt, args.out / "last.pt")
        if step == args.steps:
            break

        idx = torch.randint(0, len(train), (args.batch_size,), generator=gen).to(device)
        batch = augment(train[idx], gen)
        drop = torch.rand(len(batch), generator=gen).to(device) < args.cond_dropout
        batch[drop, 1] = tok.UNCOND

        factor = lr_factor(step, args.steps, args.warmup, args.decay_frac)
        for opt, lrs in zip(optimizers, base_lrs):
            for group, lr in zip(opt.param_groups, lrs):
                group["lr"] = lr * factor
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = step_fn(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        tokens += batch.numel()

        if step % 50 == 0:
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"loss diverged at step {step}")
            emit({"step": step, "train_loss": loss.item(), "lr_factor": factor, "grad_norm": grad_norm.item()})

    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": args.steps}, args.out / "final.pt")
    print(f"done in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
