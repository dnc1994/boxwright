"""Evaluation harness (Task 6).

    python -m boxwright.eval.harness --checkpoint ckpt.pt --out evals/run1
    python -m boxwright.eval.harness --levels levels.jsonl --out evals/frontier

Samples levels from a checkpoint (or reads `{"target": int|null, "text": str}`
lines), checks each with the solver, and reports, per group (conditional,
unconditional, each requested bucket):

- valid / playable: structurally valid / solved within the node budget
- novel: Todd et al. (2023) definition, edit distance ≥ k to every training level
  on the raw text; novel_strict: no rotation/reflection of the level is in train
- diversity: largest set of samples pairwise ≥ k apart (greedy lower bound, as the
  paper's clique search is), over the playable and novel samples
- score: Todd et al.'s headline number, |that set| / n
- bucket_exact / bucket_within1 / spearman: difficulty control for conditional samples

Writes summary.json, levels.jsonl, calibration.png and confusion.png.
Heavy on CPU (edit distance against 1.35M levels); run it on Modal, see infra/.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import boxwright_solver
import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from boxwright import tokenizer as tok
from boxwright.data.symmetry import orbit
from boxwright.eval.metrics import _spearman, bucket_of

EDIT_CAP = 20  # distances above this are reported as EDIT_CAP + 1


@dataclass
class EvalConfig:
    n_per_bucket: int = 200
    n_uncond: int = 1000
    temperature: float = 1.0
    cfg_scale: float = 1.0
    seed: int = 0
    max_nodes: int = 200_000
    novelty_k: int = 5


# ---------------------------------------------------------------- sampling


def sample_levels(checkpoint: Path, cfg: EvalConfig, device: str = "cuda", batch: int = 1000) -> tuple[list[dict], float]:
    """Generate levels; returns records and samples/sec."""
    import torch

    from boxwright.model import LevelModel, ModelConfig

    ckpt = torch.load(checkpoint, map_location=device)
    model = LevelModel(ModelConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])

    targets = [b for b in range(tok.N_BUCKETS) for _ in range(cfg.n_per_bucket)] + [None] * cfg.n_uncond
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    records, elapsed = [], 0.0
    for i in range(0, len(targets), batch):
        chunk = targets[i : i + batch]
        conds = torch.tensor([tok.condition_token(t) for t in chunk], device=device)
        start = time.perf_counter()
        seqs = model.generate(conds, temperature=cfg.temperature, cfg_scale=cfg.cfg_scale, generator=gen)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        records += [{"target": t, "text": tok.decode(s)} for t, s in zip(chunk, seqs.cpu())]
    return records, len(targets) / elapsed


# ---------------------------------------------------------------- distances


def nearest_edit(texts: list[str], corpus: list[str], symmetric: bool, cap: int = EDIT_CAP) -> np.ndarray:
    """Edit distance from each text to its nearest corpus level, capped at cap + 1.

    With `symmetric`, the minimum over all 8 rotations/reflections of the text.
    """
    if not texts:
        return np.zeros(0, dtype=np.int32)
    k = 8 if symmetric else 1
    out = np.empty(len(texts), dtype=np.int32)
    chunk = max(1, 64 // k)
    for i in range(0, len(texts), chunk):
        part = texts[i : i + chunk]
        queries = [img for t in part for img in (orbit(t) if symmetric else [t])]
        d = process.cdist(queries, corpus, scorer=Levenshtein.distance, score_cutoff=cap, dtype=np.uint8, workers=-1)
        out[i : i + len(part)] = d.min(axis=1).reshape(len(part), k).min(axis=1)
    return out


def diversity(texts: list[str], k: int) -> int:
    """Size of a large subset whose members are pairwise ≥ k edits apart.

    Greedy: repeatedly drop the level with the most too-close neighbours. A lower
    bound on the maximum clique, like the paper's iteration-capped search.
    """
    if len(texts) <= 1:
        return len(texts)
    d = process.cdist(texts, texts, scorer=Levenshtein.distance, score_cutoff=k, dtype=np.uint8, workers=-1)
    close = d < k
    np.fill_diagonal(close, False)
    alive = np.ones(len(texts), dtype=bool)
    while True:
        degree = (close & alive[None, :]).sum(axis=1) * alive
        worst = int(degree.argmax())
        if degree[worst] == 0:
            return int(alive.sum())
        alive[worst] = False


# ---------------------------------------------------------------- scoring


def score_records(records: list[dict], corpus: list[str], edges: list[float], cfg: EvalConfig) -> list[dict]:
    texts = [r["text"] for r in records]
    valid = [t is not None and tok.is_valid(t) for t in texts]
    sols = iter(boxwright_solver.solve_batch([t for t, v in zip(texts, valid) if v], max_nodes=cfg.max_nodes))
    decodable = [t for t in texts if t is not None]
    nn_plain = iter(nearest_edit(decodable, corpus, symmetric=False))
    nn_sym = iter(nearest_edit(decodable, corpus, symmetric=True))

    rows = []
    for rec, text, ok in zip(records, texts, valid):
        sol = next(sols) if ok else None
        solved = bool(sol and sol.solved)
        row = {
            "target": rec["target"],
            "text": text,
            "valid": ok,
            "status": sol.status if sol else ("invalid" if text is not None else "undecodable"),
            "solved": solved,
            "moves": sol.moves if solved else None,
            "pushes": sol.pushes if solved else None,
            "bucket": bucket_of(sol.moves, edges) if solved else None,
            "nn_edit": int(next(nn_plain)) if text is not None else None,
            "nn_edit_sym": int(next(nn_sym)) if text is not None else None,
        }
        rows.append(row)
    return rows


def summarize(rows: list[dict], k: int) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    playable_novel = [r for r in rows if r["solved"] and r["nn_edit"] is not None and r["nn_edit"] >= k]
    out = {
        "n": n,
        "valid": sum(r["valid"] for r in rows) / n,
        "playable": sum(r["solved"] for r in rows) / n,
        "novel": sum(r["nn_edit"] is not None and r["nn_edit"] >= k for r in rows) / n,
        "novel_strict": sum(r["nn_edit_sym"] is not None and r["nn_edit_sym"] > 0 for r in rows) / n,
        "playable_novel": len(playable_novel) / n,
        "score": diversity([r["text"] for r in playable_novel], k) / n,
        "unique": len({r["text"] for r in rows if r["valid"]}) / n,
    }
    solved = [r for r in rows if r["solved"]]
    if solved:
        out["moves_median"] = float(np.median([r["moves"] for r in solved]))
    targeted = [r for r in solved if r["target"] is not None]
    if targeted:
        out["bucket_exact"] = float(np.mean([r["bucket"] == r["target"] for r in targeted]))
        out["bucket_within1"] = float(np.mean([abs(r["bucket"] - r["target"]) <= 1 for r in targeted]))
        if len({r["target"] for r in targeted}) > 1:
            out["spearman"] = _spearman([r["target"] for r in targeted], [r["moves"] for r in targeted])
    return out


def summarize_groups(rows: list[dict], k: int) -> dict:
    cond = [r for r in rows if r["target"] is not None]
    groups = {"all": summarize(rows, k), "conditional": summarize(cond, k)}
    groups["unconditional"] = summarize([r for r in rows if r["target"] is None], k)
    groups["per_bucket"] = {b: summarize([r for r in cond if r["target"] == b], k) for b in range(tok.N_BUCKETS)}
    return groups


# ---------------------------------------------------------------- plots

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, BAND = "#2a78d6", "#cde2fb"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def plot_calibration(rows: list[dict], edges: list[float], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    lo = [0.0] + list(edges)
    hi = list(edges) + [max(edges) * 1.5]
    data = []
    for b in range(tok.N_BUCKETS):
        ax.add_patch(plt.Rectangle((b - 0.4, lo[b]), 0.8, hi[b] - lo[b], color=BAND, lw=0, zorder=0))
        data.append([r["moves"] for r in rows if r["target"] == b and r["solved"]] or [np.nan])
    ax.boxplot(
        data,
        positions=range(tok.N_BUCKETS),
        widths=0.35,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": SERIES_1, "edgecolor": SERIES_1, "linewidth": 1},
        whiskerprops={"color": SERIES_1, "linewidth": 1.5},
        capprops={"color": SERIES_1, "linewidth": 1.5},
        medianprops={"color": SURFACE, "linewidth": 2},
    )
    ax.set_xticks(range(tok.N_BUCKETS), [f"d{b}" for b in range(tok.N_BUCKETS)])
    ax.set_xlabel("Requested difficulty bucket", color=INK_2)
    ax.set_ylabel("Optimal solution length (moves)", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.02, "shaded: target move range · box: 25–75%, whiskers 1.5 IQR",
            transform=ax.transAxes, ha="right", color=INK_2, fontsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    m = np.zeros((tok.N_BUCKETS, tok.N_BUCKETS))
    for r in rows:
        if r["target"] is not None and r["solved"]:
            m[r["target"], r["bucket"]] += 1
    m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=SURFACE)
    _style(ax)
    im = ax.imshow(m, cmap=LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE), vmin=0, vmax=1)
    for i in range(tok.N_BUCKETS):
        for j in range(tok.N_BUCKETS):
            if m[i, j] >= 0.1:
                ax.text(j, i, f"{m[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color=SURFACE if m[i, j] > 0.45 else INK)
    labels = [f"d{b}" for b in range(tok.N_BUCKETS)]
    ax.set_xticks(range(tok.N_BUCKETS), labels)
    ax.set_yticks(range(tok.N_BUCKETS), labels)
    ax.set_xlabel("Actual bucket (solved levels)", color=INK_2)
    ax.set_ylabel("Requested bucket", color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=lambda x, _: f"{x:.0%}")
    cbar.ax.tick_params(colors=INK_2, labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- CLI


def run(out: Path, cfg: EvalConfig, data_dir: Path, checkpoint: Path | None = None, levels: Path | None = None,
        device: str = "cuda", title: str | None = None) -> dict:
    if (checkpoint is None) == (levels is None):
        raise ValueError("pass exactly one of checkpoint or levels")
    out.mkdir(parents=True, exist_ok=True)
    edges = json.loads((data_dir / "buckets.json").read_text())["edges"]
    corpus = pq.read_table(data_dir / "train.parquet", columns=["text"])["text"].to_pylist()

    samples_per_sec = None
    if checkpoint is not None:
        records, samples_per_sec = sample_levels(checkpoint, cfg, device)
    else:
        records = [json.loads(line) for line in levels.read_text().splitlines() if line.strip()]

    start = time.perf_counter()
    rows = score_records(records, corpus, edges, cfg)
    summary = {
        "source": str(checkpoint or levels),
        "config": asdict(cfg),
        "samples_per_sec": samples_per_sec,
        "scoring_seconds": time.perf_counter() - start,
        **summarize_groups(rows, cfg.novelty_k),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "levels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    title = title or out.name
    if any(r["target"] is not None for r in rows):
        plot_calibration(rows, edges, out / "calibration.png", f"Difficulty calibration · {title}")
        plot_confusion(rows, out / "confusion.png", f"Requested vs. actual bucket · {title}")
    return summary


def main(argv: list[str] | None = None) -> None:
    import os

    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path)
    src.add_argument("--levels", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("BOXWRIGHT_DATA", "data/processed")))
    for field, default in asdict(EvalConfig()).items():
        ap.add_argument(f"--{field.replace('_', '-')}", type=type(default), default=default)
    args = ap.parse_args(argv)
    cfg = EvalConfig(**{f: getattr(args, f) for f in asdict(EvalConfig())})
    summary = run(args.out, cfg, args.data, args.checkpoint, args.levels, args.device, args.title)
    for group in ["conditional", "unconditional"]:
        print(group, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary[group].items()}))


if __name__ == "__main__":
    main()
