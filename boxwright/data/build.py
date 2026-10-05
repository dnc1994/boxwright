"""Build the Boxwright dataset from Boxoban.

    uv run python -m boxwright.data.build [--limit N]

Steps: label every level with the solver (cached per source split), deduplicate
across splits under the 8 grid symmetries, drop levels the solver could not
solve, assign difficulty buckets from training-set move-count quantiles, then
write Parquet files and a dataset card.

Outputs in data/processed/: {train,val,test,hard}.parquet, buckets.json,
and docs/dataset.md.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import boxwright_solver
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from boxwright.data.boxoban import read_split
from boxwright.data.symmetry import canonical

REPO = Path(__file__).resolve().parents[2]
BOXOBAN = REPO / "data" / "external" / "boxoban-levels"
OUT = REPO / "data" / "processed"
CARD = REPO / "docs" / "dataset.md"

MAX_NODES = 1_000_000
N_BUCKETS = 10

# Output split -> Boxoban source splits. Earlier output splits win deduplication,
# so held-out levels are never also in train.
SPLITS = {
    "test": ["unfiltered/test", "medium/valid"],
    "val": ["unfiltered/valid"],
    "hard": ["hard"],
    "train": ["unfiltered/train", "medium/train"],
}


def label_source(source: str, limit: int | None) -> pa.Table:
    """Solve every level of a source split, caching the result."""
    suffix = f"-limit{limit}" if limit else ""
    cache = OUT / "labels" / f"{source.replace('/', '_')}{suffix}.parquet"
    if cache.exists():
        return pq.read_table(cache)

    levels = read_split(BOXOBAN, source)[:limit]
    texts = [lvl.text for lvl in levels]
    results = []
    start = time.perf_counter()
    for i in range(0, len(texts), 20_000):
        results += boxwright_solver.solve_batch(texts[i : i + 20_000], max_nodes=MAX_NODES)
        rate = len(results) / (time.perf_counter() - start)
        print(f"  {source}: {len(results):,}/{len(texts):,} ({rate:,.0f}/s)", flush=True)

    table = pa.table(
        {
            "id": [f"{source}/{lvl.file:03d}#{lvl.index}" for lvl in levels],
            "source": [source] * len(levels),
            "text": texts,
            "canonical": [canonical(t) for t in texts],
            "status": [r.status for r in results],
            "moves": pa.array([r.moves for r in results], pa.int32()),
            "pushes": pa.array([r.pushes for r in results], pa.int32()),
            "nodes_expanded": pa.array([r.nodes_expanded for r in results], pa.int64()),
        }
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, cache)
    return table


def build(limit: int | None) -> dict:
    stats: dict = {"sources": {}, "splits": {}}
    seen: set[str] = set()
    splits: dict[str, pa.Table] = {}

    for split, sources in SPLITS.items():
        table = pa.concat_tables([label_source(s, limit) for s in sources])
        for source in sources:
            statuses = Counter(
                st for st, so in zip(table["status"].to_pylist(), table["source"].to_pylist()) if so == source
            )
            stats["sources"][source] = {"levels": sum(statuses.values()), "status": dict(statuses)}

        keep, duplicates, unsolved = [], 0, 0
        for i, (canon, status) in enumerate(zip(table["canonical"].to_pylist(), table["status"].to_pylist())):
            if canon in seen:
                duplicates += 1
            elif status != "solved":
                unsolved += 1
            else:
                keep.append(i)
            seen.add(canon)
        splits[split] = table.take(keep)
        stats["splits"][split] = {"kept": len(keep), "duplicates_removed": duplicates, "unsolved_removed": unsolved}

    train_moves = splits["train"]["moves"].to_numpy()
    edges = np.quantile(train_moves, np.arange(1, N_BUCKETS) / N_BUCKETS).tolist()
    stats["bucket_edges"] = edges

    for split, table in splits.items():
        moves = table["moves"].to_numpy()
        buckets = np.searchsorted(edges, moves, side="right")
        table = table.append_column("bucket", pa.array(buckets, pa.int8()))
        pq.write_table(table, OUT / f"{split}.parquet")
        s = stats["splits"][split]
        s["moves_percentiles"] = dict(zip(["min", "p10", "p50", "p90", "max"], np.percentile(moves, [0, 10, 50, 90, 100]).astype(int).tolist()))
        s["pushes_median"] = int(np.median(table["pushes"].to_numpy()))
        s["nodes_median"] = int(np.median(table["nodes_expanded"].to_numpy()))
        s["bucket_counts"] = np.bincount(buckets, minlength=N_BUCKETS).tolist()

    (OUT / "buckets.json").write_text(json.dumps({"edges": edges, "n_buckets": N_BUCKETS, "max_nodes": MAX_NODES}, indent=2))
    return stats


def bucket_ranges(edges: list[float]) -> list[str]:
    """Human-readable move ranges, e.g. '≤ 18', '19–24', '≥ 61'."""
    bounds = [int(np.floor(e)) for e in edges]  # bucket k holds moves in (bounds[k-1], bounds[k]]
    ranges = [f"≤ {bounds[0]}"]
    ranges += [f"{lo + 1}–{hi}" if hi > lo + 1 else f"{hi}" for lo, hi in zip(bounds, bounds[1:])]
    return ranges + [f"≥ {bounds[-1] + 1}"]


def write_card(stats: dict) -> None:
    lines = [
        "# Boxwright dataset",
        "",
        "Generated by `uv run python -m boxwright.data.build` from "
        "[Boxoban](https://github.com/google-deepmind/boxoban-levels) (10×10, 4 boxes). "
        f"Every level is labeled by our solver (move-optimal A*, {MAX_NODES:,}-node budget).",
        "",
        "## Sources",
        "",
        "| Boxoban split | Levels | Solved | Not solved within budget |",
        "|---|---:|---:|---:|",
    ]
    for source, s in stats["sources"].items():
        solved = s["status"].get("solved", 0)
        lines.append(f"| {source} | {s['levels']:,} | {solved:,} | {s['levels'] - solved:,} |")

    lines += [
        "",
        "## Splits",
        "",
        "Deduplicated under the 8 rotations/reflections, in priority order test → val → hard → train, "
        "so no held-out level (or any symmetric copy) appears in train. Levels not solved within budget are dropped.",
        "",
        "| Split | Sources | Levels | Duplicates removed | Unsolved removed | Moves (min / p10 / median / p90 / max) | Median pushes | Median nodes |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for split, s in stats["splits"].items():
        m = s["moves_percentiles"]
        lines.append(
            f"| {split} | {', '.join(SPLITS[split])} | {s['kept']:,} | {s['duplicates_removed']:,} | {s['unsolved_removed']:,} "
            f"| {m['min']} / {m['p10']} / {m['p50']} / {m['p90']} / {m['max']} | {s['pushes_median']} | {s['nodes_median']:,} |"
        )

    ranges = bucket_ranges(stats["bucket_edges"])
    lines += [
        "",
        "## Difficulty buckets",
        "",
        f"`<d0>`…`<d{N_BUCKETS - 1}>` split the training set's optimal move counts at deciles. "
        "Ties make bucket sizes uneven. The same edges apply to every split (`data/processed/buckets.json`).",
        "",
        "| Bucket | Moves | " + " | ".join(stats["splits"]) + " |",
        "|---|---|" + "---:|" * len(stats["splits"]),
    ]
    for k in range(N_BUCKETS):
        counts = " | ".join(f"{s['bucket_counts'][k]:,}" for s in stats["splits"].values())
        lines.append(f"| d{k} | {ranges[k]} | {counts} |")

    lines += [
        "",
        "## Augmentation",
        "",
        "Not stored. Training applies one of the 8 grid symmetries at load time "
        "(`boxwright.data.symmetry.transform`), which preserves solvability and move count.",
        "",
    ]
    CARD.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="levels per source split (for quick runs)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = build(args.limit)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    if args.limit is None:
        write_card(stats)
    print(json.dumps(stats["splits"], indent=2))


if __name__ == "__main__":
    main()
