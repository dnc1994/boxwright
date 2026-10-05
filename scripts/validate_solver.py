"""Check the solver against AlignmentResearch's A* labels and measure throughput.

Usage: uv run scripts/validate_solver.py [--split medium/valid] [--limit N]
Expects data/external/boxoban-levels and data/external/<split>.csv.gz.
"""

import argparse
import csv
import gzip
import os
import time
from pathlib import Path

import boxwright_solver

from boxwright.data.boxoban import read_split

ROOT = Path(__file__).resolve().parents[1] / "data" / "external"
LURD = "urdl"  # label action codes 0..3 = up, right, down, left


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="medium/valid")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=1_000_000)
    args = ap.parse_args()

    levels = read_split(ROOT / "boxoban-levels", args.split)[: args.limit]
    labels = {}
    with gzip.open(ROOT / f"{args.split.replace('/', '_')}.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            labels[(int(row["File"]), int(row["Level"]))] = row

    start = time.perf_counter()
    results = boxwright_solver.solve_batch([l.text for l in levels], max_nodes=args.max_nodes)
    elapsed = time.perf_counter() - start

    statuses: dict[str, int] = {}
    agree = disagree = label_unsolved = invalid_paths = 0
    examples = []
    for lvl, sol in zip(levels, results):
        statuses[sol.status] = statuses.get(sol.status, 0) + 1
        if sol.solved and not boxwright_solver.verify(lvl.text, sol.path):
            invalid_paths += 1
        label = labels.get((lvl.file, lvl.index))
        if label is None or not label["Steps"].isdigit():
            label_unsolved += 1
            continue
        if not sol.solved:
            continue
        if int(label["Steps"]) == sol.moves:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append((lvl.file, lvl.index, int(label["Steps"]), sol.moves))

    n = len(levels)
    print(f"levels: {n}  cores: {os.cpu_count()}  time: {elapsed:.1f}s  ({n / elapsed:,.0f} levels/s)")
    print("status:", {k: f"{v} ({v / n:.2%})" for k, v in sorted(statuses.items())})
    print(f"paths failing verification: {invalid_paths}")
    print(f"move counts vs labels: agree {agree}, disagree {disagree}, label unsolved/missing {label_unsolved}")
    for file, index, theirs, ours in examples:
        print(f"  {file:03d}#{index}: label {theirs} moves, ours {ours}")
    nodes = sorted(s.nodes_expanded for s in results)
    print(f"nodes expanded: median {nodes[n // 2]:,}, p99 {nodes[int(n * 0.99)]:,}, max {nodes[-1]:,}")


if __name__ == "__main__":
    main()
