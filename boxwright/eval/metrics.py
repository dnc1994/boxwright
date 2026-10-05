"""Scoring generated levels with the solver.

Lightweight core used during training; the full eval harness (Task 6) builds on it.
"""

from dataclasses import dataclass

import boxwright_solver
import numpy as np

from boxwright import tokenizer as tok
from boxwright.data.novelty import NoveltyIndex

EVAL_MAX_NODES = 200_000


@dataclass
class LevelScore:
    text: str | None  # None if the tokens did not decode to a 10×10 grid
    target: int | None  # requested bucket
    valid: bool
    solved: bool
    novel: bool
    moves: int
    bucket: int | None  # actual difficulty bucket if solved


def bucket_of(moves: int, edges: list[float]) -> int:
    return int(np.searchsorted(edges, moves, side="right"))


def score(
    texts: list[str | None],
    targets: list[int | None],
    edges: list[float],
    novelty: NoveltyIndex,
    max_nodes: int = EVAL_MAX_NODES,
) -> list[LevelScore]:
    valid = [t is not None and tok.is_valid(t) for t in texts]
    to_solve = [t for t, v in zip(texts, valid) if v]
    solutions = iter(boxwright_solver.solve_batch(to_solve, max_nodes=max_nodes))
    scores = []
    for text, target, ok in zip(texts, targets, valid):
        sol = next(solutions) if ok else None
        solved = bool(sol and sol.solved)
        scores.append(
            LevelScore(
                text=text,
                target=target,
                valid=ok,
                solved=solved,
                novel=ok and novelty.is_novel(text),
                moves=sol.moves if solved else 0,
                bucket=bucket_of(sol.moves, edges) if solved else None,
            )
        )
    return scores


def summarize(scores: list[LevelScore]) -> dict[str, float]:
    n = len(scores)
    solved = [s for s in scores if s.solved]
    targeted = [s for s in solved if s.target is not None]
    out = {
        "valid": sum(s.valid for s in scores) / n,
        "solvable": len(solved) / n,
        "novel_solvable": sum(s.solved and s.novel for s in scores) / n,
        "unique": len({s.text for s in scores if s.valid}) / n,
    }
    if targeted:
        hits = [s.bucket == s.target for s in targeted]
        near = [abs(s.bucket - s.target) <= 1 for s in targeted]
        out["bucket_exact"] = float(np.mean(hits))
        out["bucket_within1"] = float(np.mean(near))
        if len({s.target for s in targeted}) > 1:
            out["target_moves_spearman"] = _spearman([s.target for s in targeted], [s.moves for s in targeted])
    return out


def _spearman(x: list[float], y: list[float]) -> float:
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    return float(np.corrcoef(rx, ry)[0, 1])
