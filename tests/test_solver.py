import json
from pathlib import Path

import boxwright_solver as solver

FIXTURE = Path(__file__).parent / "fixtures" / "boxoban_medium_valid_sample.json"
SIMPLE = "#####\n#@$.#\n#####"


def test_matches_reference_move_counts():
    """Move counts must equal the AlignmentResearch A* labels (move-optimal)."""
    cases = json.loads(FIXTURE.read_text())
    results = solver.solve_batch([c["text"] for c in cases])
    for case, sol in zip(cases, results):
        assert sol.solved
        assert sol.moves == case["moves"]
        assert solver.verify(case["text"], sol.path)


def test_statuses():
    assert solver.solve(SIMPLE).status == "solved"
    assert solver.solve("#####\n#$ .#\n# @ #\n#####").status == "unsolvable"
    hard = json.loads(FIXTURE.read_text())[0]["text"]
    assert solver.solve(hard, max_nodes=1).status == "budget_exceeded"


def test_invalid_levels_do_not_raise():
    sol = solver.solve("#####\n#@$ #\n#####")
    assert sol.status == "invalid" and not sol.solved
    assert "goals" in sol.error


def test_batch_preserves_order_and_threads():
    levels = [SIMPLE, "garbage", "######\n#@ $.#\n######"]
    statuses = [s.status for s in solver.solve_batch(levels, threads=2)]
    assert statuses == ["solved", "invalid", "solved"]


def test_verify():
    assert solver.verify(SIMPLE, "R")
    assert not solver.verify(SIMPLE, "r")
