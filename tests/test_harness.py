import json
from pathlib import Path

from boxwright.data.symmetry import transform
from boxwright.eval.harness import EvalConfig, diversity, nearest_edit, score_records, summarize, summarize_groups

FIXTURE = Path(__file__).parent / "fixtures" / "boxoban_medium_valid_sample.json"
LEVELS = [c["text"] for c in json.loads(FIXTURE.read_text())]
EDGES = [21, 25, 28, 32, 35, 39, 44, 50, 59]


def test_nearest_edit_plain_vs_symmetric():
    corpus = LEVELS[:10]
    rotated = transform(LEVELS[0], 1)
    plain, sym = nearest_edit([LEVELS[0], rotated], corpus, False), nearest_edit([LEVELS[0], rotated], corpus, True)
    assert plain[0] == 0 and sym[0] == 0
    assert plain[1] > 0, "a rotation is far in raw edit distance"
    assert sym[1] == 0, "but identical under symmetry"


def test_diversity_greedy():
    a = LEVELS[0]
    near = a[:15] + ("#" if a[15] != "#" else " ") + a[16:]  # 1 edit from a
    assert diversity([a, near], k=5) == 1
    assert diversity(LEVELS[:5], k=5) == 5
    assert diversity([], k=5) == 0


def test_score_and_summarize():
    corpus = LEVELS[10:]
    records = [{"target": 3, "text": LEVELS[0]}, {"target": 3, "text": LEVELS[10]},
               {"target": None, "text": LEVELS[0].replace("@", " ")}, {"target": None, "text": None}]
    rows = score_records(records, corpus, EDGES, EvalConfig())
    assert [r["status"] for r in rows] == ["solved", "solved", "invalid", "undecodable"]
    assert rows[1]["nn_edit"] == 0, "copied from the corpus"
    s = summarize(rows, k=5)
    assert s["valid"] == 0.5 and s["playable"] == 0.5
    assert s["playable_novel"] == 0.25 and s["score"] == 0.25
    groups = summarize_groups(rows, k=5)
    assert groups["conditional"]["n"] == 2 and groups["unconditional"]["valid"] == 0.0
    assert "bucket_exact" in groups["conditional"]
