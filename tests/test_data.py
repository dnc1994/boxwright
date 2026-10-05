import json
from pathlib import Path

import boxwright_solver as solver
import pytest

from boxwright.data.build import bucket_ranges
from boxwright.data.novelty import NoveltyIndex
from boxwright.data.symmetry import canonical, orbit, transform

FIXTURE = Path(__file__).parent / "fixtures" / "boxoban_medium_valid_sample.json"
ASYMMETRIC = "#####\n#@$.#\n#  ##\n#####"  # 4 rows x 5 cols


def test_transforms_form_the_dihedral_group():
    images = orbit(ASYMMETRIC)
    assert len(set(images)) == 8, "an asymmetric level has 8 distinct images"
    assert images[0] == ASYMMETRIC
    # Four clockwise rotations return to the start; mirroring twice is the identity.
    assert transform(transform(transform(transform(ASYMMETRIC, 1), 1), 1), 1) == ASYMMETRIC
    assert transform(transform(ASYMMETRIC, 4), 4) == ASYMMETRIC
    # Rotation swaps width and height.
    assert len(transform(ASYMMETRIC, 1).split("\n")) == 5


def test_canonical_is_shared_by_the_orbit():
    assert {canonical(t) for t in orbit(ASYMMETRIC)} == {canonical(ASYMMETRIC)}


def test_symmetry_preserves_optimal_moves():
    for case in json.loads(FIXTURE.read_text())[:10]:
        results = solver.solve_batch(orbit(case["text"]))
        assert {r.moves for r in results} == {case["moves"]}


def test_transform_rejects_bad_input():
    with pytest.raises(ValueError):
        transform(ASYMMETRIC, 8)
    with pytest.raises(ValueError):
        transform("###\n#", 1)


def test_novelty_index_sees_through_symmetry():
    index = NoveltyIndex({canonical(ASYMMETRIC)})
    assert not index.is_novel(transform(ASYMMETRIC, 6))
    assert index.is_novel(ASYMMETRIC.replace("@", " ").replace("  #", "@ #"))


def test_bucket_ranges():
    assert bucket_ranges([10.0, 15.0, 16.0]) == ["≤ 10", "11–15", "16", "≥ 17"]
