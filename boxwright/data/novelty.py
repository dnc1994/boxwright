"""Symmetry-aware membership test against the training set."""

from pathlib import Path

import pyarrow.parquet as pq

from boxwright.data.symmetry import canonical


class NoveltyIndex:
    def __init__(self, canonical_levels: set[str]):
        self._seen = canonical_levels

    @classmethod
    def from_parquet(cls, path: Path) -> "NoveltyIndex":
        column = pq.read_table(path, columns=["canonical"]).column("canonical")
        return cls(set(column.to_pylist()))

    def __len__(self) -> int:
        return len(self._seen)

    def is_novel(self, text: str) -> bool:
        """True if no rotation or reflection of `text` is in the index.

        `text` must be rectangular; ragged or malformed levels raise ValueError.
        """
        return canonical(text) not in self._seen
