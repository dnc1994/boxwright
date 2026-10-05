"""Reading Boxoban level files (google-deepmind/boxoban-levels)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoxobanLevel:
    split: str  # e.g. "medium/valid"
    file: int  # file number, e.g. 0 for 000.txt
    index: int  # level number within the file
    text: str  # 10 rows joined by "\n"


def read_file(path: Path, split: str) -> list[BoxobanLevel]:
    """Parse one Boxoban file: blocks of '; <n>' followed by grid rows."""
    levels = []
    for block in path.read_text().split(";")[1:]:
        header, *rows = block.strip("\n").split("\n")
        rows = [r for r in rows if r]
        levels.append(BoxobanLevel(split, int(path.stem), int(header), "\n".join(rows)))
    return levels


def read_split(root: Path, split: str) -> list[BoxobanLevel]:
    """All levels of a split such as "medium/valid", ordered by file then index."""
    return [lvl for f in sorted((root / split).glob("*.txt")) for lvl in read_file(f, split)]
