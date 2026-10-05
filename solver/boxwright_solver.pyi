from typing import Literal

class Solution:
    status: Literal["solved", "unsolvable", "budget_exceeded", "invalid"]
    moves: int
    pushes: int
    path: str
    nodes_expanded: int
    error: str | None
    @property
    def solved(self) -> bool: ...

def solve(level: str, max_nodes: int = 1_000_000) -> Solution: ...
def solve_batch(levels: list[str], max_nodes: int = 1_000_000, threads: int | None = None) -> list[Solution]: ...
def verify(level: str, path: str) -> bool: ...
