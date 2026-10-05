//! Move-optimal A* over pushes.
//!
//! Each search edge is one push. Its cost is the player's walk to the pushing
//! position plus the push itself, so the returned solution is optimal in total
//! moves. The state is (box set, exact player cell), and the exact player cell is
//! always the square the last pushed box came from.
//!
//! The heuristic is a minimum-cost assignment of boxes to goals, using per-goal
//! push distances on an empty board. Every push costs at least one move and
//! changes one box's distance by at most one, so the heuristic is consistent and
//! closed states never need reopening.
//!
//! Pruning (sound: never discards a solvable state):
//! - dead squares: non-goal cells from which a box can never reach any goal;
//! - 2×2 freeze: a pushed box completing a 2×2 block of walls/boxes that holds
//!   a box off its goal;
//! - an infeasible assignment (some box cannot reach any unclaimed goal).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use crate::level::{DIRS, Dir, Level, bit, has};

const UNREACHABLE: u16 = u16::MAX;
const INFEASIBLE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Solved,
    /// The whole reachable state space was searched without finding a solution.
    Unsolvable,
    /// The node budget ran out first; solvability is unknown.
    BudgetExceeded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    pub status: Status,
    /// Total moves of the optimal solution (0 unless solved).
    pub moves: u32,
    /// Pushes within that move-optimal solution (0 unless solved).
    pub pushes: u32,
    /// LURD path: lowercase walks, uppercase pushes (empty unless solved).
    pub path: String,
    /// States expanded, a rough measure of search difficulty.
    pub nodes_expanded: u64,
}

/// Static tables derived from the level layout (walls and goals only).
struct Tables {
    /// `goal_dist[g][cell]`: pushes needed to move a lone box from `cell` to goal `g`.
    goal_dist: Vec<Vec<u16>>,
    /// Cells where a box can still reach some goal.
    live: u128,
}

impl Tables {
    fn new(level: &Level) -> Tables {
        let goals: Vec<usize> = (0..level.cells()).filter(|&i| has(level.goals, i)).collect();
        let goal_dist: Vec<Vec<u16>> = goals.iter().map(|&g| pull_distances(level, g)).collect();
        let live = (0..level.cells())
            .filter(|&i| goal_dist.iter().any(|d| d[i] != UNREACHABLE))
            .fold(0u128, |acc, i| acc | bit(i));
        Tables { goal_dist, live }
    }

    /// Minimum-cost box→goal assignment, or `INFEASIBLE`.
    fn heuristic(&self, boxes: u128, scratch: &mut Vec<i64>) -> u32 {
        let n = self.goal_dist.len();
        scratch.clear();
        let mut b = boxes;
        while b != 0 {
            let cell = b.trailing_zeros() as usize;
            b &= b - 1;
            scratch.extend(self.goal_dist.iter().map(|d| match d[cell] {
                UNREACHABLE => BIG,
                v => v as i64,
            }));
        }
        let cost = min_assignment(scratch, n);
        if cost >= BIG { INFEASIBLE } else { cost as u32 }
    }
}

/// Reverse BFS from a goal: how many pushes a lone box needs from each cell.
fn pull_distances(level: &Level, goal: usize) -> Vec<u16> {
    let mut dist = vec![UNREACHABLE; level.cells()];
    let mut queue = std::collections::VecDeque::from([goal]);
    dist[goal] = 0;
    while let Some(t) = queue.pop_front() {
        for dir in DIRS {
            // A push in `dir` moves the box from `b` to `t` with the player on `p`.
            let back = opposite(dir);
            let Some(b) = level.open_step(t, back) else { continue };
            if level.open_step(b, back).is_none() || dist[b] != UNREACHABLE {
                continue;
            }
            dist[b] = dist[t] + 1;
            queue.push_back(b);
        }
    }
    dist
}

fn opposite(dir: Dir) -> Dir {
    DIRS[(dir.index() + 2) % 4]
}

/// True if the box just pushed onto `cell` completes a 2×2 block of walls/boxes
/// that contains a box not on a goal.
fn freeze_deadlock(level: &Level, boxes: u128, cell: usize) -> bool {
    let solid = |i: Option<usize>| i.is_none_or(|i| has(level.walls, i) || has(boxes, i));
    let off_goal = |i: Option<usize>| i.is_some_and(|i| has(boxes, i) && !has(level.goals, i));
    // The four 2×2 squares containing `cell`, each given as (vertical, horizontal) offsets.
    for (v, h) in [(Dir::Up, Dir::Left), (Dir::Up, Dir::Right), (Dir::Down, Dir::Left), (Dir::Down, Dir::Right)] {
        let a = level.step(cell, v);
        let b = level.step(cell, h);
        let c = a.and_then(|a| level.step(a, h));
        let square = [Some(cell), a, b, c];
        if square.iter().all(|&i| solid(i)) && square.iter().any(|&i| off_goal(i)) {
            return true;
        }
    }
    false
}

const BIG: i64 = 1 << 40;

/// Hungarian algorithm (O(n³)) on a row-major `n×n` cost matrix.
fn min_assignment(cost: &[i64], n: usize) -> i64 {
    let inf = i64::MAX / 4;
    let (mut u, mut v) = (vec![0i64; n + 1], vec![0i64; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let (i0, mut delta, mut j1) = (p[j0], inf, 0);
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    (1..=n).map(|j| cost[(p[j] - 1) * n + (j - 1)]).sum()
}

struct Node {
    boxes: u128,
    player: u8,
    parent: u32,
    g: u32,
}

/// Walking distances from `start`, avoiding walls and boxes.
fn walk_distances(level: &Level, boxes: u128, start: usize, dist: &mut Vec<u16>, queue: &mut Vec<usize>) {
    dist.clear();
    dist.resize(level.cells(), UNREACHABLE);
    queue.clear();
    dist[start] = 0;
    queue.push(start);
    let mut head = 0;
    while head < queue.len() {
        let c = queue[head];
        head += 1;
        for dir in DIRS {
            if let Some(n) = level.open_step(c, dir)
                && !has(boxes, n)
                && dist[n] == UNREACHABLE
            {
                dist[n] = dist[c] + 1;
                queue.push(n);
            }
        }
    }
}

/// Searches for a move-optimal solution, expanding at most `max_nodes` states.
pub fn solve(level: &Level, max_nodes: u64) -> Solution {
    let unsolved = |status, nodes_expanded| Solution {
        status,
        moves: 0,
        pushes: 0,
        path: String::new(),
        nodes_expanded,
    };
    let tables = Tables::new(level);
    let mut scratch = Vec::new();
    // Initial boxes on dead squares (or an infeasible assignment) mean no solution.
    let h0 = tables.heuristic(level.boxes, &mut scratch);
    if level.boxes & !level.goals & !tables.live != 0 || h0 == INFEASIBLE {
        return unsolved(Status::Unsolvable, 0);
    }

    let mut nodes = vec![Node { boxes: level.boxes, player: level.player as u8, parent: u32::MAX, g: 0 }];
    let mut best: FxHashMap<(u128, u8), u32> = FxHashMap::default();
    best.insert((level.boxes, level.player as u8), 0);
    // Min-heap on f; ties prefer larger g (deeper), then insertion order.
    let mut open = BinaryHeap::from([(Reverse(h0), 0u32, Reverse(0u32))]);
    let (mut dist, mut queue) = (Vec::new(), Vec::new());
    let mut expanded = 0u64;

    while let Some((_, g, Reverse(idx))) = open.pop() {
        let (boxes, player) = (nodes[idx as usize].boxes, nodes[idx as usize].player);
        if best.get(&(boxes, player)) != Some(&g) {
            continue; // stale heap entry
        }
        if boxes == level.goals {
            return reconstruct(level, &nodes, idx, expanded);
        }
        if expanded >= max_nodes {
            return unsolved(Status::BudgetExceeded, expanded);
        }
        expanded += 1;

        walk_distances(level, boxes, player as usize, &mut dist, &mut queue);
        let mut bs = boxes;
        while bs != 0 {
            let b = bs.trailing_zeros() as usize;
            bs &= bs - 1;
            for dir in DIRS {
                let Some(stand) = level.open_step(b, opposite(dir)) else { continue };
                if dist[stand] == UNREACHABLE {
                    continue;
                }
                let Some(t) = level.open_step(b, dir) else { continue };
                if has(boxes, t) || !has(tables.live, t) {
                    continue;
                }
                let next_boxes = boxes & !bit(b) | bit(t);
                if freeze_deadlock(level, next_boxes, t) {
                    continue;
                }
                let next_g = g + dist[stand] as u32 + 1;
                let key = (next_boxes, b as u8);
                if best.get(&key).is_some_and(|&old| old <= next_g) {
                    continue;
                }
                let h = tables.heuristic(next_boxes, &mut scratch);
                if h == INFEASIBLE {
                    continue;
                }
                best.insert(key, next_g);
                let next_idx = nodes.len() as u32;
                nodes.push(Node { boxes: next_boxes, player: b as u8, parent: idx, g: next_g });
                open.push((Reverse(next_g + h), next_g, Reverse(next_idx)));
            }
        }
    }
    unsolved(Status::Unsolvable, expanded)
}

/// Rebuilds the LURD path by replaying each push and walking between them.
fn reconstruct(level: &Level, nodes: &[Node], goal: u32, expanded: u64) -> Solution {
    let mut chain = vec![goal];
    while nodes[*chain.last().unwrap() as usize].parent != u32::MAX {
        chain.push(nodes[*chain.last().unwrap() as usize].parent);
    }
    chain.reverse();

    let mut path = String::new();
    let (mut dist, mut queue) = (Vec::new(), Vec::new());
    for pair in chain.windows(2) {
        let (from, to) = (&nodes[pair[0] as usize], &nodes[pair[1] as usize]);
        let old = (from.boxes & !to.boxes).trailing_zeros() as usize;
        let new = (to.boxes & !from.boxes).trailing_zeros() as usize;
        let dir = DIRS.into_iter().find(|&d| level.step(old, d) == Some(new)).unwrap();
        let stand = level.step(old, opposite(dir)).unwrap();

        // Walk from the push target backwards along decreasing distance.
        walk_distances(level, from.boxes, from.player as usize, &mut dist, &mut queue);
        let mut walk = Vec::new();
        let mut cur = stand;
        while cur != from.player as usize {
            let (prev, d) = DIRS
                .into_iter()
                .filter_map(|d| level.step(cur, opposite(d)).map(|p| (p, d)))
                .find(|&(p, _)| dist[p] != UNREACHABLE && dist[p] + 1 == dist[cur])
                .unwrap();
            walk.push(d.lurd(false));
            cur = prev;
        }
        path.extend(walk.iter().rev());
        path.push(dir.lurd(true));
    }

    let moves = path.len() as u32;
    let pushes = path.chars().filter(char::is_ascii_uppercase).count() as u32;
    debug_assert_eq!(moves, nodes[goal as usize].g);
    Solution { status: Status::Solved, moves, pushes, path, nodes_expanded: expanded }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve_text(text: &str) -> Solution {
        solve(&Level::parse(text).unwrap(), 1_000_000)
    }

    #[test]
    fn one_push() {
        let s = solve_text("#####\n#@$.#\n#####");
        assert_eq!((s.status, s.moves, s.pushes, s.path.as_str()), (Status::Solved, 1, 1, "R"));
    }

    #[test]
    fn walk_then_push() {
        let s = solve_text("######\n#@ $.#\n######");
        assert_eq!((s.moves, s.path.as_str()), (2, "rR"));
    }

    #[test]
    fn already_solved() {
        let s = solve_text("####\n#@*#\n####");
        assert_eq!((s.status, s.moves), (Status::Solved, 0));
    }

    #[test]
    fn box_in_corner_is_unsolvable() {
        let s = solve_text("#####\n#$ .#\n# @ #\n#####");
        assert_eq!(s.status, Status::Unsolvable);
        assert_eq!(s.nodes_expanded, 0, "rejected by dead-square check before search");
    }

    #[test]
    fn freeze_against_wall_is_pruned() {
        // Pushing the left box up makes a 2×2 block with the wall row: never optimal or solvable.
        let s = solve_text("#######\n#  .  #\n# $$. #\n#  @  #\n#######");
        assert_eq!(s.status, Status::Solved);
        assert!(Level::parse("#######\n#  .  #\n# $$. #\n#  @  #\n#######").unwrap().verify(&s.path));
    }

    #[test]
    fn budget_exceeded() {
        let level = "##########\n#   #    #\n# $ # $  #\n#  .  .  #\n# $ # $  #\n#.  #  . #\n#  @#    #\n##########";
        let s = solve(&Level::parse(level).unwrap(), 1);
        assert_eq!(s.status, Status::BudgetExceeded);
    }

    #[test]
    fn assignment_matches_brute_force() {
        let cost = [4, 1, 3, 2, 0, 5, 3, 2, 2];
        assert_eq!(min_assignment(&cost, 3), 5);
        let infeasible = [BIG, 1, BIG, 2];
        assert!(min_assignment(&infeasible, 2) >= BIG);
    }

    #[test]
    fn solutions_are_move_optimal() {
        // Exhaustive move-level BFS on small levels must agree with A*.
        for text in [
            "#######\n#  .  #\n# $$. #\n#  @  #\n#######",
            "########\n#.  $ .#\n# ## $ #\n#  @   #\n########",
            "#######\n#.$ @ #\n# $   #\n#.    #\n#######",
        ] {
            let level = Level::parse(text).unwrap();
            let s = solve(&level, 1_000_000);
            assert!(level.verify(&s.path), "{text}");
            assert_eq!(Some(s.moves), bfs_moves(&level), "{text}");
        }
    }

    /// Brute-force move-level BFS, for testing only.
    fn bfs_moves(level: &Level) -> Option<u32> {
        let mut seen = std::collections::HashSet::from([(level.boxes, level.player)]);
        let mut frontier = vec![level.clone()];
        for depth in 0.. {
            if frontier.is_empty() {
                return None;
            }
            let mut next = Vec::new();
            for lvl in frontier {
                if lvl.is_solved() {
                    return Some(depth);
                }
                for dir in DIRS {
                    for push in [false, true] {
                        let mut l = lvl.clone();
                        if l.apply(dir.lurd(push)) && seen.insert((l.boxes, l.player)) {
                            next.push(l);
                        }
                    }
                }
            }
            frontier = next;
        }
        unreachable!()
    }
}
