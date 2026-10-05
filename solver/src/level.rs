//! Level parsing, validation, rendering and move simulation.
//!
//! Cells are indexed row-major (`r * width + c`). Boxes, goals and walls are
//! stored as `u128` bitsets, so a level may have at most 128 cells, enough for
//! Boxoban (10×10). Anything outside the grid counts as wall.

use std::fmt;

pub const MAX_CELLS: usize = 128;

/// Directions in Boxoban / LURD order used by the A* labels: up, right, down, left.
pub const DIRS: [Dir; 4] = [Dir::Up, Dir::Right, Dir::Down, Dir::Left];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Right,
    Down,
    Left,
}

impl Dir {
    pub fn index(self) -> usize {
        self as usize
    }

    /// LURD character: lowercase for a walk, uppercase for a push.
    pub fn lurd(self, push: bool) -> char {
        let c = match self {
            Dir::Up => 'u',
            Dir::Right => 'r',
            Dir::Down => 'd',
            Dir::Left => 'l',
        };
        if push { c.to_ascii_uppercase() } else { c }
    }

    pub fn from_lurd(c: char) -> Option<Dir> {
        match c.to_ascii_lowercase() {
            'u' => Some(Dir::Up),
            'r' => Some(Dir::Right),
            'd' => Some(Dir::Down),
            'l' => Some(Dir::Left),
            _ => None,
        }
    }
}

#[inline]
pub fn has(bits: u128, i: usize) -> bool {
    bits >> i & 1 == 1
}

#[inline]
pub fn bit(i: usize) -> u128 {
    1u128 << i
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level {
    pub width: usize,
    pub height: usize,
    pub walls: u128,
    pub goals: u128,
    pub boxes: u128,
    pub player: usize,
    /// `neighbors[cell][dir]`: the adjacent cell, or `None` if off the grid.
    neighbors: Vec<[Option<u8>; 4]>,
}

impl Level {
    /// Parses a level in standard Sokoban text (`#` wall, space/`-`/`_` floor,
    /// `@` player, `+` player on goal, `$` box, `*` box on goal, `.` goal).
    ///
    /// Ragged rows are padded with floor; blank leading/trailing lines are ignored.
    /// Requires exactly one player and as many boxes as goals (at least one).
    pub fn parse(text: &str) -> Result<Level, ParseError> {
        let rows: Vec<&str> = text
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .skip_while(|l| l.trim().is_empty())
            .collect();
        let rows: Vec<&str> = {
            let end = rows.iter().rposition(|l| !l.trim().is_empty()).map_or(0, |i| i + 1);
            rows[..end].to_vec()
        };
        let height = rows.len();
        let width = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
        if height == 0 || width == 0 {
            return Err(ParseError("empty level".into()));
        }
        if width * height > MAX_CELLS {
            return Err(ParseError(format!(
                "level is {width}x{height}; at most {MAX_CELLS} cells are supported"
            )));
        }

        let (mut walls, mut goals, mut boxes) = (0u128, 0u128, 0u128);
        let mut players = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            for (c, ch) in row.chars().enumerate() {
                let i = r * width + c;
                match ch {
                    '#' => walls |= bit(i),
                    ' ' | '-' | '_' => {}
                    '.' => goals |= bit(i),
                    '$' => boxes |= bit(i),
                    '*' => {
                        boxes |= bit(i);
                        goals |= bit(i);
                    }
                    '@' => players.push(i),
                    '+' => {
                        players.push(i);
                        goals |= bit(i);
                    }
                    _ => return Err(ParseError(format!("unknown tile {ch:?} at row {r}, col {c}"))),
                }
            }
        }

        if players.len() != 1 {
            return Err(ParseError(format!("expected 1 player, found {}", players.len())));
        }
        let (n_boxes, n_goals) = (boxes.count_ones(), goals.count_ones());
        if n_boxes == 0 {
            return Err(ParseError("level has no boxes".into()));
        }
        if n_boxes != n_goals {
            return Err(ParseError(format!("{n_boxes} boxes but {n_goals} goals")));
        }

        let neighbors = (0..width * height)
            .map(|i| {
                let (r, c) = (i / width, i % width);
                [
                    (r > 0).then(|| (i - width) as u8),
                    (c + 1 < width).then(|| (i + 1) as u8),
                    (r + 1 < height).then(|| (i + width) as u8),
                    (c > 0).then(|| (i - 1) as u8),
                ]
            })
            .collect();

        Ok(Level { width, height, walls, goals, boxes, player: players[0], neighbors })
    }

    pub fn cells(&self) -> usize {
        self.width * self.height
    }

    /// The cell one step from `cell` in `dir`, or `None` if that is off the grid.
    #[inline]
    pub fn step(&self, cell: usize, dir: Dir) -> Option<usize> {
        self.neighbors[cell][dir.index()].map(usize::from)
    }

    /// Like [`Level::step`], but also `None` if the target is a wall.
    #[inline]
    pub fn open_step(&self, cell: usize, dir: Dir) -> Option<usize> {
        self.step(cell, dir).filter(|&n| !has(self.walls, n))
    }

    pub fn is_solved(&self) -> bool {
        self.boxes == self.goals
    }

    /// Renders back to standard text, one row per line.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(self.cells() + self.height);
        for r in 0..self.height {
            for c in 0..self.width {
                let i = r * self.width + c;
                let (w, g, b, p) =
                    (has(self.walls, i), has(self.goals, i), has(self.boxes, i), i == self.player);
                out.push(match (w, g, b, p) {
                    (true, ..) => '#',
                    (_, true, true, _) => '*',
                    (_, true, _, true) => '+',
                    (_, true, ..) => '.',
                    (_, _, true, _) => '$',
                    (.., true) => '@',
                    _ => ' ',
                });
            }
            out.push('\n');
        }
        out
    }

    /// Applies one LURD move in place. Returns false (leaving the level unchanged)
    /// if the move is illegal or its case disagrees with whether it pushes.
    pub fn apply(&mut self, c: char) -> bool {
        let Some(dir) = Dir::from_lurd(c) else { return false };
        let Some(next) = self.open_step(self.player, dir) else { return false };
        let pushes = has(self.boxes, next);
        if pushes != c.is_ascii_uppercase() {
            return false;
        }
        if pushes {
            match self.open_step(next, dir) {
                Some(dest) if !has(self.boxes, dest) => {
                    self.boxes = self.boxes & !bit(next) | bit(dest);
                }
                _ => return false,
            }
        }
        self.player = next;
        true
    }

    /// True if `path` (LURD) is legal from this position and ends solved.
    pub fn verify(&self, path: &str) -> bool {
        let mut lvl = self.clone();
        path.chars().all(|c| lvl.apply(c)) && lvl.is_solved()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = "#####\n#@$.#\n#####";

    #[test]
    fn parse_and_render_round_trip() {
        let text = "#######\n#.@ $ #\n#  *  #\n#######\n";
        let lvl = Level::parse(text).unwrap();
        assert_eq!(lvl.render(), text);
        assert_eq!((lvl.width, lvl.height), (7, 4));
    }

    #[test]
    fn rejects_invalid_levels() {
        assert!(Level::parse("").is_err());
        assert!(Level::parse("#####\n# $.#\n#####").is_err(), "no player");
        assert!(Level::parse("#####\n#@@$.#\n#####").is_err(), "two players");
        assert!(Level::parse("#####\n#@$ #\n#####").is_err(), "box without goal");
        assert!(Level::parse("####\n#@ #\n####").is_err(), "no boxes");
        assert!(Level::parse("#####\n#@$x#\n#####").is_err(), "unknown tile");
        assert!(Level::parse(&"#".repeat(129)).is_err(), "too large");
    }

    #[test]
    fn ragged_rows_are_padded() {
        let lvl = Level::parse("####\n#@$.#\n####").unwrap();
        assert_eq!(lvl.width, 5);
    }

    #[test]
    fn apply_and_verify() {
        let lvl = Level::parse(SIMPLE).unwrap();
        assert!(lvl.verify("R"));
        assert!(!lvl.verify("r"), "lowercase must not push");
        assert!(!lvl.verify(""), "not solved");
        assert!(!lvl.verify("L"), "into a wall");
        assert!(!lvl.verify("RR"), "box into a wall");
    }
}
