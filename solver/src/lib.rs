//! Sokoban solver for Boxwright: move-optimal A* with deadlock pruning.
//!
//! The same core is used for data labeling and RL rewards (via the `python`
//! feature) and in the browser demo (compiled to WASM).

pub mod level;
pub mod search;

#[cfg(feature = "python")]
mod python;

pub use level::{Level, ParseError};
pub use search::{Solution, Status, solve};

/// Outcome of solving one level given as text.
pub type Outcome = Result<Solution, ParseError>;

/// Parses and solves one level.
pub fn solve_text(text: &str, max_nodes: u64) -> Outcome {
    Level::parse(text).map(|level| solve(&level, max_nodes))
}

/// Solves many levels in parallel, preserving order.
#[cfg(feature = "parallel")]
pub fn solve_batch(texts: &[String], max_nodes: u64) -> Vec<Outcome> {
    use rayon::prelude::*;
    texts.par_iter().map(|t| solve_text(t, max_nodes)).collect()
}
