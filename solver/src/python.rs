//! Python bindings (`boxwright_solver` module).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{Level, Outcome, Status};

#[pyclass(name = "Solution", module = "boxwright_solver", frozen, get_all)]
struct PySolution {
    /// One of "solved", "unsolvable", "budget_exceeded", "invalid".
    status: &'static str,
    moves: u32,
    pushes: u32,
    path: String,
    nodes_expanded: u64,
    /// Parse error message when `status == "invalid"`.
    error: Option<String>,
}

#[pymethods]
impl PySolution {
    #[getter]
    fn solved(&self) -> bool {
        self.status == "solved"
    }

    fn __repr__(&self) -> String {
        match &self.error {
            Some(e) => format!("Solution(status='invalid', error={e:?})"),
            None => format!(
                "Solution(status={:?}, moves={}, pushes={}, nodes_expanded={})",
                self.status, self.moves, self.pushes, self.nodes_expanded
            ),
        }
    }
}

impl From<Outcome> for PySolution {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Ok(s) => PySolution {
                status: match s.status {
                    Status::Solved => "solved",
                    Status::Unsolvable => "unsolvable",
                    Status::BudgetExceeded => "budget_exceeded",
                },
                moves: s.moves,
                pushes: s.pushes,
                path: s.path,
                nodes_expanded: s.nodes_expanded,
                error: None,
            },
            Err(e) => PySolution {
                status: "invalid",
                moves: 0,
                pushes: 0,
                path: String::new(),
                nodes_expanded: 0,
                error: Some(e.0),
            },
        }
    }
}

/// Solve one level given as text. Never raises for bad levels; check `status`.
#[pyfunction]
#[pyo3(signature = (level, max_nodes = 1_000_000))]
fn solve(py: Python<'_>, level: &str, max_nodes: u64) -> PySolution {
    py.detach(|| crate::solve_text(level, max_nodes)).into()
}

/// Solve many levels in parallel (order preserved), releasing the GIL.
#[pyfunction]
#[pyo3(signature = (levels, max_nodes = 1_000_000, threads = None))]
fn solve_batch(
    py: Python<'_>,
    levels: Vec<String>,
    max_nodes: u64,
    threads: Option<usize>,
) -> PyResult<Vec<PySolution>> {
    let outcomes = py.detach(|| match threads {
        None => Ok(crate::solve_batch(&levels, max_nodes)),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(|pool| pool.install(|| crate::solve_batch(&levels, max_nodes))),
    });
    let outcomes = outcomes.map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(outcomes.into_iter().map(PySolution::from).collect())
}

/// True if the LURD `path` is legal from `level` and ends with every box on a goal.
#[pyfunction]
fn verify(level: &str, path: &str) -> PyResult<bool> {
    let level = Level::parse(level).map_err(|e| PyValueError::new_err(e.0))?;
    Ok(level.verify(path))
}

#[pymodule]
fn boxwright_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(solve_batch, m)?)?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    Ok(())
}
