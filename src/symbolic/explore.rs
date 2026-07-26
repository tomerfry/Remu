//! Concolic exploration driver (feature `symbolic-solver`).
//!
//! Repeatedly re-executes a program concretely, negating one branch per new
//! input to discover paths, until a goal is reached — the generational search
//! used by SAGE/Triton. The engine can't own the guest's memory/bus, so the
//! caller supplies a `harness` closure that builds a fresh machine, applies the
//! chosen input, runs it, and reports whether the goal was met plus the run's
//! [`SymEngine`] (which holds the path constraints).

use std::collections::{HashMap, HashSet, VecDeque};

use super::solver::Solver;
use super::state::SymEngine;

/// A concrete input assignment: input name → value.
pub type InputMap = HashMap<String, u64>;

/// Search for an input that drives `harness` to report success.
///
/// `harness(inputs)` must build a fresh machine, symbolize and **seed** its
/// inputs from `inputs` (an absent name uses that input's default seed), run to
/// completion, and return `(reached, engine)`. Exploration starts from the
/// default seed (an empty map) and, for each run that misses the goal, solves
/// for inputs that flip each branch on that path, enqueueing the fresh ones.
///
/// Returns the first input map that reaches the goal, or `None` after
/// `max_iters` executions. Uses [`Solver::new`] (`REMU_SMT_SOLVER`).
pub fn find_input<H>(max_iters: usize, harness: &mut H) -> Option<InputMap>
where
    H: FnMut(&InputMap) -> (bool, SymEngine),
{
    let solver = Solver::new();
    let mut queue: VecDeque<InputMap> = VecDeque::from([InputMap::new()]);
    let mut seen: HashSet<Vec<(String, u64)>> = HashSet::new();
    let mut iters = 0;

    while let Some(input) = queue.pop_front() {
        if iters >= max_iters {
            break;
        }
        iters += 1;

        let (reached, eng) = harness(&input);
        if reached {
            return Some(input);
        }

        // Fan out: negate each branch on this path and solve for the input that
        // takes the other side.
        for i in 0..eng.constraints().len() {
            let Some(raw) = solver.solve(&eng.smtlib(Some(i))) else {
                continue;
            };
            let child = named_model(&raw, &eng);
            if seen.insert(canonical(&child)) {
                queue.push_back(child);
            }
        }
    }
    None
}

/// Translate a solver model keyed by SMT name (`x!<id>`) to input names.
fn named_model(raw: &HashMap<String, u64>, eng: &SymEngine) -> InputMap {
    let mut out = InputMap::new();
    for (k, v) in raw {
        if let Some(id) = k.strip_prefix("x!").and_then(|d| d.parse::<u32>().ok())
            && let Some(name) = eng.input_name(id)
        {
            out.insert(name.to_string(), *v);
        }
    }
    out
}

/// A stable key for de-duplicating input assignments.
fn canonical(m: &InputMap) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = m.iter().map(|(k, val)| (k.clone(), *val)).collect();
    v.sort();
    v
}
