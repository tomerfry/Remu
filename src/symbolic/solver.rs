//! SMT solver driver (feature `symbolic-solver`).
//!
//! Deliberately dependency-free: it spawns a solver binary that speaks SMT-LIB
//! 2 over stdio (z3, cvc5, bitwuzla, …), feeds it a script from [`smtlib`], and
//! parses the model back. No native linking, which keeps the Windows-first,
//! minimal-dependency build simple.
//!
//! The binary and its arguments default to `z3 -in`; override the binary with
//! the `REMU_SMT_SOLVER` environment variable (e.g. a full path, or `cvc5`).
//!
//! [`smtlib`]: super::smtlib

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

/// A handle to an external SMT solver.
#[derive(Debug, Clone)]
pub struct Solver {
    binary: String,
    args: Vec<String>,
}

impl Default for Solver {
    fn default() -> Self {
        Solver::new()
    }
}

impl Solver {
    /// A solver from `REMU_SMT_SOLVER` (default `z3`), run in incremental stdio
    /// mode.
    pub fn new() -> Self {
        let binary = std::env::var("REMU_SMT_SOLVER").unwrap_or_else(|_| "z3".to_string());
        Solver {
            binary,
            args: vec!["-in".to_string()],
        }
    }

    /// Whether the configured solver binary can be launched.
    pub fn available(&self) -> bool {
        Command::new(&self.binary)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    /// Run `script` and, if satisfiable, return the model as a map from SMT
    /// variable name (`x!<id>`) to its concrete value. Returns `None` on
    /// `unsat`, a spawn/IO failure, or an unparseable response.
    pub fn solve(&self, script: &str) -> Option<HashMap<String, u64>> {
        let mut child = Command::new(&self.binary)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        child.stdin.take()?.write_all(script.as_bytes()).ok()?;
        let out = child.wait_with_output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        if !text.contains("sat") || first_result(&text)? != "sat" {
            return None;
        }
        Some(parse_model(&text))
    }
}

/// The first `sat`/`unsat`/`unknown` token in the solver's output.
fn first_result(text: &str) -> Option<&str> {
    for line in text.lines() {
        let t = line.trim();
        match t {
            "sat" | "unsat" | "unknown" => return Some(t),
            _ => {}
        }
    }
    None
}

/// Extract `x!<id> -> value` pairs from a `(get-value …)` response. Values may
/// be `#x<hex>`, `#b<bin>`, or `(_ bv<dec> <w>)`.
fn parse_model(text: &str) -> HashMap<String, u64> {
    let mut model = HashMap::new();
    let mut rest = text;
    while let Some(p) = rest.find("x!") {
        let name_start = p;
        let after = &rest[name_start + 2..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            rest = &rest[name_start + 2..];
            continue;
        }
        let name = format!("x!{digits}");
        let value_region = &after[digits.len()..];
        if let Some(v) = parse_value(value_region) {
            model.insert(name, v);
        }
        rest = value_region;
    }
    model
}

/// Parse the first bitvector literal appearing in `s`.
fn parse_value(s: &str) -> Option<u64> {
    let s = s.trim_start();
    if let Some(hex) = s.strip_prefix("#x") {
        let digits: String = hex.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        return u64::from_str_radix(&digits, 16).ok();
    }
    if let Some(bin) = s.strip_prefix("#b") {
        let digits: String = bin.chars().take_while(|c| *c == '0' || *c == '1').collect();
        return u64::from_str_radix(&digits, 2).ok();
    }
    // (_ bv<dec> <width>)
    if let Some(idx) = s.find("bv") {
        let after = &s[idx + 2..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return digits.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_bin_and_decimal_models() {
        let hex = "sat\n((x!0 #x12345678))\n";
        assert_eq!(parse_model(hex).get("x!0"), Some(&0x1234_5678));

        let bin = "sat\n((x!1 #b1010))\n";
        assert_eq!(parse_model(bin).get("x!1"), Some(&10));

        let dec = "sat\n((x!2 (_ bv255 8)))\n";
        assert_eq!(parse_model(dec).get("x!2"), Some(&255));

        let multi = "sat\n((x!0 #x01) (x!3 #x02))\n";
        let m = parse_model(multi);
        assert_eq!(m.get("x!0"), Some(&1));
        assert_eq!(m.get("x!3"), Some(&2));
    }

    #[test]
    fn first_result_finds_verdict() {
        assert_eq!(first_result("sat\n((x!0 #x00))"), Some("sat"));
        assert_eq!(first_result("unsat\n"), Some("unsat"));
        assert_eq!(first_result("garbage\n"), None);
    }
}
