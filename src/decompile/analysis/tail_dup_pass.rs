// Cross-jump tail de-duplication: un-merges a gcc-cross-jumped shared tail pre-structuring, so dominance scopes the side-effecting tail under each context's own guard.

use std::collections::{HashMap, HashSet};

use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::x86::types::*;

const SYNTH_BIT: u64 = 1u64 << 62;
const MAX_TAIL_LEN: usize = 8;

fn is_terminal(stmt: &CsharpminorStmt) -> bool {
    matches!(
        stmt,
        CsharpminorStmt::Sjump(_)
            | CsharpminorStmt::Sreturn(_)
            | CsharpminorStmt::Stailcall(_, _, _)
            | CsharpminorStmt::Scond(_, _, _, _)
            | CsharpminorStmt::Sjumptable(_, _)
    )
}

fn is_call(stmt: &CsharpminorStmt) -> bool {
    matches!(stmt, CsharpminorStmt::Scall(_, _, _, _))
}

// Explicit (non-fallthrough) control-flow targets of a statement.
fn branch_targets(stmt: &CsharpminorStmt) -> Vec<Node> {
    match stmt {
        CsharpminorStmt::Sjump(t) => vec![*t],
        CsharpminorStmt::Scond(_, _, t, f) => vec![*t, *f],
        CsharpminorStmt::Sjumptable(_, targets) => targets.as_ref().to_vec(),
        _ => vec![],
    }
}

// Redirect every from edge of an Sjump/Scond to to, returning true if anything changed; Sjumptable preds are out of scope.
fn retarget(stmt: &mut CsharpminorStmt, from: Node, to: Node) -> bool {
    let mut changed = false;
    match stmt {
        CsharpminorStmt::Sjump(t) => {
            if *t == from {
                *t = to;
                changed = true;
            }
        }
        CsharpminorStmt::Scond(_, _, t, f) => {
            if *t == from {
                *t = to;
                changed = true;
            }
            if *f == from {
                *f = to;
                changed = true;
            }
        }
        _ => {}
    }
    changed
}

pub struct TailDupPass;

impl IRPass for TailDupPass {
    fn name(&self) -> &'static str {
        "tail_dup"
    }

    fn inputs(&self) -> &'static [&'static str] {
        &["csharp_stmt_candidate", "next", "instr_in_function"]
    }

    fn outputs(&self) -> &'static [&'static str] {
        &["csharp_stmt_candidate"]
    }

    fn run(&self, db: &mut DecompileDB) {
        let stmts: Vec<(Node, CsharpminorStmt)> = db
            .rel_iter::<(Node, CsharpminorStmt)>("csharp_stmt_candidate")
            .map(|(n, s)| (*n, s.clone()))
            .collect();
        if stmts.is_empty() {
            return;
        }
        let stmt_map: HashMap<Node, CsharpminorStmt> = stmts.iter().cloned().collect();
        let stmt_nodes: HashSet<Node> = stmts.iter().map(|(n, _)| *n).collect();

        let in_func: HashMap<Node, Address> = db
            .rel_iter::<(Node, Address)>("instr_in_function")
            .map(|(n, f)| (*n, *f))
            .collect();

        // Raw address-order successor (may point at a non-stmt node).
        let next_raw: HashMap<Node, Node> = db
            .rel_iter::<(Node, Node)>("next")
            .map(|(a, b)| (*a, *b))
            .collect();

        // Statement-level fall-through: walk `next` past non-stmt nodes to the next stmt node.
        let stmt_next = |mut n: Node| -> Option<Node> {
            for _ in 0..64 {
                let nx = *next_raw.get(&n)?;
                if stmt_nodes.contains(&nx) {
                    return Some(nx);
                }
                n = nx;
            }
            None
        };

        // Predecessor classification, per kind, keyed by target stmt node.
        let mut branch_preds: HashMap<Node, Vec<Node>> = HashMap::new();
        let mut fall_preds: HashMap<Node, Vec<Node>> = HashMap::new();
        for (n, s) in &stmts {
            for t in branch_targets(s) {
                if stmt_nodes.contains(&t) {
                    branch_preds.entry(t).or_default().push(*n);
                }
            }
            // A non-terminal stmt falls through to its next stmt node.
            if !is_terminal(s) {
                if let Some(nx) = stmt_next(*n) {
                    fall_preds.entry(nx).or_default().push(*n);
                }
            }
        }

        // Total stmt-pred count (branch + fall) for the linearity test on the tail body.
        let pred_count = |node: Node| -> usize {
            branch_preds.get(&node).map_or(0, |v| v.len())
                + fall_preds.get(&node).map_or(0, |v| v.len())
        };

        let mut new_stmts: Vec<(Node, CsharpminorStmt)> = Vec::new(); // synthetic copies
        let mut rewires: HashMap<Node, Vec<(Node, Node)>> = HashMap::new(); // pred -> [(from,to)]

        for (entry, _) in &stmts {
            let entry = *entry;
            let synth = entry | SYNTH_BIT;
            if stmt_nodes.contains(&synth) {
                continue; // synthetic id collision: skip
            }
            // Cross-jump merge: exactly one explicit-branch pred + >=1 fall-through pred.
            let bps = match branch_preds.get(&entry) {
                Some(v) if v.len() == 1 => v,
                _ => continue,
            };
            if fall_preds.get(&entry).map_or(0, |v| v.len()) == 0 {
                continue;
            }
            let branch_pred = bps[0];
            // The split is only meaningful inside one function.
            if in_func.get(&branch_pred) != in_func.get(&entry) {
                continue;
            }
            // Collect the bounded single-entry straight-line tail: entry, then fall-through successors each reached ONLY from the previous node.
            let mut chain: Vec<Node> = vec![entry];
            let mut cur = entry;
            let mut ok = false;
            loop {
                let s = &stmt_map[&cur];
                if is_terminal(s) {
                    // Must end in a plain jump to common code (the merge), not a return/cond.
                    ok = matches!(s, CsharpminorStmt::Sjump(_));
                    break;
                }
                let nx = match stmt_next(cur) {
                    Some(nx) if stmt_nodes.contains(&nx) => nx,
                    _ => break,
                };
                // tail body node must be single-entry (reached only from `cur`)
                if pred_count(nx) != 1 {
                    break;
                }
                if chain.len() >= MAX_TAIL_LEN {
                    break;
                }
                chain.push(nx);
                cur = nx;
            }
            if !ok {
                continue;
            }
            // Hazard gate: the tail must carry a call, the side effect that must stay conditional; pure-data tails are left to structuring.
            if !chain.iter().any(|n| is_call(&stmt_map[n])) {
                continue;
            }
            // Build the duplicated tail as one Sseq node and redirect the branch edge to it.
            let seq: Vec<CsharpminorStmt> = chain.iter().map(|n| stmt_map[n].clone()).collect();
            new_stmts.push((synth, CsharpminorStmt::Sseq(seq)));
            rewires
                .entry(branch_pred)
                .or_default()
                .push((entry, synth));
        }

        if new_stmts.is_empty() {
            return;
        }

        // Emit the rewritten candidate relation: rewired branch preds + appended synth copies.
        let out = ascent::boxcar::Vec::<(Node, CsharpminorStmt)>::new();
        for (n, s) in &stmts {
            if let Some(edits) = rewires.get(n) {
                let mut s2 = s.clone();
                for &(from, to) in edits {
                    retarget(&mut s2, from, to);
                }
                out.push((*n, s2));
            } else {
                out.push((*n, s.clone()));
            }
        }
        for (n, s) in &new_stmts {
            out.push((*n, s.clone()));
        }
        db.rel_set("csharp_stmt_candidate", out);
        // Synthetic copies stay out of instr_in_function (declaring it an output would cycle the schedule); downstream maps fall back to the masked base id.
    }
}
