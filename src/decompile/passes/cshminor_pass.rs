use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::{declare_io_from, run_pass};

use crate::decompile::passes::cminor_pass::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::mreg::Mreg;
use crate::x86::op::{Addressing, Condition, Operation};
use crate::x86::types::*;
use ascent::ascent_par;
use ascent::lattice::set::Set;
use ascent::Dual;

// Helper: add node n to a Set<Node>, returning new Set
fn dom_set_with_self(strict: &Set<Node>, n: Node) -> Set<Node> {
    let mut s = strict.0.clone();
    s.insert(n);
    Set(s)
}

// Substitute every Evar(old) with Evar(new) inside a Csharpminor expression.
fn cond_subst_var_in_expr(expr: &CsharpminorExpr, old: RTLReg, new: RTLReg) -> CsharpminorExpr {
    match expr {
        CsharpminorExpr::Evar(r) if *r == old => CsharpminorExpr::Evar(new),
        CsharpminorExpr::Eunop(op, inner) => CsharpminorExpr::Eunop(
            op.clone(),
            Box::new(cond_subst_var_in_expr(inner, old, new)),
        ),
        CsharpminorExpr::Ebinop(op, l, r) => CsharpminorExpr::Ebinop(
            op.clone(),
            Box::new(cond_subst_var_in_expr(l, old, new)),
            Box::new(cond_subst_var_in_expr(r, old, new)),
        ),
        CsharpminorExpr::Eload(chunk, addr) => {
            CsharpminorExpr::Eload(*chunk, Box::new(cond_subst_var_in_expr(addr, old, new)))
        }
        CsharpminorExpr::Econdition(c, t, f) => CsharpminorExpr::Econdition(
            Box::new(cond_subst_var_in_expr(c, old, new)),
            Box::new(cond_subst_var_in_expr(t, old, new)),
            Box::new(cond_subst_var_in_expr(f, old, new)),
        ),
        _ => expr.clone(),
    }
}

fn cond_subst_exprs(args: &[CsharpminorExpr], old: RTLReg, new: RTLReg) -> Vec<CsharpminorExpr> {
    args.iter()
        .map(|e| cond_subst_var_in_expr(e, old, new))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum LoopType {
    PreTested,
    PostTested,
    MidTested,
    Infinite,
}
use either::Either;

const SYNTHETIC_NODE_MASK: Node = (1u64 << 62) | (1u64 << 63);

// Keep byte offsets byte-scaled regardless of the backing local's declared
// scalar type.  Csharpminor has no cast node dedicated to char pointers, so
// carry &local through the existing 64-bit integer address path; Eload/Sstore
// subsequently cast that numeric byte address to the authoritative chunk's
// pointer type.
fn home_backing_address(slot: RTLReg, byte_ofs: i64) -> CsharpminorExpr {
    let base = CsharpminorExpr::Eunop(
        CminorUnop::Olongofintu,
        Box::new(CsharpminorExpr::Eaddrof(ident_from_reg(slot))),
    );
    if byte_ofs == 0 {
        base
    } else {
        CsharpminorExpr::Ebinop(
            CminorBinop::Oaddl,
            Box::new(base),
            Box::new(CsharpminorExpr::Econst(Constant::Olongconst(byte_ofs))),
        )
    }
}

fn rewrite_home_backing_expr(
    expr: &CsharpminorExpr,
    slot: RTLReg,
    chunk: MemoryChunk,
    address: &CsharpminorExpr,
) -> CsharpminorExpr {
    match expr {
        CsharpminorExpr::Evar(reg) if *reg == slot => {
            CsharpminorExpr::Eload(chunk, Box::new(address.clone()))
        }
        // One x86 instruction has at most one decoded memory operand.  At an
        // authenticated backing-access node, an explicit load is that operand
        // even when ordinary stack lowering did not first scalarize it.
        CsharpminorExpr::Eload(_, _) => {
            CsharpminorExpr::Eload(chunk, Box::new(address.clone()))
        }
        CsharpminorExpr::Eunop(op, inner) => CsharpminorExpr::Eunop(
            op.clone(),
            Box::new(rewrite_home_backing_expr(inner, slot, chunk, address)),
        ),
        CsharpminorExpr::Ebinop(op, left, right) => CsharpminorExpr::Ebinop(
            op.clone(),
            Box::new(rewrite_home_backing_expr(left, slot, chunk, address)),
            Box::new(rewrite_home_backing_expr(right, slot, chunk, address)),
        ),
        CsharpminorExpr::Econdition(condition, if_true, if_false) => {
            CsharpminorExpr::Econdition(
                Box::new(rewrite_home_backing_expr(
                    condition, slot, chunk, address,
                )),
                Box::new(rewrite_home_backing_expr(if_true, slot, chunk, address)),
                Box::new(rewrite_home_backing_expr(if_false, slot, chunk, address)),
            )
        }
        _ => expr.clone(),
    }
}

fn rewrite_home_backing_builtin(
    arg: &BuiltinArg<CsharpminorExpr>,
    slot: RTLReg,
    chunk: MemoryChunk,
    address: &CsharpminorExpr,
) -> BuiltinArg<CsharpminorExpr> {
    match arg {
        BuiltinArg::BA(expr) => BuiltinArg::BA(rewrite_home_backing_expr(
            expr, slot, chunk, address,
        )),
        BuiltinArg::BASplitLong(left, right) => BuiltinArg::BASplitLong(
            Box::new(rewrite_home_backing_builtin(left, slot, chunk, address)),
            Box::new(rewrite_home_backing_builtin(right, slot, chunk, address)),
        ),
        BuiltinArg::BAAddPtr(left, right) => BuiltinArg::BAAddPtr(
            Box::new(rewrite_home_backing_builtin(left, slot, chunk, address)),
            Box::new(rewrite_home_backing_builtin(right, slot, chunk, address)),
        ),
        _ => arg.clone(),
    }
}

fn rewrite_home_backing_stmt(
    stmt: &CsharpminorStmt,
    slot: RTLReg,
    byte_ofs: i64,
    chunk: MemoryChunk,
    read: bool,
    write: bool,
    address_only: bool,
) -> CsharpminorStmt {
    let address = home_backing_address(slot, byte_ofs);
    if address_only {
        return match stmt {
            CsharpminorStmt::Sset(destination, _) => {
                let value = if byte_ofs == 0 {
                    CsharpminorExpr::Eaddrof(ident_from_reg(slot))
                } else {
                    address
                };
                CsharpminorStmt::Sset(*destination, value)
            }
            _ => stmt.clone(),
        };
    }

    let expr = |value: &CsharpminorExpr| {
        if read {
            rewrite_home_backing_expr(value, slot, chunk, &address)
        } else {
            value.clone()
        }
    };
    match stmt {
        CsharpminorStmt::Sset(destination, value) if write && *destination == slot => {
            CsharpminorStmt::Sstore(chunk, address.clone(), expr(value))
        }
        CsharpminorStmt::Sset(destination, value) => {
            CsharpminorStmt::Sset(*destination, expr(value))
        }
        CsharpminorStmt::Sstore(_, _, value) => {
            CsharpminorStmt::Sstore(chunk, address.clone(), expr(value))
        }
        CsharpminorStmt::Scall(destination, signature, callee, args) => {
            let callee = match callee {
                Either::Left(value) => Either::Left(expr(value)),
                other => other.clone(),
            };
            CsharpminorStmt::Scall(
                *destination,
                signature.clone(),
                callee,
                args.iter().map(expr).collect(),
            )
        }
        CsharpminorStmt::Stailcall(signature, callee, args) => {
            let callee = match callee {
                Either::Left(value) => Either::Left(expr(value)),
                other => other.clone(),
            };
            CsharpminorStmt::Stailcall(
                signature.clone(),
                callee,
                args.iter().map(expr).collect(),
            )
        }
        CsharpminorStmt::Sbuiltin(destination, name, args, result) => {
            CsharpminorStmt::Sbuiltin(
                *destination,
                name.clone(),
                args.iter()
                    .map(|arg| rewrite_home_backing_builtin(arg, slot, chunk, &address))
                    .collect(),
                rewrite_home_backing_builtin(result, slot, chunk, &address),
            )
        }
        CsharpminorStmt::Scond(condition, args, if_true, if_false) => {
            CsharpminorStmt::Scond(
                condition.clone(),
                args.iter().map(expr).collect(),
                *if_true,
                *if_false,
            )
        }
        CsharpminorStmt::Sjumptable(value, targets) => {
            CsharpminorStmt::Sjumptable(expr(value), targets.clone())
        }
        CsharpminorStmt::Sreturn(value) => CsharpminorStmt::Sreturn(expr(value)),
        CsharpminorStmt::Sseq(statements) => CsharpminorStmt::Sseq(
            statements
                .iter()
                .map(|statement| {
                    rewrite_home_backing_stmt(
                        statement,
                        slot,
                        byte_ofs,
                        chunk,
                        read,
                        write,
                        address_only,
                    )
                })
                .collect(),
        ),
        CsharpminorStmt::Sifthenelse(condition, args, if_true, if_false) => {
            CsharpminorStmt::Sifthenelse(
                condition.clone(),
                args.iter().map(expr).collect(),
                Box::new(rewrite_home_backing_stmt(
                    if_true,
                    slot,
                    byte_ofs,
                    chunk,
                    read,
                    write,
                    address_only,
                )),
                Box::new(rewrite_home_backing_stmt(
                    if_false,
                    slot,
                    byte_ofs,
                    chunk,
                    read,
                    write,
                    address_only,
                )),
            )
        }
        CsharpminorStmt::Sloop(body) => CsharpminorStmt::Sloop(Box::new(
            rewrite_home_backing_stmt(
                body,
                slot,
                byte_ofs,
                chunk,
                read,
                write,
                address_only,
            ),
        )),
        _ => stmt.clone(),
    }
}

fn home_backing_candidate_reads(inst: &RTLInst, slot: RTLReg) -> bool {
    match inst {
        RTLInst::Iload(..) => true,
        RTLInst::Iop(_, args, _) | RTLInst::Icond(_, args, _, _) => args.contains(&slot),
        RTLInst::Istore(_, _, args, source) => {
            args.contains(&slot) || *source == slot
        }
        _ => false,
    }
}

fn home_backing_candidate_writes(inst: &RTLInst, slot: RTLReg) -> bool {
    matches!(inst, RTLInst::Istore(..))
        || matches!(inst, RTLInst::Iop(_, _, destination) if *destination == slot)
}

fn home_backing_candidate_takes_address(inst: &RTLInst) -> bool {
    matches!(
        inst,
        RTLInst::Iop(
            Operation::Olea(Addressing::Ainstack(_))
                | Operation::Oleal(Addressing::Ainstack(_)),
            _,
            _
        )
    )
}

fn materialize_win64_home_backing(db: &mut DecompileDB) {
    let mut accesses: BTreeMap<
        Node,
        BTreeSet<(RTLReg, i64, usize, MemoryChunk, bool, bool, bool)>,
    > = BTreeMap::new();
    for (node, slot, byte_ofs, width, chunk, read, write, address) in db
        .rel_iter::<(
            Node,
            RTLReg,
            i64,
            usize,
            MemoryChunk,
            bool,
            bool,
            bool,
        )>("win64_home_backing_access")
    {
        accesses.entry(*node).or_default().insert((
            *slot,
            *byte_ofs,
            *width,
            *chunk,
            *read,
            *write,
            *address,
        ));
    }
    if accesses.is_empty() {
        return;
    }
    let mut selected: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (_, candidate, inst) in db.rel_iter::<(Node, Node, RTLInst)>(
        "win64_home_backing_selected_candidate",
    ) {
        selected.entry(*candidate).or_default().push(inst.clone());
    }
    for rows in selected.values_mut() {
        rows.sort_by_cached_key(|inst| format!("{inst:?}"));
        rows.dedup();
    }

    let mut rewritten = Vec::new();
    for (node, stmt) in db.rel_iter::<(Node, CsharpminorStmt)>("csharp_stmt_candidate") {
        let real = *node & !SYNTHETIC_NODE_MASK;
        let Some(rows) = accesses.get(&real) else {
            rewritten.push((*node, stmt.clone()));
            continue;
        };
        // RTL selection admitted a backing cell only after proving exactly
        // one cell/access interpretation at this real instruction.
        let Some(&(slot, byte_ofs, _width, chunk, read, write, address)) =
            rows.iter().next()
        else {
            rewritten.push((*node, stmt.clone()));
            continue;
        };
        if rows.len() != 1 {
            rewritten.push((*node, stmt.clone()));
            continue;
        }
        let Some(candidate_rows) = selected.get(node) else {
            rewritten.push((*node, stmt.clone()));
            continue;
        };
        let Some(candidate) = candidate_rows.iter().next() else {
            rewritten.push((*node, stmt.clone()));
            continue;
        };
        if candidate_rows.len() != 1 {
            rewritten.push((*node, stmt.clone()));
            continue;
        }
        let candidate_read = read && home_backing_candidate_reads(candidate, slot);
        let candidate_write = write && home_backing_candidate_writes(candidate, slot);
        let candidate_address = address && home_backing_candidate_takes_address(candidate);
        rewritten.push((
            *node,
            rewrite_home_backing_stmt(
                stmt,
                slot,
                byte_ofs,
                chunk,
                candidate_read,
                candidate_write,
                candidate_address,
            ),
        ));
    }
    rewritten.sort_by_cached_key(|(node, stmt)| (*node, format!("{stmt:?}")));
    rewritten.dedup();
    db.rel_set(
        "csharp_stmt_candidate",
        rewritten
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
}

/// Retain a CR8 byte marker only when final pre-Csh type evidence remains
/// integral. A pointer fact or any candidate which outranks Xlongunsigned
/// cannot safely receive the comparison-local byte cast. If one value at an
/// ambiguous node conflicts, reject that whole node rather than filtering
/// ambiguity into an apparently unique marker.
fn filter_cr8_byte_compares_with_incompatible_types(db: &mut DecompileDB) {
    let markers: Vec<(Node, RTLReg)> = db
        .rel_iter::<(Node, RTLReg)>("cr8_byte_compare")
        .copied()
        .collect();
    if markers.is_empty() {
        return;
    }
    let marker_values: BTreeSet<RTLReg> = markers.iter().map(|(_, value)| *value).collect();
    let threshold =
        crate::decompile::passes::clight_pass::xtype_refine_priority(&XType::Xlongunsigned);
    let mut incompatible_values: BTreeSet<RTLReg> = db
        .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
        .filter_map(|(value, xtype)| {
            (marker_values.contains(value)
                && crate::decompile::passes::clight_pass::xtype_refine_priority(xtype) > threshold)
                .then_some(*value)
        })
        .collect();
    incompatible_values.extend(
        db.rel_iter::<(RTLReg,)>("is_ptr")
            .filter_map(|(value,)| marker_values.contains(value).then_some(*value)),
    );
    if incompatible_values.is_empty() {
        return;
    }
    let blocked_nodes: BTreeSet<Node> = markers
        .iter()
        .filter_map(|(node, value)| incompatible_values.contains(value).then_some(*node))
        .collect();
    db.rel_set(
        "cr8_byte_compare",
        markers
            .into_iter()
            .filter(|(node, _)| !blocked_nodes.contains(node))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
}

fn scalar_lvalue_integral_type_width(xtype: XType) -> Option<usize> {
    match xtype {
        XType::Xint8signed | XType::Xint8unsigned => Some(1),
        XType::Xint16signed | XType::Xint16unsigned => Some(2),
        XType::Xint | XType::Xintunsigned | XType::Xany32 => Some(4),
        XType::Xlong | XType::Xlongunsigned | XType::Xany64 => Some(8),
        XType::Xbool
        | XType::Xfloat
        | XType::Xsingle
        | XType::Xptr
        | XType::Xcharptr
        | XType::Xcharptrptr
        | XType::Xintptr
        | XType::Xfloatptr
        | XType::Xsingleptr
        | XType::Xfuncptr
        | XType::Xvoid
        | XType::XstructPtr(_) => None,
    }
}

pub(crate) fn scalar_lvalue_extension_result_type(
    proof: &ScalarMemoryAccessProof,
) -> Option<XType> {
    match (proof.direction, proof.extension, proof.value_width) {
        (ScalarMemoryDirection::Read, ScalarMemoryExtension::SignExtend, 4) => Some(XType::Xint),
        (ScalarMemoryDirection::Read, ScalarMemoryExtension::ZeroExtend, 4) => {
            Some(XType::Xintunsigned)
        }
        (ScalarMemoryDirection::Read, ScalarMemoryExtension::SignExtend, 8) => Some(XType::Xlong),
        (ScalarMemoryDirection::Read, ScalarMemoryExtension::ZeroExtend, 8) => {
            Some(XType::Xlongunsigned)
        }
        _ => None,
    }
}

fn scalar_lvalue_extension_chunk_type(proof: &ScalarMemoryAccessProof) -> Option<XType> {
    match proof.chunk {
        MemoryChunk::MInt8Signed => Some(XType::Xint8signed),
        MemoryChunk::MInt8Unsigned => Some(XType::Xint8unsigned),
        MemoryChunk::MInt16Signed => Some(XType::Xint16signed),
        MemoryChunk::MInt16Unsigned => Some(XType::Xint16unsigned),
        MemoryChunk::MInt32 => Some(XType::Xint),
        _ => None,
    }
}

fn scalar_lvalue_extension_companion_type(proof: &ScalarMemoryAccessProof) -> Option<XType> {
    match (proof.chunk, proof.width) {
        (MemoryChunk::MInt8Signed, 1) => Some(XType::Xint8unsigned),
        (MemoryChunk::MInt8Unsigned, 1) => Some(XType::Xint8signed),
        (MemoryChunk::MInt16Signed, 2) => Some(XType::Xint16unsigned),
        (MemoryChunk::MInt16Unsigned, 2) => Some(XType::Xint16signed),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScalarExtensionTypeRewrite {
    pub(crate) transport_types: BTreeSet<XType>,
    pub(crate) result_type: XType,
}

/// Validate the complete late type set for one authenticated extension load.
/// The selected transport and its sole same-width signedness companion are
/// lowering artifacts; the raw opcode's exact architectural result type is
/// the only other admissible row. This is shared by SignatureReconciliation
/// and Cshminor so the ABI and final source cannot disagree.
pub(crate) fn scalar_lvalue_extension_type_rewrite(
    proof: &ScalarMemoryAccessProof,
    types: &BTreeSet<XType>,
    pointer: bool,
) -> Option<ScalarExtensionTypeRewrite> {
    if pointer
        || !proof.is_closed_v1()
        || proof.direction != ScalarMemoryDirection::Read
        || proof.extension == ScalarMemoryExtension::Plain
        || types.is_empty()
    {
        return None;
    }
    let selected = scalar_lvalue_extension_chunk_type(proof)?;
    let result_type = scalar_lvalue_extension_result_type(proof)?;
    let mut transport_types = BTreeSet::from([selected]);
    if let Some(companion) = scalar_lvalue_extension_companion_type(proof) {
        transport_types.insert(companion);
    }
    types
        .iter()
        .all(|xtype| transport_types.contains(xtype) || *xtype == result_type)
        .then_some(ScalarExtensionTypeRewrite {
            transport_types,
            result_type,
        })
}

/// Revalidate scalar-lvalue proof values after TypePass, PtrTo, struct, and
/// signature reconciliation have reached their final relation state.  This is
/// deliberately the last gate before Csharp structuring: an authenticated
/// memory operand is not permission to reinterpret a late float, pointer,
/// bool, or wider value as an integer lvalue.
///
/// MOVSX/MOVZX are special only in one closed way.  The decoder proves their
/// architectural result width/signedness, while the ordinary type pipeline
/// may leave the loaded source chunk's sealed signedness pair on the result
/// register. For a surviving proof, remove only those transport artifacts and
/// publish exactly the opcode-defined EAX/RAX result type. Any other
/// conflicting late type rejects every proof sharing that value instead of
/// being filtered away.
fn filter_scalar_lvalues_with_final_types(db: &mut DecompileDB) {
    let mut proofs: Vec<(Node, ScalarMemoryAccessProof)> = db
        .rel_iter::<(Node, ScalarMemoryAccessProof)>("cminor_scalar_memory_access")
        .cloned()
        .collect();
    if proofs.is_empty() {
        return;
    }
    proofs.sort();
    proofs.dedup();

    let pointer_values: BTreeSet<RTLReg> = db
        .rel_iter::<(RTLReg,)>("is_ptr")
        .map(|row| row.0)
        .collect();
    let mut types_by_value: BTreeMap<RTLReg, BTreeSet<XType>> = BTreeMap::new();
    let mut type_rows: Vec<(RTLReg, XType)> = db
        .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
        .copied()
        .collect();
    type_rows.sort();
    type_rows.dedup();
    for (value, xtype) in &type_rows {
        types_by_value.entry(*value).or_default().insert(*xtype);
    }

    let signature_extensions: BTreeSet<(Node, ScalarMemoryAccessProof)> = db
        .rel_iter::<(Node, ScalarMemoryAccessProof)>("signature_scalar_extension_access")
        .cloned()
        .collect();
    let mut returns_by_function: BTreeMap<Address, BTreeSet<RTLReg>> = BTreeMap::new();
    for (function, value) in db.rel_iter::<(Address, RTLReg)>("emit_function_return") {
        returns_by_function
            .entry(*function)
            .or_default()
            .insert(*value);
    }
    let mut final_returns_by_function: BTreeMap<Address, BTreeSet<XType>> = BTreeMap::new();
    for (function, xtype) in db.rel_iter::<(Address, XType)>("emit_function_return_type_xtype") {
        final_returns_by_function
            .entry(*function)
            .or_default()
            .insert(*xtype);
    }

    let mut individually_valid = BTreeSet::new();
    let mut blocked_values = BTreeSet::new();
    let mut extension_rewrites: BTreeMap<RTLReg, ScalarExtensionTypeRewrite> = BTreeMap::new();
    for (node, proof) in &proofs {
        let types = types_by_value.get(&proof.value);
        let valid = proof.is_closed_v1()
            && !pointer_values.contains(&proof.value)
            && types.is_some_and(|types| {
                if types.is_empty() {
                    return false;
                }
                match (proof.direction, proof.extension) {
                    (ScalarMemoryDirection::Write, ScalarMemoryExtension::Plain) => types
                        .iter()
                        .all(|xtype| scalar_lvalue_integral_type_width(*xtype).is_some()),
                    (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain) => {
                        types.iter().all(|xtype| {
                            scalar_lvalue_integral_type_width(*xtype) == Some(proof.value_width)
                        })
                    }
                    (ScalarMemoryDirection::Read, _) => {
                        signature_extensions.contains(&(*node, proof.clone()))
                            && scalar_lvalue_extension_type_rewrite(proof, types, false).is_some()
                            && returns_by_function
                                .get(&proof.function)
                                .map_or(true, |returns| {
                                    !returns.contains(&proof.value)
                                        || (returns == &BTreeSet::from([proof.value])
                                            && final_returns_by_function.get(&proof.function)
                                                == Some(&BTreeSet::from([
                                                    scalar_lvalue_extension_result_type(proof)
                                                        .expect("validated extension result"),
                                                ])))
                                })
                    }
                    (ScalarMemoryDirection::Write, _) => false,
                }
            });
        if valid {
            individually_valid.insert((*node, proof.clone()));
            if proof.direction == ScalarMemoryDirection::Read
                && proof.extension != ScalarMemoryExtension::Plain
            {
                let rewrite = scalar_lvalue_extension_type_rewrite(
                    proof,
                    types.expect("validated extension type set"),
                    false,
                )
                .expect("validated extension rewrite");
                if extension_rewrites
                    .insert(proof.value, rewrite.clone())
                    .is_some_and(|previous| previous != rewrite)
                {
                    blocked_values.insert(proof.value);
                }
            }
        } else {
            blocked_values.insert(proof.value);
        }
    }

    let accepted: Vec<_> = individually_valid
        .into_iter()
        .filter(|(_, proof)| !blocked_values.contains(&proof.value))
        .collect();
    let accepted_values: BTreeSet<RTLReg> = accepted.iter().map(|(_, proof)| proof.value).collect();
    extension_rewrites.retain(|value, _| accepted_values.contains(value));

    db.rel_set(
        "cminor_scalar_memory_access",
        accepted.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
    if extension_rewrites.is_empty() {
        return;
    }

    type_rows.retain(|(value, xtype)| {
        !extension_rewrites
            .get(value)
            .is_some_and(|rewrite| rewrite.transport_types.contains(xtype))
    });
    type_rows.extend(
        extension_rewrites
            .into_iter()
            .map(|(value, rewrite)| (value, rewrite.result_type)),
    );
    type_rows.sort();
    type_rows.dedup();
    db.rel_set(
        "emit_var_type_candidate",
        type_rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct CshminorPassProgram;


    relation cminor_stmt(Node, CminorStmt);
    relation cminor_scalar_memory_access(Node, ScalarMemoryAccessProof);
    relation signature_scalar_extension_access(Node, ScalarMemoryAccessProof);
    relation trim_jump_table_impl(Node);
    // RTLOptimize's post-rewrite proof that one exact CR8 condition consumes
    // the matching low byte of its still-64-bit intrinsic result.
    relation cr8_byte_compare(Node, RTLReg);
    // Provenance marker kept separate from csharp_stmt_candidate: the
    // structuring pass intentionally selects one canonical statement per
    // node, while Clight may safely retain bounded source alternatives.
    relation scalar_lvalue_candidate(Node, ScalarMemoryAccessProof);
    #[local] relation scalar_lvalue_param_leaf_missing(Node);

    #[local] relation active_cminor_stmt(Node, CminorStmt);
    active_cminor_stmt(node, stmt.clone()) <--
        cminor_stmt(node, stmt),
        !trim_jump_table_impl(node);
    active_cminor_stmt(node, stmt.clone()) <--
        cminor_stmt(node, stmt),
        trim_jump_table_impl(node),
        if matches!(stmt, CminorStmt::Sjumptable(_, _));

    scalar_lvalue_param_leaf_missing(node) <--
        cminor_scalar_memory_access(node, proof),
        for leaf in proof.address_param_leaves.iter(),
        !emit_function_param(proof.function, *leaf);

    scalar_lvalue_candidate(node, proof.clone()) <--
        cminor_scalar_memory_access(node, proof),
        active_cminor_stmt(node, stmt),
        !scalar_lvalue_param_leaf_missing(node),
        if crate::decompile::passes::cminor_pass::scalar_memory_proof_matches_cminor(proof, stmt);

    relation instr_in_function(Node, Address);
    relation rtl_succ(Node, Node);
    relation cminor_fallthrough(Node, Node);
    relation emit_function(Address, Symbol, Node);
    relation emit_function_param(Address, RTLReg);
    relation emit_function_return(Address, RTLReg);
    relation emit_function_return_type_xtype(Address, XType);
    relation next(Address, Address);
    relation emit_var_type_candidate(RTLReg, XType);
    relation idom(Address, Node, Node);
    relation stack_var(Address, Address, i64, RTLReg);
    // rtl_pass export: node-keyed proof that this stack-address expression is
    // &one canonical Win64 home slot.  It intentionally carries no raw offset.
    relation win64_home_address(Node, RTLReg);
    // rtl_pass export: a selected byte-range access through one canonical
    // eight-byte home backing local.  The final booleans are
    // (memory-readable, memory-writable, address-only).
    relation win64_home_backing_access(
        Node, RTLReg, i64, usize, MemoryChunk, bool, bool, bool
    );
    relation win64_home_backing_selected_candidate(Node, Node, RTLInst);
    // rtl_pass export: the single escaped canonical local per (func, offset), used to resolve synthetic-only stack loads that have no per-node stack_var.
    relation slot_escaped_canonical(Address, i64, RTLReg);
    // Noreturn recognition inputs: the always-noreturn symbol set, single-def constants for resolving an error-family status arg, and function_noreturn for user-defined wrappers.
    relation is_known_noreturn_function(Symbol);
    relation single_def_const(RTLReg, Constant);
    relation function_noreturn(Address);
    relation symbol_resolved_addr(Symbol, Address);
    // Memory-indirect call: the rtl_pass surfaces the load addressing so we can inline the function-pointer load into the Scall callee (renders as `(*(base + disp))(args)`).
    relation call_through_memory_load(Node, RTLReg, MemoryChunk, Addressing, Args);
    // signature_pass: the function returns void, so a loop exit flowing to its return stays a plain break; used to gate loop_exit_to_return to value-returning functions only.
    relation emit_function_void_candidate(Address);


    relation stmt_can_fallthrough(Node);
    relation csh_next(Node, Node);
    relation cminor_succ(Node, Node);
    relation func_entry_node(Address, Node);
    relation pred(Address, Node, Node);

    // A call whose callee never returns gets no cminor_succ edge, so its dead tail drops: an always-noreturn symbol, or an error-family symbol with a nonzero constant status arg.
    relation call_node_noreturn(Node);
    call_node_noreturn(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(_, _, Either::Right(Either::Right(name)), _)),
        if crate::abi::is_always_noreturn(name);
    // (2) error-family: join arg 0's register to a single-definition nonzero int constant.
    call_node_noreturn(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(_, _, Either::Right(Either::Right(name)), args)),
        if crate::abi::is_status_noreturn_callee(name),
        single_def_const(status_reg, cst),
        if args.first() == Some(status_reg),
        if matches!(cst, Constant::Ointconst(n) | Constant::Olongconst(n) if *n != 0);
    // (3) user-defined noreturn wrapper called by direct address (die/usage/...).
    call_node_noreturn(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(_, _, Either::Right(Either::Left(addr)), _)),
        function_noreturn(addr);
    // (4) user-defined noreturn wrapper called by symbol that resolves to its entry.
    call_node_noreturn(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(_, _, Either::Right(Either::Right(name)), _)),
        symbol_resolved_addr(name, addr),
        function_noreturn(addr);

    stmt_can_fallthrough(node) <--
        active_cminor_stmt(node, stmt),
        !call_node_noreturn(node),
        if !matches!(stmt,
            CminorStmt::Sjump(_) |
            CminorStmt::Sreturn(_) |
            CminorStmt::Stailcall(_, _, _) |
            CminorStmt::Sjumptable(_, _));

    // Walk `next` chain to first cminor node (transitive closure)
    #[local] relation next_to_cminor(Node, Node);
    next_to_cminor(a, b) <-- next(a, b), active_cminor_stmt(b, _);
    next_to_cminor(a, c) <-- next(a, b), !active_cminor_stmt(b, _), next_to_cminor(b, c);

    // Natural fallthrough via `next` chain
    cminor_fallthrough(node, next_cminor) <--
        active_cminor_stmt(node, _),
        stmt_can_fallthrough(node),
        next_to_cminor(node, next_cminor);

    csh_next(node, next_node) <--
        stmt_can_fallthrough(node),
        cminor_fallthrough(node, next_node);

    // Direct branch targets (target IS a cminor node)
    cminor_succ(*node, *target) <-- active_cminor_stmt(node, ?CminorStmt::Sjump(target)), active_cminor_stmt(target, _);
    cminor_succ(*node, *ifso) <-- active_cminor_stmt(node, ?CminorStmt::Sbranch(_, _, ifso, _)), active_cminor_stmt(ifso, _);
    cminor_succ(*node, *ifnot) <-- active_cminor_stmt(node, ?CminorStmt::Sbranch(_, _, _, ifnot)), active_cminor_stmt(ifnot, _);
    cminor_succ(*node, *ifso) <-- active_cminor_stmt(node, ?CminorStmt::Sifthenelse(_, _, ifso, _)), active_cminor_stmt(ifso, _);
    cminor_succ(*node, *ifnot) <-- active_cminor_stmt(node, ?CminorStmt::Sifthenelse(_, _, _, ifnot)), active_cminor_stmt(ifnot, _);
    cminor_succ(*node, target) <-- active_cminor_stmt(node, ?CminorStmt::Sjumptable(_, targets)), for target in targets.iter(), active_cminor_stmt(target, _);

    // Resolved branch targets (non-cminor target -> nearest cminor via `next`)
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sjump(target)), !active_cminor_stmt(target, _), next_to_cminor(*target, resolved);
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sbranch(_, _, ifso, _)), !active_cminor_stmt(ifso, _), next_to_cminor(*ifso, resolved);
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sbranch(_, _, _, ifnot)), !active_cminor_stmt(ifnot, _), next_to_cminor(*ifnot, resolved);
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sifthenelse(_, _, ifso, _)), !active_cminor_stmt(ifso, _), next_to_cminor(*ifso, resolved);
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sifthenelse(_, _, _, ifnot)), !active_cminor_stmt(ifnot, _), next_to_cminor(*ifnot, resolved);
    cminor_succ(*node, resolved) <-- active_cminor_stmt(node, ?CminorStmt::Sjumptable(_, targets)), for target in targets.iter(), !active_cminor_stmt(target, _), next_to_cminor(*target, resolved);

    // synth_rerouted guards against re-fabricating the rtl_edge_negated-killed fallthrough for synth-carrying nodes (bits 62/63), which would give an unconditional stmt two successors and detach synth stores from their if-arms.
    #[local] relation synth_rerouted(Node);
    synth_rerouted(*node) <--
        rtl_succ(node, dst),
        if (*dst & (3u64 << 62)) != 0,
        active_cminor_stmt(dst, _);

    // Fallthrough suppressed when a synth chain carries the real flow; bridge rules below provide node -> synth -> ... -> next instead.
    cminor_succ(node, next) <--
        csh_next(node, next),
        !synth_rerouted(node),
        active_cminor_stmt(node, _),
        active_cminor_stmt(next, _);

    // Synth-node successor: synthetic nodes lack next entries, so bridge via rtl_succ, which drives the join one tuple per CFG edge; scanning active_cminor_stmt twice was an O(n^2) self-join (36s).
    cminor_succ(*node, *dst) <--
        rtl_succ(node, dst),
        if (*node & (3u64 << 62)) != 0,
        active_cminor_stmt(node, _),
        active_cminor_stmt(dst, _);

    // Synth chain exiting to a trimmed address resolves via `next`, so suppressing the bypass edge above cannot strand the chain.
    cminor_succ(*node, resolved) <--
        rtl_succ(node, dst),
        if (*node & (3u64 << 62)) != 0,
        active_cminor_stmt(node, _),
        !active_cminor_stmt(dst, _),
        next_to_cminor(*dst, resolved);

    // The predecessor of a synth node needs an explicit cminor_succ; `next` skips synth, leaving it unreachable from its real-address predecessor.
    cminor_succ(*node, *dst) <--
        rtl_succ(node, dst),
        if (*dst & (3u64 << 62)) != 0,
        active_cminor_stmt(node, _),
        active_cminor_stmt(dst, _);


    // emit_function entry may point at a non-stmt node (first instr like `mov $0x2,%r8d` later trimmed); follow `next` to the first active cminor node so reachable_from_entry/dom cover the full body.
    func_entry_node(func, entry) <--
        emit_function(func, _, entry),
        active_cminor_stmt(entry, _);

    func_entry_node(func, resolved) <--
        emit_function(func, _, entry),
        !active_cminor_stmt(entry, _),
        next_to_cminor(entry, resolved);


    pred(func, to_node, from_node) <--
        cminor_succ(from_node, to_node),
        instr_in_function(from_node, func),
        instr_in_function(to_node, func);


    // dom/not_idom are #[local] intermediates; only idom is exported.
    #[local] relation dom(Address, Node, Node);
    #[local] relation not_idom(Address, Node, Node);

    // Lattice-based dom: intersection of pred dom-sets via Dual<Set<Node>>; replaces the path_avoiding O(V^2) computation (~115 MB on /bin/ls).
    #[local] lattice strict_dom_set(Address, Node, Dual<Set<Node>>);
    lattice dom_set(Address, Node, Dual<Set<Node>>);

    // Cooper-Harvey-Kennedy dom: dom_set(n) = strict_dom_set(n) U {n}; strict_dom_set(n) = intersect dom_set(p) over preds; entry seed = {entry}.

    // Entry case: dom_set = {entry}.
    dom_set(func, entry, Dual(Set::singleton(*entry))) <--
        func_entry_node(func, entry);

    // Non-entry case: strict_dom_set accumulates intersection of preds' dom_sets
    strict_dom_set(func, n, Dual(p_doms_dual.0.clone())) <--
        cminor_succ(p, n),
        instr_in_function(n, func),
        instr_in_function(p, func),
        dom_set(func, p, p_doms_dual),
        !func_entry_node(func, n);

    // dom_set = strict union {n}.
    dom_set(func, n, Dual(dom_set_with_self(&strict_dual.0, *n))) <--
        strict_dom_set(func, n, strict_dual),
        !func_entry_node(func, n);

    // Materialize dom relation by iterating each node's dom_set
    dom(func, n, *d) <--
        dom_set(func, n, doms_dual),
        for d in doms_dual.0.iter();

    // Immediate dominator: d is not n's idom iff some mid sits strictly between them on the dom chain; dropping the implied dom(n,d) arm removes one arm of the dom-triple-join (20s).
    not_idom(func, n, d) <--
        dom(func, n, mid),
        dom(func, mid, d),
        if mid != n,
        if mid != d;
    idom(func, n, d) <--
        dom(func, n, d),
        !not_idom(func, n, d),
        if n != d;


    relation csharp_stmt_candidate(Node, CsharpminorStmt);

    // Track nodes where a stack address was resolved to Eaddrof via stack_var.
    #[local] relation stack_addr_resolved(Node);

    // Per-function stack offset -> canonical local, so a memory operand materialized only at cminor (a CMP's [rsp+ofs]) resolves to the slot's local instead of a bare *(const).
    #[local] relation stack_local_at(Address, i64, RTLReg);
    stack_local_at(func, ofs, reg) <-- stack_var(func, _, ofs, reg);

    // Canonical home addresses bypass raw-offset stack_var lookup entirely,
    // so a post-prologue local with the same literal displacement cannot be
    // selected.  RTL preserves the 64-bit operation as Oleal.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eop(op, args))),
        if let Operation::Oleal(Addressing::Ainstack(_)) = op,
        if args.is_empty(),
        win64_home_address(node, stack_rtl),
        let var_ident = ident_from_reg(*stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Eaddrof(var_ident));

    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Econst(Constant::Oaddrstack(_)))),
        win64_home_address(node, stack_rtl),
        let var_ident = ident_from_reg(*stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Eaddrof(var_ident));

    // Resolve Eop(Olea(Ainstack(ofs)), []) -> Eaddrof(var_ident) when stack_var maps the offset.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eop(op, args))),
        if let Operation::Olea(Addressing::Ainstack(ofs)) = op,
        if args.is_empty(),
        !win64_home_address(node, _),
        instr_in_function(node, func_start),
        stack_var(func_start, node, *ofs, stack_rtl),
        let var_ident = ident_from_reg(*stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Eaddrof(var_ident));

    // Resolve Eop(Oleal(Ainstack(ofs)), []) -> Eaddrof(var_ident) when stack_var maps the offset.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eop(op, args))),
        if let Operation::Oleal(Addressing::Ainstack(ofs)) = op,
        if args.is_empty(),
        !win64_home_address(node, _),
        instr_in_function(node, func_start),
        stack_var(func_start, node, *ofs, stack_rtl),
        let var_ident = ident_from_reg(*stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Eaddrof(var_ident));

    // Resolve Econst(Oaddrstack(ofs)) -> Eaddrof(var_ident) when stack_var maps the offset. This handles the case where cminor_pass already converted Olea(Ainstack) to Oaddrstack.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Econst(Constant::Oaddrstack(ofs)))),
        !win64_home_address(node, _),
        instr_in_function(node, func_start),
        stack_var(func_start, node, *ofs, stack_rtl),
        let var_ident = ident_from_reg(*stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Eaddrof(var_ident));

    // Resolve an untracked stack Eload to the function's local for that offset; an address-escaped slot uses rtl_pass's single slot_escaped_canonical, not the multi-valued fallback.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eload(_chunk, addr, args))),
        if let Addressing::Ainstack(ofs) = addr,
        if args.is_empty(),
        instr_in_function(node, func_start),
        slot_escaped_canonical(func_start, *ofs, stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Evar(*stack_rtl));

    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eload(_chunk, addr, args))),
        if let Addressing::Ainstack(ofs) = addr,
        if args.is_empty(),
        instr_in_function(node, func_start),
        !slot_escaped_canonical(func_start, *ofs, _),
        stack_local_at(func_start, *ofs, stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Evar(*stack_rtl));

    // Upper half of a wide XMM/YMM spill: bind an untracked base+8 offset to the base slot's local, gated on no def at ofs with a tracked local 8 bytes below.
    csharp_stmt_candidate(node, stmt), stack_addr_resolved(node) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, CminorExpr::Eload(_chunk, addr, args))),
        if let Addressing::Ainstack(ofs) = addr,
        if args.is_empty(),
        instr_in_function(node, func_start),
        !slot_escaped_canonical(func_start, *ofs, _),
        !stack_local_at(func_start, *ofs, _),
        stack_local_at(func_start, *ofs - 8, stack_rtl),
        let stmt = CsharpminorStmt::Sset(*dst, CsharpminorExpr::Evar(*stack_rtl));

    // Generic Sassign -> Sset conversion (skipped when stack_addr_resolved handles the node).
    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, expr)),
        !stack_addr_resolved(node),
        if let Some(converted) = csharp_expr_from_cminor(expr),
        let stmt = CsharpminorStmt::Sset(*dst, converted);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, expr)),
        !stack_addr_resolved(node),
        if let Some(converted) = csharp_expr_from_cminor_sized(expr, true),
        let stmt = CsharpminorStmt::Sset(*dst, converted);

    csharp_stmt_candidate(node, CsharpminorStmt::Snop) <--
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, expr)),
        !stack_addr_resolved(node),
        if csharp_expr_from_cminor(expr).is_none();

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sstore(chunk, addr, args, src)),
        if let Some(addr_expr) = addressing_to_csharp_expr(addr, args.as_slice()),
        let value_expr = CsharpminorExpr::Evar(*src),
        let stmt = CsharpminorStmt::Sstore(chunk.clone(), addr_expr, value_expr);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sstore(chunk, addr, args, src)),
        if let Some(addr_expr) = addressing_to_csharp_expr_sized(addr, args.as_slice(), true),
        let value_expr = CsharpminorExpr::Evar(*src),
        let stmt = CsharpminorStmt::Sstore(chunk.clone(), addr_expr, value_expr);

    csharp_stmt_candidate(node, CsharpminorStmt::Snop) <--
        active_cminor_stmt(node, ?CminorStmt::Sstore(chunk, addr, args, src)),
        if addressing_to_csharp_expr(addr, args.as_slice()).is_none();

    // Default Scall conversion (callee is a plain register reference).
    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(dst, sig, func, args)),
        !call_through_memory_load(node, _, _, _, _),
        let converted_func = match func.clone() {
            Either::Left(reg) => Either::Left(CsharpminorExpr::Evar(reg)),
            Either::Right(id) => Either::Right(id),
        },
        let call_args = args.iter().map(|r| CsharpminorExpr::Evar(*r)).collect(),
        let stmt = CsharpminorStmt::Scall(dst.clone(), sig.clone(), converted_func, call_args);

    // Memory-indirect call: render callee as the load expression `(*(base + disp))` so the function pointer slot is dereferenced explicitly in the C output.
    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Scall(dst, sig, _, args)),
        call_through_memory_load(node, _temp, chunk, addressing, load_args),
        if let Some(addr_expr) = addressing_to_csharp_expr_sized(addressing, load_args.as_slice(), true),
        let converted_func = Either::Left(CsharpminorExpr::Eload(*chunk, Box::new(addr_expr))),
        let call_args = args.iter().map(|r| CsharpminorExpr::Evar(*r)).collect(),
        let stmt = CsharpminorStmt::Scall(dst.clone(), sig.clone(), converted_func, call_args);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Stailcall(sig, func, args)),
        let converted_func = match func.clone() {
            Either::Left(reg) => Either::Left(CsharpminorExpr::Evar(reg)),
            Either::Right(id) => Either::Right(id),
        },
        let call_args = args.iter().map(|r| CsharpminorExpr::Evar(*r)).collect(),
        let stmt = CsharpminorStmt::Stailcall(sig.clone(), converted_func, call_args);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sbuiltin(dst, name, args, res)),
        let converted_args = args.iter().map(csharp_builtin_arg_from).collect(),
        let converted_res = csharp_builtin_arg_from(&res),
        let stmt = CsharpminorStmt::Sbuiltin(dst.clone(), name.clone(), converted_args, converted_res);

    #[local] relation cr8_byte_condition_row_count(Node, usize);
    cr8_byte_condition_row_count(*node, count) <--
        cr8_byte_compare(node, _),
        agg count = ascent::aggregators::count() in active_cminor_stmt(node, _);

    // A marker is usable only when it identifies one value at this condition.
    // Competing marker rows retain the ordinary 32-bit condition rendering.
    #[local] relation cr8_byte_marker_count(Node, usize);
    cr8_byte_marker_count(*node, count) <--
        cr8_byte_compare(node, _),
        agg count = ascent::aggregators::count() in cr8_byte_compare(node, _);

    #[local] relation cr8_byte_condition_resolved(Node);
    cr8_byte_condition_resolved(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Sbranch(cond, regs, _, _)),
        cr8_byte_compare(node, value),
        cr8_byte_condition_row_count(node, 1),
        cr8_byte_marker_count(node, 1),
        if matches!(cond, Condition::Ccompuimm(_, 1)),
        if regs.as_slice() == [*value];

    cr8_byte_condition_resolved(*node) <--
        active_cminor_stmt(node, ?CminorStmt::Sifthenelse(cond, regs, _, _)),
        cr8_byte_compare(node, value),
        cr8_byte_condition_row_count(node, 1),
        cr8_byte_marker_count(node, 1),
        if matches!(cond, Condition::Ccompuimm(_, 1)),
        if regs.as_slice() == [*value];

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sbranch(cond, regs, ifso, ifnot)),
        cr8_byte_compare(node, value),
        cr8_byte_condition_row_count(node, 1),
        cr8_byte_marker_count(node, 1),
        if matches!(cond, Condition::Ccompuimm(_, 1)),
        if regs.as_slice() == [*value],
        let converted_args = vec![CsharpminorExpr::Eunop(
            CminorUnop::Ocast8unsigned,
            Box::new(CsharpminorExpr::Evar(*value)),
        )],
        let stmt = CsharpminorStmt::Scond(cond.clone(), converted_args, *ifso, *ifnot);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sifthenelse(cond, regs, ifso, ifnot)),
        cr8_byte_compare(node, value),
        cr8_byte_condition_row_count(node, 1),
        cr8_byte_marker_count(node, 1),
        if matches!(cond, Condition::Ccompuimm(_, 1)),
        if regs.as_slice() == [*value],
        let converted_args = vec![CsharpminorExpr::Eunop(
            CminorUnop::Ocast8unsigned,
            Box::new(CsharpminorExpr::Evar(*value)),
        )],
        let stmt = CsharpminorStmt::Scond(cond.clone(), converted_args, *ifso, *ifnot);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sbranch(cond, regs, ifso, ifnot)),
        !cr8_byte_condition_resolved(node),
        let converted_args = regs
            .iter()
            .map(|r| CsharpminorExpr::Evar(*r))
            .collect::<Vec<CsharpminorExpr>>(),
        let stmt = CsharpminorStmt::Scond(cond.clone(), converted_args, *ifso, *ifnot);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sifthenelse(cond, regs, ifso, ifnot)),
        !cr8_byte_condition_resolved(node),
        let converted_args = regs.iter().map(|r| CsharpminorExpr::Evar(*r)).collect(),
        let stmt = CsharpminorStmt::Scond(cond.clone(), converted_args, *ifso, *ifnot);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sjumptable(reg, targets)),
        let stmt = CsharpminorStmt::Sjumptable(CsharpminorExpr::Evar(*reg), targets.clone());

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Sreturn(result)),
        let converted = CsharpminorExpr::Evar(*result),
        let stmt = CsharpminorStmt::Sreturn(converted);

    csharp_stmt_candidate(node, stmt) <--
        active_cminor_stmt(node, ?CminorStmt::Snop),
        let stmt = CsharpminorStmt::Snop;

    // Drop unreachable bare Sjumps: the structurer reads such an orphan goto as a loop back-edge, making the loop span everything between its label and the dead jump.
    relation cminor_node_has_pred(Node);
    cminor_node_has_pred(*n) <-- cminor_succ(_, n);

    csharp_stmt_candidate(node, CsharpminorStmt::Sjump(*target)) <--
        active_cminor_stmt(node, ?CminorStmt::Sjump(target)),
        cminor_node_has_pred(node);



    relation func_exit_node(Address, Node);

    func_exit_node(func, node) <--
        csharp_stmt_candidate(node, ?CsharpminorStmt::Sreturn(_)),
        instr_in_function(node, func);

    func_exit_node(func, node) <--
        csharp_stmt_candidate(node, ?CsharpminorStmt::Stailcall(_, _, _)),
        instr_in_function(node, func);


    relation loop_back_edge(Address, Node, Node);
    relation loop_head(Address, Node);

    loop_back_edge(func, latch, header) <--
        cminor_succ(latch, header),
        instr_in_function(latch, func),
        dom(func, latch, header);

    loop_head(func, header) <--
        loop_back_edge(func, _, header);


    // The natural loop body (reverse-from-latch, dominance-filtered), kept as a STABLE seed the abort-region rule reads without feeding back, so folding cannot cascade.
    relation loop_body_core(Address, Node, Node);

    loop_body_core(func, header, header) <--
        loop_back_edge(func, _, header);

    loop_body_core(func, header, latch) <--
        loop_back_edge(func, latch, header);

    loop_body_core(func, header, p) <--
        loop_body_core(func, header, node),
        pred(func, node, p),
        dom(func, p, header),
        if *p != *header;

    // n reaches a function return along the CFG; this is what separates a loop's normal exit continuation from an in-loop abort arm that only hits a noreturn sink.
    relation reaches_return(Node);
    reaches_return(n) <-- func_exit_node(_, n);
    reaches_return(p) <-- cminor_succ(p, n), reaches_return(n);

    // Require at least one exit that reaches a return, or a wholly-noreturn function would fold its post-loop tail into an inner loop.
    relation loop_has_returning_exit(Address, Node);
    loop_has_returning_exit(func, header) <--
        loop_body_core(func, header, b),
        cminor_succ(b, t),
        !loop_body_core(func, header, t),
        reaches_return(t);

    // In-loop noreturn abort region: entered from the body via an exit edge whose target never reaches a return, grown forward inside the header's dominance region so the dispatch default stays in the loop.
    relation loop_abort_region(Address, Node, Node);
    loop_abort_region(func, header, t) <--
        loop_has_returning_exit(func, header),
        loop_body_core(func, header, b),
        cminor_succ(b, t),
        !loop_body_core(func, header, t),
        !reaches_return(t),
        dom(func, t, header);
    loop_abort_region(func, header, n) <--
        loop_abort_region(func, header, m),
        cminor_succ(m, n),
        !loop_body_core(func, header, n),
        !reaches_return(n),
        dom(func, n, header);

    relation loop_body(Address, Node, Node);
    loop_body(func, header, n) <--
        loop_body_core(func, header, n);
    loop_body(func, header, n) <--
        loop_abort_region(func, header, n);

    // CF-2/O-6 (removed 2026-06-10): SCC-body-extension was provably empty for recognized loops, and headerless-SCC creation regressed goto_per_func by 11%; reviving it needs emission costs cut first.


    relation loop_nesting_via_idom(Address, Node, Node);
    relation directly_nested_dom(Address, Node, Node);
    relation intermediate_loop_dom(Address, Node, Node);
    relation immediate_loop_parent(Address, Node, Node);

    loop_nesting_via_idom(func, outer, inner) <--
        loop_head(func, outer),
        loop_head(func, inner),
        dom(func, inner, outer),
        loop_body(func, outer, inner),
        if *outer != *inner;

    directly_nested_dom(func, outer, inner) <--
        loop_nesting_via_idom(func, outer, inner),
        !intermediate_loop_dom(func, outer, inner);

    intermediate_loop_dom(func, outer, inner) <--
        loop_nesting_via_idom(func, outer, mid),
        loop_nesting_via_idom(func, mid, inner);

    immediate_loop_parent(func, inner, outer) <--
        directly_nested_dom(func, outer, inner);


    relation innermost_loop(Address, Node, Node);
    relation node_in_nested_inner_loop(Address, Node, Node);

    innermost_loop(func, node, header) <--
        loop_body(func, header, node),
        !node_in_nested_inner_loop(func, header, node);

    node_in_nested_inner_loop(func, outer, node) <--
        directly_nested_dom(func, outer, inner),
        loop_body(func, inner, node),
        if *node != *inner;


    relation loop_exit_edge(Address, Node, Node, Node);
    relation loop_exit_branch(Address, Node, Node, Condition, Arc<Vec<CsharpminorExpr>>, Node, Node, bool);
    relation loop_jumptable_exit(Address, Node, Node, Node);
    relation loop_exit_count(Address, Node, usize);
    relation loop_has_exit(Address, Node);
    relation exit_at_header(Address, Node);
    relation exit_at_latch(Address, Node, Node);

    loop_exit_edge(func, header, from_node, to_node) <--
        loop_body(func, header, from_node),
        cminor_succ(from_node, to_node),
        !loop_body(func, header, to_node);

    // Substitute dst->src for a loop-exit comparison reading a copy that is eliminated downstream, using the copy at the branch's unique predecessor where v == src provably holds.
    relation cond_other_pred(Node, Node);
    cond_other_pred(n, p) <--
        cminor_succ(p, n),
        cminor_succ(p2, n),
        if p2 != p;

    relation cond_read_copy(Node, RTLReg, RTLReg);
    cond_read_copy(*branch_node, *v, *w) <--
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Scond(_, _, _, _)),
        cminor_succ(pred, branch_node),
        !cond_other_pred(branch_node, pred),
        csharp_stmt_candidate(pred, ?CsharpminorStmt::Sset(v, src_expr)),
        if let CsharpminorExpr::Evar(w) = src_expr;

    loop_exit_branch(func, header, branch_node, cond.clone(), Arc::new(args.clone()), *ifso, *ifnot, false) <--
        loop_body(func, header, branch_node),
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Scond(cond, args, ifso, ifnot)),
        !loop_body(func, header, *ifso),
        loop_body(func, header, *ifnot),
        !cond_read_copy(branch_node, _, _);

    loop_exit_branch(func, header, branch_node, cond.clone(), Arc::new(cond_subst_exprs(args, *v, *w)), *ifso, *ifnot, false) <--
        loop_body(func, header, branch_node),
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Scond(cond, args, ifso, ifnot)),
        !loop_body(func, header, *ifso),
        loop_body(func, header, *ifnot),
        cond_read_copy(branch_node, v, w);

    loop_exit_branch(func, header, branch_node, cond.clone(), Arc::new(args.clone()), *ifnot, *ifso, true) <--
        loop_body(func, header, branch_node),
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Scond(cond, args, ifso, ifnot)),
        loop_body(func, header, *ifso),
        !loop_body(func, header, *ifnot),
        !cond_read_copy(branch_node, _, _);

    loop_exit_branch(func, header, branch_node, cond.clone(), Arc::new(cond_subst_exprs(args, *v, *w)), *ifnot, *ifso, true) <--
        loop_body(func, header, branch_node),
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Scond(cond, args, ifso, ifnot)),
        loop_body(func, header, *ifso),
        !loop_body(func, header, *ifnot),
        cond_read_copy(branch_node, v, w);

    loop_jumptable_exit(func, header, branch_node, *target) <--
        loop_body(func, header, branch_node),
        csharp_stmt_candidate(branch_node, ?CsharpminorStmt::Sjumptable(_, targets)),
        for target in targets.iter(),
        !loop_body(func, header, *target);

    relation loop_unconditional_exit(Address, Node, Node);

    loop_unconditional_exit(func, header, node) <--
        loop_exit_edge(func, header, node, _),
        !loop_exit_branch(func, header, node, _, _, _, _, _),
        !loop_jumptable_exit(func, header, node, _);

    loop_exit_count(func, header, count) <--
        loop_head(func, header),
        agg count = ascent::aggregators::count() in loop_exit_branch(func, header, _, _, _, _, _, _);

    loop_has_exit(func, header) <--
        loop_exit_count(func, header, count),
        if *count > 0;

    loop_has_exit(func, header) <--
        loop_jumptable_exit(func, header, _, _);

    loop_has_exit(func, header) <--
        loop_exit_edge(func, header, _, _);

    exit_at_header(func, header) <--
        loop_exit_branch(func, header, node, _, _, _, _, _),
        if *node == *header;

    exit_at_header(func, header) <--
        loop_unconditional_exit(func, header, node),
        if *node == *header;

    exit_at_latch(func, header, latch) <--
        loop_back_edge(func, latch, header),
        loop_exit_branch(func, header, latch, _, _, _, _, _),
        if *latch != *header;

    exit_at_latch(func, header, latch) <--
        loop_back_edge(func, latch, header),
        loop_unconditional_exit(func, header, latch),
        if *latch != *header;


    relation loop_type(Address, Node, LoopType);
    relation primary_exit_node(Address, Node, Node);

    loop_type(func, header, LoopType::Infinite) <--
        loop_head(func, header),
        !loop_has_exit(func, header);

    loop_type(func, header, LoopType::PreTested) <--
        loop_head(func, header),
        exit_at_header(func, header);

    loop_type(func, header, LoopType::PostTested) <--
        loop_head(func, header),
        exit_at_latch(func, header, _),
        !exit_at_header(func, header);

    loop_type(func, header, LoopType::MidTested) <--
        loop_head(func, header),
        loop_has_exit(func, header),
        !exit_at_header(func, header),
        !exit_at_latch(func, header, _);

    primary_exit_node(func, header, header) <--
        loop_type(func, header, LoopType::PreTested);

    primary_exit_node(func, header, latch) <--
        loop_type(func, header, LoopType::PostTested),
        exit_at_latch(func, header, latch);

    primary_exit_node(func, header, exit_node) <--
        loop_type(func, header, LoopType::MidTested),
        loop_exit_count(func, header, 1),
        loop_exit_branch(func, header, exit_node, _, _, _, _, _);


    relation loop_self_modify(Address, Node, Node, RTLReg);
    relation loop_increment(Address, Node, Node, RTLReg);
    relation exit_condition_uses_reg(Address, Node, RTLReg);
    relation induction_var(Address, Node, RTLReg);
    relation has_induction_var(Address, Node);

    loop_self_modify(func, header, node, dst) <--
        loop_body(func, header, node),
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, expr)),
        if crate::decompile::passes::clight_pass::extract_self_modifying_reg(dst, expr).is_some();

    loop_increment(func, header, node, dst) <--
        loop_body(func, header, node),
        active_cminor_stmt(node, ?CminorStmt::Sassign(dst, expr)),
        if let CminorExpr::Ebinop(op, arg1, arg2) = expr,
        if crate::decompile::passes::clight_pass::is_increment_decrement_op(op),
        if *arg1 == *dst || *arg2 == *dst;

    exit_condition_uses_reg(func, header, reg) <--
        primary_exit_node(func, header, exit_node),
        loop_exit_branch(func, header, exit_node, _, args, _, _, _),
        let vars = crate::decompile::passes::clight_pass::extract_vars_from_csharp_exprs(args.as_slice()),
        for reg in vars;

    induction_var(func, header, reg) <--
        loop_self_modify(func, header, _, reg),
        exit_condition_uses_reg(func, header, reg);

    induction_var(func, header, reg) <--
        loop_increment(func, header, _, reg),
        exit_condition_uses_reg(func, header, reg);

    has_induction_var(func, header) <--
        induction_var(func, header, _);


    relation body_stmt(Address, Node, Node);
    relation body_stmt_count(Address, Node, usize);
    relation has_substantial_body(Address, Node);
    relation step_node(Address, Node, Node, RTLReg);
    relation valid_loop(Address, Node);

    body_stmt(func, header, node) <--
        loop_body(func, header, node),
        active_cminor_stmt(node, stmt),
        if !crate::decompile::passes::clight_pass::is_trivial_cminor_stmt(stmt),
        if *node != *header;

    body_stmt_count(func, header, count) <--
        loop_head(func, header),
        agg count = ascent::aggregators::count() in body_stmt(func, header, _);

    has_substantial_body(func, header) <--
        body_stmt_count(func, header, count),
        if *count >= 1;

    step_node(func, header, step, ind_var) <--
        induction_var(func, header, ind_var),
        loop_increment(func, header, step, ind_var);

    valid_loop(func, header) <--
        has_induction_var(func, header),
        loop_has_exit(func, header);

    valid_loop(func, header) <--
        has_substantial_body(func, header),
        loop_has_exit(func, header);

    valid_loop(func, header) <--
        loop_type(func, header, LoopType::Infinite),
        has_substantial_body(func, header);

    valid_loop(func, header) <--
        loop_back_edge(func, _, header),
        body_stmt_count(func, header, count),
        if *count >= 1,
        loop_has_exit(func, header);


    relation emit_loop_body(Address, Node, Node);
    relation emit_loop_exit(Address, Node, Node, Condition, Arc<Vec<CsharpminorExpr>>, Node, Node);
    relation trim_step(Address, Node, Node);
    relation emit_break_stmt(Address, Node, Node, ClightStmt);
    // A non-primary loop exit flowing to a function return rather than post-loop code: a valueless break would drop the returned value, so select tail-duplicates value_node and the return into it.
    relation loop_exit_to_return(Address, Node, Node, Node, Node);
    relation node_multi_pred(Address, Node);
    // node flows to a function return (ret_node) through zero or more pure jump nodes (no intervening assignment, branch, or side effect), so a value-set-then-`jmp epilogue` exit is recognized as a return even with the jump on its own node between them.
    relation returns_after(Address, Node, Node);
    // Any loop exit flowing to a function return, with the per-loop count: a genuine early-return defect needs >=2, while a single return exit must keep its exit as the loop condition.
    relation loop_return_exit_any(Address, Node, Node);
    relation loop_return_exit_count(Address, Node, usize);
    relation node_owned_by_loop(Address, Node, Node);

    emit_loop_body(func, header, body_node) <--
        valid_loop(func, header),
        loop_body(func, header, body_node);

    emit_loop_exit(func, header, branch_node, cond.clone(), args.clone(), *exit_target, *continue_target) <--
        valid_loop(func, header),
        loop_exit_branch(func, header, branch_node, cond, args, exit_target, continue_target, _);

    trim_step(func, header, step) <--
        step_node(func, header, step, _);

    emit_break_stmt(func, header, exit_node, break_stmt) <--
        valid_loop(func, header),
        loop_exit_branch(func, header, exit_node, cond, args, _, _, inverted),
        !primary_exit_node(func, header, exit_node),
        agg all_var_types = crate::decompile::passes::clight_pass::collect_all_var_types(reg, xty) in emit_var_type_candidate(reg, xty),
        let vars_used = crate::decompile::passes::clight_pass::extract_vars_from_csharp_exprs(args.as_slice()),
        let var_types = crate::decompile::passes::clight_pass::filter_and_build_multi_var_type_map(&all_var_types, &vars_used),
        let break_cond_raw = if *inverted {
            crate::decompile::passes::clight_pass::invert_condition(&cond)
        } else {
            cond.clone()
        },
        if let Some(break_cond) = crate::decompile::passes::clight_pass::clight_condition_expr_with_types(&break_cond_raw, args.as_slice(), &var_types),
        let break_stmt = ClightStmt::Sifthenelse(
            break_cond,
            Box::new(ClightStmt::Sbreak),
            Box::new(ClightStmt::Sskip),
        );

    node_multi_pred(func, n) <--
        pred(func, n, p1),
        pred(func, n, p2),
        if p1 != p2;

    returns_after(func, node, node) <--
        func_exit_node(func, node);

    returns_after(func, node, ret_node) <--
        returns_after(func, succ, ret_node),
        cminor_succ(node, succ),
        active_cminor_stmt(node, ?CminorStmt::Sjump(_)),
        instr_in_function(node, func);

    loop_return_exit_any(func, header, exit_node) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        active_cminor_stmt(exit_target, ?CminorStmt::Sassign(_, _)),
        cminor_succ(exit_target, succ),
        returns_after(func, succ, _);

    loop_return_exit_any(func, header, exit_node) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        func_exit_node(func, exit_target);

    loop_return_exit_count(func, header, c) <--
        loop_head(func, header),
        agg c = ascent::aggregators::count() in loop_return_exit_any(func, header, _);

    // A non-primary return-exit tail-duplicates its return unconditionally, since a valueless break would fall through to whatever the layout places after the loop and drop the value.
    loop_exit_to_return(func, header, exit_node, exit_target, ret_node) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        !emit_function_void_candidate(func),
        !primary_exit_node(func, header, exit_node),
        !loop_body(func, header, exit_target),
        active_cminor_stmt(exit_target, ?CminorStmt::Sassign(_, _)),
        cminor_succ(exit_target, succ),
        returns_after(func, succ, ret_node),
        pred(func, exit_target, exit_node);

    // 0-hop: the exit target is itself the function return (value computed inline at the exit).
    loop_exit_to_return(func, header, exit_node, exit_target, exit_target) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        !emit_function_void_candidate(func),
        !primary_exit_node(func, header, exit_node),
        !loop_body(func, header, exit_target),
        func_exit_node(func, exit_target),
        pred(func, exit_target, exit_node);

    // The primary return-exit tail-duplicates only when a second distinct return-exit exists; as the sole return-exit it stays the loop condition.
    loop_exit_to_return(func, header, exit_node, exit_target, ret_node) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        !emit_function_void_candidate(func),
        primary_exit_node(func, header, exit_node),
        !loop_body(func, header, exit_target),
        active_cminor_stmt(exit_target, ?CminorStmt::Sassign(_, _)),
        cminor_succ(exit_target, succ),
        returns_after(func, succ, ret_node),
        pred(func, exit_target, exit_node),
        loop_return_exit_count(func, header, c),
        if *c >= 2;

    loop_exit_to_return(func, header, exit_node, exit_target, exit_target) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        !emit_function_void_candidate(func),
        primary_exit_node(func, header, exit_node),
        !loop_body(func, header, exit_target),
        func_exit_node(func, exit_target),
        pred(func, exit_target, exit_node),
        loop_return_exit_count(func, header, c),
        if *c >= 2;

    // Relocate a break-edge flag write (reg = INT_CONSTANT only) before the break so the post-loop test sees it; a loop increment is not a constant and is never relocated.
    relation loop_exit_pre_break_assign(Address, Node, Node, Node);
    loop_exit_pre_break_assign(func, header, exit_node, exit_target) <--
        loop_exit_branch(func, header, exit_node, _, _, exit_target, _, _),
        !loop_body(func, header, exit_target),
        active_cminor_stmt(exit_target, ?CminorStmt::Sassign(_, rhs)),
        if matches!(rhs, CminorExpr::Econst(Constant::Ointconst(_)) | CminorExpr::Econst(Constant::Olongconst(_))),
        pred(func, exit_target, exit_node),
        !node_multi_pred(func, exit_target),
        !func_exit_node(func, exit_target),
        cminor_succ(exit_target, succ),
        !returns_after(func, succ, _);

    node_owned_by_loop(func, node, header) <--
        innermost_loop(func, node, header),
        valid_loop(func, header),
        !loop_head(func, node);

    node_owned_by_loop(func, node, parent) <--
        loop_head(func, node),
        valid_loop(func, node),
        immediate_loop_parent(func, node, parent);


    relation ternary_true_assignment(Address, Node, RTLReg, CsharpminorExpr, Node);
    relation ternary_false_assignment(Address, Node, RTLReg, CsharpminorExpr, Node);
    relation valid_ternary(Address, Node, RTLReg, CsharpminorExpr, CsharpminorExpr, Node);

    ternary_true_assignment(func, branch, var, expr.clone(), merge) <--
        csharp_stmt_candidate(branch, ?CsharpminorStmt::Scond(_, _, ifso, _)),
        instr_in_function(branch, func),
        csharp_stmt_candidate(ifso, ?CsharpminorStmt::Sset(var, expr)),
        cminor_succ(ifso, merge);

    ternary_false_assignment(func, branch, var, expr.clone(), merge) <--
        csharp_stmt_candidate(branch, ?CsharpminorStmt::Scond(_, _, _, ifnot)),
        instr_in_function(branch, func),
        csharp_stmt_candidate(ifnot, ?CsharpminorStmt::Sset(var, expr)),
        cminor_succ(ifnot, merge);

    valid_ternary(func, branch, var, true_expr.clone(), false_expr.clone(), merge) <--
        ternary_true_assignment(func, branch, var, true_expr, merge),
        ternary_false_assignment(func, branch, var, false_expr, merge),
        !valid_loop(func, branch),
        !node_owned_by_loop(func, branch, _);

}

pub struct CshminorPass;

impl CshminorPass {
    fn prepare_jump_tables(db: &mut DecompileDB) {
        if db
            .rel_iter::<(Node, usize, Node)>("jump_table_target")
            .next()
            .is_none()
        {
            return;
        }

        let mut tables: HashMap<Node, Vec<(usize, Node)>> = HashMap::new();
        for &(jmp, idx, target) in db.rel_iter::<(Node, usize, Node)>("jump_table_target") {
            tables.entry(jmp).or_default().push((idx, target));
        }
        for targets in tables.values_mut() {
            targets.sort_by_key(|(idx, _)| *idx);
        }

        let mut impls: HashMap<Node, Vec<Node>> = HashMap::new();
        for &(impl_addr, jmp_addr) in db.rel_iter::<(Node, Node)>("jump_table_impl") {
            impls.entry(jmp_addr).or_default().push(impl_addr);
        }

        let mut impl_set = std::collections::HashSet::new();
        for &(impl_addr, _) in db.rel_iter::<(Node, Node)>("jump_table_impl") {
            impl_set.insert(impl_addr);
        }

        // Sort jump-table nodes and pick lowest RTLReg discriminant for determinism.
        let mut jmp_nodes: Vec<Node> = tables.keys().copied().collect();
        jmp_nodes.sort();
        for jmp_node in jmp_nodes {
            let targets = &tables[&jmp_node];
            let impl_nodes = match impls.get(&jmp_node) {
                Some(v) => v,
                None => continue,
            };
            let entry = match impl_nodes.iter().min() {
                Some(&e) => e,
                None => continue,
            };
            let impl_set_local: std::collections::HashSet<Node> = impl_nodes
                .iter()
                .copied()
                .chain(std::iter::once(jmp_node))
                .collect();

            let ordered_targets: Vec<Node> = targets.iter().map(|(_, t)| *t).collect();

            // Pick smallest RTLReg across plausible discriminants for deterministic output.
            let mut candidates: Vec<RTLReg> = Vec::new();
            for (_node, stmt) in db.rel_iter::<(Node, CminorStmt)>("cminor_stmt") {
                match stmt {
                    CminorStmt::Sbranch(_, args, ifso, ifnot)
                    | CminorStmt::Sifthenelse(_, args, ifso, ifnot) => {
                        if (impl_set_local.contains(ifso) || impl_set_local.contains(ifnot))
                            && args.len() == 1
                        {
                            candidates.push(args[0]);
                        }
                    }
                    _ => {}
                }
            }

            if candidates.is_empty() {
                let cmp_addrs: Vec<Node> = db
                    .rel_iter::<(Node, Node)>("jump_table_cmp")
                    .filter(|(j, _)| *j == jmp_node)
                    .map(|(_, a)| *a)
                    .collect();
                let index_regs: Vec<&'static str> = db
                    .rel_iter::<(Node, &'static str)>("jump_table_index_reg")
                    .filter(|(j, _)| *j == jmp_node)
                    .map(|(_, r)| *r)
                    .collect();
                for &cmp_addr in &cmp_addrs {
                    for &idx_name in &index_regs {
                        let mreg = Mreg::x86(idx_name);
                        for &(addr, ref reg, rtl_reg) in
                            db.rel_iter::<(Node, Mreg, RTLReg)>("reg_rtl")
                        {
                            if addr == cmp_addr && *reg == mreg {
                                candidates.push(rtl_reg);
                            }
                        }
                    }
                }
            }

            candidates.sort();
            candidates.dedup();
            let reg = match candidates.into_iter().next() {
                Some(r) => r,
                None => continue,
            };

            db.rel_push(
                "cminor_stmt",
                (
                    entry,
                    CminorStmt::Sjumptable(reg, Arc::new(ordered_targets)),
                ),
            );
        }

        let mut impl_nodes_sorted: Vec<Node> = impl_set.into_iter().collect();
        impl_nodes_sorted.sort();
        for node in impl_nodes_sorted {
            db.rel_push("trim_jump_table_impl", (node,));
        }
    }
}

impl IRPass for CshminorPass {
    fn name(&self) -> &'static str {
        "cshminor"
    }

    fn run(&self, db: &mut DecompileDB) {
        filter_cr8_byte_compares_with_incompatible_types(db);
        filter_scalar_lvalues_with_final_types(db);
        Self::prepare_jump_tables(db);

        run_pass!(db, CshminorPassProgram);
        materialize_win64_home_backing(db);
    }

    fn extra_reads(&self) -> &'static [&'static str] {
        &[
            "jump_table_target",
            "jump_table_impl",
            "jump_table_cmp",
            "jump_table_index_reg",
            "reg_rtl",
            "is_ptr",
            "win64_home_backing_access",
            "win64_home_backing_selected_candidate",
        ]
    }

    declare_io_from!(CshminorPassProgram);
}

#[cfg(test)]
mod cr8_byte_condition_tests {
    use super::*;
    use crate::x86::op::Comparison;

    const CONDITION: Node = 0x1010;
    const TAKEN: Node = 0x1020;
    const FALLTHROUGH: Node = 0x1030;
    const VALUE: RTLReg = 0x8000_0000_0000_1010;

    fn condition_db(
        markers: &[RTLReg],
        competing_condition: bool,
        incompatible_type: Option<(RTLReg, XType)>,
        is_pointer: bool,
    ) -> DecompileDB {
        let mut db = DecompileDB::default();
        let condition = CminorStmt::Sbranch(
            Condition::Ccompuimm(Comparison::Cle, 1),
            Arc::new(vec![VALUE]),
            TAKEN,
            FALLTHROUGH,
        );
        db.rel_push("cminor_stmt", (CONDITION, condition));
        if competing_condition {
            db.rel_push(
                "cminor_stmt",
                (
                    CONDITION,
                    CminorStmt::Sbranch(
                        Condition::Ccompuimm(Comparison::Cle, 1),
                        Arc::new(vec![VALUE]),
                        FALLTHROUGH,
                        TAKEN,
                    ),
                ),
            );
        }
        for value in markers {
            db.rel_push("cr8_byte_compare", (CONDITION, *value));
        }
        db.rel_push("emit_var_type_candidate", (VALUE, XType::Xint));
        if let Some((value, xtype)) = incompatible_type {
            db.rel_push("emit_var_type_candidate", (value, xtype));
        }
        if is_pointer {
            db.rel_push("is_ptr", (VALUE,));
        }
        CshminorPass.run(&mut db);
        db
    }

    fn condition_args(db: &DecompileDB) -> Vec<Vec<CsharpminorExpr>> {
        db.rel_iter::<(Node, CsharpminorStmt)>("csharp_stmt_candidate")
            .filter_map(|(node, stmt)| match stmt {
                CsharpminorStmt::Scond(_, args, _, _) if *node == CONDITION => Some(args.clone()),
                _ => None,
            })
            .collect()
    }

    fn has_cr8_long_type(db: &DecompileDB, value: RTLReg) -> bool {
        db.rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .any(|(reg, xtype)| *reg == value && *xtype == XType::Xlongunsigned)
    }

    #[test]
    fn unique_cr8_marker_narrows_only_the_condition_expression() {
        let db = condition_db(&[VALUE], false, None, false);
        assert_eq!(
            condition_args(&db),
            vec![vec![CsharpminorExpr::Eunop(
                CminorUnop::Ocast8unsigned,
                Box::new(CsharpminorExpr::Evar(VALUE)),
            )]]
        );
        assert!(!has_cr8_long_type(&db, VALUE));
    }

    #[test]
    fn competing_cr8_marker_values_fail_closed_to_generic_condition() {
        let db = condition_db(&[VALUE, VALUE + 1], false, None, false);
        assert_eq!(
            condition_args(&db),
            vec![vec![CsharpminorExpr::Evar(VALUE)]]
        );
        assert!(!has_cr8_long_type(&db, VALUE));
    }

    #[test]
    fn unresolved_or_competing_condition_rows_fail_closed_to_generic_condition() {
        let mismatched = condition_db(&[VALUE + 1], false, None, false);
        assert_eq!(
            condition_args(&mismatched),
            vec![vec![CsharpminorExpr::Evar(VALUE)]]
        );
        assert!(!has_cr8_long_type(&mismatched, VALUE));

        let competing = condition_db(&[VALUE], true, None, false);
        let args = condition_args(&competing);
        assert!(!args.is_empty());
        assert!(args
            .iter()
            .all(|row| row.as_slice() == [CsharpminorExpr::Evar(VALUE)]));
        assert!(!has_cr8_long_type(&competing, VALUE));
    }

    #[test]
    fn incompatible_final_type_or_pointer_evidence_rejects_the_marker() {
        for db in [
            condition_db(&[VALUE], false, Some((VALUE, XType::Xfloat)), false),
            condition_db(&[VALUE], false, None, true),
            condition_db(
                &[VALUE, VALUE + 1],
                false,
                Some((VALUE + 1, XType::Xfloat)),
                false,
            ),
        ] {
            assert_eq!(
                condition_args(&db),
                vec![vec![CsharpminorExpr::Evar(VALUE)]]
            );
            assert!(!has_cr8_long_type(&db, VALUE));
            assert!(!db
                .rel_iter::<(Node, RTLReg)>("cr8_byte_compare")
                .any(|(node, _)| *node == CONDITION));
        }
    }
}

#[cfg(test)]
mod scalar_lvalue_final_param_tests {
    use super::*;

    const FUNCTION: Address = 0x2000;
    const NODE: Node = 0x2010;
    const BASE: RTLReg = 0x8000_0000_0000_2010;
    const INDEX: RTLReg = 0x8000_0000_0000_2020;
    const VALUE: RTLReg = 0x8000_0000_0000_2030;

    fn proof(
        direction: ScalarMemoryDirection,
        extension: ScalarMemoryExtension,
        width: usize,
        value_width: usize,
        chunk: MemoryChunk,
    ) -> ScalarMemoryAccessProof {
        ScalarMemoryAccessProof {
            function: FUNCTION,
            origin_node: NODE,
            selected_node: NODE,
            operand: "scalar_final_param_memory",
            direction,
            extension,
            address_size: 8,
            base_register: Mreg::CX,
            index_register: Some(Mreg::DX),
            scale: 4,
            displacement: 8,
            width,
            value_width,
            downstream_value_width: (direction == ScalarMemoryDirection::Read)
                .then_some(value_width),
            chunk,
            base_value: Some(BASE),
            index_value: Some(INDEX),
            value: VALUE,
            address_param_leaves: Arc::new(vec![BASE, INDEX]),
            synthetic_stack_origin: false,
            exact_scaled_index: false,
        }
    }

    fn run_case(
        proof: ScalarMemoryAccessProof,
        final_params: &[RTLReg],
        types: &[XType],
        pointer: bool,
    ) -> DecompileDB {
        run_case_with_return(proof, final_params, types, pointer, None)
    }

    fn run_case_with_return(
        proof: ScalarMemoryAccessProof,
        final_params: &[RTLReg],
        types: &[XType],
        pointer: bool,
        final_return: Option<XType>,
    ) -> DecompileDB {
        run_case_with_marker_and_return(proof, final_params, types, pointer, true, final_return)
    }

    fn run_case_with_marker_and_return(
        proof: ScalarMemoryAccessProof,
        final_params: &[RTLReg],
        types: &[XType],
        pointer: bool,
        signature_marker: bool,
        final_return: Option<XType>,
    ) -> DecompileDB {
        let mut db = DecompileDB::default();
        let statement = match proof.direction {
            ScalarMemoryDirection::Read => CminorStmt::Sassign(
                VALUE,
                CminorExpr::Eload(
                    proof.chunk,
                    Addressing::Aindexed2scaled(4, 8),
                    Arc::new(vec![BASE, INDEX]),
                ),
            ),
            ScalarMemoryDirection::Write => CminorStmt::Sstore(
                proof.chunk,
                Addressing::Aindexed2scaled(4, 8),
                Arc::new(vec![BASE, INDEX]),
                VALUE,
            ),
        };
        db.rel_push("cminor_stmt", (NODE, statement));
        db.rel_push("cminor_scalar_memory_access", (NODE, proof.clone()));
        if signature_marker && proof.extension != ScalarMemoryExtension::Plain {
            db.rel_push("signature_scalar_extension_access", (NODE, proof));
        }
        if let Some(xtype) = final_return {
            db.rel_push("emit_function_return", (FUNCTION, VALUE));
            db.rel_push("emit_function_return_type_xtype", (FUNCTION, xtype));
        }
        for parameter in final_params {
            db.rel_push("emit_function_param", (FUNCTION, *parameter));
        }
        for xtype in types {
            db.rel_push("emit_var_type_candidate", (VALUE, *xtype));
        }
        if pointer {
            db.rel_push("is_ptr", (VALUE,));
        }
        CshminorPass.run(&mut db);
        db
    }

    fn candidate_count(db: &DecompileDB) -> usize {
        db.rel_iter::<(Node, ScalarMemoryAccessProof)>("scalar_lvalue_candidate")
            .filter(|(node, _)| *node == NODE)
            .count()
    }

    fn plain_read_proof() -> ScalarMemoryAccessProof {
        proof(
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::Plain,
            4,
            4,
            MemoryChunk::MInt32,
        )
    }

    #[test]
    fn scalar_lvalue_revalidates_every_address_leaf_after_param_reconciliation() {
        for (params, expected) in [
            (&[BASE, INDEX][..], 1),
            (&[BASE, INDEX, VALUE][..], 1),
            (&[BASE][..], 0),
            (&[INDEX][..], 0),
            (&[][..], 0),
        ] {
            let db = run_case(plain_read_proof(), params, &[XType::Xint], false);
            assert_eq!(candidate_count(&db), expected, "params={params:?}");
        }
    }

    #[test]
    fn scalar_lvalue_rejects_late_nonintegral_pointer_bool_and_wider_load_types() {
        for (types, pointer) in [
            (&[XType::Xfloat][..], false),
            (&[XType::Xptr][..], false),
            (&[XType::Xint][..], true),
            (&[XType::Xbool][..], false),
            (&[XType::Xlong][..], false),
        ] {
            let db = run_case(plain_read_proof(), &[BASE, INDEX], types, pointer);
            assert_eq!(
                candidate_count(&db),
                0,
                "late types={types:?}, pointer={pointer}"
            );
        }
    }

    #[test]
    fn scalar_lvalue_store_rejects_late_float_source_without_numeric_cast() {
        let store = proof(
            ScalarMemoryDirection::Write,
            ScalarMemoryExtension::Plain,
            4,
            4,
            MemoryChunk::MInt32,
        );
        let db = run_case(store, &[BASE, INDEX], &[XType::Xfloat], false);
        assert_eq!(candidate_count(&db), 0);
    }

    #[test]
    fn scalar_lvalue_extension_replaces_only_chunk_type_with_exact_result_type() {
        for (proof, original, result) in [
            (
                proof(
                    ScalarMemoryDirection::Read,
                    ScalarMemoryExtension::ZeroExtend,
                    1,
                    4,
                    MemoryChunk::MInt8Unsigned,
                ),
                XType::Xint8unsigned,
                XType::Xintunsigned,
            ),
            (
                proof(
                    ScalarMemoryDirection::Read,
                    ScalarMemoryExtension::SignExtend,
                    1,
                    8,
                    MemoryChunk::MInt8Signed,
                ),
                XType::Xint8signed,
                XType::Xlong,
            ),
        ] {
            let db = run_case(proof, &[BASE, INDEX], &[original], false);
            assert_eq!(candidate_count(&db), 1);
            let types: BTreeSet<XType> = db
                .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
                .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
                .collect();
            assert_eq!(types, BTreeSet::from([result]));
        }
    }

    #[test]
    fn scalar_lvalue_signed_extension_retypes_selected_unsigned_transport() {
        let signed_movsx = proof(
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::SignExtend,
            1,
            8,
            MemoryChunk::MInt8Unsigned,
        );
        let db = run_case(signed_movsx, &[BASE, INDEX], &[XType::Xint8unsigned], false);
        assert_eq!(candidate_count(&db), 1);
        let types: BTreeSet<XType> = db
            .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
            .collect();
        assert_eq!(types, BTreeSet::from([XType::Xlong]));
    }

    #[test]
    fn scalar_lvalue_extension_retypes_sole_late_signedness_companion() {
        let zero_extended = proof(
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::ZeroExtend,
            1,
            4,
            MemoryChunk::MInt8Unsigned,
        );
        let db = run_case(zero_extended, &[BASE, INDEX], &[XType::Xint8signed], false);
        assert_eq!(candidate_count(&db), 1);
        let types: BTreeSet<XType> = db
            .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
            .collect();
        assert_eq!(types, BTreeSet::from([XType::Xintunsigned]));
    }

    #[test]
    fn scalar_lvalue_extension_requires_signature_marker() {
        let zero_extended = proof(
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::ZeroExtend,
            1,
            4,
            MemoryChunk::MInt8Unsigned,
        );
        let db = run_case_with_marker_and_return(
            zero_extended,
            &[BASE, INDEX],
            &[XType::Xint8signed],
            false,
            false,
            None,
        );
        assert_eq!(candidate_count(&db), 0);
        let types: BTreeSet<XType> = db
            .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
            .collect();
        assert_eq!(types, BTreeSet::from([XType::Xint8signed]));
    }

    #[test]
    fn scalar_lvalue_direct_return_respects_prototype_selected_final_type() {
        for (final_return, expected) in [
            (XType::Xintunsigned, 1),
            (XType::Xint, 0),
            (XType::Xint8unsigned, 0),
            (XType::Xlong, 0),
        ] {
            let zero_extended = proof(
                ScalarMemoryDirection::Read,
                ScalarMemoryExtension::ZeroExtend,
                1,
                4,
                MemoryChunk::MInt8Unsigned,
            );
            let db = run_case_with_return(
                zero_extended,
                &[BASE, INDEX],
                &[XType::Xintunsigned],
                false,
                Some(final_return),
            );
            assert_eq!(
                candidate_count(&db),
                expected,
                "final return {final_return:?}"
            );
        }
    }

    #[test]
    fn scalar_lvalue_extension_rejects_late_type_conflict_without_sanitizing_it() {
        for incompatible in [
            XType::Xfloat,
            XType::Xptr,
            XType::Xbool,
            XType::Xint16unsigned,
            XType::Xlong,
        ] {
            let extension = proof(
                ScalarMemoryDirection::Read,
                ScalarMemoryExtension::ZeroExtend,
                1,
                4,
                MemoryChunk::MInt8Unsigned,
            );
            let db = run_case(
                extension,
                &[BASE, INDEX],
                &[XType::Xint8unsigned, incompatible],
                false,
            );
            assert_eq!(candidate_count(&db), 0, "late type={incompatible:?}");
            let types: BTreeSet<XType> = db
                .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
                .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
                .collect();
            assert_eq!(
                types,
                BTreeSet::from([XType::Xint8unsigned, incompatible]),
                "rejected type evidence must not be sanitized"
            );
        }
    }

    #[test]
    fn scalar_lvalue_extension_retains_only_the_exact_opcode_result_type() {
        let extension = proof(
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::ZeroExtend,
            1,
            4,
            MemoryChunk::MInt8Unsigned,
        );
        let db = run_case(
            extension,
            &[BASE, INDEX],
            &[XType::Xint8unsigned, XType::Xintunsigned],
            false,
        );
        assert_eq!(candidate_count(&db), 1);
        let types: BTreeSet<XType> = db
            .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .filter_map(|(value, xtype)| (*value == VALUE).then_some(*xtype))
            .collect();
        assert_eq!(types, BTreeSet::from([XType::Xintunsigned]));
    }
}
