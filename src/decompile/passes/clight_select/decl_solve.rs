//! P5 decl solve: program-level decl decisions from the SELECTED statements, which per-function solves cannot make; v1 has no cross-decl coupling, so it is an exact weighted vote in plain Rust.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::decompile::passes::c_pass::types::{CType, TypeQualifiers};
use crate::decompile::passes::clight_select::query::CalleeSignature;
use crate::decompile::passes::clight_select::select::SelectedFunction;
use crate::x86::types::{
    Address, ClightBinaryOp, ClightExpr, ClightStmt, ClightType, Ident, RTLReg, XType,
};

#[derive(Default)]
pub struct DeclSolveOut {
    /// Fields with integer-arithmetic use evidence: joins TR-3's conflict veto so pointer patches never land on them.
    pub field_int_veto: HashSet<(String, String)>,
    /// Fields whose use evidence is integer-DOMINATED (int uses, no bare deref uses): if the emitted def says pointer, retype to long (the `%`-on-void* family). Applied by apply_struct_field_type_selection.
    pub field_force_long: HashSet<(String, String)>,
    /// Per-(struct,field) pointer retype decision: positive pointer evidence minus the conflict veto; the field authority relocated here from the emit pass, valued with the chosen candidate string.
    pub field_ptr_selection: HashMap<(String, String), String>,
    /// Data globals invoked as functions: name -> the fn-pointer decl type that makes every `name(args)` call site compile (signature from the callee signature table, which is derived from the call sites themselves).
    pub fnptr_globals: HashMap<String, CType>,
    /// Per-function registers selected float/pointer but used as a BARE operand of an integer-only operator with no float or deref use: the operator proves an integer, so force long.
    pub force_long_regs: HashMap<Address, HashSet<RTLReg>>,
    /// The pointerness-flip companion to force_long_regs: registers selected int_* but dereferenced or stored through, seeded void * BEFORE usage inference so decl_solve is the pointerness authority.
    pub force_ptr_regs: HashMap<Address, HashSet<RTLReg>>,
}

/// XType -> CType for fn-pointer signatures; mirrors helpers.rs convert_param_type_from_param's Typed arm at class level.
fn xtype_to_ctype(xt: &XType) -> CType {
    use crate::decompile::passes::c_pass::types::{FloatSize, IntSize, Signedness};
    match xt {
        XType::Xvoid => CType::Void,
        XType::Xfloat => CType::Float(FloatSize::Double),
        XType::Xsingle => CType::Float(FloatSize::Float),
        XType::Xbool => CType::Bool,
        XType::Xint8signed => CType::Int(IntSize::Char, Signedness::Signed),
        XType::Xint8unsigned => CType::Int(IntSize::Char, Signedness::Unsigned),
        XType::Xint16signed => CType::Int(IntSize::Short, Signedness::Signed),
        XType::Xint16unsigned => CType::Int(IntSize::Short, Signedness::Unsigned),
        XType::Xint | XType::Xany32 => CType::int(),
        XType::Xintunsigned => CType::uint(),
        XType::Xlongunsigned => CType::ulong(),
        XType::XstructPtr(sid) => CType::ptr(CType::Struct(format!("struct_{:x}", sid))),
        XType::Xcharptr => CType::ptr(CType::char_signed()),
        XType::Xcharptrptr => CType::ptr(CType::ptr(CType::char_signed())),
        XType::Xintptr => CType::ptr(CType::int()),
        XType::Xfloatptr => CType::ptr(CType::double()),
        XType::Xsingleptr => CType::ptr(CType::float()),
        XType::Xptr => CType::ptr(CType::Void),
        XType::Xfuncptr => CType::ptr(CType::func_unprototyped(CType::Void)),
        _ => CType::long(),
    }
}

fn function_pointer_global_type(
    name: &str,
    observed_arities: &BTreeSet<usize>,
    shared_fixed_arities: &HashMap<String, usize>,
    known_fnptr_signatures: &HashMap<String, (XType, Vec<XType>, bool)>,
) -> CType {
    let function_type = if let Some((return_type, param_types, is_varargs)) =
        known_fnptr_signatures.get(name)
    {
        if *is_varargs && param_types.is_empty() {
            // ISO C cannot spell a variadic prototype without at least one
            // named parameter.  Preserve the call ABI by leaving it
            // unprototyped instead of fabricating a fixed `(void)` function.
            CType::func_unprototyped(xtype_to_ctype(return_type))
        } else {
            CType::Function(
                Box::new(xtype_to_ctype(return_type)),
                param_types.iter().map(xtype_to_ctype).collect::<Vec<_>>(),
                *is_varargs,
                false,
            )
        }
    } else if let Some(&arity) = shared_fixed_arities
        .get(name)
        .filter(|&&arity| observed_arities.len() == 1 && observed_arities.contains(&arity))
    {
        CType::Function(
            Box::new(CType::long()),
            vec![CType::long(); arity],
            false,
            false,
        )
    } else {
        CType::func_unprototyped(CType::long())
    };
    CType::Pointer(Box::new(function_type), TypeQualifiers::none())
}

/// The annotation type of an expression (leaf accessor; mirrors ctyping::expr_annotation for the shapes a field base can take).
fn ann(e: &ClightExpr) -> Option<&ClightType> {
    use ClightExpr::*;
    match e {
        EconstInt(_, t) | EconstFloat(_, t) | EconstSingle(_, t) | EconstLong(_, t)
        | Evar(_, t) | EvarSymbol(_, t) | Etempvar(_, t) | Ederef(_, t) | Eaddrof(_, t)
        | Eunop(_, _, t) | Ebinop(_, _, _, t) | Ecast(_, t) | Efield(_, _, t)
        | Esizeof(_, t) | Ealignof(_, t) | Econdition(_, _, _, t) => Some(t),
    }
}

/// Struct sid when `e` is struct-pointer-typed (annotation or cast target).
fn struct_ptr_sid(e: &ClightExpr) -> Option<usize> {
    if let Some(ClightType::Tpointer(inner, _)) = ann(e) {
        if let ClightType::Tstruct(sid, _) = inner.as_ref() {
            return Some(*sid);
        }
    }
    None
}

/// The TR-3 field key of a field USE, either a materialized Efield or the raw deref form from_relations later prints as ->ofs_N; struct identity comes from the base's Tstruct annotation.
fn field_key(e: &ClightExpr) -> Option<(String, String)> {
    match e {
        ClightExpr::Efield(base, fid, _) => {
            let bt = match base.as_ref() {
                ClightExpr::Ederef(_, t) => t,
                ClightExpr::Etempvar(_, t) | ClightExpr::Evar(_, t) | ClightExpr::EvarSymbol(_, t) => t,
                _ => return None,
            };
            if let ClightType::Tstruct(sid, _) = bt {
                // DECL-1: the Efield id is a two's-complement-encoded usize, so a negative offset must be spelled by the one canonical decoder, keeping every field-key producer in one key space.
                return Some((
                    format!("struct_{:x}", sid),
                    crate::decompile::passes::csh_pass::field_ident_to_name(*fid),
                ));
            }
            None
        }
        ClightExpr::Ederef(inner, _) => match inner.as_ref() {
            ClightExpr::Ebinop(ClightBinaryOp::Oadd, l, r, _) => {
                let sid = struct_ptr_sid(l)?;
                let ofs = match r.as_ref() {
                    ClightExpr::EconstInt(n, _) => *n as i64,
                    ClightExpr::EconstLong(n, _) => *n,
                    _ => return None,
                };
                // DECL-1: negative struct-relative offsets are legal, so spell the field name with the canonical decoder and this raw-deref key space agrees with the materialized-Efield one for every sign.
                Some((
                    format!("struct_{:x}", sid),
                    crate::decompile::passes::csh_pass::field_ident_to_name(
                        ofs as crate::x86::types::Ident,
                    ),
                ))
            }
            base => {
                let sid = struct_ptr_sid(base)?;
                Some((format!("struct_{:x}", sid), "ofs_0".to_string()))
            }
        },
        _ => None,
    }
}

#[derive(Default, Clone)]
struct FieldVotes {
    int_ops: u32,
    bare_deref: u32,
}

struct Collector<'a> {
    fields: BTreeMap<(String, String), FieldVotes>,
    /// Called data-global name -> the set of argument counts observed at its call sites (the signature source: data globals have no recovered CalleeSignature).
    called_globals: BTreeMap<String, std::collections::BTreeSet<usize>>,
    /// Names of OBJECT-kind symbols (data globals) from the symbol table -- a call through one of these is the call-through-non-function family.
    global_names: &'a HashSet<String>,
    /// Clight global idents -> names (callees arrive as Evar(id) for ident-addressed globals, EvarSymbol for named externs).
    id_to_name: &'a HashMap<usize, String>,
    /// Current function's register candidate lists: a BARE field load into a register whose every candidate is integral is integer evidence for the FIELD, since cast-wrapped forms compile regardless.
    var_cands: &'a HashMap<crate::x86::types::RTLReg, Vec<String>>,
    /// Current function's registers used as a BARE operand of an integer-only operator: the use-based authority, right even when the solver mistyped the register as a pointer (the RC-3 family).
    reg_int_use: &'a HashSet<crate::x86::types::RTLReg>,
}

fn all_int_candidates(cands: &[String]) -> bool {
    !cands.is_empty() && cands.iter().all(|c| c.starts_with("int_"))
}

fn direct_global_object_name(
    expr: &ClightExpr,
    id_to_name: &HashMap<usize, String>,
) -> Option<String> {
    match expr {
        ClightExpr::EvarSymbol(name, _) => Some(name.clone()),
        ClightExpr::Evar(id, _) => id_to_name.get(id).cloned(),
        ClightExpr::Ecast(inner, _) => direct_global_object_name(inner, id_to_name),
        _ => None,
    }
}

/// Recover the data object supplying a call target.  In addition to the
/// historical direct Evar/EvarSymbol form, exact IAT lowering is deliberately
/// `cast(deref(cast(addrof(Evar(slot)))))`: the dereference is semantic and may
/// not be stripped, but the declaration authority still has to discover the
/// addressed slot so it can declare that object as a function pointer.
fn called_global_object_name(
    expr: &ClightExpr,
    id_to_name: &HashMap<usize, String>,
) -> Option<String> {
    match expr {
        ClightExpr::Ecast(inner, _) => called_global_object_name(inner, id_to_name),
        ClightExpr::EvarSymbol(_, _) | ClightExpr::Evar(_, _) => {
            direct_global_object_name(expr, id_to_name)
        }
        ClightExpr::Ederef(address, _) => {
            let mut address = address.as_ref();
            while let ClightExpr::Ecast(inner, _) = address {
                address = inner.as_ref();
            }
            let ClightExpr::Eaddrof(object, _) = address else {
                return None;
            };
            direct_global_object_name(object, id_to_name).map(|name| {
                crate::decompile::passes::c_pass::convert::from_relations::sanitize_c_symbol_name(
                    &name,
                )
            })
        }
        _ => None,
    }
}

/// True when `op` rejects pointer operands outright: the bitwise/shift/mod set, plus `*`/`/` when the result is not float (a Tfloat `*`/`/` is genuine float arithmetic). A BARE register operand here is integer-used.
fn is_int_only_binop(op: &ClightBinaryOp, ty: &ClightType) -> bool {
    matches!(
        op,
        ClightBinaryOp::Omod
            | ClightBinaryOp::Oand
            | ClightBinaryOp::Oor
            | ClightBinaryOp::Oxor
            | ClightBinaryOp::Oshl
            | ClightBinaryOp::Oshr
    ) || (matches!(op, ClightBinaryOp::Omul | ClightBinaryOp::Odiv)
        && !matches!(ty, ClightType::Tfloat(..)))
}

/// The register at the root of an operand, peeling casts (the printer drops these, so a cast-wrapped `reg` is still `reg` in an integer position). Mirrors `bare_reg` used by force_register_decls.
fn bare_reg_of(e: &ClightExpr) -> Option<crate::x86::types::RTLReg> {
    match e {
        ClightExpr::Etempvar(id, _) => Some(*id as crate::x86::types::RTLReg),
        ClightExpr::Ecast(inner, _) => bare_reg_of(inner),
        _ => None,
    }
}

/// Registers used as a BARE operand of an integer-only operator anywhere in `stmts`. A field flowing into such a register is integer-used through it.
fn collect_int_op_regs<'a>(
    stmts: impl Iterator<Item = &'a ClightStmt>,
) -> HashSet<crate::x86::types::RTLReg> {
    fn expr(e: &ClightExpr, out: &mut HashSet<crate::x86::types::RTLReg>) {
        use ClightExpr::*;
        match e {
            Ebinop(op, l, r, t) => {
                if is_int_only_binop(op, t) {
                    for side in [l.as_ref(), r.as_ref()] {
                        if let Some(reg) = bare_reg_of(side) {
                            out.insert(reg);
                        }
                    }
                }
                expr(l, out);
                expr(r, out);
            }
            Eunop(crate::x86::types::ClightUnaryOp::Onotint, inner, _) => {
                if let Some(reg) = bare_reg_of(inner) {
                    out.insert(reg);
                }
                expr(inner, out);
            }
            Eaddrof(inner, _) | Eunop(_, inner, _) | Ecast(inner, _) | Ederef(inner, _)
            | Efield(inner, _, _) => expr(inner, out),
            Econdition(c, a, b, _) => {
                expr(c, out);
                expr(a, out);
                expr(b, out);
            }
            _ => {}
        }
    }
    fn stmt(s: &ClightStmt, out: &mut HashSet<crate::x86::types::RTLReg>) {
        use ClightStmt::*;
        match s {
            Sassign(l, r) => {
                expr(l, out);
                expr(r, out);
            }
            Sset(_, e) => expr(e, out),
            Scall(_, f, args) => {
                expr(f, out);
                for a in args {
                    expr(a, out);
                }
            }
            Sbuiltin(_, _, _, args) => {
                for a in args {
                    expr(a, out);
                }
            }
            Sreturn(Some(e)) => expr(e, out),
            Sifthenelse(c, a, b) => {
                expr(c, out);
                stmt(a, out);
                stmt(b, out);
            }
            Ssequence(ss) => {
                for s in ss {
                    stmt(s, out);
                }
            }
            Sloop(a, b) => {
                stmt(a, out);
                stmt(b, out);
            }
            Slabel(_, inner) => stmt(inner, out),
            Sswitch(e, cases) => {
                expr(e, out);
                for (_, s) in cases {
                    stmt(s, out);
                }
            }
            _ => {}
        }
    }
    let mut out = HashSet::new();
    for s in stmts {
        stmt(s, &mut out);
    }
    out
}

impl<'a> Collector<'a> {
    fn expr(&mut self, e: &ClightExpr) {
        use ClightExpr::*;
        match e {
            Ebinop(op, l, r, _) => {
                // Integer-only operators reject pointers outright, so a BARE field operand is decl-level integer evidence; * and / are undefined on pointers too, making field * 8 the same hard evidence.
                if matches!(
                    op,
                    ClightBinaryOp::Omod
                        | ClightBinaryOp::Oand
                        | ClightBinaryOp::Oor
                        | ClightBinaryOp::Oxor
                        | ClightBinaryOp::Oshl
                        | ClightBinaryOp::Oshr
                        | ClightBinaryOp::Omul
                        | ClightBinaryOp::Odiv
                ) {
                    for side in [l.as_ref(), r.as_ref()] {
                        if let Some(key) = field_key(side) {
                            self.fields.entry(key).or_default().int_ops += 1;
                        }
                    }
                }
                self.expr(l);
                self.expr(r);
            }
            Ederef(inner, _) => {
                // A bare field as a deref base is pointer evidence that blocks the force-long override (dual-use fields stay untouched).
                if let Some(key) = field_key(inner) {
                    self.fields.entry(key).or_default().bare_deref += 1;
                }
                self.expr(inner);
            }
            Efield(base, _, _) => self.expr(base),
            Eaddrof(inner, _) | Eunop(_, inner, _) | Ecast(inner, _) => self.expr(inner),
            Econdition(c, a, b, _) => {
                self.expr(c);
                self.expr(a);
                self.expr(b);
            }
            _ => {}
        }
    }

    fn stmt(&mut self, s: &ClightStmt) {
        use ClightStmt::*;
        match s {
            Sassign(l, r) => {
                self.expr(l);
                self.expr(r);
            }
            Sset(id, e) => {
                if let Some(key) = field_key(e) {
                    let reg = *id as crate::x86::types::RTLReg;
                    // Two routes to integer evidence for a loaded field: every candidate integral, or a USE fact that the register feeds an integer-only operator, which is the authority for the RC-3 family.
                    let int_only = self
                        .var_cands
                        .get(&reg)
                        .map(|c| all_int_candidates(c))
                        .unwrap_or(false)
                        || self.reg_int_use.contains(&reg);
                    if int_only {
                        self.fields.entry(key).or_default().int_ops += 1;
                    }
                }
                self.expr(e)
            }
            Scall(_, f, args) => {
                // A call through a global SYMBOL that is a data object (not a recovered function) is the call-through-non-function family: the decl must become a function pointer.
                let callee_name = called_global_object_name(f, self.id_to_name);
                if let Some(name) = callee_name {
                    if std::env::var("MANIFOLD_DECL_SOLVE_TRACE").is_ok() {
                        eprintln!(
                            "[decl-solve trace] call via {:?} (object-symbol: {})",
                            name,
                            self.global_names.contains(name.as_str())
                        );
                    }
                    if self.global_names.contains(name.as_str()) {
                        self.called_globals
                            .entry(name)
                            .or_default()
                            .insert(args.len());
                    }
                }
                self.expr(f);
                for a in args {
                    self.expr(a);
                }
            }
            Sbuiltin(_, _, _, args) => {
                for a in args {
                    self.expr(a);
                }
            }
            Sreturn(Some(e)) => self.expr(e),
            Sifthenelse(c, a, b) => {
                self.expr(c);
                self.stmt(a);
                self.stmt(b);
            }
            Ssequence(ss) => {
                for s in ss {
                    self.stmt(s);
                }
            }
            Sloop(a, b) => {
                self.stmt(a);
                self.stmt(b);
            }
            Slabel(_, inner) => self.stmt(inner),
            Sswitch(e, cases) => {
                self.expr(e);
                for (_, s) in cases {
                    self.stmt(s);
                }
            }
            _ => {}
        }
    }
}

/// Per-(struct,field) pointer retype decision, the field authority relocated from the emit pass; the same evidence -> widen -> conflicts shape run over the EMITTED function set.
fn field_ptr_selection(
    funcs: &[SelectedFunction],
    internal_addrs: &HashSet<Address>,
    callee_sigs: &HashMap<Ident, CalleeSignature>,
    id_to_name: &HashMap<usize, String>,
    int_veto: &HashSet<(String, String)>,
) -> HashMap<(String, String), String> {
    use crate::decompile::passes::clight_select::query as csq;

    // Field evidence follows a symbol-form callee only when the spelling has
    // one Ident owner.  A deterministic first-wins choice is still the wrong
    // object under an address/kind or sanitizer collision.
    let mut name_owners: BTreeMap<String, BTreeSet<Ident>> = BTreeMap::new();
    for (id, name) in id_to_name {
        let ident = *id as Ident;
        let sanitized =
            crate::decompile::passes::c_pass::convert::from_relations::sanitize_c_symbol_name(name);
        name_owners.entry(sanitized).or_default().insert(ident);
        name_owners.entry(name.clone()).or_default().insert(ident);
    }
    for func in funcs {
        let sanitized =
            crate::decompile::passes::c_pass::convert::from_relations::sanitize_c_symbol_name(
                &func.name,
            );
        name_owners
            .entry(func.name.clone())
            .or_default()
            .insert(func.address as Ident);
        name_owners
            .entry(sanitized)
            .or_default()
            .insert(func.address as Ident);
    }
    let name_to_ident: HashMap<String, Ident> = name_owners
        .into_iter()
        .filter_map(|(name, owners)| {
            (owners.len() == 1).then(|| (name, *owners.iter().next().unwrap()))
        })
        .collect();

    let mut field_cands: HashMap<(String, String), Vec<String>> = HashMap::new();
    let mut evidence: HashSet<(String, String)> = HashSet::new();
    let mut conflicts: HashSet<(String, String)> = HashSet::new();

    // Emitted functions only, address-sorted for a deterministic first-wins candidate merge (candidates are identical per key, so order is moot for correctness; the discipline matches the retired scan).
    let mut sorted_funcs: Vec<&SelectedFunction> = funcs
        .iter()
        .filter(|f| internal_addrs.contains(&f.address))
        .collect();
    sorted_funcs.sort_by_key(|f| f.address);

    // Phase 1: per-function selected register classes (retained for phase 2), field candidates, and positive pointer evidence.
    let mut func_reg_classes: Vec<(HashSet<RTLReg>, HashSet<RTLReg>)> =
        Vec::with_capacity(sorted_funcs.len());
    for func in &sorted_funcs {
        let mut ptr_regs: HashSet<RTLReg> = HashSet::new();
        let mut float_regs: HashSet<RTLReg> = HashSet::new();
        for (reg, cands) in &func.var_type_candidates {
            let idx = func.var_decl_idx.get(reg).copied().unwrap_or(0);
            let sel = cands.get(idx).or_else(|| cands.first());
            if sel.map(|s| s.starts_with("ptr_")).unwrap_or(false) {
                ptr_regs.insert(*reg);
            }
            if sel.map(|s| s.starts_with("float_")).unwrap_or(false) {
                float_regs.insert(*reg);
            }
        }
        for (reg, ty) in &func.var_types {
            if !func.var_type_candidates.contains_key(reg) {
                if ty.starts_with("ptr_") {
                    ptr_regs.insert(*reg);
                } else if ty.starts_with("float_") {
                    float_regs.insert(*reg);
                }
            }
        }
        for (key, cands) in csq::collect_struct_field_type_candidates(func.statements.values()) {
            field_cands.entry(key).or_insert(cands);
        }
        for stmt in func.statements.values() {
            csq::scan_field_ptr_evidence(
                stmt,
                &ptr_regs,
                callee_sigs,
                &name_to_ident,
                &mut evidence,
            );
        }
        func_reg_classes.push((ptr_regs, float_regs));
    }

    // Fields the pick rule would patch: phase 2's veto must treat loads of these as pointer-valued, since the patch is what makes them pointers in the printed C.
    let patchable: HashSet<(String, String)> = evidence
        .iter()
        .filter(|k| {
            field_cands
                .get(*k)
                .map_or(false, |cands| csq::pick_field_ptr_candidate(cands).is_some())
        })
        .cloned()
        .collect();

    // Phase 2: widen the register classes to a fixpoint, then collect the pointer-retype conflicts (float context, integer/offset arithmetic).
    for (func, (ptr_regs, float_regs)) in sorted_funcs.iter().zip(func_reg_classes.iter_mut()) {
        csq::widen_float_regs_by_defs(func.statements.values(), float_regs);
        csq::widen_ptr_regs_by_defs(func.statements.values(), ptr_regs, &patchable);
        for stmt in func.statements.values() {
            csq::scan_field_ptr_conflicts(
                stmt,
                ptr_regs,
                float_regs,
                callee_sigs,
                &name_to_ident,
                &mut conflicts,
            );
        }
    }

    // Integer-use evidence joins the conflict veto -- a field sitting in `%`/`&`/shift arithmetic must never receive a pointer patch.
    conflicts.extend(int_veto.iter().cloned());

    let mut selection: HashMap<(String, String), String> = HashMap::new();
    let mut sorted_evidence: Vec<&(String, String)> = evidence.iter().collect();
    sorted_evidence.sort();
    for key in sorted_evidence {
        if conflicts.contains(key) {
            continue;
        }
        if let Some(cands) = field_cands.get(key) {
            if let Some(i) = csq::pick_field_ptr_candidate(cands) {
                selection.insert(key.clone(), cands[i].clone());
            }
        }
    }
    selection
}

/// The register at the root of an operand, peeling casts: forcing its decl to long makes the peeled cast redundant, and the float/deref vetoes still catch genuinely float or pointer registers.
fn bare_reg(e: &ClightExpr) -> Option<RTLReg> {
    match e {
        ClightExpr::Etempvar(id, _) => Some(*id as RTLReg),
        ClightExpr::Ecast(inner, _) => bare_reg(inner),
        _ => None,
    }
}

#[derive(Default, Clone)]
struct RegVotes {
    int_only: u32,
    float_use: u32,
    ptr_use: u32,
}

/// Per-function register decl authority returning (forced_long, forced_ptr): an int-only operator proves an integer, a deref proves a pointer, and each verdict requires unambiguous evidence.
fn force_register_decls(func: &SelectedFunction) -> (HashSet<RTLReg>, HashSet<RTLReg>) {
    // Eligible-long = selected float/pointer; eligible-ptr = selected int. An already-correct register needs no forcing, so each verdict only reconsiders the OTHER class (the two sets are disjoint).
    let class_of = |reg: &RTLReg| -> Option<&str> {
        if let Some(cands) = func.var_type_candidates.get(reg) {
            let idx = func.var_decl_idx.get(reg).copied().unwrap_or(0);
            return cands.get(idx).or_else(|| cands.first()).map(|s| s.as_str());
        }
        func.var_types.get(reg).map(|s| s.as_str())
    };
    let mut all_regs: HashSet<RTLReg> = func.var_type_candidates.keys().copied().collect();
    all_regs.extend(func.var_types.keys().copied());
    let mut eligible_long: HashSet<RTLReg> = HashSet::new();
    let mut eligible_ptr: HashSet<RTLReg> = HashSet::new();
    for reg in &all_regs {
        match class_of(reg) {
            Some(s) if s.starts_with("float_") || s.starts_with("ptr_") => {
                eligible_long.insert(*reg);
            }
            Some(s) if s.starts_with("int_") => {
                eligible_ptr.insert(*reg);
            }
            _ => {}
        }
    }
    if eligible_long.is_empty() && eligible_ptr.is_empty() {
        return (HashSet::new(), HashSet::new());
    }

    fn is_int_only_op(e: &ClightExpr) -> bool {
        // `*`/`/` are integer-only ONLY when the result is not float; a Tfloat `*`/`/` is genuine float arithmetic and must NOT be read as an int op (that would suppress legitimate float-source evidence).
        let int_mul_div = matches!(
            e,
            ClightExpr::Ebinop(ClightBinaryOp::Omul | ClightBinaryOp::Odiv, _, _, t)
                if !matches!(t, ClightType::Tfloat(..))
        );
        int_mul_div
            || matches!(
                e,
                ClightExpr::Ebinop(
                    ClightBinaryOp::Omod
                        | ClightBinaryOp::Oand
                        | ClightBinaryOp::Oor
                        | ClightBinaryOp::Oxor
                        | ClightBinaryOp::Oshl
                        | ClightBinaryOp::Oshr,
                    ..,
                ) | ClightExpr::Eunop(crate::x86::types::ClightUnaryOp::Onotint, ..)
            )
    }

    fn walk_expr(e: &ClightExpr, votes: &mut HashMap<RTLReg, RegVotes>) {
        use ClightExpr::*;
        match e {
            Ebinop(op, l, r, t) => {
                // * and / reject pointers, so a non-float multiply or divide is integer evidence too; a Tfloat one is genuine float arithmetic, so gate the int reading on a non-float result.
                let int_mul_div = matches!(op, ClightBinaryOp::Omul | ClightBinaryOp::Odiv)
                    && !matches!(t, ClightType::Tfloat(..));
                let int_only = int_mul_div
                    || matches!(
                        op,
                        ClightBinaryOp::Omod
                            | ClightBinaryOp::Oand
                            | ClightBinaryOp::Oor
                            | ClightBinaryOp::Oxor
                            | ClightBinaryOp::Oshl
                            | ClightBinaryOp::Oshr
                    );
                // Genuine float arithmetic -- NOT a bit op that merely inherited a Tfloat annotation from the mistyped operand.
                let float_arith = matches!(t, ClightType::Tfloat(..))
                    && matches!(
                        op,
                        ClightBinaryOp::Omul
                            | ClightBinaryOp::Odiv
                            | ClightBinaryOp::Oadd
                            | ClightBinaryOp::Osub
                    );
                for side in [l.as_ref(), r.as_ref()] {
                    if let Some(reg) = bare_reg(side) {
                        if int_only {
                            votes.entry(reg).or_default().int_only += 1;
                        }
                        if float_arith {
                            votes.entry(reg).or_default().float_use += 1;
                        }
                    }
                }
                walk_expr(l, votes);
                walk_expr(r, votes);
            }
            Eunop(crate::x86::types::ClightUnaryOp::Onotint, inner, _) => {
                if let Some(reg) = bare_reg(inner) {
                    votes.entry(reg).or_default().int_only += 1;
                }
                walk_expr(inner, votes);
            }
            Ederef(inner, _) => {
                // The dereferenced base is a genuine pointer.
                let base = match inner.as_ref() {
                    Ebinop(ClightBinaryOp::Oadd, l, _, _) => l.as_ref(),
                    other => other,
                };
                if let Some(reg) = bare_reg(base) {
                    votes.entry(reg).or_default().ptr_use += 1;
                }
                walk_expr(inner, votes);
            }
            Efield(base, _, _) => {
                let p = match base.as_ref() {
                    Ederef(inner, _) => inner.as_ref(),
                    other => other,
                };
                if let Some(reg) = bare_reg(p) {
                    votes.entry(reg).or_default().ptr_use += 1;
                }
                walk_expr(base, votes);
            }
            Ecast(inner, t) => {
                // A cast TO float proves the operand is floating, so it vetoes a wrong force-long without forcing float in the decl.
                if matches!(t, ClightType::Tfloat(..)) {
                    if let Some(reg) = bare_reg(inner) {
                        votes.entry(reg).or_default().float_use += 1;
                    }
                }
                walk_expr(inner, votes);
            }
            Eaddrof(inner, _) | Eunop(_, inner, _) => walk_expr(inner, votes),
            Econdition(c, a, b, _) => {
                walk_expr(c, votes);
                walk_expr(a, votes);
                walk_expr(b, votes);
            }
            _ => {}
        }
    }

    fn walk_stmt(s: &ClightStmt, votes: &mut HashMap<RTLReg, RegVotes>) {
        use ClightStmt::*;
        match s {
            Sassign(l, r) => {
                // `*R = ...` stores through a genuine pointer register.
                if let ClightExpr::Ederef(inner, _) = l {
                    if let Some(reg) = bare_reg(inner) {
                        votes.entry(reg).or_default().ptr_use += 1;
                    }
                }
                walk_expr(l, votes);
                walk_expr(r, votes);
            }
            Sset(id, e) => {
                // Assigned a genuine float SOURCE (literal or float arithmetic / float load), not a bit op that merely inherited Tfloat.
                let genuine_float = !is_int_only_op(e)
                    && (matches!(e, ClightExpr::EconstFloat(..) | ClightExpr::EconstSingle(..))
                        || matches!(ann(e), Some(ClightType::Tfloat(..))));
                if genuine_float {
                    votes.entry(*id as RTLReg).or_default().float_use += 1;
                }
                // NOTE: pointer-SOURCE evidence was tried here and REVERTED -- the emitter materializes annotation casts liberally, so counting them spuriously vetoed correct force-long verdicts (+11 gcc errors).
                walk_expr(e, votes);
            }
            Scall(_, f, args) => {
                walk_expr(f, votes);
                for a in args {
                    walk_expr(a, votes);
                }
            }
            Sbuiltin(_, _, _, args) => {
                for a in args {
                    walk_expr(a, votes);
                }
            }
            Sreturn(Some(e)) => walk_expr(e, votes),
            Sifthenelse(c, a, b) => {
                walk_expr(c, votes);
                walk_stmt(a, votes);
                walk_stmt(b, votes);
            }
            Ssequence(ss) => {
                for s in ss {
                    walk_stmt(s, votes);
                }
            }
            Sloop(a, b) => {
                walk_stmt(a, votes);
                walk_stmt(b, votes);
            }
            Slabel(_, inner) => walk_stmt(inner, votes),
            Sswitch(e, cases) => {
                walk_expr(e, votes);
                for (_, s) in cases {
                    walk_stmt(s, votes);
                }
            }
            _ => {}
        }
    }

    let mut votes: HashMap<RTLReg, RegVotes> = HashMap::new();
    let mut nodes: Vec<_> = func.statements.keys().copied().collect();
    nodes.sort_unstable();
    for n in nodes {
        walk_stmt(&func.statements[&n], &mut votes);
    }

    let forced_long: HashSet<RTLReg> = eligible_long
        .iter()
        .filter(|r| {
            votes
                .get(r)
                .map_or(false, |v| v.int_only > 0 && v.float_use == 0 && v.ptr_use == 0)
        })
        .copied()
        .collect();
    let forced_ptr: HashSet<RTLReg> = eligible_ptr
        .iter()
        .filter(|r| {
            votes
                .get(r)
                .map_or(false, |v| v.ptr_use > 0 && v.int_only == 0 && v.float_use == 0)
        })
        .copied()
        .collect();

    if std::env::var("MANIFOLD_FORCE_LONG_TRACE").is_ok() {
        let cands: Vec<_> = eligible_long
            .iter()
            .chain(eligible_ptr.iter())
            .filter_map(|r| votes.get(r).map(|v| (*r, v.int_only, v.float_use, v.ptr_use)))
            .filter(|(_, i, _, p)| *i > 0 || *p > 0)
            .collect();
        if !cands.is_empty() {
            eprintln!(
                "[force-reg] fn {:x}: elig(long={},ptr={}) forced(long={},ptr={}) cands(reg,int,float,ptr)={:?}",
                func.address,
                eligible_long.len(),
                eligible_ptr.len(),
                forced_long.len(),
                forced_ptr.len(),
                cands,
            );
        }
    }

    (forced_long, forced_ptr)
}

pub fn run(
    funcs: &[SelectedFunction],
    internal_addrs: &HashSet<Address>,
    callee_sigs: &HashMap<Ident, CalleeSignature>,
    global_names: &HashSet<String>,
    id_to_name: &HashMap<usize, String>,
    shared_fixed_arities: &HashMap<String, usize>,
    known_fnptr_signatures: &HashMap<String, (XType, Vec<XType>, bool)>,
) -> DeclSolveOut {
    let empty: HashMap<crate::x86::types::RTLReg, Vec<String>> = HashMap::new();
    let empty_regs: HashSet<crate::x86::types::RTLReg> = HashSet::new();
    let mut fields: BTreeMap<(String, String), FieldVotes> = BTreeMap::new();
    let mut called: BTreeMap<String, std::collections::BTreeSet<usize>> = BTreeMap::new();
    // Address-sorted walk for deterministic counts (counts are order-free, but keep the discipline).
    let mut sorted: Vec<&SelectedFunction> = funcs.iter().collect();
    sorted.sort_by_key(|f| f.address);
    for f in &sorted {
        // Pre-pass: integer-only-op-used registers (def can follow use, so gather them first).
        let reg_int_use = collect_int_op_regs(f.statements.values());
        let mut c = Collector {
            fields: std::mem::take(&mut fields),
            called_globals: std::mem::take(&mut called),
            global_names,
            id_to_name,
            var_cands: &f.var_type_candidates,
            reg_int_use: &reg_int_use,
        };
        let _ = &empty;
        let mut nodes: Vec<_> = f.statements.keys().copied().collect();
        nodes.sort_unstable();
        for n in nodes {
            c.stmt(&f.statements[&n]);
        }
        fields = c.fields;
        called = c.called_globals;
    }
    let c = Collector {
        fields,
        called_globals: called,
        global_names,
        id_to_name,
        var_cands: &empty,
        reg_int_use: &empty_regs,
    };

    if std::env::var("MANIFOLD_DECL_SOLVE_TRACE").is_ok() {
        eprintln!(
            "[decl-solve trace] fields seen: {:?}",
            c.fields.keys().take(5).collect::<Vec<_>>()
        );
        eprintln!(
            "[decl-solve trace] called globals: {:?}; global_names sample: {:?}",
            c.called_globals.keys().take(5).collect::<Vec<_>>(),
            global_names.iter().take(5).collect::<Vec<_>>()
        );
    }
    let mut out = DeclSolveOut::default();
    for (key, v) in &c.fields {
        if v.int_ops > 0 {
            out.field_int_veto.insert(key.clone());
            // The vote: integer uses with NO bare-deref pointer evidence means the decl must be integral; with both, the field is dual-use and `long` would shift errors rather than fix them -- leave it.
            if v.bare_deref == 0 {
                out.field_force_long.insert(key.clone());
            }
        }
    }

    // CTYPING_PLAN 5.3: the per-(struct,field) pointer retype decision, the field authority relocated from the emit-pass TR-3 scan. Runs over the emitted function set with the int-arith veto just computed joined into its conflicts (the emit pass joined it the same way).
    out.field_ptr_selection = field_ptr_selection(
        funcs,
        internal_addrs,
        callee_sigs,
        id_to_name,
        &out.field_int_veto,
    );
    for (name, arg_counts) in &c.called_globals {
        // An authoritative known prototype wins.  Otherwise a data-global
        // callee receives a fixed type only from the same anchored coherent
        // cross-site proof used for ordinary externs.  Single-site,
        // conflicting, variadic, and forwarded-only observations stay truly
        // unprototyped so every site's real argument vector is preserved.
        out.fnptr_globals.insert(
            name.clone(),
            function_pointer_global_type(
                name,
                arg_counts,
                shared_fixed_arities,
                known_fnptr_signatures,
            ),
        );
    }

    // Register force-long (phase 5 register decl authority): per emitted function, registers whose usage proves they are integers despite a float/pointer selection.
    for f in &sorted {
        if !internal_addrs.contains(&f.address) {
            continue;
        }
        let (forced_long, forced_ptr) = force_register_decls(f);
        if !forced_long.is_empty() {
            out.force_long_regs.insert(f.address, forced_long);
        }
        if !forced_ptr.is_empty() {
            out.force_ptr_regs.insert(f.address, forced_ptr);
        }
    }

    out
}

#[cfg(test)]
mod function_pointer_declaration_tests {
    use super::*;
    use crate::decompile::passes::csh_pass::{
        clight_function_pointer_type, default_int_type, default_long_type, pointer_to,
    };
    use crate::x86::types::{Node, Signature};
    use std::sync::Arc;

    #[test]
    fn emitted_function_pointer_uses_authoritative_or_proven_fixed_signature() {
        let known = HashMap::from([(
            "known".to_string(),
            (XType::Xptr, vec![XType::Xcharptr, XType::Xlong], true),
        ), (
            "takes_cb".to_string(),
            (XType::Xvoid, vec![XType::Xfuncptr], false),
        ), (
            "invalid_zero_va".to_string(),
            (XType::Xint, Vec::new(), true),
        )]);
        let shared = HashMap::from([("shared".to_string(), 2usize)]);
        let observed_two = BTreeSet::from([2usize]);

        assert_eq!(
            function_pointer_global_type("known", &observed_two, &shared, &known),
            CType::ptr(CType::Function(
                Box::new(CType::ptr(CType::Void)),
                vec![CType::ptr(CType::char_signed()), CType::long()],
                true,
                false,
            ))
        );
        assert_eq!(
            function_pointer_global_type("shared", &observed_two, &shared, &known),
            CType::ptr(CType::Function(
                Box::new(CType::long()),
                vec![CType::long(), CType::long()],
                false,
                false,
            ))
        );
        assert_eq!(
            function_pointer_global_type("takes_cb", &observed_two, &shared, &known),
            CType::ptr(CType::Function(
                Box::new(CType::Void),
                vec![CType::ptr(CType::Function(
                Box::new(CType::Void),
                    Vec::new(),
                    false,
                    true,
                ))],
                false,
                false,
            ))
        );
        assert_eq!(
            function_pointer_global_type(
                "ambiguous",
                &BTreeSet::new(),
                &HashMap::new(),
                &HashMap::new(),
            ),
            CType::ptr(CType::func_unprototyped(CType::long()))
        );
        assert_eq!(
            function_pointer_global_type(
                "invalid_zero_va",
                &BTreeSet::new(),
                &shared,
                &known,
            ),
            CType::ptr(CType::func_unprototyped(CType::int()))
        );

        // A shared zero-arity guess must not retype a selected four-argument
        // IAT call.  The per-site vector wins and the object remains K&R.
        assert_eq!(
            function_pointer_global_type(
                "__imp_opaque_fixed_arity_target",
                &BTreeSet::from([4usize]),
                &HashMap::from([("__imp_opaque_fixed_arity_target".to_string(), 0usize)]),
                &HashMap::new(),
            ),
            CType::ptr(CType::func_unprototyped(CType::long()))
        );
    }

    #[test]
    fn casted_dereferenced_iat_object_is_typed_as_function_pointer_global() {
        let function: Address = 0x7000;
        let node: Node = 0x7010;
        let iat: Ident = 0x9000;
        let name = "__imp_imported_callback".to_string();
        let signature = Signature {
            sig_args: Arc::new(vec![XType::Xany64]),
            sig_res: XType::Xvoid,
            ..Signature::default()
        };
        let slot_address = ClightExpr::Eaddrof(
            Box::new(ClightExpr::Evar(iat, default_int_type())),
            pointer_to(default_int_type()),
        );
        let loaded_slot = ClightExpr::Ederef(
            Box::new(ClightExpr::Ecast(
                Box::new(slot_address),
                pointer_to(default_long_type()),
            )),
            default_long_type(),
        );
        let callee = ClightExpr::Ecast(
            Box::new(loaded_slot),
            clight_function_pointer_type(&signature),
        );
        let selected = SelectedFunction {
            address: function,
            name: "iat_thunk".to_string(),
            entry_node: node,
            return_type: default_int_type(),
            param_regs: Vec::new(),
            param_types: Vec::new(),
            stack_size: 0,
            statements: HashMap::from([(
                node,
                ClightStmt::Scall(
                    None,
                    callee,
                    vec![ClightExpr::EconstLong(7, default_long_type())],
                ),
            )]),
            successors: HashMap::new(),
            used_regs: HashSet::new(),
            struct_fields: HashMap::new(),
            sseq_groups: HashMap::new(),
            var_types: HashMap::new(),
            var_type_candidates: HashMap::new(),
            var_decl_idx: HashMap::new(),
            loop_headers: HashSet::new(),
            switch_heads: HashSet::new(),
            reg_struct_ids: HashMap::new(),
            loop_info: HashMap::new(),
        };
        let known = HashMap::from([(
            name.clone(),
            (XType::Xvoid, vec![XType::Xany64], false),
        )]);

        let out = run(
            &[selected],
            &HashSet::from([function]),
            &HashMap::new(),
            &HashSet::from([name.clone()]),
            &HashMap::from([(iat, name.clone())]),
            &HashMap::new(),
            &known,
        );

        assert_eq!(
            out.fnptr_globals.get(&name),
            Some(&CType::ptr(CType::Function(
                Box::new(CType::Void),
                vec![CType::long()],
                false,
                false,
            )))
        );
    }
}
