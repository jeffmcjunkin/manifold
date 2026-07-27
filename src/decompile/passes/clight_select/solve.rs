use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use rayon::prelude::*;

use z3::ast::{Bool, Dynamic, Int};
use z3::{DatatypeAccessor, DatatypeBuilder, DatatypeSort, Optimize, SatResult, Sort};

use crate::x86::types::{Address, ClightBinaryOp, ClightExpr, ClightIntSize, ClightStmt, ClightType, Ident, Node, RTLReg, XType};
use crate::decompile::passes::clight_select::query::FunctionData;
use crate::decompile::passes::clight_select::select::{
    callee_ident_from_expr, ProgramSelectionState,
};

// Variant order in the Ty datatype; indices into DatatypeSort::variants.
const V_VOID: usize = 0;
const V_INT: usize = 1;
const V_FLOAT: usize = 2;
const V_STRUCT: usize = 3;
const V_UNION: usize = 4;
const V_FUNC: usize = 5;
const V_PTR: usize = 6;

// Per-Context handles to the recursive Ty datatype; the algebra models type CLASSES only, since width/signedness payloads carried no constraint information and exploded maxres.
struct TypeDt {
    dt: DatatypeSort,
}

impl TypeDt {
    fn build() -> Self {
        let dt = DatatypeBuilder::new("Ty")
            .variant("void", vec![])
            .variant("int", vec![])
            .variant("float", vec![])
            .variant("struct", vec![("sid", DatatypeAccessor::Sort(Sort::int()))])
            .variant("union", vec![("uid", DatatypeAccessor::Sort(Sort::int()))])
            .variant("func", vec![])
            .variant("ptr", vec![("pointee", DatatypeAccessor::datatype("Ty"))])
            .finish();
        TypeDt { dt }
    }

    fn void(&self) -> Dynamic { self.dt.variants[V_VOID].constructor.apply(&[]) }
    fn int(&self) -> Dynamic { self.dt.variants[V_INT].constructor.apply(&[]) }
    fn float(&self) -> Dynamic { self.dt.variants[V_FLOAT].constructor.apply(&[]) }
    fn struct_(&self, sid: i64) -> Dynamic {
        self.dt.variants[V_STRUCT].constructor.apply(&[&Int::from_i64(sid)])
    }
    fn union_(&self, uid: i64) -> Dynamic {
        self.dt.variants[V_UNION].constructor.apply(&[&Int::from_i64(uid)])
    }
    fn func(&self) -> Dynamic { self.dt.variants[V_FUNC].constructor.apply(&[]) }
    fn ptr(&self, pointee: &Dynamic) -> Dynamic {
        self.dt.variants[V_PTR].constructor.apply(&[pointee])
    }

    fn is_void(&self, x: &Dynamic) -> Bool { self.dt.variants[V_VOID].tester.apply(&[x]).as_bool().unwrap() }
    fn is_int(&self, x: &Dynamic) -> Bool { self.dt.variants[V_INT].tester.apply(&[x]).as_bool().unwrap() }
    fn is_float(&self, x: &Dynamic) -> Bool { self.dt.variants[V_FLOAT].tester.apply(&[x]).as_bool().unwrap() }
    fn is_struct(&self, x: &Dynamic) -> Bool { self.dt.variants[V_STRUCT].tester.apply(&[x]).as_bool().unwrap() }
    fn is_union(&self, x: &Dynamic) -> Bool { self.dt.variants[V_UNION].tester.apply(&[x]).as_bool().unwrap() }
    fn is_func(&self, x: &Dynamic) -> Bool { self.dt.variants[V_FUNC].tester.apply(&[x]).as_bool().unwrap() }
    fn is_ptr(&self, x: &Dynamic) -> Bool { self.dt.variants[V_PTR].tester.apply(&[x]).as_bool().unwrap() }
    fn pointee(&self, x: &Dynamic) -> Dynamic { self.dt.variants[V_PTR].accessors[0].apply(&[x]) }
    fn struct_sid(&self, x: &Dynamic) -> Dynamic { self.dt.variants[V_STRUCT].accessors[0].apply(&[x]) }

    // The post-typeconv integer universe (one class on this algebra).
    fn is_intish(&self, x: &Dynamic) -> Bool {
        self.is_int(x)
    }
    // Arithmetic class: int/float -- what classify_binarith accepts.
    fn is_arith(&self, x: &Dynamic) -> Bool {
        Bool::or(&[self.is_int(x), self.is_float(x)])
    }
    // Pointer-shaped after decay: a pointer or a function (functions decay).
    fn is_ptrish(&self, x: &Dynamic) -> Bool {
        Bool::or(&[self.is_ptr(x), self.is_func(x)])
    }
    // classify_bool: int/float/ptr accept, struct/union/void reject.
    fn is_boolish(&self, x: &Dynamic) -> Bool {
        Bool::or(&[self.is_arith(x), self.is_ptrish(x)])
    }

    // A fresh free type variable.
    fn fresh(&self) -> Dynamic {
        Dynamic::from_ast(&z3::ast::Datatype::fresh_const("ty", &self.dt.sort))
    }
}

// XType (signature types) -> a concrete Ty term.
fn xtype_to_ty(t: &TypeDt, xt: &XType) -> Dynamic {
    match xt {
        XType::Xvoid => t.void(),
        XType::Xfloat | XType::Xsingle => t.float(),
        XType::XstructPtr(sid) => t.ptr(&t.struct_(*sid as i64)),
        XType::Xfloatptr | XType::Xsingleptr => t.ptr(&t.float()),
        XType::Xfuncptr => t.ptr(&t.func()),
        XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr => t.ptr(&t.int()),
        _ => t.int(),
    }
}

// ClightType (expression annotations) -> a concrete Ty term at OPERATOR positions: typeconv applies (the whole integer family collapses to the int class, arrays decay, function decay is handled via classify_fun's ptrish acceptance).
fn clight_to_ty(t: &TypeDt, ct: &ClightType) -> Dynamic {
    match ct {
        ClightType::Tvoid => t.void(),
        ClightType::Tint(..) | ClightType::Tlong(..) | ClightType::Tint128(..) => t.int(),
        ClightType::Tfloat(..) => t.float(),
        ClightType::Tstruct(sid, _) => t.struct_(*sid as i64),
        ClightType::Tunion(uid, _) => t.union_(*uid as i64),
        ClightType::Tfunction(..) => t.func(),
        ClightType::Tpointer(inner, _) => {
            let pointee = clight_to_ty(t, inner);
            t.ptr(&pointee)
        }
        ClightType::Tarray(inner, _, _) => {
            // typeconv: arrays decay to element pointers.
            let pointee = clight_to_ty(t, inner);
            t.ptr(&pointee)
        }
    }
}

// Candidate type-string -> a concrete Ty term (the provenance prior); classes only -- width/signedness/size live in the candidate strings and the anchor.
fn cand_str_to_ty(t: &TypeDt, s: &str) -> Dynamic {
    if let Some(rest) = s.strip_prefix("ptr_struct_") {
        if let Ok(sid) = i64::from_str_radix(rest, 16) {
            return t.ptr(&t.struct_(sid));
        }
        return t.ptr(&t.int());
    }
    match s {
        "ptr_double" | "ptr_float" => t.ptr(&t.float()),
        "ptr_void" => t.ptr(&t.void()),
        "ptr_func" => t.ptr(&t.func()),
        _ if s.starts_with("ptr_") => t.ptr(&t.int()),
        _ if s.starts_with("float_") => t.float(),
        _ => t.int(),
    }
}

// wt_cast at the class level: float<->ptr, cross-kind composites, a void source and a function target are errors, while int<->ptr/width/float conversions are silent; composite IDs are deliberately not compared.
fn wt_cast_term(t: &TypeDt, from: &Dynamic, to: &Dynamic) -> Bool {
    let bad_f2p = Bool::and(&[t.is_float(from), t.is_ptrish(to)]);
    let bad_p2f = Bool::and(&[t.is_ptrish(from), t.is_float(to)]);
    let comp_ok = Bool::or(&[
        Bool::and(&[t.is_struct(from), t.is_struct(to)]),
        Bool::and(&[t.is_union(from), t.is_union(to)]),
    ]);
    let comp_involved = Bool::or(&[
        t.is_struct(from),
        t.is_union(from),
        t.is_struct(to),
        t.is_union(to),
    ]);
    let comp_bad = Bool::and(&[comp_involved, comp_ok.not()]);
    let bad_void_src = t.is_void(from);
    let bad_to_func = t.is_func(to);
    let illegal = Bool::or(&[bad_f2p, bad_p2f, comp_bad, bad_void_src, bad_to_func]);
    // cast_case_void: ANY source converts to void.
    Bool::or(&[t.is_void(to), illegal.not()])
}

// wt_cast specialized for a pruned register variable TO-side, dropping the func/void/union arms to 8 testers instead of 14 on the hottest obligation family.
fn wt_cast_to_reg(t: &TypeDt, from: &Dynamic, lv: &Dynamic) -> Bool {
    let bad_f2p = Bool::and(&[t.is_float(from), t.is_ptr(lv)]);
    let bad_p2f = Bool::and(&[t.is_ptrish(from), t.is_float(lv)]);
    // lv can never be a union (domain pruning), so a union SOURCE is illegal for every register target; struct sources pair only with struct targets.
    let bad_comp = Bool::or(&[
        Bool::and(&[t.is_struct(from), t.is_struct(lv).not()]),
        Bool::and(&[t.is_struct(from).not(), t.is_struct(lv)]),
        t.is_union(from),
    ]);
    let bad_void_src = t.is_void(from);
    Bool::or(&[bad_f2p, bad_p2f, bad_comp, bad_void_src]).not()
}

// "a and b agree on POINTERNESS", the axis the Sset-level SEL-AGREE preference operates on, demoting a crossing whose printed cast gcc rejects.
fn ptr_class_eq(t: &TypeDt, a: &Dynamic, b: &Dynamic) -> Bool {
    t.is_ptr(a)._eq(&t.is_ptr(b))
}

// type_binop legality at the class level, one clause per node; parity with ctyping::type_binop is test-enforced.
fn binop_legal_term(t: &TypeDt, op: ClightBinaryOp, ltm: &Dynamic, rtm: &Dynamic) -> Bool {
    use ClightBinaryOp::*;
    match op {
        Oadd => Bool::or(&[
            Bool::and(&[t.is_ptrish(ltm), t.is_intish(rtm)]),
            Bool::and(&[t.is_intish(ltm), t.is_ptrish(rtm)]),
            Bool::and(&[t.is_arith(ltm), t.is_arith(rtm)]),
        ]),
        Osub => Bool::or(&[
            Bool::and(&[t.is_ptrish(ltm), t.is_intish(rtm)]),
            Bool::and(&[t.is_ptrish(ltm), t.is_ptrish(rtm)]),
            Bool::and(&[t.is_arith(ltm), t.is_arith(rtm)]),
        ]),
        Omul | Odiv => Bool::and(&[t.is_arith(ltm), t.is_arith(rtm)]),
        Omod | Oand | Oor | Oxor | Oshl | Oshr => {
            Bool::and(&[t.is_intish(ltm), t.is_intish(rtm)])
        }
        Oeq | One | Olt | Ogt | Ole | Oge => Bool::or(&[
            Bool::and(&[t.is_ptrish(ltm), t.is_ptrish(rtm)]),
            Bool::and(&[t.is_ptrish(ltm), t.is_intish(rtm)]),
            Bool::and(&[t.is_intish(ltm), t.is_ptrish(rtm)]),
            Bool::and(&[t.is_arith(ltm), t.is_arith(rtm)]),
        ]),
    }
}

// type_unop legality (Ctyping.v:52): ! accepts any scalar; ~ integers only; - and __builtin_fabs numerics only.
fn unop_legal_term(t: &TypeDt, op: crate::x86::types::ClightUnaryOp, itm: &Dynamic) -> Bool {
    use crate::x86::types::ClightUnaryOp::*;
    match op {
        Onotbool => t.is_boolish(itm),
        Onotint => t.is_intish(itm),
        Oneg | Oabsfloat => t.is_arith(itm),
    }
}

/// One recorded condition for the post-solve per-node re-pick: the raw (unguarded) condition (None = an unconditional penalty for this candidate) and its weight; weight u64::MAX marks a structural requirement (validity, not preference).
type RecordedCond = (Option<Bool>, u64);

// Z3 is the sole selector: structural requirements are always HARD; there is no relaxed retry and no non-Z3 fallback (a function whose constraints are UNSAT yields no selection and signals an upstream candidate-generation bug).
struct ConstraintSink<'a> {
    opt: &'a Optimize,
    // The per-function dominant soft weight (compat_w); carried here so the operator-typing rules in `type_of` can emit dominant-soft requirements without threading the weight through every recursive call.
    compat_w: u64,
    // Per-candidate dedup for the SEL-AGREE Etempvar class-match preference: a register read N times with the same annotation class in one candidate statement needs ONE preference, not N. Cleared by begin_candidate().
    agree_seen: RefCell<HashSet<(RTLReg, u8)>>,
    // Re-pick recording for multi-candidate nodes: every emitted condition is logged RAW and re-scored in Rust under the fixed model, since node selections are conditionally independent given it.
    record: RefCell<Option<Vec<RecordedCond>>>,
    // Per-function (guard, cond, weight) dedup for the OPTIMIZER side only: identical obligations carry the same MaxSMT meaning, and clause COUNT is the cost driver.
    asserted: RefCell<HashSet<(Option<Bool>, Bool, u64)>>,
}

impl<'a> ConstraintSink<'a> {
    fn new(opt: &'a Optimize, compat_w: u64) -> Self {
        ConstraintSink {
            opt,
            compat_w,
            agree_seen: RefCell::new(HashSet::new()),
            record: RefCell::new(None),
            asserted: RefCell::new(HashSet::new()),
        }
    }

    // True exactly once per distinct (guard, cond, w) within this function.
    fn first_assert(&self, guard: Option<&Bool>, cond: &Bool, w: u64) -> bool {
        self.asserted
            .borrow_mut()
            .insert((guard.cloned(), cond.clone(), w))
    }

    // Open a fresh per-candidate scope: resets the class-match preference dedup and (for multi-candidate nodes) starts recording conditions for the post-solve re-pick.
    fn begin_candidate(&self, recording: bool) {
        self.agree_seen.borrow_mut().clear();
        *self.record.borrow_mut() = if recording { Some(Vec::new()) } else { None };
    }

    // Close the per-candidate scope, returning the recorded conditions.
    fn take_recorded(&self) -> Vec<RecordedCond> {
        self.record.borrow_mut().take().unwrap_or_default()
    }

    fn log(&self, cond: Option<&Bool>, w: u64) {
        if let Some(buf) = self.record.borrow_mut().as_mut() {
            buf.push((cond.cloned(), w));
        }
    }

    // A selection preference: never asserted into the optimizer, only recorded for the post-solve re-pick (no-op outside a recording scope -- with a single candidate there is nothing to prefer).
    fn prefer(&self, cond: &Bool, w: u64) {
        self.log(Some(cond), w);
    }

    // An unconditional penalty against the current candidate (e.g. capturing a void callee's result); recorded for the re-pick alongside its z3 assertion.
    fn penalize_candidate(&self, w: u64) {
        self.log(None, w);
    }

    // A structural requirement: always hard.
    fn assert_structural(&self, guard: Option<&Bool>, cond: &Bool) {
        self.log(Some(cond), u64::MAX);
        if !self.first_assert(guard, cond, u64::MAX) {
            return;
        }
        let c = match guard {
            Some(g) => g.implies(cond.clone()),
            None => cond.clone(),
        };
        self.opt.assert(&c);
    }

    fn assert_soft_tagged(&self, guard: Option<&Bool>, cond: &Bool, w: u64, tag: &'static str) {
        self.log(Some(cond), w);
        if !self.first_assert(guard, cond, w) {
            return;
        }
        SOFT_CLAUSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        soft_tag_count(tag);
        match guard {
            Some(g) => self.opt.assert_soft(&g.implies(cond.clone()), w, None),
            None => self.opt.assert_soft(cond, w, None),
        }
    }

    // Soft assertion on a raw Bool (the candidate-penalty `!guard` forms and the priors).
    fn assert_soft_raw(&self, cond: &Bool, w: u64) {
        SOFT_CLAUSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.opt.assert_soft(cond, w, None);
    }
}

// Run-wide asserted-soft-clause counter (P4 perf early-warning, CTYPING_PLAN D2: MaxSMT cost scales with conflicting soft-clause count -- the 15x lesson). Swap-reset and reported by infer_select_program.
static SOFT_CLAUSES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
// Construction-vs-solve attribution for the P4 perf gate.
static BUILD_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CHECK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Per-family soft-clause breakdown behind MANIFOLD_SOFT_BREAKDOWN (perf triage).
static SOFT_TAGS: std::sync::LazyLock<std::sync::Mutex<BTreeMap<&'static str, usize>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(BTreeMap::new()));

fn soft_tag_count(tag: &'static str) {
    if std::env::var("MANIFOLD_SOFT_BREAKDOWN").is_ok() {
        *SOFT_TAGS.lock().unwrap().entry(tag).or_default() += 1;
    }
}

// True if a cast operand's type comes from an inferable register (a free Ty var the solver can drive) rather than a concrete annotation/constant; a hard structural constraint only helps on a free var, so concrete operands stay soft (else a single-candidate node can go UNSAT).
fn cast_operand_is_inferable(e: &ClightExpr, tyvar: &HashMap<RTLReg, Dynamic>) -> bool {
    match e {
        ClightExpr::Etempvar(id, _) => tyvar.contains_key(&(*id as RTLReg)),
        ClightExpr::Ecast(inner, _) => cast_operand_is_inferable(inner, tyvar),
        _ => false,
    }
}

// True when e's inferred type term carries a free register variable, so a structural constraint on it is satisfiable; fully concrete forms are skipped and left to the emitter.
fn base_drives_free_var(e: &ClightExpr, tyvar: &HashMap<RTLReg, Dynamic>) -> bool {
    match e {
        ClightExpr::Etempvar(id, _) => tyvar.contains_key(&(*id as RTLReg)),
        ClightExpr::Ederef(inner, _) => base_drives_free_var(inner, tyvar),
        ClightExpr::Ebinop(ClightBinaryOp::Oadd | ClightBinaryOp::Osub, l, r, _) => {
            base_drives_free_var(l, tyvar) || base_drives_free_var(r, tyvar)
        }
        _ => false,
    }
}

/// A typed term with freeness: Conc folds in Rust with no Z3 emission, Sym flows a free variable so obligations can steer a register type; freeness decides fold-vs-emit only.
enum TyTerm {
    Conc(ClightType),
    Sym(Dynamic),
}

impl TyTerm {
    fn is_sym(&self) -> bool {
        matches!(self, TyTerm::Sym(_))
    }
    fn term(&self, t: &TypeDt) -> Dynamic {
        match self {
            TyTerm::Conc(ct) => clight_to_ty(t, ct),
            TyTerm::Sym(d) => d.clone(),
        }
    }
}

// True when an obligation over these operands can steer the model (any free variable involved); concrete sites fold silently.
fn any_sym(parts: &[&TyTerm]) -> bool {
    parts.iter().any(|p| p.is_sym())
}

// The single operator-typing rule: returns the TyTerm of `e` and emits the classify-derived obligations its sub-expressions impose (Cop.v/Ctyping.v at the post-typeconv class level); `tyvar` holds each inferable register's free variable, non-inferable leaves fall back to their annotated type.
fn type_of(
    t: &TypeDt,
    tyvar: &HashMap<RTLReg, Dynamic>,
    sink: &ConstraintSink,
    guard: Option<&Bool>,
    e: &ClightExpr,
) -> TyTerm {
    match e {
        ClightExpr::Etempvar(id, ct) => {
            match tyvar.get(&(*id as RTLReg)) {
                Some(tv) => {
                    // SEL-AGREE (class match): prefer the candidate whose annotation CLASS matches the register's model class, weight 2, so it outranks the candidate-[0] prior but yields to every semantic constraint.
                    {
                        let class_id: u8 = match ct {
                            ClightType::Tpointer(..) => 0,
                            ClightType::Tfloat(..) => 1,
                            ClightType::Tstruct(..) => 2,
                            ClightType::Tint(..) | ClightType::Tlong(..) => 3,
                            _ => u8::MAX,
                        };
                        if class_id != u8::MAX
                            && sink.agree_seen.borrow_mut().insert((*id as RTLReg, class_id))
                        {
                            let agree = match class_id {
                                0 => t.is_ptr(tv),
                                1 => t.is_float(tv),
                                2 => t.is_struct(tv),
                                _ => t.is_intish(tv),
                            };
                            sink.prefer(&agree, 2);
                        }
                    }
                    TyTerm::Sym(tv.clone())
                }
                None => TyTerm::Conc(ct.clone()),
            }
        }
        ClightExpr::EconstInt(_, ct)
        | ClightExpr::EconstLong(_, ct)
        | ClightExpr::EconstFloat(_, ct)
        | ClightExpr::EconstSingle(_, ct) => TyTerm::Conc(ct.clone()),
        ClightExpr::Evar(_, ct) | ClightExpr::EvarSymbol(_, ct) => TyTerm::Conc(ct.clone()),
        ClightExpr::Ederef(inner, ct) => {
            let it = type_of(t, tyvar, sink, guard, inner);
            // deref requires a pointer and the result is the pointee; assert only when the base drives a free variable (on a concrete base it is redundant or forces whole-function UNSAT) -- the D1-frozen hard set.
            if base_drives_free_var(inner, tyvar) {
                sink.assert_structural(guard, &t.is_ptr(&it.term(t)));
            }
            match it {
                TyTerm::Sym(d) => TyTerm::Sym(t.pointee(&d)),
                TyTerm::Conc(c) => {
                    // Fold: pointer/array yield the element; anything else is the re-pick's business -- recover with the annotation.
                    match crate::decompile::passes::clight_select::ctyping::type_deref(&c) {
                        Some(elem) => TyTerm::Conc(elem),
                        None => TyTerm::Conc(ct.clone()),
                    }
                }
            }
        }
        ClightExpr::Eaddrof(inner, _) => {
            let it = type_of(t, tyvar, sink, guard, inner);
            match it {
                TyTerm::Sym(d) => TyTerm::Sym(t.ptr(&d)),
                TyTerm::Conc(c) => TyTerm::Conc(ClightType::Tpointer(
                    std::sync::Arc::new(c),
                    crate::x86::types::ClightAttr::default(),
                )),
            }
        }
        ClightExpr::Efield(base, _, ct) => {
            let bt = type_of(t, tyvar, sink, guard, base);
            // the field base (for `e.f`, or the pointee for `e->f` = (*p).f) must be a struct; assert only when it drives a free variable, for the same reason as deref above.
            if base_drives_free_var(base, tyvar) {
                sink.assert_structural(guard, &t.is_struct(&bt.term(t)));
            } else if matches!(
                base.as_ref(),
                ClightExpr::Evar(_, ClightType::Tstruct(..))
                    | ClightExpr::Etempvar(_, ClightType::Tstruct(..))
            ) {
                // A struct-VALUE member access whose base folds only to a concrete Tstruct annotation is the member-of-non-struct error, priced here so a raw byte-offset alternative wins; p->f is untouched.
                sink.penalize_candidate(8);
            }
            TyTerm::Conc(ct.clone())
        }
        ClightExpr::Ecast(inner, ct) => {
            let it = type_of(t, tyvar, sink, guard, inner);
            if it.is_sym() {
                let itm = it.term(t);
                // Casting a floating operand to a pointer is illegal C, enforced HARD for an inferable register but soft otherwise so a single-candidate node is not forced UNSAT.
                if matches!(ct, ClightType::Tpointer(..)) {
                    let not_float = t.is_float(&itm).not();
                    if cast_operand_is_inferable(inner, tyvar) {
                        sink.assert_structural(guard, &not_float);
                    } else {
                        sink.assert_soft_tagged(guard, &not_float, 8, "cast_to_ptr");
                    }
                }
                // The mirror image, casting a pointer operand to a float, is dominant-SOFT rather than hard: not_ptr can conflict with a hard is_ptr from a deref of the same register and would force UNSAT.
                if matches!(ct, ClightType::Tfloat(..)) {
                    let not_ptr = t.is_ptrish(&itm).not();
                    let w = if cast_operand_is_inferable(inner, tyvar) { sink.compat_w } else { 8 };
                    sink.assert_soft_tagged(guard, &not_ptr, w, "cast_to_float");
                }
                // Remaining wt_cast mismatches (struct<->scalar, void source) are RECORDED-ONLY: the re-pick prices every cast exactly, and asserting these 16k+ clauses bought nothing.
                let tgt = clight_to_ty(t, ct);
                sink.prefer(&wt_cast_term(t, &itm, &tgt), 8);
                // The weight-1 SEL-AGREE cast-level pointerness nudge was retired as subsumed; the Sset-level and Etempvar class-match preferences are not subsumed and stay.
            }
            TyTerm::Conc(ct.clone())
        }
        ClightExpr::Ebinop(op, l, r, ct) => {
            let lt = type_of(t, tyvar, sink, guard, l);
            let rt = type_of(t, tyvar, sink, guard, r);
            if any_sym(&[&lt, &rt]) {
                let ltm = lt.term(t);
                let rtm = rt.term(t);
                // A float-typed arithmetic node requires non-pointer operands, since p + 0.1f is an invalid-operands error nothing silences; dominant-soft like the float-cast rule above.
                if matches!(ct, ClightType::Tfloat(..)) {
                    sink.assert_soft_tagged(guard, &t.is_ptrish(&ltm).not(), sink.compat_w, "float_binop");
                    sink.assert_soft_tagged(guard, &t.is_ptrish(&rtm).not(), sink.compat_w, "float_binop");
                }
                // The integer mirror: an INT-annotated int-only op has integer operands by RTL construction, stronger than classify_binarith, which legally admits floats and let registers drift float.
                if matches!(
                    op,
                    ClightBinaryOp::Omul
                        | ClightBinaryOp::Odiv
                        | ClightBinaryOp::Omod
                        | ClightBinaryOp::Oand
                        | ClightBinaryOp::Oor
                        | ClightBinaryOp::Oxor
                        | ClightBinaryOp::Oshl
                        | ClightBinaryOp::Oshr
                ) && matches!(ct, ClightType::Tint(..) | ClightType::Tlong(..))
                {
                    if lt.is_sym() {
                        sink.assert_soft_tagged(guard, &t.is_intish(&ltm), sink.compat_w, "int_binop");
                    }
                    if rt.is_sym() {
                        sink.assert_soft_tagged(guard, &t.is_intish(&rtm), sink.compat_w, "int_binop");
                    }
                }
                sink.assert_soft_tagged(guard, &binop_legal_term(t, *op, &ltm, &rtm), sink.compat_w, "binop_legal");
                // Pointer arithmetic carries the pointer operand's type so a deref of the result constrains the base's POINTEE; the pointer operand is itself inferred, so select it with ite.
                match op {
                    ClightBinaryOp::Oadd | ClightBinaryOp::Osub => {
                        // The result is a pointer ONLY if an operand is, so force a non-pointer fallback or a deref could be satisfied by the annotation and the base register would escape is_ptr.
                        let ann = match ct {
                            ClightType::Tpointer(..) => t.int(),
                            other => clight_to_ty(t, other),
                        };
                        let res = if matches!(op, ClightBinaryOp::Oadd) {
                            let inner = t.is_ptr(&rtm).ite(&rtm, &ann);
                            t.is_ptr(&ltm).ite(&ltm, &inner)
                        } else {
                            let both = Bool::and(&[t.is_ptr(&ltm), t.is_ptr(&rtm)]);
                            let l_only = t.is_ptr(&ltm).ite(&ltm, &ann);
                            both.ite(&t.int(), &l_only)
                        };
                        TyTerm::Sym(res)
                    }
                    _ => TyTerm::Conc(ct.clone()),
                }
            } else {
                // Fully concrete: fold (the re-pick prices any error); annotation is the result.
                TyTerm::Conc(ct.clone())
            }
        }
        ClightExpr::Eunop(op, inner, ct) => {
            let it = type_of(t, tyvar, sink, guard, inner);
            // type_unop legality (Ctyping.v:52): ! accepts any scalar; ~ integers only; -/__builtin_fabs numerics only. New in P4 -- previously unops were entirely unconstrained.
            if it.is_sym() {
                sink.assert_soft_tagged(guard, &unop_legal_term(t, *op, &it.term(t)), sink.compat_w, "unop_legal");
            }
            TyTerm::Conc(ct.clone())
        }
        ClightExpr::Econdition(c, a, b, ct) => {
            let tc = type_of(t, tyvar, sink, guard, c);
            // wt_bool on the ?: condition (new in P4): struct/union/void-typed conditions are frontend errors.
            if tc.is_sym() {
                sink.assert_soft_tagged(guard, &t.is_boolish(&tc.term(t)), sink.compat_w, "cond_bool");
            }
            let _ = type_of(t, tyvar, sink, guard, a);
            let _ = type_of(t, tyvar, sink, guard, b);
            TyTerm::Conc(ct.clone())
        }
        ClightExpr::Esizeof(_, ct) | ClightExpr::Ealignof(_, ct) => TyTerm::Conc(ct.clone()),
    }
}

// Walk a statement emitting operator-typing constraints, guarded so they bind only for the selected candidate; the old constrain_int_ops register scan is retired as subsumed.
fn constrain_stmt(
    t: &TypeDt,
    tyvar: &HashMap<RTLReg, Dynamic>,
    sink: &ConstraintSink,
    guard: Option<&Bool>,
    func: &FunctionData,
    name_to_ident: &HashMap<String, Ident>,
    func_ret: &Dynamic,
    compat_w: u64,
    stmt: &crate::x86::types::ClightStmt,
) {
    use crate::x86::types::ClightStmt;
    match stmt {
        ClightStmt::Sassign(lhs, rhs) => {
            let lt = type_of(t, tyvar, sink, guard, lhs);
            let rt = type_of(t, tyvar, sink, guard, rhs);
            if any_sym(&[&lt, &rt]) {
                sink.assert_soft_tagged(guard, &wt_cast_term(t, &rt.term(t), &lt.term(t)), compat_w, "assign_compat");
            }
        }
        ClightStmt::Sset(id, e) => {
            let rt = type_of(t, tyvar, sink, guard, e);
            if let Some(lv) = tyvar.get(&(*id as RTLReg)) {
                sink.assert_soft_tagged(guard, &wt_cast_to_reg(t, &rt.term(t), lv), compat_w, "set_compat");
                // SEL-AGREE (Sset level): prefer an rhs whose pointerness matches the destination, weight 2, which also demotes a float rhs into a pointer destination that the wt count cannot foresee.
                sink.prefer(&ptr_class_eq(t, lv, &rt.term(t)), 2);
            }
        }
        ClightStmt::Scall(ret, f, args) => {
            let ft = type_of(t, tyvar, sink, guard, f);
            // classify_fun (new in P4): a call through a non-function object is a frontend error; steer a register-typed callee toward pointer-ness (the model has no fnptr pointee discipline beyond ptrish).
            if ft.is_sym() {
                sink.assert_soft_tagged(guard, &t.is_ptrish(&ft.term(t)), compat_w, "callee_fun");
            }
            // Cross-function: bind each argument and the result to the callee's signature.
            let sig = callee_ident_from_expr(f, name_to_ident)
                .and_then(|id| func.callee_signatures.get(&id));
            for (i, a) in args.iter().enumerate() {
                let at = type_of(t, tyvar, sink, guard, a);
                if at.is_sym() {
                    if let Some(sig) = sig {
                        if let Some(pt) = sig.param_types.get(i) {
                            let want = xtype_to_ty(t, pt);
                            sink.assert_soft_tagged(guard, &wt_cast_term(t, &at.term(t), &want), compat_w, "call_arg");
                        }
                    }
                }
            }
            if let (Some(rid), Some(sig)) = (ret, sig) {
                if let Some(lv) = tyvar.get(&(*rid as RTLReg)) {
                    let want = xtype_to_ty(t, &sig.return_type);
                    sink.assert_soft_tagged(guard, &wt_cast_to_reg(t, &want, lv), compat_w, "call_ret");
                }
            }
            // return/void agreement (caller side): capturing the result of a void callee yields "void value not ignored as it ought to be", so when this call is a refinable candidate, penalize the result-capturing form so a non-capturing candidate is chosen.
            if let (Some(_), Some(sig)) = (ret, sig) {
                if matches!(sig.return_type, XType::Xvoid) {
                    if let Some(g) = guard {
                        sink.assert_soft_raw(&g.not(), compat_w);
                        sink.penalize_candidate(compat_w);
                    }
                }
            }
        }
        ClightStmt::Sreturn(ret) => {
            // return/void agreement on the callee side: a value-return in a void function or a valueless return in a non-void one is a frontend error, so penalize the mismatching form.
            let is_void_fn = matches!(func.return_type, ClightType::Tvoid);
            match ret {
                Some(e) => {
                    let et = type_of(t, tyvar, sink, guard, e);
                    if et.is_sym() {
                        sink.assert_soft_tagged(guard, &wt_cast_term(t, &et.term(t), func_ret), compat_w, "ret_compat");
                    }
                    if is_void_fn {
                        if let Some(g) = guard {
                            sink.assert_soft_raw(&g.not(), compat_w);
                            sink.penalize_candidate(compat_w);
                        }
                    }
                }
                None => {
                    if !is_void_fn {
                        if let Some(g) = guard {
                            sink.assert_soft_raw(&g.not(), compat_w);
                            sink.penalize_candidate(compat_w);
                        }
                    }
                }
            }
        }
        ClightStmt::Sifthenelse(c, a, b) => {
            let tc = type_of(t, tyvar, sink, guard, c);
            // wt_bool on the branch condition (new in P4): struct/union/void conditions are frontend errors.
            if tc.is_sym() {
                sink.assert_soft_tagged(guard, &t.is_boolish(&tc.term(t)), compat_w, "if_bool");
            }
            constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, a);
            constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, b);
        }
        ClightStmt::Ssequence(ss) => {
            for s in ss {
                constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, s);
            }
        }
        ClightStmt::Sloop(a, b) => {
            constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, a);
            constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, b);
        }
        ClightStmt::Slabel(_, inner) => {
            constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, inner);
        }
        ClightStmt::Sswitch(e, cases) => {
            let te = type_of(t, tyvar, sink, guard, e);
            // classify_switch (new in P4): the scrutinee must be an integer.
            if te.is_sym() {
                sink.assert_soft_tagged(guard, &t.is_intish(&te.term(t)), compat_w, "switch_int");
            }
            for (_, s) in cases {
                constrain_stmt(t, tyvar, sink, guard, func, name_to_ident, func_ret, compat_w, s);
            }
        }
        _ => {}
    }
}

// Render a model's inferred Ty for reg to a type-string; allow_void permits ptr_void only when the register is never a deref/field base. CLASS-level renders only, widths chosen at anchor_index.
fn render_type(
    t: &TypeDt,
    model: &z3::Model,
    tv: &Dynamic,
    reg: RTLReg,
    func: &FunctionData,
    allow_void: bool,
) -> String {
    let truth = |b: Bool| model.eval(&b, true).and_then(|x| x.as_bool()).unwrap_or(false);
    if truth(t.is_ptr(tv)) {
        let p = t.pointee(tv);
        if truth(t.is_struct(&p)) {
            if let Some(&sid) = func.reg_struct_ids.get(&reg) {
                return format!("ptr_struct_{:x}", sid);
            }
            // sid from the model if upstream didn't record one.
            if let Some(sid) = model
                .eval(&t.struct_sid(&p), true)
                .and_then(|x| x.as_int())
                .and_then(|i| i.as_i64())
            {
                return format!("ptr_struct_{:x}", sid);
            }
            return "ptr_I64".to_string();
        }
        if truth(t.is_float(&p)) {
            return "ptr_double".to_string();
        }
        if truth(t.is_func(&p)) {
            return "ptr_func".to_string();
        }
        if allow_void && truth(t.is_void(&p)) {
            return "ptr_void".to_string();
        }
        return "ptr_I64".to_string();
    }
    if truth(t.is_float(tv)) {
        return "float_F64".to_string();
    }
    "int_I64".to_string()
}

// Bit width of a candidate type-string, for nearest-width anchoring within a class.
fn type_str_width(s: &str) -> u32 {
    if s.starts_with("ptr_") { return 64; }
    if s == "float_F32" { return 32; }
    if s == "float_F64" { return 64; }
    if s == "int_IBool" { return 8; }
    if s.starts_with("int_I8") { return 8; }
    if s.starts_with("int_I16") { return 16; }
    if s.starts_with("int_I32") || s == "int_U32" { return 32; }
    if s.starts_with("int_I64") || s == "int_U64" { return 64; }
    64
}

// Pick the candidate index best matching the rendered inferred type; None means genuinely-new type (carry as override). `allow_void` mirrors render_type's license: ptr_void anchoring requires the register is never deref/field-based.
fn anchor_index(cands: &[String], rendered: &str, allow_void: bool) -> Option<usize> {
    // ptr_void without the license emits an incomplete pointee that can't be deref-assigned; fall through to prefer a complete scalar pointer.
    if rendered != "ptr_void" || allow_void {
        if let Some(i) = cands.iter().position(|c| c == rendered) {
            return Some(i);
        }
    }
    // Classes: struct-pointer is distinct from scalar/generic pointer, so a scalar-pointer inference never anchors back onto a `ptr_struct` candidate (which would reintroduce the struct mismatch).
    let class = |s: &str| -> u8 {
        if s.starts_with("ptr_struct_") { 0 }
        else if s.starts_with("ptr_") { 1 }
        else if s.starts_with("float_") { 2 }
        else { 3 }
    };
    let rc = class(rendered);
    if rc == 1 {
        // A scalar/generic pointer: prefer a COMPLETE pointee (int*/long*/char*) over void* (incomplete, not assignable through); only fall back to ptr_void if nothing complete is available.
        for pref in ["ptr_I64", "ptr_int", "ptr_char"] {
            if let Some(i) = cands.iter().position(|c| c == pref) {
                return Some(i);
            }
        }
        if let Some(i) = cands.iter().position(|c| class(c) == 1 && c != "ptr_void") {
            return Some(i);
        }
        // ptr_void is still a pointer-class anchor when nothing with a complete pointee is available.
        return cands.iter().position(|c| c == "ptr_void");
    }
    // Same class, no exact width: take nearest-width candidate; ties break to the earlier (higher-priority) index.
    let rw = type_str_width(rendered);
    cands
        .iter()
        .enumerate()
        .filter(|(_, c)| class(c) == rc)
        .min_by_key(|(i, c)| (type_str_width(c).abs_diff(rw), *i))
        .map(|(i, _)| i)
}

// Z3-only solve: a Sat model IS the selection, and UNSAT gives no selection so the consumer keeps upstream defaults; there is deliberately no relaxed retry, since UNSAT is an upstream bug.
fn solve_function(
    func: &FunctionData,
    name_to_ident: &HashMap<String, Ident>,
) -> (HashMap<Node, Option<usize>>, HashMap<RTLReg, usize>, HashMap<RTLReg, String>) {
    match solve_function_z3(func, name_to_ident) {
        Ok((c, ty, ov, grade)) => {
            if matches!(grade, SatResult::Unknown) {
                TYPE_SOLVE_PARTIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "[clight-select] WARN: type solve for {} ({:#x}) exhausted its rlimit budget mid-optimization; reusing the best model found ({} nodes, {} regs)",
                    func.name,
                    func.address,
                    func.node_statements.len(),
                    func.var_type_candidates.len()
                );
            }
            (c, ty, ov)
        }
        Err(first) => {
            TYPE_SOLVE_NO_MODEL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[clight-select] WARN: type solve for {} ({:#x}) produced no model ({}); emitting no selection, upstream defaults apply ({} nodes, {} regs)",
                func.name,
                func.address,
                match first {
                    SatResult::Unsat => "UNSAT",
                    SatResult::Unknown => "unknown (rlimit/timeout)",
                    SatResult::Sat => "sat but no model",
                },
                func.node_statements.len(),
                func.var_type_candidates.len()
            );
            (HashMap::new(), HashMap::new(), HashMap::new())
        }
    }
}

// The single Z3 solve. Returns the selection + the SatResult that yielded the model, or the failing SatResult when no model is available.
#[allow(clippy::type_complexity)]
fn solve_function_z3(
    func: &FunctionData,
    name_to_ident: &HashMap<String, Ident>,
) -> Result<
    (HashMap<Node, Option<usize>>, HashMap<RTLReg, usize>, HashMap<RTLReg, String>, SatResult),
    SatResult,
> {
    // Fresh Z3 context per function: reusing one leaks MaxSMT state across functions and makes tie-breaking depend on rayon order. Do not set random_seed -- z3 rejects it and warns to stderr.
    let cfg = z3::Config::new();
    z3::with_z3_config(&cfg, || {
    let build_t0 = std::time::Instant::now();
    let t = TypeDt::build();
    let opt = Optimize::new();
    let mut params = z3::Params::new();
    // Use the deterministic rlimit budget, not wall-clock timeout, which is a hang backstop only; z3 4.8.x ignores rlimit through Optimize params, so the model-reuse path guards itself.
    params.set_u32("rlimit", crate::decompile::elevator::config::SOLVE_RLIMIT);
    params.set_u32("timeout", crate::decompile::elevator::config::SOLVE_TIMEOUT_MS);
    params.set_u32("random_seed", 0);
    opt.set_params(&params);

    // Free type variable per inferable register (deterministic order).
    let mut regs: Vec<RTLReg> = func.var_type_candidates.keys().copied().collect();
    regs.sort();
    let mut tyvar: HashMap<RTLReg, Dynamic> = HashMap::new();
    for &reg in &regs {
        let tv = t.fresh();
        // Domain pruning: a register decl never renders as bare func/void/union (render_type falls back to int_I64), so excluding those variants HARD shrinks every clause's search space at zero MaxSMT cost and lets the Sset compat formula drop their arms (wt_cast_to_reg).
        opt.assert(&t.is_func(&tv).not());
        opt.assert(&t.is_void(&tv).not());
        opt.assert(&t.is_union(&tv).not());
        // Float-class pruning (RC-1): the is_float axis is otherwise free, so require positive upstream float evidence; excluding the float variant HARD cannot cause UNSAT since nothing forces is_float.
        let cands = func.var_type_candidates.get(&reg);
        let has_float_cand = cands
            .map(|cs| {
                cs.iter().any(|c| {
                    c.starts_with("float_") || c == "ptr_double" || c == "ptr_float"
                })
            })
            .unwrap_or(false);
        // RC-1: a register carrying a concrete non-float pointer candidate is a pointer, so close the is_float axis to stop MaxSMT fabricating double; ptr_double/ptr_float excluded.
        let has_concrete_ptr = cands
            .map(|cs| {
                cs.iter().any(|c| {
                    c.starts_with("ptr_") && c != "ptr_double" && c != "ptr_float"
                })
            })
            .unwrap_or(false);
        let has_float_evidence = has_float_cand && !has_concrete_ptr;
        if !has_float_evidence {
            opt.assert(&t.is_float(&tv).not());
        }
        tyvar.insert(reg, tv);
    }

    // Diagnostic: dump the candidates the lifting pass produced for a target function (by name substring), to distinguish "correct candidate absent (lifting bug)" from "present but not selected (selector bug)".
    if std::env::var("MANIFOLD_DUMP_CANDS").map(|s| !s.is_empty() && func.name.contains(&s)).unwrap_or(false) {
        eprintln!("=== CANDS fn={} addr={:#x} ===", func.name, func.address);
        for &reg in &regs {
            let nm = crate::decompile::passes::c_pass::helpers::param_name_for_reg(reg);
            eprintln!("  {} (reg {:#x}): {:?}", nm, reg, func.var_type_candidates[&reg]);
        }
        let mut ns: Vec<Node> = func.node_statements.keys().copied().collect();
        ns.sort();
        for n in &ns {
            let stmts = &func.node_statements[n];
            let dbg = format!("{:?}", stmts);
            if stmts.len() > 1 || dbg.contains("Ederef") {
                eprintln!("  node {:#x} ({} cand stmts):", n, stmts.len());
                for (i, s) in stmts.iter().enumerate() {
                    eprintln!("    [{}] {:?}", i, s);
                }
            }
        }
    }

    // Statement-selection booleans for multi-candidate nodes (exactly one per node).
    let mut nodes: Vec<Node> = func.node_statements.keys().copied().collect();
    nodes.sort();
    let mut selvar: HashMap<Node, Vec<Bool>> = HashMap::new();
    for &node in &nodes {
        let m = func.node_statements[&node].len();
        if m > 1 {
            let vars: Vec<Bool> = (0..m).map(|_| Bool::fresh_const("s")).collect();
            let refs: Vec<(&Bool, i32)> = vars.iter().map(|b| (b, 1)).collect();
            opt.assert(&Bool::pb_eq(&refs, 1));
            selvar.insert(node, vars);
        }
    }

    let func_ret = match &func.return_type {
        ClightType::Tvoid => t.void(),
        other => clight_to_ty(&t, other),
    };

    // Weight strata: priors 1 < SEL-AGREE 2 < cast nudges 8 < dominant compat_w. A x4 rescale with a separate canonical stratum was reverted, sending maxres from 13s to 75s on ls.
    let compat_w: u64 = (regs.len() + selvar.len()) as u64 + 1;
    let sink = ConstraintSink::new(&opt, compat_w);

    // Emit constraints for every candidate statement, guarded by its selection boolean when refinable; multi-candidate nodes also record their conditions for the post-solve re-pick.
    let mut recorded: HashMap<Node, Vec<Vec<RecordedCond>>> = HashMap::new();
    for &node in &nodes {
        let cands = &func.node_statements[&node];
        match selvar.get(&node) {
            Some(vars) => {
                let mut per_cand = Vec::with_capacity(cands.len());
                for (ci, stmt) in cands.iter().enumerate() {
                    sink.begin_candidate(true);
                    constrain_stmt(&t, &tyvar, &sink, Some(&vars[ci]), func, name_to_ident, &func_ret, compat_w, stmt);
                    per_cand.push(sink.take_recorded());
                }
                recorded.insert(node, per_cand);
            }
            None => {
                sink.begin_candidate(false);
                constrain_stmt(&t, &tyvar, &sink, None, func, name_to_ident, &func_ret, compat_w, &cands[0]);
            }
        }
    }

    // Soft provenance: prefer each register's priority-0 candidate type, and the priority-0 statement.
    for &reg in &regs {
        if let Some(c0) = func.var_type_candidates[&reg].first() {
            let prior = cand_str_to_ty(&t, c0);
            sink.assert_soft_raw(&tyvar[&reg]._eq(prior), 1u64);
        }
    }
    let mut sel_nodes: Vec<Node> = selvar.keys().copied().collect();
    sel_nodes.sort();
    for node in &sel_nodes {
        sink.assert_soft_raw(&selvar[node][0], 1u64);
    }
    // (No canonical tie-break clauses: the width/signedness/float-size payload dimensions that created solver-side ties were REMOVED from the datatype -- see the TypeDt comment. Class-level ties cannot reach the output: the anchor maps each class to one deterministic candidate.)

    let mut cand_out: HashMap<Node, Option<usize>> = HashMap::new();
    let mut ty_out: HashMap<RTLReg, usize> = HashMap::new();
    let mut ty_override: HashMap<RTLReg, String> = HashMap::new();

    BUILD_NS.fetch_add(
        build_t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let check_t0 = std::time::Instant::now();
    let check_result = opt.check(&[]);
    CHECK_NS.fetch_add(
        check_t0.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let model = match check_result {
        SatResult::Sat => opt.get_model(),
        // Partial-solve reuse (TR-1): on unknown, reuse the best-so-far model only when the deterministic rlimit was exhausted, since a wall-clock unknown would be timing-dependent.
        SatResult::Unknown => {
            let resource_budget = opt
                .get_reason_unknown()
                .map(|r| r.contains("resource"))
                .unwrap_or(false);
            if resource_budget { opt.get_model() } else { None }
        }
        SatResult::Unsat => None,
    };

    match model {
        Some(m) => {
            // P3: the re-pick scores each candidate by its EXACT frontend-error count under the model, covering concrete bases that the z3 structural asserts exempt; allow_void=false breaks the cycle.
            let wt_env = {
                let mut temps: HashMap<Ident, ClightType> = HashMap::new();
                for &reg in &regs {
                    let rendered = render_type(&t, &m, &tyvar[&reg], reg, func, false);
                    // Type by the ANCHORED candidate, not the raw class render: the emitter prints candidates[anchor], so the anchored string is the decl gcc will see -- width/signedness fidelity the class-level model no longer carries (deterministic: anchor_index is a pure function).
                    let cands = &func.var_type_candidates[&reg];
                    let decl_str = anchor_index(cands, &rendered, false)
                        .and_then(|i| cands.get(i).cloned())
                        .unwrap_or(rendered);
                    temps.insert(
                        reg as Ident,
                        crate::decompile::passes::clight_select::wt_audit::type_string_to_clight(&decl_str),
                    );
                }
                for (reg, s) in &func.var_types {
                    if !tyvar.contains_key(reg) {
                        temps.entry(*reg as Ident).or_insert_with(|| {
                            crate::decompile::passes::clight_select::wt_audit::type_string_to_clight(s)
                        });
                    }
                }
                // Params are a gap-filler ONLY: for an inferable param reg the MODEL rendering must win, or the signature's ParamType clobbers a model-typed struct pointer and flips selection.
                for (reg, pt) in func.param_regs.iter().zip(func.param_types.iter()) {
                    if tyvar.contains_key(reg) {
                        continue;
                    }
                    temps
                        .entry(*reg as Ident)
                        .or_insert_with(|| crate::decompile::passes::clight_select::wt_audit::param_type_to_clight(pt));
                }
                let (fields, known) =
                    crate::decompile::passes::clight_select::wt_audit::fields_from_struct_fields(&func.struct_fields);
                crate::decompile::passes::clight_select::wt_audit::AuditEnv::from_parts(
                    temps,
                    fields,
                    known,
                    func.return_type.clone(),
                    &func.callee_signatures,
                    name_to_ident,
                )
            };
            for &node in &nodes {
                match selvar.get(&node) {
                    Some(vars) => {
                        // z3's own pick (consistent with every asserted constraint) is the fallback when nothing was recorded.
                        let mut chosen = 0usize;
                        for (i, b) in vars.iter().enumerate() {
                            if m.eval(b, true).and_then(|x| x.as_bool()).unwrap_or(false) {
                                chosen = i;
                                break;
                            }
                        }
                        // Exact-wt re-pick (P3): with the model fixed, score each candidate by fewest exact frontend errors, then structural conditions, then soft weight, then index, and take the minimum.
                        if let Some(per_cand) = recorded.get(&node) {
                            let truth = |c: &Bool| {
                                m.eval(c, true).and_then(|x| x.as_bool()).unwrap_or(true)
                            };
                            let cands = &func.node_statements[&node];
                            let mut best: Option<(u64, u64, u64, u64, usize)> = None;
                            for (ci, conds) in per_cand.iter().enumerate() {
                                let wt = cands
                                    .get(ci)
                                    .map(|stmt| {
                                        let errs = crate::decompile::passes::clight_select::ctyping::wt_check_stmt(stmt, &wt_env);
                                        crate::decompile::passes::clight_select::ctyping::error_count(&errs) as u64
                                    })
                                    .unwrap_or(0);
                                // Lossy 64->32 truncations: ranked above soft preferences, uniform when truncation is unavoidable.
                                let narrow = cands.get(ci).map(count_truncating_casts_stmt).unwrap_or(0);
                                let mut sviol = 0u64;
                                let mut wviol = 0u64;
                                for (cond, w) in conds {
                                    let ok = match cond {
                                        Some(c) => truth(c),
                                        None => false,
                                    };
                                    if !ok {
                                        if *w == u64::MAX {
                                            sviol += 1;
                                        } else {
                                            wviol = wviol.saturating_add(*w);
                                        }
                                    }
                                }
                                if std::env::var("MANIFOLD_DUMP_CANDS").map(|s| !s.is_empty() && func.name.contains(&s)).unwrap_or(false) {
                                    let errs = cands.get(ci).map(|stmt| crate::decompile::passes::clight_select::ctyping::wt_check_stmt(stmt, &wt_env)).unwrap_or_default();
                                    eprintln!("  repick node {:#x} cand[{}]: wt={} sviol={} narrow={} wviol={} errs={:?}", node, ci, wt, sviol, narrow, wviol,
                                        errs.iter().map(|e| format!("{}:{}", e.kind.gcc_family(), e.detail)).collect::<Vec<_>>());
                                }
                                let key = (wt, sviol, narrow, wviol, ci);
                                if best.is_none_or(|b| key < b) {
                                    best = Some(key);
                                }
                            }
                            if let Some((_, _, _, _, ci)) = best {
                                chosen = ci;
                            }
                        }
                        cand_out.insert(node, Some(chosen));
                    }
                    None => {
                        cand_out.insert(node, Some(0));
                    }
                }
            }
            // TR-6: void* license -- denied for any register used as deref/field base (casts included) in any selected statement, so an incomplete pointee can never break an emitted deref.
            let mut ptr_ev: BTreeMap<RTLReg, RegPtrEvidence> = BTreeMap::new();
            for &node in &nodes {
                if let Some(Some(ci)) = cand_out.get(&node) {
                    if let Some(stmt) = func.node_statements[&node].get(*ci) {
                        collect_reg_ptr_evidence_stmt(stmt, &mut ptr_ev);
                    }
                }
            }
            for &reg in &regs {
                let allow_void = ptr_ev.get(&reg).map(|e| !e.any_deref).unwrap_or(true);
                let rendered = render_type(&t, &m, &tyvar[&reg], reg, func, allow_void);
                let cands = &func.var_type_candidates[&reg];
                match anchor_index(cands, &rendered, allow_void) {
                    Some(idx) => {
                        ty_out.insert(reg, idx);
                    }
                    None => {
                        // Belt-and-suspenders (RC-1): an override may widen width within an evidenced class but must never invent the float CLASS, so a float render absent from the candidate set falls back to the anchor.
                        let render_is_float =
                            rendered.starts_with("float_");
                        let cands_have_float = cands.iter().any(|c| {
                            c.starts_with("float_") || c == "ptr_double" || c == "ptr_float"
                        });
                        if render_is_float && !cands_have_float {
                            ty_out.insert(reg, 0);
                        } else {
                            // Genuinely-new inferred type: carry it out-of-band so the emitter still sees it.
                            ty_out.insert(reg, 0);
                            ty_override.insert(reg, rendered);
                        }
                    }
                }
            }
            // TEMP DEBUG: dump the solved model (rendered reg types + chosen candidate per node) for MANIFOLD_DUMP_CANDS-matched functions.
            if std::env::var("MANIFOLD_DUMP_CANDS").map(|s| !s.is_empty() && func.name.contains(&s)).unwrap_or(false) {
                eprintln!("=== MODEL fn={} addr={:#x} ({:?}) ===", func.name, func.address, check_result);
                for &reg in &regs {
                    let nm = crate::decompile::passes::c_pass::helpers::param_name_for_reg(reg);
                    let allow_void = ptr_ev.get(&reg).map(|e| !e.any_deref).unwrap_or(true);
                    let rendered = render_type(&t, &m, &tyvar[&reg], reg, func, allow_void);
                    eprintln!("  {} -> {} (anchor={:?}, override={:?})", nm, rendered, ty_out.get(&reg), ty_override.get(&reg));
                }
                let mut ns: Vec<Node> = cand_out.keys().copied().collect();
                ns.sort();
                for n in ns {
                    if func.node_statements.get(&n).map(|s| s.len() > 1).unwrap_or(false) {
                        eprintln!("  node {:#x} chose [{}]", n, cand_out[&n].map(|i| i as i64).unwrap_or(-1));
                    }
                }
            }
            Ok((cand_out, ty_out, ty_override, check_result))
        }
        None => Err(check_result),
    }
    })
}

// Run-wide counters: NO_MODEL = z3 yielded no model (function shipped with no selection), PARTIAL = over-budget best-so-far model reused.
static TYPE_SOLVE_NO_MODEL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TYPE_SOLVE_PARTIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// ---- per-register structural pointer-use evidence (TR-6 void license) ----

#[derive(Default, Clone, Copy)]
struct RegPtrEvidence {
    // Deref/field involvement in any form including casts -- denies TR-6 void* license even when solver does not hard-assert (memory is still read through the register).
    any_deref: bool,
}

// Deref/field chain root (tempvar, deref, add/sub), peeling casts -- conservative root set for denying the void-pointee license.
fn loose_chain_root(e: &ClightExpr) -> Option<RTLReg> {
    match e {
        ClightExpr::Etempvar(id, _) => Some(*id as RTLReg),
        ClightExpr::Ederef(inner, _) | ClightExpr::Ecast(inner, _) => loose_chain_root(inner),
        ClightExpr::Ebinop(ClightBinaryOp::Oadd | ClightBinaryOp::Osub, l, r, _) => {
            loose_chain_root(l).or_else(|| loose_chain_root(r))
        }
        _ => None,
    }
}

fn collect_reg_ptr_evidence_expr(e: &ClightExpr, out: &mut BTreeMap<RTLReg, RegPtrEvidence>) {
    match e {
        ClightExpr::Ederef(inner, _) => {
            if let Some(r) = loose_chain_root(inner) {
                out.entry(r).or_default().any_deref = true;
            }
            collect_reg_ptr_evidence_expr(inner, out);
        }
        ClightExpr::Efield(base, _, _) => {
            if let Some(r) = loose_chain_root(base) {
                out.entry(r).or_default().any_deref = true;
            }
            collect_reg_ptr_evidence_expr(base, out);
        }
        ClightExpr::Ecast(inner, _)
        | ClightExpr::Eaddrof(inner, _)
        | ClightExpr::Eunop(_, inner, _) => {
            collect_reg_ptr_evidence_expr(inner, out);
        }
        ClightExpr::Ebinop(_, l, r, _) => {
            collect_reg_ptr_evidence_expr(l, out);
            collect_reg_ptr_evidence_expr(r, out);
        }
        ClightExpr::Econdition(c, a, b, _) => {
            collect_reg_ptr_evidence_expr(c, out);
            collect_reg_ptr_evidence_expr(a, out);
            collect_reg_ptr_evidence_expr(b, out);
        }
        _ => {}
    }
}

fn collect_reg_ptr_evidence_stmt(stmt: &ClightStmt, out: &mut BTreeMap<RTLReg, RegPtrEvidence>) {
    match stmt {
        ClightStmt::Sassign(l, r) => {
            collect_reg_ptr_evidence_expr(l, out);
            collect_reg_ptr_evidence_expr(r, out);
        }
        ClightStmt::Sset(_, e) => collect_reg_ptr_evidence_expr(e, out),
        ClightStmt::Scall(_, f, args) => {
            collect_reg_ptr_evidence_expr(f, out);
            for a in args {
                collect_reg_ptr_evidence_expr(a, out);
            }
        }
        ClightStmt::Sreturn(Some(e)) => collect_reg_ptr_evidence_expr(e, out),
        ClightStmt::Sifthenelse(c, a, b) => {
            collect_reg_ptr_evidence_expr(c, out);
            collect_reg_ptr_evidence_stmt(a, out);
            collect_reg_ptr_evidence_stmt(b, out);
        }
        ClightStmt::Ssequence(ss) => {
            for s in ss {
                collect_reg_ptr_evidence_stmt(s, out);
            }
        }
        ClightStmt::Sloop(a, b) => {
            collect_reg_ptr_evidence_stmt(a, out);
            collect_reg_ptr_evidence_stmt(b, out);
        }
        ClightStmt::Slabel(_, inner) => collect_reg_ptr_evidence_stmt(inner, out),
        ClightStmt::Sswitch(e, cases) => {
            collect_reg_ptr_evidence_expr(e, out);
            for (_, s) in cases {
                collect_reg_ptr_evidence_stmt(s, out);
            }
        }
        _ => {}
    }
}

// Static type annotation carried by any Clight expression (the last field of every variant).
fn clight_expr_type(e: &ClightExpr) -> &ClightType {
    match e {
        ClightExpr::EconstInt(_, t)
        | ClightExpr::EconstFloat(_, t)
        | ClightExpr::EconstSingle(_, t)
        | ClightExpr::EconstLong(_, t)
        | ClightExpr::Evar(_, t)
        | ClightExpr::EvarSymbol(_, t)
        | ClightExpr::Etempvar(_, t)
        | ClightExpr::Ederef(_, t)
        | ClightExpr::Eaddrof(_, t)
        | ClightExpr::Eunop(_, _, t)
        | ClightExpr::Ebinop(_, _, _, t)
        | ClightExpr::Ecast(_, t)
        | ClightExpr::Efield(_, _, t)
        | ClightExpr::Esizeof(_, t)
        | ClightExpr::Ealignof(_, t)
        | ClightExpr::Econdition(_, _, _, t) => t,
    }
}

// Price lossy 64->32 Ecast narrowings as a tie-breaker only, steering away from a needless truncation when a width-preserving candidate exists; float sources excluded.
fn is_wide64_lossy_source(t: &ClightType) -> bool {
    matches!(
        t,
        ClightType::Tpointer(_, _) | ClightType::Tfunction(_, _, _) | ClightType::Tlong(_, _)
    )
}

fn is_narrow_int_target(t: &ClightType) -> bool {
    matches!(
        t,
        ClightType::Tint(
            ClightIntSize::I8 | ClightIntSize::I16 | ClightIntSize::I32 | ClightIntSize::IBool,
            _,
            _
        )
    )
}

fn count_truncating_casts_expr(e: &ClightExpr) -> u64 {
    let mut n = 0u64;
    if let ClightExpr::Ecast(inner, target) = e {
        if is_wide64_lossy_source(clight_expr_type(inner)) && is_narrow_int_target(target) {
            n += 1;
        }
    }
    match e {
        ClightExpr::Ederef(i, _)
        | ClightExpr::Eaddrof(i, _)
        | ClightExpr::Eunop(_, i, _)
        | ClightExpr::Ecast(i, _)
        | ClightExpr::Efield(i, _, _) => n += count_truncating_casts_expr(i),
        ClightExpr::Ebinop(_, l, r, _) => {
            n += count_truncating_casts_expr(l);
            n += count_truncating_casts_expr(r);
        }
        ClightExpr::Econdition(c, a, b, _) => {
            n += count_truncating_casts_expr(c);
            n += count_truncating_casts_expr(a);
            n += count_truncating_casts_expr(b);
        }
        _ => {}
    }
    n
}

fn count_truncating_casts_stmt(stmt: &ClightStmt) -> u64 {
    match stmt {
        ClightStmt::Sassign(l, r) => {
            count_truncating_casts_expr(l) + count_truncating_casts_expr(r)
        }
        ClightStmt::Sset(_, e) => count_truncating_casts_expr(e),
        ClightStmt::Scall(_, f, args) => {
            count_truncating_casts_expr(f) + args.iter().map(count_truncating_casts_expr).sum::<u64>()
        }
        ClightStmt::Sreturn(Some(e)) => count_truncating_casts_expr(e),
        ClightStmt::Sifthenelse(c, _, _) => count_truncating_casts_expr(c),
        ClightStmt::Sswitch(e, _) => count_truncating_casts_expr(e),
        ClightStmt::Slabel(_, inner) => count_truncating_casts_stmt(inner),
        _ => 0,
    }
}

// Whole-program selection: one Z3 solve per function (in parallel), merged into a ProgramSelectionState. Z3 is the only selector; functions whose solve yields no model simply contribute no entries.
pub(crate) fn infer_select_program(
    functions: &[FunctionData],
    name_to_ident: &HashMap<String, Ident>,
) -> ProgramSelectionState {
    let per_func: Vec<(Address, HashMap<Node, Option<usize>>, HashMap<RTLReg, usize>, HashMap<RTLReg, String>)> =
        functions
            .par_iter()
            .map(|f| {
                let (c, t, o) = solve_function(f, name_to_ident);
                (f.address, c, t, o)
            })
            .collect();

    // Run summary: how many functions shipped a degraded type solve (per-function WARN lines above carry the detail). Swap-reset so in-process multi-binary runs (tests) report per program.
    let no_model = TYPE_SOLVE_NO_MODEL.swap(0, std::sync::atomic::Ordering::Relaxed);
    let partial = TYPE_SOLVE_PARTIAL.swap(0, std::sync::atomic::Ordering::Relaxed);
    if no_model > 0 || partial > 0 {
        eprintln!(
            "[clight-select] WARN: type solve degraded for {}/{} functions ({} partial-model reuse, {} no model / no selection)",
            no_model + partial,
            functions.len(),
            partial,
            no_model
        );
    }
    let softs = SOFT_CLAUSES.swap(0, std::sync::atomic::Ordering::Relaxed);
    let build_ms = BUILD_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / 1_000_000;
    let check_ms = CHECK_NS.swap(0, std::sync::atomic::Ordering::Relaxed) / 1_000_000;
    eprintln!(
        "[clight-select] asserted soft clauses: {} across {} functions (cpu: build {}ms, check {}ms)",
        softs,
        functions.len(),
        build_ms,
        check_ms
    );
    if std::env::var("MANIFOLD_SOFT_BREAKDOWN").is_ok() {
        let mut tags = SOFT_TAGS.lock().unwrap();
        let line = tags
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[clight-select] soft breakdown: {}", line);
        tags.clear();
    }

    let mut candidate_idx = HashMap::new();
    let mut var_decl_idx = HashMap::new();
    let mut var_type_override = HashMap::new();
    for (addr, cmap, tmap, omap) in per_func {
        for (node, idx) in cmap {
            candidate_idx.insert((addr, node), idx);
        }
        for (reg, idx) in tmap {
            var_decl_idx.insert((addr, reg), idx);
        }
        for (reg, s) in omap {
            var_type_override.insert((addr, reg), s);
        }
    }

    ProgramSelectionState {
        candidate_idx,
        var_decl_idx,
        var_type_override,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::passes::clight_select::ctyping;
    use crate::x86::types::{ClightAttr, ClightFloatSize, ClightIntSize, ClightSignedness, ClightUnaryOp};
    use std::sync::Arc;
    use z3::ast::Ast; // for `.simplify()` on Bool, used in tests below

    fn a() -> ClightAttr {
        ClightAttr::default()
    }

    // Class-level representatives: every Ty variant, both composite-id cases.
    fn type_set() -> Vec<ClightType> {
        use ClightFloatSize::*;
        use ClightIntSize::*;
        use ClightSignedness::*;
        vec![
            ClightType::Tvoid,
            ClightType::Tint(I32, Signed, a()),
            ClightType::Tint(I32, Unsigned, a()),
            ClightType::Tlong(Signed, a()),
            ClightType::Tlong(Unsigned, a()),
            ClightType::Tfloat(F32, a()),
            ClightType::Tfloat(F64, a()),
            ClightType::Tpointer(Arc::new(ClightType::Tlong(Signed, a())), a()),
            ClightType::Tpointer(Arc::new(ClightType::Tvoid), a()),
            ClightType::Tpointer(Arc::new(ClightType::Tfloat(F64, a())), a()),
            ClightType::Tpointer(Arc::new(ClightType::Tstruct(1, a())), a()),
            ClightType::Tstruct(1, a()),
            ClightType::Tstruct(2, a()),
            ClightType::Tunion(7, a()),
            ClightType::Tunion(8, a()),
            ClightType::Tfunction(Arc::new(vec![]), Arc::new(ClightType::Tvoid), Default::default()),
        ]
    }

    // A closed (variable-free) Bool term is valid iff it is true; the solver check is the robust ground-truth evaluator (simplify alone can leave datatype tester applications unreduced).
    fn eval_closed(b: &Bool) -> bool {
        if let Some(v) = b.clone().simplify().as_bool() {
            return v;
        }
        let s = z3::Solver::new();
        s.assert(&b.not());
        matches!(s.check(), SatResult::Unsat)
    }

    /// The Z3 relaxation and the executable checker must agree on every concrete class-level pair except the composite-id relaxation, where the test asserts soundness only.
    #[test]
    fn wt_cast_term_matches_ctyping() {
        z3::with_z3_config(&z3::Config::new(), || {
            let t = TypeDt::build();
            for from in type_set() {
                for to in type_set() {
                    let term = wt_cast_term(&t, &clight_to_ty(&t, &from), &clight_to_ty(&t, &to));
                    let z3_legal = eval_closed(&term);
                    let ct_legal = ctyping::wt_cast(&from, &to);
                    let same_kind_composites = matches!(
                        (&from, &to),
                        (ClightType::Tstruct(..), ClightType::Tstruct(..))
                            | (ClightType::Tunion(..), ClightType::Tunion(..))
                    );
                    if same_kind_composites {
                        assert!(
                            z3_legal || !ct_legal,
                            "wt_cast soundness: {} -> {}",
                            ctyping::ty_str(&from),
                            ctyping::ty_str(&to)
                        );
                    } else {
                        assert_eq!(
                            z3_legal,
                            ct_legal,
                            "wt_cast parity: {} -> {} (z3 {}, ctyping {})",
                            ctyping::ty_str(&from),
                            ctyping::ty_str(&to),
                            z3_legal,
                            ct_legal
                        );
                    }
                }
            }
        });
    }

    /// wt_cast_to_reg is wt_cast_term specialized to the pruned register domain (no bare func/void/union targets) -- they must agree there.
    #[test]
    fn wt_cast_to_reg_matches_full_on_pruned_domain() {
        use ClightFloatSize::*;
        use ClightSignedness::*;
        let reg_targets = vec![
            ClightType::Tint(ClightIntSize::I32, Signed, a()),
            ClightType::Tlong(Unsigned, a()),
            ClightType::Tfloat(F64, a()),
            ClightType::Tpointer(Arc::new(ClightType::Tlong(Signed, a())), a()),
            ClightType::Tstruct(1, a()),
        ];
        z3::with_z3_config(&z3::Config::new(), || {
            let t = TypeDt::build();
            for from in type_set() {
                for to in &reg_targets {
                    let full = eval_closed(&wt_cast_term(&t, &clight_to_ty(&t, &from), &clight_to_ty(&t, to)));
                    let spec = eval_closed(&wt_cast_to_reg(&t, &clight_to_ty(&t, &from), &clight_to_ty(&t, to)));
                    assert_eq!(
                        full,
                        spec,
                        "to-reg specialization: {} -> {}",
                        ctyping::ty_str(&from),
                        ctyping::ty_str(to)
                    );
                }
            }
        });
    }

    #[test]
    fn binop_legality_matches_ctyping() {
        use ClightBinaryOp::*;
        let ops = [
            Oadd, Osub, Omul, Odiv, Omod, Oand, Oor, Oxor, Oshl, Oshr, Oeq, One, Olt, Ogt, Ole,
            Oge,
        ];
        z3::with_z3_config(&z3::Config::new(), || {
            let t = TypeDt::build();
            for op in ops {
                for l in type_set() {
                    for r in type_set() {
                        let term =
                            binop_legal_term(&t, op, &clight_to_ty(&t, &l), &clight_to_ty(&t, &r));
                        let z3_legal = eval_closed(&term);
                        let ct_legal = ctyping::type_binop(op, &l, &r).is_some();
                        assert_eq!(
                            z3_legal,
                            ct_legal,
                            "binop parity: {:?} on ({}, {})",
                            op,
                            ctyping::ty_str(&l),
                            ctyping::ty_str(&r)
                        );
                    }
                }
            }
        });
    }

    #[test]
    fn unop_legality_matches_ctyping() {
        use ClightUnaryOp::*;
        z3::with_z3_config(&z3::Config::new(), || {
            let t = TypeDt::build();
            for op in [Onotbool, Onotint, Oneg, Oabsfloat] {
                for x in type_set() {
                    let term = unop_legal_term(&t, op, &clight_to_ty(&t, &x));
                    let z3_legal = eval_closed(&term);
                    let ct_legal = ctyping::type_unop(op, &x).is_some();
                    assert_eq!(
                        z3_legal,
                        ct_legal,
                        "unop parity: {:?} on {}",
                        op,
                        ctyping::ty_str(&x)
                    );
                }
            }
        });
    }

    /// D8 lockstep: every string render_type can produce parses to the intended class in BOTH consumers (the emitter's xtype_string_to_ctype and the audit/re-pick bridge type_string_to_clight) -- no silent long-fallbacks.
    #[test]
    fn render_vocabulary_lockstep() {
        use crate::decompile::passes::c_pass::types::CType;
        use crate::decompile::passes::clight_select::wt_audit::type_string_to_clight;
        // Exactly the strings render_type can emit (class-level set).
        let renders = [
            "int_I64",
            "float_F64",
            "ptr_I64",
            "ptr_void",
            "ptr_double",
            "ptr_func",
            "ptr_struct_2a",
        ];
        for s in renders {
            let ctype = crate::decompile::passes::c_pass::helpers::xtype_string_to_ctype(s);
            let clight = type_string_to_clight(s);
            let want_ptr = s.starts_with("ptr_");
            assert_eq!(
                matches!(ctype, CType::Pointer(..)),
                want_ptr,
                "helpers parse of {} lost pointerness",
                s
            );
            assert_eq!(
                matches!(clight, ClightType::Tpointer(..)),
                want_ptr,
                "bridge parse of {} lost pointerness",
                s
            );
            if s.starts_with("float_") {
                assert!(matches!(ctype, CType::Float(_)), "helpers parse of {}", s);
                assert!(matches!(clight, ClightType::Tfloat(..)), "bridge parse of {}", s);
            }
        }
    }
}
