//! Bounded, provider-owned snapshots of semantically equivalent final C forms.
//!
//! `ForLoopPass` and `VarReducePass` are destructive post-selection passes.  Their
//! inputs are already typed C trees, so retaining an input function when the pass
//! changes it preserves a much stronger provenance boundary than reconstructing an
//! alternative later from printed source.  The canonical translation unit remains
//! unchanged; this module only records at most one input snapshot per boundary and
//! function for downstream best-candidate evaluation.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::abi::BinaryFormat;
use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::c_pass::types::{
    CBlockItem, CExpr, CStmt, FuncDef, TopLevelDecl, TranslationUnit,
};
use crate::decompile::passes::clight_select::select::{
    ScalarSourceAlternativeFamily, SelectedFunction, Stage4SourceProfile,
};

pub const SOURCE_ALTERNATIVES_SCHEMA: &str = "manifold-source-alternatives-v3";
pub const SOURCE_ALTERNATIVES_SCHEMA_V4: &str = "manifold-source-alternatives-v4";
/// Two ordinary rows plus at most two typed, two field, and four extension
/// rows. No unused wire capacity is advertised.
pub const MAX_SOURCE_ALTERNATIVES_PER_FUNCTION: usize = 10;
/// V4 adds at most two content-distinct forms for each of sixteen independent
/// Stage-4 roots while retaining all ten historical v3 rows.
pub const MAX_SOURCE_ALTERNATIVES_PER_FUNCTION_V4: usize = 42;
pub const MAX_TOTAL_SOURCE_ALTERNATIVES: usize = 4096;
pub const MAX_SOURCE_ALTERNATIVE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES: usize = 64 * 1024 * 1024;

/// Final provider classifications whose ordinary C remains available only as
/// fallback or is omitted altogether.  An alternative source form must never
/// compete with the structured partial/unsupported/machine-state artifact for
/// the same function.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct SourceAlternativeExclusions {
    suppress_all: bool,
    partial_functions: HashSet<u64>,
    unsupported_functions: HashSet<u64>,
    omitted_functions: HashSet<u64>,
    machine_state_functions: HashSet<(u64, String)>,
}

impl SourceAlternativeExclusions {
    pub fn suppress_all() -> Self {
        Self {
            suppress_all: true,
            ..Default::default()
        }
    }

    fn excludes(&self, name: &str, address: u64) -> bool {
        self.partial_functions.contains(&address)
            || self.unsupported_functions.contains(&address)
            || self.omitted_functions.contains(&address)
            || self
                .machine_state_functions
                .iter()
                .any(|(stub_address, stub_name)| {
                    *stub_address == address && stub_name == name
                })
    }
}

/// Derive source-alternative eligibility from the same authenticated provider
/// classifiers used by Clight selection/export.  No printed JSON, function
/// spelling, or downstream score participates in this decision.
pub fn final_source_alternative_exclusions(
    db: &DecompileDB,
    exact_function_identities: &[(String, u64)],
) -> SourceAlternativeExclusions {
    let validation =
        crate::decompile::passes::clight_select::select::validate_partial_unsupported_functions(db);
    let suppress_all = !validation.artifact_bundle_is_valid();
    let partial_functions: HashSet<u64> = validation.partial_functions.keys().copied().collect();
    let unsupported_functions: HashSet<u64> = validation
        .diagnosed_functions
        .iter()
        .filter(|function| !partial_functions.contains(function))
        .copied()
        .collect();
    let omitted_functions = crate::decompile::passes::clight_select::select::
        unsupported_functions_requiring_omission_from_validation(&validation);
    let exact_function_identities: HashSet<(u64, &str)> = exact_function_identities
        .iter()
        .map(|(name, address)| (*address, name.as_str()))
        .collect();
    let machine_state_functions = db
        .coff_address_map
        .as_ref()
        .map(|map| {
            crate::decompile::disassembly::machine_state::recognize_machine_state_stubs(db, map)
                .into_iter()
                .filter_map(|record| {
                    let address = record.function_address_value()?;
                    let name = record.function_name();
                    exact_function_identities
                        .contains(&(address, name))
                        .then(|| (address, name.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    SourceAlternativeExclusions {
        suppress_all,
        partial_functions,
        unsupported_functions,
        omitted_functions,
        machine_state_functions,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceAlternativeBoundary {
    PreForLoop,
    PreVarReduce,
    ScalarLvaluePreVarReduce,
    ScalarLvaluePostVarReduce,
    ScalarFieldPreVarReduce,
    ScalarFieldPostVarReduce,
    ScalarExtensionPerUsePreVarReduce,
    ScalarExtensionPerUsePostVarReduce,
    ScalarExtensionHoistedPreVarReduce,
    ScalarExtensionHoistedPostVarReduce,
    Stage4AffinePreVarReduce,
    Stage4AffinePostVarReduce,
    Stage4ZeroPreVarReduce,
    Stage4ZeroPostVarReduce,
    Stage4RmwPreVarReduce,
    Stage4RmwPostVarReduce,
}

impl SourceAlternativeBoundary {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::PreForLoop => "pre_forloop",
            Self::PreVarReduce => "pre_var_reduce",
            Self::ScalarLvaluePreVarReduce => "scalar_lvalue_pre_var_reduce",
            Self::ScalarLvaluePostVarReduce => "scalar_lvalue_post_var_reduce",
            Self::ScalarFieldPreVarReduce => "scalar_field_pre_var_reduce",
            Self::ScalarFieldPostVarReduce => "scalar_field_post_var_reduce",
            Self::ScalarExtensionPerUsePreVarReduce => {
                "scalar_extension_per_use_pre_var_reduce"
            }
            Self::ScalarExtensionPerUsePostVarReduce => {
                "scalar_extension_per_use_post_var_reduce"
            }
            Self::ScalarExtensionHoistedPreVarReduce => {
                "scalar_extension_hoisted_pre_var_reduce"
            }
            Self::ScalarExtensionHoistedPostVarReduce => {
                "scalar_extension_hoisted_post_var_reduce"
            }
            Self::Stage4AffinePreVarReduce => "stage4_affine_pre_var_reduce",
            Self::Stage4AffinePostVarReduce => "stage4_affine_post_var_reduce",
            Self::Stage4ZeroPreVarReduce => "stage4_zero_pre_var_reduce",
            Self::Stage4ZeroPostVarReduce => "stage4_zero_post_var_reduce",
            Self::Stage4RmwPreVarReduce => "stage4_rmw_pre_var_reduce",
            Self::Stage4RmwPostVarReduce => "stage4_rmw_post_var_reduce",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceAlternativeSnapshot {
    /// Private declaration coordinate retained across the destructive passes.
    /// This is never serialized: the wire ordinal is derived later from the
    /// canonical printer's emitted `FuncDef` order.
    pub declaration_index: usize,
    pub manifold_name: String,
    pub manifold_address: u64,
    pub boundary: SourceAlternativeBoundary,
    pub kinds: Vec<&'static str>,
    /// Present only for v4 Stage-4 rows.  The root is provider provenance; a
    /// deterministic per-function ordinal is derived during rendering.
    pub stage4_profile: Option<Stage4SourceProfile>,
    pub function: FuncDef,
}

/// Provider-owned typed-lvalue function assembled through the same Clight/C
/// path as the canonical definition.  It is kept private until the two
/// destructive post-selection boundaries have run; only then can the exact
/// pre/post forms be admitted to the current family-atomic wire manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingScalarLvalueAlternative {
    pub manifold_name: String,
    pub manifold_address: u64,
    pub family: ScalarSourceAlternativeFamily,
    pub function: FuncDef,
}

/// One private Stage-4 root assembled through the canonical Clight-to-C path.
/// Profiles are independent: no pending function contains two Stage-4 roots,
/// and VarReduce records this profile's bounded pre/post subsequence only.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingStage4Alternative {
    pub manifold_name: String,
    pub manifold_address: u64,
    pub profile: Stage4SourceProfile,
    pub function: FuncDef,
}

/// Retain a globally bounded snapshot set.  The first overflow clears all
/// snapshots and permanently disables capture for this DB, so a later pass can
/// never repopulate a partial sidecar.
pub fn record_bounded_snapshot(
    snapshots: &mut Vec<SourceAlternativeSnapshot>,
    overflowed: &mut bool,
    snapshot: SourceAlternativeSnapshot,
) {
    if *overflowed {
        return;
    }
    if snapshots.len() >= MAX_TOTAL_SOURCE_ALTERNATIVES {
        snapshots.clear();
        *overflowed = true;
        return;
    }
    snapshots.push(snapshot);
}

/// Bind final emitted provider names to exactly one selected function address.
/// Missing or duplicate names are deliberately absent from the returned map.
pub fn exact_function_addresses(
    selected_functions: &[SelectedFunction],
    emitted_names: &HashMap<usize, String>,
) -> HashMap<String, u64> {
    let mut addresses = HashMap::new();
    let mut duplicate_names = HashSet::new();
    for function in selected_functions {
        let Some(name) = emitted_names.get(&(function.address as usize)) else {
            continue;
        };
        if addresses.insert(name.clone(), function.address).is_some() {
            duplicate_names.insert(name.clone());
        }
    }
    for name in duplicate_names {
        addresses.remove(&name);
    }
    addresses
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct ControlShapeProfile {
    conditionals: usize,
    conditional_return_arms: usize,
    return_calls: usize,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
struct CallMaterializationProfile {
    calls: usize,
    assignment_calls: usize,
    initializer_calls: usize,
    return_calls: usize,
    condition_calls: usize,
    expression_statement_calls: usize,
}

fn profile_call_expr(expr: &CExpr, profile: &mut CallMaterializationProfile) {
    match expr {
        CExpr::Call(callee, args) => {
            profile.calls += 1;
            profile_call_expr(callee, profile);
            for arg in args {
                profile_call_expr(arg, profile);
            }
        }
        CExpr::Unary(_, value)
        | CExpr::Cast(_, value)
        | CExpr::SizeofExpr(value)
        | CExpr::Paren(value) => profile_call_expr(value, profile),
        CExpr::Binary(_, lhs, rhs) | CExpr::Index(lhs, rhs) => {
            profile_call_expr(lhs, profile);
            profile_call_expr(rhs, profile);
        }
        CExpr::Assign(_, lhs, rhs) => {
            let calls_before = profile.calls;
            profile_call_expr(rhs, profile);
            profile.assignment_calls += profile.calls - calls_before;
            profile_call_expr(lhs, profile);
        }
        CExpr::Ternary(condition, then_expr, else_expr) => {
            profile_call_expr(condition, profile);
            profile_call_expr(then_expr, profile);
            profile_call_expr(else_expr, profile);
        }
        CExpr::Member(base, _) | CExpr::MemberPtr(base, _) => profile_call_expr(base, profile),
        CExpr::CompoundLit(_, initializers) => {
            for initializer in initializers {
                profile_call_initializer(initializer, profile);
            }
        }
        CExpr::Generic(control, associations) => {
            profile_call_expr(control, profile);
            for (_, expr) in associations {
                profile_call_expr(expr, profile);
            }
        }
        CExpr::StmtExpr(statements, result) => {
            for statement in statements {
                profile_call_stmt(statement, profile);
            }
            profile_call_expr(result, profile);
        }
        CExpr::Var(_)
        | CExpr::IntLit(_)
        | CExpr::FloatLit(_)
        | CExpr::CharLit(_)
        | CExpr::StringLit(_)
        | CExpr::SizeofType(_)
        | CExpr::AlignofType(_) => {}
    }
}

fn profile_call_initializer(
    initializer: &crate::decompile::passes::c_pass::types::Initializer,
    profile: &mut CallMaterializationProfile,
) {
    use crate::decompile::passes::c_pass::types::Initializer;
    match initializer {
        Initializer::Expr(expr) => {
            let calls_before = profile.calls;
            profile_call_expr(expr, profile);
            profile.initializer_calls += profile.calls - calls_before;
        }
        Initializer::List(items) => {
            for item in items {
                profile_call_initializer(&item.init, profile);
            }
        }
        Initializer::String(_) => {}
    }
}

fn profile_call_stmt(stmt: &CStmt, profile: &mut CallMaterializationProfile) {
    match stmt {
        CStmt::Expr(expr) => {
            let calls_before = profile.calls;
            profile_call_expr(expr, profile);
            profile.expression_statement_calls += profile.calls - calls_before;
        }
        CStmt::Block(items) => {
            for item in items {
                match item {
                    CBlockItem::Stmt(stmt) => profile_call_stmt(stmt, profile),
                    CBlockItem::Decl(decls) => {
                        for decl in decls {
                            if let Some(initializer) = &decl.init {
                                profile_call_initializer(initializer, profile);
                            }
                        }
                    }
                }
            }
        }
        CStmt::If(condition, then_stmt, else_stmt) => {
            let calls_before = profile.calls;
            profile_call_expr(condition, profile);
            profile.condition_calls += profile.calls - calls_before;
            profile_call_stmt(then_stmt, profile);
            if let Some(else_stmt) = else_stmt {
                profile_call_stmt(else_stmt, profile);
            }
        }
        CStmt::Switch(expr, body) | CStmt::While(expr, body) => {
            let calls_before = profile.calls;
            profile_call_expr(expr, profile);
            profile.condition_calls += profile.calls - calls_before;
            profile_call_stmt(body, profile);
        }
        CStmt::DoWhile(body, expr) => {
            profile_call_stmt(body, profile);
            let calls_before = profile.calls;
            profile_call_expr(expr, profile);
            profile.condition_calls += profile.calls - calls_before;
        }
        CStmt::For(init, condition, step, body) => {
            if let Some(init) = init {
                match init {
                    crate::decompile::passes::c_pass::types::ForInit::Expr(expr) => {
                        profile_call_expr(expr, profile)
                    }
                    crate::decompile::passes::c_pass::types::ForInit::Decl(decls) => {
                        for decl in decls {
                            if let Some(initializer) = &decl.init {
                                profile_call_initializer(initializer, profile);
                            }
                        }
                    }
                }
            }
            if let Some(condition) = condition {
                let calls_before = profile.calls;
                profile_call_expr(condition, profile);
                profile.condition_calls += profile.calls - calls_before;
            }
            if let Some(step) = step {
                profile_call_expr(step, profile);
            }
            profile_call_stmt(body, profile);
        }
        CStmt::Return(Some(expr)) => {
            let calls_before = profile.calls;
            profile_call_expr(expr, profile);
            profile.return_calls += profile.calls - calls_before;
        }
        CStmt::Labeled(_, stmt) => profile_call_stmt(stmt, profile),
        CStmt::Decl(decls) => {
            for decl in decls {
                if let Some(initializer) = &decl.init {
                    profile_call_initializer(initializer, profile);
                }
            }
        }
        CStmt::Sequence(statements) => {
            for statement in statements {
                profile_call_stmt(statement, profile);
            }
        }
        CStmt::Empty | CStmt::Goto(_) | CStmt::Continue | CStmt::Break | CStmt::Return(None) => {}
    }
}

fn call_materialization_profile(function: &FuncDef) -> CallMaterializationProfile {
    let mut profile = CallMaterializationProfile::default();
    for local in &function.local_vars {
        if let Some(initializer) = &local.init {
            profile_call_initializer(initializer, &mut profile);
        }
    }
    profile_call_stmt(&function.body, &mut profile);
    profile
}

fn stmt_ends_in_return(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Return(_) => true,
        CStmt::Block(items) => items.last().is_some_and(|item| match item {
            CBlockItem::Stmt(stmt) => stmt_ends_in_return(stmt),
            CBlockItem::Decl(_) => false,
        }),
        CStmt::Sequence(statements) => statements.last().is_some_and(stmt_ends_in_return),
        CStmt::If(_, then_stmt, Some(else_stmt)) => {
            stmt_ends_in_return(then_stmt) && stmt_ends_in_return(else_stmt)
        }
        CStmt::Labeled(_, stmt) => stmt_ends_in_return(stmt),
        _ => false,
    }
}

fn profile_control_shapes(stmt: &CStmt, profile: &mut ControlShapeProfile) {
    match stmt {
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(stmt) = item {
                    profile_control_shapes(stmt, profile);
                }
            }
        }
        CStmt::If(_, then_stmt, else_stmt) => {
            profile.conditionals += 1;
            profile.conditional_return_arms += usize::from(stmt_ends_in_return(then_stmt));
            if let Some(else_stmt) = else_stmt {
                profile.conditional_return_arms += usize::from(stmt_ends_in_return(else_stmt));
            }
            profile_control_shapes(then_stmt, profile);
            if let Some(else_stmt) = else_stmt {
                profile_control_shapes(else_stmt, profile);
            }
        }
        CStmt::Switch(_, body)
        | CStmt::While(_, body)
        | CStmt::DoWhile(body, _)
        | CStmt::For(_, _, _, body)
        | CStmt::Labeled(_, body) => profile_control_shapes(body, profile),
        CStmt::Return(Some(expr)) => {
            if matches!(expr, CExpr::Call(_, _)) {
                profile.return_calls += 1;
            }
        }
        CStmt::Sequence(statements) => {
            for statement in statements {
                profile_control_shapes(statement, profile);
            }
        }
        CStmt::Empty
        | CStmt::Expr(_)
        | CStmt::Goto(_)
        | CStmt::Continue
        | CStmt::Break
        | CStmt::Return(None)
        | CStmt::Decl(_) => {}
    }
}

fn control_shape_profile(function: &FuncDef) -> ControlShapeProfile {
    let mut profile = ControlShapeProfile::default();
    profile_control_shapes(&function.body, &mut profile);
    profile
}

/// Retain one exact pre-pass function only when a post-selection pass changed it.
pub fn snapshot_if_changed(
    declaration_index: usize,
    manifold_address: u64,
    boundary: SourceAlternativeBoundary,
    before: &FuncDef,
    after: &FuncDef,
) -> Option<SourceAlternativeSnapshot> {
    if before == after || before.name != after.name {
        return None;
    }
    let before_profile = control_shape_profile(before);
    let after_profile = control_shape_profile(after);
    let mut kinds = match boundary {
        SourceAlternativeBoundary::PreForLoop => vec!["control_layout"],
        SourceAlternativeBoundary::PreVarReduce => vec!["local_lifetime"],
        SourceAlternativeBoundary::ScalarLvaluePreVarReduce
        | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
        | SourceAlternativeBoundary::ScalarFieldPreVarReduce
        | SourceAlternativeBoundary::ScalarFieldPostVarReduce
        | SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce
        | SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce
        | SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce
        | SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce
        | SourceAlternativeBoundary::Stage4AffinePreVarReduce
        | SourceAlternativeBoundary::Stage4AffinePostVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPreVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
        | SourceAlternativeBoundary::Stage4RmwPreVarReduce
        | SourceAlternativeBoundary::Stage4RmwPostVarReduce => return None,
    };
    match boundary {
        SourceAlternativeBoundary::PreForLoop => {
            if before_profile.conditionals != after_profile.conditionals {
                kinds.push("conditional");
            }
            if before_profile.conditional_return_arms != after_profile.conditional_return_arms {
                kinds.push("early_return");
            }
            if before_profile.return_calls != after_profile.return_calls {
                kinds.push("tail_call");
            }
        }
        SourceAlternativeBoundary::PreVarReduce => {
            let before_calls = call_materialization_profile(before);
            let after_calls = call_materialization_profile(after);
            if before_calls != after_calls && (before_calls.calls > 0 || after_calls.calls > 0) {
                kinds.push("call_result");
            }
        }
        SourceAlternativeBoundary::ScalarLvaluePreVarReduce
        | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
        | SourceAlternativeBoundary::ScalarFieldPreVarReduce
        | SourceAlternativeBoundary::ScalarFieldPostVarReduce
        | SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce
        | SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce
        | SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce
        | SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce
        | SourceAlternativeBoundary::Stage4AffinePreVarReduce
        | SourceAlternativeBoundary::Stage4AffinePostVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPreVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
        | SourceAlternativeBoundary::Stage4RmwPreVarReduce
        | SourceAlternativeBoundary::Stage4RmwPostVarReduce => unreachable!(),
    }
    kinds.sort_unstable();
    kinds.dedup();
    Some(SourceAlternativeSnapshot {
        declaration_index,
        manifold_name: before.name.clone(),
        manifold_address,
        boundary,
        kinds,
        stage4_profile: None,
        function: before.clone(),
    })
}

fn scalar_feature_boundary_data(
    family: ScalarSourceAlternativeFamily,
) -> [
    (
        SourceAlternativeBoundary,
        Vec<&'static str>,
    );
    2
] {
    match family {
        ScalarSourceAlternativeFamily::TypedLvalue => [
            (
                SourceAlternativeBoundary::ScalarLvaluePreVarReduce,
                vec!["local_lifetime", "typed_lvalue"],
            ),
            (
                SourceAlternativeBoundary::ScalarLvaluePostVarReduce,
                vec!["typed_lvalue"],
            ),
        ],
        ScalarSourceAlternativeFamily::FieldLvalue => [
            (
                SourceAlternativeBoundary::ScalarFieldPreVarReduce,
                vec!["local_lifetime", "typed_lvalue", "field_lvalue"],
            ),
            (
                SourceAlternativeBoundary::ScalarFieldPostVarReduce,
                vec!["typed_lvalue", "field_lvalue"],
            ),
        ],
        ScalarSourceAlternativeFamily::ExtensionPerUse => [
            (
                SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce,
                vec!["local_lifetime", "typed_lvalue", "per_use_extension"],
            ),
            (
                SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce,
                vec!["typed_lvalue", "per_use_extension"],
            ),
        ],
        ScalarSourceAlternativeFamily::ExtensionHoisted => [
            (
                SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce,
                vec!["local_lifetime", "typed_lvalue", "hoisted_extension"],
            ),
            (
                SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce,
                vec!["typed_lvalue", "hoisted_extension"],
            ),
        ],
    }
}

/// Construct the useful printed forms of one v3 family in fixed `[pre, post]`
/// order. Forms equal to canonical source are discarded and equal alternative
/// sources collapse to their first boundary. This deliberately permits a
/// stable one-record profile while keeping identity and signature checks closed.
pub fn scalar_feature_snapshots(
    declaration_index: usize,
    manifold_address: u64,
    family: ScalarSourceAlternativeFamily,
    canonical: &FuncDef,
    pre_var_reduce: &FuncDef,
    post_var_reduce: &FuncDef,
) -> Option<Vec<SourceAlternativeSnapshot>> {
    scalar_feature_snapshots_for_format(
        declaration_index,
        manifold_address,
        family,
        canonical,
        pre_var_reduce,
        post_var_reduce,
        BinaryFormat::Coff,
    )
}

pub fn scalar_feature_snapshots_for_format(
    declaration_index: usize,
    manifold_address: u64,
    family: ScalarSourceAlternativeFamily,
    canonical: &FuncDef,
    pre_var_reduce: &FuncDef,
    post_var_reduce: &FuncDef,
    format: BinaryFormat,
) -> Option<Vec<SourceAlternativeSnapshot>> {
    if !same_signature(canonical, pre_var_reduce)
        || !same_signature(canonical, post_var_reduce)
    {
        return None;
    }
    let canonical_source = one_function_source(canonical, format);
    let mut seen_sources = HashSet::new();
    let mut snapshots = Vec::new();
    for ((boundary, kinds), function) in scalar_feature_boundary_data(family)
        .into_iter()
        .zip([pre_var_reduce, post_var_reduce])
    {
        let source = one_function_source(function, format);
        if source == canonical_source || !seen_sources.insert(source) {
            continue;
        }
        snapshots.push(SourceAlternativeSnapshot {
            declaration_index,
            manifold_name: canonical.name.clone(),
            manifold_address,
            boundary,
            kinds,
            stage4_profile: None,
            function: function.clone(),
        });
    }
    (!snapshots.is_empty()).then_some(snapshots)
}

fn stage4_boundary_data(
    kind: crate::x86::types::Stage4SourceKind,
) -> [(SourceAlternativeBoundary, Vec<&'static str>); 2] {
    use crate::x86::types::Stage4SourceKind;
    match kind {
        Stage4SourceKind::AffineAddress => [
            (
                SourceAlternativeBoundary::Stage4AffinePreVarReduce,
                vec!["address_expression", "local_lifetime"],
            ),
            (
                SourceAlternativeBoundary::Stage4AffinePostVarReduce,
                vec!["address_expression"],
            ),
        ],
        Stage4SourceKind::Zeroing => [
            (
                SourceAlternativeBoundary::Stage4ZeroPreVarReduce,
                vec!["local_lifetime", "zeroing"],
            ),
            (
                SourceAlternativeBoundary::Stage4ZeroPostVarReduce,
                vec!["zeroing"],
            ),
        ],
        Stage4SourceKind::Add
        | Stage4SourceKind::Sub
        | Stage4SourceKind::Mul
        | Stage4SourceKind::And
        | Stage4SourceKind::Or
        | Stage4SourceKind::Xor => [
            (
                SourceAlternativeBoundary::Stage4RmwPreVarReduce,
                vec!["compound_assignment", "local_lifetime"],
            ),
            (
                SourceAlternativeBoundary::Stage4RmwPostVarReduce,
                vec!["compound_assignment"],
            ),
        ],
    }
}

pub fn stage4_feature_snapshots_for_format(
    declaration_index: usize,
    manifold_address: u64,
    profile: Stage4SourceProfile,
    canonical: &FuncDef,
    pre_var_reduce: &FuncDef,
    post_var_reduce: &FuncDef,
    format: BinaryFormat,
) -> Option<Vec<SourceAlternativeSnapshot>> {
    if !same_signature(canonical, pre_var_reduce)
        || !same_signature(canonical, post_var_reduce)
    {
        return None;
    }
    let canonical_source = one_function_source(canonical, format);
    let mut seen_sources = HashSet::new();
    let mut snapshots = Vec::new();
    for ((boundary, kinds), function) in stage4_boundary_data(profile.kind)
        .into_iter()
        .zip([pre_var_reduce, post_var_reduce])
    {
        let source = one_function_source(function, format);
        if source == canonical_source || !seen_sources.insert(source) {
            continue;
        }
        snapshots.push(SourceAlternativeSnapshot {
            declaration_index,
            manifold_name: canonical.name.clone(),
            manifold_address,
            boundary,
            kinds,
            stage4_profile: Some(profile),
            function: function.clone(),
        });
    }
    (!snapshots.is_empty()).then_some(snapshots)
}

/// Backward-compatible internal constructor used by the v2 regression
/// fixtures. The wire schema is v3, but the Stage-2 family bytes and kinds are
/// intentionally unchanged.
pub fn scalar_lvalue_snapshot_pair(
    declaration_index: usize,
    manifold_address: u64,
    canonical: &FuncDef,
    pre_var_reduce: &FuncDef,
    post_var_reduce: &FuncDef,
) -> Option<[SourceAlternativeSnapshot; 2]> {
    if pre_var_reduce == post_var_reduce
        || pre_var_reduce == canonical
        || post_var_reduce == canonical
        || !same_signature(canonical, pre_var_reduce)
        || !same_signature(canonical, post_var_reduce)
    {
        return None;
    }
    let [(pre_boundary, pre_kinds), (post_boundary, post_kinds)] =
        scalar_feature_boundary_data(ScalarSourceAlternativeFamily::TypedLvalue);
    Some([
        SourceAlternativeSnapshot {
            declaration_index,
            manifold_name: canonical.name.clone(),
            manifold_address,
            boundary: pre_boundary,
            kinds: pre_kinds,
            stage4_profile: None,
            function: pre_var_reduce.clone(),
        },
        SourceAlternativeSnapshot {
            declaration_index,
            manifold_name: canonical.name.clone(),
            manifold_address,
            boundary: post_boundary,
            kinds: post_kinds,
            stage4_profile: None,
            function: post_var_reduce.clone(),
        },
    ])
}

#[derive(Serialize)]
struct SourceAlternativeManifest {
    schema: &'static str,
    max_per_function: usize,
    max_total: usize,
    max_source_bytes: usize,
    max_manifest_bytes: usize,
    truncated: bool,
    canonical_translation_unit_sha256: String,
    ordered_set_sha256: String,
    alternatives: Vec<SourceAlternativeRecord>,
}

#[derive(Clone, Serialize)]
struct SourceAlternativeRecord {
    id: String,
    function_ordinal: usize,
    manifold_name: String,
    manifold_address: String,
    boundary: &'static str,
    kinds: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_ordinal: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_root_node: Option<String>,
    canonical_source_sha256: String,
    alternative_source_sha256: String,
    canonical_source: String,
    alternative_source: String,
}

fn one_function_source(function: &FuncDef, format: BinaryFormat) -> String {
    use crate::decompile::passes::c_pass::print::{IntegerModel, PrintConfig, Printer};

    let mut config = PrintConfig::default();
    if matches!(format, BinaryFormat::Pe | BinaryFormat::Coff) {
        config.integer_model = IntegerModel::MsvcLlp64;
    }
    let mut printer = Printer::new(config);
    printer.print_func_def(function);
    printer.into_string()
}

fn same_signature(left: &FuncDef, right: &FuncDef) -> bool {
    left.name == right.name
        && left.return_type == right.return_type
        && left.params == right.params
        && left.is_variadic == right.is_variadic
        && left.storage_class == right.storage_class
}

fn feature_boundary_family(
    boundary: SourceAlternativeBoundary,
) -> Option<(ScalarSourceAlternativeFamily, bool)> {
    match boundary {
        SourceAlternativeBoundary::ScalarLvaluePreVarReduce => {
            Some((ScalarSourceAlternativeFamily::TypedLvalue, true))
        }
        SourceAlternativeBoundary::ScalarLvaluePostVarReduce => {
            Some((ScalarSourceAlternativeFamily::TypedLvalue, false))
        }
        SourceAlternativeBoundary::ScalarFieldPreVarReduce => {
            Some((ScalarSourceAlternativeFamily::FieldLvalue, true))
        }
        SourceAlternativeBoundary::ScalarFieldPostVarReduce => {
            Some((ScalarSourceAlternativeFamily::FieldLvalue, false))
        }
        SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce => {
            Some((ScalarSourceAlternativeFamily::ExtensionPerUse, true))
        }
        SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce => {
            Some((ScalarSourceAlternativeFamily::ExtensionPerUse, false))
        }
        SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce => {
            Some((ScalarSourceAlternativeFamily::ExtensionHoisted, true))
        }
        SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce => {
            Some((ScalarSourceAlternativeFamily::ExtensionHoisted, false))
        }
        SourceAlternativeBoundary::PreForLoop
        | SourceAlternativeBoundary::PreVarReduce
        | SourceAlternativeBoundary::Stage4AffinePreVarReduce
        | SourceAlternativeBoundary::Stage4AffinePostVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPreVarReduce
        | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
        | SourceAlternativeBoundary::Stage4RmwPreVarReduce
        | SourceAlternativeBoundary::Stage4RmwPostVarReduce => None,
    }
}

fn stage4_boundary_data_for_snapshot(
    snapshot: &SourceAlternativeSnapshot,
) -> Option<(bool, Vec<&'static str>)> {
    let profile = snapshot.stage4_profile?;
    stage4_boundary_data(profile.kind)
        .into_iter()
        .find(|(boundary, _)| *boundary == snapshot.boundary)
        .map(|(boundary, kinds)| {
            let pre = matches!(
                boundary,
                SourceAlternativeBoundary::Stage4AffinePreVarReduce
                    | SourceAlternativeBoundary::Stage4ZeroPreVarReduce
                    | SourceAlternativeBoundary::Stage4RmwPreVarReduce
            );
            (pre, kinds)
        })
}

fn is_stage4_boundary(boundary: SourceAlternativeBoundary) -> bool {
    matches!(
        boundary,
        SourceAlternativeBoundary::Stage4AffinePreVarReduce
            | SourceAlternativeBoundary::Stage4AffinePostVarReduce
            | SourceAlternativeBoundary::Stage4ZeroPreVarReduce
            | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
            | SourceAlternativeBoundary::Stage4RmwPreVarReduce
            | SourceAlternativeBoundary::Stage4RmwPostVarReduce
    )
}

fn is_feature_boundary(boundary: SourceAlternativeBoundary) -> bool {
    feature_boundary_family(boundary).is_some() || is_stage4_boundary(boundary)
}

fn append_digest_field(material: &mut Vec<u8>, field: &[u8]) {
    material.extend_from_slice(&(field.len() as u64).to_be_bytes());
    material.extend_from_slice(field);
}

/// Hash the complete ordered identity/source-digest set with unambiguous
/// length-prefix framing.  The source text itself is separately authenticated by
/// each record's digest and by the provider artifact bundle.
fn ordered_set_sha256(
    schema: &str,
    canonical_translation_unit_sha256: &str,
    records: &[SourceAlternativeRecord],
) -> String {
    let mut material = Vec::new();
    append_digest_field(&mut material, schema.as_bytes());
    append_digest_field(&mut material, canonical_translation_unit_sha256.as_bytes());
    material.extend_from_slice(&(records.len() as u64).to_be_bytes());
    for record in records {
        append_digest_field(&mut material, record.id.as_bytes());
        material.extend_from_slice(&(record.function_ordinal as u64).to_be_bytes());
        append_digest_field(&mut material, record.manifold_name.as_bytes());
        append_digest_field(&mut material, record.manifold_address.as_bytes());
        append_digest_field(&mut material, record.boundary.as_bytes());
        material.extend_from_slice(&(record.kinds.len() as u64).to_be_bytes());
        for kind in &record.kinds {
            append_digest_field(&mut material, kind.as_bytes());
        }
        if schema == SOURCE_ALTERNATIVES_SCHEMA_V4 {
            material.extend_from_slice(
                &record
                    .profile_ordinal
                    .map(|ordinal| ordinal as u64)
                    .unwrap_or(u64::MAX)
                    .to_be_bytes(),
            );
            append_digest_field(
                &mut material,
                record.profile_root_node.as_deref().unwrap_or("").as_bytes(),
            );
        }
        append_digest_field(&mut material, record.canonical_source_sha256.as_bytes());
        append_digest_field(&mut material, record.alternative_source_sha256.as_bytes());
    }
    crate::decompile::disassembly::coff::sha256_hex(&material)
}

/// Render a deterministic sidecar.  Invalid or stale snapshots are omitted rather
/// than guessed: the private declaration coordinate and provider identity must
/// still select the same exact function in the final canonical translation unit.
/// The serialized ordinal is independently derived from the canonical printer's
/// emitted, nonempty `FuncDef` order; other declarations never affect it.
pub fn render_manifest(
    final_tu: &TranslationUnit,
    snapshots: &[SourceAlternativeSnapshot],
    capture_overflowed: bool,
    exclusions: &SourceAlternativeExclusions,
    exact_function_identities: &[(String, u64)],
    canonical_translation_unit_source: &str,
    format: BinaryFormat,
) -> serde_json::Result<Option<String>> {
    render_manifest_with_limits(
        final_tu,
        snapshots,
        capture_overflowed,
        exclusions,
        exact_function_identities,
        canonical_translation_unit_source,
        format,
        MAX_TOTAL_SOURCE_ALTERNATIVES,
        MAX_SOURCE_ALTERNATIVE_BYTES,
        MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
    )
}

fn render_manifest_with_limits(
    final_tu: &TranslationUnit,
    snapshots: &[SourceAlternativeSnapshot],
    capture_overflowed: bool,
    exclusions: &SourceAlternativeExclusions,
    exact_function_identities: &[(String, u64)],
    canonical_translation_unit_source: &str,
    format: BinaryFormat,
    max_total: usize,
    max_source_bytes: usize,
    max_manifest_bytes: usize,
) -> serde_json::Result<Option<String>> {
    render_manifest_with_limits_inner(
        final_tu,
        snapshots,
        capture_overflowed,
        exclusions,
        exact_function_identities,
        canonical_translation_unit_source,
        format,
        max_total,
        max_source_bytes,
        max_manifest_bytes,
        true,
    )
}

fn render_manifest_with_limits_inner(
    final_tu: &TranslationUnit,
    snapshots: &[SourceAlternativeSnapshot],
    capture_overflowed: bool,
    exclusions: &SourceAlternativeExclusions,
    exact_function_identities: &[(String, u64)],
    canonical_translation_unit_source: &str,
    format: BinaryFormat,
    max_total: usize,
    max_source_bytes: usize,
    max_manifest_bytes: usize,
    allow_feature_fallback: bool,
) -> serde_json::Result<Option<String>> {
    if capture_overflowed || exclusions.suppress_all {
        return Ok(None);
    }
    let has_stage4 = snapshots
        .iter()
        .any(|snapshot| is_stage4_boundary(snapshot.boundary));
    let schema = if has_stage4 {
        SOURCE_ALTERNATIVES_SCHEMA_V4
    } else {
        SOURCE_ALTERNATIVES_SCHEMA
    };
    let max_per_function = if has_stage4 {
        MAX_SOURCE_ALTERNATIVES_PER_FUNCTION_V4
    } else {
        MAX_SOURCE_ALTERNATIVES_PER_FUNCTION
    };
    let reproduced_source =
        crate::decompile::passes::c_pass::print_translation_unit_for_format(final_tu, format);
    if reproduced_source != canonical_translation_unit_source {
        return Err(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "canonical optimized translation unit changed before alternative emission",
        )));
    }
    let fallback_to_ordinary = || {
        if has_stage4 {
            let prior: Vec<_> = snapshots
                .iter()
                .filter(|snapshot| !is_stage4_boundary(snapshot.boundary))
                .cloned()
                .collect();
            return render_manifest_with_limits_inner(
                final_tu,
                &prior,
                capture_overflowed,
                exclusions,
                exact_function_identities,
                canonical_translation_unit_source,
                format,
                max_total,
                max_source_bytes,
                max_manifest_bytes,
                allow_feature_fallback,
            );
        }
        if !allow_feature_fallback
            || !snapshots
                .iter()
                .any(|snapshot| is_feature_boundary(snapshot.boundary))
        {
            return Ok(None);
        }
        let ordinary: Vec<_> = snapshots
            .iter()
            .filter(|snapshot| !is_feature_boundary(snapshot.boundary))
            .cloned()
            .collect();
        render_manifest_with_limits_inner(
            final_tu,
            &ordinary,
            capture_overflowed,
            exclusions,
            exact_function_identities,
            canonical_translation_unit_source,
            format,
            max_total,
            max_source_bytes,
            max_manifest_bytes,
            false,
        )
    };
    // The translation-unit printer category-orders declarations while retaining
    // relative order within each category.  FuncDefs therefore appear in their
    // declaration-relative order, minus definitions the printer omits as empty.
    let emitted_functions: Vec<(usize, &FuncDef)> = final_tu
        .decls
        .iter()
        .enumerate()
        .filter_map(|(declaration_index, declaration)| match declaration {
            TopLevelDecl::FuncDef(function)
                if crate::decompile::passes::c_pass::print::is_emitted_func_def(function) =>
            {
                Some((declaration_index, function))
            }
            _ => None,
        })
        .collect();
    let wire_ordinal_by_declaration: HashMap<usize, usize> = emitted_functions
        .iter()
        .enumerate()
        .map(|(function_ordinal, (declaration_index, _))| (*declaration_index, function_ordinal))
        .collect();
    let snapshot_source_if_valid = |snapshot: &SourceAlternativeSnapshot| {
        if exclusions.excludes(&snapshot.manifold_name, snapshot.manifold_address)
            || !wire_ordinal_by_declaration.contains_key(&snapshot.declaration_index)
        {
            return None;
        }
        let canonical = final_tu
            .decls
            .get(snapshot.declaration_index)
            .and_then(|declaration| match declaration {
                TopLevelDecl::FuncDef(function) => Some(function),
                _ => None,
            })?;
        if snapshot.manifold_name != canonical.name
            || !same_signature(canonical, &snapshot.function)
            || exact_function_identities
                .iter()
                .filter(|(name, address)| {
                    name == &snapshot.manifold_name && *address == snapshot.manifold_address
                })
                .count()
                != 1
        {
            return None;
        }
        let canonical_source = one_function_source(canonical, format);
        let alternative_source = one_function_source(&snapshot.function, format);
        (canonical_source != alternative_source).then_some(alternative_source)
    };

    // The project adapter rejects duplicate printed alternatives across every
    // family of one function. Ordinary Stage-1 rows have permanent priority:
    // compute their exact emitted source set first, then admit later feature
    // families atomically only when their whole source set is disjoint. The
    // normal Stage-1 pipeline cannot produce equal pre-forloop and
    // pre-var-reduce sources (the first exists only when ForLoop changed it),
    // but a forged duplicate must still suppress the malformed sidecar rather
    // than silently dropping either historical row.
    let mut ordinary_key_counts = HashMap::new();
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| !is_feature_boundary(snapshot.boundary))
    {
        *ordinary_key_counts
            .entry((snapshot.declaration_index, snapshot.boundary))
            .or_insert(0usize) += 1;
    }
    let mut admitted_sources: HashMap<usize, HashSet<String>> = HashMap::new();
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| !is_feature_boundary(snapshot.boundary))
    {
        if ordinary_key_counts
            .get(&(snapshot.declaration_index, snapshot.boundary))
            .copied()
            != Some(1)
        {
            continue;
        }
        let Some(source) = snapshot_source_if_valid(snapshot) else {
            continue;
        };
        if !admitted_sources
            .entry(snapshot.declaration_index)
            .or_default()
            .insert(source)
        {
            return Ok(None);
        }
    }

    let mut feature_groups: HashMap<
        (usize, ScalarSourceAlternativeFamily),
        Vec<&SourceAlternativeSnapshot>,
    > = HashMap::new();
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| feature_boundary_family(snapshot.boundary).is_some())
    {
        let (family, _) = feature_boundary_family(snapshot.boundary)
            .expect("feature boundary filtered above");
        feature_groups
            .entry((snapshot.declaration_index, family))
            .or_default()
            .push(snapshot);
    }
    let group_sources_if_valid = |family: ScalarSourceAlternativeFamily,
                                  group: &[&SourceAlternativeSnapshot]| {
            if group.is_empty() || group.len() > 2 {
                return None;
            }
            let mut boundaries = HashSet::new();
            if group.iter().any(|snapshot| {
                feature_boundary_family(snapshot.boundary)
                    .map_or(true, |(snapshot_family, _)| snapshot_family != family)
                    || scalar_feature_boundary_data(family)
                        .into_iter()
                        .find(|(boundary, _)| *boundary == snapshot.boundary)
                        .map_or(true, |(_, kinds)| kinds != snapshot.kinds)
                    || !boundaries.insert(snapshot.boundary)
            }) {
                return None;
            }
            let mut alternative_sources = HashSet::new();
            for snapshot in group {
                let source = snapshot_source_if_valid(snapshot)?;
                if !alternative_sources.insert(source) {
                    return None;
                }
            }
            Some(alternative_sources)
        };
    let mut valid_feature_groups = HashSet::new();
    let feature_declarations: std::collections::BTreeSet<usize> = feature_groups
        .keys()
        .map(|(declaration_index, _)| *declaration_index)
        .collect();
    for declaration_index in feature_declarations {
        for family in [
            ScalarSourceAlternativeFamily::TypedLvalue,
            ScalarSourceAlternativeFamily::FieldLvalue,
        ] {
            let key = (declaration_index, family);
            let Some(sources) = feature_groups
                .get(&key)
                .and_then(|group| group_sources_if_valid(family, group))
            else {
                continue;
            };
            let used = admitted_sources.entry(declaration_index).or_default();
            if sources.is_disjoint(used) {
                used.extend(sources);
                valid_feature_groups.insert(key);
            }
        }

        let per_use = (
            declaration_index,
            ScalarSourceAlternativeFamily::ExtensionPerUse,
        );
        let hoisted = (
            declaration_index,
            ScalarSourceAlternativeFamily::ExtensionHoisted,
        );
        let per_use_sources = feature_groups
            .get(&per_use)
            .and_then(|group| group_sources_if_valid(per_use.1, group));
        let hoisted_sources = feature_groups
            .get(&hoisted)
            .and_then(|group| group_sources_if_valid(hoisted.1, group));
        if let (Some(per_use_sources), Some(hoisted_sources)) =
            (per_use_sources, hoisted_sources)
        {
            let mut extension_sources = per_use_sources;
            let placements_are_disjoint = hoisted_sources
                .iter()
                .all(|source| extension_sources.insert(source.clone()));
            let used = admitted_sources.entry(declaration_index).or_default();
            if placements_are_disjoint && extension_sources.is_disjoint(used) {
                used.extend(extension_sources);
                valid_feature_groups.insert(per_use);
                valid_feature_groups.insert(hoisted);
            }
        }
    }

    let mut stage4_groups: HashMap<
        (usize, Stage4SourceProfile),
        Vec<&SourceAlternativeSnapshot>,
    > = HashMap::new();
    for snapshot in snapshots
        .iter()
        .filter(|snapshot| is_stage4_boundary(snapshot.boundary))
    {
        let Some(profile) = snapshot.stage4_profile else {
            continue;
        };
        stage4_groups
            .entry((snapshot.declaration_index, profile))
            .or_default()
            .push(snapshot);
    }
    let mut profiles_by_declaration: BTreeMap<usize, BTreeSet<Stage4SourceProfile>> =
        BTreeMap::new();
    for &(declaration_index, profile) in stage4_groups.keys() {
        profiles_by_declaration
            .entry(declaration_index)
            .or_default()
            .insert(profile);
    }
    if profiles_by_declaration
        .values()
        .any(|profiles| profiles.len() > crate::decompile::passes::clight_select::select::MAX_STAGE4_PROFILES_PER_FUNCTION)
        || stage4_groups.len()
            > crate::decompile::passes::clight_select::select::MAX_STAGE4_PROFILES_PER_TU
    {
        return fallback_to_ordinary();
    }
    let mut valid_stage4_groups = HashSet::new();
    let mut sorted_stage4_keys: Vec<_> = stage4_groups.keys().copied().collect();
    sorted_stage4_keys.sort_unstable();
    for key @ (declaration_index, profile) in sorted_stage4_keys {
        let group = &stage4_groups[&key];
        if group.is_empty() || group.len() > 2 {
            continue;
        }
        let mut boundaries = HashSet::new();
        let mut sources = HashSet::new();
        let valid = group.iter().all(|snapshot| {
            snapshot.stage4_profile == Some(profile)
                && stage4_boundary_data_for_snapshot(snapshot)
                    .is_some_and(|(_, kinds)| kinds == snapshot.kinds)
                && boundaries.insert(snapshot.boundary)
                && snapshot_source_if_valid(snapshot)
                    .is_some_and(|source| sources.insert(source))
        });
        if !valid {
            continue;
        }
        let used = admitted_sources.entry(declaration_index).or_default();
        if sources.is_disjoint(used) {
            used.extend(sources);
            valid_stage4_groups.insert(key);
        }
    }
    if has_stage4 && valid_stage4_groups.is_empty() {
        return fallback_to_ordinary();
    }
    // Derive contiguous wire ordinals only from admitted profiles. Invalid or
    // content-colliding siblings remain canonical and cannot perturb the
    // identity of an otherwise valid profile.
    let mut admitted_profiles_by_declaration: BTreeMap<usize, BTreeSet<Stage4SourceProfile>> =
        BTreeMap::new();
    for &(declaration_index, profile) in &valid_stage4_groups {
        admitted_profiles_by_declaration
            .entry(declaration_index)
            .or_default()
            .insert(profile);
    }
    let profile_ordinals: HashMap<(usize, Stage4SourceProfile), usize> =
        admitted_profiles_by_declaration
            .iter()
            .flat_map(|(declaration_index, profiles)| {
                profiles.iter().enumerate().map(move |(ordinal, profile)| {
                    ((*declaration_index, *profile), ordinal)
                })
            })
            .collect();
    let mut ordered: Vec<(usize, &SourceAlternativeSnapshot)> = snapshots
        .iter()
        .filter(|snapshot| {
            if let Some((family, _)) = feature_boundary_family(snapshot.boundary) {
                valid_feature_groups.contains(&(snapshot.declaration_index, family))
            } else if is_stage4_boundary(snapshot.boundary) {
                snapshot.stage4_profile.is_some_and(|profile| {
                    valid_stage4_groups.contains(&(snapshot.declaration_index, profile))
                })
            } else {
                true
            }
        })
        .filter(|snapshot| {
            !exclusions.excludes(&snapshot.manifold_name, snapshot.manifold_address)
        })
        .filter_map(|snapshot| {
            wire_ordinal_by_declaration
                .get(&snapshot.declaration_index)
                .copied()
                .map(|function_ordinal| (function_ordinal, snapshot))
        })
        .collect();
    ordered.sort_by_key(|(function_ordinal, snapshot)| {
        let stage4 = is_stage4_boundary(snapshot.boundary);
        let profile_ordinal = snapshot
            .stage4_profile
            .and_then(|profile| {
                profile_ordinals
                    .get(&(snapshot.declaration_index, profile))
                    .copied()
            })
            .unwrap_or(usize::MAX);
        (
            *function_ordinal,
            stage4,
            profile_ordinal,
            snapshot.boundary,
        )
    });
    let mut snapshot_key_counts = std::collections::HashMap::new();
    for (_, snapshot) in &ordered {
        *snapshot_key_counts
            .entry((
                snapshot.declaration_index,
                snapshot.boundary,
                snapshot.stage4_profile,
            ))
            .or_insert(0usize) += 1;
    }

    let mut alternatives = Vec::new();
    let mut last_ordinal = None;
    let mut count_for_function = 0usize;
    let mut source_bytes = 0usize;
    for (function_ordinal, snapshot) in &ordered {
        if last_ordinal != Some(*function_ordinal) {
            last_ordinal = Some(*function_ordinal);
            count_for_function = 0;
        }
        if count_for_function >= max_per_function {
            return fallback_to_ordinary();
        }
        if alternatives.len() >= max_total {
            return fallback_to_ordinary();
        }
        if snapshot_key_counts
            .get(&(
                snapshot.declaration_index,
                snapshot.boundary,
                snapshot.stage4_profile,
            ))
            .copied()
            != Some(1)
        {
            continue;
        }
        let Some(&(resolved_declaration_index, canonical)) =
            emitted_functions.get(*function_ordinal)
        else {
            continue;
        };
        if resolved_declaration_index != snapshot.declaration_index {
            continue;
        }
        if canonical.name != snapshot.manifold_name
            || canonical == &snapshot.function
            || !same_signature(canonical, &snapshot.function)
            || exact_function_identities
                .iter()
                .filter(|(name, address)| {
                    name == &snapshot.manifold_name && *address == snapshot.manifold_address
                })
                .count()
                != 1
        {
            continue;
        }
        let canonical_source = one_function_source(canonical, format);
        let alternative_source = one_function_source(&snapshot.function, format);
        if canonical_source == alternative_source {
            continue;
        }
        let candidate_bytes = canonical_source
            .len()
            .saturating_add(alternative_source.len());
        if candidate_bytes > max_source_bytes
            || source_bytes.saturating_add(candidate_bytes) > max_source_bytes
        {
            return fallback_to_ordinary();
        }
        source_bytes += candidate_bytes;
        let canonical_source_sha256 =
            crate::decompile::disassembly::coff::sha256_hex(canonical_source.as_bytes());
        let alternative_source_sha256 =
            crate::decompile::disassembly::coff::sha256_hex(alternative_source.as_bytes());
        let profile_ordinal = snapshot.stage4_profile.and_then(|profile| {
            profile_ordinals
                .get(&(snapshot.declaration_index, profile))
                .copied()
        });
        let id = if let Some(profile_ordinal) = profile_ordinal {
            format!(
                "function-{:06}:{}:profile-{:04}",
                function_ordinal,
                snapshot.boundary.wire_name(),
                profile_ordinal,
            )
        } else {
            format!(
                "function-{:06}:{}",
                function_ordinal,
                snapshot.boundary.wire_name(),
            )
        };
        alternatives.push(SourceAlternativeRecord {
            id,
            function_ordinal: *function_ordinal,
            manifold_name: snapshot.manifold_name.clone(),
            manifold_address: format!("0x{:x}", snapshot.manifold_address),
            boundary: snapshot.boundary.wire_name(),
            kinds: snapshot.kinds.clone(),
            profile_ordinal,
            profile_root_node: snapshot
                .stage4_profile
                .map(|profile| format!("0x{:x}", profile.root_node)),
            canonical_source_sha256,
            alternative_source_sha256,
            canonical_source,
            alternative_source,
        });
        count_for_function += 1;
    }

    let canonical_translation_unit_sha256 = crate::decompile::disassembly::coff::sha256_hex(
        canonical_translation_unit_source.as_bytes(),
    );
    if alternatives.is_empty() {
        return Ok(None);
    }
    let ordered_set_sha256 =
        ordered_set_sha256(schema, &canonical_translation_unit_sha256, &alternatives);
    let rendered = serde_json::to_string_pretty(&SourceAlternativeManifest {
        schema,
        max_per_function,
        max_total: MAX_TOTAL_SOURCE_ALTERNATIVES,
        max_source_bytes: MAX_SOURCE_ALTERNATIVE_BYTES,
        max_manifest_bytes: MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        truncated: false,
        canonical_translation_unit_sha256,
        ordered_set_sha256,
        alternatives,
    })?;
    if rendered.len() > max_manifest_bytes {
        return fallback_to_ordinary();
    }
    Ok(Some(rendered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::passes::c_pass::types::{
        BinaryOp, CType, FuncParam, SourceLoc, StorageClass,
    };
    use crate::decompile::passes::clight_select::select::MAX_STAGE4_PROFILES_PER_FUNCTION;

    fn function(name: &str, body: CStmt) -> FuncDef {
        FuncDef {
            name: name.to_string(),
            return_type: CType::int(),
            params: vec![FuncParam::named("x", CType::int())],
            is_variadic: false,
            storage_class: StorageClass::Auto,
            body,
            local_vars: Vec::new(),
            loc: SourceLoc::unknown(),
        }
    }

    fn seed_decoded_owned(db: &mut DecompileDB, function: u64, node: u64) {
        db.rel_push("instr_in_function", (node, function));
        db.rel_push(
            "instruction",
            (node, 1usize, "", "nop", "", "", "", "", 0usize, 0usize),
        );
    }

    #[test]
    fn unchanged_or_identity_mismatched_functions_are_not_snapshotted() {
        let original = function("f", CStmt::Return(Some(CExpr::int(0))));
        assert!(snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &original,
            &original,
        )
        .is_none());
        let renamed = function("g", CStmt::Return(Some(CExpr::int(1))));
        assert!(snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &original,
            &renamed,
        )
        .is_none());
    }

    #[test]
    fn typed_lvalue_pair_is_atomic_closed_and_v3_preserves_legacy_names() {
        let canonical = function("f", CStmt::Return(Some(CExpr::int(0))));
        let pre = function(
            "f",
            CStmt::Block(vec![
                CBlockItem::Stmt(CStmt::Expr(CExpr::assign(
                    CExpr::var("temporary"),
                    CExpr::int(1),
                ))),
                CBlockItem::Stmt(CStmt::Return(Some(CExpr::var("temporary")))),
            ]),
        );
        let post = function("f", CStmt::Return(Some(CExpr::int(1))));
        let pair = scalar_lvalue_snapshot_pair(3, 0x1000, &canonical, &pre, &post)
            .expect("two distinct signature-preserving typed-lvalue forms");
        assert_eq!(SOURCE_ALTERNATIVES_SCHEMA, "manifold-source-alternatives-v3");
        assert_eq!(
            pair[0].boundary.wire_name(),
            "scalar_lvalue_pre_var_reduce"
        );
        assert_eq!(pair[0].kinds, vec!["local_lifetime", "typed_lvalue"]);
        assert_eq!(
            pair[1].boundary.wire_name(),
            "scalar_lvalue_post_var_reduce"
        );
        assert_eq!(pair[1].kinds, vec!["typed_lvalue"]);
        assert!(scalar_lvalue_snapshot_pair(3, 0x1000, &canonical, &post, &post).is_none());

        let mut wrong_signature = post.clone();
        wrong_signature.return_type = CType::long();
        assert!(
            scalar_lvalue_snapshot_pair(3, 0x1000, &canonical, &pre, &wrong_signature)
                .is_none()
        );
    }

    #[test]
    fn v3_feature_profiles_collapse_by_ordered_printed_source_identity() {
        let canonical = function("f", CStmt::Return(Some(CExpr::int(0))));
        let pre = function("f", CStmt::Return(Some(CExpr::int(1))));
        let post = function("f", CStmt::Return(Some(CExpr::int(2))));
        let both = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical,
            &pre,
            &post,
            BinaryFormat::Coff,
        )
        .expect("two useful forms");
        assert_eq!(both.len(), 2);

        let pre_only = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical,
            &pre,
            &canonical,
            BinaryFormat::Coff,
        )
        .expect("pre remains useful");
        assert_eq!(pre_only.len(), 1);
        assert_eq!(pre_only[0].boundary, SourceAlternativeBoundary::ScalarLvaluePreVarReduce);

        let post_only = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical,
            &canonical,
            &post,
            BinaryFormat::Coff,
        )
        .expect("post remains useful");
        assert_eq!(post_only.len(), 1);
        assert_eq!(post_only[0].boundary, SourceAlternativeBoundary::ScalarLvaluePostVarReduce);

        let duplicate = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical,
            &pre,
            &pre,
            BinaryFormat::Coff,
        )
        .expect("the first equal alternative survives");
        assert_eq!(duplicate.len(), 1);
        assert_eq!(duplicate[0].boundary, SourceAlternativeBoundary::ScalarLvaluePreVarReduce);
        assert!(scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical,
            &canonical,
            &canonical,
            BinaryFormat::Coff,
        )
        .is_none());
    }

    fn stage4_profile(kind: crate::x86::types::Stage4SourceKind, root_node: u64) -> Stage4SourceProfile {
        Stage4SourceProfile { kind, root_node }
    }

    #[test]
    fn v4_stage4_profiles_collapse_to_one_closed_pre_post_subsequence() {
        use crate::x86::types::Stage4SourceKind;

        let canonical = function("f", CStmt::Return(Some(CExpr::int(0))));
        let pre = function("f", CStmt::Return(Some(CExpr::int(1))));
        let post = function("f", CStmt::Return(Some(CExpr::int(2))));
        let profile = stage4_profile(Stage4SourceKind::AffineAddress, 0x1010);
        let both = stage4_feature_snapshots_for_format(
            0,
            0x1000,
            profile,
            &canonical,
            &pre,
            &post,
            BinaryFormat::Coff,
        )
        .expect("two content-distinct affine forms");
        assert_eq!(both.len(), 2);
        assert!(both.iter().all(|snapshot| snapshot.stage4_profile == Some(profile)));
        assert_eq!(both[0].boundary, SourceAlternativeBoundary::Stage4AffinePreVarReduce);
        assert_eq!(both[0].kinds, vec!["address_expression", "local_lifetime"]);
        assert_eq!(both[1].boundary, SourceAlternativeBoundary::Stage4AffinePostVarReduce);
        assert_eq!(both[1].kinds, vec!["address_expression"]);

        let one = stage4_feature_snapshots_for_format(
            0,
            0x1000,
            profile,
            &canonical,
            &pre,
            &pre,
            BinaryFormat::Coff,
        )
        .expect("the first duplicate source survives");
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].boundary, SourceAlternativeBoundary::Stage4AffinePreVarReduce);

        let mut wrong_signature = post.clone();
        wrong_signature.return_type = CType::long();
        assert!(stage4_feature_snapshots_for_format(
            0,
            0x1000,
            profile,
            &canonical,
            &pre,
            &wrong_signature,
            BinaryFormat::Coff,
        )
        .is_none());
    }

    #[test]
    fn v4_is_cumulative_and_stage4_failure_preserves_exact_v3_manifest() {
        use crate::x86::types::Stage4SourceKind;

        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let ordinary_function = function("f", CStmt::Return(Some(CExpr::int(90))));
        let ordinary = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &ordinary_function,
            &canonical_function,
        )
        .expect("ordinary snapshot");
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];
        let render = |snapshots: &[SourceAlternativeSnapshot]| {
            render_manifest(
                &final_tu,
                snapshots,
                false,
                &SourceAlternativeExclusions::default(),
                &identities,
                &canonical,
                BinaryFormat::Coff,
            )
            .expect("render cumulative fixture")
        };
        let v3 = render(std::slice::from_ref(&ordinary))
            .expect("ordinary v3 manifest");
        let v3_json: serde_json::Value = serde_json::from_str(&v3).unwrap();
        assert_eq!(v3_json["schema"], SOURCE_ALTERNATIVES_SCHEMA);
        assert!(v3_json["alternatives"][0].get("profile_ordinal").is_none());

        let profile = stage4_profile(Stage4SourceKind::Zeroing, 0x1010);
        let mut cumulative = vec![ordinary.clone()];
        cumulative.extend(
            stage4_feature_snapshots_for_format(
                0,
                0x1000,
                profile,
                &canonical_function,
                &function("f", CStmt::Return(Some(CExpr::int(1)))),
                &function("f", CStmt::Return(Some(CExpr::int(2)))),
                BinaryFormat::Coff,
            )
            .expect("zeroing profile"),
        );
        let v4 = render(&cumulative).expect("cumulative v4 manifest");
        let v4_json: serde_json::Value = serde_json::from_str(&v4).unwrap();
        assert_eq!(v4_json["schema"], SOURCE_ALTERNATIVES_SCHEMA_V4);
        assert_eq!(
            v4_json["max_per_function"],
            MAX_SOURCE_ALTERNATIVES_PER_FUNCTION_V4
        );
        assert_eq!(v4_json["alternatives"][0], v3_json["alternatives"][0]);
        assert_eq!(
            v4_json["alternatives"][1]["id"],
            "function-000000:stage4_zero_pre_var_reduce:profile-0000"
        );
        assert_eq!(v4_json["alternatives"][1]["profile_root_node"], "0x1010");

        // An exact source collision drops only the affected Stage-4 group and
        // recursively renders the historical bytes; it cannot evict ordinary.
        let mut collision = vec![ordinary.clone()];
        collision.extend(
            stage4_feature_snapshots_for_format(
                0,
                0x1000,
                profile,
                &canonical_function,
                &ordinary_function,
                &canonical_function,
                BinaryFormat::Coff,
            )
            .expect("colliding stage4 row"),
        );
        assert_eq!(render(&collision).expect("v3 fallback"), v3);

        let mut missing_profile = cumulative.clone();
        for snapshot in &mut missing_profile {
            if is_stage4_boundary(snapshot.boundary) {
                snapshot.stage4_profile = None;
            }
        }
        assert_eq!(render(&missing_profile).expect("missing profile fallback"), v3);
    }

    #[test]
    fn v4_profiles_are_independent_contiguous_and_bounded_without_eviction() {
        use crate::x86::types::Stage4SourceKind;

        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let ordinary = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &function("f", CStmt::Return(Some(CExpr::int(90)))),
            &canonical_function,
        )
        .unwrap();
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];
        let render = |snapshots: &[SourceAlternativeSnapshot]| {
            render_manifest(
                &final_tu,
                snapshots,
                false,
                &SourceAlternativeExclusions::default(),
                &identities,
                &canonical,
                BinaryFormat::Coff,
            )
            .expect("render bounded profile fixture")
        };
        let v3 = render(std::slice::from_ref(&ordinary)).unwrap();

        let colliding_profile = stage4_profile(Stage4SourceKind::AffineAddress, 0x1010);
        let valid_profile = stage4_profile(Stage4SourceKind::Zeroing, 0x1020);
        let mut rows = vec![ordinary.clone()];
        rows.extend(
            stage4_feature_snapshots_for_format(
                0,
                0x1000,
                colliding_profile,
                &canonical_function,
                &function("f", CStmt::Return(Some(CExpr::int(90)))),
                &canonical_function,
                BinaryFormat::Coff,
            )
            .unwrap(),
        );
        rows.extend(
            stage4_feature_snapshots_for_format(
                0,
                0x1000,
                valid_profile,
                &canonical_function,
                &function("f", CStmt::Return(Some(CExpr::int(1)))),
                &canonical_function,
                BinaryFormat::Coff,
            )
            .unwrap(),
        );
        let rendered = render(&rows).expect("valid sibling remains");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let alternatives = parsed["alternatives"].as_array().unwrap();
        assert_eq!(alternatives.len(), 2);
        assert_eq!(alternatives[1]["profile_ordinal"], 0);
        assert_eq!(alternatives[1]["profile_root_node"], "0x1020");

        let mut overflow = vec![ordinary];
        for index in 0..=MAX_STAGE4_PROFILES_PER_FUNCTION {
            let profile = stage4_profile(Stage4SourceKind::AffineAddress, 0x2000 + index as u64);
            overflow.extend(
                stage4_feature_snapshots_for_format(
                    0,
                    0x1000,
                    profile,
                    &canonical_function,
                    &function(
                        "f",
                        CStmt::Return(Some(CExpr::int(1000 + index as i64))),
                    ),
                    &canonical_function,
                    BinaryFormat::Coff,
                )
                .unwrap(),
            );
        }
        assert_eq!(render(&overflow).expect("profile overflow preserves v3"), v3);
    }

    #[test]
    fn profiles_control_and_call_result_boundaries_without_source_names() {
        let pre_control = function(
            "f",
            CStmt::If(
                CExpr::var("x"),
                Box::new(CStmt::Return(Some(CExpr::call(
                    CExpr::var("callee"),
                    vec![],
                )))),
                None,
            ),
        );
        let post_control = function(
            "f",
            CStmt::Return(Some(CExpr::call(CExpr::var("callee"), vec![]))),
        );
        let control = snapshot_if_changed(
            7,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &pre_control,
            &post_control,
        )
        .unwrap();
        assert_eq!(
            control.kinds,
            vec!["conditional", "control_layout", "early_return"]
        );

        let mut pre_value = function(
            "f",
            CStmt::Block(vec![
                CBlockItem::Stmt(CStmt::Expr(CExpr::assign(
                    CExpr::var("temporary"),
                    CExpr::call(CExpr::var("callee"), vec![]),
                ))),
                CBlockItem::Stmt(CStmt::Return(Some(CExpr::var("temporary")))),
            ]),
        );
        pre_value
            .local_vars
            .push(crate::decompile::passes::c_pass::types::VarDecl::new(
                "temporary",
                CType::int(),
            ));
        let post_value = function(
            "f",
            CStmt::Return(Some(CExpr::call(CExpr::var("callee"), vec![]))),
        );
        let value = snapshot_if_changed(
            7,
            0x1000,
            SourceAlternativeBoundary::PreVarReduce,
            &pre_value,
            &post_value,
        )
        .unwrap();
        assert_eq!(value.kinds, vec!["call_result", "local_lifetime"]);

        // An unrelated call does not authenticate a call-result alternative:
        // its closed materialization profile is identical on both sides.
        let unrelated_call_before = function(
            "f",
            CStmt::Return(Some(CExpr::Binary(
                BinaryOp::Add,
                Box::new(CExpr::call(CExpr::var("callee"), vec![])),
                Box::new(CExpr::var("x")),
            ))),
        );
        let unrelated_call_after = function(
            "f",
            CStmt::Return(Some(CExpr::Binary(
                BinaryOp::Add,
                Box::new(CExpr::call(CExpr::var("callee"), vec![])),
                Box::new(CExpr::Binary(
                    BinaryOp::Add,
                    Box::new(CExpr::var("x")),
                    Box::new(CExpr::int(0)),
                )),
            ))),
        );
        let unrelated_call = snapshot_if_changed(
            7,
            0x1000,
            SourceAlternativeBoundary::PreVarReduce,
            &unrelated_call_before,
            &unrelated_call_after,
        )
        .unwrap();
        assert_eq!(unrelated_call.kinds, vec!["local_lifetime"]);

        let no_call_before = function("f", CStmt::Return(Some(CExpr::var("x"))));
        let no_call_after = function(
            "f",
            CStmt::Return(Some(CExpr::Binary(
                BinaryOp::Add,
                Box::new(CExpr::var("x")),
                Box::new(CExpr::int(0)),
            ))),
        );
        let no_call = snapshot_if_changed(
            7,
            0x1000,
            SourceAlternativeBoundary::PreVarReduce,
            &no_call_before,
            &no_call_after,
        )
        .unwrap();
        assert_eq!(no_call.kinds, vec!["local_lifetime"]);
    }

    #[test]
    fn profiles_bounded_return_tail_duplication() {
        let call = || CExpr::call(CExpr::var("callee"), vec![]);
        let pre = function(
            "f",
            CStmt::Block(vec![
                CBlockItem::Stmt(CStmt::If(
                    CExpr::var("x"),
                    Box::new(CStmt::Goto("return_tail".to_string())),
                    None,
                )),
                CBlockItem::Stmt(CStmt::Labeled(
                    crate::decompile::passes::c_pass::types::Label::Named(
                        "return_tail".to_string(),
                    ),
                    Box::new(CStmt::Return(Some(call()))),
                )),
            ]),
        );
        let post = function(
            "f",
            CStmt::Block(vec![
                CBlockItem::Stmt(CStmt::If(
                    CExpr::var("x"),
                    Box::new(CStmt::Return(Some(call()))),
                    None,
                )),
                CBlockItem::Stmt(CStmt::Return(Some(call()))),
            ]),
        );
        let snapshot = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &pre,
            &post,
        )
        .unwrap();
        assert!(snapshot.kinds.contains(&"tail_call"));
        assert!(snapshot.kinds.contains(&"early_return"));
    }

    #[test]
    fn manifest_is_bounded_sorted_and_bound_to_final_decl_identity() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let final_tu_before = final_tu.clone();
        let pre_forloop = function("f", CStmt::Return(Some(CExpr::int(0))));
        let pre_var = function("f", CStmt::Return(Some(CExpr::int(1))));
        let stale = function("other", CStmt::Return(Some(CExpr::int(3))));
        let snapshots = vec![
            snapshot_if_changed(
                0,
                0x1000,
                SourceAlternativeBoundary::PreVarReduce,
                &pre_var,
                &final_function,
            )
            .unwrap(),
            snapshot_if_changed(
                0,
                0x1000,
                SourceAlternativeBoundary::PreForLoop,
                &pre_forloop,
                &final_function,
            )
            .unwrap(),
            SourceAlternativeSnapshot {
                declaration_index: 1,
                manifold_name: stale.name.clone(),
                manifold_address: 0x2000,
                boundary: SourceAlternativeBoundary::PreForLoop,
                kinds: vec!["control_layout"],
                stage4_profile: None,
                function: stale,
            },
        ];
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let rendered = render_manifest(
            &final_tu,
            &snapshots,
            false,
            &SourceAlternativeExclusions::default(),
            &[("f".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap()
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let records = parsed["alternatives"].as_array().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["id"], "function-000000:pre_forloop");
        assert_eq!(records[1]["id"], "function-000000:pre_var_reduce");
        assert_ne!(
            records[0]["canonical_source"],
            records[0]["alternative_source"]
        );
        assert_eq!(
            records[0]["canonical_source_sha256"],
            crate::decompile::disassembly::coff::sha256_hex(
                records[0]["canonical_source"].as_str().unwrap().as_bytes(),
            )
        );
        assert_eq!(
            records[0]["alternative_source_sha256"],
            crate::decompile::disassembly::coff::sha256_hex(
                records[0]["alternative_source"]
                    .as_str()
                    .unwrap()
                    .as_bytes(),
            )
        );
        assert_eq!(records[0]["manifold_address"], "0x1000");
        assert_eq!(
            parsed["canonical_translation_unit_sha256"],
            crate::decompile::disassembly::coff::sha256_hex(canonical.as_bytes())
        );
        assert_eq!(
            parsed["max_per_function"],
            MAX_SOURCE_ALTERNATIVES_PER_FUNCTION
        );
        assert_eq!(parsed["truncated"], false);
        assert_eq!(final_tu, final_tu_before);
    }

    #[test]
    fn manifest_omits_non_clight_identity_and_retains_adjacent_valid_row() {
        let kept = function("coff_fn_kept", CStmt::Return(Some(CExpr::int(2))));
        let omitted = function("FUN_10002000", CStmt::Return(Some(CExpr::int(4))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(kept.clone());
        final_tu.add_function(omitted.clone());
        let snapshots = [
            snapshot_if_changed(
                0,
                0x1000,
                SourceAlternativeBoundary::PreVarReduce,
                &function("coff_fn_kept", CStmt::Return(Some(CExpr::int(1)))),
                &kept,
            )
            .unwrap(),
            snapshot_if_changed(
                1,
                0x2000,
                SourceAlternativeBoundary::PreVarReduce,
                &function("FUN_10002000", CStmt::Return(Some(CExpr::int(3)))),
                &omitted,
            )
            .unwrap(),
        ];
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let rendered = render_manifest(
            &final_tu,
            &snapshots,
            false,
            &SourceAlternativeExclusions::default(),
            &[("coff_fn_kept".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Coff,
        )
        .unwrap()
        .expect("the adjacent emitted identity remains eligible");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let records = parsed["alternatives"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["manifold_name"], "coff_fn_kept");
        assert_eq!(records[0]["manifold_address"], "0x1000");
        assert_eq!(records[0]["id"], "function-000000:pre_var_reduce");
    }

    #[test]
    fn manifest_emits_the_exact_ten_record_cumulative_v3_portfolio() {
        // Regression shape from
        // wbemcomn.dll|admin/wmi/wbem/winmgmt/wbemcomn/smallarr|
        // ?InsertAt@CSmallArrayBlob@@QEAAPEAV1@HPEAX@Z: Stage-2 cap2
        // displaced its profitable ordinary pre_forloop row. The selector is
        // wholly generic; this fixture pins only the cumulative wire shape.
        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let mut snapshots = Vec::new();
        for (boundary, value) in [
            (SourceAlternativeBoundary::PreForLoop, 90),
            (SourceAlternativeBoundary::PreVarReduce, 91),
        ] {
            snapshots.push(
                snapshot_if_changed(
                    0,
                    0x1000,
                    boundary,
                    &function("f", CStmt::Return(Some(CExpr::int(value)))),
                    &canonical_function,
                )
                .expect("ordinary snapshot"),
            );
        }
        for (index, family) in [
            ScalarSourceAlternativeFamily::TypedLvalue,
            ScalarSourceAlternativeFamily::FieldLvalue,
            ScalarSourceAlternativeFamily::ExtensionPerUse,
            ScalarSourceAlternativeFamily::ExtensionHoisted,
        ]
        .into_iter()
        .enumerate()
        {
            snapshots.extend(
                scalar_feature_snapshots_for_format(
                    0,
                    0x1000,
                    family,
                    &canonical_function,
                    &function("f", CStmt::Return(Some(CExpr::int(10 + index as i64 * 2)))),
                    &function("f", CStmt::Return(Some(CExpr::int(11 + index as i64 * 2)))),
                    BinaryFormat::Coff,
                )
                .expect("two useful feature forms"),
            );
        }
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function);
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];
        let rendered = render_manifest(
            &final_tu,
            &snapshots,
            false,
            &SourceAlternativeExclusions::default(),
            &identities,
            &canonical,
            BinaryFormat::Coff,
        )
        .unwrap()
        .expect("complete pair renders");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let records = parsed["alternatives"].as_array().unwrap();
        assert_eq!(records.len(), MAX_SOURCE_ALTERNATIVES_PER_FUNCTION);
        assert_eq!(MAX_SOURCE_ALTERNATIVES_PER_FUNCTION, 10);
        assert_eq!(records[0]["id"], "function-000000:pre_forloop");
        assert_eq!(records[1]["id"], "function-000000:pre_var_reduce");
        assert_eq!(records[2]["id"], "function-000000:scalar_lvalue_pre_var_reduce");
        assert_eq!(records[9]["id"], "function-000000:scalar_extension_hoisted_post_var_reduce");
        assert_eq!(parsed["schema"], SOURCE_ALTERNATIVES_SCHEMA);
    }

    #[test]
    fn v3_extension_requires_both_nonempty_collapsed_placements() {
        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let per_use = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::ExtensionPerUse,
            &canonical_function,
            &function("f", CStmt::Return(Some(CExpr::int(1)))),
            &canonical_function,
            BinaryFormat::Coff,
        )
        .expect("one useful per-use boundary");
        let hoisted = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::ExtensionHoisted,
            &canonical_function,
            &canonical_function,
            &function("f", CStmt::Return(Some(CExpr::int(2)))),
            BinaryFormat::Coff,
        )
        .expect("one useful hoisted boundary");
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function);
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];

        assert!(render_manifest(
            &final_tu,
            &per_use,
            false,
            &SourceAlternativeExclusions::default(),
            &identities,
            &canonical,
            BinaryFormat::Coff,
        )
        .unwrap()
        .is_none());

        let mut complete = per_use;
        complete.extend(hoisted);
        let rendered = render_manifest(
            &final_tu,
            &complete,
            false,
            &SourceAlternativeExclusions::default(),
            &identities,
            &canonical,
            BinaryFormat::Coff,
        )
        .unwrap()
        .expect("one boundary from each placement is atomic and useful");
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let ids: Vec<_> = parsed["alternatives"]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| record["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            [
                "function-000000:scalar_extension_per_use_pre_var_reduce",
                "function-000000:scalar_extension_hoisted_post_var_reduce",
            ]
        );
    }

    #[test]
    fn cross_family_source_collisions_preserve_ordinary_and_drop_features_atomically() {
        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let ordinary_function = function("f", CStmt::Return(Some(CExpr::int(1))));
        let ordinary = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &ordinary_function,
            &canonical_function,
        )
        .expect("ordinary snapshot");
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];
        let render = |snapshots: &[SourceAlternativeSnapshot]| {
            render_manifest(
                &final_tu,
                snapshots,
                false,
                &SourceAlternativeExclusions::default(),
                &identities,
                &canonical,
                BinaryFormat::Coff,
            )
            .expect("render collision fixture")
        };
        let ordinary_rendered = render(std::slice::from_ref(&ordinary))
            .expect("ordinary collision priority row");

        let typed_equal_to_ordinary = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical_function,
            &ordinary_function,
            &canonical_function,
            BinaryFormat::Coff,
        )
        .expect("typed collision row");
        let mut ordinary_collision = vec![ordinary.clone()];
        ordinary_collision.extend(typed_equal_to_ordinary);
        assert_eq!(
            render(&ordinary_collision).expect("ordinary survives feature collision"),
            ordinary_rendered,
        );

        let shared_feature = function("f", CStmt::Return(Some(CExpr::int(2))));
        let mut typed_field_collision = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical_function,
            &shared_feature,
            &canonical_function,
            BinaryFormat::Coff,
        )
        .expect("typed feature row");
        typed_field_collision.extend(
            scalar_feature_snapshots_for_format(
                0,
                0x1000,
                ScalarSourceAlternativeFamily::FieldLvalue,
                &canonical_function,
                &shared_feature,
                &canonical_function,
                BinaryFormat::Coff,
            )
            .expect("field collision row"),
        );
        let typed_field = render(&typed_field_collision).expect("typed priority sidecar");
        let typed_field: serde_json::Value =
            serde_json::from_str(&typed_field).expect("typed priority sidecar parses");
        let records = typed_field["alternatives"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["boundary"], "scalar_lvalue_pre_var_reduce");

        let shared_extension = function("f", CStmt::Return(Some(CExpr::int(3))));
        let mut extension_collision = vec![ordinary.clone()];
        extension_collision.extend(
            scalar_feature_snapshots_for_format(
                0,
                0x1000,
                ScalarSourceAlternativeFamily::ExtensionPerUse,
                &canonical_function,
                &shared_extension,
                &canonical_function,
                BinaryFormat::Coff,
            )
            .expect("per-use collision row"),
        );
        extension_collision.extend(
            scalar_feature_snapshots_for_format(
                0,
                0x1000,
                ScalarSourceAlternativeFamily::ExtensionHoisted,
                &canonical_function,
                &canonical_function,
                &shared_extension,
                BinaryFormat::Coff,
            )
            .expect("hoisted collision row"),
        );
        assert_eq!(
            render(&extension_collision).expect("ordinary survives extension collision"),
            ordinary_rendered,
        );

        // A duplicate between the two historical boundaries cannot arise from
        // the pass chain, but forged input is rejected rather than silently
        // choosing one Stage-1 row and changing its ordered-set identity.
        let mut duplicate_ordinary = ordinary;
        duplicate_ordinary.boundary = SourceAlternativeBoundary::PreVarReduce;
        assert!(render(&[ordinary_collision[0].clone(), duplicate_ordinary]).is_none());
    }

    #[test]
    fn feature_count_overflow_preserves_exact_ordinary_4095_record_manifest() {
        let mut final_tu = TranslationUnit::new();
        let mut ordinary = Vec::new();
        let mut identities = Vec::new();
        let mut first_canonical = None;
        for index in 0..4095usize {
            let name = format!("f_{index:04}");
            let address = 0x1000 + index as u64 * 0x10;
            let canonical_function =
                function(&name, CStmt::Return(Some(CExpr::int(index as i64))));
            let alternative = function(
                &name,
                CStmt::Return(Some(CExpr::int(index as i64 + 10_000))),
            );
            if index == 0 {
                first_canonical = Some(canonical_function.clone());
            }
            ordinary.push(
                snapshot_if_changed(
                    index,
                    address,
                    SourceAlternativeBoundary::PreForLoop,
                    &alternative,
                    &canonical_function,
                )
                .expect("ordinary snapshot"),
            );
            identities.push((name, address));
            final_tu.add_function(canonical_function);
        }
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let exclusions = SourceAlternativeExclusions::default();
        let ordinary_rendered = render_manifest_with_limits(
            &final_tu,
            &ordinary,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            4096,
            MAX_SOURCE_ALTERNATIVE_BYTES,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("4095 ordinary rows fit");
        let first_canonical = first_canonical.expect("first function");
        let one_feature = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &first_canonical,
            &function("f_0000", CStmt::Return(Some(CExpr::int(20_000)))),
            &first_canonical,
            BinaryFormat::Coff,
        )
        .expect("one collapsed feature row");
        let mut exact_4096 = ordinary.clone();
        exact_4096.extend(one_feature);
        let rendered_4096 = render_manifest_with_limits(
            &final_tu,
            &exact_4096,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            4096,
            MAX_SOURCE_ALTERNATIVE_BYTES,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("exactly 4096 rows fit");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&rendered_4096).unwrap()["alternatives"]
                .as_array()
                .unwrap()
                .len(),
            4096
        );

        let two_features = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &first_canonical,
            &function("f_0000", CStmt::Return(Some(CExpr::int(20_001)))),
            &function("f_0000", CStmt::Return(Some(CExpr::int(20_002)))),
            BinaryFormat::Coff,
        )
        .expect("two distinct feature rows");
        let mut overflow_4097 = ordinary.clone();
        overflow_4097.extend(two_features);
        let fallback = render_manifest_with_limits(
            &final_tu,
            &overflow_4097,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            4096,
            MAX_SOURCE_ALTERNATIVE_BYTES,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("feature overflow falls back to ordinary");
        assert_eq!(fallback, ordinary_rendered);
    }

    #[test]
    fn feature_source_and_manifest_byte_overflow_preserve_ordinary_bytes() {
        let canonical_function = function("f", CStmt::Return(Some(CExpr::int(100))));
        let ordinary_function = function("f", CStmt::Return(Some(CExpr::int(1))));
        let ordinary = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &ordinary_function,
            &canonical_function,
        )
        .expect("ordinary snapshot");
        let feature = scalar_feature_snapshots_for_format(
            0,
            0x1000,
            ScalarSourceAlternativeFamily::TypedLvalue,
            &canonical_function,
            &function("f", CStmt::Return(Some(CExpr::int(2)))),
            &canonical_function,
            BinaryFormat::Coff,
        )
        .expect("one feature row");
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(canonical_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Coff,
        );
        let identities = [("f".to_string(), 0x1000)];
        let exclusions = SourceAlternativeExclusions::default();
        let ordinary_source_bytes = one_function_source(&canonical_function, BinaryFormat::Coff)
            .len()
            + one_function_source(&ordinary_function, BinaryFormat::Coff).len();
        let ordinary_rendered = render_manifest_with_limits(
            &final_tu,
            std::slice::from_ref(&ordinary),
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            10,
            ordinary_source_bytes,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("ordinary row fits its exact source-byte bound");
        let mut combined = vec![ordinary];
        combined.extend(feature);
        let source_fallback = render_manifest_with_limits(
            &final_tu,
            &combined,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            10,
            ordinary_source_bytes,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("feature source bytes fall back to ordinary");
        assert_eq!(source_fallback, ordinary_rendered);

        let ordinary_unlimited = render_manifest_with_limits(
            &final_tu,
            &combined[..1],
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            10,
            MAX_SOURCE_ALTERNATIVE_BYTES,
            MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        )
        .unwrap()
        .expect("ordinary manifest");
        let manifest_fallback = render_manifest_with_limits(
            &final_tu,
            &combined,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Coff,
            10,
            MAX_SOURCE_ALTERNATIVE_BYTES,
            ordinary_unlimited.len(),
        )
        .unwrap()
        .expect("feature manifest bytes fall back to ordinary");
        assert_eq!(manifest_fallback, ordinary_unlimited);
    }

    #[test]
    fn manifest_rejects_each_final_ineligible_function_class() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let alternative = function("f", CStmt::Return(Some(CExpr::int(1))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let snapshot = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &final_function,
        )
        .unwrap();
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let identities = [("f".to_string(), 0x1000)];

        assert!(render_manifest(
            &final_tu,
            &[snapshot.clone()],
            false,
            &SourceAlternativeExclusions::default(),
            &identities,
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap()
        .is_some());

        for exclusions in [
            SourceAlternativeExclusions {
                partial_functions: HashSet::from([0x1000]),
                ..Default::default()
            },
            SourceAlternativeExclusions {
                unsupported_functions: HashSet::from([0x1000]),
                ..Default::default()
            },
            SourceAlternativeExclusions {
                omitted_functions: HashSet::from([0x1000]),
                ..Default::default()
            },
            SourceAlternativeExclusions {
                machine_state_functions: HashSet::from([(0x1000, "f".to_string())]),
                ..Default::default()
            },
        ] {
            assert!(render_manifest(
                &final_tu,
                &[snapshot.clone()],
                false,
                &exclusions,
                &identities,
                &canonical,
                BinaryFormat::Pe,
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn final_exclusions_use_authenticated_partial_and_omission_relations() {
        const PARTIAL_FUNCTION: u64 = 0x1000;
        const PARTIAL_ACCESS: u64 = 0x1010;
        let mut partial = DecompileDB::default();
        partial.rel_push(
            "unsupported_stack_address",
            (
                PARTIAL_FUNCTION,
                PARTIAL_ACCESS,
                "unsupported-stack-address",
            ),
        );
        partial.rel_push(
            "suppressed_unsupported_address",
            (
                PARTIAL_FUNCTION,
                PARTIAL_ACCESS,
                "unsupported-stack-address",
            ),
        );
        partial.rel_push(
            "suppressed_unsupported_address_node",
            (PARTIAL_FUNCTION, PARTIAL_ACCESS, PARTIAL_ACCESS),
        );
        partial.rel_push(
            "partial_unsupported_function",
            (PARTIAL_FUNCTION, 1usize, 8usize),
        );
        for node in PARTIAL_ACCESS..PARTIAL_ACCESS + 8 {
            seed_decoded_owned(&mut partial, PARTIAL_FUNCTION, node);
        }
        let partial_exclusions = final_source_alternative_exclusions(&partial, &[]);
        assert!(partial_exclusions
            .partial_functions
            .contains(&PARTIAL_FUNCTION));
        assert!(!partial_exclusions
            .unsupported_functions
            .contains(&PARTIAL_FUNCTION));
        assert!(!partial_exclusions
            .omitted_functions
            .contains(&PARTIAL_FUNCTION));
        assert!(partial_exclusions.excludes("partial", PARTIAL_FUNCTION));

        const UNSUPPORTED_FUNCTION: u64 = 0x2000;
        const UNSUPPORTED_ACCESS: u64 = 0x2010;
        let mut unsupported = DecompileDB::default();
        unsupported.rel_push(
            "unsupported_stack_address",
            (
                UNSUPPORTED_FUNCTION,
                UNSUPPORTED_ACCESS,
                "unsupported-stack-address",
            ),
        );
        let unsupported_exclusions = final_source_alternative_exclusions(&unsupported, &[]);
        assert!(!unsupported_exclusions
            .partial_functions
            .contains(&UNSUPPORTED_FUNCTION));
        assert!(unsupported_exclusions
            .unsupported_functions
            .contains(&UNSUPPORTED_FUNCTION));
        assert!(unsupported_exclusions
            .omitted_functions
            .contains(&UNSUPPORTED_FUNCTION));
        assert!(unsupported_exclusions.excludes("unsupported", UNSUPPORTED_FUNCTION));
    }

    #[test]
    fn invalid_partial_artifact_atomically_suppresses_clean_sibling_sidecar() {
        let mut invalid = DecompileDB::default();
        invalid.rel_push(
            "suppressed_unsupported_address",
            (0x9000u64, 0x9010u64, "unsupported-stack-address"),
        );
        let identities = [("clean".to_string(), 0x5000u64)];
        let exclusions = final_source_alternative_exclusions(&invalid, &identities);
        assert!(exclusions.suppress_all);

        let final_function = function("clean", CStmt::Return(Some(CExpr::int(2))));
        let alternative = function("clean", CStmt::Return(Some(CExpr::int(1))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let snapshot = snapshot_if_changed(
            0,
            0x5000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &final_function,
        )
        .unwrap();
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        assert!(render_manifest(
            &final_tu,
            &[snapshot],
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn machine_state_exclusion_requires_exact_name_and_address() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let alternative = function("f", CStmt::Return(Some(CExpr::int(1))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let snapshot = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &final_function,
        )
        .unwrap();
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let exclusions = SourceAlternativeExclusions {
            machine_state_functions: HashSet::from([(0x1000, "other".to_string())]),
            ..Default::default()
        };
        assert!(render_manifest(
            &final_tu,
            &[snapshot],
            false,
            &exclusions,
            &[("f".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap()
        .is_some());
    }

    #[test]
    fn final_ineligible_functions_do_not_suppress_clean_same_tu_sibling() {
        let cases = [
            ("partial", 0x1000),
            ("unsupported", 0x2000),
            ("omitted", 0x3000),
            ("machine_state", 0x4000),
            ("clean", 0x5000),
        ];
        let mut final_tu = TranslationUnit::new();
        let mut snapshots = Vec::new();
        let mut identities = Vec::new();
        for (declaration_index, (name, address)) in cases.into_iter().enumerate() {
            let final_function = function(name, CStmt::Return(Some(CExpr::int(2))));
            let alternative = function(name, CStmt::Return(Some(CExpr::int(1))));
            snapshots.push(
                snapshot_if_changed(
                    declaration_index,
                    address,
                    SourceAlternativeBoundary::PreForLoop,
                    &alternative,
                    &final_function,
                )
                .unwrap(),
            );
            final_tu.add_function(final_function);
            identities.push((name.to_string(), address));
        }
        let exclusions = SourceAlternativeExclusions {
            suppress_all: false,
            partial_functions: HashSet::from([0x1000]),
            unsupported_functions: HashSet::from([0x2000]),
            omitted_functions: HashSet::from([0x3000]),
            machine_state_functions: HashSet::from([(0x4000, "machine_state".to_string())]),
        };
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let rendered = render_manifest(
            &final_tu,
            &snapshots,
            false,
            &exclusions,
            &identities,
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap()
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let records = parsed["alternatives"].as_array().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["manifold_name"], "clean");
        assert_eq!(records[0]["manifold_address"], "0x5000");
        assert_eq!(records[0]["function_ordinal"], 4);
    }

    #[test]
    fn wire_ordinal_and_canonical_slice_follow_emitted_function_order() {
        use crate::decompile::passes::c_pass::types::{
            FuncDecl, StructDef, StructField, TypedefDecl, VarDecl,
        };

        let target = function("target", CStmt::Return(Some(CExpr::var("x"))));
        let alternative = function("target", CStmt::Return(Some(CExpr::int(7))));
        let mut final_tu = TranslationUnit::new();
        final_tu.decls.push(TopLevelDecl::Typedef(TypedefDecl {
            name: "word_type".to_string(),
            ty: CType::int(),
            loc: SourceLoc::unknown(),
        }));
        final_tu.add_func_decl(FuncDecl::new("callee", CType::int(), vec![]));
        final_tu.add_function(function("other", CStmt::Return(Some(CExpr::int(0)))));
        final_tu.add_global_var(VarDecl::new("global_value", CType::int()));
        // This definition has a declaration coordinate but no printed-function
        // ordinal because the canonical printer deliberately omits it.
        final_tu.add_function(function("empty", CStmt::Empty));
        final_tu
            .decls
            .push(TopLevelDecl::StructDef(StructDef::new_struct(
                "record_type",
                vec![StructField::new("field", CType::int())],
            )));
        let target_declaration_index = final_tu.decls.len();
        final_tu.add_function(target.clone());
        final_tu.add_function(function("later", CStmt::Return(Some(CExpr::int(3)))));
        let full_source = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let target_source = one_function_source(&target, BinaryFormat::Pe);
        assert_eq!(full_source.matches(&target_source).count(), 1);
        assert!(!full_source.contains("empty("));

        let snapshot = snapshot_if_changed(
            target_declaration_index,
            0x3000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &target,
        )
        .unwrap();
        let rendered = render_manifest(
            &final_tu,
            &[snapshot],
            false,
            &SourceAlternativeExclusions::default(),
            &[("target".to_string(), 0x3000)],
            &full_source,
            BinaryFormat::Pe,
        )
        .unwrap()
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        let record = &parsed["alternatives"][0];
        assert_eq!(target_declaration_index, 6);
        assert_eq!(record["function_ordinal"], 1);
        assert_eq!(record["id"], "function-000001:pre_forloop");
        assert_eq!(record["manifold_name"], "target");
        assert_eq!(record["canonical_source"], target_source);
    }

    #[test]
    fn capture_overflow_clears_and_permanently_disables_snapshots() {
        let original = function("f", CStmt::Return(Some(CExpr::int(0))));
        let changed = function("f", CStmt::Return(Some(CExpr::int(1))));
        let template = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &original,
            &changed,
        )
        .unwrap();
        let mut snapshots = Vec::new();
        let mut overflowed = false;
        for _ in 0..=MAX_TOTAL_SOURCE_ALTERNATIVES {
            record_bounded_snapshot(&mut snapshots, &mut overflowed, template.clone());
        }
        assert!(overflowed);
        assert!(snapshots.is_empty());
        record_bounded_snapshot(&mut snapshots, &mut overflowed, template);
        assert!(snapshots.is_empty());
    }

    #[test]
    fn manifest_rejects_competing_identity_and_boundary_evidence() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let alternative = function("f", CStmt::Return(Some(CExpr::int(1))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let snapshot = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &final_function,
        )
        .unwrap();
        let mut stale_declaration_coordinate = snapshot.clone();
        stale_declaration_coordinate.declaration_index = 1;

        for (snapshots, identities) in [
            (
                vec![snapshot.clone(), snapshot.clone()],
                vec![("f".to_string(), 0x1000)],
            ),
            (vec![snapshot.clone()], vec![("f".to_string(), 0x2000)]),
            (
                vec![snapshot.clone()],
                vec![("f".to_string(), 0x1000), ("f".to_string(), 0x1000)],
            ),
            (
                vec![stale_declaration_coordinate],
                vec![("f".to_string(), 0x1000)],
            ),
        ] {
            let rendered = render_manifest(
                &final_tu,
                &snapshots,
                false,
                &SourceAlternativeExclusions::default(),
                &identities,
                &canonical,
                BinaryFormat::Pe,
            )
            .unwrap();
            assert!(rendered.is_none());
        }
    }

    #[test]
    fn manifest_rejects_stale_canonical_source_and_signature() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let mut wrong_signature = function("f", CStmt::Return(Some(CExpr::int(1))));
        wrong_signature.return_type = CType::ulong();
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let snapshot = SourceAlternativeSnapshot {
            declaration_index: 0,
            manifold_name: "f".to_string(),
            manifold_address: 0x1000,
            boundary: SourceAlternativeBoundary::PreForLoop,
            kinds: vec!["control_layout"],
            stage4_profile: None,
            function: wrong_signature,
        };
        let rendered = render_manifest(
            &final_tu,
            &[snapshot.clone()],
            false,
            &SourceAlternativeExclusions::default(),
            &[("f".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Pe,
        )
        .unwrap();
        assert!(rendered.is_none());

        assert!(render_manifest(
            &final_tu,
            &[snapshot],
            false,
            &SourceAlternativeExclusions::default(),
            &[("f".to_string(), 0x1000)],
            "stale canonical source\n",
            BinaryFormat::Pe,
        )
        .is_err());
    }

    #[test]
    fn manifest_is_omitted_instead_of_truncated_at_closed_bounds() {
        let final_function = function("f", CStmt::Return(Some(CExpr::int(2))));
        let alternative = function("f", CStmt::Return(Some(CExpr::int(1))));
        let mut final_tu = TranslationUnit::new();
        final_tu.add_function(final_function.clone());
        let canonical = crate::decompile::passes::c_pass::print_translation_unit_for_format(
            &final_tu,
            BinaryFormat::Pe,
        );
        let snapshot = snapshot_if_changed(
            0,
            0x1000,
            SourceAlternativeBoundary::PreForLoop,
            &alternative,
            &final_function,
        )
        .unwrap();
        let rendered = render_manifest_with_limits(
            &final_tu,
            &[snapshot],
            false,
            &SourceAlternativeExclusions::default(),
            &[("f".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Pe,
            1,
            1,
            4096,
        )
        .unwrap();
        assert!(rendered.is_none());

        let rendered = render_manifest_with_limits(
            &final_tu,
            &[snapshot_if_changed(
                0,
                0x1000,
                SourceAlternativeBoundary::PreForLoop,
                &alternative,
                &final_function,
            )
            .unwrap()],
            false,
            &SourceAlternativeExclusions::default(),
            &[("f".to_string(), 0x1000)],
            &canonical,
            BinaryFormat::Pe,
            1,
            4096,
            1,
        )
        .unwrap();
        assert!(rendered.is_none());
    }
}
