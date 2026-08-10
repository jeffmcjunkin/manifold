//! Bounded, provider-owned snapshots of semantically equivalent final C forms.
//!
//! `ForLoopPass` and `VarReducePass` are destructive post-selection passes.  Their
//! inputs are already typed C trees, so retaining an input function when the pass
//! changes it preserves a much stronger provenance boundary than reconstructing an
//! alternative later from printed source.  The canonical translation unit remains
//! unchanged; this module only records at most one input snapshot per boundary and
//! function for downstream best-candidate evaluation.

use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::abi::BinaryFormat;
use crate::decompile::passes::c_pass::types::{
    CBlockItem, CExpr, CStmt, FuncDef, TopLevelDecl, TranslationUnit,
};
use crate::decompile::passes::clight_select::select::SelectedFunction;

pub const SOURCE_ALTERNATIVES_SCHEMA: &str = "manifold-source-alternatives-v1";
pub const MAX_SOURCE_ALTERNATIVES_PER_FUNCTION: usize = 2;
pub const MAX_TOTAL_SOURCE_ALTERNATIVES: usize = 4096;
pub const MAX_SOURCE_ALTERNATIVE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SourceAlternativeBoundary {
    PreForLoop,
    PreVarReduce,
}

impl SourceAlternativeBoundary {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::PreForLoop => "pre_forloop",
            Self::PreVarReduce => "pre_var_reduce",
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
    }
    kinds.sort_unstable();
    kinds.dedup();
    Some(SourceAlternativeSnapshot {
        declaration_index,
        manifold_name: before.name.clone(),
        manifold_address,
        boundary,
        kinds,
        function: before.clone(),
    })
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

fn append_digest_field(material: &mut Vec<u8>, field: &[u8]) {
    material.extend_from_slice(&(field.len() as u64).to_be_bytes());
    material.extend_from_slice(field);
}

/// Hash the complete ordered identity/source-digest set with unambiguous
/// length-prefix framing.  The source text itself is separately authenticated by
/// each record's digest and by the provider artifact bundle.
fn ordered_set_sha256(
    canonical_translation_unit_sha256: &str,
    records: &[SourceAlternativeRecord],
) -> String {
    let mut material = Vec::new();
    append_digest_field(&mut material, SOURCE_ALTERNATIVES_SCHEMA.as_bytes());
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
    exact_function_identities: &[(String, u64)],
    canonical_translation_unit_source: &str,
    format: BinaryFormat,
) -> serde_json::Result<Option<String>> {
    render_manifest_with_limits(
        final_tu,
        snapshots,
        capture_overflowed,
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
    exact_function_identities: &[(String, u64)],
    canonical_translation_unit_source: &str,
    format: BinaryFormat,
    max_total: usize,
    max_source_bytes: usize,
    max_manifest_bytes: usize,
) -> serde_json::Result<Option<String>> {
    if capture_overflowed {
        return Ok(None);
    }
    let reproduced_source =
        crate::decompile::passes::c_pass::print_translation_unit_for_format(final_tu, format);
    if reproduced_source != canonical_translation_unit_source {
        return Err(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "canonical optimized translation unit changed before alternative emission",
        )));
    }
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
    let mut ordered: Vec<(usize, &SourceAlternativeSnapshot)> = snapshots
        .iter()
        .filter_map(|snapshot| {
            wire_ordinal_by_declaration
                .get(&snapshot.declaration_index)
                .copied()
                .map(|function_ordinal| (function_ordinal, snapshot))
        })
        .collect();
    ordered.sort_by_key(|(function_ordinal, snapshot)| (*function_ordinal, snapshot.boundary));
    let mut snapshot_key_counts = std::collections::HashMap::new();
    for (_, snapshot) in &ordered {
        *snapshot_key_counts
            .entry((snapshot.declaration_index, snapshot.boundary))
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
        if count_for_function >= MAX_SOURCE_ALTERNATIVES_PER_FUNCTION {
            continue;
        }
        if alternatives.len() >= max_total {
            return Ok(None);
        }
        if snapshot_key_counts
            .get(&(snapshot.declaration_index, snapshot.boundary))
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
            return Ok(None);
        }
        source_bytes += candidate_bytes;
        let canonical_source_sha256 =
            crate::decompile::disassembly::coff::sha256_hex(canonical_source.as_bytes());
        let alternative_source_sha256 =
            crate::decompile::disassembly::coff::sha256_hex(alternative_source.as_bytes());
        alternatives.push(SourceAlternativeRecord {
            id: format!(
                "function-{:06}:{}",
                function_ordinal,
                snapshot.boundary.wire_name(),
            ),
            function_ordinal: *function_ordinal,
            manifold_name: snapshot.manifold_name.clone(),
            manifold_address: format!("0x{:x}", snapshot.manifold_address),
            boundary: snapshot.boundary.wire_name(),
            kinds: snapshot.kinds.clone(),
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
    let ordered_set_sha256 = ordered_set_sha256(&canonical_translation_unit_sha256, &alternatives);
    let rendered = serde_json::to_string_pretty(&SourceAlternativeManifest {
        schema: SOURCE_ALTERNATIVES_SCHEMA,
        max_per_function: MAX_SOURCE_ALTERNATIVES_PER_FUNCTION,
        max_total: MAX_TOTAL_SOURCE_ALTERNATIVES,
        max_source_bytes: MAX_SOURCE_ALTERNATIVE_BYTES,
        max_manifest_bytes: MAX_SOURCE_ALTERNATIVE_MANIFEST_BYTES,
        truncated: false,
        canonical_translation_unit_sha256,
        ordered_set_sha256,
        alternatives,
    })?;
    if rendered.len() > max_manifest_bytes {
        return Ok(None);
    }
    Ok(Some(rendered))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::passes::c_pass::types::{
        BinaryOp, CType, FuncParam, SourceLoc, StorageClass,
    };

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
        assert_eq!(parsed["max_per_function"], 2);
        assert_eq!(parsed["truncated"], false);
        assert_eq!(final_tu, final_tu_before);
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
            function: wrong_signature,
        };
        let rendered = render_manifest(
            &final_tu,
            &[snapshot.clone()],
            false,
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
