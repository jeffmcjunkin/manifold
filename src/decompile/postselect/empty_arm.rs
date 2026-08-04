//! Final empty-conditional-arm canonicalization on the selected C AST.
//!
//! Earlier passes may leave an empty preferred arm, and variable coalescing can
//! create new empty arms when it removes identity copies.  Normalize only the
//! final syntax shape here, after those transformations have finished:
//!
//! `if (condition) { } else { body }` becomes `if (!condition) { body }`.
//!
//! The condition is wrapped in a literal logical-not expression.  In
//! particular, this pass never flips comparison operators: `!(a < b)` is not
//! interchangeable with `a >= b` for floating-point unordered values.

use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::c_pass::types::*;
use crate::decompile::passes::pass::IRPass;

#[derive(Default)]
struct CanonicalizeStats {
    inverted: usize,
    removed_empty_else: usize,
}

/// Empty for arm-selection purposes.  Declarations and labels are deliberately
/// structural: even a label whose body is empty must remain a possible goto
/// target, and a declaration-only block must retain its scope.
fn is_structurally_empty(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Empty => true,
        CStmt::Block(items) => items.iter().all(|item| match item {
            CBlockItem::Stmt(stmt) => is_structurally_empty(stmt),
            CBlockItem::Decl(_) => false,
        }),
        CStmt::Sequence(stmts) => stmts.iter().all(is_structurally_empty),
        _ => false,
    }
}

/// Preserve the original condition as exactly one operand evaluation.  Do not
/// share this with readability helpers that rewrite comparison operators.
fn literal_logical_not(condition: &CExpr) -> CExpr {
    CExpr::Unary(UnaryOp::Not, Box::new(condition.clone()))
}

struct EmptyArmCanonicalizer<'a> {
    stats: &'a mut CanonicalizeStats,
}

impl EmptyArmCanonicalizer<'_> {
    fn transform_designator(&mut self, designator: Designator) -> Designator {
        match designator {
            Designator::Field(field) => Designator::Field(field),
            Designator::Index(index) => Designator::Index(self.transform_expr(index)),
            Designator::Range(start, end) => {
                Designator::Range(self.transform_expr(start), self.transform_expr(end))
            }
        }
    }

    fn transform_var_decl(&mut self, mut decl: VarDecl) -> VarDecl {
        decl.init = decl
            .init
            .map(|initializer| self.transform_initializer(initializer));
        decl
    }
}

impl ExprTransform for EmptyArmCanonicalizer<'_> {
    fn transform_initializer(&mut self, initializer: Initializer) -> Initializer {
        match initializer {
            Initializer::Expr(expr) => Initializer::Expr(self.transform_expr(expr)),
            Initializer::List(items) => Initializer::List(
                items
                    .into_iter()
                    .map(|item| InitItem {
                        designator: item
                            .designator
                            .map(|designator| self.transform_designator(designator)),
                        init: self.transform_initializer(item.init),
                    })
                    .collect(),
            ),
            Initializer::String(literal) => Initializer::String(literal),
        }
    }

    fn transform_stmt_in_expr(&mut self, stmt: CStmt) -> CStmt {
        StmtTransform::transform_stmt(self, stmt)
    }
}

impl StmtTransform for EmptyArmCanonicalizer<'_> {
    fn transform_stmt(&mut self, stmt: CStmt) -> CStmt {
        let stmt = self.walk_stmt(stmt);
        let CStmt::If(condition, then_stmt, else_stmt) = stmt else {
            return stmt;
        };

        match else_stmt {
            Some(else_stmt) if is_structurally_empty(&else_stmt) => {
                self.stats.removed_empty_else += 1;
                CStmt::If(condition, then_stmt, None)
            }
            Some(else_stmt) if is_structurally_empty(&then_stmt) => {
                self.stats.inverted += 1;
                CStmt::If(literal_logical_not(&condition), else_stmt, None)
            }
            Some(else_stmt) => CStmt::If(condition, then_stmt, Some(else_stmt)),
            None => CStmt::If(condition, then_stmt, None),
        }
    }
}

#[cfg(test)]
fn canonicalize_stmt(stmt: &CStmt, stats: &mut CanonicalizeStats) -> CStmt {
    let mut canonicalizer = EmptyArmCanonicalizer { stats };
    StmtTransform::transform_stmt(&mut canonicalizer, stmt.clone())
}

pub struct EmptyArmCanonicalizerPass;

impl IRPass for EmptyArmCanonicalizerPass {
    fn name(&self) -> &'static str {
        "empty_arm_canonicalizer"
    }

    fn run(&self, db: &mut DecompileDB) {
        let Some(tu) = db.cast_optimized_translation_unit.as_mut() else {
            return;
        };

        let mut stats = CanonicalizeStats::default();
        let mut changed_functions = 0usize;
        let mut changed_declarations = 0usize;
        let mut canonicalizer = EmptyArmCanonicalizer { stats: &mut stats };
        for decl in &mut tu.decls {
            match decl {
                TopLevelDecl::FuncDef(function) => {
                    let body =
                        StmtTransform::transform_stmt(&mut canonicalizer, function.body.clone());
                    let local_vars = function
                        .local_vars
                        .iter()
                        .cloned()
                        .map(|decl| canonicalizer.transform_var_decl(decl))
                        .collect::<Vec<_>>();
                    if body != function.body || local_vars != function.local_vars {
                        function.body = body;
                        function.local_vars = local_vars;
                        changed_functions += 1;
                    }
                }
                TopLevelDecl::VarDecl(var) => {
                    let transformed = canonicalizer.transform_var_decl(var.clone());
                    if transformed != *var {
                        *var = transformed;
                        changed_declarations += 1;
                    }
                }
                TopLevelDecl::EnumDef(enum_def) => {
                    let mut changed = false;
                    for constant in &mut enum_def.constants {
                        if let Some(value) = constant.value.take() {
                            let transformed = canonicalizer.transform_expr(value.clone());
                            changed |= transformed != value;
                            constant.value = Some(transformed);
                        }
                    }
                    if changed {
                        changed_declarations += 1;
                    }
                }
                TopLevelDecl::FuncDecl(_)
                | TopLevelDecl::StructDef(_)
                | TopLevelDecl::Typedef(_) => {}
            }
        }
        drop(canonicalizer);

        log::info!(
            "empty_arm_canonicalizer: changed {} funcs and {} non-function declarations, inverted {} empty-then arms, removed {} empty else arms",
            changed_functions,
            changed_declarations,
            stats.inverted,
            stats.removed_empty_else,
        );
    }

    fn inputs(&self) -> &'static [&'static str] {
        &["cast_optimized_translation_unit"]
    }

    fn outputs(&self) -> &'static [&'static str] {
        &["cast_optimized_translation_unit"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::passes::c_pass::print::print_stmt;
    use crate::decompile::passes::pass::PassScheduler;
    use crate::decompile::postselect::var_reduce::VarReducePass;

    fn expr_call(name: &str) -> CStmt {
        CStmt::Expr(CExpr::call(CExpr::var(name), vec![]))
    }

    fn nonempty_block(stmt: CStmt) -> CStmt {
        CStmt::Block(vec![CBlockItem::Stmt(stmt)])
    }

    fn canonicalize(stmt: &CStmt) -> CStmt {
        canonicalize_stmt(stmt, &mut CanonicalizeStats::default())
    }

    fn function(body: CStmt) -> FuncDef {
        FuncDef {
            name: "fixture".into(),
            return_type: CType::int(),
            params: vec![],
            is_variadic: false,
            storage_class: StorageClass::Auto,
            body,
            local_vars: vec![],
            loc: SourceLoc::unknown(),
        }
    }

    fn invertible_if(name: &str) -> CStmt {
        CStmt::If(
            CExpr::var(format!("{name}_condition")),
            Box::new(CStmt::Empty),
            Some(Box::new(expr_call(&format!("{name}_work")))),
        )
    }

    fn nested_stmt_expr(name: &str) -> CExpr {
        CExpr::StmtExpr(
            vec![
                expr_call(&format!("{name}_before")),
                invertible_if(name),
                expr_call(&format!("{name}_after")),
            ],
            Box::new(CExpr::call(CExpr::var(format!("{name}_tail")), vec![])),
        )
    }

    fn assert_inverted(stmt: &CStmt, name: &str) {
        assert!(
            matches!(
                stmt,
                CStmt::If(
                    CExpr::Unary(UnaryOp::Not, condition),
                    _,
                    None,
                ) if condition.as_ref() == &CExpr::var(format!("{name}_condition"))
            ),
            "{name}: {stmt:?}"
        );
    }

    fn assert_canonical_designator(designator: &Designator) {
        match designator {
            Designator::Field(_) => {}
            Designator::Index(index) => assert_canonical_expr(index),
            Designator::Range(start, end) => {
                assert_canonical_expr(start);
                assert_canonical_expr(end);
            }
        }
    }

    fn assert_canonical_initializer(initializer: &Initializer) {
        match initializer {
            Initializer::Expr(expr) => assert_canonical_expr(expr),
            Initializer::List(items) => {
                for item in items {
                    if let Some(designator) = &item.designator {
                        assert_canonical_designator(designator);
                    }
                    assert_canonical_initializer(&item.init);
                }
            }
            Initializer::String(_) => {}
        }
    }

    fn assert_canonical_decl(decl: &VarDecl) {
        if let Some(initializer) = &decl.init {
            assert_canonical_initializer(initializer);
        }
    }

    fn assert_canonical_expr(expr: &CExpr) {
        match expr {
            CExpr::IntLit(_)
            | CExpr::FloatLit(_)
            | CExpr::StringLit(_)
            | CExpr::CharLit(_)
            | CExpr::Var(_)
            | CExpr::SizeofType(_)
            | CExpr::AlignofType(_) => {}
            CExpr::Unary(_, inner)
            | CExpr::Cast(_, inner)
            | CExpr::Member(inner, _)
            | CExpr::MemberPtr(inner, _)
            | CExpr::SizeofExpr(inner)
            | CExpr::Paren(inner) => assert_canonical_expr(inner),
            CExpr::Binary(_, left, right) | CExpr::Assign(_, left, right) => {
                assert_canonical_expr(left);
                assert_canonical_expr(right);
            }
            CExpr::Ternary(condition, then_expr, else_expr) => {
                assert_canonical_expr(condition);
                assert_canonical_expr(then_expr);
                assert_canonical_expr(else_expr);
            }
            CExpr::Call(callee, args) => {
                assert_canonical_expr(callee);
                args.iter().for_each(assert_canonical_expr);
            }
            CExpr::Index(base, index) => {
                assert_canonical_expr(base);
                assert_canonical_expr(index);
            }
            CExpr::CompoundLit(_, initializers) => {
                initializers.iter().for_each(assert_canonical_initializer);
            }
            CExpr::Generic(control, associations) => {
                assert_canonical_expr(control);
                for (_, association) in associations {
                    assert_canonical_expr(association);
                }
            }
            CExpr::StmtExpr(stmts, tail) => {
                stmts.iter().for_each(assert_canonical_stmt);
                assert_canonical_expr(tail);
            }
        }
    }

    fn assert_canonical_stmt(stmt: &CStmt) {
        match stmt {
            CStmt::Empty | CStmt::Goto(_) | CStmt::Continue | CStmt::Break => {}
            CStmt::Expr(expr) => assert_canonical_expr(expr),
            CStmt::Block(items) => {
                for item in items {
                    match item {
                        CBlockItem::Stmt(stmt) => assert_canonical_stmt(stmt),
                        CBlockItem::Decl(decls) => decls.iter().for_each(assert_canonical_decl),
                    }
                }
            }
            CStmt::If(condition, then_stmt, else_stmt) => {
                assert_canonical_expr(condition);
                if let Some(else_stmt) = else_stmt {
                    assert!(
                        !is_structurally_empty(then_stmt) && !is_structurally_empty(else_stmt),
                        "noncanonical empty conditional arm: {stmt:?}"
                    );
                }
                assert_canonical_stmt(then_stmt);
                if let Some(else_stmt) = else_stmt {
                    assert_canonical_stmt(else_stmt);
                }
            }
            CStmt::Switch(expr, body) | CStmt::While(expr, body) => {
                assert_canonical_expr(expr);
                assert_canonical_stmt(body);
            }
            CStmt::DoWhile(body, condition) => {
                assert_canonical_stmt(body);
                assert_canonical_expr(condition);
            }
            CStmt::For(init, condition, update, body) => {
                if let Some(init) = init {
                    match init {
                        ForInit::Expr(expr) => assert_canonical_expr(expr),
                        ForInit::Decl(decls) => decls.iter().for_each(assert_canonical_decl),
                    }
                }
                if let Some(condition) = condition {
                    assert_canonical_expr(condition);
                }
                if let Some(update) = update {
                    assert_canonical_expr(update);
                }
                assert_canonical_stmt(body);
            }
            CStmt::Return(expr) => {
                if let Some(expr) = expr {
                    assert_canonical_expr(expr);
                }
            }
            CStmt::Labeled(label, body) => {
                if let Label::Case(expr) = label {
                    assert_canonical_expr(expr);
                }
                assert_canonical_stmt(body);
            }
            CStmt::Decl(decls) => decls.iter().for_each(assert_canonical_decl),
            CStmt::Sequence(stmts) => stmts.iter().for_each(assert_canonical_stmt),
        }
    }

    #[test]
    fn side_effecting_condition_is_inverted_once_and_rewrite_is_idempotent() {
        let condition = CExpr::call(CExpr::var("probe"), vec![]);
        let input = CStmt::If(
            condition.clone(),
            Box::new(CStmt::Block(vec![])),
            Some(Box::new(nonempty_block(expr_call("work")))),
        );

        let output = canonicalize(&input);
        assert_eq!(canonicalize(&output), output);
        match &output {
            CStmt::If(CExpr::Unary(UnaryOp::Not, inner), _, None) => {
                assert_eq!(inner.as_ref(), &condition);
            }
            other => panic!("expected a literal logical-not condition: {other:?}"),
        }
        let printed = print_stmt(&output);
        assert_eq!(printed.matches("probe(").count(), 1, "{printed}");
        assert_eq!(printed.matches("work(").count(), 1, "{printed}");
    }

    #[test]
    fn floating_relational_condition_is_wrapped_not_operator_flipped() {
        // `NaN < 0.0` is unordered.  The inverse is literally
        // `!(NaN < 0.0)`, never `NaN >= 0.0`.
        let condition = CExpr::Binary(
            BinaryOp::Lt,
            Box::new(CExpr::FloatLit(FloatLiteral {
                value: f64::NAN,
                suffix: FloatLiteralSuffix::None,
            })),
            Box::new(CExpr::FloatLit(FloatLiteral {
                value: 0.0,
                suffix: FloatLiteralSuffix::None,
            })),
        );
        let input = CStmt::If(
            condition.clone(),
            Box::new(CStmt::Empty),
            Some(Box::new(expr_call("unordered_or_greater"))),
        );

        let output = canonicalize(&input);
        assert!(matches!(
            output,
            CStmt::If(CExpr::Unary(UnaryOp::Not, inner), _, None)
                if inner.as_ref() == &condition
        ));
    }

    #[test]
    fn empty_else_is_removed_without_changing_the_condition() {
        let condition = CExpr::call(CExpr::var("probe"), vec![]);
        let input = CStmt::If(
            condition.clone(),
            Box::new(expr_call("work")),
            Some(Box::new(CStmt::Block(vec![CBlockItem::Stmt(CStmt::Empty)]))),
        );

        assert_eq!(
            canonicalize(&input),
            CStmt::If(condition, Box::new(expr_call("work")), None),
        );
    }

    #[test]
    fn labels_and_declaration_blocks_are_not_treated_as_empty() {
        let labeled_then = CStmt::Labeled(
            Label::Named("landing".into()),
            Box::new(CStmt::Sequence(vec![CStmt::Empty])),
        );
        let labeled_input = CStmt::If(
            CExpr::var("condition"),
            Box::new(labeled_then.clone()),
            Some(Box::new(expr_call("work"))),
        );
        assert_eq!(canonicalize(&labeled_input), labeled_input);

        let declaration = VarDecl::new("kept", CType::int());
        let declaration_block = CStmt::Block(vec![CBlockItem::Decl(vec![declaration])]);
        let declaration_input = CStmt::If(
            CExpr::var("condition"),
            Box::new(CStmt::Empty),
            Some(Box::new(declaration_block.clone())),
        );
        assert_eq!(
            canonicalize(&declaration_input),
            CStmt::If(
                CExpr::Unary(UnaryOp::Not, Box::new(CExpr::var("condition"))),
                Box::new(declaration_block),
                None,
            ),
        );
    }

    #[test]
    fn every_body_bearing_statement_is_walked_and_retained() {
        let input = CStmt::Sequence(vec![
            invertible_if("direct_if"),
            CStmt::Switch(
                CExpr::var("selector"),
                Box::new(invertible_if("switch_body")),
            ),
            CStmt::While(
                CExpr::var("while_condition"),
                Box::new(invertible_if("while_body")),
            ),
            CStmt::DoWhile(
                Box::new(invertible_if("do_body")),
                CExpr::var("do_condition"),
            ),
            CStmt::For(
                Some(ForInit::Expr(CExpr::int(0))),
                Some(CExpr::var("for_condition")),
                Some(CExpr::var("for_update")),
                Box::new(invertible_if("for_body")),
            ),
            CStmt::Labeled(
                Label::Named("kept_label".into()),
                Box::new(invertible_if("labeled_body")),
            ),
        ]);

        let output = canonicalize(&input);
        assert_eq!(canonicalize(&output), output);
        let CStmt::Sequence(stmts) = &output else {
            panic!("expected retained sequence: {output:?}");
        };
        assert_eq!(stmts.len(), 6);
        assert_inverted(&stmts[0], "direct_if");
        match &stmts[1] {
            CStmt::Switch(selector, body) => {
                assert_eq!(selector, &CExpr::var("selector"));
                assert_inverted(body, "switch_body");
            }
            other => panic!("expected switch: {other:?}"),
        }
        match &stmts[2] {
            CStmt::While(condition, body) => {
                assert_eq!(condition, &CExpr::var("while_condition"));
                assert_inverted(body, "while_body");
            }
            other => panic!("expected while: {other:?}"),
        }
        match &stmts[3] {
            CStmt::DoWhile(body, condition) => {
                assert_inverted(body, "do_body");
                assert_eq!(condition, &CExpr::var("do_condition"));
            }
            other => panic!("expected do-while: {other:?}"),
        }
        match &stmts[4] {
            CStmt::For(init, condition, update, body) => {
                assert_eq!(init, &Some(ForInit::Expr(CExpr::int(0))));
                assert_eq!(condition, &Some(CExpr::var("for_condition")));
                assert_eq!(update, &Some(CExpr::var("for_update")));
                assert_inverted(body, "for_body");
            }
            other => panic!("expected for: {other:?}"),
        }
        match &stmts[5] {
            CStmt::Labeled(Label::Named(label), body) => {
                assert_eq!(label, "kept_label");
                assert_inverted(body, "labeled_body");
            }
            other => panic!("expected named label: {other:?}"),
        }
        assert_canonical_stmt(&output);
    }

    #[test]
    fn expression_initializer_and_designator_walk_is_exhaustive_and_idempotent() {
        let expressions = vec![
            CExpr::Unary(UnaryOp::Plus, Box::new(nested_stmt_expr("unary"))),
            CExpr::Binary(
                BinaryOp::Add,
                Box::new(nested_stmt_expr("binary_left")),
                Box::new(nested_stmt_expr("binary_right")),
            ),
            CExpr::Assign(
                AssignOp::Assign,
                Box::new(CExpr::var("assignment_target")),
                Box::new(nested_stmt_expr("assignment_value")),
            ),
            CExpr::Ternary(
                Box::new(nested_stmt_expr("ternary_condition")),
                Box::new(nested_stmt_expr("ternary_then")),
                Box::new(nested_stmt_expr("ternary_else")),
            ),
            CExpr::Call(
                Box::new(nested_stmt_expr("callee")),
                vec![nested_stmt_expr("call_argument")],
            ),
            CExpr::Cast(CType::int(), Box::new(nested_stmt_expr("cast"))),
            CExpr::Member(Box::new(nested_stmt_expr("member")), "field".into()),
            CExpr::MemberPtr(Box::new(nested_stmt_expr("member_pointer")), "field".into()),
            CExpr::Index(
                Box::new(nested_stmt_expr("index_base")),
                Box::new(nested_stmt_expr("index_value")),
            ),
            CExpr::SizeofExpr(Box::new(nested_stmt_expr("sizeof"))),
            CExpr::CompoundLit(
                CType::int(),
                vec![
                    Initializer::Expr(nested_stmt_expr("compound_expr")),
                    Initializer::List(vec![
                        InitItem {
                            designator: Some(Designator::Field("field".into())),
                            init: Initializer::Expr(nested_stmt_expr("field_init")),
                        },
                        InitItem {
                            designator: Some(Designator::Index(nested_stmt_expr("compound_index"))),
                            init: Initializer::Expr(nested_stmt_expr("indexed_init")),
                        },
                        InitItem {
                            designator: Some(Designator::Range(
                                nested_stmt_expr("range_start"),
                                nested_stmt_expr("range_end"),
                            )),
                            init: Initializer::String(StringLiteral {
                                value: "kept".into(),
                                is_wide: false,
                            }),
                        },
                    ]),
                ],
            ),
            CExpr::Generic(
                Box::new(nested_stmt_expr("generic_control")),
                vec![(Some(CType::int()), nested_stmt_expr("generic_arm"))],
            ),
            CExpr::Paren(Box::new(nested_stmt_expr("paren"))),
            CExpr::StmtExpr(
                vec![
                    expr_call("outer_before"),
                    invertible_if("outer_stmt_expr"),
                    expr_call("outer_after"),
                ],
                Box::new(nested_stmt_expr("stmt_expr_tail")),
            ),
        ];

        let declaration =
            VarDecl::new("with_initializer", CType::int()).with_init(Initializer::List(vec![
                InitItem {
                    designator: Some(Designator::Index(nested_stmt_expr("decl_index"))),
                    init: Initializer::Expr(nested_stmt_expr("decl_value")),
                },
            ]));
        let standalone_declaration = VarDecl::new("standalone", CType::int())
            .with_init(Initializer::Expr(nested_stmt_expr("standalone_decl")));
        let for_declaration = VarDecl::new("loop_decl", CType::int())
            .with_init(Initializer::Expr(nested_stmt_expr("for_decl")));
        let mut items = vec![
            CBlockItem::Decl(vec![declaration]),
            CBlockItem::Stmt(CStmt::Decl(vec![standalone_declaration])),
        ];
        items.extend(
            expressions
                .into_iter()
                .map(|expr| CBlockItem::Stmt(CStmt::Expr(expr))),
        );
        items.extend([
            CBlockItem::Stmt(CStmt::Return(Some(nested_stmt_expr("return")))),
            CBlockItem::Stmt(CStmt::If(
                nested_stmt_expr("if_condition_expr"),
                Box::new(CStmt::Empty),
                None,
            )),
            CBlockItem::Stmt(CStmt::Switch(
                nested_stmt_expr("switch_expr"),
                Box::new(CStmt::Empty),
            )),
            CBlockItem::Stmt(CStmt::While(
                nested_stmt_expr("while_expr"),
                Box::new(CStmt::Empty),
            )),
            CBlockItem::Stmt(CStmt::DoWhile(
                Box::new(CStmt::Empty),
                nested_stmt_expr("do_expr"),
            )),
            CBlockItem::Stmt(CStmt::For(
                Some(ForInit::Expr(nested_stmt_expr("for_init"))),
                Some(nested_stmt_expr("for_condition_expr")),
                Some(nested_stmt_expr("for_update_expr")),
                Box::new(CStmt::Empty),
            )),
            CBlockItem::Stmt(CStmt::For(
                Some(ForInit::Decl(vec![for_declaration])),
                None,
                None,
                Box::new(CStmt::Empty),
            )),
            CBlockItem::Stmt(CStmt::Labeled(
                Label::Case(nested_stmt_expr("case_expr")),
                Box::new(CStmt::Empty),
            )),
        ]);
        let input = CStmt::Block(items);

        let output = canonicalize(&input);
        assert_eq!(canonicalize(&output), output);
        assert_canonical_stmt(&output);

        let CStmt::Block(items) = &output else {
            panic!("expected retained block: {output:?}");
        };
        assert!(matches!(items.first(), Some(CBlockItem::Decl(decls)) if decls.len() == 1));
        let outer_stmt_expr = items.iter().find_map(|item| match item {
            CBlockItem::Stmt(CStmt::Expr(CExpr::StmtExpr(stmts, _))) => Some(stmts),
            _ => None,
        });
        let stmts = outer_stmt_expr.expect("retained outer statement expression");
        assert_eq!(stmts.len(), 3);
        assert_eq!(stmts[0], expr_call("outer_before"));
        assert_inverted(&stmts[1], "outer_stmt_expr");
        assert_eq!(stmts[2], expr_call("outer_after"));
    }

    #[test]
    fn pass_walks_function_local_global_and_enum_expression_containers() {
        let mut fixture = function(CStmt::Empty);
        fixture.local_vars.push(
            VarDecl::new("local", CType::int())
                .with_init(Initializer::Expr(nested_stmt_expr("local_initializer"))),
        );

        let mut tu = TranslationUnit::new();
        tu.add_function(fixture);
        tu.add_global_var(
            VarDecl::new("global", CType::int())
                .with_init(Initializer::Expr(nested_stmt_expr("global_initializer"))),
        );
        tu.decls.push(TopLevelDecl::EnumDef(EnumDef {
            name: Some("fixture_enum".into()),
            constants: vec![EnumConst {
                name: "FIXTURE_VALUE".into(),
                value: Some(nested_stmt_expr("enum_value")),
            }],
            loc: SourceLoc::unknown(),
        }));
        let mut db = DecompileDB::default();
        db.cast_optimized_translation_unit = Some(tu);

        EmptyArmCanonicalizerPass.run(&mut db);
        let optimized = db
            .cast_optimized_translation_unit
            .as_ref()
            .expect("pass must retain the optimized TU");
        match &optimized.decls[0] {
            TopLevelDecl::FuncDef(function) => {
                function.local_vars.iter().for_each(assert_canonical_decl)
            }
            other => panic!("expected function definition: {other:?}"),
        }
        match &optimized.decls[1] {
            TopLevelDecl::VarDecl(var) => assert_canonical_decl(var),
            other => panic!("expected global declaration: {other:?}"),
        }
        match &optimized.decls[2] {
            TopLevelDecl::EnumDef(enum_def) => {
                assert_canonical_expr(
                    enum_def.constants[0]
                        .value
                        .as_ref()
                        .expect("retained enum value"),
                );
            }
            other => panic!("expected enum definition: {other:?}"),
        }

        let once = optimized.clone();
        EmptyArmCanonicalizerPass.run(&mut db);
        assert_eq!(db.cast_optimized_translation_unit.as_ref(), Some(&once));
    }

    #[test]
    fn pass_rewrites_the_optimized_tu_and_scheduler_orders_it_after_var_reduce() {
        let mut tu = TranslationUnit::new();
        tu.add_function(function(CStmt::If(
            CExpr::var("condition"),
            Box::new(CStmt::Empty),
            Some(Box::new(expr_call("work"))),
        )));
        let mut db = DecompileDB::default();
        db.cast_optimized_translation_unit = Some(tu);

        VarReducePass.run(&mut db);
        EmptyArmCanonicalizerPass.run(&mut db);
        let body = match &db
            .cast_optimized_translation_unit
            .as_ref()
            .expect("pass must retain the optimized TU")
            .decls[0]
        {
            TopLevelDecl::FuncDef(function) => &function.body,
            other => panic!("expected function definition: {other:?}"),
        };
        assert!(matches!(
            body,
            CStmt::If(CExpr::Unary(UnaryOp::Not, _), _, None)
        ));

        let passes: Vec<Box<dyn IRPass>> =
            vec![Box::new(VarReducePass), Box::new(EmptyArmCanonicalizerPass)];
        let schedule = PassScheduler::build_schedule(&passes);
        assert_eq!(schedule.stages.len(), 2);
        assert_eq!(schedule.stages[0].passes, vec![0]);
        assert_eq!(schedule.stages[1].passes, vec![1]);
    }
}
