use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::c_pass::print::{IntegerModel, PrintConfig, Printer};
use manifold::decompile::passes::c_pass::types::{
    AssignOp, BinaryOp, CBlockItem, CExpr, CStmt, CType, FuncDef, Initializer, IntSize, Signedness,
    TopLevelDecl, UnaryOp,
};
use manifold::decompile::postselect::source_alternatives::SourceAlternativeBoundary;
use manifold::x86::types::{
    Address, ClightBinaryOp, ClightExpr, ClightIntSize, ClightSignedness, ClightStmt, ClightType,
    Node, ScalarMemoryAccessProof, ScalarMemoryDirection, ScalarMemoryExtension, Symbol,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// Immutable stage-1 oracle. Generated independently with the exact provider
// bundle and fixture bytes below; the literal file is deliberately checked in
// so a patched run cannot manufacture its own canonical "golden".
const BASE718_PROVIDER_COMMIT: &str = "718dcc773f5629197b319ac0ffcd3eccc828273d";
const BASE718_PROVIDER_BINARY_SHA256: &str =
    "a84f4feddb0272cf5da0e243e3ba9443a5009f8516d438263ca305106be993c3";
const BASE718_PROVIDER_IDENTITY_SHA256: &str =
    "ce09ec2b11d3b2c49ece836a2122d75a3cc9c1a6f4f5b371247f4d1a30ee28fa";
const BASE718_FIXTURE_SHA256: &str =
    "092a47a170c973fc23d96923eec6ae92a848a66b5c69590f317e121b1cc062cb";
const BASE718_CANONICAL_SOURCE_SHA256: &str =
    "c8c17dc4ac091a31fc34079cb51fb9d974b497c3d4a55788d4345cd4bdbf36c0";
const BASE718_CANONICAL_SOURCE: &str = include_str!("fixtures/scalar_lvalue_base718.c");

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn sha256_file(path: &Path) -> String {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .unwrap_or_else(|error| panic!("run sha256sum over {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "sha256sum failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum emitted a digest")
        .to_string()
}

fn build_fixture() -> PathBuf {
    assert!(
        command_exists("clang"),
        "scalar-lvalue integration prerequisite missing: clang is unavailable"
    );
    let directory =
        std::env::temp_dir().join(format!("manifold_scalar_lvalue_{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create scalar-lvalue fixture directory");
    let source = directory.join("fixture.s");
    let object = directory.join("fixture.obj");
    std::fs::write(
        &source,
        r#"
        .section .text$scalar_lvalue_plain32,"xr"
        .globl scalar_lvalue_plain32
        .def scalar_lvalue_plain32; .scl 2; .type 32; .endef
scalar_lvalue_plain32:
        movl 12(%rcx,%rdx,2), %eax
        retq

        .section .text$scalar_lvalue_plain64,"xr"
        .globl scalar_lvalue_plain64
        .def scalar_lvalue_plain64; .scl 2; .type 32; .endef
scalar_lvalue_plain64:
        movq 8(%rcx,%rdx,4), %rax
        retq

        .section .text$scalar_lvalue_movzx_indexed,"xr"
        .globl scalar_lvalue_movzx_indexed
        .def scalar_lvalue_movzx_indexed; .scl 2; .type 32; .endef
scalar_lvalue_movzx_indexed:
        movzbl -3(%rcx,%rdx,4), %eax
        retq

        .section .text$scalar_lvalue_movsx64,"xr"
        .globl scalar_lvalue_movsx64
        .def scalar_lvalue_movsx64; .scl 2; .type 32; .endef
scalar_lvalue_movsx64:
        movsbq 5(%rcx,%rdx,2), %rax
        addq %r8, %rax
        retq

        .section .text$scalar_lvalue_store16,"xr"
        .globl scalar_lvalue_store16
        .def scalar_lvalue_store16; .scl 2; .type 32; .endef
scalar_lvalue_store16:
        movw %r8w, -6(%rcx,%rdx,4)
        xorl %eax, %eax
        retq

        .section .text$scalar_lvalue_inline_index,"xr"
        .globl scalar_lvalue_inline_index
        .def scalar_lvalue_inline_index; .scl 2; .type 32; .endef
scalar_lvalue_inline_index:
        leaq 24(%rcx), %r8
        movl (%r8,%rdx,4), %eax
        retq

        # These three controls are deliberately outside the sealed v1
        # descriptor. With no authenticated/cminor/final marker, their only
        # candidate path is the unchanged base-718dcc canonical path.
        .section .text$scalar_lvalue_base718_high8_control,"xr"
        .globl scalar_lvalue_base718_high8_control
        .def scalar_lvalue_base718_high8_control; .scl 2; .type 32; .endef
scalar_lvalue_base718_high8_control:
        movb %ah, 4(%rcx)
        xorl %eax, %eax
        retq

        .section .text$scalar_lvalue_base718_word_dest_control,"xr"
        .globl scalar_lvalue_base718_word_dest_control
        .def scalar_lvalue_base718_word_dest_control; .scl 2; .type 32; .endef
scalar_lvalue_base718_word_dest_control:
        movzbw 4(%rcx), %ax
        movzwl %ax, %eax
        retq

        .section .text$scalar_lvalue_base718_wide_use_control,"xr"
        .globl scalar_lvalue_base718_wide_use_control
        .def scalar_lvalue_base718_wide_use_control; .scl 2; .type 32; .endef
scalar_lvalue_base718_wide_use_control:
        movl (%rcx), %eax
        addq %rdx, %rax
        retq
"#,
    )
    .expect("write scalar-lvalue fixture assembly");
    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("run clang over scalar-lvalue fixture");
    assert!(status.success(), "scalar-lvalue fixture assembly failed");
    object
}

fn fixture_object() -> &'static Path {
    assert!(
        command_exists("clang"),
        "scalar-lvalue integration prerequisite missing: clang is unavailable"
    );
    static OBJECT: OnceLock<PathBuf> = OnceLock::new();
    OBJECT.get_or_init(build_fixture).as_path()
}

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    let coff_name = format!("coff_fn_{name}");
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .find_map(|(symbol, start, end)| {
            (*symbol == name || *symbol == coff_name).then_some((*start, *end))
        })
        .unwrap_or_else(|| panic!("fixture function {name} has no authenticated span"))
}

fn in_span(node: Node, span: (Address, Address)) -> bool {
    node >= span.0 && node < span.1
}

fn proofs_in_function(db: &DecompileDB, name: &str) -> Vec<ScalarMemoryAccessProof> {
    let span = function_span(db, name);
    db.rel_iter::<(Node, ScalarMemoryAccessProof)>("authenticated_scalar_memory_access")
        .filter(|(_, proof)| in_span(proof.origin_node, span))
        .map(|(_, proof)| proof.clone())
        .collect()
}

fn printed_function_definition<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    for candidate in [name.to_string(), format!("coff_fn_{name}")] {
        let needle = format!("{candidate}(");
        for (start, _) in text.match_indices(&needle) {
            let tail = &text[start..];
            let Some(open_brace) = tail.find('{') else {
                continue;
            };
            if tail
                .find(';')
                .is_some_and(|semicolon| semicolon < open_brace)
            {
                continue;
            }
            let end = tail.find("\n}\n").map_or(tail.len(), |end| end + 3);
            return Some(&tail[..end]);
        }
    }
    None
}

fn clight_scalar_type(ty: &ClightType) -> Option<(usize, ClightSignedness)> {
    match ty {
        ClightType::Tint(ClightIntSize::I8, signedness, _) => Some((1, *signedness)),
        ClightType::Tint(ClightIntSize::I16, signedness, _) => Some((2, *signedness)),
        ClightType::Tint(ClightIntSize::I32, signedness, _) => Some((4, *signedness)),
        ClightType::Tlong(signedness, _) => Some((8, *signedness)),
        _ => None,
    }
}

fn visit_clight_expr<'a>(expr: &'a ClightExpr, visit: &mut impl FnMut(&'a ClightExpr)) {
    visit(expr);
    match expr {
        ClightExpr::Ederef(inner, _)
        | ClightExpr::Eaddrof(inner, _)
        | ClightExpr::Eunop(_, inner, _)
        | ClightExpr::Ecast(inner, _) => visit_clight_expr(inner, visit),
        ClightExpr::Ebinop(_, left, right, _) => {
            visit_clight_expr(left, visit);
            visit_clight_expr(right, visit);
        }
        ClightExpr::Efield(base, _, _) => visit_clight_expr(base, visit),
        ClightExpr::Econdition(condition, if_true, if_false, _) => {
            visit_clight_expr(condition, visit);
            visit_clight_expr(if_true, visit);
            visit_clight_expr(if_false, visit);
        }
        ClightExpr::EconstInt(..)
        | ClightExpr::EconstFloat(..)
        | ClightExpr::EconstSingle(..)
        | ClightExpr::EconstLong(..)
        | ClightExpr::Evar(..)
        | ClightExpr::EvarSymbol(..)
        | ClightExpr::Etempvar(..)
        | ClightExpr::Esizeof(..)
        | ClightExpr::Ealignof(..) => {}
    }
}

fn clight_expr_has_constant(expr: &ClightExpr, expected: i64) -> bool {
    let mut found = false;
    visit_clight_expr(expr, &mut |candidate| {
        found |= matches!(candidate, ClightExpr::EconstInt(value, _) if i64::from(*value) == expected)
            || matches!(candidate, ClightExpr::EconstLong(value, _) if *value == expected);
    });
    found
}

fn clight_expr_has_scaled_mul(expr: &ClightExpr, expected: i64) -> bool {
    let mut found = false;
    visit_clight_expr(expr, &mut |candidate| {
        if let ClightExpr::Ebinop(ClightBinaryOp::Omul, left, right, _) = candidate {
            found |= clight_expr_has_constant(left, expected)
                || clight_expr_has_constant(right, expected);
        }
    });
    found
}

fn scalar_clight_expr<'a>(
    statement: &'a ClightStmt,
    proof: &ScalarMemoryAccessProof,
) -> Option<&'a ClightExpr> {
    match (proof.direction, statement) {
        (ScalarMemoryDirection::Read, ClightStmt::Sset(_, value)) => Some(value),
        (ScalarMemoryDirection::Write, ClightStmt::Sassign(lvalue, _)) => Some(lvalue),
        _ => None,
    }
}

fn clight_expr_has_access(expr: &ClightExpr, proof: &ScalarMemoryAccessProof) -> bool {
    let expected_signedness = match proof.extension {
        ScalarMemoryExtension::SignExtend => Some(ClightSignedness::Signed),
        ScalarMemoryExtension::ZeroExtend => Some(ClightSignedness::Unsigned),
        ScalarMemoryExtension::Plain => None,
    };
    let mut found = false;
    visit_clight_expr(expr, &mut |candidate| {
        if let ClightExpr::Ederef(_, access_type) = candidate {
            found |= clight_scalar_type(access_type).is_some_and(|(width, signedness)| {
                width == proof.width
                    && expected_signedness.map_or(true, |expected| expected == signedness)
            });
        }
    });
    found
}

fn assert_scalar_clight_candidate(db: &DecompileDB, name: &str, proof: &ScalarMemoryAccessProof) {
    let expected_form = if proof.exact_scaled_index {
        manifold::x86::types::ScalarLvalueSourceForm::TypedScaled
    } else {
        manifold::x86::types::ScalarLvalueSourceForm::RawByte
    };
    let tagged: Vec<_> = db
        .rel_iter::<(
            Node,
            manifold::x86::types::ScalarLvalueSourceForm,
            ClightStmt,
        )>("scalar_lvalue_source_candidate")
        .filter(|(node, _, _)| *node == proof.selected_node)
        .collect();
    assert!(
        tagged.iter().any(|(_, form, _)| *form == expected_form),
        "{name}: isolated candidate set lacks expected form {expected_form:?}: {tagged:#?}"
    );
    let candidates: Vec<&ClightStmt> = tagged
        .into_iter()
        .map(|(_, _, statement)| statement)
        .collect();
    assert!(
        !candidates.is_empty(),
        "{name}: no isolated feature Clight statements at proof node"
    );
    let matching: Vec<&ClightExpr> = candidates
        .iter()
        .filter_map(|statement| scalar_clight_expr(statement, proof))
        .filter(|expr| clight_expr_has_access(expr, proof))
        .collect();
    assert!(
        !matching.is_empty(),
        "{name}: no width/direction-compatible scalar Clight lvalue: {candidates:#?}"
    );
    assert_eq!(proof.extension, ScalarMemoryExtension::Plain);
    if proof.displacement != 0 {
        assert!(
            matching
                .iter()
                .any(|expr| clight_expr_has_constant(expr, proof.displacement)),
            "{name}: signed displacement {} was lost: {matching:#?}",
            proof.displacement
        );
    }
    if proof.index_value.is_some() && proof.scale != 1 {
        assert!(
            matching
                .iter()
                .any(|expr| clight_expr_has_scaled_mul(expr, proof.scale)),
            "{name}: no raw byte-address candidate preserves scale {}: {matching:#?}",
            proof.scale
        );
        if proof.exact_scaled_index {
            assert!(
                matching
                    .iter()
                    .any(|expr| !clight_expr_has_scaled_mul(expr, proof.scale)),
                "{name}: no direct typed scaled-index candidate accompanied the raw form"
            );
        }
    }
}

fn function_definition<'a>(
    translation_unit: &'a manifold::decompile::passes::c_pass::TranslationUnit,
    name: &str,
) -> &'a FuncDef {
    let names = [name.to_string(), format!("coff_fn_{name}")];
    translation_unit
        .decls
        .iter()
        .find_map(|decl| match decl {
            TopLevelDecl::FuncDef(function) if names.contains(&function.name) => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("final C AST lost function {name}"))
}

fn visit_c_expr<'a>(expr: &'a CExpr, visit: &mut impl FnMut(&'a CExpr)) {
    visit(expr);
    match expr {
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _)
        | CExpr::SizeofExpr(inner)
        | CExpr::Paren(inner) => visit_c_expr(inner, visit),
        CExpr::Binary(_, left, right)
        | CExpr::Assign(_, left, right)
        | CExpr::Index(left, right) => {
            visit_c_expr(left, visit);
            visit_c_expr(right, visit);
        }
        CExpr::Ternary(condition, if_true, if_false) => {
            visit_c_expr(condition, visit);
            visit_c_expr(if_true, visit);
            visit_c_expr(if_false, visit);
        }
        CExpr::Call(function, arguments) => {
            visit_c_expr(function, visit);
            for argument in arguments {
                visit_c_expr(argument, visit);
            }
        }
        CExpr::SizeofType(_)
        | CExpr::AlignofType(_)
        | CExpr::IntLit(_)
        | CExpr::FloatLit(_)
        | CExpr::StringLit(_)
        | CExpr::CharLit(_)
        | CExpr::Var(_) => {}
        CExpr::CompoundLit(_, initializers) => {
            for initializer in initializers {
                visit_c_initializer(initializer, visit);
            }
        }
        CExpr::Generic(control, associations) => {
            visit_c_expr(control, visit);
            for (_, expression) in associations {
                visit_c_expr(expression, visit);
            }
        }
        CExpr::StmtExpr(statements, value) => {
            for statement in statements {
                visit_c_stmt(statement, visit);
            }
            visit_c_expr(value, visit);
        }
    }
}

fn visit_c_initializer<'a>(initializer: &'a Initializer, visit: &mut impl FnMut(&'a CExpr)) {
    match initializer {
        Initializer::Expr(expr) => visit_c_expr(expr, visit),
        Initializer::List(items) => {
            for item in items {
                visit_c_initializer(&item.init, visit);
            }
        }
        Initializer::String(_) => {}
    }
}

fn visit_c_stmt<'a>(statement: &'a CStmt, visit: &mut impl FnMut(&'a CExpr)) {
    match statement {
        CStmt::Expr(expr) | CStmt::Return(Some(expr)) => visit_c_expr(expr, visit),
        CStmt::Block(items) => {
            for item in items {
                match item {
                    CBlockItem::Stmt(statement) => visit_c_stmt(statement, visit),
                    CBlockItem::Decl(declarations) => {
                        for declaration in declarations {
                            if let Some(initializer) = &declaration.init {
                                visit_c_initializer(initializer, visit);
                            }
                        }
                    }
                }
            }
        }
        CStmt::If(condition, if_true, if_false) => {
            visit_c_expr(condition, visit);
            visit_c_stmt(if_true, visit);
            if let Some(if_false) = if_false {
                visit_c_stmt(if_false, visit);
            }
        }
        CStmt::Switch(condition, body) | CStmt::While(condition, body) => {
            visit_c_expr(condition, visit);
            visit_c_stmt(body, visit);
        }
        CStmt::DoWhile(body, condition) => {
            visit_c_stmt(body, visit);
            visit_c_expr(condition, visit);
        }
        CStmt::For(initializer, condition, update, body) => {
            if let Some(initializer) = initializer {
                match initializer {
                    manifold::decompile::passes::c_pass::types::ForInit::Expr(expr) => {
                        visit_c_expr(expr, visit)
                    }
                    manifold::decompile::passes::c_pass::types::ForInit::Decl(declarations) => {
                        for declaration in declarations {
                            if let Some(initializer) = &declaration.init {
                                visit_c_initializer(initializer, visit);
                            }
                        }
                    }
                }
            }
            if let Some(condition) = condition {
                visit_c_expr(condition, visit);
            }
            if let Some(update) = update {
                visit_c_expr(update, visit);
            }
            visit_c_stmt(body, visit);
        }
        CStmt::Labeled(_, body) => visit_c_stmt(body, visit),
        CStmt::Decl(declarations) => {
            for declaration in declarations {
                if let Some(initializer) = &declaration.init {
                    visit_c_initializer(initializer, visit);
                }
            }
        }
        CStmt::Sequence(statements) => {
            for statement in statements {
                visit_c_stmt(statement, visit);
            }
        }
        CStmt::Empty | CStmt::Goto(_) | CStmt::Continue | CStmt::Break | CStmt::Return(None) => {}
    }
}

fn c_scalar_width(ty: &CType) -> Option<usize> {
    match ty {
        CType::Int(IntSize::Char, _) => Some(1),
        CType::Int(IntSize::Short, _) => Some(2),
        CType::Int(IntSize::Int, _) => Some(4),
        CType::Int(IntSize::Long | IntSize::LongLong, _) => Some(8),
        _ => None,
    }
}

fn unparen(expr: &CExpr) -> &CExpr {
    match expr {
        CExpr::Paren(inner) => unparen(inner),
        _ => expr,
    }
}

fn c_int_value(expr: &CExpr) -> Option<i64> {
    match unparen(expr) {
        CExpr::IntLit(value) => i64::try_from(value.value).ok(),
        _ => None,
    }
}

fn c_access_type_matches(ty: &CType, proof: &ScalarMemoryAccessProof) -> bool {
    if c_scalar_width(ty) != Some(proof.width) {
        return false;
    }
    match (proof.extension, ty) {
        (ScalarMemoryExtension::SignExtend, CType::Int(_, Signedness::Signed)) => true,
        (ScalarMemoryExtension::ZeroExtend, CType::Int(_, Signedness::Unsigned)) => true,
        (ScalarMemoryExtension::Plain, CType::Int(_, _)) => true,
        _ => false,
    }
}

fn cast_pointer_pointee(expr: &CExpr) -> Option<&CType> {
    match unparen(expr) {
        CExpr::Cast(CType::Pointer(pointee, _), _) => Some(pointee.as_ref()),
        _ => None,
    }
}

fn is_unsigned_byte_pointer_cast(expr: &CExpr) -> bool {
    matches!(
        cast_pointer_pointee(expr),
        Some(CType::Int(IntSize::Char, Signedness::Unsigned))
    )
}

fn raw_index_offset_matches(expr: &CExpr, proof: &ScalarMemoryAccessProof) -> bool {
    if proof.scale == 1 {
        return !matches!(unparen(expr), CExpr::IntLit(_));
    }
    let CExpr::Binary(BinaryOp::Mul, index, scale) = unparen(expr) else {
        return false;
    };
    c_int_value(scale) == Some(proof.scale) && !matches!(unparen(index), CExpr::IntLit(_))
}

fn raw_feature_lvalue_matches(expr: &CExpr, proof: &ScalarMemoryAccessProof) -> bool {
    if proof.address_size != 8 || proof.synthetic_stack_origin {
        return false;
    }
    let CExpr::Unary(UnaryOp::Deref, address) = unparen(expr) else {
        return false;
    };
    let CExpr::Cast(CType::Pointer(access_type, _), byte_address) = unparen(address) else {
        return false;
    };
    if !c_access_type_matches(access_type, proof) {
        return false;
    }
    let address_without_displacement = if proof.displacement == 0 {
        unparen(byte_address)
    } else {
        let CExpr::Binary(BinaryOp::Add, address, displacement) = unparen(byte_address) else {
            return false;
        };
        if c_int_value(displacement) != Some(proof.displacement) {
            return false;
        }
        unparen(address)
    };
    match proof.index_value {
        None => is_unsigned_byte_pointer_cast(address_without_displacement),
        Some(_) => {
            let CExpr::Binary(BinaryOp::Add, base, index_offset) = address_without_displacement
            else {
                return false;
            };
            is_unsigned_byte_pointer_cast(base) && raw_index_offset_matches(index_offset, proof)
        }
    }
}

fn typed_scaled_feature_lvalue_matches(expr: &CExpr, proof: &ScalarMemoryAccessProof) -> bool {
    if !proof.exact_scaled_index || proof.address_size != 8 || proof.index_value.is_none() {
        return false;
    }
    let CExpr::Unary(UnaryOp::Deref, address) = unparen(expr) else {
        return false;
    };
    let CExpr::Binary(BinaryOp::Add, base, index) = unparen(address) else {
        return false;
    };
    let Some(access_type) = cast_pointer_pointee(base) else {
        return false;
    };
    c_access_type_matches(access_type, proof)
        && !matches!(
            unparen(index),
            CExpr::IntLit(_) | CExpr::Binary(BinaryOp::Mul, _, _)
        )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FeatureLvalueSpelling {
    RawBytePointer,
    TypedScaledIndex,
}

fn feature_lvalue_spelling(
    expr: &CExpr,
    proof: &ScalarMemoryAccessProof,
) -> Option<FeatureLvalueSpelling> {
    if raw_feature_lvalue_matches(expr, proof) {
        Some(FeatureLvalueSpelling::RawBytePointer)
    } else if typed_scaled_feature_lvalue_matches(expr, proof) {
        Some(FeatureLvalueSpelling::TypedScaledIndex)
    } else {
        None
    }
}

fn c_extension_result_type(proof: &ScalarMemoryAccessProof) -> Option<CType> {
    let signedness = match proof.extension {
        ScalarMemoryExtension::SignExtend => Signedness::Signed,
        ScalarMemoryExtension::ZeroExtend => Signedness::Unsigned,
        ScalarMemoryExtension::Plain => return None,
    };
    match proof.value_width {
        4 => Some(CType::Int(IntSize::Int, signedness)),
        8 => Some(CType::Int(IntSize::Long, signedness)),
        _ => None,
    }
}

fn selected_feature_expression<'a>(
    expr: &'a CExpr,
    proof: &ScalarMemoryAccessProof,
) -> Option<(FeatureLvalueSpelling, &'a CExpr)> {
    match proof.direction {
        ScalarMemoryDirection::Read if proof.extension == ScalarMemoryExtension::Plain => {
            feature_lvalue_spelling(expr, proof).map(|spelling| (spelling, expr))
        }
        ScalarMemoryDirection::Read => {
            let CExpr::Cast(result_type, lvalue) = unparen(expr) else {
                return None;
            };
            if c_extension_result_type(proof).as_ref() != Some(result_type) {
                return None;
            }
            feature_lvalue_spelling(lvalue, proof).map(|spelling| (spelling, expr))
        }
        ScalarMemoryDirection::Write => {
            let CExpr::Assign(AssignOp::Assign, lvalue, _) = unparen(expr) else {
                return None;
            };
            feature_lvalue_spelling(lvalue, proof).map(|spelling| (spelling, lvalue.as_ref()))
        }
    }
}

fn print_c_expr(expr: &CExpr) -> String {
    let mut printer = Printer::new(PrintConfig {
        integer_model: IntegerModel::MsvcLlp64,
        ..PrintConfig::default()
    });
    printer.print_expr(expr);
    printer.into_string()
}

fn assert_selected_feature_lvalue(
    translation_unit: &manifold::decompile::passes::c_pass::TranslationUnit,
    source: &str,
    name: &str,
    proof: &ScalarMemoryAccessProof,
) {
    let function = function_definition(translation_unit, name);
    let definition = printed_function_definition(source, &function.name)
        .unwrap_or_else(|| panic!("printed source lost selected function {}", function.name));
    let mut spellings = BTreeSet::new();
    visit_c_stmt(&function.body, &mut |expr| {
        if let Some((kind, feature_expr)) = selected_feature_expression(expr, proof) {
            spellings.insert((kind, print_c_expr(feature_expr)));
        }
    });
    assert_eq!(
        spellings.len(),
        1,
        "{name}: selected C AST must contain exactly one sealed raw/scaled feature spelling: {spellings:#?}\n{definition}"
    );
    let (kind, spelling) = spellings.iter().next().expect("one feature lvalue");
    assert!(
        *kind == FeatureLvalueSpelling::RawBytePointer || proof.exact_scaled_index,
        "{name}: typed scaled spelling escaped its proof gate"
    );
    assert_eq!(
        definition.matches(spelling).count(),
        1,
        "{name}: exact selected feature spelling {spelling:?} was not emitted once:\n{definition}"
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DecodedMemoryShape {
    mnemonic: &'static str,
    segment: &'static str,
    base: &'static str,
    index: &'static str,
    scale: i64,
    displacement: i64,
    width: usize,
    read: bool,
    write: bool,
}

fn decoded_memory_shapes(db: &DecompileDB, name: &str) -> Vec<DecodedMemoryShape> {
    let span = function_span(db, name);
    let reads: BTreeSet<(Node, Symbol)> = db
        .rel_iter::<(Node, Symbol)>("decoded_memory_read_operand")
        .copied()
        .collect();
    let writes: BTreeSet<(Node, Symbol)> = db
        .rel_iter::<(Node, Symbol)>("decoded_memory_write_operand")
        .copied()
        .collect();
    let mut indirects: BTreeMap<
        Symbol,
        Vec<(&'static str, &'static str, &'static str, i64, i64, usize)>,
    > = BTreeMap::new();
    for (operand, segment, base, index, scale, displacement, width) in db.rel_iter::<(
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )>("op_indirect")
    {
        indirects.entry(*operand).or_default().push((
            *segment,
            *base,
            *index,
            *scale,
            *displacement,
            *width,
        ));
    }
    let mut shapes = Vec::new();
    for (node, _, _, mnemonic, op1, op2, op3, op4, _, _) in db.rel_iter::<(
        Node,
        usize,
        &'static str,
        &'static str,
        Symbol,
        Symbol,
        Symbol,
        Symbol,
        usize,
        usize,
    )>("unrefinedinstruction")
    {
        if !in_span(*node, span) {
            continue;
        }
        for operand in [*op1, *op2, *op3, *op4] {
            let Some(rows) = indirects.get(&operand).filter(|rows| rows.len() == 1) else {
                continue;
            };
            let (segment, base, index, scale, displacement, width) = rows[0];
            shapes.push(DecodedMemoryShape {
                mnemonic: *mnemonic,
                segment,
                base,
                index,
                scale,
                displacement,
                width,
                read: reads.contains(&(*node, operand)),
                write: writes.contains(&(*node, operand)),
            });
        }
    }
    shapes
}

fn assert_rebuilt_memory_semantics(db: &DecompileDB, name: &str, proof: &ScalarMemoryAccessProof) {
    let shapes = decoded_memory_shapes(db, name);
    let effective_displacement = if name.ends_with("scalar_lvalue_inline_index") {
        24
    } else {
        proof.displacement
    };
    let expected_mnemonic = if proof.extension == ScalarMemoryExtension::Plain {
        "MOV"
    } else if proof.extension == ScalarMemoryExtension::ZeroExtend {
        "MOVZX"
    } else if proof.width == 4 {
        "MOVSXD"
    } else {
        "MOVSX"
    };
    assert!(
        shapes.iter().any(|shape| {
            shape.mnemonic == expected_mnemonic
                && shape.segment == "NONE"
                && shape.base != "RIP"
                && shape.width == proof.width
                && shape.read == (proof.direction == ScalarMemoryDirection::Read)
                && shape.write == (proof.direction == ScalarMemoryDirection::Write)
                && (if proof.index_register.is_some() {
                    shape.index != "NONE" && shape.scale == proof.scale
                } else {
                    shape.index == "NONE" && shape.scale == 1
                })
                && shape.displacement == effective_displacement
        }),
        "{name}: rebuilt object lost width/direction/extension/indexed/raw semantics for {proof:#?}: {shapes:#?}"
    );
}

fn run_fixture_pipeline(object: &Path, pool: &rayon::ThreadPool) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, false, false));
    db
}

#[test]
fn scalar_lvalue_real_coff_pipeline_emits_compilable_candidates_and_fallbacks() {
    let object = fixture_object();
    assert_eq!(
        sha256_file(object),
        BASE718_FIXTURE_SHA256,
        "fixture bytes drifted; immutable base-718 canonical oracle is no longer applicable"
    );
    let base718_golden_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/scalar_lvalue_base718.c"
    ));
    assert_eq!(BASE718_CANONICAL_SOURCE.as_bytes().len(), 2167);
    assert_eq!(
        sha256_file(base718_golden_path),
        BASE718_CANONICAL_SOURCE_SHA256,
        "checked-in base-718 canonical source literal drifted"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("build scalar-lvalue test pool");
    let db = run_fixture_pipeline(object, &pool);
    let mut positive_proofs = BTreeMap::new();
    let mut deferred_extension_proofs = BTreeMap::new();

    for (name, direction, extension, width, value_width) in [
        (
            "scalar_lvalue_plain32",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::Plain,
            4,
            4,
        ),
        (
            "scalar_lvalue_plain64",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::Plain,
            8,
            8,
        ),
        (
            "scalar_lvalue_movzx_indexed",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::ZeroExtend,
            1,
            4,
        ),
        (
            "scalar_lvalue_movsx64",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::SignExtend,
            1,
            8,
        ),
        (
            "scalar_lvalue_store16",
            ScalarMemoryDirection::Write,
            ScalarMemoryExtension::Plain,
            2,
            2,
        ),
        (
            "scalar_lvalue_inline_index",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::Plain,
            4,
            4,
        ),
    ] {
        let proofs = proofs_in_function(&db, name);
        assert_eq!(proofs.len(), 1, "{name}: {proofs:#?}");
        let proof = &proofs[0];
        assert_eq!(proof.direction, direction, "{name}");
        assert_eq!(proof.extension, extension, "{name}");
        assert_eq!(proof.width, width, "{name}");
        assert_eq!(proof.value_width, value_width, "{name}");
        if name == "scalar_lvalue_plain64" {
            assert_eq!(proof.chunk, manifold::x86::types::MemoryChunk::MAny64);
        }
        if name == "scalar_lvalue_movzx_indexed" {
            assert_eq!(
                proof.chunk,
                manifold::x86::types::MemoryChunk::MInt8Unsigned
            );
        }
        if name == "scalar_lvalue_movsx64" {
            assert_eq!(proof.extension, ScalarMemoryExtension::SignExtend);
            assert_eq!(
                proof.chunk,
                manifold::x86::types::MemoryChunk::MInt8Unsigned
            );
        }
        assert!(proof.is_closed_v1(), "{name}: {proof:#?}");
        let reaches_feature = db
            .rel_iter::<(Node, ScalarMemoryAccessProof)>("scalar_lvalue_candidate")
            .any(|(node, candidate)| *node == proof.selected_node && candidate == proof);
        if proof.extension == ScalarMemoryExtension::Plain {
            assert!(
                reaches_feature,
                "{name} lost its proof before isolated feature generation"
            );
            assert_scalar_clight_candidate(&db, name, proof);
            positive_proofs.insert(name, proof.clone());
        } else {
            assert!(
                !reaches_feature,
                "{name}: stage-2 extension proof reached the feature emitter"
            );
            assert!(
                !db.rel_iter::<(Node, ScalarMemoryAccessProof)>(
                    "signature_scalar_extension_access"
                )
                .any(|(node, candidate)| *node == proof.selected_node && candidate == proof),
                "{name}: extension proof changed the canonical signature boundary"
            );
            deferred_extension_proofs.insert(name, proof.clone());
        }
    }

    let base718_canonical_controls = [
        "scalar_lvalue_base718_high8_control",
        "scalar_lvalue_base718_word_dest_control",
        "scalar_lvalue_base718_wide_use_control",
    ];
    for name in base718_canonical_controls {
        assert!(
            proofs_in_function(&db, name).is_empty(),
            "negative fixture {name} acquired a scalar-lvalue proof"
        );
        let span = function_span(&db, name);
        for relation in ["cminor_scalar_memory_access", "scalar_lvalue_candidate"] {
            assert!(
                !db.rel_iter::<(Node, ScalarMemoryAccessProof)>(relation)
                    .any(|(node, _)| in_span(*node, span)),
                "base-718dcc control {name} reached feature relation {relation}"
            );
        }
        let memory_nodes: BTreeSet<Node> = db
            .rel_iter::<(Node, Symbol)>("decoded_memory_read_operand")
            .chain(db.rel_iter::<(Node, Symbol)>("decoded_memory_write_operand"))
            .filter_map(|(node, _)| in_span(*node, span).then_some(*node))
            .collect();
        assert!(
            !memory_nodes.is_empty(),
            "negative fixture {name} has no decoded memory effect"
        );
        assert!(
            db.rel_iter::<(Node, ClightStmt)>("clight_stmt")
                .any(|(node, _)| in_span(*node, span)),
            "negative fixture {name} lost its ordinary canonical Clight fallback"
        );
    }

    let translation_unit = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("scalar-lvalue fixture pipeline emitted no translation unit");
    let source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        translation_unit,
        BinaryFormat::Coff,
    );
    let patched_canonical_path = object.with_file_name("scalar_lvalue_patched_canonical.c");
    std::fs::write(&patched_canonical_path, &source)
        .expect("write patched canonical source for immutable-oracle check");
    assert_eq!(
        sha256_file(&patched_canonical_path),
        BASE718_CANONICAL_SOURCE_SHA256,
        "patched canonical primary SHA differs from immutable base-718 output"
    );
    assert_eq!(
        source, BASE718_CANONICAL_SOURCE,
        "stage-2 evidence changed the canonical primary emitted by immutable base provider {} (binary {}, identity {})",
        BASE718_PROVIDER_COMMIT,
        BASE718_PROVIDER_BINARY_SHA256,
        BASE718_PROVIDER_IDENTITY_SHA256,
    );
    for name in [
        "scalar_lvalue_plain32",
        "scalar_lvalue_plain64",
        "scalar_lvalue_movzx_indexed",
        "scalar_lvalue_movsx64",
        "scalar_lvalue_store16",
        "scalar_lvalue_inline_index",
        "scalar_lvalue_base718_high8_control",
        "scalar_lvalue_base718_word_dest_control",
        "scalar_lvalue_base718_wide_use_control",
    ] {
        assert!(
            printed_function_definition(&source, name).is_some(),
            "final source lost candidate or canonical fallback {name}:\n{source}"
        );
    }
    let mut feature_translation_unit = translation_unit.clone();
    let mut selected_feature_names = BTreeSet::new();
    for (name, proof) in &positive_proofs {
        let snapshots: Vec<_> = db
            .cast_source_alternatives
            .iter()
            .filter(|snapshot| {
                snapshot.manifold_address == proof.function
                    && (snapshot.manifold_name == *name
                        || snapshot.manifold_name == format!("coff_fn_{name}"))
                    && matches!(
                        snapshot.boundary,
                        SourceAlternativeBoundary::ScalarLvaluePreVarReduce
                            | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                    )
            })
            .collect();
        assert!(
            snapshots.is_empty() || snapshots.len() == 2,
            "{name}: typed-lvalue selection must publish no record or one atomic pre/post pair: {snapshots:#?}"
        );
        if snapshots.is_empty() {
            continue;
        }
        selected_feature_names.insert(*name);
        assert!(snapshots.iter().any(|snapshot| {
            snapshot.boundary == SourceAlternativeBoundary::ScalarLvaluePreVarReduce
                && snapshot.kinds == vec!["local_lifetime", "typed_lvalue"]
        }));
        let post = snapshots
            .iter()
            .find(|snapshot| {
                snapshot.boundary == SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                    && snapshot.kinds == vec!["typed_lvalue"]
            })
            .expect("post-VarReduce typed-lvalue form");
        let canonical = function_definition(translation_unit, name);
        assert_ne!(canonical, &post.function, "feature form equals primary");
        let target = feature_translation_unit
            .decls
            .iter_mut()
            .find_map(|declaration| match declaration {
                TopLevelDecl::FuncDef(function) if function.name == canonical.name => {
                    Some(function)
                }
                _ => None,
            })
            .expect("feature TU lost canonical function identity");
        *target = post.function.clone();
    }
    for (name, proof) in &deferred_extension_proofs {
        assert!(
            !db.cast_source_alternatives.iter().any(|snapshot| {
                snapshot.manifold_address == proof.function
                    && (snapshot.manifold_name == *name
                        || snapshot.manifold_name == format!("coff_fn_{name}"))
                    && matches!(
                        snapshot.boundary,
                        SourceAlternativeBoundary::ScalarLvaluePreVarReduce
                            | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                    )
            }),
            "{name}: deferred extension proof emitted a stage-2 typed-lvalue sidecar"
        );
    }
    let feature_source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        &feature_translation_unit,
        BinaryFormat::Coff,
    );
    assert!(
        selected_feature_names.contains("scalar_lvalue_inline_index"),
        "the real dead-at-use LEA fixture must produce its two scored feature forms"
    );
    for name in &selected_feature_names {
        let proof = positive_proofs
            .get(*name)
            .expect("selected feature retains its authenticated proof");
        assert_selected_feature_lvalue(
            &feature_translation_unit,
            &feature_source,
            name,
            proof,
        );
    }

    let all_canonical_names = [
        "scalar_lvalue_plain32",
        "scalar_lvalue_plain64",
        "scalar_lvalue_movzx_indexed",
        "scalar_lvalue_movsx64",
        "scalar_lvalue_store16",
        "scalar_lvalue_inline_index",
        "scalar_lvalue_base718_high8_control",
        "scalar_lvalue_base718_word_dest_control",
        "scalar_lvalue_base718_wide_use_control",
    ];
    let base718_definitions: BTreeMap<&str, String> = all_canonical_names
        .iter()
        .map(|name| {
            (
                *name,
                printed_function_definition(BASE718_CANONICAL_SOURCE, name)
                    .unwrap_or_else(|| panic!("base-718 golden lacks definition {name}"))
                    .to_string(),
            )
        })
        .collect();
    let replay_db = run_fixture_pipeline(object, &pool);
    let replay_translation_unit = replay_db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("scalar-lvalue replay emitted no translation unit");
    let replay_source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        replay_translation_unit,
        BinaryFormat::Coff,
    );
    assert_eq!(
        replay_translation_unit, translation_unit,
        "feature evidence changed the replayed canonical TU AST"
    );
    assert_eq!(
        replay_source, source,
        "feature evidence changed the canonical primary TU bytes, including extension controls"
    );
    assert_eq!(replay_source, BASE718_CANONICAL_SOURCE);
    for (name, golden) in &base718_definitions {
        assert_eq!(
            printed_function_definition(&replay_source, name),
            Some(golden.as_str()),
            "{name} did not replay its exact byte-stable canonical definition"
        );
    }

    let emitted = object.with_file_name("scalar_lvalue_generated.c");
    let rebuilt = object.with_file_name("scalar_lvalue_generated.obj");
    std::fs::write(&emitted, &feature_source).expect("write emitted scalar-lvalue feature C");
    let compiled = Command::new("clang")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-O1",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            "c",
            "-c",
        ])
        .arg(&emitted)
        .arg("-o")
        .arg(&rebuilt)
        .output()
        .expect("compile emitted scalar-lvalue C");
    assert!(
        compiled.status.success(),
        "emitted scalar-lvalue C did not compile:\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr),
        feature_source
    );

    let mut rebuilt_db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut rebuilt_db, &rebuilt);
    manifold::decompile::disassembly::load_preset(&mut rebuilt_db);
    for (name, proof) in &positive_proofs {
        let emitted_name = &function_definition(&feature_translation_unit, name).name;
        assert_rebuilt_memory_semantics(&rebuilt_db, emitted_name, proof);
    }
}
