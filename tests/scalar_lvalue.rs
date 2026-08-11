use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::c_pass::print::{IntegerModel, PrintConfig, Printer};
use manifold::decompile::passes::c_pass::types::{
    AssignOp, BinaryOp, CBlockItem, CExpr, CStmt, CType, FuncDef, Initializer, IntSize, Signedness,
    TopLevelDecl, UnaryOp,
};
use manifold::decompile::postselect::source_alternatives::SourceAlternativeBoundary;
use manifold::x86::types::{
    Address, BuiltinArg, ClightBinaryOp, ClightExpr, ClightIntSize, ClightSignedness, ClightStmt,
    ClightType, Node, RTLInst, RTLReg, ScalarLvaluePlacement, ScalarMemoryAccessProof,
    ScalarMemoryDirection, ScalarMemoryExtension, ScalarMemoryResultChain, ScalarMemoryUsePlan,
    ScalarMemoryUseType, Symbol,
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

// Immutable Stage-2 provider oracle for the expanded Stage-3 fixture. The
// object is assembled from the checked-in bytes above, then the exact bd762
// binary emits both the selected Clight AST JSON and canonical C golden. A
// patched test must match both byte streams before it may inspect or compile a
// feature view.
const BD762_PROVIDER_COMMIT: &str = "bd7620e8af85a4760ce49419a80a543fae9b88a4";
const BD762_PROVIDER_TREE: &str = "b62e04682a2694e002521650a2f774a44bc8ef01";
const BD762_PROVIDER_BINARY_SHA256: &str =
    "319dbd2021eb183611f1c6e9fcff42282270e1cac502cb376b877ceca074c60c";
const BD762_PROVIDER_BINARY_SIZE: u64 = 732000992;
const BD762_PROVIDER_BUILD_ID: &str = "manifold-6c3ae804cadc9c7679f5dcc7";
const BD762_PROVIDER_IDENTITY_SHA256: &str =
    "500cf19da0f424125bee5625c30ce4557e9be29f1a2d260241b125d17c2e528f";
const BD762_FIXTURE_ASSEMBLY_SHA256: &str =
    "505b6d5cf9925528af8468732ee34161feaca3c8fdf19e8673e7fe97f0cbae96";
const BD762_FIXTURE_OBJECT_SHA256: &str =
    "e528469bc7b853151b9a178d6eff7abac3bea3a8aa8cf954b374ccb5b5508c0b";
const BD762_CANONICAL_CLIGHT_JSON_SHA256: &str =
    "63995bc35f20dfaa447374c06c2d440b45e62c4fe74b60b7ee77f7fc68a3b231";
const BD762_CANONICAL_SOURCE_SHA256: &str =
    "1df6ebc0a37fde7bea56fe966df7deb542e6f38d50d836ce9861ea41a58100cb";
const BD762_CANONICAL_SOURCE: &str = include_str!("fixtures/scalar_extension_bd762.c");

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

        # The high-byte and word-destination forms remain outside the sealed
        # descriptor. The r32 load followed by a 64-bit consumer is a Stage-3
        # positive: the encoded EAX write proves architectural upper-32 zeroing.
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

fn build_stage3_extension_fixture() -> PathBuf {
    assert!(command_exists("clang"), "clang is required for Stage-3 COFF coverage");
    let directory = std::env::temp_dir().join(format!(
        "manifold_scalar_extension_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Stage-3 fixture directory");
    let source = directory.join("stage3-fixture.s");
    let object = directory.join("stage3-fixture.obj");
    std::fs::write(&source, include_str!("fixtures/scalar_extension_bd762.s"))
    .expect("write Stage-3 extension assembly");
    assert_eq!(
        sha256_file(&source),
        BD762_FIXTURE_ASSEMBLY_SHA256,
        "checked-in Stage-3 assembly fixture bytes drifted",
    );
    let compiled = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("assemble Stage-3 extension fixture");
    assert!(
        compiled.status.success(),
        "Stage-3 extension fixture failed to assemble:\n{}",
        String::from_utf8_lossy(&compiled.stderr),
    );
    assert_eq!(
        sha256_file(&object),
        BD762_FIXTURE_OBJECT_SHA256,
        "Stage-3 COFF fixture differs from the immutable bd762 oracle input",
    );
    object
}

fn stage3_extension_fixture_object() -> &'static Path {
    static OBJECT: OnceLock<PathBuf> = OnceLock::new();
    OBJECT.get_or_init(build_stage3_extension_fixture).as_path()
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

fn builtin_uses_value(argument: &BuiltinArg<RTLReg>, value: RTLReg) -> usize {
    match argument {
        BuiltinArg::BA(register) => usize::from(*register == value),
        BuiltinArg::BASplitLong(left, right) | BuiltinArg::BAAddPtr(left, right) => {
            builtin_uses_value(left, value) + builtin_uses_value(right, value)
        }
        _ => 0,
    }
}

fn rtl_value_def_use(instruction: &RTLInst, value: RTLReg) -> (usize, usize) {
    match instruction {
        RTLInst::Inop | RTLInst::Ibranch(_) => (0, 0),
        RTLInst::Iop(_, arguments, destination)
        | RTLInst::Iload(_, _, arguments, destination) => (
            usize::from(*destination == value),
            arguments.iter().filter(|argument| **argument == value).count(),
        ),
        RTLInst::Istore(_, _, arguments, source) => (
            0,
            arguments.iter().filter(|argument| **argument == value).count()
                + usize::from(*source == value),
        ),
        RTLInst::Icall(_, callee, arguments, destination, _) => (
            usize::from(destination == &Some(value)),
            arguments.iter().filter(|argument| **argument == value).count()
                + usize::from(matches!(callee, either::Either::Left(register) if *register == value)),
        ),
        RTLInst::Itailcall(_, callee, arguments) => (
            0,
            arguments.iter().filter(|argument| **argument == value).count()
                + usize::from(matches!(callee, either::Either::Left(register) if *register == value)),
        ),
        RTLInst::Ibuiltin(_, arguments, result) => (
            usize::from(matches!(result, BuiltinArg::BA(register) if *register == value)),
            arguments
                .iter()
                .map(|argument| builtin_uses_value(argument, value))
                .sum(),
        ),
        RTLInst::Icond(_, arguments, _, _) => (
            0,
            arguments.iter().filter(|argument| **argument == value).count(),
        ),
        RTLInst::Ijumptable(register, _) | RTLInst::Ireturn(register) => {
            (0, usize::from(*register == value))
        }
    }
}

fn assert_final_root_store_fanout(
    db: &DecompileDB,
    proof: &ScalarMemoryAccessProof,
    minimum_stores: usize,
) {
    let plans: Vec<_> = db
        .rel_iter::<(Node, ScalarMemoryUsePlan)>("scalar_memory_use_plan")
        .filter(|(node, plan)| {
            *node == proof.selected_node
                && plan.function == proof.function
                && plan.value == proof.value
        })
        .map(|(_, plan)| plan)
        .collect();
    assert_eq!(plans.len(), 1, "fanout proof lacks one final-revalidated use plan");
    let plan = plans[0];
    assert!(plan.is_closed_v1(proof));
    assert!(plan.transports.is_empty(), "bd762 canonical RTLOpt should coalesce redundant qword transports");
    let site_nodes: BTreeSet<_> = plan.sites.iter().map(|site| site.node).collect();
    assert_eq!(site_nodes.len(), plan.sites.len());
    assert!(plan.sites.len() >= minimum_stores);
    assert!(plan.sites.iter().all(|site| {
        site.value == proof.value && site.required_type == ScalarMemoryUseType::Unsigned64
    }));

    let owners: BTreeMap<Node, BTreeSet<Address>> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .fold(BTreeMap::new(), |mut owners, (node, function)| {
            owners.entry(*node).or_default().insert(*function);
            owners
        });
    let mut definitions = Vec::new();
    let mut uses = Vec::new();
    for (node, instruction) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        let (definition_count, use_count) = rtl_value_def_use(instruction, proof.value);
        for _ in 0..definition_count {
            definitions.push(*node);
        }
        for _ in 0..use_count {
            uses.push(*node);
        }
        if site_nodes.contains(node) {
            assert_eq!(owners.get(node), Some(&BTreeSet::from([proof.function])));
            assert!(matches!(
                instruction,
                RTLInst::Istore(
                    manifold::x86::types::MemoryChunk::MAny64,
                    _,
                    _,
                    source,
                ) if *source == proof.value
            ));
        }
    }
    definitions.sort_unstable();
    uses.sort_unstable();
    let mut expected_uses: Vec<_> = site_nodes.into_iter().collect();
    expected_uses.sort_unstable();
    assert_eq!(definitions, vec![proof.selected_node]);
    assert_eq!(uses, expected_uses, "fanout root has a hidden or duplicate final RTL use");
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
        ScalarMemoryExtension::ZeroExtend | ScalarMemoryExtension::ImplicitZeroExtend => {
            Some(ClightSignedness::Unsigned)
        }
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
            ScalarLvaluePlacement,
            ClightStmt,
        )>("scalar_lvalue_source_candidate")
        .filter(|(node, _, placement, _)| {
            *node == proof.selected_node && *placement == ScalarLvaluePlacement::Plain
        })
        .collect();
    assert!(
        tagged.iter().any(|(_, form, _, _)| *form == expected_form),
        "{name}: isolated candidate set lacks expected form {expected_form:?}: {tagged:#?}"
    );
    let candidates: Vec<&ClightStmt> = tagged
        .into_iter()
        .map(|(_, _, _, statement)| statement)
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

fn assert_extension_clight_candidates(
    db: &DecompileDB,
    name: &str,
    proof: &ScalarMemoryAccessProof,
) {
    let tagged: Vec<_> = db
        .rel_iter::<(
            Node,
            manifold::x86::types::ScalarLvalueSourceForm,
            ScalarLvaluePlacement,
            ClightStmt,
        )>("scalar_lvalue_source_candidate")
        .filter(|(node, _, placement, _)| {
            *node == proof.selected_node
                && matches!(
                    placement,
                    ScalarLvaluePlacement::ExtensionPerUse
                        | ScalarLvaluePlacement::ExtensionHoisted
                )
        })
        .collect();
    for placement in [
        ScalarLvaluePlacement::ExtensionPerUse,
        ScalarLvaluePlacement::ExtensionHoisted,
    ] {
        let candidates: Vec<_> = tagged
            .iter()
            .filter(|(_, _, candidate_placement, _)| *candidate_placement == placement)
            .filter_map(|(_, _, _, statement)| scalar_clight_expr(statement, proof))
            .filter(|expression| clight_expr_has_access(expression, proof))
            .collect();
        assert!(
            !candidates.is_empty(),
            "{name}: {placement:?} lacks an exact width/signedness lvalue: {tagged:#?}"
        );
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
        (
            ScalarMemoryExtension::ZeroExtend | ScalarMemoryExtension::ImplicitZeroExtend,
            CType::Int(_, Signedness::Unsigned),
        ) => true,
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

fn test_x86_register_width(register: &str) -> Option<usize> {
    match register {
        "RAX" | "RBX" | "RCX" | "RDX" | "RSI" | "RDI" | "RBP" | "RSP" | "R8"
        | "R9" | "R10" | "R11" | "R12" | "R13" | "R14" | "R15" => Some(8),
        "EAX" | "EBX" | "ECX" | "EDX" | "ESI" | "EDI" | "EBP" | "ESP" | "R8D"
        | "R9D" | "R10D" | "R11D" | "R12D" | "R13D" | "R14D" | "R15D" => Some(4),
        "AX" | "BX" | "CX" | "DX" | "SI" | "DI" | "BP" | "SP" | "R8W" | "R9W"
        | "R10W" | "R11W" | "R12W" | "R13W" | "R14W" | "R15W" => Some(2),
        "AL" | "BL" | "CL" | "DL" | "SIL" | "DIL" | "BPL" | "SPL" | "R8B"
        | "R9B" | "R10B" | "R11B" | "R12B" | "R13B" | "R14B" | "R15B" => Some(1),
        _ => None,
    }
}

fn test_x86_register_parent(register: &str) -> Option<&'static str> {
    match register {
        "RAX" | "EAX" | "AX" | "AL" | "AH" => Some("RAX"),
        "RBX" | "EBX" | "BX" | "BL" | "BH" => Some("RBX"),
        "RCX" | "ECX" | "CX" | "CL" | "CH" => Some("RCX"),
        "RDX" | "EDX" | "DX" | "DL" | "DH" => Some("RDX"),
        "RSI" | "ESI" | "SI" | "SIL" => Some("RSI"),
        "RDI" | "EDI" | "DI" | "DIL" => Some("RDI"),
        "RBP" | "EBP" | "BP" | "BPL" => Some("RBP"),
        "RSP" | "ESP" | "SP" | "SPL" => Some("RSP"),
        "R8" | "R8D" | "R8W" | "R8B" => Some("R8"),
        "R9" | "R9D" | "R9W" | "R9B" => Some("R9"),
        "R10" | "R10D" | "R10W" | "R10B" => Some("R10"),
        "R11" | "R11D" | "R11W" | "R11B" => Some("R11"),
        "R12" | "R12D" | "R12W" | "R12B" => Some("R12"),
        "R13" | "R13D" | "R13W" | "R13B" => Some("R13"),
        "R14" | "R14D" | "R14W" | "R14B" => Some("R14"),
        "R15" | "R15D" | "R15W" | "R15B" => Some("R15"),
        _ => None,
    }
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
    if proof.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32) {
        return Some(CType::Int(IntSize::Long, Signedness::Unsigned));
    }
    let signedness = match proof.extension {
        ScalarMemoryExtension::SignExtend => Signedness::Signed,
        ScalarMemoryExtension::ZeroExtend | ScalarMemoryExtension::ImplicitZeroExtend => {
            Signedness::Unsigned
        }
        ScalarMemoryExtension::Plain => return None,
    };
    match proof.value_width {
        4 => Some(CType::Int(IntSize::Int, signedness)),
        8 => Some(CType::Int(IntSize::Long, signedness)),
        _ => None,
    }
}

fn extension_result_input<'a>(
    expr: &'a CExpr,
    proof: &ScalarMemoryAccessProof,
) -> Option<&'a CExpr> {
    let CExpr::Cast(result_type, inner) = unparen(expr) else {
        return None;
    };
    if c_extension_result_type(proof).as_ref() != Some(result_type) {
        return None;
    }
    if proof.result_chain != Some(ScalarMemoryResultChain::ZeroUpper32) {
        return Some(unparen(inner));
    }
    let unsigned32 = CType::Int(IntSize::Int, Signedness::Unsigned);
    let CExpr::Cast(encoded_type, encoded_input) = unparen(inner) else {
        return None;
    };
    if encoded_type != &unsigned32 {
        return None;
    }
    if proof.extension != ScalarMemoryExtension::SignExtend {
        return Some(unparen(encoded_input));
    }
    let signed32 = CType::Int(IntSize::Int, Signedness::Signed);
    match unparen(encoded_input) {
        CExpr::Cast(sign_type, lvalue) if sign_type == &signed32 => Some(unparen(lvalue)),
        // A signed byte/word lvalue or local undergoes C's integral promotion
        // to signed int before the explicit unsigned-int conversion.  The
        // caller independently checks the access/local declaration type.
        promoted => Some(promoted),
    }
}

fn selected_feature_expression<'a>(
    expr: &'a CExpr,
    proof: &ScalarMemoryAccessProof,
) -> Option<(bool, FeatureLvalueSpelling, &'a CExpr)> {
    match proof.direction {
        ScalarMemoryDirection::Read if proof.extension == ScalarMemoryExtension::Plain => {
            feature_lvalue_spelling(expr, proof).map(|spelling| (false, spelling, expr))
        }
        ScalarMemoryDirection::Read => {
            // After the provider rejects an encoded-qword or implicit-wide
            // family with no surviving full-width use, a remaining narrow
            // return may legitimately collapse only the outer result cast.
            if let Some(lvalue) = extension_result_input(expr, proof) {
                feature_lvalue_spelling(lvalue, proof)
                    .map(|spelling| (true, spelling, expr))
            } else {
                feature_lvalue_spelling(expr, proof).map(|spelling| (false, spelling, expr))
            }
        }
        ScalarMemoryDirection::Write => {
            let CExpr::Assign(AssignOp::Assign, lvalue, _) = unparen(expr) else {
                return None;
            };
            feature_lvalue_spelling(lvalue, proof)
                .map(|spelling| (false, spelling, lvalue.as_ref()))
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
    let mut candidates = BTreeSet::new();
    visit_c_stmt(&function.body, &mut |expr| {
        if let Some((has_result_chain, kind, feature_expr)) =
            selected_feature_expression(expr, proof)
        {
            candidates.insert((has_result_chain, kind, print_c_expr(feature_expr)));
        }
    });
    let has_result_chain = candidates.iter().any(|(explicit, _, _)| *explicit);
    let spellings: BTreeSet<_> = candidates
        .into_iter()
        .filter(|(explicit, _, _)| !has_result_chain || *explicit)
        .map(|(_, kind, spelling)| (kind, spelling))
        .collect();
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

fn assert_selected_per_use_feature_lvalue(
    translation_unit: &manifold::decompile::passes::c_pass::TranslationUnit,
    source: &str,
    name: &str,
    proof: &ScalarMemoryAccessProof,
) {
    assert_ne!(proof.extension, ScalarMemoryExtension::Plain);
    let function = function_definition(translation_unit, name);
    let definition = printed_function_definition(source, &function.name)
        .unwrap_or_else(|| panic!("printed source lost per-use function {}", function.name));
    let mut lvalues = BTreeSet::new();
    let mut exact_result_casts = BTreeSet::new();
    visit_c_stmt(&function.body, &mut |expr| {
        if let Some(spelling) = feature_lvalue_spelling(expr, proof) {
            lvalues.insert((spelling, print_c_expr(expr)));
        }
        if extension_result_input(expr, proof).is_some_and(|input| match unparen(input) {
            CExpr::Var(name) => function
                .local_vars
                .iter()
                .any(|declaration| {
                    declaration.name == *name && c_access_type_matches(&declaration.ty, proof)
                }),
            _ => false,
        }) {
            exact_result_casts.insert(print_c_expr(expr));
        }
    });
    assert_eq!(
        lvalues.len(),
        1,
        "{name}: per-use source must contain one exact authenticated load lvalue: {lvalues:#?}\n{definition}",
    );
    let narrow_result_collapse = c_scalar_width(&function.return_type)
        .is_some_and(|width| width < proof.value_width);
    assert!(
        !exact_result_casts.is_empty() || narrow_result_collapse,
        "{name}: per-use source lacks the exact architectural result cast at its consumer:\n{definition}",
    );
    let (_, lvalue) = lvalues.iter().next().expect("one per-use lvalue");
    assert_eq!(
        definition.matches(lvalue).count(),
        1,
        "{name}: per-use load lvalue was not emitted exactly once:\n{definition}",
    );
}

fn assert_selected_field_feature(
    translation_unit: &manifold::decompile::passes::c_pass::TranslationUnit,
    source: &str,
    name: &str,
    expected_members: usize,
) {
    let function = function_definition(translation_unit, name);
    let definition = printed_function_definition(source, &function.name)
        .unwrap_or_else(|| panic!("printed source lost field function {}", function.name));
    let mut members = BTreeSet::new();
    visit_c_stmt(&function.body, &mut |expression| {
        if matches!(
            unparen(expression),
            CExpr::Member(_, _) | CExpr::MemberPtr(_, _)
        ) {
            members.insert(print_c_expr(expression));
        }
    });
    assert_eq!(
        members.len(),
        expected_members,
        "{name}: field profile lost or invented an existing layout member: {members:#?}\n{definition}",
    );
    for member in members {
        assert_eq!(definition.matches(&member).count(), 1);
    }
}

fn replace_function_definition(
    translation_unit: &mut manifold::decompile::passes::c_pass::TranslationUnit,
    canonical_name: &str,
    replacement: &FuncDef,
) {
    let target = translation_unit
        .decls
        .iter_mut()
        .find_map(|declaration| match declaration {
            TopLevelDecl::FuncDef(function) if function.name == canonical_name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("feature TU lost canonical function {canonical_name}"));
    *target = replacement.clone();
}

fn choose_collapsed_snapshot<'a>(
    rows: &[&'a manifold::decompile::postselect::source_alternatives::SourceAlternativeSnapshot],
    post_boundary: SourceAlternativeBoundary,
) -> &'a manifold::decompile::postselect::source_alternatives::SourceAlternativeSnapshot {
    rows.iter()
        .find(|snapshot| snapshot.boundary == post_boundary)
        .copied()
        .unwrap_or(rows[0])
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DecodedMemoryShape {
    node: Node,
    mnemonic: &'static str,
    segment: &'static str,
    base: &'static str,
    index: &'static str,
    scale: i64,
    displacement: i64,
    width: usize,
    read: bool,
    write: bool,
    value_register: Option<&'static str>,
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
    let registers: BTreeMap<Symbol, &'static str> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .copied()
        .collect();
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
            let read = reads.contains(&(*node, operand));
            let write = writes.contains(&(*node, operand));
            let value_register = if read {
                registers.get(op2).copied()
            } else if write {
                registers.get(op1).copied()
            } else {
                None
            };
            shapes.push(DecodedMemoryShape {
                node: *node,
                mnemonic: *mnemonic,
                segment,
                base,
                index,
                scale,
                displacement,
                width,
                read,
                write,
                value_register,
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
    let expected_mnemonic = if matches!(
        proof.extension,
        ScalarMemoryExtension::Plain | ScalarMemoryExtension::ImplicitZeroExtend
    ) {
        "MOV"
    } else if proof.extension == ScalarMemoryExtension::ZeroExtend {
        "MOVZX"
    } else if proof.width == 4 {
        "MOVSXD"
    } else {
        "MOVSX"
    };
    let matching: Vec<_> = shapes
        .iter()
        .filter(|shape| {
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
        })
        .collect();
    assert!(
        !matching.is_empty(),
        "{name}: rebuilt object lost width/direction/extension/indexed/raw semantics for {proof:#?}: {shapes:#?}"
    );
    if proof.direction == ScalarMemoryDirection::Read {
        let encoded_width = proof
            .encoded_destination_width
            .expect("authenticated read has encoded destination width");
        let direct_encoded_destination = matching.iter().any(|shape| {
            shape
                .value_register
                .is_some_and(|register| test_x86_register_width(register) == Some(encoded_width))
        });
        let span = function_span(db, name);
        let split_zero_upper_destination = proof.result_chain
            == Some(ScalarMemoryResultChain::ZeroUpper32)
            && encoded_width == 4
            && matching.iter().any(|shape| {
                let Some(parent) = shape.value_register.and_then(test_x86_register_parent) else {
                    return false;
                };
                test_x86_register_width(shape.value_register.unwrap()) == Some(8)
                    && db
                        .rel_iter::<(
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
                        .any(|(node, _, _, mnemonic, op1, op2, _, _, _, _)| {
                            if !in_span(*node, span)
                                || *node <= shape.node
                                || *mnemonic != "MOV"
                            {
                                return false;
                            }
                            let register = |operand| {
                                db.rel_iter::<(Symbol, &'static str)>("op_register")
                                    .find_map(|(symbol, register)| {
                                        (*symbol == operand).then_some(*register)
                                    })
                            };
                            let (Some(source), Some(destination)) =
                                (register(*op1), register(*op2))
                            else {
                                return false;
                            };
                            test_x86_register_width(source) == Some(4)
                                && test_x86_register_width(destination) == Some(4)
                                && test_x86_register_parent(source) == Some(parent)
                                && test_x86_register_parent(destination) == Some(parent)
                        })
            });
        assert!(
            direct_encoded_destination || split_zero_upper_destination,
            "{name}: rebuilt memory instruction lost encoded destination width {encoded_width} or its exact later r32 zero-upper normalization: {matching:#?}",
        );
        if proof.result_chain == Some(ScalarMemoryResultChain::ZeroUpper32) {
            let encoded_parents: BTreeSet<_> = matching
                .iter()
                .filter_map(|shape| shape.value_register)
                .filter_map(test_x86_register_parent)
                .collect();
            assert_eq!(
                encoded_parents.len(),
                1,
                "{name}: ZeroUpper32 load destination is not a unique r32 register: {matching:#?}",
            );
            let requires_later_wide_use = db
                .rel_iter::<(Node, ScalarMemoryUsePlan)>("scalar_memory_use_plan")
                .filter(|(node, plan)| {
                    *node == proof.selected_node
                        && plan.function == proof.function
                        && plan.value == proof.value
                })
                .any(|(_, plan)| {
                    plan.sites
                        .iter()
                        .any(|site| site.required_type.width() == 8)
                });
            if requires_later_wide_use {
                let has_later_wide_use = db
                    .rel_iter::<(
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
                    .any(|(node, _, _, _, op1, op2, op3, op4, _, _)| {
                        in_span(*node, span)
                            && matching.iter().any(|shape| *node > shape.node)
                            && [*op1, *op2, *op3, *op4].into_iter().any(|operand| {
                                db.rel_iter::<(Symbol, &'static str)>("op_register")
                                    .any(|(symbol, register)| {
                                        *symbol == operand
                                            && test_x86_register_width(register) == Some(8)
                                            && test_x86_register_parent(register)
                                                .is_some_and(|parent| {
                                                    encoded_parents.contains(parent)
                                                })
                                    })
                            })
                    });
                assert!(
                    has_later_wide_use,
                    "{name}: rebuilt ZeroUpper32 result has no authenticated later 64-bit consumer",
                );
            }
        }
    }
}

fn compile_and_check_feature_tu(
    object: &Path,
    label: &str,
    language: &str,
    source: &str,
    translation_unit: &manifold::decompile::passes::c_pass::TranslationUnit,
    positive_proofs: &BTreeMap<&str, ScalarMemoryAccessProof>,
) {
    let extension = if language == "c++" { "cpp" } else { "c" };
    let emitted = object.with_file_name(format!("scalar_lvalue_{label}.{extension}"));
    let rebuilt = object.with_file_name(format!("scalar_lvalue_{label}_{extension}.obj"));
    let compiled_source = if language == "c++" {
        format!("extern \"C\" {{\n{source}\n}}\n")
    } else {
        source.to_string()
    };
    std::fs::write(&emitted, &compiled_source)
        .unwrap_or_else(|error| panic!("write {label} {language} feature source: {error}"));
    let compiled = Command::new("clang")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-O1",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            language,
            "-c",
        ])
        .arg(&emitted)
        .arg("-o")
        .arg(&rebuilt)
        .output()
        .unwrap_or_else(|error| panic!("compile {label} scalar-lvalue {language}: {error}"));
    assert!(
        compiled.status.success(),
        "{label} scalar-lvalue {language} did not compile:\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr),
        compiled_source,
    );

    let mut rebuilt_db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut rebuilt_db, &rebuilt);
    manifold::decompile::disassembly::load_preset(&mut rebuilt_db);
    for (name, proof) in positive_proofs {
        let emitted_name = &function_definition(translation_unit, name).name;
        assert_rebuilt_memory_semantics(&rebuilt_db, emitted_name, proof);
    }
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
            8,
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
        (
            "scalar_lvalue_base718_wide_use_control",
            ScalarMemoryDirection::Read,
            ScalarMemoryExtension::ImplicitZeroExtend,
            4,
            8,
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
            assert_eq!(proof.encoded_destination_width, Some(4));
            assert_eq!(
                proof.result_chain,
                Some(ScalarMemoryResultChain::ZeroUpper32)
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
        assert!(
            reaches_feature,
            "{name} lost its proof before isolated feature generation"
        );
        if proof.extension == ScalarMemoryExtension::Plain {
            assert_scalar_clight_candidate(&db, name, proof);
        } else {
            assert_extension_clight_candidates(&db, name, proof);
        }
        positive_proofs.insert(name, proof.clone());
    }

    let base718_canonical_controls = [
        "scalar_lvalue_base718_high8_control",
        "scalar_lvalue_base718_word_dest_control",
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
    let mut per_use_translation_unit = translation_unit.clone();
    let mut selected_feature_names = BTreeSet::new();
    for (name, proof) in &positive_proofs {
        let matching_identity = |snapshot: &&manifold::decompile::postselect::source_alternatives::SourceAlternativeSnapshot| {
            snapshot.manifold_address == proof.function
                && (snapshot.manifold_name == *name
                    || snapshot.manifold_name == format!("coff_fn_{name}"))
        };
        let snapshots: Vec<_> = db.cast_source_alternatives.iter().filter(|snapshot| {
            matching_identity(snapshot)
                && if proof.extension == ScalarMemoryExtension::Plain {
                    matches!(
                        snapshot.boundary,
                        SourceAlternativeBoundary::ScalarLvaluePreVarReduce
                            | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                    )
                } else {
                    matches!(
                        snapshot.boundary,
                        SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce
                            | SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce
                    )
                }
        }).collect();
        let per_use_snapshots: Vec<_> = if proof.extension != ScalarMemoryExtension::Plain {
            db.cast_source_alternatives
                .iter()
                .filter(|snapshot| {
                    matching_identity(snapshot)
                        && matches!(
                            snapshot.boundary,
                            SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce
                                | SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce
                        )
                })
                .collect()
        } else {
            Vec::new()
        };
        if proof.extension == ScalarMemoryExtension::Plain {
            assert!(
                matches!(snapshots.len(), 1 | 2),
                "{name}: selected profile must retain one or two collapsed source forms: {snapshots:#?}"
            );
        } else if snapshots.is_empty() || per_use_snapshots.is_empty() {
            assert!(
                snapshots.is_empty() && per_use_snapshots.is_empty(),
                "{name}: content collapse left a partial extension superfamily"
            );
            continue;
        } else {
            assert!(matches!(snapshots.len(), 1 | 2));
            assert!(matches!(per_use_snapshots.len(), 1 | 2));
        }
        selected_feature_names.insert(*name);
        let chosen = snapshots
            .iter()
            .find(|snapshot| {
                matches!(
                    snapshot.boundary,
                    SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                        | SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce
                )
            })
            .copied()
            .unwrap_or(snapshots[0]);
        let canonical = function_definition(translation_unit, name);
        assert_ne!(canonical, &chosen.function, "feature form equals primary");
        replace_function_definition(
            &mut feature_translation_unit,
            &canonical.name,
            &chosen.function,
        );
        let per_use_chosen = if proof.extension == ScalarMemoryExtension::Plain {
            chosen
        } else {
            per_use_snapshots
                .iter()
                .find(|snapshot| {
                    snapshot.boundary
                        == SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce
                })
                .copied()
                .unwrap_or(per_use_snapshots[0])
        };
        replace_function_definition(
            &mut per_use_translation_unit,
            &canonical.name,
            &per_use_chosen.function,
        );
    }
    let feature_source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        &feature_translation_unit,
        BinaryFormat::Coff,
    );
    let per_use_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            &per_use_translation_unit,
            BinaryFormat::Coff,
        );
    assert!(
        selected_feature_names.contains("scalar_lvalue_inline_index"),
        "the real dead-at-use LEA fixture must produce its two scored feature forms"
    );
    assert!(
        !selected_feature_names.contains("scalar_lvalue_movsx64")
            && !selected_feature_names.contains("scalar_lvalue_base718_wide_use_control"),
        "a qword or implicit-wide result lost from final RTL must remain provenance-only"
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
        if proof.extension != ScalarMemoryExtension::Plain {
            assert_selected_per_use_feature_lvalue(
                &per_use_translation_unit,
                &per_use_source,
                name,
                proof,
            );
        }
    }
    let selected_feature_proofs: BTreeMap<_, _> = positive_proofs
        .iter()
        .filter(|(name, _)| selected_feature_names.contains(**name))
        .map(|(name, proof)| (*name, proof.clone()))
        .collect();

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

    for language in ["c", "c++"] {
        compile_and_check_feature_tu(
            object,
            "hoisted",
            language,
            &feature_source,
            &feature_translation_unit,
            &selected_feature_proofs,
        );
        compile_and_check_feature_tu(
            object,
            "per_use",
            language,
            &per_use_source,
            &per_use_translation_unit,
            &selected_feature_proofs,
        );
    }
}

#[test]
fn scalar_extension_real_coff_pipeline_covers_result_chains_fanout_and_both_placements() {
    let object = stage3_extension_fixture_object();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("build Stage-3 scalar fixture pool");
    let db = run_fixture_pipeline(object, &pool);
    let mut positive_proofs = BTreeMap::new();
    for (name, extension, encoded_width, result_chain) in [
        (
            "stage3_movsxd64",
            ScalarMemoryExtension::SignExtend,
            8,
            ScalarMemoryResultChain::Direct,
        ),
        (
            "stage3_movsx32_wide",
            ScalarMemoryExtension::SignExtend,
            4,
            ScalarMemoryResultChain::ZeroUpper32,
        ),
        (
            "stage3_movzx32_wide",
            ScalarMemoryExtension::ZeroExtend,
            4,
            ScalarMemoryResultChain::ZeroUpper32,
        ),
        (
            "stage3_plain32_fanout",
            ScalarMemoryExtension::ImplicitZeroExtend,
            4,
            ScalarMemoryResultChain::ZeroUpper32,
        ),
    ] {
        let proofs = proofs_in_function(&db, name);
        assert_eq!(proofs.len(), 1, "{name}: {proofs:#?}");
        let proof = &proofs[0];
        assert_eq!(proof.extension, extension, "{name}");
        assert_eq!(proof.encoded_destination_width, Some(encoded_width), "{name}");
        assert_eq!(proof.result_chain, Some(result_chain), "{name}");
        assert_eq!(proof.value_width, 8, "{name}");
        assert!(proof.is_closed_v1(), "{name}: {proof:#?}");
        assert_extension_clight_candidates(&db, name, proof);
        if name == "stage3_plain32_fanout" {
            assert_final_root_store_fanout(&db, proof, 3);
        }
        positive_proofs.insert(name, proof.clone());
    }
    assert!(
        proofs_in_function(&db, "stage3_partial_merge_control").is_empty(),
        "partial-register merge acquired a Stage-3 proof",
    );

    let canonical_tu = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("Stage-3 fixture emitted no canonical translation unit");
    let canonical_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            canonical_tu,
            BinaryFormat::Coff,
        );
    let patched_source_path = object.with_file_name("stage3-patched-canonical.c");
    std::fs::write(&patched_source_path, &canonical_source)
        .expect("write patched Stage-3 canonical source oracle candidate");
    assert_eq!(
        sha256_file(&patched_source_path),
        BD762_CANONICAL_SOURCE_SHA256,
        "patched Stage-3 canonical source differs from immutable provider {} tree {} binary {} ({} bytes, build {}, identity {})",
        BD762_PROVIDER_COMMIT,
        BD762_PROVIDER_TREE,
        BD762_PROVIDER_BINARY_SHA256,
        BD762_PROVIDER_BINARY_SIZE,
        BD762_PROVIDER_BUILD_ID,
        BD762_PROVIDER_IDENTITY_SHA256,
    );
    assert_eq!(
        canonical_source, BD762_CANONICAL_SOURCE,
        "patched Stage-3 canonical C bytes differ from the checked-in bd762 golden",
    );
    let patched_clight_path = object.with_file_name("stage3-patched-canonical.clight.json");
    manifold::export_clight_json(
        &db,
        patched_clight_path
            .to_str()
            .expect("Stage-3 Clight oracle path is UTF-8"),
    )
    .expect("export patched Stage-3 canonical Clight AST");
    assert_eq!(
        sha256_file(&patched_clight_path),
        BD762_CANONICAL_CLIGHT_JSON_SHA256,
        "patched selected Clight AST bytes differ from the immutable bd762 oracle",
    );
    let replay_db = run_fixture_pipeline(object, &pool);
    let replay_tu = replay_db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("Stage-3 canonical replay emitted no translation unit");
    assert_eq!(
        replay_tu, canonical_tu,
        "patched Stage-3 canonical C AST is nondeterministic",
    );
    let replay_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            replay_tu,
            BinaryFormat::Coff,
        );
    assert_eq!(
        replay_source, BD762_CANONICAL_SOURCE,
        "patched Stage-3 replay differs from immutable bd762 canonical C bytes",
    );
    let field_proofs = proofs_in_function(&db, "stage3_field_profile");
    assert_eq!(field_proofs.len(), 1, "field fixture proof drifted: {field_proofs:#?}");
    let field_proof = field_proofs[0].clone();
    assert_eq!(field_proof.extension, ScalarMemoryExtension::Plain);
    let field_snapshots: Vec<_> = db
        .cast_source_alternatives
        .iter()
        .filter(|snapshot| {
            snapshot.manifold_address == field_proof.function
                && matches!(
                    snapshot.boundary,
                    SourceAlternativeBoundary::ScalarFieldPreVarReduce
                        | SourceAlternativeBoundary::ScalarFieldPostVarReduce
                )
        })
        .collect();
    let field_profile_emitted = !field_snapshots.is_empty();
    let mut field_tu = canonical_tu.clone();
    let field_canonical = function_definition(canonical_tu, "stage3_field_profile");
    if field_snapshots.is_empty() {
        let raw_profile_survives = db.cast_source_alternatives.iter().any(|snapshot| {
            snapshot.manifold_address == field_proof.function
                && matches!(
                    snapshot.boundary,
                    SourceAlternativeBoundary::ScalarLvaluePreVarReduce
                        | SourceAlternativeBoundary::ScalarLvaluePostVarReduce
                )
        });
        assert!(
            raw_profile_survives,
            "field/canonical collision must not evict the independent raw profile"
        );
    } else {
        assert!(matches!(field_snapshots.len(), 1 | 2));
        let field_snapshot = field_snapshots
            .iter()
            .find(|snapshot| {
                snapshot.boundary == SourceAlternativeBoundary::ScalarFieldPostVarReduce
            })
            .copied()
            .unwrap_or(field_snapshots[0]);
        assert_ne!(field_canonical, &field_snapshot.function);
        replace_function_definition(
            &mut field_tu,
            &field_canonical.name,
            &field_snapshot.function,
        );
    }
    let field_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            &field_tu,
            BinaryFormat::Coff,
        );
    assert_selected_field_feature(
        &field_tu,
        &field_source,
        "stage3_field_profile",
        2,
    );
    let field_compile_proofs = if field_profile_emitted {
        BTreeMap::from([("stage3_field_profile", field_proof.clone())])
    } else {
        // The existing canonical field layout was byte-identical to the
        // feature view and therefore correctly deduplicated. Compile the
        // complete TU, but do not attribute Clang's commutative reordering of
        // its two canonical loads to a non-emitted feature alternative.
        BTreeMap::new()
    };
    let mut per_use_tu = canonical_tu.clone();
    let mut hoisted_tu = canonical_tu.clone();
    let mut emitted_extension_names = BTreeSet::new();
    for (name, proof) in &positive_proofs {
        let matching_identity = |snapshot: &&manifold::decompile::postselect::source_alternatives::SourceAlternativeSnapshot| {
            snapshot.manifold_address == proof.function
                && (snapshot.manifold_name == *name
                    || snapshot.manifold_name == format!("coff_fn_{name}"))
        };
        let family = |per_use: bool| {
            db.cast_source_alternatives
                .iter()
                .filter(|snapshot| {
                    matching_identity(snapshot)
                        && if per_use {
                            matches!(
                                snapshot.boundary,
                                SourceAlternativeBoundary::ScalarExtensionPerUsePreVarReduce
                                    | SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce
                            )
                        } else {
                            matches!(
                                snapshot.boundary,
                                SourceAlternativeBoundary::ScalarExtensionHoistedPreVarReduce
                                    | SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce
                            )
                        }
                })
                .collect::<Vec<_>>()
        };
        let per_use = family(true);
        let hoisted = family(false);
        if per_use.is_empty() || hoisted.is_empty() {
            assert!(
                per_use.is_empty() && hoisted.is_empty(),
                "{name}: content collapse left a partial extension superfamily"
            );
            continue;
        }
        assert!(matches!(per_use.len(), 1 | 2));
        assert!(matches!(hoisted.len(), 1 | 2));
        emitted_extension_names.insert(*name);
        let canonical = function_definition(canonical_tu, name);
        let per_use = choose_collapsed_snapshot(
            &per_use,
            SourceAlternativeBoundary::ScalarExtensionPerUsePostVarReduce,
        );
        let hoisted = choose_collapsed_snapshot(
            &hoisted,
            SourceAlternativeBoundary::ScalarExtensionHoistedPostVarReduce,
        );
        assert_ne!(canonical, &per_use.function, "{name}: per-use equals canonical");
        assert_ne!(canonical, &hoisted.function, "{name}: hoisted equals canonical");
        replace_function_definition(&mut per_use_tu, &canonical.name, &per_use.function);
        replace_function_definition(&mut hoisted_tu, &canonical.name, &hoisted.function);
    }
    assert!(
        emitted_extension_names.contains("stage3_movsx32_wide"),
        "real COFF suite must retain at least one complete four-record extension superfamily"
    );
    let per_use_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            &per_use_tu,
            BinaryFormat::Coff,
        );
    let hoisted_source =
        manifold::decompile::passes::c_pass::print_translation_unit_for_format(
            &hoisted_tu,
            BinaryFormat::Coff,
        );
    for name in &emitted_extension_names {
        let proof = positive_proofs
            .get(name)
            .expect("emitted extension retains its authenticated proof");
        assert_selected_per_use_feature_lvalue(
            &per_use_tu,
            &per_use_source,
            name,
            proof,
        );
        assert_selected_feature_lvalue(&hoisted_tu, &hoisted_source, name, proof);
    }
    let emitted_extension_proofs: BTreeMap<_, _> = positive_proofs
        .iter()
        .filter(|(name, _)| emitted_extension_names.contains(**name))
        .map(|(name, proof)| (*name, proof.clone()))
        .collect();
    for language in ["c", "c++"] {
        compile_and_check_feature_tu(
            object,
            "stage3_per_use",
            language,
            &per_use_source,
            &per_use_tu,
            &emitted_extension_proofs,
        );
        compile_and_check_feature_tu(
            object,
            "stage3_hoisted",
            language,
            &hoisted_source,
            &hoisted_tu,
            &emitted_extension_proofs,
        );
        compile_and_check_feature_tu(
            object,
            "stage3_field",
            language,
            &field_source,
            &field_tu,
            &field_compile_proofs,
        );
    }
}
