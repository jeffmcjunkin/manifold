use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::c_pass::print::{IntegerModel, PrintConfig, Printer};
use manifold::decompile::passes::c_pass::types::{FuncDef, TopLevelDecl, TranslationUnit};
use manifold::decompile::passes::clight_select::select::Stage4SourceProfile;
use manifold::decompile::postselect::source_alternatives::{
    exact_function_addresses, final_source_alternative_exclusions, render_manifest,
    SourceAlternativeBoundary, SourceAlternativeSnapshot, MAX_SOURCE_ALTERNATIVES_PER_FUNCTION_V4,
    SOURCE_ALTERNATIVES_SCHEMA_V4,
};
use manifold::mreg::Mreg;
use manifold::x86::types::{
    Address, Node, RTLInst, Stage4RootBoundary, Stage4SourceKind, Stage4SourceProof,
    Stage4TerminalUse, Stage4UsePlan, Symbol,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// Immutable Stage-3 authority used to generate the canonical fixture oracle.
// The test binds the exact assembly/object plus canonical C, Clight, and RTL
// bytes below; these identity fields make that provenance independently
// auditable without requiring the 150 MiB provider bundle at test runtime.
const C8_PROVIDER_COMMIT: &str = "c8f34f5d964b5a5bd4377dfb47ecc63702c8717c";
const C8_PROVIDER_TREE: &str = "531b634096862c0ef56c25900fd041b0615653fc";
const C8_PROVIDER_BUILD_ID: &str = "manifold-5c5cf24c5c967c44ee9c3e2f";
const C8_PROVIDER_BINARY_SHA256: &str =
    "93716de3cb495a449236d9285fe66476fb83bbb8168069fa08f4ab73bc2ae86b";
const C8_PROVIDER_BINARY_SIZE: u64 = 157_273_264;
const C8_PROVIDER_IDENTITY_SHA256: &str =
    "2fc6df66fb970fd3a36d3b488ef068a53198dda94780d6ad7291330561248d7d";
const C8_ORACLE_MANIFEST_SHA256: &str =
    "4d62ecef924f6f943f8f4ad7f020c91720475c45d1af528f8639839fdcd4fdb5";
const C8_FIXTURE_ASSEMBLY_SHA256: &str =
    "f07f8ff0bc5197adf8d6b48f5b04690cd37a20dfc32135d52a752d68245325a4";
const C8_FIXTURE_OBJECT_SHA256: &str =
    "bae2b0415aa18d9619973f772daa6e41c3287d6968704fb793cdd918375bbd21";
const C8_CANONICAL_SOURCE_SHA256: &str =
    "9aa905b6959a9a0983b44f4b37d50d5d5153e8919ca24fcbd49dd939bf3a5dcb";
const C8_CANONICAL_CLIGHT_JSON_SHA256: &str =
    "b5d33620cdd98da8f6d9efbe556a81994242cd998c6abcfeac05bb23342942cc";
const C8_CANONICAL_RTL_SHA256: &str =
    "801f290a31d8321ca4e8a47b2d41284b7607d4d94306494d4d0cb90788db0dd7";
const C8_V3_SIDECAR_SHA256: &str =
    "cf09a8d82f8cff039c7f212e4bd2b99aff6170924dc31c42196454fb9cccc46a";
const C8_V3_ORDINARY_PROJECTION_SHA256: &str =
    "3e394a48ee0731d44ca945f1228e68bab542991f230c02a62de6ae995a96c2a3";
const C8_CANONICAL_SOURCE: &str = include_str!("fixtures/stage4_source_forms_c8f.c");

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
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

fn append_projection_field(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_be_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn ordinary_manifest_projection(records: &[serde_json::Value]) -> Vec<u8> {
    let mut output = Vec::new();
    for record in records
        .iter()
        .filter(|record| record.get("profile_ordinal").is_none())
    {
        append_projection_field(&mut output, record["id"].as_str().expect("ordinary id"));
        output.extend_from_slice(
            &record["function_ordinal"]
                .as_u64()
                .expect("ordinary function ordinal")
                .to_be_bytes(),
        );
        for field in ["manifold_name", "manifold_address", "boundary"] {
            append_projection_field(
                &mut output,
                record[field].as_str().expect("ordinary string identity"),
            );
        }
        let kinds = record["kinds"].as_array().expect("ordinary kinds");
        output.extend_from_slice(&(kinds.len() as u64).to_be_bytes());
        for kind in kinds {
            append_projection_field(&mut output, kind.as_str().expect("ordinary kind"));
        }
        for field in [
            "canonical_source_sha256",
            "alternative_source_sha256",
            "canonical_source",
            "alternative_source",
        ] {
            append_projection_field(
                &mut output,
                record[field].as_str().expect("ordinary source field"),
            );
        }
    }
    output
}

fn fixture_assembly() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/stage4_source_forms.s"
    ))
}

fn build_fixture() -> PathBuf {
    assert!(
        command_exists("clang"),
        "Stage-4 integration prerequisite missing: clang is unavailable"
    );
    let assembly = fixture_assembly();
    assert_eq!(sha256_file(assembly), C8_FIXTURE_ASSEMBLY_SHA256);
    let directory = std::env::temp_dir().join(format!(
        "manifold_stage4_source_forms_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Stage-4 fixture directory");
    let object = directory.join("stage4_source_forms.obj");
    let output = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(assembly)
        .arg("-o")
        .arg(&object)
        .output()
        .expect("assemble Stage-4 COFF fixture");
    assert!(
        output.status.success(),
        "Stage-4 fixture failed to assemble:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        sha256_file(&object),
        C8_FIXTURE_OBJECT_SHA256,
        "fixture bytes drifted from immutable provider {} tree {} binary {} ({} bytes, build {}, identity {}, oracle manifest {})",
        C8_PROVIDER_COMMIT,
        C8_PROVIDER_TREE,
        C8_PROVIDER_BINARY_SHA256,
        C8_PROVIDER_BINARY_SIZE,
        C8_PROVIDER_BUILD_ID,
        C8_PROVIDER_IDENTITY_SHA256,
        C8_ORACLE_MANIFEST_SHA256,
    );
    object
}

fn fixture_object() -> &'static Path {
    static OBJECT: OnceLock<PathBuf> = OnceLock::new();
    OBJECT.get_or_init(build_fixture).as_path()
}

fn run_pipeline(object: &Path, pool: &rayon::ThreadPool) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, false, false));
    db
}

fn fixture_name(name: &str) -> &str {
    name.strip_prefix("coff_fn_").unwrap_or(name)
}

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    let emitted = format!("coff_fn_{name}");
    // The immutable input object starts with the raw fixture symbol and the
    // COFF loader publishes its provider identity with one `coff_fn_` prefix.
    // A rebuilt object starts from that emitted provider name, so the same
    // loader deterministically adds the second prefix.  Admit exactly those
    // three closed identities and still require one unambiguous span.
    let rebuilt = format!("coff_fn_{emitted}");
    let spans: Vec<_> = db
        .rel_iter::<(Symbol, Address, Address)>("func_span")
        .filter_map(|(symbol, start, end)| {
            (*symbol == name || *symbol == emitted || *symbol == rebuilt).then_some((*start, *end))
        })
        .collect();
    let [span] = spans.as_slice() else {
        panic!(
            "Stage-4 fixture function {name} has {} authenticated spans, expected exactly one: {spans:?}",
            spans.len(),
        )
    };
    *span
}

fn in_span(node: Node, span: (Address, Address)) -> bool {
    node >= span.0 && node < span.1
}

fn proofs_in_function(
    db: &DecompileDB,
    relation: &'static str,
    name: &str,
) -> Vec<Stage4SourceProof> {
    let span = function_span(db, name);
    let mut proofs: Vec<_> = db
        .rel_iter::<(Node, Stage4SourceProof)>(relation)
        .filter(|(node, proof)| {
            in_span(*node, span)
                && proof.function == span.0
                && proof.selected_node == *node
                && in_span(proof.origin_node, span)
        })
        .map(|(_, proof)| proof.clone())
        .collect();
    proofs.sort();
    proofs
}

fn plans_in_function(db: &DecompileDB, relation: &'static str, name: &str) -> Vec<Stage4UsePlan> {
    let span = function_span(db, name);
    let mut plans: Vec<_> = db
        .rel_iter::<(Node, Stage4UsePlan)>(relation)
        .filter(|(node, plan)| {
            in_span(*node, span)
                && plan.function == span.0
                && plan.definition_node == *node
                && in_span(plan.placement_node, span)
        })
        .map(|(_, plan)| plan.clone())
        .collect();
    plans.sort();
    plans
}

fn rtl_rows_at(db: &DecompileDB, relation: &'static str, node: Node) -> Vec<RTLInst> {
    db.rel_iter::<(Node, RTLInst)>(relation)
        .filter(|(candidate, _)| *candidate == node)
        .map(|(_, instruction)| instruction.clone())
        .collect()
}

fn successors_at(db: &DecompileDB, relation: &'static str, node: Node) -> Vec<Node> {
    let mut rows: Vec<_> = db
        .rel_iter::<(Node, Node)>(relation)
        .filter_map(|(source, target)| (*source == node).then_some(*target))
        .collect();
    rows.sort_unstable();
    rows
}

fn raw_instruction_at(db: &DecompileDB, node: Node) -> (usize, &'static str) {
    let rows: Vec<_> = db
        .rel_iter::<(
            Address,
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
        .filter(|(candidate, ..)| *candidate == node)
        .collect();
    let [row] = rows.as_slice() else {
        panic!("Stage-4 node 0x{node:x} has non-unique raw instruction rows: {rows:#?}");
    };
    (row.1, row.3)
}

fn function_definition<'a>(translation_unit: &'a TranslationUnit, name: &str) -> &'a FuncDef {
    let emitted = format!("coff_fn_{name}");
    translation_unit
        .decls
        .iter()
        .find_map(|declaration| match declaration {
            TopLevelDecl::FuncDef(function)
                if function.name == name || function.name == emitted =>
            {
                Some(function)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("canonical Stage-4 TU lost function {name}"))
}

fn replace_function_definition(
    translation_unit: &mut TranslationUnit,
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

fn one_function_source(function: &FuncDef) -> String {
    let mut config = PrintConfig::default();
    config.integer_model = IntegerModel::MsvcLlp64;
    let mut printer = Printer::new(config);
    printer.print_func_def(function);
    printer.into_string()
}

fn canonical_rtl_dump(db: &DecompileDB) -> String {
    let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        owners.entry(*node).or_default().insert(*function);
    }
    let mut names: BTreeMap<Address, BTreeSet<&str>> = BTreeMap::new();
    for (name, function) in db.rel_iter::<(Symbol, Address)>("func_entry") {
        names.entry(*function).or_default().insert(*name);
    }
    let mut functions: BTreeMap<Address, Vec<(Node, RTLInst)>> = BTreeMap::new();
    for (node, instruction) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        let Some(function_owners) = owners.get(node) else {
            continue;
        };
        if function_owners.len() != 1 {
            panic!("canonical RTL node 0x{node:x} has ambiguous function ownership");
        }
        let function = *function_owners.iter().next().expect("one owner");
        functions
            .entry(function)
            .or_default()
            .push((*node, instruction.clone()));
    }
    let mut output = String::from(";; RTL IR dump\n\n");
    for (function, rows) in &mut functions {
        rows.sort_by_key(|(node, _)| *node);
        assert!(
            rows.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "canonical RTL fixture contains candidate rows"
        );
        let function_names = names
            .get(function)
            .unwrap_or_else(|| panic!("RTL function 0x{function:x} has no name"));
        if function_names.len() != 1 {
            panic!("RTL function 0x{function:x} has ambiguous names");
        }
        let name = function_names.iter().next().expect("one function name");
        output.push_str(&format!(";; ---- {name} (0x{function:x}) ----\n"));
        for (node, instruction) in rows {
            output.push_str(&format!("  0x{node:x}:  {instruction:?}\n"));
        }
        output.push('\n');
    }
    output
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

fn expected_boundaries(kind: Stage4SourceKind) -> [SourceAlternativeBoundary; 2] {
    match kind {
        Stage4SourceKind::AffineAddress => [
            SourceAlternativeBoundary::Stage4AffinePreVarReduce,
            SourceAlternativeBoundary::Stage4AffinePostVarReduce,
        ],
        Stage4SourceKind::Zeroing => [
            SourceAlternativeBoundary::Stage4ZeroPreVarReduce,
            SourceAlternativeBoundary::Stage4ZeroPostVarReduce,
        ],
        Stage4SourceKind::Add
        | Stage4SourceKind::Sub
        | Stage4SourceKind::Mul
        | Stage4SourceKind::And
        | Stage4SourceKind::Or
        | Stage4SourceKind::Xor => [
            SourceAlternativeBoundary::Stage4RmwPreVarReduce,
            SourceAlternativeBoundary::Stage4RmwPostVarReduce,
        ],
    }
}

fn expected_kinds(boundary: SourceAlternativeBoundary) -> &'static [&'static str] {
    match boundary {
        SourceAlternativeBoundary::Stage4AffinePreVarReduce => {
            &["address_expression", "local_lifetime"]
        }
        SourceAlternativeBoundary::Stage4AffinePostVarReduce => &["address_expression"],
        SourceAlternativeBoundary::Stage4ZeroPreVarReduce => &["local_lifetime", "zeroing"],
        SourceAlternativeBoundary::Stage4ZeroPostVarReduce => &["zeroing"],
        SourceAlternativeBoundary::Stage4RmwPreVarReduce => {
            &["compound_assignment", "local_lifetime"]
        }
        SourceAlternativeBoundary::Stage4RmwPostVarReduce => &["compound_assignment"],
        _ => panic!("non-Stage-4 boundary in Stage-4 profile"),
    }
}

fn compound_token(kind: Stage4SourceKind) -> Option<&'static str> {
    match kind {
        Stage4SourceKind::Add => Some(" += "),
        Stage4SourceKind::Sub => Some(" -= "),
        Stage4SourceKind::Mul => Some(" *= "),
        Stage4SourceKind::And => Some(" &= "),
        Stage4SourceKind::Or => Some(" |= "),
        Stage4SourceKind::Xor => Some(" ^= "),
        Stage4SourceKind::AffineAddress | Stage4SourceKind::Zeroing => None,
    }
}

fn compound_token_count(source: &str) -> usize {
    [" += ", " -= ", " *= ", " &= ", " |= ", " ^= "]
        .into_iter()
        .map(|token| source.matches(token).count())
        .sum()
}

fn choose_collapsed_snapshot<'a>(
    rows: &[&'a SourceAlternativeSnapshot],
) -> &'a SourceAlternativeSnapshot {
    rows.iter()
        .find(|snapshot| {
            matches!(
                snapshot.boundary,
                SourceAlternativeBoundary::Stage4AffinePostVarReduce
                    | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
                    | SourceAlternativeBoundary::Stage4RmwPostVarReduce
            )
        })
        .copied()
        .unwrap_or(rows[0])
}

fn x86_register_width(register: &str) -> Option<usize> {
    if ["AL", "BL", "CL", "DL", "SIL", "DIL", "BPL", "SPL"].contains(&register)
        || register.ends_with('B') && register.starts_with('R')
    {
        Some(1)
    } else if ["AX", "BX", "CX", "DX", "SI", "DI", "BP", "SP"].contains(&register)
        || register.ends_with('W') && register.starts_with('R')
    {
        Some(2)
    } else if register.starts_with('E') || register.ends_with('D') && register.starts_with('R') {
        Some(4)
    } else if ["RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP"].contains(&register)
        || register
            .strip_prefix('R')
            .is_some_and(|suffix| suffix.parse::<u8>().is_ok())
    {
        Some(8)
    } else {
        None
    }
}

fn rebuilt_mnemonics(db: &DecompileDB, name: &str) -> BTreeSet<&'static str> {
    let span = function_span(db, name);
    db.rel_iter::<(
        Address,
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
        .filter_map(|(node, _, _, mnemonic, ..)| in_span(*node, span).then_some(*mnemonic))
        .collect()
}

fn rebuilt_has_nonstack_memory_effect(
    db: &DecompileDB,
    name: &str,
    maximum_width: usize,
    exact_width: bool,
    read: bool,
    write: bool,
) -> bool {
    let span = function_span(db, name);
    let reads: BTreeSet<_> = db
        .rel_iter::<(Node, Symbol)>("decoded_memory_read_operand")
        .copied()
        .collect();
    let writes: BTreeSet<_> = db
        .rel_iter::<(Node, Symbol)>("decoded_memory_write_operand")
        .copied()
        .collect();
    db.rel_iter::<(
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )>("op_indirect")
        .any(|(operand, segment, base, _, _, _, width)| {
            *segment == "NONE"
                && !matches!(*base, "NONE" | "RIP" | "RSP" | "RBP")
                && (if exact_width {
                    *width == maximum_width
                } else {
                    *width != 0 && *width <= maximum_width
                })
                && db
                    .rel_iter::<(
                        Address,
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
                            && [*op1, *op2, *op3, *op4].contains(operand)
                            && reads.contains(&(*node, *operand)) == read
                            && writes.contains(&(*node, *operand)) == write
                    })
        })
}

fn rebuilt_has_affine_lea(db: &DecompileDB, name: &str, width: usize) -> bool {
    let span = function_span(db, name);
    type DecodedInstruction = (
        Address,
        usize,
        &'static str,
        &'static str,
        Symbol,
        Symbol,
        Symbol,
        Symbol,
        usize,
        usize,
    );
    let mut instructions: BTreeMap<Address, Vec<DecodedInstruction>> = BTreeMap::new();
    for row in db
        .rel_iter::<DecodedInstruction>("unrefinedinstruction")
        .filter(|row| in_span(row.0, span))
    {
        instructions.entry(row.0).or_default().push(*row);
    }
    let mut registers: BTreeMap<Symbol, Vec<&'static str>> = BTreeMap::new();
    for (operand, register) in db.rel_iter::<(Symbol, &'static str)>("op_register") {
        registers.entry(*operand).or_default().push(*register);
    }
    type IndirectOperand = (
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    );
    let mut addresses: BTreeMap<Symbol, Vec<IndirectOperand>> = BTreeMap::new();
    for row in db.rel_iter::<(
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )>("op_indirect")
    {
        addresses.entry(row.0).or_default().push(*row);
    }
    let mut immediates: BTreeMap<Symbol, Vec<(Symbol, i64, usize)>> = BTreeMap::new();
    for row in db.rel_iter::<(Symbol, i64, usize)>("op_immediate") {
        immediates.entry(row.0).or_default().push(*row);
    }
    let decoded_defs: BTreeMap<_, Vec<_>> = {
        let mut rows: BTreeMap<Address, Vec<Mreg>> = BTreeMap::new();
        for (node, register) in db.rel_iter::<(Address, Mreg)>("decoded_reg_def") {
            if in_span(*node, span) {
                rows.entry(*node).or_default().push(*register);
            }
        }
        for values in rows.values_mut() {
            values.sort();
        }
        rows
    };
    let decoded_uses: BTreeMap<_, Vec<_>> = {
        let mut rows: BTreeMap<Address, Vec<Mreg>> = BTreeMap::new();
        for (node, register) in db.rel_iter::<(Address, Mreg)>("decoded_reg_use") {
            if in_span(*node, span) {
                rows.entry(*node).or_default().push(*register);
            }
        }
        for values in rows.values_mut() {
            values.sort();
        }
        rows
    };
    let memory_reads: Vec<_> = db
        .rel_iter::<(Address, Symbol)>("decoded_memory_read_operand")
        .filter(|(node, _)| in_span(*node, span))
        .copied()
        .collect();
    let memory_writes: Vec<_> = db
        .rel_iter::<(Address, Symbol)>("decoded_memory_write_operand")
        .filter(|(node, _)| in_span(*node, span))
        .copied()
        .collect();

    let mut matches = 0usize;
    for (node, instruction_rows) in &instructions {
        let [instruction] = instruction_rows.as_slice() else {
            continue;
        };
        let (_, size, _, mnemonic, op1, op2, op3, op4, _, _) = *instruction;
        if mnemonic != "LEA" || op3 != "0" || op4 != "0" {
            continue;
        }
        // The loader swaps Capstone's first two Intel-order operands before it
        // publishes unrefinedinstruction: LEA is [address, destination].
        let Some([destination_name]) = registers.get(op2).map(Vec::as_slice) else {
            continue;
        };
        let destination = Mreg::x86(destination_name);
        let destination_is_exact = !destination.is_unknown()
            && x86_register_width(destination_name) == Some(width)
            && decoded_defs.get(node).map(Vec::as_slice) == Some(&[destination]);
        let Some([(_, segment, base, index, scale, displacement, address_width)]) =
            addresses.get(op1).map(Vec::as_slice)
        else {
            continue;
        };
        let base_register = Mreg::x86(base);
        let index_register = Mreg::x86(index);
        let lea_memory_reads: Vec<_> = memory_reads
            .iter()
            .filter_map(|(owner, operand)| (*owner == *node).then_some(*operand))
            .collect();
        let address_is_exact = *segment == "NONE"
            && !matches!(*base, "NONE" | "RIP" | "RSP" | "RBP")
            && *index != "NONE"
            && !base_register.is_unknown()
            && !index_register.is_unknown()
            && *scale == 4
            && *address_width == width
            && decoded_uses.get(node).is_some_and(|uses| {
                uses.as_slice() == [base_register, index_register]
                    || uses.as_slice() == [index_register, base_register]
            })
            && lea_memory_reads.as_slice() == [op1]
            && memory_writes.iter().all(|(owner, _)| owner != node);
        if !destination_is_exact || !address_is_exact {
            continue;
        }
        let final_node = if *displacement == 12 {
            Some(*node)
        } else if *displacement == 0 {
            let Some(next_node) = node.checked_add(size as u64) else {
                continue;
            };
            let Some([next_instruction]) = instructions.get(&next_node).map(Vec::as_slice) else {
                continue;
            };
            let (_, _, _, next_mnemonic, next_op1, next_op2, next_op3, next_op4, _, _) =
                *next_instruction;
            (next_mnemonic == "ADD"
                && next_op3 == "0"
                && next_op4 == "0"
                // ADD-immediate follows the same loader contract: [immediate,
                // destination], not Capstone's original Intel order.
                && registers.get(next_op2).map(Vec::as_slice) == Some(&[*destination_name])
                && immediates.get(next_op1).map(Vec::as_slice) == Some(&[(next_op1, 12, 0)])
                && decoded_uses.get(&next_node).map(Vec::as_slice) == Some(&[destination])
                && decoded_defs.get(&next_node).map(Vec::as_slice) == Some(&[destination])
                && memory_reads.iter().all(|(owner, _)| *owner != next_node)
                && memory_writes.iter().all(|(owner, _)| *owner != next_node))
            .then_some(next_node)
        } else {
            None
        };
        let Some(final_node) = final_node else {
            continue;
        };
        let return_nodes: Vec<_> = instructions
            .iter()
            .filter_map(|(candidate, rows)| {
                let [(_, _, _, mnemonic, ..)] = rows.as_slice() else {
                    return None;
                };
                (*candidate > final_node && *mnemonic == "RET").then_some(*candidate)
            })
            .collect();
        let [return_node] = return_nodes.as_slice() else {
            continue;
        };
        let carrier_uses: Vec<_> = decoded_uses
            .iter()
            .filter_map(|(candidate, uses)| {
                (*candidate > final_node
                    && *candidate < *return_node
                    && uses.contains(&destination))
                .then_some(*candidate)
            })
            .collect();
        let mut store_displacements = Vec::new();
        let stores_are_exact = carrier_uses.iter().all(|candidate| {
            let Some([(_, _, _, mnemonic, op1, op2, op3, op4, _, _)]) =
                instructions.get(candidate).map(Vec::as_slice)
            else {
                return false;
            };
            let operands = [*op1, *op2, *op3, *op4];
            let writes: Vec<_> = memory_writes
                .iter()
                .filter_map(|(owner, operand)| (*owner == *candidate).then_some(*operand))
                .collect();
            let [write] = writes.as_slice() else {
                return false;
            };
            let Some([(_, segment, base, index, scale, displacement, memory_width)]) =
                addresses.get(write).map(Vec::as_slice)
            else {
                return false;
            };
            let base_register = Mreg::x86(base);
            let Some(uses) = decoded_uses.get(candidate) else {
                return false;
            };
            let mut expected_uses = vec![destination, base_register];
            expected_uses.sort();
            let source_operands = operands
                .iter()
                .filter(|operand| {
                    registers.get(**operand).map(Vec::as_slice) == Some(&[*destination_name])
                })
                .count();
            let exact = *mnemonic == "MOV"
                && operands.contains(write)
                && source_operands == 1
                && uses == &expected_uses
                && memory_reads.iter().all(|(owner, _)| *owner != *candidate)
                && *segment == "NONE"
                && !matches!(*base, "NONE" | "RIP" | "RSP" | "RBP")
                && !base_register.is_unknown()
                && *index == "NONE"
                && *scale == 1
                && *memory_width == width;
            if exact {
                store_displacements.push(*displacement);
            }
            exact
        });
        store_displacements.sort();
        let roles_are_exact = destination == Mreg::AX
            && decoded_defs.iter().all(|(candidate, definitions)| {
                *candidate <= final_node
                    || *candidate >= *return_node
                    || !definitions.contains(&destination)
            })
            && carrier_uses.len() == 2
            && stores_are_exact
            && store_displacements == [0, width as i64];
        if roles_are_exact {
            matches += 1;
        }
    }
    matches == 1 && !rebuilt_mnemonics(db, name).contains("CALL")
}

#[test]
fn rebuilt_affine_oracle_is_closed_to_fused_or_exact_split_lowering() {
    #[derive(Clone, Copy)]
    struct Fixture {
        split: bool,
        displacement: i64,
        width: usize,
        lea_register: &'static str,
        add_register: &'static str,
        add_gap: u64,
        lea_extra: &'static str,
        add_immediate: i64,
        add_immediate_width: usize,
        add_memory: bool,
        decoy: bool,
        reversed_loader_slots: bool,
    }

    let fixture = |shape: Fixture| {
        const START: Address = 0x1000;
        const LEA_SIZE: usize = 4;
        const ADD_SIZE: usize = 3;
        let add = START + LEA_SIZE as u64 + shape.add_gap;
        let final_node = if shape.split { add } else { START };
        let final_size = if shape.split { ADD_SIZE } else { LEA_SIZE };
        let store0 = final_node + final_size as u64;
        let store1 = store0 + 3;
        let ret = store1 + 4;
        let decoy = ret + 1;
        let end = if shape.decoy { decoy + 1 } else { ret + 1 };
        let (lea_op1, lea_op2) = if shape.reversed_loader_slots {
            ("lea-dst", "lea-address")
        } else {
            ("lea-address", "lea-dst")
        };
        let mut db = DecompileDB::default();
        db.rel_push("func_span", ("coff_fn_fixture", START, end));
        db.rel_push(
            "unrefinedinstruction",
            (
                START,
                LEA_SIZE,
                "",
                "LEA",
                lea_op1,
                lea_op2,
                shape.lea_extra,
                "0",
                0usize,
                0usize,
            ),
        );
        db.rel_push("op_register", ("lea-dst", shape.lea_register));
        db.rel_push(
            "op_indirect",
            (
                "lea-address",
                "NONE",
                "RCX",
                "RDX",
                4i64,
                shape.displacement,
                shape.width,
            ),
        );
        db.rel_push("decoded_reg_def", (START, Mreg::x86(shape.lea_register)));
        db.rel_push("decoded_reg_use", (START, Mreg::CX));
        db.rel_push("decoded_reg_use", (START, Mreg::DX));
        db.rel_push("decoded_memory_read_operand", (START, "lea-address"));
        if shape.split {
            let (add_op1, add_op2) = if shape.reversed_loader_slots {
                ("add-dst", "add-immediate")
            } else {
                ("add-immediate", "add-dst")
            };
            db.rel_push(
                "unrefinedinstruction",
                (
                    add, ADD_SIZE, "", "ADD", add_op1, add_op2, "0", "0", 0usize, 0usize,
                ),
            );
            db.rel_push("op_register", ("add-dst", shape.add_register));
            db.rel_push(
                "op_immediate",
                (
                    "add-immediate",
                    shape.add_immediate,
                    shape.add_immediate_width,
                ),
            );
            db.rel_push("decoded_reg_use", (add, Mreg::x86(shape.add_register)));
            db.rel_push("decoded_reg_def", (add, Mreg::x86(shape.add_register)));
            if shape.add_memory {
                db.rel_push("decoded_memory_read_operand", (add, "hidden-memory"));
                db.rel_push(
                    "op_indirect",
                    (
                        "hidden-memory",
                        "NONE",
                        "RCX",
                        "NONE",
                        1i64,
                        0i64,
                        shape.width,
                    ),
                );
            }
        }
        for (node, source, output, displacement) in [
            (store0, "store-source-0", "store-output-0", 0i64),
            (
                store1,
                "store-source-1",
                "store-output-1",
                shape.width as i64,
            ),
        ] {
            db.rel_push(
                "unrefinedinstruction",
                (
                    node,
                    if node == store0 { 3usize } else { 4usize },
                    "",
                    "MOV",
                    source,
                    output,
                    "0",
                    "0",
                    0usize,
                    0usize,
                ),
            );
            db.rel_push("op_register", (source, shape.lea_register));
            db.rel_push(
                "op_indirect",
                (
                    output,
                    "NONE",
                    "R8",
                    "NONE",
                    1i64,
                    displacement,
                    shape.width,
                ),
            );
            db.rel_push("decoded_reg_use", (node, Mreg::x86(shape.lea_register)));
            db.rel_push("decoded_reg_use", (node, Mreg::R8));
            db.rel_push("decoded_memory_write_operand", (node, output));
        }
        db.rel_push(
            "unrefinedinstruction",
            (ret, 1usize, "", "RET", "0", "0", "0", "0", 0usize, 0usize),
        );
        if shape.decoy {
            db.rel_push(
                "unrefinedinstruction",
                (
                    decoy,
                    1usize,
                    "",
                    "LEA",
                    "decoy-address",
                    "decoy-dst",
                    "0",
                    "0",
                    0usize,
                    0usize,
                ),
            );
            db.rel_push("op_register", ("decoy-dst", "EDX"));
            db.rel_push(
                "op_indirect",
                ("decoy-address", "NONE", "RCX", "RDX", 4i64, 12i64, 4usize),
            );
            db.rel_push("decoded_reg_def", (decoy, Mreg::DX));
            db.rel_push("decoded_reg_use", (decoy, Mreg::CX));
            db.rel_push("decoded_reg_use", (decoy, Mreg::DX));
            db.rel_push("decoded_memory_read_operand", (decoy, "decoy-address"));
        }
        db
    };
    let fused = Fixture {
        split: false,
        displacement: 12,
        width: 4,
        lea_register: "EAX",
        add_register: "EAX",
        add_gap: 0,
        lea_extra: "0",
        add_immediate: 12,
        add_immediate_width: 0,
        add_memory: false,
        decoy: false,
        reversed_loader_slots: false,
    };
    assert!(rebuilt_has_affine_lea(&fixture(fused), "fixture", 4));
    let split = Fixture {
        split: true,
        displacement: 0,
        ..fused
    };
    assert!(rebuilt_has_affine_lea(&fixture(split), "fixture", 4));
    assert!(rebuilt_has_affine_lea(
        &fixture(Fixture {
            decoy: true,
            ..split
        }),
        "fixture",
        4,
    ));
    for invalid in [
        Fixture {
            add_register: "EDX",
            ..split
        },
        Fixture {
            width: 8,
            lea_register: "RAX",
            add_register: "RAX",
            ..split
        },
        Fixture {
            displacement: 8,
            split: false,
            ..fused
        },
        Fixture {
            displacement: 8,
            split: false,
            decoy: true,
            ..fused
        },
        Fixture {
            add_gap: 1,
            ..split
        },
        Fixture {
            lea_extra: "forged-extra",
            ..split
        },
        Fixture {
            add_immediate: 11,
            ..split
        },
        Fixture {
            add_immediate_width: 4,
            ..split
        },
        Fixture {
            add_memory: true,
            ..split
        },
        // This is the unswapped Capstone/Intel spelling. The loader contract
        // is the opposite ordering and must reject it.
        Fixture {
            reversed_loader_slots: true,
            ..split
        },
    ] {
        assert!(!rebuilt_has_affine_lea(&fixture(invalid), "fixture", 4));
    }
}

fn compile_feature_tu(
    directory: &Path,
    label: &str,
    language: &str,
    translation_unit: &TranslationUnit,
) -> DecompileDB {
    let source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        translation_unit,
        BinaryFormat::Coff,
    );
    let source = if language == "c++" {
        format!("extern \"C\" {{\n{source}\n}}\n")
    } else {
        source
    };
    let extension = if language == "c++" { "cpp" } else { "c" };
    let source_path = directory.join(format!("{label}.{extension}"));
    let object_path = directory.join(format!("{label}-{extension}.obj"));
    std::fs::write(&source_path, &source).expect("write Stage-4 feature source");
    let output = Command::new("clang")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-O1",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            language,
            "-c",
        ])
        .arg(&source_path)
        .arg("-o")
        .arg(&object_path)
        .output()
        .unwrap_or_else(|error| panic!("compile Stage-4 {label} {language}: {error}"));
    assert!(
        output.status.success(),
        "Stage-4 {label} {language} did not compile:\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        source,
    );
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, &object_path);
    manifold::decompile::disassembly::load_preset(&mut db);
    db
}

fn native_harness(two_roots_kind: Stage4SourceKind) -> String {
    let (two_first, two_second) = match two_roots_kind {
        Stage4SourceKind::Add => (7, 6),
        Stage4SourceKind::Mul => (4, 30),
        _ => panic!("two-root harness received non-RMW profile"),
    };
    format!(
        r#"
int main(void)
{{
    __int64 affine64_storage[2] = {{0, 0}};
    if (coff_fn_stage4_affine64(10, 2, (struct struct_2 *)affine64_storage) != 30 ||
        affine64_storage[0] != 30 || affine64_storage[1] != 30) return 1;
    struct struct_1 affine32_storage = {{0, 0}};
    if (coff_fn_stage4_affine32(10, 2, &affine32_storage) != 30 ||
        affine32_storage.ofs_0 != 30 || affine32_storage.ofs_4 != 30) return 2;
    struct struct_4 qword = {{5}};
    if (coff_fn_stage4_add_reg64(&qword, 7) != 12) return 3;
    if (coff_fn_stage4_add_imm64(&qword) != 12) return 4;
    int dword = 9;
    if (coff_fn_stage4_sub_imm32(&dword) != 4) return 5;
    qword.ofs_0 = 6;
    if (coff_fn_stage4_mul_reg64(&qword, 7) != 42) return 6;
    dword = 6;
    if (coff_fn_stage4_mul_imm32(&dword) != 18) return 7;
    struct struct_4 and_in = {{0x123}}, and_out = {{0}};
    if (coff_fn_stage4_and_imm64(&and_in, &and_out, 77) != 77 ||
        and_out.ofs_0 != 0x23) return 8;
    qword.ofs_0 = 0x10;
    if (coff_fn_stage4_or_reg64(&qword, 3) != 0x13) return 9;
    qword.ofs_0 = 0x1f;
    if (coff_fn_stage4_xor_imm64(&qword) != 0) return 10;
    struct struct_1 two_in = {{4, 6}}, two_out = {{0, 0}};
    if (coff_fn_stage4_two_roots(&two_in, 0, &two_out, 91) != 91 ||
        two_out.ofs_0 != {two_first} || two_out.ofs_4 != {two_second}) return 11;
    return 0;
}}
"#
    )
}

fn compile_and_run_native_feature_tu(
    directory: &Path,
    label: &str,
    language: &str,
    translation_unit: &TranslationUnit,
    two_roots_kind: Stage4SourceKind,
) {
    let source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        translation_unit,
        BinaryFormat::Coff,
    );
    let source = if language == "c++" {
        format!(
            "extern \"C\" {{\n{source}\n}}\n{}",
            native_harness(two_roots_kind)
        )
    } else {
        format!("{source}\n{}", native_harness(two_roots_kind))
    };
    let extension = if language == "c++" { "cpp" } else { "c" };
    let source_path = directory.join(format!("{label}-native.{extension}"));
    let executable = directory.join(format!("{label}-native-{extension}"));
    std::fs::write(&source_path, &source).expect("write native Stage-4 semantic oracle");
    let output = Command::new("clang")
        .args(["-O1", "-fms-extensions", "-Wno-everything", "-x", language])
        .arg(&source_path)
        .arg("-o")
        .arg(&executable)
        .output()
        .unwrap_or_else(|error| panic!("compile native Stage-4 {label} {language}: {error}"));
    assert!(
        output.status.success(),
        "native Stage-4 {label} {language} did not compile:\n{}\n{source}",
        String::from_utf8_lossy(&output.stderr)
    );
    let executed = Command::new(&executable)
        .output()
        .unwrap_or_else(|error| panic!("execute native Stage-4 {label} {language}: {error}"));
    assert!(
        executed.status.success(),
        "native Stage-4 {label} {language} semantic oracle failed with {:?}:\nstdout:\n{}\nstderr:\n{}",
        executed.status.code(),
        String::from_utf8_lossy(&executed.stdout),
        String::from_utf8_lossy(&executed.stderr),
    );
}

fn assert_rebuilt_profile(
    db: &DecompileDB,
    name: &str,
    proof: &Stage4SourceProof,
    plan: &Stage4UsePlan,
) {
    match proof.kind {
        Stage4SourceKind::AffineAddress => {
            assert!(
                rebuilt_has_affine_lea(db, name, proof.width),
                "{name}: rebuilt source lost exact {}-bit base+index*4+12 LEA",
                proof.width * 8
            );
        }
        Stage4SourceKind::Zeroing => {
            panic!("canonical-identical zeroing profile reached the v4 wire")
        }
        kind => {
            // Clang may legally fold the load into the compound operation,
            // strength-reduce multiplication to LEA, or narrow an AND mask
            // to MOVZX. The immutable input object above carries the exact
            // load -> destructive operation -> terminal proof. Keep this
            // rebuilt-COFF oracle limited to a closed lowering family and
            // memory/terminal effects; the native harness below proves the
            // emitted C and C++ semantics for every retained source row.
            let accepted_mnemonics: &[&str] = match kind {
                Stage4SourceKind::Add => &["ADD", "LEA"],
                Stage4SourceKind::Sub => &["SUB", "ADD", "LEA"],
                Stage4SourceKind::Mul => &["IMUL", "LEA"],
                Stage4SourceKind::And => &["AND", "MOVZX"],
                Stage4SourceKind::Or => &["OR"],
                Stage4SourceKind::Xor => &["XOR"],
                Stage4SourceKind::AffineAddress | Stage4SourceKind::Zeroing => unreachable!(),
            };
            let mnemonics = rebuilt_mnemonics(db, name);
            assert!(
                accepted_mnemonics
                    .iter()
                    .any(|mnemonic| mnemonics.contains(mnemonic)),
                "{name}: rebuilt {kind:?} escaped its sealed lowering family {accepted_mnemonics:?}: {mnemonics:?}"
            );
            assert!(
                !mnemonics.contains("CALL"),
                "{name}: rebuilt feature unexpectedly introduced a call"
            );
            assert!(
                rebuilt_has_nonstack_memory_effect(db, name, proof.width, false, true, false,),
                "{name}: rebuilt {kind:?} lost its nonstack source read"
            );
            match plan
                .terminal_use
                .as_ref()
                .expect("rebuilt RMW profile has terminal role")
            {
                Stage4TerminalUse::Return => {
                    assert!(
                        mnemonics.contains("RET"),
                        "{name}: rebuilt return profile has no RET"
                    );
                }
                Stage4TerminalUse::Store { .. } => {
                    assert!(
                        rebuilt_has_nonstack_memory_effect(
                            db,
                            name,
                            proof.width,
                            true,
                            false,
                            true,
                        ),
                        "{name}: rebuilt store profile lost its exact-width nonstack write"
                    );
                }
            }
        }
    }
}

#[test]
fn stage4_real_coff_pipeline_preserves_primary_and_emits_bounded_independent_profiles() {
    let object = fixture_object();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("build Stage-4 fixture pool");
    let db = run_pipeline(object, &pool);

    let expected = [
        (
            "stage4_affine64",
            Stage4SourceKind::AffineAddress,
            8,
            Stage4RootBoundary::FinalRtlDefinition,
            1usize,
            1usize,
        ),
        (
            "stage4_affine32",
            Stage4SourceKind::AffineAddress,
            4,
            Stage4RootBoundary::FinalRtlDefinition,
            1,
            1,
        ),
        (
            "stage4_zero_xor32",
            Stage4SourceKind::Zeroing,
            4,
            Stage4RootBoundary::FinalRtlDefinition,
            1,
            1,
        ),
        (
            "stage4_zero_sub64",
            Stage4SourceKind::Zeroing,
            8,
            Stage4RootBoundary::FinalRtlDefinition,
            1,
            1,
        ),
        (
            "stage4_add_reg32",
            Stage4SourceKind::Add,
            4,
            Stage4RootBoundary::EliminatedMutation,
            1,
            0,
        ),
        (
            "stage4_add_reg64",
            Stage4SourceKind::Add,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_add_imm64",
            Stage4SourceKind::Add,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_sub_imm32",
            Stage4SourceKind::Sub,
            4,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_mul_reg64",
            Stage4SourceKind::Mul,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_mul_imm32",
            Stage4SourceKind::Mul,
            4,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_and_imm64",
            Stage4SourceKind::And,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_or_reg64",
            Stage4SourceKind::Or,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_xor_imm64",
            Stage4SourceKind::Xor,
            8,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_two_roots",
            Stage4SourceKind::Add,
            4,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
        (
            "stage4_two_roots",
            Stage4SourceKind::Mul,
            4,
            Stage4RootBoundary::EliminatedMutation,
            1,
            1,
        ),
    ];
    let controls = [
        "stage4_memory_rmw_control",
        "stage4_three_operand_mul_control",
        "stage4_stack_lea_control",
        "stage4_narrow_zero_control",
    ];

    let mut expected_late_by_profile = BTreeMap::new();
    let mut machine_count = 0usize;
    let mut late_count = 0usize;
    for (name, kind, width, boundary, expected_machine, expected_late) in expected {
        let machine: Vec<_> = proofs_in_function(&db, "authenticated_stage4_source", name)
            .into_iter()
            .filter(|proof| proof.kind == kind)
            .collect();
        let late: Vec<_> = proofs_in_function(&db, "stage4_source_candidate", name)
            .into_iter()
            .filter(|proof| proof.kind == kind)
            .collect();
        assert_eq!(machine.len(), expected_machine, "{name}: {machine:#?}");
        assert_eq!(late.len(), expected_late, "{name}: {late:#?}");
        machine_count += machine.len();
        late_count += late.len();
        let machine_nodes: BTreeSet<_> = machine.iter().map(|proof| proof.selected_node).collect();
        let late_nodes: BTreeSet<_> = late.iter().map(|proof| proof.selected_node).collect();
        let machine_plans: Vec<_> = plans_in_function(&db, "authenticated_stage4_use_plan", name)
            .into_iter()
            .filter(|plan| machine_nodes.contains(&plan.definition_node))
            .collect();
        let late_plans: Vec<_> = plans_in_function(&db, "stage4_use_plan", name)
            .into_iter()
            .filter(|plan| late_nodes.contains(&plan.definition_node))
            .collect();
        assert_eq!(
            machine_plans.len(),
            expected_machine,
            "{name}: {machine_plans:#?}"
        );
        assert_eq!(late_plans.len(), expected_late, "{name}: {late_plans:#?}");
        for proof in &machine {
            assert_eq!(proof.kind, kind, "{name}");
            assert_eq!(proof.width, width, "{name}");
            assert_eq!(proof.root_boundary, boundary, "{name}");
            assert!(proof.is_closed_v1(), "{name}: {proof:#?}");
            let matching: Vec<_> = machine_plans
                .iter()
                .filter(|plan| plan.definition_node == proof.selected_node)
                .collect();
            let [plan] = matching.as_slice() else {
                panic!("{name}: proof has no unique final-RTL use plan: {matching:#?}");
            };
            assert!(plan.is_closed_v1(proof), "{name}: {plan:#?}");
            if boundary == Stage4RootBoundary::EliminatedMutation {
                let placement = plan
                    .placement_load
                    .as_ref()
                    .unwrap_or_else(|| panic!("{name}: eliminated plan lacks placement load"));
                assert!(plan.transports.is_empty(), "{name}");
                assert_eq!(plan.sites.len(), 1, "{name}");
                let site = &plan.sites[0];
                assert_eq!(site.value, plan.value, "{name}");

                let placement_final = rtl_rows_at(&db, "rtl_inst", plan.placement_node);
                assert!(
                    matches!(
                        placement_final.as_slice(),
                        [RTLInst::Iload(chunk, addressing, args, value)]
                            if chunk == &placement.chunk
                                && addressing == &placement.addressing
                                && args == &placement.args
                                && value == &plan.value
                    ),
                    "{name}: placement is not the exact surviving final load: {placement_final:#?}"
                );
                assert!(
                    rtl_rows_at(&db, "rtl_inst", proof.selected_node).is_empty(),
                    "{name}: eliminated mutation unexpectedly survived final RTL"
                );
                let root_preopt =
                    rtl_rows_at(&db, "stage4_preopt_selected_rtl", proof.selected_node);
                assert!(
                    matches!(
                        root_preopt.as_slice(),
                        [RTLInst::Iop(operation, args, result)]
                            if operation == &proof.operation
                                && args.as_ref() == proof.args.as_ref()
                                && result == &proof.selected_result
                    ),
                    "{name}: pre-optimization root is not the authenticated destructive op: {root_preopt:#?}"
                );
                let terminal_final = rtl_rows_at(&db, "rtl_inst", site.node);
                let terminal_preopt = rtl_rows_at(&db, "stage4_preopt_selected_rtl", site.node);
                assert_eq!(terminal_preopt, terminal_final, "{name}: terminal drifted");
                match plan
                    .terminal_use
                    .as_ref()
                    .unwrap_or_else(|| panic!("{name}: eliminated plan lacks terminal role"))
                {
                    Stage4TerminalUse::Return => {
                        assert!(
                            matches!(terminal_final.as_slice(), [RTLInst::Ireturn(value)] if value == &plan.value),
                            "{name}: terminal is not the exact carrier return: {terminal_final:#?}"
                        );
                        assert_eq!(raw_instruction_at(&db, site.node).1, "RET", "{name}");
                    }
                    Stage4TerminalUse::Store {
                        chunk,
                        addressing,
                        args,
                    } => {
                        assert!(
                            matches!(
                                terminal_final.as_slice(),
                                [RTLInst::Istore(final_chunk, final_addressing, final_args, value)]
                                    if final_chunk == chunk
                                        && final_addressing == addressing
                                        && final_args == args
                                        && value == &plan.value
                            ),
                            "{name}: terminal is not the exact carrier store: {terminal_final:#?}"
                        );
                        assert_eq!(raw_instruction_at(&db, site.node).1, "MOV", "{name}");
                    }
                }
                assert_eq!(
                    successors_at(&db, "stage4_preopt_selected_succ", plan.placement_node),
                    vec![proof.selected_node],
                    "{name}: placement does not have exact preopt D->R edge"
                );
                assert_eq!(
                    successors_at(&db, "stage4_preopt_selected_succ", proof.selected_node),
                    vec![site.node],
                    "{name}: root does not have exact preopt R->U edge"
                );
                assert_eq!(
                    successors_at(&db, "rtl_succ", plan.placement_node),
                    vec![site.node],
                    "{name}: final load does not flow directly to its sole terminal"
                );
                let placement_size = raw_instruction_at(&db, plan.placement_node).0 as u64;
                let root_size = raw_instruction_at(&db, proof.selected_node).0 as u64;
                assert_eq!(
                    plan.placement_node.checked_add(placement_size),
                    Some(proof.selected_node),
                    "{name}: placement and eliminated root are not byte-contiguous"
                );
                assert_eq!(
                    proof.selected_node.checked_add(root_size),
                    Some(site.node),
                    "{name}: eliminated root and terminal are not byte-contiguous"
                );
            }
        }
        for proof in late {
            let matching_plans: Vec<_> = late_plans
                .iter()
                .filter(|plan| plan.definition_node == proof.selected_node)
                .collect();
            let [plan] = matching_plans.as_slice() else {
                panic!("{name}: late proof lacks one exact late plan: {matching_plans:#?}");
            };
            expected_late_by_profile.insert(
                (
                    name.to_string(),
                    Stage4SourceProfile {
                        kind: proof.kind,
                        root_node: proof.selected_node,
                    },
                ),
                (proof, (*plan).clone()),
            );
        }
    }
    assert_eq!(
        machine_count, 15,
        "real final-RTL Stage-4 proof inventory drifted"
    );
    assert_eq!(
        late_count, 14,
        "late type/ABI-gated Stage-4 proof inventory drifted"
    );
    for name in controls {
        assert!(
            proofs_in_function(&db, "authenticated_stage4_source", name).is_empty(),
            "negative fixture {name} acquired a Stage-4 machine proof"
        );
        assert!(
            proofs_in_function(&db, "stage4_source_candidate", name).is_empty(),
            "negative fixture {name} acquired a late Stage-4 source candidate"
        );
    }

    let canonical_tu = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("Stage-4 pipeline emitted no canonical C translation unit");
    let canonical_source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        canonical_tu,
        BinaryFormat::Coff,
    );
    let canonical_path = object.with_file_name("stage4-patched-canonical.c");
    std::fs::write(&canonical_path, &canonical_source).expect("write Stage-4 canonical oracle");
    assert_eq!(sha256_file(&canonical_path), C8_CANONICAL_SOURCE_SHA256);
    assert_eq!(
        canonical_source, C8_CANONICAL_SOURCE,
        "Stage-4 changed the canonical primary emitted by immutable c8 authority"
    );
    let clight_path = object.with_file_name("stage4-patched-canonical.clight.json");
    manifold::export_clight_json(
        &db,
        clight_path
            .to_str()
            .expect("Stage-4 Clight oracle path is UTF-8"),
    )
    .expect("export Stage-4 canonical Clight AST");
    assert_eq!(sha256_file(&clight_path), C8_CANONICAL_CLIGHT_JSON_SHA256);
    let ir_base = object.with_file_name("stage4-patched-canonical");
    let rtl_path = ir_base.with_extension("rtl");
    std::fs::write(&rtl_path, canonical_rtl_dump(&db)).expect("write Stage-4 canonical RTL dump");
    assert_eq!(
        sha256_file(&rtl_path),
        C8_CANONICAL_RTL_SHA256,
        "Stage-4 changed immutable c8 final RTL before private source selection"
    );

    let ordinary_rows: Vec<_> = db
        .cast_source_alternatives
        .iter()
        .filter(|snapshot| !is_stage4_boundary(snapshot.boundary))
        .cloned()
        .collect();
    assert_eq!(
        ordinary_rows.len(),
        27,
        "the cumulative v3 portfolio drifted from immutable sidecar {C8_V3_SIDECAR_SHA256}"
    );
    let stage4_rows: Vec<_> = db
        .cast_source_alternatives
        .iter()
        .filter(|snapshot| is_stage4_boundary(snapshot.boundary))
        .collect();
    assert!(
        stage4_rows.iter().all(|snapshot| {
            snapshot.stage4_profile.is_some()
                && snapshot.kinds.as_slice() == expected_kinds(snapshot.boundary)
        }),
        "v4 row lost its closed profile/boundary/kind identity"
    );
    assert!(
        stage4_rows.iter().all(|snapshot| {
            !matches!(
                snapshot.boundary,
                SourceAlternativeBoundary::Stage4ZeroPreVarReduce
                    | SourceAlternativeBoundary::Stage4ZeroPostVarReduce
            )
        }),
        "canonical-identical zero forms must be content-deduped before wire capture"
    );

    let mut groups: BTreeMap<(String, Stage4SourceProfile), Vec<&SourceAlternativeSnapshot>> =
        BTreeMap::new();
    for snapshot in stage4_rows {
        groups
            .entry((
                fixture_name(&snapshot.manifold_name).to_string(),
                snapshot.stage4_profile.expect("filtered Stage-4 profile"),
            ))
            .or_default()
            .push(snapshot);
    }
    assert_eq!(
        groups.len(),
        12,
        "v4 profile inventory drifted: {groups:#?}"
    );
    assert_eq!(
        groups
            .keys()
            .filter(|(name, _)| name == "stage4_two_roots")
            .count(),
        2,
        "same-function disjoint roots were merged or dropped"
    );
    assert!(
        !groups.keys().any(|(name, _)| name == "stage4_add_reg32"),
        "late signed/type-mismatched register RHS reached the v4 source portfolio"
    );

    let mut emitted_sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut selected_profiles = BTreeMap::new();
    for ((name, profile), rows) in &mut groups {
        rows.sort_by_key(|snapshot| snapshot.boundary);
        assert!(matches!(rows.len(), 1 | 2), "{name}/{profile:?}: {rows:#?}");
        let allowed = expected_boundaries(profile.kind);
        assert!(
            rows.iter()
                .all(|snapshot| allowed.contains(&snapshot.boundary)),
            "{name}/{profile:?} crossed its closed family boundary"
        );
        if rows.len() == 2 {
            assert_eq!(
                rows.iter()
                    .map(|snapshot| snapshot.boundary)
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(allowed),
                "{name}/{profile:?} did not retain an atomic pre/post pair"
            );
        }
        let canonical = function_definition(canonical_tu, name);
        for snapshot in rows.iter() {
            let source = one_function_source(&snapshot.function);
            assert_ne!(source, one_function_source(canonical), "{name}/{profile:?}");
            assert!(
                emitted_sources
                    .entry(name.clone())
                    .or_default()
                    .insert(source.clone()),
                "{name}: duplicate printed source crossed ordinary/feature profile identity"
            );
            if let Some(token) = compound_token(profile.kind) {
                assert_eq!(
                    source.matches(token).count(),
                    1,
                    "{name}/{profile:?}: {source}"
                );
                assert_eq!(
                    compound_token_count(&source),
                    1,
                    "{name}/{profile:?}: {source}"
                );
            }
            if name == "stage4_two_roots" {
                assert_eq!(
                    compound_token_count(&source),
                    1,
                    "two roots formed a cross-product"
                );
                assert_ne!(source.contains(" += "), source.contains(" *= "));
            }
        }
        let chosen = choose_collapsed_snapshot(rows);
        let (proof, plan) = expected_late_by_profile
            .get(&(name.clone(), *profile))
            .unwrap_or_else(|| panic!("wire profile lacks exact late proof {name}/{profile:?}"));
        selected_profiles.insert(
            (name.clone(), *profile),
            (chosen.function.clone(), proof.clone(), plan.clone()),
        );
    }
    for snapshot in &ordinary_rows {
        let name = fixture_name(&snapshot.manifold_name).to_string();
        let source = one_function_source(&snapshot.function);
        assert!(
            emitted_sources
                .entry(name.clone())
                .or_default()
                .insert(source),
            "{name}: v4 feature source collided with retained cumulative-v3 source"
        );
    }

    let exact_identities: Vec<_> =
        exact_function_addresses(&db.cast_selected_functions, &db.cast_id_to_name)
            .into_iter()
            .collect();
    let exclusions = final_source_alternative_exclusions(&db, &exact_identities);
    let ordinary_manifest = render_manifest(
        canonical_tu,
        &ordinary_rows,
        db.cast_source_alternatives_overflowed,
        &exclusions,
        &exact_identities,
        &canonical_source,
        BinaryFormat::Coff,
    )
    .expect("render cumulative v3 source-alternative manifest")
    .expect("immutable cumulative v3 portfolio unexpectedly disappeared");
    let ordinary_manifest_path = object.with_file_name("stage4-cumulative-v3.json");
    std::fs::write(&ordinary_manifest_path, format!("{ordinary_manifest}\n"))
        .expect("write cumulative v3 source-alternative oracle");
    assert_eq!(
        sha256_file(&ordinary_manifest_path),
        C8_V3_SIDECAR_SHA256,
        "Stage-4 changed any ordered v3 boundary/kind/source/digest byte"
    );
    let manifest = render_manifest(
        canonical_tu,
        &db.cast_source_alternatives,
        db.cast_source_alternatives_overflowed,
        &exclusions,
        &exact_identities,
        &canonical_source,
        BinaryFormat::Coff,
    )
    .expect("render v4 source-alternative manifest")
    .expect("valid Stage-4 fixture unexpectedly fell back to no manifest");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest).expect("parse v4 source-alternative manifest");
    assert_eq!(manifest["schema"], SOURCE_ALTERNATIVES_SCHEMA_V4);
    assert_eq!(
        manifest["max_per_function"].as_u64(),
        Some(MAX_SOURCE_ALTERNATIVES_PER_FUNCTION_V4 as u64)
    );
    assert_eq!(manifest["truncated"], false);
    let records = manifest["alternatives"]
        .as_array()
        .expect("v4 alternatives array");
    let ordinary_record_count = records
        .iter()
        .filter(|record| record.get("profile_ordinal").is_none())
        .count();
    assert_eq!(ordinary_record_count, 27);
    let ordinary_projection_path = object.with_file_name("stage4-v4-ordinary-projection.bin");
    std::fs::write(
        &ordinary_projection_path,
        ordinary_manifest_projection(records),
    )
    .expect("write v4 ordinary-row projection");
    assert_eq!(
        sha256_file(&ordinary_projection_path),
        C8_V3_ORDINARY_PROJECTION_SHA256,
        "v4 serialization changed retained v3 row identity/source bytes"
    );
    let mut manifest_profiles: BTreeMap<String, BTreeMap<u64, BTreeSet<String>>> = BTreeMap::new();
    for record in records
        .iter()
        .filter(|record| record.get("profile_ordinal").is_some())
    {
        let name = fixture_name(
            record["manifold_name"]
                .as_str()
                .expect("v4 record manifold_name"),
        )
        .to_string();
        let ordinal = record["profile_ordinal"]
            .as_u64()
            .expect("v4 record profile ordinal");
        let root = record["profile_root_node"]
            .as_str()
            .expect("v4 record profile root")
            .to_string();
        let id = record["id"].as_str().expect("v4 record id");
        assert!(id.ends_with(&format!(":profile-{ordinal:04}")));
        manifest_profiles
            .entry(name)
            .or_default()
            .entry(ordinal)
            .or_default()
            .insert(root);
    }
    assert_eq!(
        manifest_profiles.values().map(BTreeMap::len).sum::<usize>(),
        12
    );
    for (name, profiles) in &manifest_profiles {
        assert_eq!(
            profiles.keys().copied().collect::<Vec<_>>(),
            (0..profiles.len() as u64).collect::<Vec<_>>(),
            "{name}: profile ordinals are not contiguous"
        );
        let span = function_span(&db, name);
        for roots in profiles.values() {
            assert_eq!(roots.len(), 1, "one profile has competing wire roots");
            let root =
                u64::from_str_radix(roots.iter().next().unwrap().trim_start_matches("0x"), 16)
                    .expect("hex profile root");
            assert!(in_span(root, span), "{name}: wire root escaped mapped span");
        }
    }

    let mut profiles_by_function: BTreeMap<
        String,
        Vec<(
            Stage4SourceProfile,
            Vec<FuncDef>,
            Stage4SourceProof,
            Stage4UsePlan,
        )>,
    > = BTreeMap::new();
    for ((name, profile), (_, proof, plan)) in selected_profiles {
        let functions = groups[&(name.clone(), profile)]
            .iter()
            .map(|snapshot| snapshot.function.clone())
            .collect();
        profiles_by_function
            .entry(name)
            .or_default()
            .push((profile, functions, proof, plan));
    }
    for profiles in profiles_by_function.values_mut() {
        profiles.sort_by_key(|(profile, _, _, _)| *profile);
    }
    let build_directory = object
        .parent()
        .expect("Stage-4 object has parent directory");
    let expected_compiled_rows: BTreeSet<_> = groups
        .iter()
        .flat_map(|((name, profile), rows)| {
            rows.iter()
                .map(move |snapshot| (name.clone(), *profile, snapshot.boundary))
        })
        .collect();
    let mut compiled_rows = BTreeSet::new();
    for root_variant in 0..2 {
        for boundary_variant in 0..2 {
            let mut feature_tu = canonical_tu.clone();
            let mut selected = Vec::new();
            for (name, profiles) in &profiles_by_function {
                let profile_index = if name == "stage4_two_roots" {
                    root_variant
                } else {
                    0
                };
                let (profile, functions, proof, plan) = &profiles[profile_index];
                let function_index = boundary_variant.min(functions.len() - 1);
                let function = &functions[function_index];
                let snapshot = groups[&(name.clone(), *profile)][function_index];
                let canonical = function_definition(canonical_tu, name);
                replace_function_definition(&mut feature_tu, &canonical.name, function);
                compiled_rows.insert((name.clone(), *profile, snapshot.boundary));
                selected.push((name.clone(), *profile, proof.clone(), plan.clone()));
            }
            for language in ["c", "c++"] {
                let label = format!("stage4-feature-{root_variant}-{boundary_variant}");
                let rebuilt = compile_feature_tu(build_directory, &label, language, &feature_tu);
                for (name, _, proof, plan) in &selected {
                    assert_rebuilt_profile(&rebuilt, name, proof, plan);
                }
                let two_roots = selected
                    .iter()
                    .find(|(name, _, _, _)| name == "stage4_two_roots")
                    .expect("selected disjoint-root profile");
                compile_and_run_native_feature_tu(
                    build_directory,
                    &label,
                    language,
                    &feature_tu,
                    two_roots.2.kind,
                );
            }
        }
    }
    assert_eq!(compiled_rows, expected_compiled_rows);
}
