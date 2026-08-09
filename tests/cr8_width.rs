use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::x86::types::{Address, CminorUnop, CsharpminorExpr, CsharpminorStmt, RTLReg, Symbol};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> PathBuf {
    assert!(
        command_exists("clang"),
        "CR8 width integration prerequisite missing: clang is unavailable"
    );
    let directory = std::env::temp_dir().join(format!("manifold_cr8_width_{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create CR8 width fixture directory");
    let source = directory.join("fixture.s");
    let object = directory.join("fixture.obj");
    std::fs::write(
        &source,
        r#"
        # Encoding bytes are used for MOV-from-CR8 so the fixture does not
        # depend on an assembler's privileged-register spelling.
        .section .text$cr8_width_positive,"xr"
        .globl cr8_width_positive
        .def cr8_width_positive; .scl 2; .type 32; .endef
cr8_width_positive:
        .byte 0x44, 0x0f, 0x20, 0xc0
        cmpb $1, %al
        jbe .Lpositive_low
        movl $2, %eax
        retq
.Lpositive_low:
        movl $1, %eax
        retq

        .section .text$cr8_width_wide,"xr"
        .globl cr8_width_wide
        .def cr8_width_wide; .scl 2; .type 32; .endef
cr8_width_wide:
        .byte 0x44, 0x0f, 0x20, 0xc0
        cmpl $1, %eax
        jbe .Lwide_low
        movl $2, %eax
        retq
.Lwide_low:
        movl $1, %eax
        retq

        .section .text$cr8_width_wrong_register,"xr"
        .globl cr8_width_wrong_register
        .def cr8_width_wrong_register; .scl 2; .type 32; .endef
cr8_width_wrong_register:
        .byte 0x44, 0x0f, 0x20, 0xc0
        cmpb $1, %cl
        jbe .Lwrong_register_low
        movl $2, %eax
        retq
.Lwrong_register_low:
        movl $1, %eax
        retq

        .section .text$cr8_width_wrong_constant,"xr"
        .globl cr8_width_wrong_constant
        .def cr8_width_wrong_constant; .scl 2; .type 32; .endef
cr8_width_wrong_constant:
        .byte 0x44, 0x0f, 0x20, 0xc0
        cmpb $2, %al
        jbe .Lwrong_constant_low
        movl $2, %eax
        retq
.Lwrong_constant_low:
        movl $1, %eax
        retq

        .section .text$cr8_width_nonadjacent,"xr"
        .globl cr8_width_nonadjacent
        .def cr8_width_nonadjacent; .scl 2; .type 32; .endef
cr8_width_nonadjacent:
        .byte 0x44, 0x0f, 0x20, 0xc0
        nop
        cmpb $1, %al
        jbe .Lnonadjacent_low
        movl $2, %eax
        retq
.Lnonadjacent_low:
        movl $1, %eax
        retq

        .section .text$cr8_width_repeated,"xr"
        .globl cr8_width_repeated
        .def cr8_width_repeated; .scl 2; .type 32; .endef
cr8_width_repeated:
        .byte 0x44, 0x0f, 0x20, 0xc0
        cmpb $1, %al
        ja .Lrepeated_second
        movl $1, %eax
        retq
.Lrepeated_second:
        .byte 0x44, 0x0f, 0x20, 0xc1
        cmpb $1, %cl
        jbe .Lrepeated_low
        movl $2, %eax
        retq
.Lrepeated_low:
        movl $1, %eax
        retq
"#,
    )
    .expect("write CR8 width fixture assembly");
    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("run clang over CR8 width fixture");
    assert!(status.success(), "CR8 width fixture assembly failed");
    object
}

fn fixture_object() -> Option<&'static Path> {
    if !command_exists("clang") {
        eprintln!("skipping CR8 width integration test: clang unavailable");
        return None;
    }
    static OBJECT: OnceLock<PathBuf> = OnceLock::new();
    Some(OBJECT.get_or_init(build_fixture).as_path())
}

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    let coff_name = format!("coff_fn_{name}");
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .find_map(|(symbol, start, end)| {
            (*symbol == name || *symbol == coff_name).then_some((*start, *end))
        })
        .unwrap_or_else(|| panic!("fixture function {name} has no authenticated span"))
}

fn in_span(node: Address, span: (Address, Address)) -> bool {
    node >= span.0 && node < span.1
}

fn compare_uses_register(
    db: &DecompileDB,
    span: (Address, Address),
    required_node: Option<Address>,
    register: &str,
    immediate: Option<i64>,
) -> bool {
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
        .any(|(node, _, _, mnemonic, op1, op2, op3, op4, _, _)| {
            in_span(*node, span)
                && required_node.is_none_or(|wanted| *node == wanted)
                && *mnemonic == "CMP"
                && [*op1, *op2, *op3, *op4].iter().any(|operand| {
                    db.rel_iter::<(Symbol, &'static str)>("op_register")
                        .any(|(candidate, name)| candidate == operand && *name == register)
                })
                && immediate.is_none_or(|wanted| {
                    [*op1, *op2, *op3, *op4].iter().any(|operand| {
                        db.rel_iter::<(Symbol, i64, usize)>("op_immediate")
                            .any(|(candidate, value, _)| candidate == operand && *value == wanted)
                    })
                })
        })
}

fn exact_register_operand(db: &DecompileDB, operand: Symbol, register: &str) -> bool {
    let rows: Vec<&'static str> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .filter_map(|(candidate, name)| (*candidate == operand).then_some(*name))
        .collect();
    rows == [register]
        && !db
            .rel_iter::<(Symbol, i64, usize)>("op_immediate")
            .any(|(candidate, _, _)| *candidate == operand)
        && !db
            .rel_iter::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>("op_indirect")
            .any(|(candidate, ..)| *candidate == operand)
}

fn exact_immediate_operand(db: &DecompileDB, operand: Symbol, value: i64) -> bool {
    let rows: Vec<(i64, usize)> = db
        .rel_iter::<(Symbol, i64, usize)>("op_immediate")
        .filter_map(|(candidate, value, width)| (*candidate == operand).then_some((*value, *width)))
        .collect();
    rows.len() == 1
        && rows[0].0 == value
        && !db
            .rel_iter::<(Symbol, &'static str)>("op_register")
            .any(|(candidate, _)| *candidate == operand)
        && !db
            .rel_iter::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>("op_indirect")
            .any(|(candidate, ..)| *candidate == operand)
}

fn zero_extends_al_before_compare(db: &DecompileDB, span: (Address, Address)) -> bool {
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
        .filter(|(node, ..)| in_span(*node, span))
        .collect();
    let zero_extensions: Vec<Address> = rows
        .iter()
        .filter_map(
            |(node, size, prefix, mnemonic, source, destination, op3, op4, meta0, meta1)| {
                (*size == 3
                    && *prefix == ""
                    && *mnemonic == "MOVZX"
                    && *op3 == "0"
                    && *op4 == "0"
                    && *meta0 == 0
                    && *meta1 == 0
                    && exact_register_operand(db, *source, "AL")
                    && exact_register_operand(db, *destination, "EAX"))
                .then_some(*node)
            },
        )
        .collect();
    let compares: Vec<Address> = rows
        .iter()
        .filter_map(
            |(node, _, _, mnemonic, immediate, register, op3, op4, _, _)| {
                (*mnemonic == "CMP"
                    && *op3 == "0"
                    && *op4 == "0"
                    && exact_immediate_operand(db, *immediate, 1)
                    && exact_register_operand(db, *register, "EAX"))
                .then_some(*node)
            },
        )
        .collect();
    zero_extensions.len() == 1
        && compares.len() == 1
        && db
            .rel_iter::<(Address, Address)>("next")
            .any(|(from, to)| *from == zero_extensions[0] && *to == compares[0])
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

#[test]
fn real_coff_cr8_low_byte_compare_is_narrowed_only_at_the_proved_site() {
    let Some(object) = fixture_object() else {
        return;
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("build CR8 width test pool");
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, false, false));

    let positive = function_span(&db, "cr8_width_positive");
    let positive_markers: Vec<(Address, RTLReg)> = db
        .rel_iter::<(Address, RTLReg)>("cr8_byte_compare")
        .filter(|(node, _)| in_span(*node, positive))
        .copied()
        .collect();
    assert_eq!(positive_markers.len(), 1, "{positive_markers:#x?}");
    let (compare, value) = positive_markers[0];
    assert!(
        compare_uses_register(&db, positive, Some(compare), "AL", Some(1)),
        "marker was not bound to the fixture's exact low-byte target CMP"
    );
    assert!(db
        .rel_iter::<(Address, CsharpminorStmt)>("csharp_stmt_candidate")
        .any(|(node, stmt)| {
            *node == compare
                && matches!(
                    stmt,
                    CsharpminorStmt::Scond(_, args, _, _)
                        if args.as_slice() == [CsharpminorExpr::Eunop(
                            CminorUnop::Ocast8unsigned,
                            Box::new(CsharpminorExpr::Evar(value)),
                        )]
                )
        }));

    for negative in [
        "cr8_width_wide",
        "cr8_width_wrong_register",
        "cr8_width_wrong_constant",
        "cr8_width_nonadjacent",
        "cr8_width_repeated",
    ] {
        let span = function_span(&db, negative);
        assert!(
            !db.rel_iter::<(Address, RTLReg)>("cr8_byte_compare")
                .any(|(node, _)| in_span(*node, span)),
            "negative fixture {negative} acquired a CR8 byte marker"
        );
    }

    let translation_unit = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("CR8 fixture pipeline emitted no translation unit");
    let source = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        translation_unit,
        BinaryFormat::Coff,
    );
    let positive_body = printed_function_definition(&source, "cr8_width_positive")
        .expect("final translation unit lost CR8 positive function");
    assert_eq!(
        positive_body.matches("__readcr8()").count(),
        1,
        "positive function must contain exactly one CR8 read:\n{positive_body}"
    );
    assert_eq!(
        positive_body.matches("(unsigned char)").count(),
        1,
        "proved comparison must contain exactly one unsigned-byte cast:\n{positive_body}"
    );
    let proved_condition = positive_body
        .lines()
        .find(|line| line.trim_start().starts_with("if (") && line.contains("<= 1"))
        .expect("positive function lost its proved CR8 condition");
    if let Some(assignment_lhs) = positive_body
        .lines()
        .find(|line| line.contains("__readcr8()") && line.contains(" = "))
        .and_then(|line| line.split_once(" = ").map(|(lhs, _)| lhs.trim()))
    {
        let local = assignment_lhs
            .split_whitespace()
            .next_back()
            .expect("CR8 assignment has no destination");
        assert!(
            positive_body
                .lines()
                .any(|line| line.contains("unsigned __int64") && line.contains(local)),
            "retained CR8 local lost its unsigned-64 declaration:\n{positive_body}"
        );
        assert!(
            proved_condition.contains(&format!("(unsigned char){local}"))
                && !proved_condition.contains("__readcr8()"),
            "retained CR8 local was not narrowed only at the proved condition:\n{positive_body}"
        );
    } else {
        assert!(
            proved_condition.contains("(unsigned char)__readcr8()"),
            "inlined CR8 read was not narrowed only at the proved condition:\n{positive_body}"
        );
    }
    assert!(
        source.contains("unsigned __int64 __readcr8(void);")
            && source.contains("#pragma intrinsic(__readcr8)"),
        "CR8 intrinsic declaration was globally narrowed:\n{source}"
    );

    let emitted = object.with_file_name("cr8_width_generated.c");
    let rebuilt = object.with_file_name("cr8_width_generated.obj");
    std::fs::write(&emitted, &source).expect("write emitted CR8 C");
    let compiled = Command::new("clang")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-O0",
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
        .expect("compile emitted CR8 C");
    assert!(
        compiled.status.success(),
        "emitted CR8 C did not compile:\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr),
        source
    );

    let mut rebuilt_db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut rebuilt_db, &rebuilt);
    manifold::decompile::disassembly::load_preset(&mut rebuilt_db);
    // The first COFF load emitted the provider-prefixed function name.  Loading
    // that rebuilt object adds the same namespace prefix a second time.
    let rebuilt_positive = function_span(&rebuilt_db, "coff_fn_cr8_width_positive");
    assert!(
        zero_extends_al_before_compare(&rebuilt_db, rebuilt_positive),
        "recompiled proved function did not preserve byte truncation before comparison:\n{positive_body}"
    );
}
