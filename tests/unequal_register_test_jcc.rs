use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::mreg::Mreg;
use manifold::x86::op::{Comparison, Condition};
use manifold::x86::types::{Address, MachInst, Symbol};

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping unequal-register TEST/Jcc test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_unequal_register_test_jcc_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl test_byte_je
        .def test_byte_je; .scl 2; .type 32; .endef
test_byte_je:
        testb %cl, %dl
        je test_byte_je_zero
        xorl %eax, %eax
        retq
test_byte_je_zero:
        movl $1, %eax
        retq

        .globl test_word_jne
        .def test_word_jne; .scl 2; .type 32; .endef
test_word_jne:
        testw %cx, %dx
        jne test_word_jne_nonzero
        xorl %eax, %eax
        retq
test_word_jne_nonzero:
        movl $1, %eax
        retq

        .globl test_dword_je
        .def test_dword_je; .scl 2; .type 32; .endef
test_dword_je:
        testl %ecx, %edx
        je test_dword_je_zero
        xorl %eax, %eax
        retq
test_dword_je_zero:
        movl $1, %eax
        retq

        .globl test_qword_jne
        .def test_qword_jne; .scl 2; .type 32; .endef
test_qword_jne:
        testq %rcx, %rdx
        jne test_qword_jne_nonzero
        xorl %eax, %eax
        retq
test_qword_jne_nonzero:
        movl $1, %eax
        retq

        .globl test_secondary_safe
        .def test_secondary_safe; .scl 2; .type 32; .endef
test_secondary_safe:
        testl %ecx, %edx
        je test_secondary_safe_zero
        movl %r8d, %r9d
        jne test_secondary_safe_nonzero
        movl $3, %eax
        retq
test_secondary_safe_zero:
        movl $1, %eax
        retq
test_secondary_safe_nonzero:
        movl $2, %eax
        retq

        .globl test_secondary_second_redefined
        .def test_secondary_second_redefined; .scl 2; .type 32; .endef
test_secondary_second_redefined:
        testl %ecx, %edx
        je test_secondary_second_redefined_zero
        movl %r8d, %ecx
        jne test_secondary_second_redefined_nonzero
        movl $3, %eax
        retq
test_secondary_second_redefined_zero:
        movl $1, %eax
        retq
test_secondary_second_redefined_nonzero:
        movl $2, %eax
        retq

        .globl test_high8_rejected
        .def test_high8_rejected; .scl 2; .type 32; .endef
test_high8_rejected:
        testb %ah, %dl
        je test_high8_rejected_zero
        xorl %eax, %eax
        retq
test_high8_rejected_zero:
        movl $1, %eax
        retq

        .globl test_carry_rejected
        .def test_carry_rejected; .scl 2; .type 32; .endef
test_carry_rejected:
        testl %ecx, %edx
        jb test_carry_rejected_below
        xorl %eax, %eax
        retq
test_carry_rejected_below:
        movl $1, %eax
        retq

        .globl test_same_register_preserved
        .def test_same_register_preserved; .scl 2; .type 32; .endef
test_same_register_preserved:
        testl %ecx, %ecx
        je test_same_register_preserved_zero
        xorl %eax, %eax
        retq
test_same_register_preserved_zero:
        movl $1, %eax
        retq

        .globl test_immediate_preserved
        .def test_immediate_preserved; .scl 2; .type 32; .endef
test_immediate_preserved:
        testl $4, %ecx
        jne test_immediate_preserved_nonzero
        xorl %eax, %eax
        retq
test_immediate_preserved_nonzero:
        movl $1, %eax
        retq
"#,
    )
    .ok()?;

    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("skipping unequal-register TEST/Jcc test: fixture assembly failed");
        return None;
    }
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
}

fn build_call_result_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping unequal-register TEST/Jcc pipeline test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_unequal_register_test_call_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl word_source
        .def word_source; .scl 2; .type 32; .endef
word_source:
        movl %ecx, %eax
        retq

        .globl call_word_test
        .def call_word_test; .scl 2; .type 32; .endef
call_word_test:
        subq $40, %rsp
        callq word_source
        addq $40, %rsp
        movw $0x8000, %r10w
        testw %ax, %r10w
        je call_word_test_zero
        movl $7, %eax
        retq
call_word_test_zero:
        movl $9, %eax
        retq
"#,
    )
    .ok()?;

    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("skipping unequal-register TEST/Jcc pipeline test: fixture assembly failed");
        return None;
    }
    Some(object)
}

fn call_result_fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_call_result_fixture).as_deref()
}

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    let coff_name = format!("coff_fn_{name}");
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .find_map(|(symbol, start, end)| {
            (*symbol == name || *symbol == coff_name).then_some((*start, *end))
        })
        .unwrap_or_else(|| panic!("missing fixture function {name}"))
}

fn in_span(address: Address, span: (Address, Address)) -> bool {
    address >= span.0 && address < span.1
}

fn instruction_addresses(
    db: &DecompileDB,
    span: (Address, Address),
    mnemonic: &str,
) -> Vec<Address> {
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
    )>("instruction")
        .filter_map(|(address, _, _, row_mnemonic, _, _, _, _, _, _)| {
            (in_span(*address, span) && *row_mnemonic == mnemonic).then_some(*address)
        })
        .collect()
}

fn mconds_in_span(db: &DecompileDB, span: (Address, Address)) -> Vec<(Address, MachInst)> {
    db.rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| {
            (in_span(*address, span) && matches!(inst, MachInst::Mcond(..)))
                .then_some((*address, inst.clone()))
        })
        .collect()
}

fn assert_width_condition(
    db: &DecompileDB,
    name: &str,
    expected_condition: Condition,
    expected_registers: [Mreg; 2],
) {
    let span = function_span(db, name);
    let test = instruction_addresses(db, span, "TEST");
    assert_eq!(test.len(), 1, "{name} must contain one TEST");
    let conditions = mconds_in_span(db, span);
    assert!(
        conditions.iter().any(|(address, inst)| {
            *address == test[0]
                && matches!(inst, MachInst::Mcond(condition, args, _)
                if *condition == expected_condition && {
                    let mut actual = args.as_ref().clone();
                    actual.sort();
                    let mut expected = expected_registers.to_vec();
                    expected.sort();
                    actual == expected
                })
        }),
        "{name} lost {expected_condition:?} over {expected_registers:?}: {conditions:#x?}"
    );
}

fn assert_asm_results(object: &Path) {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    AsmPass.run(&mut db);

    assert_width_condition(
        &db,
        "test_byte_je",
        Condition::Ctestzero(1),
        [Mreg::CX, Mreg::DX],
    );
    assert_width_condition(
        &db,
        "test_word_jne",
        Condition::Ctestnotzero(2),
        [Mreg::CX, Mreg::DX],
    );
    assert_width_condition(
        &db,
        "test_dword_je",
        Condition::Ctestzero(4),
        [Mreg::CX, Mreg::DX],
    );
    assert_width_condition(
        &db,
        "test_qword_jne",
        Condition::Ctestnotzero(8),
        [Mreg::CX, Mreg::DX],
    );

    let safe_span = function_span(&db, "test_secondary_safe");
    let safe_jne = instruction_addresses(&db, safe_span, "JNE");
    assert_eq!(safe_jne.len(), 1);
    let safe_jne = safe_jne[0];
    assert!(
        db.rel_iter::<(Address,)>("mcond_at_jcc")
            .any(|(address,)| *address == safe_jne),
        "safe secondary consumer did not retain an Mcond at its own JNE"
    );
    assert!(
        mconds_in_span(&db, safe_span)
            .iter()
            .any(|(address, inst)| {
                *address == safe_jne
                    && matches!(inst, MachInst::Mcond(Condition::Ctestnotzero(4), args, _)
                    if args.as_ref().iter().copied().collect::<BTreeSet<_>>()
                        == BTreeSet::from([Mreg::CX, Mreg::DX]))
            }),
        "safe secondary JNE lost its two-register TEST condition"
    );
    let safe_uses: BTreeSet<_> = db
        .rel_iter::<(Address, Mreg)>("reg_use")
        .filter_map(|(address, reg)| (*address == safe_jne).then_some(*reg))
        .collect();
    assert!(
        safe_uses.contains(&Mreg::CX) && safe_uses.contains(&Mreg::DX),
        "safe secondary JNE did not publish both TEST register uses: {safe_uses:?}"
    );

    let unsafe_span = function_span(&db, "test_secondary_second_redefined");
    let unsafe_test = instruction_addresses(&db, unsafe_span, "TEST");
    let unsafe_jne = instruction_addresses(&db, unsafe_span, "JNE");
    assert_eq!((unsafe_test.len(), unsafe_jne.len()), (1, 1));
    let unsafe_jne = unsafe_jne[0];
    let (_, first_operand, second_operand) = db
        .rel_iter::<(Address, Symbol, Symbol)>("ptest")
        .find(|(address, _, _)| *address == unsafe_test[0])
        .copied()
        .expect("unsafe fixture TEST was not decoded");
    let first_name = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .find_map(|(operand, name)| (*operand == first_operand).then_some(*name))
        .expect("missing first TEST register");
    let second_name = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .find_map(|(operand, name)| (*operand == second_operand).then_some(*name))
        .expect("missing second TEST register");
    assert_eq!(
        (first_name, second_name),
        ("EDX", "ECX"),
        "fixture must specifically exercise redefinition of ptest's second operand"
    );
    assert!(
        !db.rel_iter::<(Address,)>("mcond_at_jcc")
            .any(|(address,)| *address == unsafe_jne),
        "redefining TEST's second operand incorrectly left the secondary consumer safe"
    );
    assert!(
        !mconds_in_span(&db, unsafe_span)
            .iter()
            .any(|(address, _)| *address == unsafe_jne),
        "unsafe secondary consumer reread a redefined TEST operand"
    );

    for name in ["test_high8_rejected", "test_carry_rejected"] {
        let conditions = mconds_in_span(&db, function_span(&db, name));
        assert!(
            !conditions.iter().any(|(_, inst)| matches!(
                inst,
                MachInst::Mcond(Condition::Ctestzero(_) | Condition::Ctestnotzero(_), _, _)
            )),
            "{name} incorrectly acquired an unequal-register TEST condition: {conditions:#x?}"
        );
    }

    let same_conditions = mconds_in_span(&db, function_span(&db, "test_same_register_preserved"));
    assert!(
        same_conditions.iter().any(|(_, inst)| matches!(
            inst,
            MachInst::Mcond(
                Condition::Ccompimm(Comparison::Ceq, 0)
                    | Condition::Ccompuimm(Comparison::Ceq, 0),
                args,
                _
            ) if args.as_ref() == &[Mreg::CX]
        )),
        "same-register TEST no longer follows its established compare-to-zero path: {same_conditions:#x?}"
    );
    assert!(
        !same_conditions.iter().any(|(_, inst)| matches!(
            inst,
            MachInst::Mcond(Condition::Ctestzero(_) | Condition::Ctestnotzero(_), _, _)
        )),
        "same-register TEST was captured by the new unequal-register path"
    );

    let immediate_conditions = mconds_in_span(&db, function_span(&db, "test_immediate_preserved"));
    assert!(
        immediate_conditions.iter().any(|(_, inst)| matches!(
            inst,
            MachInst::Mcond(Condition::Cmasknotzero(4), args, _)
                if args.as_ref() == &[Mreg::CX]
        )),
        "TEST-immediate no longer follows its established mask path: {immediate_conditions:#x?}"
    );
    assert!(
        !immediate_conditions.iter().any(|(_, inst)| matches!(
            inst,
            MachInst::Mcond(Condition::Ctestzero(_) | Condition::Ctestnotzero(_), _, _)
        )),
        "TEST-immediate was captured by the new unequal-register path"
    );
}

fn assert_call_result_pipeline(object: &Path) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("failed to build unequal-register TEST/Jcc pipeline pool");
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, false, false));

    let tu = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("pipeline must emit an optimized translation unit");
    let text = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        tu,
        BinaryFormat::Coff,
    );
    let start = text
        .find("call_word_test(")
        .expect("pipeline lost call_word_test");
    let body = &text[start..];
    assert!(
        body.contains(" & "),
        "call-result TEST did not survive as a bitwise intersection:\n{body}"
    );
    assert!(
        body.contains("return 7;") && body.contains("return 9;"),
        "call-result TEST lost one of its CFG arms:\n{body}"
    );

    let output = object.with_extension("generated.c");
    std::fs::write(&output, &text).expect("failed to write generated call-result C");
    let compiled = Command::new("clang")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            "c",
            "-fsyntax-only",
        ])
        .arg(&output)
        .output()
        .expect("failed to run clang over generated call-result C");
    assert!(
        compiled.status.success(),
        "generated call-result C does not compile:\n{}\n{text}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn unequal_register_test_jcc_is_width_exact_and_secondary_safe() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("unequal-register-test-jcc".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_asm_results(&object))
        .expect("failed to spawn unequal-register TEST/Jcc test thread")
        .join()
        .expect("unequal-register TEST/Jcc test thread panicked");
}

#[test]
fn unequal_register_test_of_call_result_reaches_compilable_c() {
    let Some(object) = call_result_fixture() else {
        return;
    };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("unequal-register-test-call-result".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_call_result_pipeline(&object))
        .expect("failed to spawn unequal-register TEST/Jcc pipeline test thread")
        .join()
        .expect("unequal-register TEST/Jcc pipeline test thread panicked");
}
