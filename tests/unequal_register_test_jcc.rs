use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::clight_pass::ClightPass;
use manifold::decompile::passes::cshminor_pass::CshminorPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::mreg::Mreg;
use manifold::x86::asm::TestCond;
use manifold::x86::op::{Comparison, Condition};
use manifold::x86::types::{
    Address, ClightBinaryOp, ClightExpr, ClightStmt, MachInst, RTLInst, RTLReg, Symbol,
    TestOperandExtension,
};

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

        .globl test_secondary_snapshot
        .def test_secondary_snapshot; .scl 2; .type 32; .endef
test_secondary_snapshot:
        testl %ecx, %edx
        je test_secondary_snapshot_primary_zero
        movl %r8d, %ecx
        je test_secondary_snapshot_secondary_zero
        movl $3, %eax
        retq
test_secondary_snapshot_primary_zero:
        movl $1, %eax
        retq
test_secondary_snapshot_secondary_zero:
        movl $2, %eax
        retq

        .globl cmp_secondary_snapshot
        .def cmp_secondary_snapshot; .scl 2; .type 32; .endef
cmp_secondary_snapshot:
        cmpl %ecx, %edx
        jb cmp_secondary_snapshot_below
        movl %r8d, %edx
        jne cmp_secondary_snapshot_not_equal
        movl $3, %eax
        retq
cmp_secondary_snapshot_below:
        movl $1, %eax
        retq
cmp_secondary_snapshot_not_equal:
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

        .globl cmp_secondary_second_redefined
        .def cmp_secondary_second_redefined; .scl 2; .type 32; .endef
cmp_secondary_second_redefined:
        cmpl %ecx, %edx
        je cmp_secondary_second_redefined_equal
        movl %r8d, %ecx
        jne cmp_secondary_second_redefined_not_equal
        movl $3, %eax
        retq
cmp_secondary_second_redefined_equal:
        movl $1, %eax
        retq
cmp_secondary_second_redefined_not_equal:
        movl $2, %eax
        retq

        .globl test_secondary_external_pred
        .def test_secondary_external_pred; .scl 2; .type 32; .endef
test_secondary_external_pred:
        cmpl $0, %r8d
        je test_secondary_external_pred_entry
        testl %ecx, %edx
        je test_secondary_external_pred_zero
test_secondary_external_pred_fallthrough:
        movl %r9d, %r10d
        jne test_secondary_external_pred_nonzero
        movl $3, %eax
        retq
test_secondary_external_pred_zero:
        movl $1, %eax
        retq
test_secondary_external_pred_nonzero:
        movl $2, %eax
        retq
test_secondary_external_pred_entry:
        jmp test_secondary_external_pred_fallthrough

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

        .globl call_qword_unknown_test
        .def call_qword_unknown_test; .scl 2; .type 32; .endef
call_qword_unknown_test:
        subq $40, %rsp
        callq word_source
        addq $40, %rsp
        movabsq $0x8000000000000000, %rdx
        testq %rax, %rdx
        jne call_qword_unknown_test_high_bit
        movl $15, %eax
        retq
call_qword_unknown_test_high_bit:
        movl $17, %eax
        retq

        .globl qword_zero_extend_test
        .def qword_zero_extend_test; .scl 2; .type 32; .endef
qword_zero_extend_test:
        movl $0x80000000, %ecx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne qword_zero_extend_test_high_bit
        movl $11, %eax
        retq
qword_zero_extend_test_high_bit:
        movl $13, %eax
        retq

        .globl copied_qword_zero_extend_test
        .def copied_qword_zero_extend_test; .scl 2; .type 32; .endef
copied_qword_zero_extend_test:
        movl $0x80000000, %eax
        movq %rax, %rcx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne copied_qword_zero_extend_test_high_bit
        movl $19, %eax
        retq
copied_qword_zero_extend_test_high_bit:
        movl $21, %eax
        retq

        .globl mixed_width_join_qword_test
        .def mixed_width_join_qword_test; .scl 2; .type 32; .endef
mixed_width_join_qword_test:
        testl %r8d, %r8d
        je mixed_width_join_qword_test_full
        movl $0x80000000, %ecx
        jmp mixed_width_join_qword_test_join
mixed_width_join_qword_test_full:
        movq %r9, %rcx
mixed_width_join_qword_test_join:
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne mixed_width_join_qword_test_high_bit
        movl $23, %eax
        retq
mixed_width_join_qword_test_high_bit:
        movl $25, %eax
        retq

        .globl movsbl_dword_test
        .def movsbl_dword_test; .scl 2; .type 32; .endef
movsbl_dword_test:
        movsbl (%r8), %ecx
        movl $0x80000000, %edx
        testl %ecx, %edx
        jne movsbl_dword_test_set
        xorl %eax, %eax
        retq
movsbl_dword_test_set:
        movl $1, %eax
        retq

        .globl movsbl_qword_test
        .def movsbl_qword_test; .scl 2; .type 32; .endef
movsbl_qword_test:
        movsbl (%r8), %ecx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne movsbl_qword_test_set
        xorl %eax, %eax
        retq
movsbl_qword_test_set:
        movl $1, %eax
        retq

        .globl movswl_dword_test
        .def movswl_dword_test; .scl 2; .type 32; .endef
movswl_dword_test:
        movswl (%r8), %ecx
        movl $0x80000000, %edx
        testl %ecx, %edx
        jne movswl_dword_test_set
        xorl %eax, %eax
        retq
movswl_dword_test_set:
        movl $1, %eax
        retq

        .globl movswl_qword_test
        .def movswl_qword_test; .scl 2; .type 32; .endef
movswl_qword_test:
        movswl (%r8), %ecx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne movswl_qword_test_set
        xorl %eax, %eax
        retq
movswl_qword_test_set:
        movl $1, %eax
        retq

        .globl movsbq_qword_test
        .def movsbq_qword_test; .scl 2; .type 32; .endef
movsbq_qword_test:
        movsbq (%r8), %rcx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne movsbq_qword_test_set
        xorl %eax, %eax
        retq
movsbq_qword_test_set:
        movl $1, %eax
        retq

        .globl incoming_qword_test
        .def incoming_qword_test; .scl 2; .type 32; .endef
incoming_qword_test:
        cmpl $0, %ecx
        sete %al
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne incoming_qword_test_set
        xorl %eax, %eax
        retq
incoming_qword_test_set:
        movl $1, %eax
        retq

        .globl partial_word_qword_test
        .def partial_word_qword_test; .scl 2; .type 32; .endef
partial_word_qword_test:
        movw %r8w, %cx
        movabsq $0x8000000000000000, %rdx
        testq %rcx, %rdx
        jne partial_word_qword_test_set
        xorl %eax, %eax
        retq
partial_word_qword_test_set:
        movl $1, %eax
        retq

        .globl high8_write_low8_test
        .def high8_write_low8_test; .scl 2; .type 32; .endef
high8_write_low8_test:
        movb %cl, %ah
        testb %al, %dl
        jne high8_write_low8_test_set
        xorl %eax, %eax
        retq
high8_write_low8_test_set:
        movl $1, %eax
        retq

        .globl test_secondary_snapshot_pipeline
        .def test_secondary_snapshot_pipeline; .scl 2; .type 32; .endef
test_secondary_snapshot_pipeline:
        testl %ecx, %edx
        je test_secondary_snapshot_pipeline_primary_zero
        movl %r8d, %ecx
        je test_secondary_snapshot_pipeline_secondary_zero
        movl $3, %eax
        retq
test_secondary_snapshot_pipeline_primary_zero:
        movl $1, %eax
        retq
test_secondary_snapshot_pipeline_secondary_zero:
        movl $2, %eax
        retq

        .globl cmp_secondary_snapshot_pipeline
        .def cmp_secondary_snapshot_pipeline; .scl 2; .type 32; .endef
cmp_secondary_snapshot_pipeline:
        cmpl %ecx, %edx
        jb cmp_secondary_snapshot_pipeline_below
        movl %r8d, %edx
        jne cmp_secondary_snapshot_pipeline_not_equal
        movl $3, %eax
        retq
cmp_secondary_snapshot_pipeline_below:
        movl $1, %eax
        retq
cmp_secondary_snapshot_pipeline_not_equal:
        movl $2, %eax
        retq

        .globl unsafe_secondary_pipeline
        .def unsafe_secondary_pipeline; .scl 2; .type 32; .endef
unsafe_secondary_pipeline:
        testl %ecx, %edx
        je unsafe_secondary_pipeline_zero
        movl %r8d, %ecx
        jne unsafe_secondary_pipeline_nonzero
        movl $3, %eax
        retq
unsafe_secondary_pipeline_zero:
        movl $1, %eax
        retq
unsafe_secondary_pipeline_nonzero:
        movl $2, %eax
        retq

        .globl unsafe_secondary_cmp_pipeline
        .def unsafe_secondary_cmp_pipeline; .scl 2; .type 32; .endef
unsafe_secondary_cmp_pipeline:
        cmpl %ecx, %edx
        je unsafe_secondary_cmp_pipeline_equal
        movl %r8d, %ecx
        jne unsafe_secondary_cmp_pipeline_not_equal
        movl $3, %eax
        retq
unsafe_secondary_cmp_pipeline_equal:
        movl $1, %eax
        retq
unsafe_secondary_cmp_pipeline_not_equal:
        movl $2, %eax
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

fn clight_stmts_in_span(db: &DecompileDB, span: (Address, Address)) -> Vec<ClightStmt> {
    // Structuring may relocate a compound statement to a synthetic node
    // outside the original instruction-address interval. Use the explicit
    // function owner published for emission rather than node arithmetic.
    db.rel_iter::<(Address, Address, ClightStmt)>("emit_clight_stmt")
        .filter_map(|(function, _, stmt)| (*function == span.0).then_some(stmt.clone()))
        .collect()
}

fn rtl_condition_args_at(db: &DecompileDB, node: Address) -> Vec<Vec<RTLReg>> {
    db.rel_iter::<(Address, RTLInst)>("rtl_inst")
        .filter_map(|(address, inst)| match inst {
            RTLInst::Icond(_, args, _, _) if *address == node => Some(args.as_ref().clone()),
            _ => None,
        })
        .collect()
}

fn assert_secondary_folded_to_its_target(db: &DecompileDB, secondary: Address) {
    let target = db
        .rel_iter::<(Address, TestCond, Symbol)>("pjcc")
        .find_map(|(address, _, target)| (*address == secondary).then_some(*target))
        .expect("unsafe secondary fixture lost its JCC target");
    let gotos: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| {
            (*address == secondary)
                .then_some(inst)
                .and_then(|inst| match inst {
                    MachInst::Mgoto(destination) => Some(*destination),
                    _ => None,
                })
        })
        .collect();
    assert_eq!(
        gotos,
        vec![target],
        "unsafe inverse secondary was not folded to its proven target"
    );
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
    let safe_test = instruction_addresses(&db, safe_span, "TEST");
    assert_eq!(safe_test.len(), 1);
    assert!(
        db.rel_iter::<(Address, Address)>("mcond_compare_site")
            .any(|(consumer, producer)| *consumer == safe_jne && *producer == safe_test[0]),
        "safe secondary JNE did not retain its producer-site condition snapshot"
    );

    for (name, producer_mnemonic, secondary_mnemonic) in [
        ("test_secondary_snapshot", "TEST", "JE"),
        ("cmp_secondary_snapshot", "CMP", "JNE"),
    ] {
        let span = function_span(&db, name);
        let producer = instruction_addresses(&db, span, producer_mnemonic);
        let secondaries = instruction_addresses(&db, span, secondary_mnemonic);
        assert_eq!(producer.len(), 1, "{name} lost its flag producer");
        let secondary = *secondaries
            .iter()
            .max()
            .expect("snapshot fixture lost its secondary JCC");
        assert!(
            db.rel_iter::<(Address, Address)>("mcond_compare_site")
                .any(|(consumer, compare)| *consumer == secondary && *compare == producer[0]),
            "{name} did not bind the secondary to producer-site SSA values"
        );
        assert!(
            mconds_in_span(&db, span).iter().any(|(address, _)| *address == secondary),
            "{name} lost its nonconstant secondary condition"
        );
    }

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
    let primary_je = instruction_addresses(&db, unsafe_span, "JE");
    assert_eq!(primary_je.len(), 1);
    let primary_target = db
        .rel_iter::<(Address, TestCond, Symbol)>("pjcc")
        .find_map(|(address, _, target)| (*address == primary_je[0]).then_some(*target))
        .expect("unsafe-secondary fixture lost its primary JE target");
    let primary_conditions: Vec<_> = mconds_in_span(&db, unsafe_span)
        .into_iter()
        .filter(|(address, inst)| {
            *address == unsafe_test[0]
                && matches!(
                    inst,
                    MachInst::Mcond(Condition::Ctestzero(_) | Condition::Ctestnotzero(_), _, _)
                )
        })
        .collect();
    assert_eq!(
        primary_conditions.len(),
        1,
        "unsafe secondary installed a competing Mcond at the TEST root: {primary_conditions:#x?}"
    );
    assert!(
        matches!(
            &primary_conditions[0].1,
            MachInst::Mcond(Condition::Ctestzero(4), _, target)
                if *target == primary_target
        ),
        "the sole TEST-root Mcond no longer represents the primary JE: {primary_conditions:#x?}"
    );
    assert_secondary_folded_to_its_target(&db, unsafe_jne);

    let cmp_span = function_span(&db, "cmp_secondary_second_redefined");
    let cmp = instruction_addresses(&db, cmp_span, "CMP");
    let cmp_je = instruction_addresses(&db, cmp_span, "JE");
    let cmp_jne = instruction_addresses(&db, cmp_span, "JNE");
    assert_eq!((cmp.len(), cmp_je.len(), cmp_jne.len()), (1, 1, 1));
    let cmp_primary_target = db
        .rel_iter::<(Address, TestCond, Symbol)>("pjcc")
        .find_map(|(address, _, target)| (*address == cmp_je[0]).then_some(*target))
        .expect("unsafe CMP fixture lost its primary JE target");
    let cmp_conditions = mconds_in_span(&db, cmp_span);
    assert!(
        cmp_conditions.iter().any(|(address, inst)| {
            *address == cmp[0]
                && matches!(
                    inst,
                    MachInst::Mcond(
                        Condition::Ccomp(Comparison::Ceq)
                            | Condition::Ccompu(Comparison::Ceq),
                        _,
                        target
                    ) if *target == cmp_primary_target
                )
        }),
        "unsafe secondary erased the primary CMP/JE: {cmp_conditions:#x?}"
    );
    assert!(
        !cmp_conditions
            .iter()
            .any(|(address, _)| *address == cmp_jne[0]),
        "unsafe CMP secondary reread a redefined operand: {cmp_conditions:#x?}"
    );
    assert_secondary_folded_to_its_target(&db, cmp_jne[0]);

    let shared_span = function_span(&db, "test_secondary_external_pred");
    let shared_test = instruction_addresses(&db, shared_span, "TEST");
    let shared_je = instruction_addresses(&db, shared_span, "JE");
    let shared_jne = instruction_addresses(&db, shared_span, "JNE");
    assert_eq!(
        (shared_test.len(), shared_je.len(), shared_jne.len()),
        (1, 2, 1)
    );
    let primary_je = *shared_je.iter().max().unwrap();
    let primary_block = db
        .rel_iter::<(Address, Address)>("code_in_block")
        .find_map(|(address, block)| (*address == primary_je).then_some(*block))
        .expect("shared-entry fixture lost the primary block");
    let secondary_block = db
        .rel_iter::<(Address, Address)>("code_in_block")
        .find_map(|(address, block)| (*address == shared_jne[0]).then_some(*block))
        .expect("shared-entry fixture lost the secondary block");
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("ddisasm_cfg_edge")
            .any(|(source, destination, _)| {
                let source_block = db
                    .rel_iter::<(Address, Address)>("code_in_block")
                    .find_map(|(address, block)| (address == source).then_some(*block));
                source_block.is_some_and(|block| {
                    block != primary_block && block != secondary_block
                }) && db
                    .rel_iter::<(Address, Address)>("code_in_block")
                    .any(|(address, block)| address == destination && *block == secondary_block)
            }),
        "shared-entry fixture does not exercise an external predecessor"
    );
    assert!(
        !db.rel_iter::<(Address, MachInst)>("mach_inst")
            .any(|(address, inst)| {
                *address == shared_jne[0] && matches!(inst, MachInst::Mgoto(_))
            }),
        "unsafe inverse secondary was folded despite another CFG predecessor"
    );
    assert!(
        !db.rel_iter::<(Address, MachInst)>("mach_inst")
            .any(|(address, inst)| *address == shared_jne[0] && matches!(inst, MachInst::Mcond(..))),
        "multiply-entered secondary retained a condition with unauthenticated EFLAGS"
    );
    assert!(
        !db.rel_iter::<(Address, Mreg)>("decoded_reg_def").any(|(address, reg)| {
            *address > shared_test[0]
                && *address < shared_jne[0]
                && matches!(*reg, Mreg::CX | Mreg::DX)
        }),
        "external-predecessor fixture must not rely on operand redefinition"
    );
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_control_flow")
            .any(|(function, consumer, reason)| {
                *function == shared_span.0
                    && *consumer == shared_jne[0]
                    && *reason == "ambiguous-secondary-flags"
            }),
        "ambiguous secondary did not produce an explicit fail-closed diagnostic"
    );
    assert!(
        mconds_in_span(&db, shared_span)
            .iter()
            .any(|(address, inst)| {
                *address == shared_test[0]
                    && matches!(inst, MachInst::Mcond(Condition::Ctestzero(4), _, _))
            }),
        "rejecting the unproven secondary erased its primary TEST branch"
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

fn printed_function<'a>(text: &'a str, name: &str) -> &'a str {
    let needle = format!("{name}(");
    for (start, _) in text.match_indices(&needle) {
        let suffix = &text[start..];
        let Some(open) = suffix.find('{') else {
            continue;
        };
        if suffix.find(';').is_some_and(|semi| semi < open) {
            continue;
        }
        let mut depth = 0usize;
        for (offset, byte) in suffix[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &suffix[..open + offset + 1];
                    }
                }
                _ => {}
            }
        }
    }
    panic!("printed translation unit lost function {name}")
}

fn expr_contains_and(expr: &ClightExpr) -> bool {
    match expr {
        ClightExpr::Ebinop(ClightBinaryOp::Oand, _, _, _) => true,
        ClightExpr::Eunop(_, inner, _)
        | ClightExpr::Ecast(inner, _)
        | ClightExpr::Ederef(inner, _)
        | ClightExpr::Eaddrof(inner, _)
        | ClightExpr::Efield(inner, _, _) => expr_contains_and(inner),
        ClightExpr::Ebinop(_, lhs, rhs, _) => expr_contains_and(lhs) || expr_contains_and(rhs),
        ClightExpr::Econdition(condition, yes, no, _) => {
            expr_contains_and(condition) || expr_contains_and(yes) || expr_contains_and(no)
        }
        _ => false,
    }
}

fn expr_contains_equality(expr: &ClightExpr) -> bool {
    match expr {
        ClightExpr::Ebinop(ClightBinaryOp::Oeq | ClightBinaryOp::One, _, _, _) => true,
        ClightExpr::Eunop(_, inner, _)
        | ClightExpr::Ecast(inner, _)
        | ClightExpr::Ederef(inner, _)
        | ClightExpr::Eaddrof(inner, _)
        | ClightExpr::Efield(inner, _, _) => expr_contains_equality(inner),
        ClightExpr::Ebinop(_, lhs, rhs, _) => {
            expr_contains_equality(lhs) || expr_contains_equality(rhs)
        }
        ClightExpr::Econdition(condition, yes, no, _) => {
            expr_contains_equality(condition)
                || expr_contains_equality(yes)
                || expr_contains_equality(no)
        }
        _ => false,
    }
}

fn stmt_condition_matches(stmt: &ClightStmt, predicate: fn(&ClightExpr) -> bool) -> bool {
    match stmt {
        ClightStmt::Sifthenelse(condition, yes, no) => {
            predicate(condition)
                || stmt_condition_matches(yes, predicate)
                || stmt_condition_matches(no, predicate)
        }
        ClightStmt::Ssequence(statements) => statements
            .iter()
            .any(|statement| stmt_condition_matches(statement, predicate)),
        ClightStmt::Sloop(body, increment) => {
            stmt_condition_matches(body, predicate) || stmt_condition_matches(increment, predicate)
        }
        ClightStmt::Sswitch(_, cases) => cases
            .iter()
            .any(|(_, statement)| stmt_condition_matches(statement, predicate)),
        ClightStmt::Slabel(_, statement) => stmt_condition_matches(statement, predicate),
        _ => false,
    }
}

fn assert_pass_owned_diagnostics_are_rebuilt(db: &mut DecompileDB) {
    const STALE_FUNCTION: Address = Address::MAX - 1;
    const STALE_NODE: Address = Address::MAX;
    const STALE_REASON: Symbol = "stale-test-diagnostic";
    let stale = (STALE_FUNCTION, STALE_NODE, STALE_REASON);

    db.rel_push("unsupported_control_flow", stale);
    AsmPass.run(db);
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_control_flow")
            .any(|row| *row == stale),
        "AsmPass retained a diagnostic from its previous run"
    );

    db.rel_push("unsupported_rtl_condition", stale);
    RTLPass.run(db);
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_rtl_condition")
            .any(|row| *row == stale),
        "RTLPass retained a diagnostic from its previous run"
    );

    db.rel_push("unsupported_clight_condition_seed", stale);
    CshminorPass.run(db);
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_clight_condition_seed")
            .any(|row| *row == stale),
        "CshminorPass retained a diagnostic seed from its previous run"
    );

    db.rel_push("unsupported_clight_condition", stale);
    ClightPass.run(db);
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_clight_condition")
            .any(|row| *row == stale),
        "ClightPass retained a diagnostic from its previous run"
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
    let body = printed_function(&text, "call_word_test");
    assert!(
        body.contains(" & "),
        "call-result TEST did not survive as a bitwise intersection:\n{body}"
    );
    assert!(
        body.contains("return 7;") && body.contains("return 9;"),
        "call-result TEST lost one of its CFG arms:\n{body}"
    );

    let qword_span = function_span(&db, "qword_zero_extend_test");
    let qword_test = instruction_addresses(&db, qword_span, "TEST");
    assert_eq!(qword_test.len(), 1);
    let qword_rcx: BTreeSet<_> = db
        .rel_iter::<(Address, Mreg, RTLReg)>("reg_rtl")
        .filter_map(|(node, mreg, rtl)| {
            (*node == qword_test[0] && *mreg == Mreg::CX).then_some(*rtl)
        })
        .collect();
    assert!(
        db.rel_iter::<(RTLReg, TestOperandExtension)>("test_operand_extension")
            .any(|(rtl, extension)| {
                qword_rcx.contains(rtl)
                    && *extension == TestOperandExtension::ZeroExtended(4)
            }),
        "E32 definition did not publish zero-extension provenance before TESTq"
    );

    let copied_span = function_span(&db, "copied_qword_zero_extend_test");
    let copied_test = instruction_addresses(&db, copied_span, "TEST");
    assert_eq!(copied_test.len(), 1);
    let copied_rcx: BTreeSet<_> = db
        .rel_iter::<(Address, Mreg, RTLReg)>("reg_rtl")
        .filter_map(|(node, mreg, rtl)| {
            (*node == copied_test[0] && *mreg == Mreg::CX).then_some(*rtl)
        })
        .collect();
    assert!(db
        .rel_iter::<(RTLReg, TestOperandExtension)>("test_operand_extension")
        .any(|(rtl, extension)| {
            copied_rcx.contains(rtl)
                && *extension == TestOperandExtension::ZeroExtended(4)
        }),
        "E32 zero extension did not survive a full-width RAX-to-RCX copy");
    let call_qword_span = function_span(&db, "call_qword_unknown_test");
    let call_qword_test = instruction_addresses(&db, call_qword_span, "TEST");
    assert_eq!(call_qword_test.len(), 1);
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_rtl_condition")
        .any(|(function, node, reason)| {
            *function == call_qword_span.0
                && *node == call_qword_test[0]
                && *reason == "unknown-test-register-definition"
        }),
        "qword TEST widened an uncertified CALL result instead of failing closed");

    let mixed_span = function_span(&db, "mixed_width_join_qword_test");
    let mixed_tests = instruction_addresses(&db, mixed_span, "TEST");
    assert_eq!(mixed_tests.len(), 2);
    let mixed_qword_test = *mixed_tests.iter().max().unwrap();
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_rtl_condition")
        .any(|(function, node, reason)| {
            *function == mixed_span.0
                && *node == mixed_qword_test
                && *reason == "conflicting-test-extension-provenance"
        }),
        "joined E32/full-width definitions did not fail closed before TESTq");

    for name in ["movsbl_dword_test", "movswl_dword_test"] {
        let span = function_span(&db, name);
        let test = instruction_addresses(&db, span, "TEST");
        assert_eq!(test.len(), 1);
        let clight = clight_stmts_in_span(&db, span);
        assert!(
            clight.iter().any(|stmt| stmt_condition_matches(stmt, expr_contains_and)),
            "{name} lost its equal-width E-register extension at TESTd: {clight:#x?}"
        );
    }
    for name in ["movsbl_qword_test", "movswl_qword_test"] {
        let span = function_span(&db, name);
        let test = instruction_addresses(&db, span, "TEST");
        assert_eq!(test.len(), 1);
        let body = printed_function(&text, name);
        assert!(
            body.contains("(unsigned __int64)(unsigned int)")
                && body.contains(" & ")
                && body.contains(" != 0")
                && body.contains("return 0;")
                && body.contains("return 1;"),
            "{name} lost its width-exact qword TEST condition or CFG arms:\n{body}"
        );
    }
    let movsbq_span = function_span(&db, "movsbq_qword_test");
    let movsbq_test = instruction_addresses(&db, movsbq_span, "TEST");
    assert_eq!(movsbq_test.len(), 1);
    let movsbq_args = rtl_condition_args_at(&db, movsbq_test[0]);
    assert!(
        movsbq_args.iter().flatten().any(|arg| {
            db.rel_iter::<(RTLReg, TestOperandExtension)>("test_operand_extension")
                .any(|(rtl, extension)| {
                    rtl == arg && *extension == TestOperandExtension::SignExtended(1)
                })
        }),
        "MOVSBQ -> TESTq lost its signed parent-register extension"
    );

    let incoming_span = function_span(&db, "incoming_qword_test");
    let incoming_test = instruction_addresses(&db, incoming_span, "TEST");
    assert_eq!(incoming_test.len(), 1);
    let incoming_rcx: BTreeSet<_> = db
        .rel_iter::<(Address, Mreg, RTLReg)>("reg_rtl")
        .filter_map(|(node, mreg, rtl)| {
            (*node == incoming_test[0] && *mreg == Mreg::CX).then_some(*rtl)
        })
        .collect();
    assert!(!incoming_rcx.is_empty());
    assert!(
        !db.rel_iter::<(RTLReg, TestOperandExtension)>("test_operand_extension")
            .any(|(rtl, extension)| {
                incoming_rcx.contains(rtl)
                    && *extension == TestOperandExtension::ZeroExtended(4)
            }),
        "qword TEST invented an ECX write for an unchanged incoming RCX"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_clight_condition")
            .any(|(function, node, reason)| {
                *function == incoming_span.0
                    && *node == incoming_test[0]
                    && *reason == "condition-bits-unrepresentable"
            }),
        "incoming RCX retained a fail-closed diagnostic despite full-width type evidence"
    );
    let incoming_body = printed_function(&text, "incoming_qword_test");
    assert!(
        text.contains("int coff_fn_incoming_qword_test(__int64 p0)")
            && incoming_body.contains("if ((-9223372036854775808LL & p0) != 0)")
            && !incoming_body.contains("(unsigned int)p0")
            && incoming_body.contains("return 0;")
            && incoming_body.contains("return 1;"),
        "incoming RCX lost its full-width raw high-bit TEST or CFG arms:\n{incoming_body}"
    );

    let partial_span = function_span(&db, "partial_word_qword_test");
    let partial_test = instruction_addresses(&db, partial_span, "TEST");
    assert_eq!(partial_test.len(), 1);
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_rtl_condition")
            .any(|(function, node, reason)| {
                *function == partial_span.0
                    && *node == partial_test[0]
                    && *reason == "partial-register-test-parent-bits"
            }),
        "word write before TESTq did not fail closed on unknown parent-register bits"
    );

    let high8_span = function_span(&db, "high8_write_low8_test");
    let high8_test = instruction_addresses(&db, high8_span, "TEST");
    assert_eq!(high8_test.len(), 1);
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_rtl_condition")
            .any(|(function, node, reason)| {
                *function == high8_span.0
                    && *node == high8_test[0]
                    && *reason == "unknown-test-register-definition"
            }),
        "a legacy high-byte write was misrepresented as a low-byte definition"
    );

    for (name, producer_mnemonic, secondary_mnemonic) in [
        ("test_secondary_snapshot_pipeline", "TEST", "JE"),
        ("cmp_secondary_snapshot_pipeline", "CMP", "JNE"),
    ] {
        let span = function_span(&db, name);
        let producer = instruction_addresses(&db, span, producer_mnemonic);
        let secondaries = instruction_addresses(&db, span, secondary_mnemonic);
        assert_eq!(producer.len(), 1);
        let secondary = *secondaries
            .iter()
            .max()
            .expect("snapshot pipeline lost secondary JCC");
        let primary_args = rtl_condition_args_at(&db, producer[0]);
        let secondary_args = rtl_condition_args_at(&db, secondary);
        assert_eq!(primary_args.len(), 1, "{name} lost its primary condition");
        assert_eq!(secondary_args.len(), 1, "{name} lost its secondary condition");
        assert_eq!(
            primary_args[0], secondary_args[0],
            "{name} reread a post-compare register instead of the producer SSA snapshot"
        );
    }

    let unsafe_span = function_span(&db, "unsafe_secondary_pipeline");
    let unsafe_test = instruction_addresses(&db, unsafe_span, "TEST");
    let unsafe_jne = instruction_addresses(&db, unsafe_span, "JNE");
    assert_eq!((unsafe_test.len(), unsafe_jne.len()), (1, 1));
    assert!(
        db.rel_iter::<(Address, RTLInst)>("rtl_inst")
            .any(|(node, inst)| {
                *node == unsafe_test[0]
                    && matches!(inst, RTLInst::Icond(Condition::Ctestzero(4), _, _, _))
            }),
        "the unsafe secondary displaced the primary TEST/JE before final RTL"
    );
    assert!(
        !db.rel_iter::<(Address, RTLInst)>("rtl_inst")
            .any(|(node, inst)| {
                *node == unsafe_jne[0]
                    && matches!(
                        inst,
                        RTLInst::Icond(
                            Condition::Ctestzero(_) | Condition::Ctestnotzero(_),
                            _,
                            _,
                            _
                        )
                    )
            }),
        "an unsafe secondary TEST condition survived at its JNE"
    );
    let unsafe_clight = clight_stmts_in_span(&db, unsafe_span);
    assert!(
        unsafe_clight
            .iter()
            .any(|stmt| stmt_condition_matches(stmt, expr_contains_and)),
        "the primary TEST condition did not reach Clight: {unsafe_clight:#x?}"
    );
    let unsafe_body = printed_function(&text, "unsafe_secondary_pipeline");
    assert!(
        unsafe_body.contains(" & ")
            && unsafe_body.contains("return 1;")
            && unsafe_body.contains("return 2;"),
        "the unsafe secondary displaced or erased the primary branch in emitted C:\n{unsafe_body}"
    );

    let unsafe_cmp_span = function_span(&db, "unsafe_secondary_cmp_pipeline");
    let unsafe_cmp = instruction_addresses(&db, unsafe_cmp_span, "CMP");
    let unsafe_cmp_jne = instruction_addresses(&db, unsafe_cmp_span, "JNE");
    assert_eq!((unsafe_cmp.len(), unsafe_cmp_jne.len()), (1, 1));
    assert_secondary_folded_to_its_target(&db, unsafe_cmp_jne[0]);
    assert!(
        db.rel_iter::<(Address, RTLInst)>("rtl_inst")
            .any(|(node, inst)| {
                *node == unsafe_cmp[0]
                    && matches!(
                        inst,
                        RTLInst::Icond(
                            Condition::Ccomp(Comparison::Ceq) | Condition::Ccompu(Comparison::Ceq),
                            _,
                            _,
                            _
                        )
                    )
            }),
        "the unsafe secondary displaced the primary CMP/JE before final RTL"
    );
    assert!(
        !db.rel_iter::<(Address, RTLInst)>("rtl_inst")
            .any(|(node, inst)| {
                *node == unsafe_cmp_jne[0] && matches!(inst, RTLInst::Icond(_, _, _, _))
            }),
        "an unsafe secondary CMP condition survived at its JNE"
    );
    let unsafe_cmp_clight = clight_stmts_in_span(&db, unsafe_cmp_span);
    assert!(
        unsafe_cmp_clight
            .iter()
            .any(|stmt| stmt_condition_matches(stmt, expr_contains_equality)),
        "the primary CMP condition did not reach Clight: {unsafe_cmp_clight:#x?}"
    );
    let unsafe_cmp_body = printed_function(&text, "unsafe_secondary_cmp_pipeline");
    assert!(
        (unsafe_cmp_body.contains(" == ") || unsafe_cmp_body.contains(" != "))
            && unsafe_cmp_body.contains("return 1;")
            && unsafe_cmp_body.contains("return 2;"),
        "the unsafe secondary displaced or erased the primary CMP branch in emitted C:\n{unsafe_cmp_body}"
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

    assert_pass_owned_diagnostics_are_rebuilt(&mut db);
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
