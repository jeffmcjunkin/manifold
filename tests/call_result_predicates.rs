use std::path::PathBuf;
use std::process::Command;
use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::x86::op::{Condition, TestRegisterSlice};
use manifold::x86::types::{Address, Node, RTLInst, RTLReg, Symbol};

fn build_fixture() -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "manifold_call_result_predicates_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create call-result fixture directory");
    let source = directory.join("fixture.s");
    let object = directory.join("fixture.obj");
    std::fs::write(
        &source,
        r#"
        .text
        .extern masked_value
        .globl call_result_register_mask
        .def call_result_register_mask; .scl 2; .type 32; .endef
call_result_register_mask:
        subq $40, %rsp
        callq masked_value
        movl $0x8000, %ecx
        testw %ax, %cx
        je .Lzero
        movl $1, %eax
        addq $40, %rsp
        retq
.Lzero:
        xorl %eax, %eax
        addq $40, %rsp
        retq

        .globl testb_je_ignores_high_bits
        .def testb_je_ignores_high_bits; .scl 2; .type 32; .endef
testb_je_ignores_high_bits:
        movl $0x100, %eax
        movl $0x100, %ecx
        testb %al, %cl
        je .Lbyte_zero
        xorl %eax, %eax
        retq
.Lbyte_zero:
        movl $1, %eax
        retq

        .globl testw_jne_ignores_high_bits
        .def testw_jne_ignores_high_bits; .scl 2; .type 32; .endef
testw_jne_ignores_high_bits:
        movl $0x10000, %eax
        movl $0x10000, %ecx
        testw %ax, %cx
        jne .Lword_nonzero
        xorl %eax, %eax
        retq
.Lword_nonzero:
        movl $1, %eax
        retq

        .globl testl_je_ignores_high_bits
        .def testl_je_ignores_high_bits; .scl 2; .type 32; .endef
testl_je_ignores_high_bits:
        movabsq $0x100000000, %rax
        movabsq $0x100000000, %rcx
        testl %eax, %ecx
        je .Ldword_zero
        xorl %eax, %eax
        retq
.Ldword_zero:
        movl $1, %eax
        retq

        .globl testb_self_je_ignores_high_bits
        .def testb_self_je_ignores_high_bits; .scl 2; .type 32; .endef
testb_self_je_ignores_high_bits:
        movl $0x100, %eax
        testb %al, %al
        je .Lself_byte_zero
        xorl %eax, %eax
        retq
.Lself_byte_zero:
        movl $1, %eax
        retq

        .globl testw_self_jne_ignores_high_bits
        .def testw_self_jne_ignores_high_bits; .scl 2; .type 32; .endef
testw_self_jne_ignores_high_bits:
        movl $0x10000, %eax
        testw %ax, %ax
        jne .Lself_word_nonzero
        xorl %eax, %eax
        retq
.Lself_word_nonzero:
        movl $1, %eax
        retq

        .globl testl_self_je_ignores_high_bits
        .def testl_self_je_ignores_high_bits; .scl 2; .type 32; .endef
testl_self_je_ignores_high_bits:
        movabsq $0x100000000, %rax
        testl %eax, %eax
        je .Lself_dword_zero
        xorl %eax, %eax
        retq
.Lself_dword_zero:
        movl $1, %eax
        retq

        .globl testq_jne_keeps_high_bits
        .def testq_jne_keeps_high_bits; .scl 2; .type 32; .endef
testq_jne_keeps_high_bits:
        movabsq $0x100000000, %rax
        movabsq $0x100000000, %rcx
        testq %rax, %rcx
        jne .Lqword_nonzero
        xorl %eax, %eax
        retq
.Lqword_nonzero:
        movl $1, %eax
        retq

        .globl testb_jne_mixed_same_parent_slices
        .def testb_jne_mixed_same_parent_slices; .scl 2; .type 32; .endef
testb_jne_mixed_same_parent_slices:
        movl $0x100, %eax
        testb %al, %ah
        jne .Lmixed_wrong_slice
        movl $0x1, %eax
        testb %al, %ah
        jne .Lmixed_wrong_slice
        movl $1, %eax
        retq
.Lmixed_wrong_slice:
        xorl %eax, %eax
        retq
"#,
    )
    .expect("write call-result fixture assembly");
    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("run clang over call-result fixture");
    assert!(status.success(), "call-result fixture assembly failed");
    object
}

fn assert_emitted_c_preserves_test_widths(object: &PathBuf) {
    let directory = object.parent().expect("fixture object directory");
    let emitted = directory.join("fixture.c");
    let harness = directory.join("harness.c");
    let executable = directory.join("fixture-test");

    let manifold_output = Command::new(env!("CARGO_BIN_EXE_manifold"))
        .arg(object)
        .arg(&emitted)
        .output()
        .expect("run manifold over TEST-width fixture");
    assert!(
        manifold_output.status.success(),
        "manifold failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&manifold_output.stdout),
        String::from_utf8_lossy(&manifold_output.stderr),
    );

    std::fs::write(
        &harness,
        r#"
int coff_fn_call_result_register_mask(void);
int coff_fn_testb_je_ignores_high_bits(void);
int coff_fn_testw_jne_ignores_high_bits(void);
int coff_fn_testl_je_ignores_high_bits(void);
int coff_fn_testb_self_je_ignores_high_bits(void);
int coff_fn_testw_self_jne_ignores_high_bits(void);
int coff_fn_testl_self_je_ignores_high_bits(void);
int coff_fn_testq_jne_keeps_high_bits(void);
int coff_fn_testb_jne_mixed_same_parent_slices(void);

int coff_ext_masked_value(void) { return 0x8000; }

int main(void) {
    if (coff_fn_call_result_register_mask() != 1) return 10;
    if (coff_fn_testb_je_ignores_high_bits() != 1) return 11;
    if (coff_fn_testw_jne_ignores_high_bits() != 0) return 12;
    if (coff_fn_testl_je_ignores_high_bits() != 1) return 13;
    if (coff_fn_testq_jne_keeps_high_bits() != 1) return 14;
    if (coff_fn_testb_jne_mixed_same_parent_slices() != 1) return 15;
    if (coff_fn_testb_self_je_ignores_high_bits() != 1) return 16;
    if (coff_fn_testw_self_jne_ignores_high_bits() != 0) return 17;
    if (coff_fn_testl_self_je_ignores_high_bits() != 1) return 18;
    return 0;
}
"#,
    )
    .expect("write TEST-width C harness");

    let compile_output = Command::new("clang")
        .args(["-std=c11", "-O0"])
        .arg(&emitted)
        .arg(&harness)
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("compile emitted TEST-width C");
    let emitted_source = std::fs::read_to_string(&emitted).expect("read emitted TEST-width C");
    assert!(
        compile_output.status.success(),
        "emitted C failed to compile:\n{}\nstdout:\n{}\nstderr:\n{}",
        emitted_source,
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr),
    );

    let run_output = Command::new(&executable)
        .output()
        .expect("run emitted TEST-width C");
    assert!(
        run_output.status.success(),
        "emitted C changed TEST-width behavior (status {:?}):\n{}\nstdout:\n{}\nstderr:\n{}",
        run_output.status.code(),
        emitted_source,
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr),
    );
}

fn on_pipeline_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("call-result-predicate-test".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn call-result predicate test")
        .join()
        .expect("call-result predicate test panicked");
}

#[test]
fn register_mask_test_keeps_call_result_live_and_preserves_emitted_behavior() {
    on_pipeline_stack(|| {
        let object = build_fixture();
        let mut db = DecompileDB::default();
        manifold::decompile::disassembly::load_from_binary(&mut db, &object);
        manifold::decompile::disassembly::load_preset(&mut db);
        AbiPass.run(&mut db);
        CanaryVlaPass.run(&mut db);
        AsmPass.run(&mut db);
        StackAnalysisPass.run(&mut db);
        MachPass.run(&mut db);
        LinearPass.run(&mut db);
        RTLPass.run(&mut db);

        let (start, end) = db
            .rel_iter::<(Symbol, Address, Address)>("func_span")
            .find_map(|(name, start, end)| {
                (*name == "call_result_register_mask"
                    || *name == "coff_fn_call_result_register_mask")
                    .then_some((*start, *end))
            })
            .expect("missing call-result fixture function");
        let in_function = |address: Address| start <= address && address < end;
        let call = db
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
            )>("instruction")
            .find_map(|(address, _, _, mnemonic, _, _, _, _, _, _)| {
                (in_function(*address) && *mnemonic == "CALL").then_some(*address)
            })
            .expect("missing fixture call");
        let test = db
            .rel_iter::<(Address, Symbol, Symbol)>("ptest")
            .find_map(|(address, _, _)| in_function(*address).then_some(*address))
            .expect("missing fixture TEST");
        let return_value = db
            .rel_iter::<(Node, RTLReg)>("call_return_reg")
            .find_map(|(node, result)| (*node == call).then_some(*result))
            .expect("TEST did not keep the call return value live");
        assert!(
            db.rel_iter::<(RTLReg,)>("is_not_ptr")
                .any(|(reg,)| *reg == return_value),
            "narrow TEST result bypassed RTL integer classification",
        );

        assert!(db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| {
                *node == call
                    && matches!(
                        inst,
                        RTLInst::Icall(_, _, _, Some(result), _) if *result == return_value
                    )
            }));
        assert!(db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| {
                *node == test
                    && matches!(
                        inst,
                        RTLInst::Icond(
                            Condition::Cmaskregzero(
                                TestRegisterSlice::Low16,
                                TestRegisterSlice::Low16,
                            ),
                            args,
                            _,
                            _
                        ) if args.as_ref().len() == 2 && args.contains(&return_value)
                    )
            }));

        let qword_args = db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .find_map(|(_, inst)| match inst {
                RTLInst::Icond(
                    Condition::Cmaskregnotzero(
                        TestRegisterSlice::Full64,
                        TestRegisterSlice::Full64,
                    ),
                    args,
                    _,
                    _,
                ) => Some(args.clone()),
                _ => None,
            })
            .expect("missing full-width TEST condition");
        for arg in qword_args.iter() {
            assert!(
                db.rel_iter::<(RTLReg,)>("is_long")
                    .any(|(reg,)| reg == arg),
                "full-width TEST operand bypassed RTL width classification",
            );
        }

        drop(db);
        assert_emitted_c_preserves_test_widths(&object);
    });
}
