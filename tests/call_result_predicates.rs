use std::collections::{HashMap, HashSet};
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
use manifold::x86::op::{Condition, TestRegisterPredicate, TestRegisterSlice};
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

        .globl testb_high8_self_js_uses_slice_sign
        .def testb_high8_self_js_uses_slice_sign; .scl 2; .type 32; .endef
testb_high8_self_js_uses_slice_sign:
        movl $0x8000, %eax
        testb %ah, %ah
        js .Lhigh8_sign_set
        xorl %eax, %eax
        retq
.Lhigh8_sign_set:
        movl $1, %eax
        retq

        .globl testw_self_js_uses_word_sign
        .def testw_self_js_uses_word_sign; .scl 2; .type 32; .endef
testw_self_js_uses_word_sign:
        movl $0x8000, %eax
        testw %ax, %ax
        js .Lword_sign_set
        xorl %eax, %eax
        retq
.Lword_sign_set:
        movl $1, %eax
        retq

        .globl testl_distinct_jns_ignores_qword_sign
        .def testl_distinct_jns_ignores_qword_sign; .scl 2; .type 32; .endef
testl_distinct_jns_ignores_qword_sign:
        movabsq $0x8000000000000001, %rax
        movabsq $0x8000000000000001, %rcx
        testl %eax, %ecx
        jns .Ldword_sign_clear
        xorl %eax, %eax
        retq
.Ldword_sign_clear:
        movl $1, %eax
        retq

        .globl testq_self_jl_uses_qword_sign
        .def testq_self_jl_uses_qword_sign; .scl 2; .type 32; .endef
testq_self_jl_uses_qword_sign:
        movabsq $0x8000000000000000, %rax
        testq %rax, %rax
        jl .Lqword_sign_set
        xorl %eax, %eax
        retq
.Lqword_sign_set:
        movl $1, %eax
        retq

        .globl testw_self_jp_uses_low_byte
        .def testw_self_jp_uses_low_byte; .scl 2; .type 32; .endef
testw_self_jp_uses_low_byte:
        movl $0x100, %eax
        testw %ax, %ax
        jp .Lword_low_byte_even
        xorl %eax, %eax
        retq
.Lword_low_byte_even:
        movl $1, %eax
        retq

        .globl testb_self_all_jcc_families
        .def testb_self_all_jcc_families; .scl 2; .type 32; .endef
testb_self_all_jcc_families:
        movl $0, %eax
        testb %al, %al
        je .Lself_after_je
        jmp .Lself_family_fail
.Lself_after_je:
        movl $1, %eax
        testb %al, %al
        jne .Lself_after_jne
        jmp .Lself_family_fail
.Lself_after_jne:
        testb %al, %al
        jb .Lself_family_fail
        testb %al, %al
        jae .Lself_after_jae
        jmp .Lself_family_fail
.Lself_after_jae:
        movl $0, %eax
        testb %al, %al
        jbe .Lself_after_jbe
        jmp .Lself_family_fail
.Lself_after_jbe:
        movl $1, %eax
        testb %al, %al
        ja .Lself_after_ja
        jmp .Lself_family_fail
.Lself_after_ja:
        movl $0x80, %eax
        testb %al, %al
        js .Lself_after_js
        jmp .Lself_family_fail
.Lself_after_js:
        movl $0x7f, %eax
        testb %al, %al
        jns .Lself_after_jns
        jmp .Lself_family_fail
.Lself_after_jns:
        movl $0x80, %eax
        testb %al, %al
        jl .Lself_after_jl
        jmp .Lself_family_fail
.Lself_after_jl:
        movl $0, %eax
        testb %al, %al
        jle .Lself_after_jle
        jmp .Lself_family_fail
.Lself_after_jle:
        movl $1, %eax
        testb %al, %al
        jg .Lself_after_jg
        jmp .Lself_family_fail
.Lself_after_jg:
        movl $0x7f, %eax
        testb %al, %al
        jge .Lself_after_jge
        jmp .Lself_family_fail
.Lself_after_jge:
        movl $3, %eax
        testb %al, %al
        jp .Lself_after_jp
        jmp .Lself_family_fail
.Lself_after_jp:
        movl $1, %eax
        testb %al, %al
        jnp .Lself_after_jnp
        jmp .Lself_family_fail
.Lself_after_jnp:
        testb %al, %al
        jo .Lself_family_fail
        testb %al, %al
        jno .Lself_family_pass
        jmp .Lself_family_fail
.Lself_family_pass:
        movl $1, %eax
        retq
.Lself_family_fail:
        xorl %eax, %eax
        retq

        .globl testb_distinct_all_jcc_families
        .def testb_distinct_all_jcc_families; .scl 2; .type 32; .endef
testb_distinct_all_jcc_families:
        movl $1, %eax
        movl $1, %ecx
        testb %al, %cl
        je .Ldistinct_family_fail
        movl $1, %eax
        movl $2, %ecx
        testb %al, %cl
        jne .Ldistinct_family_fail
        movl $1, %eax
        movl $1, %ecx
        testb %al, %cl
        jb .Ldistinct_family_fail
        testb %al, %cl
        jae .Ldistinct_after_jae
        jmp .Ldistinct_family_fail
.Ldistinct_after_jae:
        testb %al, %cl
        jbe .Ldistinct_family_fail
        movl $1, %eax
        movl $2, %ecx
        testb %al, %cl
        ja .Ldistinct_family_fail
        movl $0x80, %eax
        movl $0x7f, %ecx
        testb %al, %cl
        js .Ldistinct_family_fail
        movl $0x80, %eax
        movl $0x80, %ecx
        testb %al, %cl
        jns .Ldistinct_family_fail
        movl $0x80, %eax
        movl $0x7f, %ecx
        testb %al, %cl
        jl .Ldistinct_family_fail
        movl $0x81, %eax
        movl $1, %ecx
        testb %al, %cl
        jle .Ldistinct_family_fail
        movl $1, %eax
        movl $2, %ecx
        testb %al, %cl
        jg .Ldistinct_family_fail
        movl $0x80, %eax
        movl $0x81, %ecx
        testb %al, %cl
        jge .Ldistinct_family_fail
        movl $3, %eax
        movl $5, %ecx
        testb %al, %cl
        jp .Ldistinct_family_fail
        movl $1, %eax
        movl $2, %ecx
        testb %al, %cl
        jnp .Ldistinct_family_fail
        movl $1, %eax
        movl $1, %ecx
        testb %al, %cl
        jo .Ldistinct_family_fail
        testb %al, %cl
        jno .Ldistinct_family_pass
        jmp .Ldistinct_family_fail
.Ldistinct_family_pass:
        movl $1, %eax
        retq
.Ldistinct_family_fail:
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
int coff_fn_testb_high8_self_js_uses_slice_sign(void);
int coff_fn_testw_self_js_uses_word_sign(void);
int coff_fn_testl_distinct_jns_ignores_qword_sign(void);
int coff_fn_testq_self_jl_uses_qword_sign(void);
int coff_fn_testw_self_jp_uses_low_byte(void);
int coff_fn_testb_self_all_jcc_families(void);
int coff_fn_testb_distinct_all_jcc_families(void);

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
    if (coff_fn_testb_self_all_jcc_families() != 1) return 19;
    if (coff_fn_testb_distinct_all_jcc_families() != 1) return 20;
    if (coff_fn_testb_high8_self_js_uses_slice_sign() != 1) return 21;
    if (coff_fn_testw_self_js_uses_word_sign() != 1) return 22;
    if (coff_fn_testl_distinct_jns_ignores_qword_sign() != 1) return 23;
    if (coff_fn_testq_self_jl_uses_qword_sign() != 1) return 24;
    if (coff_fn_testw_self_jp_uses_low_byte() != 1) return 25;
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
            !db.rel_iter::<(RTLReg,)>("is_not_ptr")
                .any(|(reg,)| *reg == return_value),
            "a narrow TEST alone incorrectly proved that its operand is not a pointer",
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
                            Condition::Ctestregister(
                                TestRegisterPredicate::Zero,
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
                    Condition::Ctestregister(
                        TestRegisterPredicate::Nonzero,
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

        for function_name in [
            "testb_self_all_jcc_families",
            "testb_distinct_all_jcc_families",
        ] {
            let (family_start, family_end) = db
                .rel_iter::<(Symbol, Address, Address)>("func_span")
                .find_map(|(name, start, end)| {
                    (*name == function_name || *name == format!("coff_fn_{function_name}"))
                        .then_some((*start, *end))
                })
                .unwrap_or_else(|| panic!("missing TEST/Jcc family function {function_name}"));
            let test_nodes: HashSet<_> = db
                .rel_iter::<(Address, Symbol, Symbol)>("ptest")
                .filter_map(|(address, _, _)| {
                    (family_start <= *address && *address < family_end).then_some(*address)
                })
                .collect();
            assert_eq!(
                test_nodes.len(),
                16,
                "{function_name} lost a decoded TEST/Jcc family",
            );

            let conditional_candidates: Vec<_> = db
                .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter_map(|(node, inst)| {
                    (family_start <= *node && *node < family_end)
                        .then_some((*node, inst))
                        .and_then(|(node, inst)| match inst {
                            RTLInst::Icond(condition, args, if_true, if_false) => {
                                Some((node, condition, args, if_true, if_false))
                            }
                            _ => None,
                        })
                })
                .collect();

            // Linear/RTL may preserve both the physical fallthrough through
            // an unconditional jump and its CFG-equivalent collapsed target.
            // Those candidates differ only in their false successor. Every
            // TEST must nevertheless have one condition/argument/taken-edge
            // lowering, and no condition may remain at the raw Jcc address.
            let mut lowering_by_test = HashMap::new();
            for (node, condition, args, if_true, _) in &conditional_candidates {
                assert!(
                    test_nodes.contains(node),
                    "{function_name} retained a competing Jcc-address candidate at {node:#x}: {condition:?}",
                );
                let lowering = (*condition, *args, *if_true);
                if let Some(previous) = lowering_by_test.insert(*node, lowering) {
                    assert_eq!(
                        previous, lowering,
                        "{function_name} produced conflicting RTL lowerings at {node:#x}",
                    );
                }
            }
            assert_eq!(
                lowering_by_test.keys().copied().collect::<HashSet<_>>(),
                test_nodes,
                "{function_name} produced a missing or misplaced RTL branch lowering: {conditional_candidates:#x?}",
            );
            for (node, condition, args, _, _) in conditional_candidates {
                match condition {
                    Condition::Ctestregister(_, lhs, rhs) => {
                        assert_eq!(lhs.width_bits(), 8);
                        assert_eq!(rhs.width_bits(), 8);
                        assert_eq!(args.len(), 2);
                    }
                    Condition::Cconst(_) => assert!(args.is_empty()),
                    other => panic!(
                        "{function_name} used a non-TEST condition at {node:#x}: {other:?}"
                    ),
                }
            }
        }

        drop(db);
        assert_emitted_c_preserves_test_widths(&object);
    });
}
