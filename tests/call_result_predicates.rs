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
use manifold::x86::op::Condition;
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
fn register_mask_test_keeps_call_result_live_in_rtl() {
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
                            Condition::Cmaskregzero(false),
                            args,
                            _,
                            _
                        ) if args.as_ref().len() == 2 && args.contains(&return_value)
                    )
            }));
    });
}
