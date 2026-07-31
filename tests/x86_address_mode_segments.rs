use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::abi::Arch;
use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::x86::op::{Addressing, Operation};
use manifold::x86::types::{Address, MachInst, RTLInst, Symbol};

const SYNTHETIC_NODE_MASK: Address = (1u64 << 62) | (1u64 << 63);

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn assemble(stem: &str, target: &str, source_text: &str) -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping x86 address-mode/segment test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_x86_address_mode_segments_{}_{}",
        std::process::id(),
        stem
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join(format!("{stem}.s"));
    let object = dir.join(format!("{stem}.obj"));
    std::fs::write(&source, source_text).ok()?;
    let status = Command::new("clang")
        .args(["-target", target, "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .ok()?;
    assert!(status.success(), "failed to assemble {stem} fixture");
    Some(object)
}

fn segmented_fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            assemble(
                "segments64",
                "x86_64-pc-windows-msvc",
                r#"
                .text
                .globl fs_stack_load
                .def fs_stack_load; .scl 2; .type 32; .endef
fs_stack_load:
                movl %fs:8(%rsp), %eax
                retq

                .globl gs_home_store
                .def gs_home_store; .scl 2; .type 32; .endef
gs_home_store:
                movl %ecx, %gs:8(%rsp)
                retq

                .globl addr32_pointer_load
                .def addr32_pointer_load; .scl 2; .type 32; .endef
addr32_pointer_load:
                movl 4(%ecx), %eax
                retq
"#,
            )
        })
        .as_deref()
}

fn load_asm(object: &Path) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    CanaryVlaPass.run(&mut db);
    AsmPass.run(&mut db);
    db
}

fn finish_rtl(db: &mut DecompileDB) {
    StackAnalysisPass.run(db);
    MachPass.run(db);
    LinearPass.run(db);
    RTLPass.run(db);
}

fn instruction_for_operand(db: &DecompileDB, wanted: Symbol) -> Address {
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
        .find_map(|(address, _, _, _, op1, op2, op3, op4, _, _)| {
            [*op1, *op2, *op3, *op4]
                .contains(&wanted)
                .then_some(*address)
        })
        .expect("memory operand is attached to an instruction")
}

fn rtl_bears_address_or_scalarized_stack(inst: &RTLInst) -> bool {
    matches!(
        inst,
        RTLInst::Iload(..)
            | RTLInst::Istore(..)
            | RTLInst::Iop(
                Operation::Olea(_) | Operation::Oleal(_) | Operation::Omove,
                _,
                _
            )
    )
}

fn run_with_pipeline_stack(name: &str, test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("failed to spawn x86 address-mode test thread")
        .join()
        .expect("x86 address-mode test thread panicked");
}

#[test]
fn long_mode_addr32_pointer_load_remains_explicit() {
    let Some(object) = segmented_fixture() else {
        return;
    };
    let object = object.to_path_buf();
    run_with_pipeline_stack("long-mode-addr32", move || {
        let mut db = load_asm(&object);
        assert_eq!(db.abi().arch, Arch::X86_64);

        let access = db
            .rel_iter::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>("op_indirect")
            .find_map(|(operand, segment, base, index, _, disp, size)| {
                (*segment == "NONE"
                    && *base == "ECX"
                    && *index == "NONE"
                    && *disp == 4
                    && *size == 4)
                    .then_some(instruction_for_operand(&db, *operand))
            })
            .expect("long-mode addr32 pointer operand");
        assert!(db
            .rel_iter::<(Address, u8)>("instruction_address_size")
            .any(|(address, size)| (*address, *size) == (access, 4)));

        let mach: Vec<_> = db
            .rel_iter::<(Address, MachInst)>("mach_inst")
            .filter_map(|(address, inst)| (*address == access).then_some(inst.clone()))
            .collect();
        assert!(
            mach.iter().any(|inst| matches!(
                inst,
                MachInst::Mload(_, Addressing::Aaddr32(inner), args, _)
                    if matches!(inner.as_ref(), Addressing::Aindexed(4)) && args.len() == 1
            )),
            "long-mode addr32 pointer load lost its explicit width: {mach:#?}"
        );
        assert!(!db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address_seed")
            .any(|(_, row_access, _)| *row_access == access));

        finish_rtl(&mut db);
        assert!(db
            .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| {
                (*node & !SYNTHETIC_NODE_MASK) == access
                    && matches!(inst,
                        RTLInst::Iload(_, Addressing::Aaddr32(inner), args, _)
                            if matches!(inner.as_ref(), Addressing::Aindexed(4)) && args.len() == 1)
            }));
    });
}

#[test]
fn fs_gs_stack_operands_are_structured_unsupported_and_never_folded() {
    let Some(object) = segmented_fixture() else {
        return;
    };
    let object = object.to_path_buf();
    run_with_pipeline_stack("segmented-stack-operands", move || {
        let mut db = load_asm(&object);
        assert_eq!(db.abi().arch, Arch::X86_64);

        let accesses: Vec<_> = db
            .rel_iter::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>("op_indirect")
            .filter_map(|(operand, segment, base, index, _, disp, _)| {
                (matches!(*segment, "FS" | "GS")
                    && *base == "RSP"
                    && *index == "NONE"
                    && *disp == 8)
                    .then_some((instruction_for_operand(&db, *operand), *segment))
            })
            .collect();
        assert_eq!(accesses.len(), 2, "missing explicit FS/GS stack operands");

        for (access, segment) in &accesses {
            assert!(
                db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address_seed")
                    .any(|(_, row_access, reason)| {
                        *row_access == *access && *reason == "unsupported-stack-address"
                    }),
                "{segment} stack operand lacks a structured diagnostic"
            );
            assert!(!db
                .rel_iter::<(Address, Symbol, i64)>("stack_def")
                .any(|(address, _, _)| *address == *access));
            assert!(!db
                .rel_iter::<(Address, Symbol, i64)>("stack_use")
                .any(|(address, _, _)| *address == *access));
            assert!(!db
                .rel_iter::<(Address, MachInst)>("mach_inst")
                .any(|(address, _)| *address == *access));
        }

        finish_rtl(&mut db);
        for (access, segment) in accesses {
            assert!(
                db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                    .any(|(_, row_access, reason)| {
                        (*row_access, *reason) == (access, "unsupported-stack-address")
                    }),
                "{segment} stack diagnostic was lost before RTL"
            );
            let surviving: Vec<_> = db
                .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
                .filter_map(|(node, inst)| {
                    ((*node & !SYNTHETIC_NODE_MASK) == access
                        && rtl_bears_address_or_scalarized_stack(inst))
                    .then_some((*node, inst.clone()))
                })
                .collect();
            assert!(
                surviving.is_empty(),
                "{segment} stack operand retained an address/scalar-slot candidate: {surviving:#?}"
            );
        }
    });
}
