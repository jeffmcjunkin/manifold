use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::mreg::Mreg;
use manifold::x86::op::{Addressing, Operation};
use manifold::x86::types::{Address, LTLInst, MemoryChunk, RTLInst, Symbol};

const SYNTHETIC_NODE_MASK: Address = (1u64 << 62) | (1u64 << 63);

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping stack RMW overlap test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_stack_param_rmw_overlap_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl imm_overlap_middle
        .def imm_overlap_middle; .scl 2; .type 32; .endef
imm_overlap_middle:
        movb $7, 41(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl imm_overlap_last
        .def imm_overlap_last; .scl 2; .type 32; .endef
imm_overlap_last:
        movb $7, 43(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl imm_nonoverlap_after
        .def imm_nonoverlap_after; .scl 2; .type 32; .endef
imm_nonoverlap_after:
        movb $7, 44(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl imm_nonoverlap_before
        .def imm_nonoverlap_before; .scl 2; .type 32; .endef
imm_nonoverlap_before:
        movb $7, 39(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl word_overlap_left
        .def word_overlap_left; .scl 2; .type 32; .endef
word_overlap_left:
        movw $7, 39(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl word_nonoverlap_left
        .def word_nonoverlap_left; .scl 2; .type 32; .endef
word_nonoverlap_left:
        movw $7, 38(%rsp)
        addl $1, 40(%rsp)
        retq

        .globl branch_may_overlap
        .def branch_may_overlap; .scl 2; .type 32; .endef
branch_may_overlap:
        testl %ecx, %ecx
        je 1f
        movb $7, 41(%rsp)
1:
        addl $1, 40(%rsp)
        retq

        .globl reg_overlap_middle
        .def reg_overlap_middle; .scl 2; .type 32; .endef
reg_overlap_middle:
        movb $7, 42(%rsp)
        addl %ecx, 40(%rsp)
        retq

        .globl reg_nonoverlap_after
        .def reg_nonoverlap_after; .scl 2; .type 32; .endef
reg_nonoverlap_after:
        movb $7, 44(%rsp)
        addl %ecx, 40(%rsp)
        retq

        .globl adjusted_sp_overlap
        .def adjusted_sp_overlap; .scl 2; .type 32; .endef
adjusted_sp_overlap:
        subq $32, %rsp
        movb $7, 73(%rsp)
        addl $1, 72(%rsp)
        addq $32, %rsp
        retq

        .globl bp_overlap_middle
        .def bp_overlap_middle; .scl 2; .type 32; .endef
bp_overlap_middle:
        pushq %rbp
        movq %rsp, %rbp
        movb $7, 49(%rbp)
        addl $1, 48(%rbp)
        popq %rbp
        retq

        .globl load_overlap_middle
        .def load_overlap_middle; .scl 2; .type 32; .endef
load_overlap_middle:
        movb $7, 41(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl load_overlap_same_start
        .def load_overlap_same_start; .scl 2; .type 32; .endef
load_overlap_same_start:
        movb $7, 40(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl load_nonoverlap_after
        .def load_nonoverlap_after; .scl 2; .type 32; .endef
load_nonoverlap_after:
        movb $7, 44(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl load_nonoverlap_before
        .def load_nonoverlap_before; .scl 2; .type 32; .endef
load_nonoverlap_before:
        movb $7, 39(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl load_exact_definition
        .def load_exact_definition; .scl 2; .type 32; .endef
load_exact_definition:
        movl $7, 40(%rsp)
        movl 40(%rsp), %eax
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
    assert!(
        status.success(),
        "stack RMW overlap fixture assembly failed"
    );
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
}

fn load_rtl_relations(object: &Path) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    CanaryVlaPass.run(&mut db);
    AsmPass.run(&mut db);
    StackAnalysisPass.run(&mut db);
    MachPass.run(&mut db);
    LinearPass.run(&mut db);
    RTLPass.run(&mut db);
    db
}

fn function_spans(db: &DecompileDB) -> HashMap<Symbol, (Address, Address)> {
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .map(|(name, start, end)| {
            let canonical = name.strip_prefix("coff_fn_").unwrap_or(name);
            (canonical, (*start, *end))
        })
        .collect()
}

fn in_span(address: Address, span: (Address, Address)) -> bool {
    address >= span.0 && address < span.1
}

fn immediate_rmw(db: &DecompileDB, span: (Address, Address)) -> Address {
    db.rel_iter::<(
        Address,
        manifold::x86::op::Operation,
        MemoryChunk,
        Mreg,
        i64,
    )>("arith_store_imm")
        .find_map(|(addr, _, _, _, _)| in_span(*addr, span).then_some(*addr))
        .expect("fixture function is missing its immediate stack RMW")
}

fn register_rmw(db: &DecompileDB, span: (Address, Address)) -> Address {
    db.rel_iter::<(
        Address,
        manifold::x86::op::Operation,
        MemoryChunk,
        Mreg,
        i64,
        Mreg,
    )>("arith_store_reg")
        .find_map(|(addr, _, _, _, _, _)| in_span(*addr, span).then_some(*addr))
        .expect("fixture function is missing its register stack RMW")
}

fn ordinary_load(db: &DecompileDB, span: (Address, Address)) -> Address {
    db.rel_iter::<(Address, LTLInst)>("ltl_inst")
        .find_map(|(addr, inst)| {
            let is_load = match inst {
                LTLInst::Lgetstack(_, ofs, _, _) => *ofs == 40,
                LTLInst::Lload(_, Addressing::Aindexed(ofs), args, _) => {
                    *ofs == 40 && args.iter().any(|reg| matches!(*reg, Mreg::SP | Mreg::BP))
                }
                _ => false,
            };
            (in_span(*addr, span) && is_load).then_some(*addr)
        })
        .expect("fixture function is missing its ordinary stack load")
}

fn has_param_access(db: &DecompileDB, address: Address) -> bool {
    db.rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(addr, _, _, _)| *addr == address)
}

fn is_address_bearing(inst: &RTLInst) -> bool {
    matches!(
        inst,
        RTLInst::Iload(..)
            | RTLInst::Istore(..)
            | RTLInst::Iop(Operation::Olea(_) | Operation::Oleal(_), _, _)
    )
}

fn assert_unsupported_read(
    db: &DecompileDB,
    name: &str,
    span: (Address, Address),
    address: Address,
) {
    assert!(!db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(addr, _, _, _)| *addr == address));
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, access, reason)| {
                (*func, *access, *reason) == (span.0, address, "unsupported-stack-address")
            }),
        "{name} lost its structured unsupported-stack-address diagnostic"
    );

    let candidates: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, inst)| {
            (*node & !SYNTHETIC_NODE_MASK) == address && is_address_bearing(inst)
        })
        .collect();
    assert!(
        candidates.is_empty(),
        "{name} retained address-bearing RTL candidates: {candidates:#x?}"
    );
}

#[test]
fn incoming_stack_reads_use_decoded_byte_range_overlap() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("stack-param-rmw-overlap".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let db = load_rtl_relations(&object);
            let spans = function_spans(&db);
            let overlaps: HashSet<Address> = db
                .rel_iter::<(Address, Address)>("stack_param_partial_write")
                .map(|(read, _)| *read)
                .collect();

            for name in [
                "imm_overlap_middle",
                "imm_overlap_last",
                "word_overlap_left",
                "branch_may_overlap",
                "adjusted_sp_overlap",
                "bp_overlap_middle",
            ] {
                let addr = immediate_rmw(&db, spans[name]);
                assert!(
                    overlaps.contains(&addr),
                    "{name} must be vetoed by its byte-range-overlapping write"
                );
                assert_unsupported_read(&db, name, spans[name], addr);
            }

            let reg_overlap = register_rmw(&db, spans["reg_overlap_middle"]);
            assert!(
                overlaps.contains(&reg_overlap),
                "register-source RMW must use the same overlap veto"
            );
            assert_unsupported_read(
                &db,
                "reg_overlap_middle",
                spans["reg_overlap_middle"],
                reg_overlap,
            );

            for name in ["load_overlap_middle", "load_overlap_same_start"] {
                let addr = ordinary_load(&db, spans[name]);
                assert!(
                    overlaps.contains(&addr),
                    "{name} must be vetoed by its byte-range-overlapping write"
                );
                assert_unsupported_read(&db, name, spans[name], addr);
            }

            for name in [
                "imm_nonoverlap_after",
                "imm_nonoverlap_before",
                "word_nonoverlap_left",
                "reg_nonoverlap_after",
            ] {
                let addr = if name.starts_with("reg_") {
                    register_rmw(&db, spans[name])
                } else {
                    immediate_rmw(&db, spans[name])
                };
                assert!(
                    !overlaps.contains(&addr),
                    "{name} touches only a boundary-adjacent byte"
                );
                assert!(
                    has_param_access(&db, addr),
                    "{name} lost supported incoming-parameter recovery"
                );
                assert!(!db
                    .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                    .any(|(_, access, reason)| {
                        (*access, *reason) == (addr, "unsupported-stack-address")
                    }));
            }

            for name in ["load_nonoverlap_after", "load_nonoverlap_before"] {
                let addr = ordinary_load(&db, spans[name]);
                assert!(
                    !overlaps.contains(&addr),
                    "{name} touches only a boundary-adjacent byte"
                );
                assert!(
                    has_param_access(&db, addr),
                    "{name} lost supported incoming-parameter recovery"
                );
            }

            let exact = ordinary_load(&db, spans["load_exact_definition"]);
            assert!(!overlaps.contains(&exact));
            assert!(!has_param_access(&db, exact));
            assert!(!db
                .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, access, reason)| {
                    (*func, *access, *reason)
                        == (
                            spans["load_exact_definition"].0,
                            exact,
                            "unsupported-stack-address",
                        )
                }));

            for name in ["adjusted_sp_overlap", "bp_overlap_middle"] {
                let addr = immediate_rmw(&db, spans[name]);
                assert!(db
                    .rel_iter::<(Address, Address, Mreg, i64, i64, i64)>(
                        "normalized_stack_write_range",
                    )
                    .any(|(write, _, _, _, start, end)| {
                        (*write, *start, *end) == (addr, 40, 44)
                    }));
            }
        })
        .expect("failed to spawn stack RMW overlap test thread")
        .join()
        .expect("stack RMW overlap test thread panicked");
}
