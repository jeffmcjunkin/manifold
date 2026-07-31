use std::collections::HashSet;
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
use manifold::x86::types::{Address, LTLInst, MachInst, MemoryChunk, RTLInst, Symbol, Typ, XType};

const SYNTH1: Address = 1u64 << 62;

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping stack/home relation test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_stack_home_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl sp_indexed_alias_collision
        .def sp_indexed_alias_collision; .scl 2; .type 32; .endef
sp_indexed_alias_collision:
        subq $40, %rsp
        movl %edx, 8(%rsp)
        movl %ecx, %eax
        movl %eax, 8(%rsp,%rax,4)
        movl 8(%rsp,%rax,4), %eax
        addq $40, %rsp
        retq

        .globl home_alias_roundtrip
        .def home_alias_roundtrip; .scl 2; .type 32; .endef
home_alias_roundtrip:
        movq %rsp, %rax
        movl %edx, 16(%rax)
        movl 16(%rax), %eax
        retq

        .globl home_pointer_indexed
        .def home_pointer_indexed; .scl 2; .type 32; .endef
home_pointer_indexed:
        movq %rsp, %r10
        movq %rcx, 8(%r10)
        movq 8(%rsp), %rax
        movl (%rax,%rdx,4), %r8d
        movl %r8d, %eax
        retq

        .globl wrong_home_source
        .def wrong_home_source; .scl 2; .type 32; .endef
wrong_home_source:
        movq %rsp, %rax
        movl %ecx, 16(%rax)
        movl 16(%rsp), %eax
        retq

        .globl home_reassigned
        .def home_reassigned; .scl 2; .type 32; .endef
home_reassigned:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl sp_immediate_rmw
        .def sp_immediate_rmw; .scl 2; .type 32; .endef
sp_immediate_rmw:
        subq $32, %rsp
        movl $7, 12(%rsp)
        movl %edx, %eax
        subl 12(%rsp), %eax
        addl $1, 12(%rsp)
        addq $32, %rsp
        retq

        .globl fifth_imul
        .def fifth_imul; .scl 2; .type 32; .endef
fifth_imul:
        imull $7, 40(%rsp), %eax
        retq

        .globl sixth_to_double
        .def sixth_to_double; .scl 2; .type 32; .endef
sixth_to_double:
        cvtsi2sdl 48(%rsp), %xmm0
        retq

        .globl caller_home_vs_outgoing
        .def caller_home_vs_outgoing; .scl 2; .type 32; .endef
caller_home_vs_outgoing:
        movl %edx, 16(%rsp)
        subq $40, %rsp
        movl %ecx, 32(%rsp)
        callq callee_eighth
        addq $40, %rsp
        retq

        .globl callee_eighth
        .def callee_eighth; .scl 2; .type 32; .endef
callee_eighth:
        movl 64(%rsp), %eax
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
    assert!(status.success(), "stack/home fixture assembly failed");
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

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .find_map(|(symbol, start, end)| (*symbol == name).then_some((*start, *end)))
        .unwrap_or_else(|| panic!("missing fixture function {name}"))
}

fn in_span(address: Address, span: (Address, Address)) -> bool {
    address >= span.0 && address < span.1
}

fn rtl_candidates(db: &DecompileDB, address: Address) -> Vec<RTLInst> {
    db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(row_address, inst)| (*row_address == address).then_some(inst.clone()))
        .collect()
}

fn assert_sp_indexed_relations(db: &DecompileDB) {
    let span = function_span(db, "sp_indexed_alias_collision");
    let mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, span))
        .cloned()
        .collect();

    let indexed_stores: Vec<_> = mach
        .iter()
        .filter(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mstore(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::SP, Mreg::AX]
            )
        })
        .collect();
    let indexed_loads: Vec<_> = mach
        .iter()
        .filter(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::SP, Mreg::AX]
            )
        })
        .collect();
    assert_eq!(
        indexed_stores.len(),
        1,
        "missing/duplicate SP indexed store: {mach:#x?}"
    );
    assert_eq!(
        indexed_loads.len(),
        1,
        "missing/duplicate SP indexed load: {mach:#x?}"
    );

    let store_addr = indexed_stores[0].0;
    let load_addr = indexed_loads[0].0;
    let store_candidates = rtl_candidates(db, store_addr | SYNTH1);
    let load_candidates = rtl_candidates(db, load_addr | SYNTH1);
    assert_eq!(
        store_candidates
            .iter()
            .filter(|inst| matches!(inst, RTLInst::Istore(..)))
            .count(),
        1,
        "SP indexed store must have one synthetic Istore: {store_candidates:#x?}"
    );
    assert!(store_candidates.iter().any(|inst| {
        matches!(
            inst,
            RTLInst::Istore(_, Addressing::Aindexed2scaled(4, 0), args, src)
                if args.len() == 2 && args[1] == *src
        )
    }));
    assert_eq!(
        load_candidates
            .iter()
            .filter(|inst| matches!(inst, RTLInst::Iload(..)))
            .count(),
        1,
        "SP dst/index collision must have one synthetic Iload: {load_candidates:#x?}"
    );
    assert!(load_candidates.iter().any(|inst| {
        matches!(
            inst,
            RTLInst::Iload(_, Addressing::Aindexed2scaled(4, 0), args, dst)
                if args.len() == 2 && args[1] != *dst
        )
    }));

    let scalar_addr = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span) && matches!(inst, LTLInst::Lsetstack(_, _, 8, _)))
                .then_some(*address)
        })
        .expect("missing scalar SP slot used to anchor indexed base");
    let scalar_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == span.0 && *address == scalar_addr && *offset == 8).then_some(*reg)
        })
        .collect();
    let indexed_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == span.0 && *address == store_addr && *offset == 8).then_some(*reg)
        })
        .collect();
    assert!(
        !scalar_vars.is_disjoint(&indexed_vars),
        "synthetic SP base must alias the same-offset scalar local: scalar={scalar_vars:#x?}, indexed={indexed_vars:#x?}"
    );
}

fn assert_home_relations(db: &DecompileDB) {
    let span = function_span(db, "home_alias_roundtrip");
    let spills: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(
        spills.len(),
        1,
        "expected one exact /homeparams spill: {spills:#x?}"
    );
    let (spill_addr, _, source, position) = spills[0];
    assert_eq!((source, position), (Mreg::DX, 1));
    assert!(db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .any(|(func, use_addr, alias, base)| {
            (*func, *use_addr, *alias, *base) == (span.0, spill_addr, Mreg::AX, 0)
        }));

    let reloads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(reloads.len(), 1, "expected one home reload: {reloads:#x?}");
    let (reload_addr, _, reload_source, reload_pos) = reloads[0];
    assert_eq!((reload_source, reload_pos), (Mreg::DX, 1));

    let spill_candidates = rtl_candidates(db, spill_addr);
    assert!(spill_candidates.contains(&RTLInst::Inop));
    assert!(
        !spill_candidates
            .iter()
            .any(|inst| matches!(inst, RTLInst::Istore(..))),
        "home spill must not survive as an address-valued Istore: {spill_candidates:#x?}"
    );
    let entry_dx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| (*address == span.0 && *reg == Mreg::DX).then_some(*id))
        .collect();
    let reload_candidates = rtl_candidates(db, reload_addr);
    assert!(reload_candidates.iter().any(|inst| {
        matches!(inst, RTLInst::Iop(Operation::Omove, args, _) if args.len() == 1 && entry_dx.contains(&args[0]))
    }));
    assert!(
        !reload_candidates
            .iter()
            .any(|inst| matches!(inst, RTLInst::Iload(..))),
        "home reload must be a parameter value move, not a frame load: {reload_candidates:#x?}"
    );

    let wrong = function_span(db, "wrong_home_source");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, _, _)| *func == wrong.0));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, _, _)| *func == wrong.0));

    let reassigned = function_span(db, "home_reassigned");
    assert!(db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, source, position)| {
            (*func, *source, *position) == (reassigned.0, Mreg::CX, 0)
        }));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, _, _)| *func == reassigned.0));
    let reassigned_reload = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, reassigned)
                && matches!(inst, MachInst::Mgetstack(8, Typ::Tany64, Mreg::AX)))
            .then_some(*address)
        })
        .expect("missing reassigned home-slot reload");
    let reassigned_entry_cx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| {
            (*address == reassigned.0 && *reg == Mreg::CX).then_some(*id)
        })
        .collect();
    assert!(!reassigned_entry_cx.is_empty());
    let reassigned_candidates = rtl_candidates(db, reassigned_reload);
    assert!(
        !reassigned_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(Operation::Omove, args, _)
                    if args.len() == 1 && reassigned_entry_cx.contains(&args[0])
            )
        }),
        "a reassigned home slot must read its reaching value, not the entry parameter: {reassigned_candidates:#x?}"
    );
}

fn assert_home_pointer_reload_beats_shadow_address(db: &DecompileDB) {
    let span = function_span(db, "home_pointer_indexed");
    let reloads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(
        reloads.len(),
        1,
        "expected one pointer home reload: {reloads:#x?}"
    );
    let (reload_addr, _, source, position) = reloads[0];
    assert_eq!((source, position), (Mreg::CX, 0));

    let reload_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| *address == reload_addr)
        .map(|(_, inst)| inst.clone())
        .collect();
    assert!(
        reload_mach
            .iter()
            .any(|inst| matches!(inst, MachInst::Mgetstack(8, Typ::Tany64, Mreg::AX))),
        "pointer home reload lost its value interpretation: {reload_mach:#x?}"
    );
    assert!(
        reload_mach.iter().any(|inst| {
            matches!(
                inst,
                MachInst::Mop(Operation::Olea(Addressing::Ainstack(8)), args, Mreg::AX)
                    if args.is_empty()
            )
        }),
        "fixture must retain the competing address interpretation: {reload_mach:#x?}"
    );

    let entry_cx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| (*address == span.0 && *reg == Mreg::CX).then_some(*id))
        .collect();
    let reload_candidates = rtl_candidates(db, reload_addr);
    let reload_value = reload_candidates
        .iter()
        .find_map(|inst| match inst {
            RTLInst::Iop(Operation::Omove, args, dst)
                if args.len() == 1 && entry_cx.contains(&args[0]) =>
            {
                Some(*dst)
            }
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!("home pointer reload did not select the incoming value: {reload_candidates:#x?}")
        });
    assert!(
        !reload_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(Operation::Olea(Addressing::Ainstack(8)), args, _)
                    if args.is_empty()
            )
        }),
        "home pointer reload kept a competing stack address: {reload_candidates:#x?}"
    );

    let indexed_addr = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span)
                && matches!(
                    inst,
                    MachInst::Mload(
                        MemoryChunk::MInt32,
                        Addressing::Aindexed2scaled(4, 0),
                        args,
                        Mreg::R8
                    ) if args.as_ref() == &[Mreg::AX, Mreg::DX]
                ))
            .then_some(*address)
        })
        .expect("missing pointer/index load after the home reload");
    let indexed_candidates = rtl_candidates(db, indexed_addr);
    assert!(
        indexed_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 0),
                    args,
                    _
                ) if args.len() == 2 && args[0] == reload_value
            )
        }),
        "indexed dereference did not consume the homed pointer value: reload={reload_candidates:#x?}, indexed={indexed_candidates:#x?}"
    );
}

fn assert_stack_param_and_rmw_relations(db: &DecompileDB) {
    let rmw = function_span(db, "sp_immediate_rmw");
    let inits: Vec<_> = db
        .rel_iter::<(Address, i64, i64, Typ)>("mach_imm_stack_init")
        .filter(|(address, _, _, _)| in_span(*address, rmw))
        .copied()
        .collect();
    assert!(
        inits
            .iter()
            .any(|(_, offset, value, _)| (*offset, *value) == (12, 7)),
        "SP immediate local must retain its raw displacement: {inits:#x?}"
    );
    assert!(!db
        .rel_iter::<(Address, usize)>("stack_param_ordinal")
        .any(|(func, _)| *func == rmw.0));

    let (load_addr, load_op) = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64, Mreg)>("arith_load_op")
        .find_map(|(address, op, _, base, disp, _)| {
            (in_span(*address, rmw) && *base == Mreg::SP && *disp == 12 && *op == Operation::Osub)
                .then_some((*address, op.clone()))
        })
        .expect("missing SP local memory-source RMW");
    let (store_addr, store_op) = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64)>("arith_store_imm")
        .find_map(|(address, op, _, base, disp)| {
            (in_span(*address, rmw) && *base == Mreg::SP && *disp == 12)
                .then_some((*address, op.clone()))
        })
        .expect("missing SP local memory-destination RMW");
    let load_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == rmw.0 && *address == load_addr && *offset == 12).then_some(*reg)
        })
        .collect();
    let store_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == rmw.0 && *address == store_addr && *offset == 12).then_some(*reg)
        })
        .collect();
    assert!(!load_vars.is_empty() && !store_vars.is_empty());
    assert!(rtl_candidates(db, load_addr | SYNTH1).iter().any(|inst| {
        matches!(inst, RTLInst::Iop(op, args, _) if *op == load_op && args.len() == 2 && load_vars.contains(&args[1]))
    }));
    assert!(rtl_candidates(db, store_addr).iter().any(|inst| {
        matches!(inst, RTLInst::Iop(op, args, dst) if *op == store_op && args.len() == 1 && store_vars.contains(&args[0]) && store_vars.contains(dst))
    }));

    let fifth = function_span(db, "fifth_imul");
    let sixth = function_span(db, "sixth_to_double");
    let fifth_access = db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .find(|(_, func, disp, ordinal)| *func == fifth.0 && *disp == 40 && *ordinal == 0)
        .copied()
        .expect("fifth argument must retain ABI ordinal zero");
    let sixth_access = db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .find(|(_, func, disp, ordinal)| *func == sixth.0 && *disp == 48 && *ordinal == 1)
        .copied()
        .expect("sparse sixth argument must retain ABI ordinal one");
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (fifth.0, 1)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (sixth.0, 2)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (fifth.0, 5)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (sixth.0, 6)));

    for (access, expected_op, expected_type) in [
        (fifth_access, Operation::Omulimm(7), XType::Xint),
        (sixth_access, Operation::Ofloatofint, XType::Xint),
    ] {
        let candidates = rtl_candidates(db, access.0 | SYNTH1);
        let param = candidates
            .iter()
            .find_map(|inst| match inst {
                RTLInst::Iop(op, args, _) if *op == expected_op && args.len() == 1 => Some(args[0]),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("stack unary op did not consume a parameter: {candidates:#x?}")
            });
        assert!(db
            .rel_iter::<(Address, u64, XType)>("emit_function_param_type_candidate")
            .any(|(func, reg, ty)| (*func, *reg, *ty) == (access.1, param, expected_type)));
    }
}

fn assert_home_store_is_not_outgoing(db: &DecompileDB) {
    let caller = function_span(db, "caller_home_vs_outgoing");
    let callee = function_span(db, "callee_eighth");
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (callee.0, 4)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (callee.0, 8)));
    let call = db
        .rel_iter::<(Address, Address)>("call_target_func")
        .find_map(|(call, target)| (in_span(*call, caller) && *target == callee.0).then_some(*call))
        .expect("missing local fixture call");
    let evidence: HashSet<_> = db
        .rel_iter::<(Address, usize)>("call_has_arg_evidence")
        .filter_map(|(site, position)| (*site == call).then_some(*position))
        .collect();
    assert!(
        evidence.contains(&4),
        "real outgoing fifth argument was lost: {evidence:?}"
    );
    assert!(
        !evidence.contains(&7),
        "entry home spill was misclassified as an outgoing eighth argument: {evidence:?}"
    );
}

#[test]
fn coff_stack_and_home_relations_preserve_values_and_abi_ordinals() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("stack-home-relations".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let db = load_rtl_relations(&object);
            assert_sp_indexed_relations(&db);
            assert_home_relations(&db);
            assert_home_pointer_reload_beats_shadow_address(&db);
            assert_stack_param_and_rmw_relations(&db);
            assert_home_store_is_not_outgoing(&db);
        })
        .expect("failed to spawn stack/home relation test thread")
        .join()
        .expect("stack/home relation test thread panicked");
}
