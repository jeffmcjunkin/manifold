use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::mreg::Mreg;
use manifold::x86::op::{Addressing, Operation};
use manifold::x86::types::{Address, MachInst, MemoryChunk, Symbol};

type FloatLoadOp = (
    Address,
    Operation,
    MemoryChunk,
    Addressing,
    Arc<Vec<Mreg>>,
    Mreg,
    bool,
);

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping indexed-memory test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_indexed_memory_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl indexed_and32
        .def indexed_and32; .scl 2; .type 32; .endef
indexed_and32:
        movl %r8d, %eax
        andl 12(%rcx,%rdx,4), %eax
        retq

        .globl indexed_or64
        .def indexed_or64; .scl 2; .type 32; .endef
indexed_or64:
        movq %r8, %rax
        orq 24(%rcx,%rdx,8), %rax
        retq

        .globl indexed_xor32
        .def indexed_xor32; .scl 2; .type 32; .endef
indexed_xor32:
        movl %r8d, %eax
        xorl -4(%rcx,%rdx), %eax
        retq

        .globl indexed_imul32
        .def indexed_imul32; .scl 2; .type 32; .endef
indexed_imul32:
        imull $13, (%rcx,%rdx,4), %eax
        retq

        .globl simple_imul64
        .def simple_imul64; .scl 2; .type 32; .endef
simple_imul64:
        imulq $-7, 16(%rcx), %rax
        retq

        .globl stack_imul32
        .def stack_imul32; .scl 2; .type 32; .endef
stack_imul32:
        subq $16, %rsp
        movl %ecx, 4(%rsp)
        imull $11, 4(%rsp), %eax
        addq $16, %rsp
        retq

        .globl register_imul32
        .def register_imul32; .scl 2; .type 32; .endef
register_imul32:
        imull $5, %ecx, %eax
        retq

        .globl indexed_and_store32
        .def indexed_and_store32; .scl 2; .type 32; .endef
indexed_and_store32:
        andl %r8d, 12(%rcx,%rdx,4)
        movl %r8d, %eax
        retq

        .globl simple_and32
        .def simple_and32; .scl 2; .type 32; .endef
simple_and32:
        movl %edx, %eax
        andl 8(%rcx), %eax
        retq

        .globl stack_cvtsi2ss32
        .def stack_cvtsi2ss32; .scl 2; .type 32; .endef
stack_cvtsi2ss32:
        subq $16, %rsp
        movl %ecx, 4(%rsp)
        cvtsi2ssl 4(%rsp), %xmm0
        addq $16, %rsp
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
    assert!(status.success(), "indexed-memory fixture assembly failed");
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
}

fn load_asm_relations(object: &Path) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    AsmPass.run(&mut db);
    db
}

fn instruction_addresses(db: &DecompileDB, mnemonic: &str) -> Vec<Address> {
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
            (*row_mnemonic == mnemonic).then_some(*address)
        })
        .collect()
}

fn assert_indexed_logic_relations(object: &Path) {
    let db = load_asm_relations(object);
    let loads: Vec<FloatLoadOp> = db
        .rel_iter::<FloatLoadOp>("float_load_op")
        .cloned()
        .collect();

    let expected = [
        (
            Operation::Oand,
            MemoryChunk::MInt32,
            Addressing::Aindexed2scaled(4, 12),
            Mreg::AX,
        ),
        (
            Operation::Oorl,
            MemoryChunk::MInt64,
            Addressing::Aindexed2scaled(8, 24),
            Mreg::AX,
        ),
        (
            Operation::Oxor,
            MemoryChunk::MInt32,
            Addressing::Aindexed2(-4),
            Mreg::AX,
        ),
    ];
    for (op, chunk, addressing, dst) in expected {
        assert!(
            loads.iter().any(
                |(_, row_op, row_chunk, row_addressing, args, row_dst, unary)| {
                    row_op == &op
                        && *row_chunk == chunk
                        && row_addressing == &addressing
                        && args.as_ref() == &[Mreg::CX, Mreg::DX]
                        && *row_dst == dst
                        && !*unary
                }
            ),
            "missing indexed logical load {op:?} {addressing:?}: {loads:#x?}"
        );
    }

    let arith_loads: Vec<_> = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64, Mreg)>("arith_load_op")
        .cloned()
        .collect();
    assert!(
        arith_loads.iter().any(|(_, op, chunk, base, disp, dst)| {
            *op == Operation::Oand
                && *chunk == MemoryChunk::MInt32
                && *base == Mreg::CX
                && *disp == 8
                && *dst == Mreg::AX
        }),
        "simple memory-source AND must remain on arith_load_op: {arith_loads:#x?}"
    );

    let indexed_logic_addrs: HashSet<Address> = loads
        .iter()
        .filter_map(|(address, op, _, _, _, _, _)| {
            matches!(
                op,
                Operation::Oand
                    | Operation::Oandl
                    | Operation::Oor
                    | Operation::Oorl
                    | Operation::Oxor
                    | Operation::Oxorl
            )
            .then_some(*address)
        })
        .collect();
    assert_eq!(indexed_logic_addrs.len(), 3, "unexpected logic load rows");

    let and_addrs: HashSet<Address> = instruction_addresses(&db, "AND").into_iter().collect();
    let arith_and_addrs: HashSet<Address> = arith_loads
        .iter()
        .filter_map(|(address, op, _, _, _, _)| {
            matches!(op, Operation::Oand | Operation::Oandl).then_some(*address)
        })
        .collect();
    let unlowered_as_source: Vec<_> = and_addrs
        .difference(&indexed_logic_addrs)
        .filter(|address| !arith_and_addrs.contains(*address))
        .copied()
        .collect();
    assert_eq!(
        unlowered_as_source.len(),
        1,
        "indexed memory-destination AND must not be reversed into a load: {loads:#x?}"
    );
}

fn assert_imul3_relations(object: &Path) {
    let db = load_asm_relations(object);
    let loads: Vec<FloatLoadOp> = db
        .rel_iter::<FloatLoadOp>("float_load_op")
        .cloned()
        .collect();

    assert!(
        loads
            .iter()
            .any(|(_, op, chunk, addressing, args, dst, unary)| {
                *op == Operation::Omulimm(13)
                    && *chunk == MemoryChunk::MInt32
                    && *addressing == Addressing::Aindexed2scaled(4, 0)
                    && args.as_ref() == &[Mreg::CX, Mreg::DX]
                    && *dst == Mreg::AX
                    && *unary
            }),
        "indexed IMUL3 must retain the full load address and be unary: {loads:#x?}"
    );
    assert!(
        loads
            .iter()
            .any(|(_, op, chunk, addressing, args, dst, unary)| {
                *op == Operation::Omullimm(-7)
                    && *chunk == MemoryChunk::MInt64
                    && *addressing == Addressing::Aindexed(16)
                    && args.as_ref() == &[Mreg::CX]
                    && *dst == Mreg::AX
                    && *unary
            }),
        "simple IMUL3 memory load must use the unary path: {loads:#x?}"
    );
    assert!(
        loads.iter().all(|(_, op, _, _, _, _, unary)| {
            !matches!(op, Operation::Omulimm(_) | Operation::Omullimm(_)) || *unary
        }),
        "IMUL3 memory rows must never retain old dst as a binary operand"
    );

    let stack_ops: Vec<_> = db
        .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("stack_unary_load_op")
        .cloned()
        .collect();
    assert!(
        stack_ops.iter().any(|(_, op, base, disp, dst)| {
            *op == Operation::Omulimm(11) && *base == Mreg::SP && *disp == 4 && *dst == Mreg::AX
        }),
        "stack IMUL3 must consume the canonical slot value: {stack_ops:#x?}"
    );
    assert!(
        stack_ops.iter().any(|(_, op, base, disp, dst)| {
            *op == Operation::Osingleofint && *base == Mreg::SP && *disp == 4 && *dst == Mreg::X0
        }),
        "generalized stack unary relation must preserve CVTSI2SS: {stack_ops:#x?}"
    );

    let mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .cloned()
        .collect();
    assert!(
        mach.iter().any(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mop(Operation::Omulimm(5), args, Mreg::AX)
                    if args.as_ref() == &[Mreg::CX]
            )
        }),
        "register IMUL3 must remain a one-source write to dst: {mach:#x?}"
    );
}

#[test]
fn indexed_logic_memory_sources_lower_from_coff_without_reversing_stores() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("indexed-logic-relations".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_indexed_logic_relations(&object))
        .expect("failed to spawn indexed-logic test thread")
        .join()
        .expect("indexed-logic test thread panicked");
}

#[test]
fn imul3_memory_sources_are_write_only_unary_loads_in_coff() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("imul3-memory-relations".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_imul3_relations(&object))
        .expect("failed to spawn IMUL3 test thread")
        .join()
        .expect("IMUL3 test thread panicked");
}

#[test]
fn stack_unary_relation_replaces_the_cvtsi_specific_api() {
    let outputs = AsmPass.outputs();
    assert!(outputs.contains(&"stack_unary_load_op"));
    assert!(!outputs.contains(&concat!("cvtsi2_", "stack_op")));
}
