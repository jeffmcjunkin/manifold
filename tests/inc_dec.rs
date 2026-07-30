use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::mreg::Mreg;
use manifold::x86::op::{Comparison, Condition, Operation};
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
        eprintln!("skipping INC/DEC test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!("manifold_inc_dec_fixture_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl dec_jne_64
        .def dec_jne_64; .scl 2; .type 32; .endef
dec_jne_64:
        movq %rcx, %rax
.Ldec_loop:
        decq %rax
        movq %rax, %r10
        jne .Ldec_loop
        retq

        .globl inc_je_32
        .def inc_je_32; .scl 2; .type 32; .endef
inc_je_32:
        movl %ecx, %eax
        incl %eax
        je .Linc_done
        retq
.Linc_done:
        retq

        .globl dec_je_switch_chain
        .def dec_je_switch_chain; .scl 2; .type 32; .endef
dec_je_switch_chain:
        movl %edx, %ecx
        testl %edx, %edx
        je .Lcase_zero
        decl %ecx
        je .Lcase_one
        decl %ecx
        je .Lcase_two
        leal 13(%rdx), %eax
        retq
.Lcase_zero:
        movl $101, %eax
        retq
.Lcase_one:
        movl $-7, %eax
        retq
.Lcase_two:
        movl $43, %eax
        retq

        .globl dec_jb_preserves_carry
        .def dec_jb_preserves_carry; .scl 2; .type 32; .endef
dec_jb_preserves_carry:
        movq %rcx, %rax
        decq %rax
        jb .Lcarry_done
        retq
.Lcarry_done:
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
        eprintln!("skipping INC/DEC test: fixture assembly failed");
        return None;
    }
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
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
    )>("unrefinedinstruction")
        .filter_map(|(address, _, _, row_mnemonic, _, _, _, _, _, _)| {
            (*row_mnemonic == mnemonic).then_some(*address)
        })
        .collect()
}

fn assert_inc_dec_results(object: &Path) {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);

    let decs = instruction_addresses(&db, "DEC");
    let incs = instruction_addresses(&db, "INC");
    let jnes = instruction_addresses(&db, "JNE");
    let jes = instruction_addresses(&db, "JE");
    let jbs = instruction_addresses(&db, "JB");
    assert_eq!(decs.len(), 4, "fixture must contain four DEC instructions");
    assert_eq!(incs.len(), 1, "fixture must contain one INC instruction");
    assert_eq!(jnes.len(), 1, "fixture must contain one JNE instruction");
    assert_eq!(jes.len(), 4, "fixture must contain four JE instructions");
    assert_eq!(jbs.len(), 1, "fixture must contain one JB instruction");

    let flag_pairs: Vec<_> = db
        .rel_iter::<(Address, Address, &'static str)>("flags_and_jump_pair")
        .copied()
        .collect();
    let dec_jne = flag_pairs
        .iter()
        .find(|(source, jump, cond)| decs.contains(source) && *jump == jnes[0] && *cond == "ne")
        .copied()
        .expect("DEC must remain the JNE flag source across a scheduled MOV");
    assert!(
        flag_pairs.contains(&dec_jne),
        "DEC must remain the JNE flag source across a scheduled MOV: {flag_pairs:#x?}"
    );
    let inc_je = flag_pairs
        .iter()
        .find(|(source, jump, cond)| incs.contains(source) && jes.contains(jump) && *cond == "e")
        .copied()
        .expect("INC must be the adjacent JE flag source");
    assert!(
        flag_pairs.contains(&inc_je),
        "INC must be the adjacent JE flag source: {flag_pairs:#x?}"
    );
    let dec_je_pairs: Vec<_> = flag_pairs
        .iter()
        .filter(|(source, jump, cond)| decs.contains(source) && jes.contains(jump) && *cond == "e")
        .copied()
        .collect();
    assert_eq!(
        dec_je_pairs.len(),
        2,
        "each DEC in the switch chain must feed its JE: {flag_pairs:#x?}"
    );
    let carry_dec = decs
        .iter()
        .copied()
        .filter(|address| *address < jbs[0])
        .max()
        .expect("missing DEC before JB");
    assert!(
        !flag_pairs
            .iter()
            .any(|(source, jump, _)| *source == carry_dec && *jump == jbs[0]),
        "DEC preserves CF and must not be treated as the source of JB"
    );

    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    AsmPass.run(&mut db);

    let mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .cloned()
        .collect();
    assert!(
        mach.iter().any(|(address, inst)| {
            *address == dec_jne.0
                && matches!(
                    inst,
                    MachInst::Mop(Operation::Oaddlimm(-1), args, Mreg::AX)
                        if args.as_ref() == &[Mreg::AX]
                )
        }),
        "64-bit DEC value update was not lowered: {mach:#x?}"
    );
    assert!(
        mach.iter().any(|(address, inst)| {
            *address == incs[0]
                && matches!(
                    inst,
                    MachInst::Mop(Operation::Oaddimm(1), args, Mreg::AX)
                        if args.as_ref() == &[Mreg::AX]
                )
        }),
        "32-bit INC value update was not lowered: {mach:#x?}"
    );
    assert!(
        mach.iter().any(|(address, inst)| {
            *address == jnes[0]
                && matches!(
                    inst,
                    MachInst::Mcond(Condition::Ccomplimm(Comparison::Cne, 0), args, _)
                        if args.as_ref() == &[Mreg::AX]
                )
        }),
        "DEC/JNE did not emit a 64-bit result condition at JNE: {mach:#x?}"
    );
    assert!(
        mach.iter().any(|(address, inst)| {
            *address == inc_je.1
                && matches!(
                    inst,
                    MachInst::Mcond(Condition::Ccompimm(Comparison::Ceq, 0), args, _)
                        if args.as_ref() == &[Mreg::AX]
                )
        }),
        "INC/JE did not emit a 32-bit result condition at JE: {mach:#x?}"
    );
    for (_, jump, _) in &dec_je_pairs {
        assert!(
            mach.iter().any(|(address, inst)| {
                address == jump
                    && matches!(
                        inst,
                        MachInst::Mcond(Condition::Ccompimm(Comparison::Ceq, 0), args, _)
                            if args.as_ref() == &[Mreg::CX]
                    )
            }),
            "switch-chain DEC/JE did not emit a result condition at {jump:#x}: {mach:#x?}"
        );
    }
    assert!(
        !mach
            .iter()
            .any(|(address, inst)| *address == jbs[0] && matches!(inst, MachInst::Mcond(..))),
        "DEC/JB must not fabricate a carry-derived result comparison: {mach:#x?}"
    );
}

#[test]
fn inc_dec_results_feed_equality_jccs_without_inventing_carry() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("inc-dec-pipeline".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_inc_dec_results(&object))
        .expect("failed to spawn INC/DEC test thread")
        .join()
        .expect("INC/DEC test thread panicked");
}
