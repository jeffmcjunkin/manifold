
// Arch-neutral control-transfer classification: the four facts block/CFG building need, answered per-arch so no shared consumer matches one architecture's mnemonics.

use crate::abi::Arch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchKind {
    // Not a control transfer.
    None,
    // Falls through on one edge, jumps on the other.
    CondJump,
    // Unconditional jump to a known target.
    UncondJump,
    // Unconditional jump through a register or memory (jump tables, tail calls).
    IndirectJump,
    // Call that returns to the following instruction.
    Call,
    // Call through a register or memory.
    IndirectCall,
    // Function return.
    Return,
    // Terminates the block with no successor: traps and undefined instructions.
    Trap,
    // Ends a basic block, but the CFG models it as ordinary fallthrough (x86 SYSCALL resumes at the next instruction).
    OpaqueTransfer,
}

impl BranchKind {
    pub fn is_control_transfer(&self) -> bool {
        !matches!(self, BranchKind::None)
    }

    /// True when execution may continue at the next instruction.
    pub fn falls_through(&self) -> bool {
        matches!(
            self,
            BranchKind::None
                | BranchKind::CondJump
                | BranchKind::Call
                | BranchKind::IndirectCall
                | BranchKind::OpaqueTransfer
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchInfo {
    pub kind: BranchKind,
    /// Operand slot holding the PC-relative target: x86 always slot 0, AArch64 slot 1 for CBZ and slot 2 for TBZ.
    pub target_op: Option<usize>,
}

const NOT_A_BRANCH: BranchInfo = BranchInfo { kind: BranchKind::None, target_op: None };

pub fn classify(arch: Arch, mnem: &str) -> BranchInfo {
    match arch {
        Arch::Aarch64 => classify_aarch64(mnem),
        _ => classify_x86(mnem),
    }
}

/// Classify one decoded instruction using operand semantics that cannot be
/// represented by the mnemonic alone.  Windows AMD64 fast-fail (`int 29h`)
/// never returns, while other software interrupts such as checked-build
/// `int 2ch` deliberately retain their existing fallthrough behavior.
pub fn classify_decoded(
    arch: Arch,
    mnem: &str,
    interrupt_vector: Option<u8>,
) -> BranchInfo {
    if matches!(arch, Arch::X86_64 | Arch::X86_32)
        && mnem == "INT"
        && interrupt_vector == Some(0x29)
    {
        return info(BranchKind::Trap, None);
    }
    classify(arch, mnem)
}

fn info(kind: BranchKind, target_op: Option<usize>) -> BranchInfo {
    BranchInfo { kind, target_op }
}

fn classify_x86(mnem: &str) -> BranchInfo {
    match mnem {
        "JMP" | "JMPQ" => return info(BranchKind::UncondJump, Some(0)),
        "CALL" => return info(BranchKind::Call, Some(0)),
        "RET" => return info(BranchKind::Return, None),
        "INT3" | "HLT" | "UD2" => return info(BranchKind::Trap, None),
        // Jcc including the count-register forms; LOOP/LOOPcc decrement RCX and branch, so both edges live.
        "JE" | "JNE" | "JL" | "JLE" | "JG" | "JGE" | "JB" | "JBE" | "JA" | "JAE" | "JP"
        | "JNP" | "JO" | "JNO" | "JS" | "JNS" | "JCXZ" | "JECXZ" | "JRCXZ" | "LOOP"
        | "LOOPE" | "LOOPNE" => return info(BranchKind::CondJump, Some(0)),
        // Enters the kernel and resumes at the next instruction.
        "SYSCALL" => return info(BranchKind::OpaqueTransfer, None),
        _ => {}
    }
    // Any remaining J-prefixed mnemonic ends a block but contributed no branch edge before this refactor; keep it that way.
    if mnem.starts_with('J') {
        return info(BranchKind::OpaqueTransfer, None);
    }
    NOT_A_BRANCH
}

fn classify_aarch64(mnem: &str) -> BranchInfo {
    match mnem {
        "B" => return info(BranchKind::UncondJump, Some(0)),
        "BL" => return info(BranchKind::Call, Some(0)),
        "BR" => return info(BranchKind::IndirectJump, None),
        "BLR" => return info(BranchKind::IndirectCall, None),
        "RET" => return info(BranchKind::Return, None),
        // Compare-and-branch tests one register against zero.
        "CBZ" | "CBNZ" => return info(BranchKind::CondJump, Some(1)),
        // Test-bit-and-branch takes a bit position before the target.
        "TBZ" | "TBNZ" => return info(BranchKind::CondJump, Some(2)),
        "BRK" | "UDF" => return info(BranchKind::Trap, None),
        // Supervisor/hypervisor calls resume at the next instruction.
        "SVC" | "HVC" | "SMC" => return info(BranchKind::OpaqueTransfer, None),
        _ => {}
    }
    // The B.<cond> family, matched by prefix anchored on the dot so BFI/BIC/BFXIL/BFM/BSL/BIF cannot match.
    if let Some(rest) = mnem.strip_prefix("B.") {
        if !rest.is_empty() {
            return info(BranchKind::CondJump, Some(0));
        }
    }
    NOT_A_BRANCH
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aarch64_data_processing_b_mnemonics_are_not_branches() {
        for m in ["BFI", "BIC", "BFXIL", "BFM", "BFC", "BSL", "BIF", "BICS"] {
            assert_eq!(
                classify(Arch::Aarch64, m).kind,
                BranchKind::None,
                "{m} must not classify as a branch"
            );
        }
    }

    #[test]
    fn aarch64_branch_targets_sit_in_the_right_operand_slot() {
        assert_eq!(classify(Arch::Aarch64, "B").target_op, Some(0));
        assert_eq!(classify(Arch::Aarch64, "B.LT").target_op, Some(0));
        assert_eq!(classify(Arch::Aarch64, "CBZ").target_op, Some(1));
        assert_eq!(classify(Arch::Aarch64, "TBNZ").target_op, Some(2));
    }

    // The exact set block.rs::is_branch recognized before this module existed.
    #[test]
    fn x86_block_splitting_set_is_unchanged() {
        for m in [
            "JMP", "JE", "JNE", "JRCXZ", "JUNKNOWNCC", "RET", "CALL", "LOOP", "LOOPE",
            "LOOPNE", "INT3", "HLT", "UD2", "SYSCALL",
        ] {
            assert!(
                classify(Arch::X86_64, m).kind.is_control_transfer(),
                "{m} must remain a block terminator"
            );
        }
        for m in ["MOV", "ADD", "LEA", "XOR", "CMP", "TEST", "SYSRET", "IRET", "RETF"] {
            assert_eq!(
                classify(Arch::X86_64, m).kind,
                BranchKind::None,
                "{m} was not a block terminator before and must not become one"
            );
        }
    }

    // SYSCALL and unrecognized Jcc split blocks but produced only a fallthrough edge.
    #[test]
    fn x86_opaque_transfers_still_fall_through() {
        for m in ["SYSCALL", "JUNKNOWNCC"] {
            let k = classify(Arch::X86_64, m).kind;
            assert_eq!(k, BranchKind::OpaqueTransfer);
            assert!(k.falls_through());
            assert!(k.is_control_transfer());
        }
    }
    #[test]
    fn only_fastfail_interrupt_is_terminal() {
        assert_eq!(
            classify_decoded(Arch::X86_64, "INT", Some(0x29)).kind,
            BranchKind::Trap,
        );
        assert_eq!(
            classify_decoded(Arch::X86_64, "INT", Some(0x2c)).kind,
            BranchKind::None,
        );
        assert_eq!(
            classify_decoded(Arch::X86_64, "INT", None).kind,
            BranchKind::None,
        );
    }
}
