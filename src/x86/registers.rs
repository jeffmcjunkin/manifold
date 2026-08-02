use crate::x86::asm::{Freg, Ireg};
use crate::x86::types::Operand;

pub(crate) fn is_x86_64_gp_register_name(name: &str) -> bool {
    matches!(
        name,
        "RAX"
            | "RBX"
            | "RCX"
            | "RDX"
            | "RSI"
            | "RDI"
            | "RBP"
            | "RSP"
            | "R8"
            | "R9"
            | "R10"
            | "R11"
            | "R12"
            | "R13"
            | "R14"
            | "R15"
    )
}

pub(crate) fn x86_gp_value_width(name: &str) -> Option<usize> {
    if is_x86_64_gp_register_name(name) {
        return Some(8);
    }
    matches!(
        name,
        "EAX"
            | "EBX"
            | "ECX"
            | "EDX"
            | "ESI"
            | "EDI"
            | "EBP"
            | "ESP"
            | "R8D"
            | "R9D"
            | "R10D"
            | "R11D"
            | "R12D"
            | "R13D"
            | "R14D"
            | "R15D"
    )
    .then_some(4)
}

/// Width of a decoded x86-64 general-purpose register operand in bytes.
///
/// This deliberately preserves subregister spelling.  Callers which collapse
/// registers to their parent `Mreg` must reject AH/BH/CH/DH when they cannot
/// also represent the legacy high-byte offset.
pub(crate) fn x86_gp_operand_width(name: &str) -> Option<usize> {
    if let Some(width) = x86_gp_value_width(name) {
        return Some(width);
    }
    if matches!(
        name,
        "AX" | "BX"
            | "CX"
            | "DX"
            | "SI"
            | "DI"
            | "BP"
            | "SP"
            | "R8W"
            | "R9W"
            | "R10W"
            | "R11W"
            | "R12W"
            | "R13W"
            | "R14W"
            | "R15W"
    ) {
        return Some(2);
    }
    matches!(
        name,
        "AL" | "BL"
            | "CL"
            | "DL"
            | "AH"
            | "BH"
            | "CH"
            | "DH"
            | "SIL"
            | "DIL"
            | "BPL"
            | "SPL"
            | "R8B"
            | "R9B"
            | "R10B"
            | "R11B"
            | "R12B"
            | "R13B"
            | "R14B"
            | "R15B"
    )
    .then_some(1)
}

impl From<Operand> for Ireg {
    fn from(op: Operand) -> Self {
        match op {
            Operand::Register(reg) => Ireg::from(reg.to_string()),
            Operand::Memory(base, offset) => {
                if offset == "0" {
                    return Ireg::from(*base);
                } else {
                    panic!("Offset is not zero: {}", offset);
                }
            }
            _ => panic!("Not a register {:#?}", op),
        }
    }
}

impl From<Operand> for Freg {
    fn from(op: Operand) -> Self {
        match op {
            Operand::Register(reg) => match &reg[..] {
                "%xmm0" | "%ymm0" => Freg::XMM0,
                "%xmm1" | "%ymm1" => Freg::XMM1,
                "%xmm2" | "%ymm2" => Freg::XMM2,
                "%xmm3" | "%ymm3" => Freg::XMM3,
                "%xmm4" | "%ymm4" => Freg::XMM4,
                "%xmm5" | "%ymm5" => Freg::XMM5,
                "%xmm6" | "%ymm6" => Freg::XMM6,
                "%xmm7" | "%ymm7" => Freg::XMM7,
                "%xmm8" | "%ymm8" => Freg::XMM8,
                "%xmm9" | "%ymm9" => Freg::XMM9,
                "%xmm10" | "%ymm10" => Freg::XMM10,
                "%xmm11" | "%ymm11" => Freg::XMM11,
                "%xmm12" | "%ymm12" => Freg::XMM12,
                "%xmm13" | "%ymm13" => Freg::XMM13,
                "%xmm14" | "%ymm14" => Freg::XMM14,
                "%xmm15" | "%ymm15" => Freg::XMM15,
                _ => panic!("Unknown register: {}", reg),
            },
            _ => panic!("Not a register"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::x86_gp_operand_width;

    #[test]
    fn decoded_gp_operand_width_preserves_subregister_spelling() {
        for (name, width) in [
            ("AL", 1),
            ("AH", 1),
            ("R15B", 1),
            ("AX", 2),
            ("R15W", 2),
            ("EAX", 4),
            ("R15D", 4),
            ("RAX", 8),
            ("R15", 8),
        ] {
            assert_eq!(x86_gp_operand_width(name), Some(width), "{name}");
        }
        for name in ["RIP", "XMM0", "NONE", "rax"] {
            assert_eq!(x86_gp_operand_width(name), None, "{name}");
        }
    }
}
