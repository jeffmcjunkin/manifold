use crate::aarch64::mach::A64Mreg;
use crate::mreg::Mreg;

// Noreturn classification, the single source of truth shared by abi_pass, linear_pass, cshminor_pass and structuring_pass so their lists cannot drift.

/// Functions that NEVER return regardless of arguments; the symbol-only noreturn set consumed verbatim by abi_pass.
pub const ALWAYS_NORETURN_FUNCS: &[&str] = &[
    // libc primitives
    "exit", "_exit", "_Exit",
    "abort",
    "__assert_fail", "__assert_perror_fail",
    "__stack_chk_fail",
    "pthread_exit",
    "longjmp", "_longjmp", "siglongjmp",
    "__cxa_throw", "__cxa_rethrow",
    "quick_exit", "thrd_exit",
    // always-noreturn wrappers (declared _Noreturn at their source)
    "xalloc_die",
    "__chk_fail",
    "__fortify_fail",
    "assert_failed",
];

/// True for an always-noreturn function (the symbol-only set above).
pub fn is_always_noreturn(symbol: &str) -> bool {
    ALWAYS_NORETURN_FUNCS.contains(&symbol)
}

/// error-family functions that exit(status) only when status != 0, so they are noreturn exactly when arg 0 is a nonzero constant.
pub fn is_status_noreturn_callee(symbol: &str) -> bool {
    matches!(
        symbol,
        "error" | "error_at_line" | "verror" | "verror_at_line"
    )
}

/// THE shared status-aware noreturn predicate: always-noreturn callees, or error-family callees whose arg 0 is a known nonzero constant (exact, not heuristic).
pub fn call_is_noreturn(callee: &str, status_arg0_nonzero_const: impl FnOnce() -> bool) -> bool {
    is_always_noreturn(callee)
        || (is_status_noreturn_callee(callee) && status_arg0_nonzero_const())
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFormat {
    Coff,
    Elf,
    Pe,
    MachO,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    X86_64,
    X86_32,
    Aarch64,
}


#[derive(Debug, Clone)]
pub struct AbiConfig {
    pub format: BinaryFormat,
    pub arch: Arch,

    pub pointer_size: usize,

    pub int_arg_regs: Vec<Mreg>,

    pub float_arg_regs: Vec<Mreg>,

    pub ret_int_reg: Mreg,

    #[allow(dead_code)]
    pub ret_float_reg: Mreg,

    pub caller_saved: Vec<Mreg>,

    pub callee_saved: Vec<Mreg>,

    #[allow(dead_code)]
    pub stack_alignment: usize,

    #[allow(dead_code)]
    pub red_zone_size: usize,
}

impl AbiConfig {
    pub fn sysv_x86_64() -> Self {
        Self {
            format: BinaryFormat::Elf,
            arch: Arch::X86_64,
            pointer_size: 8,
            int_arg_regs: vec![
                Mreg::DI, Mreg::SI, Mreg::DX, Mreg::CX, Mreg::R8, Mreg::R9,
            ],
            float_arg_regs: vec![
                Mreg::X0, Mreg::X1, Mreg::X2, Mreg::X3,
                Mreg::X4, Mreg::X5, Mreg::X6, Mreg::X7,
            ],
            ret_int_reg: Mreg::AX,
            ret_float_reg: Mreg::X0,
            caller_saved: vec![
                Mreg::AX, Mreg::CX, Mreg::DX,
                Mreg::SI, Mreg::DI,
                Mreg::R8, Mreg::R9, Mreg::R10, Mreg::R11,
                Mreg::X0, Mreg::X1, Mreg::X2, Mreg::X3,
                Mreg::X4, Mreg::X5, Mreg::X6, Mreg::X7,
                Mreg::X8, Mreg::X9, Mreg::X10, Mreg::X11,
                Mreg::X12, Mreg::X13, Mreg::X14, Mreg::X15,
            ],
            callee_saved: vec![
                Mreg::BX, Mreg::BP,
                Mreg::R12, Mreg::R13, Mreg::R14, Mreg::R15,
            ],
            stack_alignment: 16,
            red_zone_size: 128,
        }
    }

    pub fn win64() -> Self {
        Self {
            format: BinaryFormat::Pe,
            arch: Arch::X86_64,
            pointer_size: 8,
            int_arg_regs: vec![Mreg::CX, Mreg::DX, Mreg::R8, Mreg::R9],
            float_arg_regs: vec![Mreg::X0, Mreg::X1, Mreg::X2, Mreg::X3],
            ret_int_reg: Mreg::AX,
            ret_float_reg: Mreg::X0,
            caller_saved: vec![
                Mreg::AX, Mreg::CX, Mreg::DX,
                Mreg::R8, Mreg::R9, Mreg::R10, Mreg::R11,
                Mreg::X0, Mreg::X1, Mreg::X2, Mreg::X3,
                Mreg::X4, Mreg::X5,
            ],
            callee_saved: vec![
                Mreg::BX, Mreg::BP,
                Mreg::SI, Mreg::DI,
                Mreg::R12, Mreg::R13, Mreg::R14, Mreg::R15,
                Mreg::X6, Mreg::X7, Mreg::X8, Mreg::X9,
                Mreg::X10, Mreg::X11, Mreg::X12, Mreg::X13,
                Mreg::X14, Mreg::X15,
            ],
            stack_alignment: 16,
            red_zone_size: 0,
        }
    }

    pub fn cdecl_x86_32() -> Self {
        Self {
            format: BinaryFormat::Elf,
            arch: Arch::X86_32,
            pointer_size: 4,
            int_arg_regs: vec![],
            float_arg_regs: vec![],
            ret_int_reg: Mreg::AX,
            ret_float_reg: Mreg::FP0,
            caller_saved: vec![
                Mreg::AX, Mreg::CX, Mreg::DX,
            ],
            callee_saved: vec![
                Mreg::BX, Mreg::BP, Mreg::SI, Mreg::DI,
            ],
            stack_alignment: 4,
            red_zone_size: 0,
        }
    }

    /// AAPCS64 (ARM's standard AArch64 calling convention), as used by Linux ELF.
    pub fn aapcs64() -> Self {
        let gp = |n: u8| Mreg::A64(A64Mreg::gp(n));
        let v = |n: u8| Mreg::A64(A64Mreg::vec(n));
        Self {
            format: BinaryFormat::Elf,
            arch: Arch::Aarch64,
            pointer_size: 8,
            // NGRN/NSRN are independent counters, as in SysV x86-64 and unlike Win64.
            int_arg_regs: (0..8).map(gp).collect(),
            float_arg_regs: (0..8).map(v).collect(),
            ret_int_reg: gp(0),
            ret_float_reg: v(0),
            // X0-X17 are temporaries (X16/X17 are IP0/IP1); X18 is the platform register, a Linux temporary but reserved on Darwin/Windows, so it is in neither list.
            caller_saved: (0..18)
                .map(gp)
                .chain((0..8).map(v))
                .chain((16..32).map(v))
                .collect(),
            // V8-V15 are callee-saved only in their low 64 bits; modelling whole registers is the conservative direction for a saved-reg test.
            callee_saved: (19..=30)
                .map(gp)
                .chain((8..16).map(v))
                .collect(),
            stack_alignment: 16,
            red_zone_size: 0,
        }
    }

    pub fn is_64bit(&self) -> bool {
        matches!(self.arch, Arch::X86_64 | Arch::Aarch64)
    }

    /// Windows x64 assigns integer and FP registers from the same four ordinal slots; SysV keeps independent GP and XMM sequences.
    pub fn uses_shared_arg_slots(&self) -> bool {
        self.format == BinaryFormat::Pe && self.arch == Arch::X86_64
    }

    pub fn first_stack_arg_position(&self) -> usize {
        if self.uses_shared_arg_slots() { 4 } else { self.int_arg_regs.len() }
    }

    /// Offset from RSP immediately before CALL to the first stack argument.
    pub fn outgoing_stack_arg_base(&self) -> i64 {
        if self.uses_shared_arg_slots() { 32 } else { 0 }
    }

    /// Bytes the call instruction pushed below the incoming stack arguments: x86's CALL pushes the return address, AArch64's BL leaves it in X30 and pushes nothing.
    pub fn return_addr_stack_size(&self) -> i64 {
        match self.arch {
            Arch::Aarch64 => 0,
            _ => self.pointer_size as i64,
        }
    }

    /// Offset from the callee's entry SP to its first stack argument: past whatever the call pushed, plus Windows' shadow space.
    pub fn incoming_sp_stack_arg_base(&self) -> i64 {
        self.return_addr_stack_size() + self.outgoing_stack_arg_base()
    }

    /// Offset from the established frame pointer to the first stack argument (x86: incoming-SP base plus saved RBP; AArch64: 16 for the canonical stp x29,x30 frame record).
    pub fn incoming_bp_stack_arg_base(&self) -> i64 {
        match self.arch {
            Arch::Aarch64 => 2 * self.pointer_size as i64,
            _ => self.incoming_sp_stack_arg_base() + self.pointer_size as i64,
        }
    }
}

pub fn detect_abi(obj: &object::File) -> Result<AbiConfig, String> {
    use object::Object;

    let format = match obj.format() {
        object::BinaryFormat::Coff => BinaryFormat::Coff,
        object::BinaryFormat::Elf => BinaryFormat::Elf,
        object::BinaryFormat::Pe => BinaryFormat::Pe,
        object::BinaryFormat::MachO => BinaryFormat::MachO,
        _ => BinaryFormat::Unknown,
    };

    let arch = match obj.architecture() {
        object::Architecture::X86_64 => Arch::X86_64,
        object::Architecture::I386 => Arch::X86_32,
        object::Architecture::Aarch64 => Arch::Aarch64,
        other => return Err(format!("unsupported architecture {other:?}; manifold currently supports x86 and aarch64 binaries")),
    };

    if format == BinaryFormat::Pe && arch != Arch::X86_64 {
        return Err("unsupported PE target: only native AMD64 PE32+ images are supported".to_string());
    }

    // AArch64 is ELF-only for now: the PE .pdata RUNTIME_FUNCTION layout is x64-specific and there is no Mach-O metadata frontend.
    if arch == Arch::Aarch64 && format != BinaryFormat::Elf {
        return Err(format!("unsupported aarch64 target: only ELF images are supported, got {format:?}"));
    }

    // The Clight backend hardcodes 8-byte pointers, so a 32-bit input would silently decompile against the wrong pointer width; reject it rather than emit wrong C.
    if arch == Arch::X86_32 {
        return Err("unsupported 32-bit x86 target: manifold currently models 64-bit pointers only".to_string());
    }

    let mut config = match (format, arch) {
        (BinaryFormat::Pe | BinaryFormat::Coff, Arch::X86_64) => AbiConfig::win64(),
        (_, Arch::X86_64) => AbiConfig::sysv_x86_64(),
        (_, Arch::X86_32) => AbiConfig::cdecl_x86_32(),
        (_, Arch::Aarch64) => AbiConfig::aapcs64(),
    };

    config.format = format;
    Ok(config)
}
