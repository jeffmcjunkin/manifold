// AArch64 to Mach: the AArch64 counterpart of asm_pass, mutually exclusive with it at runtime, decomposing AArch64 into the SHARED Operation/Addressing/Condition alphabet rather than mirroring CompCert's aarch64 backend.

use crate::abi::Arch;
use crate::aarch64::mach::A64Mreg;
use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::mreg::Mreg;
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use crate::{declare_io_from, run_pass};
use ascent::aggregators;
use ascent::ascent_par;
use either::Either;
use std::sync::Arc;

// A register operand's Mreg; capstone's alias spellings (FP/LR/ZR) and width views (W9/D3) are already normalized by A64Mreg::from.
fn mreg_of(name: &str) -> Mreg {
    Mreg::a64(A64Mreg::from(name))
}

// True for a 64-bit register view; AArch64 encodes width in the register name, so this selects between the int and long halves of the shared alphabet.
fn is_64bit_reg(name: &str) -> bool {
    let u = name.trim().to_ascii_uppercase();
    matches!(u.as_str(), "SP" | "XZR" | "FP" | "LR") || u.starts_with('X') || u.starts_with('D')
}

// True for a floating-point register view (Vn/Dn/Sn/Qn/Hn/Bn).
fn is_float_reg(name: &str) -> bool {
    A64Mreg::from(name).is_vec()
}

// The zero register reads as the constant 0 rather than a value, so a rule wanting a real operand must reject it.
fn is_zero_reg(name: &str) -> bool {
    A64Mreg::from(name) == A64Mreg::ZR
}

// Comparison and signedness from a condition suffix, in the shared shorthand cfg.rs::branch_condition maps B.<cond> onto.
fn cond_parts(cond: &str) -> Option<(Comparison, bool)> {
    Some(match cond {
        "e" => (Comparison::Ceq, true),
        "ne" => (Comparison::Cne, true),
        "l" => (Comparison::Clt, true),
        "le" => (Comparison::Cle, true),
        "g" => (Comparison::Cgt, true),
        "ge" => (Comparison::Cge, true),
        "b" => (Comparison::Clt, false),
        "be" => (Comparison::Cle, false),
        "a" => (Comparison::Cgt, false),
        "ae" => (Comparison::Cge, false),
        _ => return None,
    })
}

// Register-register comparison condition.
fn cond_reg(cond: &str, wide: bool) -> Option<Condition> {
    let (cmp, signed) = cond_parts(cond)?;
    Some(match (wide, signed) {
        (true, true) => Condition::Ccompl(cmp),
        (true, false) => Condition::Ccomplu(cmp),
        (false, true) => Condition::Ccomp(cmp),
        (false, false) => Condition::Ccompu(cmp),
    })
}

// Register-immediate comparison condition.
fn cond_imm(cond: &str, wide: bool, imm: i64) -> Option<Condition> {
    let (cmp, signed) = cond_parts(cond)?;
    Some(match (wide, signed) {
        (true, true) => Condition::Ccomplimm(cmp, imm),
        (true, false) => Condition::Ccompluimm(cmp, imm),
        (false, true) => Condition::Ccompimm(cmp, imm),
        (false, false) => Condition::Ccompuimm(cmp, imm),
    })
}

// Three-operand register ALU mnemonic -> shared Operation, width-selected.
fn alu_rrr_op(mnem: &str, wide: bool) -> Option<Operation> {
    Some(match (mnem, wide) {
        ("ADD", true) => Operation::Oaddl,
        ("ADD", false) => Operation::Oadd,
        ("SUB", true) => Operation::Osubl,
        ("SUB", false) => Operation::Osub,
        ("MUL", true) => Operation::Omull,
        ("MUL", false) => Operation::Omul,
        ("SDIV", true) => Operation::Odivl,
        ("SDIV", false) => Operation::Odiv,
        ("UDIV", true) => Operation::Odivlu,
        ("UDIV", false) => Operation::Odivu,
        ("AND", true) => Operation::Oandl,
        ("AND", false) => Operation::Oand,
        ("ORR", true) => Operation::Oorl,
        ("ORR", false) => Operation::Oor,
        ("EOR", true) => Operation::Oxorl,
        ("EOR", false) => Operation::Oxor,
        ("LSL", true) => Operation::Oshll,
        ("LSL", false) => Operation::Oshl,
        ("LSR", true) => Operation::Oshrlu,
        ("LSR", false) => Operation::Oshru,
        ("ASR", true) => Operation::Oshrl,
        ("ASR", false) => Operation::Oshr,
        _ => return None,
    })
}

// Register-immediate ALU mnemonic -> shared Operation; SUB has no immediate form there, so it folds to Oaddlimm with a negated constant.
fn alu_rri_op(mnem: &str, wide: bool, imm: i64) -> Option<Operation> {
    Some(match (mnem, wide) {
        ("ADD", true) => Operation::Oaddlimm(imm),
        ("ADD", false) => Operation::Oaddimm(imm),
        ("SUB", true) => Operation::Oaddlimm(-imm),
        ("SUB", false) => Operation::Oaddimm(-imm),
        ("AND", true) => Operation::Oandlimm(imm),
        ("AND", false) => Operation::Oandimm(imm),
        ("ORR", true) => Operation::Oorlimm(imm),
        ("ORR", false) => Operation::Oorimm(imm),
        ("EOR", true) => Operation::Oxorlimm(imm),
        ("EOR", false) => Operation::Oxorimm(imm),
        ("LSL", true) => Operation::Oshllimm(imm),
        ("LSL", false) => Operation::Oshlimm(imm),
        ("LSR", true) => Operation::Oshrluimm(imm),
        ("LSR", false) => Operation::Oshruimm(imm),
        ("ASR", true) => Operation::Oshrlimm(imm),
        ("ASR", false) => Operation::Oshrimm(imm),
        ("MUL", true) => Operation::Omullimm(imm),
        ("MUL", false) => Operation::Omulimm(imm),
        _ => return None,
    })
}

// Two-operand register unary mnemonic -> shared Operation.
fn alu_rr_op(mnem: &str, wide: bool) -> Option<Operation> {
    Some(match (mnem, wide) {
        ("MOV", _) => Operation::Omove,
        ("NEG", true) => Operation::Onegl,
        ("NEG", false) => Operation::Oneg,
        ("MVN", true) => Operation::Onotl,
        ("MVN", false) => Operation::Onot,
        // Width conversions; SXTW is the 32->64 sign extension appearing wherever an int index feeds a 64-bit address computation.
        ("SXTW", _) => Operation::Ocast32signed,
        ("SXTB", _) => Operation::Ocast8signed,
        ("SXTH", _) => Operation::Ocast16signed,
        ("UXTB", _) => Operation::Ocast8unsigned,
        ("UXTH", _) => Operation::Ocast16unsigned,
        ("FMOV", _) => Operation::Omove,
        _ => return None,
    })
}

// Memory access width and signedness -> MemoryChunk from the mnemonic suffix and register view; no capstone operand size is needed or available.
fn chunk_of(mnem: &str, reg: &str) -> Option<MemoryChunk> {
    let float = is_float_reg(reg);
    let wide = is_64bit_reg(reg);
    Some(match mnem {
        "LDRB" | "STRB" | "LDURB" | "STURB" => MemoryChunk::MInt8Unsigned,
        "LDRSB" | "LDURSB" => MemoryChunk::MInt8Signed,
        "LDRH" | "STRH" | "LDURH" | "STURH" => MemoryChunk::MInt16Unsigned,
        "LDRSH" | "LDURSH" => MemoryChunk::MInt16Signed,
        "LDRSW" | "LDURSW" => MemoryChunk::MInt32,
        "LDR" | "STR" | "LDUR" | "STUR" | "LDP" | "STP" => {
            if float {
                if wide { MemoryChunk::MFloat64 } else { MemoryChunk::MFloat32 }
            } else if wide {
                MemoryChunk::MInt64
            } else {
                MemoryChunk::MInt32
            }
        }
        _ => return None,
    })
}

fn is_load_mnem(m: &str) -> bool {
    m.starts_with("LD")
}

fn is_store_mnem(m: &str) -> bool {
    m.starts_with("ST")
}

// Addressing mode for a decoded memory operand; every AArch64 mode lands in the SHARED Addressing set, with [SP, #i] becoming Ainstack so stack analysis sees it as it does x86.
fn addressing_of(
    base: &str,
    index: &str,
    scale: i64,
    disp: i64,
) -> Option<(Addressing, Vec<Mreg>)> {
    if base == "NONE" {
        return None;
    }
    if index == "NONE" {
        if base == "SP" {
            return Some((Addressing::Ainstack(disp), vec![]));
        }
        return Some((Addressing::Aindexed(disp), vec![mreg_of(base)]));
    }
    let args = vec![mreg_of(base), mreg_of(index)];
    if scale <= 1 {
        Some((Addressing::Aindexed2(disp), args))
    } else {
        Some((Addressing::Aindexed2scaled(scale, disp), args))
    }
}

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct Aarch64AsmPassProgram;

    // ---- inputs from the disassembly layer -------------------------------------
    relation unrefinedinstruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation op_register(Symbol, &'static str);
    relation op_immediate(Symbol, i64, usize);
    relation op_indirect(Symbol, &'static str, &'static str, &'static str, i64, i64, usize);
    relation code_in_block(Address, Address);
    relation code_in_refined_block(Address, Address);
    relation block_in_function(Node, Address);
    relation func_span(Symbol, Address, Address);
    relation symbol_table(Address, usize, &'static str, &'static str, &'static str, usize, &'static str, usize, &'static str);
    relation symbols(Address, Symbol, Symbol);
    relation next(Address, Address);
    relation ddisasm_function_entry(Address);
    relation flags_and_jump_pair(Address, Address, &'static str);
    relation adjusts_stack(Address, Symbol, i64);
    relation trim_instruction(Address);
    relation func_entry(Symbol, Address);

    // ---- outputs consumed downstream -------------------------------------------
    relation instruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation instr_in_function(Node, Address);
    relation is_external_function(Address);
    relation plt_entry(Address, Symbol);
    relation func_stacksz(Address, Address, Symbol, u64);
    relation jmp_target_in_own_func(Address);
    relation mcond_at_jcc(Address);
    relation node_unlowered_side_effect(Address);
    relation mach_imm_stack_init(Address, i64, i64, Typ);
    relation mach_inst(Address, MachInst);

    // SHARED-DERIVATION: restated from asm_pass because that pass is gated off for AArch64. Keep in step.

    instruction(addr, size, prefix, mnemonic, o1, o2, o3, o4, x, y) <--
        unrefinedinstruction(addr, size, prefix, mnemonic, o1, o2, o3, o4, x, y),
        code_in_refined_block(addr, _),
        !trim_instruction(addr);

    instr_in_function(addr, func) <--
        block_in_function(blockaddr, func),
        code_in_block(addr, blockaddr);

    instr_in_function(next_addr, func) <--
        instr_in_function(addr, func),
        next(addr, next_addr),
        func_span(_, func, end_addr),
        if *next_addr > *addr,
        if *next_addr < *end_addr,
        !ddisasm_function_entry(*next_addr),
        if *next_addr != *func;

    plt_entry(addr, name) <--
        symbol_table(addr, _, _, _, _, _, section_name, _, name),
        if *section_name == ".plt" || *section_name == ".plt.got";

    is_external_function(addr) <-- plt_entry(addr, _);

    is_external_function(addr) <--
        symbol_table(addr, _, sym_type, _, _, _, section_name, _, _),
        if *sym_type == "FUNC",
        if *section_name == ".init" || *section_name == ".fini";

    // Frame size: AArch64 allocates with SUB SP, #N and/or a pre-index store, both recorded as negative adjusts_stack deltas, so summing the negatives suffices.

    #[local] relation frame_alloc(Symbol, Address, i64);
    frame_alloc(func_name, addr, -*delta) <--
        adjusts_stack(addr, "SP", delta),
        if *delta < 0,
        instr_in_function(addr, func),
        func_span(func_name, func, _);

    func_stacksz(start_addr, end_addr, *func_name, stack_size_u64) <--
        func_span(func_name, start_addr, end_addr),
        agg stack_size = aggregators::sum(sz) in frame_alloc(func_name, _, sz),
        let stack_size_u64 = stack_size.max(0) as u64;

    // Control flow

    mach_inst(addr, MachInst::Mreturn) <--
        instruction(addr, _, _, "RET", _, _, _, _, _, _);

    // A B whose target is inside the same function is structured control flow; one that leaves it is a tail call, the same split asm_pass makes for x86 JMP.
    jmp_target_in_own_func(addr) <--
        instruction(addr, _, _, "B", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        instr_in_function(addr, func),
        instr_in_function(*target_addr as u64, func);

    mach_inst(addr, MachInst::Mgoto(dst)) <--
        instruction(addr, _, _, "B", dst, _, _, _, _, _),
        jmp_target_in_own_func(addr);

    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(name)))) <--
        instruction(addr, _, _, "B", dst, _, _, _, _, _),
        !jmp_target_in_own_func(addr),
        op_immediate(dst, target_addr, _),
        func_entry(name, *target_addr as u64);

    // BL is a direct call; BLR calls through a register.
    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Left(name)))) <--
        instruction(addr, _, _, "BL", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        func_entry(name, *target_addr as u64);

    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Right(*target_addr)))) <--
        instruction(addr, _, _, "BL", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        !func_entry(_, *target_addr as u64);

    mach_inst(addr, MachInst::Mcall(Either::Left(mreg_of(reg)))) <--
        instruction(addr, _, _, "BLR", dst, _, _, _, _, _),
        op_register(dst, reg);

    // Conditional branches: AArch64 sets NZCV separately like x86, so the pairing comes from flags_and_jump_pair and the operands are read back from the flag setter.

    mcond_at_jcc(br_addr) <-- flags_and_jump_pair(_, br_addr, _);

    // CMP Xn, Xm  /  B.<cond>
    mach_inst(br_addr, MachInst::Mcond(condition, Arc::new(vec![mreg_of(r1), mreg_of(r2)]), lbl)) <--
        flags_and_jump_pair(cmp_addr, br_addr, cond),
        instruction(cmp_addr, _, _, cmp_mnem, o1, o2, _, _, _, _),
        if *cmp_mnem == "CMP" || *cmp_mnem == "SUBS",
        op_register(o1, r1),
        op_register(o2, r2),
        instruction(br_addr, _, _, _, lbl, _, _, _, _, _),
        if let Some(condition) = cond_reg(cond, is_64bit_reg(r1));

    // CMP Xn, #imm  /  B.<cond>
    mach_inst(br_addr, MachInst::Mcond(condition, Arc::new(vec![mreg_of(r1)]), lbl)) <--
        flags_and_jump_pair(cmp_addr, br_addr, cond),
        instruction(cmp_addr, _, _, cmp_mnem, o1, o2, _, _, _, _),
        if *cmp_mnem == "CMP" || *cmp_mnem == "SUBS",
        op_register(o1, r1),
        op_immediate(o2, imm, _),
        instruction(br_addr, _, _, _, lbl, _, _, _, _, _),
        if let Some(condition) = cond_imm(cond, is_64bit_reg(r1), *imm);

    // CBZ/CBNZ fuse the compare-against-zero into the branch, so they need no flag-setter pairing; the target is operand slot 1.
    mach_inst(addr, MachInst::Mcond(cond, Arc::new(vec![mreg_of(reg)]), lbl)) <--
        instruction(addr, _, _, mnem, o1, lbl, _, _, _, _),
        if *mnem == "CBZ" || *mnem == "CBNZ",
        op_register(o1, reg),
        let cmp = if *mnem == "CBZ" { Comparison::Ceq } else { Comparison::Cne },
        let cond = if is_64bit_reg(reg) {
            Condition::Ccomplimm(cmp, 0)
        } else {
            Condition::Ccompimm(cmp, 0)
        };

    mcond_at_jcc(addr) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if *mnem == "CBZ" || *mnem == "CBNZ";

    // Data processing

    // Three-register ALU: ADD/SUB/MUL/SDIV/UDIV/AND/ORR/EOR/LSL/LSR/ASR.
    mach_inst(addr, MachInst::Mop(op, Arc::new(vec![mreg_of(rn), mreg_of(rm)]), mreg_of(rd))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        op_register(o0, rd),
        op_register(o1, rn),
        op_register(o2, rm),
        if !is_zero_reg(rd),
        if let Some(op) = alu_rrr_op(mnem, is_64bit_reg(rd));

    // Register/immediate ALU.
    mach_inst(addr, MachInst::Mop(op, Arc::new(vec![mreg_of(rn)]), mreg_of(rd))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        op_register(o0, rd),
        op_register(o1, rn),
        op_immediate(o2, imm, _),
        if !is_zero_reg(rd),
        if !is_zero_reg(rn),
        if let Some(op) = alu_rri_op(mnem, is_64bit_reg(rd), *imm);

    // Two-register unary: MOV/NEG/MVN and the sign/zero extensions.
    mach_inst(addr, MachInst::Mop(op, Arc::new(vec![mreg_of(rn)]), mreg_of(rd))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *o2 == "0",
        op_register(o0, rd),
        op_register(o1, rn),
        if !is_zero_reg(rd),
        if !is_zero_reg(rn),
        if let Some(op) = alu_rr_op(mnem, is_64bit_reg(rd));

    // `MOV Xd, XZR` materializes zero rather than copying a register.
    mach_inst(addr, MachInst::Mop(op, Arc::new(vec![]), mreg_of(rd))) <--
        instruction(addr, _, _, "MOV", o0, o1, o2, _, _, _),
        if *o2 == "0",
        op_register(o0, rd),
        op_register(o1, rn),
        if is_zero_reg(rn),
        if !is_zero_reg(rd),
        let op = if is_64bit_reg(rd) {
            Operation::Olongconst(0)
        } else {
            Operation::Ointconst(0)
        };

    // MOV Xd, #imm / MOVZ.
    mach_inst(addr, MachInst::Mop(op, Arc::new(vec![]), mreg_of(rd))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *mnem == "MOV" || *mnem == "MOVZ",
        if *o2 == "0",
        op_register(o0, rd),
        op_immediate(o1, imm, _),
        if !is_zero_reg(rd),
        let op = if is_64bit_reg(rd) {
            Operation::Olongconst(*imm)
        } else {
            Operation::Ointconst(*imm)
        };

    // ADD Xd, SP, #imm is frame-address arithmetic, spelled in the shared alphabet as a load-effective-address over Ainstack.
    mach_inst(addr, MachInst::Mop(Operation::Oleal(Addressing::Ainstack(*imm)), Arc::new(vec![]), mreg_of(rd))) <--
        instruction(addr, _, _, "ADD", o0, o1, o2, _, _, _),
        op_register(o0, rd),
        op_register(o1, base),
        if *base == "SP",
        op_immediate(o2, imm, _),
        if !is_zero_reg(rd);

    // Memory

    mach_inst(addr, MachInst::Mload(chunk, addressing, Arc::new(args), mreg_of(rt))) <--
        instruction(addr, _, _, mnem, o0, o1, _, _, _, _),
        if is_load_mnem(mnem),
        op_register(o0, rt),
        op_indirect(o1, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp);

    mach_inst(addr, MachInst::Mstore(chunk, addressing, Arc::new(args), mreg_of(rt))) <--
        instruction(addr, _, _, mnem, o0, o1, _, _, _, _),
        if is_store_mnem(mnem),
        op_register(o0, rt),
        op_indirect(o1, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp);

    // A load/store pair transfers two registers to consecutive slots, the second one access-width above the first, so it cannot be one wide access.
    mach_inst(addr, MachInst::Mload(chunk, addressing, Arc::new(args), mreg_of(rt))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *mnem == "LDP" || *mnem == "LDNP",
        op_register(o0, rt),
        op_indirect(o2, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp),
        let _ = o1;

    mach_inst(addr, MachInst::Mstore(chunk, addressing, Arc::new(args), mreg_of(rt))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *mnem == "STP" || *mnem == "STNP",
        op_register(o0, rt),
        op_indirect(o2, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp),
        let _ = o1;

    // Second register of a pair, one slot up.
    mach_inst(addr, MachInst::Mload(chunk, addressing, Arc::new(args), mreg_of(rt2))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *mnem == "LDP" || *mnem == "LDNP",
        op_register(o0, rt),
        op_register(o1, rt2),
        op_indirect(o2, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt2),
        let width = chunk_bytes(chunk),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp + width),
        let _ = rt;

    mach_inst(addr, MachInst::Mstore(chunk, addressing, Arc::new(args), mreg_of(rt2))) <--
        instruction(addr, _, _, mnem, o0, o1, o2, _, _, _),
        if *mnem == "STP" || *mnem == "STNP",
        op_register(o0, rt),
        op_register(o1, rt2),
        op_indirect(o2, _, base, index, scale, disp, _),
        if let Some(chunk) = chunk_of(mnem, rt2),
        let width = chunk_bytes(chunk),
        if let Some((addressing, args)) = addressing_of(base, index, *scale, *disp + width),
        let _ = rt;

    // Unmodelled: SIMD has no scalar lowering, so flagging it keeps the loss visible downstream instead of silently dropping the computation.
    node_unlowered_side_effect(addr) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if is_simd_mnem(mnem);
}

// Byte width of a memory chunk, for stepping to the second register of a pair.
fn chunk_bytes(chunk: MemoryChunk) -> i64 {
    match chunk {
        MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned => 1,
        MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => 2,
        MemoryChunk::MInt32 | MemoryChunk::MFloat32 => 4,
        _ => 8,
    }
}

// Genuinely packed SIMD mnemonics, excluding the scalar FP forms (FMOV/FADD/FCMP on Sn/Dn) and the vector MOV/LDR/STR block movers, which are ordinary memory traffic.
fn is_simd_mnem(m: &str) -> bool {
    matches!(
        m,
        "ADDV" | "DUP" | "UMOV" | "XTN" | "XTN2" | "USHLL" | "USHLL2" | "SHRN" | "SHRN2"
            | "LD2" | "LD3" | "LD4" | "ST2" | "ST3" | "ST4" | "EXT" | "TBL" | "TBX"
            | "CMHI" | "CMGT" | "CMEQ" | "CMGE" | "CMHS" | "BIF" | "BSL" | "BIT"
            | "SMAXV" | "UMAXV" | "SMINV" | "UMINV" | "UADDW" | "SADDW" | "USHR"
            | "MOVI" | "MVNI" | "SMAX" | "SMIN" | "UMAX" | "UMIN"
    )
}

pub struct Aarch64AsmPass;

impl IRPass for Aarch64AsmPass {
    fn name(&self) -> &'static str {
        "aarch64_asm"
    }

    fn run(&self, db: &mut DecompileDB) {
        // Mutually exclusive with asm_pass; see the module comment for why running both would corrupt rather than duplicate.
        if db.abi().arch != Arch::Aarch64 {
            return;
        }
        run_pass!(db, Aarch64AsmPassProgram);
    }

    declare_io_from!(Aarch64AsmPassProgram);
}
