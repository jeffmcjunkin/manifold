
// AArch64 decoding into the same DB relations as x86, with two interface facts: operand order is capstone's (destination first), and def/use comes from opcode shape since Arm64Operand has no access field.

use capstone::arch::arm64::{Arm64OperandType, Arm64Shift};
use capstone::prelude::*;
use object::{Object, ObjectSection};

use crate::aarch64::mach::A64Mreg;
use crate::decompile::disassembly::instruction::{is_executable_section, DecodedInsn};
use crate::decompile::disassembly::operand::*;
use crate::decompile::elevator::DecompileDB;
use crate::mreg::Mreg;

// Which explicit operands an instruction writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefShape {
    // Operand 0 is the destination. The common case.
    Op0,
    // Operands 0 and 1 are both destinations (load-pair).
    Op0Op1,
    // No register destination: stores, compares (they write only flags), branches.
    None,
}

// Classify by mnemonic, most-specific first, or the broad prefix checks below would shadow several tests.
fn def_shape(mnem: &str) -> DefShape {
    // Load-pair writes two registers; store-pair writes none.
    if matches!(mnem, "LDP" | "LDNP" | "LDPSW") {
        return DefShape::Op0Op1;
    }
    // Compares and test-and-set-flags write only NZCV.
    if matches!(
        mnem,
        "CMP" | "CMN" | "TST" | "CCMP" | "CCMN" | "FCMP" | "FCMPE" | "FCCMP" | "FCCMPE"
    ) {
        return DefShape::None;
    }
    // Branches. BL/BLR additionally write X30, added as an implicit def by the caller.
    if matches!(
        mnem,
        "B" | "BR" | "BL" | "BLR" | "RET" | "CBZ" | "CBNZ" | "TBZ" | "TBNZ" | "BRK" | "SVC"
    ) || mnem.starts_with("B.")
    {
        return DefShape::None;
    }
    // Every store form (STR/STRB/STRH/STUR/STP/STLR/...); the LDR-vs-STR split is the biggest def/use distinction in AArch64 code.
    if mnem.starts_with("ST") {
        return DefShape::None;
    }
    DefShape::Op0
}

// Registers whose writes are not modelled as value definitions.
fn is_discardable_def(r: A64Mreg) -> bool {
    // Writing the zero register discards the result; `SUBS XZR, Xn, Xm` is CMP.
    r == A64Mreg::ZR
}

pub fn disassemble_sections(db: &mut DecompileDB, obj: &object::File) -> Vec<DecodedInsn> {
    let cs = Capstone::new()
        .arm64()
        .mode(arch::arm64::ArchMode::Arm)
        .detail(true)
        .build()
        .expect("Failed to create capstone arm64 engine");

    let mut decoded: Vec<DecodedInsn> = Vec::new();

    let mut op_registers: Vec<(&'static str, &'static str)> = Vec::new();
    let mut op_immediates: Vec<(&'static str, i64, usize)> = Vec::new();
    let mut op_indirects: Vec<(&'static str, &'static str, &'static str, &'static str, i64, i64, usize)> = Vec::new();
    let mut op_indirect_extends: Vec<(&'static str, &'static str)> = Vec::new();
    let mut instructions: Vec<(u64, usize, &'static str, &'static str,
                                &'static str, &'static str, &'static str, &'static str,
                                usize, usize)> = Vec::new();
    let mut stack_defs: Vec<(u64, &'static str, i64)> = Vec::new();
    let mut stack_uses: Vec<(u64, &'static str, i64)> = Vec::new();
    let mut reg_defs: Vec<(u64, Mreg)> = Vec::new();
    let mut reg_uses: Vec<(u64, Mreg)> = Vec::new();
    let mut adjusts_stack: Vec<(u64, &'static str, i64)> = Vec::new();
    let mut stack_base_moves: Vec<(u64, &'static str, &'static str)> = Vec::new();
    let mut stack_base_move_offsets: Vec<(u64, &'static str, &'static str, i64)> = Vec::new();

    for section in obj.sections() {
        if !is_executable_section(&section) {
            continue;
        }
        let data = match section.data() {
            Ok(d) => d,
            Err(_) => continue,
        };
        let base_addr = section.address();

        let insns = cs
            .disasm_all(data, base_addr)
            .unwrap_or_else(|e| panic!("Disassembly failed at section {:?}: {}", section.name(), e));

        for insn in insns.as_ref() {
            let addr = insn.address();
            let size = insn.len();

            let mnemonic: &'static str = Box::leak(
                insn.mnemonic().unwrap_or("").to_ascii_uppercase().into_boxed_str(),
            );

            let detail = cs
                .insn_detail(insn)
                .expect("insn_detail failed; was detail mode enabled?");
            let arch_detail = detail.arch_detail();
            let a64 = arch_detail.arm64().expect("Not arm64");
            let ops: Vec<_> = a64.operands().collect();
            let writeback = a64.writeback();

            let shape = def_shape(mnemonic);
            // The transferred register of a load/store is operand 0; its view width is the fallback for the access size.
            let first_reg = ops.first().and_then(|o| match o.op_type {
                Arm64OperandType::Reg(r) => Some(capstone_reg_name(&cs, r)),
                _ => None,
            });
            let mem_size = access_size(mnemonic, first_reg);
            // NOP and the vector-zeroing idioms define nothing worth tracking.
            let is_nop = matches!(mnemonic, "NOP" | "HINT");

            let mut op_ids: [&'static str; 4] = [NO_OP; 4];
            let mem_pos = ops.iter().position(|o| matches!(o.op_type, Arm64OperandType::Mem(_)));

            for (i, op) in ops.iter().enumerate() {
                if i >= 4 {
                    break;
                }
                match &op.op_type {
                    Arm64OperandType::Reg(reg_id) => {
                        let id = alloc_op_id();
                        let name = capstone_reg_name(&cs, *reg_id);
                        op_registers.push((id, name));
                        op_ids[i] = id;

                        if is_nop {
                            continue;
                        }
                        let a64reg = A64Mreg::from(name);
                        if a64reg == A64Mreg::Unknown {
                            continue;
                        }
                        let mreg = Mreg::a64(a64reg);
                        let is_def = match shape {
                            DefShape::Op0 => i == 0,
                            DefShape::Op0Op1 => i <= 1,
                            DefShape::None => false,
                        };
                        if is_def && !is_discardable_def(a64reg) {
                            reg_defs.push((addr, mreg));
                        }
                        // A destination is not also a source, and the zero register reads as the constant 0 rather than a value.
                        if !is_def && a64reg != A64Mreg::ZR {
                            reg_uses.push((addr, mreg));
                        }
                    }
                    Arm64OperandType::Imm(val) => {
                        let id = alloc_op_id();
                        op_immediates.push((id, *val, 0));
                        op_ids[i] = id;
                    }
                    Arm64OperandType::Mem(mem) => {
                        let id = alloc_op_id();
                        let base = reg_or_none(&cs, mem.base());
                        let index = reg_or_none(&cs, mem.index());
                        // The scale of an indexed load lives on the operand's shifter ([Xn, Xm, LSL #3]), not inside the memory operand.
                        let scale = match op.shift {
                            Arm64Shift::Lsl(amount) => 1i64 << amount,
                            _ => 1,
                        };
                        let disp = mem.disp() as i64;
                        op_indirects.push((id, "NONE", base, index, scale, disp, mem_size));
                        op_ids[i] = id;

                        // [Xn, Wm, SXTW #2] extends a 32-bit index before scaling; no Addressing variant carries an extension, so record which kind for the frontend's explicit cast.
                        if let Some(ext) = extender_name(op.ext) {
                            op_indirect_extends.push((id, ext));
                        }

                        if is_nop {
                            continue;
                        }
                        // Address arithmetic reads the base and index registers.
                        for r in [base, index] {
                            if r == "NONE" {
                                continue;
                            }
                            let a64reg = A64Mreg::from(r);
                            if a64reg != A64Mreg::Unknown && a64reg != A64Mreg::ZR {
                                reg_uses.push((addr, Mreg::a64(a64reg)));
                            }
                        }
                        // Pre/post-index addressing writes the base back.
                        if writeback && base != "NONE" {
                            let a64reg = A64Mreg::from(base);
                            if a64reg != A64Mreg::Unknown {
                                reg_defs.push((addr, Mreg::a64(a64reg)));
                            }
                        }

                        // Frame-relative access. X29 is the AArch64 frame pointer.
                        if (base == "SP" || base == "X29") && index == "NONE" {
                            if shape == DefShape::None && mnemonic.starts_with("ST") {
                                stack_defs.push((addr, base, disp));
                            } else {
                                stack_uses.push((addr, base, disp));
                            }
                        }
                    }
                    _ => {}
                }
            }

            if !is_nop {
                // BL/BLR write the link register; capstone reports this only inconsistently and the call-return-address model depends on it.
                if matches!(mnemonic, "BL" | "BLR") {
                    reg_defs.push((addr, Mreg::a64(A64Mreg::X30)));
                }
                // RET reads it (RET with no operand is `RET X30`).
                if mnemonic == "RET" && ops.is_empty() {
                    reg_uses.push((addr, Mreg::a64(A64Mreg::X30)));
                }
            }

            // Stack pointer movement.
            if let Some(delta) = sp_adjustment(mnemonic, &ops, mem_pos, writeback, &cs) {
                adjusts_stack.push((addr, "SP", delta));
            }

            // Frame-pointer establish/teardown matched by SHAPE (<reg> := <reg> + <imm>), since clang spells it several ways and the offset is load-bearing (X29 may sit N bytes above SP).
            if matches!(mnemonic, "MOV" | "ADD") && ops.len() >= 2 {
                if let (Arm64OperandType::Reg(d), Arm64OperandType::Reg(s)) =
                    (&ops[0].op_type, &ops[1].op_type)
                {
                    let offset = match ops.get(2).map(|o| &o.op_type) {
                        None => Some(0),
                        Some(Arm64OperandType::Imm(v)) => Some(*v),
                        // A register or shifted third operand is real arithmetic, not a frame establish.
                        Some(_) => None,
                    };
                    let dst = A64Mreg::from(capstone_reg_name(&cs, *d)).name();
                    let src = A64Mreg::from(capstone_reg_name(&cs, *s)).name();
                    let frame_reg = |r: &str| r == "SP" || r == "X29";
                    if let Some(offset) = offset {
                        if frame_reg(dst) && frame_reg(src) && dst != src {
                            stack_base_move_offsets.push((addr, src, dst, offset));
                            // The offsetless relation means exactly dst := src, which every consumer relies on, so it is emitted only for the genuinely zero-offset case.
                            if offset == 0 {
                                stack_base_moves.push((addr, src, dst));
                            }
                        }
                    }
                }
            }

            instructions.push((
                addr, size, "", mnemonic,
                op_ids[0], op_ids[1], op_ids[2], op_ids[3],
                0, 0,
            ));

            let op_str: &'static str = Box::leak(
                insn.op_str().unwrap_or("").to_ascii_uppercase().into_boxed_str(),
            );

            decoded.push(DecodedInsn { address: addr, size, mnemonic, op_str });
        }
    }

    decoded.sort_by_key(|d| d.address);
    instructions.sort_by_key(|t| t.0);

    db.rel_set("unrefinedinstruction", instructions.into_iter().collect::<ascent::boxcar::Vec<_>>());

    // AArch64 instructions are all 4 bytes, but a section can end mid-stream, so adjacency is checked rather than assumed.
    let mut nexts: Vec<(u64, u64)> = Vec::with_capacity(decoded.len());
    for w in decoded.windows(2) {
        if w[0].address + w[0].size as u64 == w[1].address {
            nexts.push((w[0].address, w[1].address));
        }
    }
    db.rel_set("next", nexts.into_iter().collect::<ascent::boxcar::Vec<_>>());

    db.rel_set("op_register", op_registers.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("op_immediate", op_immediates.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("op_indirect", op_indirects.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("op_indirect_extend", op_indirect_extends.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("stack_def", stack_defs.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("stack_use", stack_uses.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("reg_def", reg_defs.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("reg_use", reg_uses.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("adjusts_stack", adjusts_stack.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("stack_base_move", stack_base_moves.into_iter().collect::<ascent::boxcar::Vec<_>>());
    db.rel_set("stack_base_move_offset", stack_base_move_offsets.into_iter().collect::<ascent::boxcar::Vec<_>>());

    decoded
}

// Canonical 64-bit name of a register operand, or "NONE"; goes through A64Mreg so a raw-string test like base == "X29" cannot silently never match.
fn reg_or_none(cs: &Capstone, reg: RegId) -> &'static str {
    if reg.0 == 0 {
        return "NONE";
    }
    A64Mreg::from(capstone_reg_name(cs, reg)).name()
}

fn extender_name(ext: capstone::arch::arm64::Arm64Extender) -> Option<&'static str> {
    use capstone::arch::arm64::Arm64Extender::*;
    match ext {
        Sxtw => Some("SXTW"),
        Sxtx => Some("SXTX"),
        Sxtb => Some("SXTB"),
        Sxth => Some("SXTH"),
        Uxtw => Some("UXTW"),
        Uxtx => Some("UXTX"),
        Uxtb => Some("UXTB"),
        Uxth => Some("UXTH"),
        Invalid => None,
    }
}

// How much this instruction moves SP: explicit SUB/ADD SP, plus pre-index and post-index writeback, which are how AArch64 spells push and pop.
fn sp_adjustment(
    mnem: &str,
    ops: &[capstone::arch::arm64::Arm64Operand],
    mem_pos: Option<usize>,
    writeback: bool,
    cs: &Capstone,
) -> Option<i64> {
    if matches!(mnem, "SUB" | "ADD") && ops.len() >= 3 {
        let dst = as_reg_name(cs, &ops[0])?;
        let src = as_reg_name(cs, &ops[1])?;
        if dst == "SP" && src == "SP" {
            if let Arm64OperandType::Imm(v) = ops[2].op_type {
                return Some(if mnem == "SUB" { -v } else { v });
            }
        }
        return None;
    }

    if !writeback {
        return None;
    }
    let mem_pos = mem_pos?;
    let Arm64OperandType::Mem(mem) = &ops[mem_pos].op_type else {
        return None;
    };
    if reg_or_none(cs, mem.base()) != "SP" {
        return None;
    }
    // Post-index carries its delta in a trailing immediate operand; pre-index carries it in the memory operand's displacement.
    match ops.get(mem_pos + 1).map(|o| &o.op_type) {
        Some(Arm64OperandType::Imm(v)) => Some(*v),
        _ => Some(mem.disp() as i64),
    }
}

// Canonical name of an explicit register operand; see reg_or_none.
fn as_reg_name(cs: &Capstone, op: &capstone::arch::arm64::Arm64Operand) -> Option<&'static str> {
    match op.op_type {
        Arm64OperandType::Reg(r) => Some(A64Mreg::from(capstone_reg_name(cs, r)).name()),
        _ => None,
    }
}

// Byte width of a load/store access: capstone carries no operand size, but the mnemonic suffix determines it, falling back to the transferred register's width.
fn access_size(mnem: &str, first_reg: Option<&str>) -> usize {
    // Suffix forms pin the width regardless of register view.
    if mnem.ends_with('B') && mnem.starts_with(['L', 'S']) {
        return 1;
    }
    if mnem.ends_with('H') && mnem.starts_with(['L', 'S']) {
        return 2;
    }
    // LDRSW/LDURSW load 4 bytes and sign-extend to 8.
    if mnem.ends_with("SW") {
        return 4;
    }
    // Otherwise the register view decides: Wn/Sn are 4 bytes, Xn/Dn are 8, Qn is 16.
    match first_reg.map(|r| r.as_bytes().first().copied().unwrap_or(b'X')) {
        Some(b'W') | Some(b'S') => 4,
        Some(b'Q') => 16,
        Some(b'H') => 2,
        Some(b'B') => 1,
        _ => 8,
    }
}
