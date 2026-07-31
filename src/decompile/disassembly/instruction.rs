use crate::decompile::disassembly::operand::*;
use crate::decompile::elevator::DecompileDB;
use crate::mreg::Mreg;
use capstone::arch::x86::X86OperandType;
use capstone::prelude::*;
use object::{Object, ObjectSection, SectionFlags, SectionKind};

const SHF_EXECINSTR: u64 = 0x4;

// Check whether an operand type is the RSP register.
fn is_rsp_reg(cs: &Capstone, op_type: &X86OperandType) -> bool {
    matches!(op_type, X86OperandType::Reg(reg_id) if {
        let name = cs.reg_name(*reg_id).unwrap_or_default().to_ascii_uppercase();
        name == "RSP"
    })
}

fn is_stack_pointer_family_reg(cs: &Capstone, op_type: &X86OperandType) -> bool {
    matches!(op_type, X86OperandType::Reg(reg_id) if {
        let name = cs.reg_name(*reg_id).unwrap_or_default().to_ascii_uppercase();
        matches!(name.as_str(), "RSP" | "SP")
    })
}

pub fn is_executable_section(section: &object::Section) -> bool {
    match section.flags() {
        SectionFlags::Elf { sh_flags } => sh_flags & SHF_EXECINSTR != 0,
        _ => section.kind() == SectionKind::Text,
    }
}

// Decoded instruction metadata retained for block detection and CFG construction.
#[derive(Debug, Clone)]
pub struct DecodedInsn {
    pub address: u64,
    pub size: usize,
    pub mnemonic: &'static str,
    pub op_str: &'static str,
}

// Disassemble executable sections into the instruction/operand/register relations, dispatching on target arch.
pub fn disassemble_sections(db: &mut DecompileDB, obj: &object::File) -> Vec<DecodedInsn> {
    match db.abi().arch {
        crate::abi::Arch::Aarch64 => {
            crate::decompile::disassembly::aarch64::disassemble_sections(db, obj)
        }
        _ => disassemble_x86_sections(db, obj),
    }
}

fn disassemble_x86_sections(db: &mut DecompileDB, obj: &object::File) -> Vec<DecodedInsn> {
    let cs = {
        let mode = if db.abi().is_64bit() {
            arch::x86::ArchMode::Mode64
        } else {
            arch::x86::ArchMode::Mode32
        };
        Capstone::new()
            .x86()
            .mode(mode)
            .syntax(arch::x86::ArchSyntax::Intel)
            .detail(true)
            .build()
            .expect("Failed to create capstone engine")
    };

    let mut decoded: Vec<DecodedInsn> = Vec::new();

    let mut op_registers: Vec<(&'static str, &'static str)> = Vec::new();
    let mut op_immediates: Vec<(&'static str, i64, usize)> = Vec::new();
    let mut op_indirects: Vec<(
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )> = Vec::new();
    let mut instructions: Vec<(
        u64,
        usize,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        usize,
        usize,
    )> = Vec::new();
    let mut instruction_address_sizes: Vec<(u64, u8)> = Vec::new();
    let mut stack_defs: Vec<(u64, &'static str, i64)> = Vec::new();
    let mut stack_uses: Vec<(u64, &'static str, i64)> = Vec::new();
    // Exact decoded memory-write operands, including non-RSP/RBP bases.  The
    // operand ID preserves the full base/index/displacement/width row in
    // op_indirect so later stack-provenance analysis need not guess writes
    // from mnemonic position.
    let mut decoded_memory_writes: Vec<(u64, &'static str)> = Vec::new();
    let mut reg_defs: Vec<(u64, Mreg)> = Vec::new();
    let mut reg_uses: Vec<(u64, Mreg)> = Vec::new();
    let mut adjusts_stack: Vec<(u64, &'static str, i64)> = Vec::new();
    let mut stack_base_moves: Vec<(u64, &'static str, &'static str)> = Vec::new();

    for section in obj.sections() {
        if !is_executable_section(&section) {
            continue;
        }
        let data = match section.data() {
            Ok(d) => d,
            Err(_) => continue,
        };
        let base_addr = section.address();

        let insns = cs.disasm_all(data, base_addr).unwrap_or_else(|e| {
            panic!("Disassembly failed at section {:?}: {}", section.name(), e)
        });

        for insn in insns.as_ref() {
            let addr = insn.address();
            let size = insn.len();

            // Strip BND/NOTRACK prefixes (treated as the underlying instruction)
            let mnem_str = insn.mnemonic().unwrap_or("").to_ascii_uppercase();
            let mnem_str = if let Some(rest) = mnem_str.strip_prefix("BND ") {
                rest.to_string()
            } else if let Some(rest) = mnem_str.strip_prefix("NOTRACK ") {
                rest.to_string()
            } else {
                mnem_str
            };
            let mnemonic: &'static str = Box::leak(mnem_str.into_boxed_str());

            let prefix: &'static str = {
                let detail = cs.insn_detail(insn).ok();
                let p = detail.as_ref().and_then(|d| {
                    let arch = d.arch_detail();
                    let x86 = arch.x86().unwrap();
                    let pfx = x86.prefix();
                    if pfx[0] != 0 {
                        match pfx[0] as u32 {
                            0xF3 => Some("REP"),
                            0xF2 => Some("REPNE"),
                            0xF0 => Some("LOCK"),
                            _ => None,
                        }
                    } else {
                        None
                    }
                });
                p.unwrap_or("")
            };

            let detail = cs
                .insn_detail(insn)
                .expect("insn_detail failed; was detail mode enabled?");
            let arch_detail = detail.arch_detail();
            let x86_detail = arch_detail.x86().expect("Not x86");
            instruction_address_sizes.push((addr, x86_detail.addr_size()));
            let ops = x86_detail.operands();

            let mut op_ids: [&'static str; 4] = [NO_OP; 4];
            let num_operands = ops.len();

            let is_nop = mnemonic == "NOP" || mnemonic == "VZEROUPPER" || mnemonic == "VZEROALL";

            let is_xor_self = if mnemonic == "XOR" && num_operands == 2 {
                let ops_vec: Vec<_> = x86_detail.operands().collect();
                match (&ops_vec[0].op_type, &ops_vec[1].op_type) {
                    (X86OperandType::Reg(r1), X86OperandType::Reg(r2)) => r1 == r2,
                    _ => false,
                }
            } else {
                false
            };

            for (i, op) in ops.enumerate() {
                if i >= 4 {
                    break;
                }
                match op.op_type {
                    X86OperandType::Reg(reg_id) => {
                        let id = alloc_op_id();
                        let name = capstone_reg_name(&cs, reg_id);
                        op_registers.push((id, name));
                        op_ids[i] = id;

                        let mreg = Mreg::x86(name);
                        if mreg != Mreg::Unknown && !is_nop {
                            if op.access.map_or(false, |a| a.is_writable()) {
                                reg_defs.push((addr, mreg));
                            }
                            if op.access.map_or(false, |a| a.is_readable()) && !is_xor_self {
                                reg_uses.push((addr, mreg));
                            }
                        }
                    }
                    X86OperandType::Imm(val) => {
                        let id = alloc_op_id();
                        op_immediates.push((id, val, 0));
                        op_ids[i] = id;
                    }
                    X86OperandType::Mem(mem) => {
                        let id = alloc_op_id();
                        let seg = if mem.segment().0 != 0 {
                            capstone_reg_name(&cs, mem.segment())
                        } else {
                            "NONE"
                        };
                        let base = if mem.base().0 != 0 {
                            capstone_reg_name(&cs, mem.base())
                        } else {
                            "NONE"
                        };
                        let index = if mem.index().0 != 0 {
                            capstone_reg_name(&cs, mem.index())
                        } else {
                            "NONE"
                        };
                        let scale = mem.scale() as i64;
                        let disp = mem.disp();
                        // Capstone operand byte width; authoritative access size (B5).
                        op_indirects.push((id, seg, base, index, scale, disp, op.size as usize));
                        op_ids[i] = id;

                        if !is_nop && op.access.map_or(false, |a| a.is_writable()) {
                            decoded_memory_writes.push((addr, id));
                        }

                        // Memory operand base/index registers count as reg_use
                        if !is_nop {
                            if base != "NONE" && base != "RIP" {
                                let mreg = Mreg::x86(base);
                                if mreg != Mreg::Unknown {
                                    reg_uses.push((addr, mreg));
                                }
                            }
                            if index != "NONE" {
                                let mreg = Mreg::x86(index);
                                if mreg != Mreg::Unknown {
                                    reg_uses.push((addr, mreg));
                                }
                            }
                        }

                        // An explicit segment contributes an unmodelled base to
                        // the effective address.  Do not publish FS/GS memory
                        // as ordinary scalar stack storage; AsmPass records a
                        // structured unsupported-address diagnostic instead.
                        if seg == "NONE"
                            && (base == "RBP" || base == "RSP")
                            && index == "NONE"
                            && mnemonic != "LEA"
                        {
                            let is_write = op.access.map_or(false, |a| a.is_writable());
                            let is_read = op.access.map_or(false, |a| a.is_readable());
                            // ddisasm: any store to stack is a def (not just MOV)
                            if is_write {
                                stack_defs.push((addr, base, disp));
                            }
                            if is_read {
                                stack_uses.push((addr, base, disp));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Implicit stack accesses for PUSH/POP
            match mnemonic {
                "PUSH" => {
                    stack_defs.push((addr, "RSP", 0));
                }
                "POP" => {
                    stack_uses.push((addr, "RSP", 0));
                }
                _ => {}
            }

            // Stack pointer modification tracking
            match mnemonic {
                "PUSH" => {
                    // In long mode PUSH is an eight-byte stack transfer unless
                    // the operand-size override selects the two-byte form.
                    let width = if x86_detail.prefix()[2] == 0x66 {
                        2
                    } else {
                        8
                    };
                    adjusts_stack.push((addr, "RSP", -width));
                }
                "POP" => {
                    let ops_vec: Vec<_> = x86_detail.operands().collect();
                    // POP into any spelling of the stack-pointer family is not
                    // affine: the popped value itself overwrites part or all of
                    // the post-increment pointer. Leave it out of adjusts_stack
                    // so the provenance analysis kills the coordinate.
                    let overwrites_sp = ops_vec
                        .first()
                        .is_some_and(|op| is_stack_pointer_family_reg(&cs, &op.op_type));
                    if !overwrites_sp {
                        let width = if x86_detail.prefix()[2] == 0x66 {
                            2
                        } else {
                            8
                        };
                        adjusts_stack.push((addr, "RSP", width));
                    }
                }
                "SUB" => {
                    if num_operands >= 2 {
                        let ops_vec: Vec<_> = x86_detail.operands().collect();
                        if let (Some(dst_op), Some(src_op)) = (ops_vec.get(0), ops_vec.get(1)) {
                            if is_rsp_reg(&cs, &dst_op.op_type) {
                                if let X86OperandType::Imm(imm_val) = src_op.op_type {
                                    adjusts_stack.push((addr, "RSP", -imm_val));
                                }
                            }
                        }
                    }
                }
                "ADD" => {
                    if num_operands >= 2 {
                        let ops_vec: Vec<_> = x86_detail.operands().collect();
                        if let (Some(dst_op), Some(src_op)) = (ops_vec.get(0), ops_vec.get(1)) {
                            if is_rsp_reg(&cs, &dst_op.op_type) {
                                if let X86OperandType::Imm(imm_val) = src_op.op_type {
                                    adjusts_stack.push((addr, "RSP", imm_val));
                                }
                            }
                        }
                    }
                }
                "LEA" => {
                    if num_operands >= 2 {
                        let ops_vec: Vec<_> = x86_detail.operands().collect();
                        if let (Some(dst_op), Some(src_op)) = (ops_vec.get(0), ops_vec.get(1)) {
                            if is_rsp_reg(&cs, &dst_op.op_type) {
                                if let X86OperandType::Mem(mem) = src_op.op_type {
                                    let base_name = if mem.base().0 != 0 {
                                        cs.reg_name(mem.base())
                                            .unwrap_or_default()
                                            .to_ascii_uppercase()
                                    } else {
                                        String::new()
                                    };
                                    let index_id = mem.index().0;
                                    if base_name == "RSP" && index_id == 0 {
                                        adjusts_stack.push((addr, "RSP", mem.disp()));
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            // Stack base register moves (mov rbp, rsp / mov rsp, rbp)
            if mnemonic == "MOV" && num_operands >= 2 {
                let ops_vec: Vec<_> = x86_detail.operands().collect();
                if let (Some(dst_op), Some(src_op)) = (ops_vec.get(0), ops_vec.get(1)) {
                    let dst_reg = match dst_op.op_type {
                        X86OperandType::Reg(reg_id) => {
                            let name = cs.reg_name(reg_id).unwrap_or_default().to_ascii_uppercase();
                            if name == "RSP" || name == "RBP" {
                                Some(name)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    let src_reg = match src_op.op_type {
                        X86OperandType::Reg(reg_id) => {
                            let name = cs.reg_name(reg_id).unwrap_or_default().to_ascii_uppercase();
                            if name == "RSP" || name == "RBP" {
                                Some(name)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };
                    if let (Some(dst), Some(src)) = (dst_reg, src_reg) {
                        if dst != src {
                            let src_static: &'static str = Box::leak(src.into_boxed_str());
                            let dst_static: &'static str = Box::leak(dst.into_boxed_str());
                            stack_base_moves.push((addr, src_static, dst_static));
                        }
                    }
                }
            }

            // Implicit register def/use from capstone's regs_write/regs_read
            if !is_nop {
                for &reg_id in detail.regs_write() {
                    let name = cs.reg_name(reg_id).unwrap_or_default().to_ascii_uppercase();
                    let mreg = Mreg::x86(name.as_str());
                    if mreg != Mreg::Unknown {
                        reg_defs.push((addr, mreg));
                    }
                }
                for &reg_id in detail.regs_read() {
                    let name = cs.reg_name(reg_id).unwrap_or_default().to_ascii_uppercase();
                    let mreg = Mreg::x86(name.as_str());
                    if mreg != Mreg::Unknown {
                        reg_uses.push((addr, mreg));
                    }
                }
            }

            // Swap op1/op2 to match ddisasm's operand ordering convention
            if num_operands >= 2 {
                op_ids.swap(0, 1);
            }

            instructions.push((
                addr, size, prefix, mnemonic, op_ids[0], op_ids[1], op_ids[2], op_ids[3], 0, 0,
            ));

            let op_str: &'static str = Box::leak(
                insn.op_str()
                    .unwrap_or("")
                    .to_ascii_uppercase()
                    .into_boxed_str(),
            );

            decoded.push(DecodedInsn {
                address: addr,
                size,
                mnemonic,
                op_str,
            });
        }
    }

    decoded.sort_by_key(|d| d.address);
    instructions.sort_by_key(|t| t.0);

    // Populate DB relations
    db.rel_set(
        "unrefinedinstruction",
        instructions.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let mut nexts: Vec<(u64, u64)> = Vec::with_capacity(decoded.len());
    for w in decoded.windows(2) {
        if w[0].address + w[0].size as u64 == w[1].address {
            nexts.push((w[0].address, w[1].address));
        }
    }
    db.rel_set(
        "next",
        nexts.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "op_register",
        op_registers.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "op_immediate",
        op_immediates
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "op_indirect",
        op_indirects.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "instruction_address_size",
        instruction_address_sizes
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "stack_def",
        stack_defs.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "stack_use",
        stack_uses.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "decoded_memory_write_operand",
        decoded_memory_writes
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    // Immutable decoder-owned register effects.  Later IR passes intentionally
    // augment the public reg_use/reg_def relations; retaining this snapshot
    // prevents a repeated AsmPass run from mistaking its own synthesized facts
    // for new Capstone evidence.
    db.rel_set(
        "decoded_reg_def",
        reg_defs.iter().copied().collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "decoded_reg_use",
        reg_uses.iter().copied().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "reg_def",
        reg_defs.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "reg_use",
        reg_uses.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    db.rel_set(
        "adjusts_stack",
        adjusts_stack
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "stack_base_move",
        stack_base_moves
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    decoded
}
