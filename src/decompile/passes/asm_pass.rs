// Asm to Mach: parses x86 assembly into Mach IR (reverses CompCert's x86/Asmgen.v).

use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::{declare_io_from, run_pass};

use crate::mreg::Mreg;
use crate::x86::asm::{Freg, Ireg, Preg, TestCond};
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use ascent::aggregators;
use ascent::ascent_par;
use ascent::lattice::constant_propagation::ConstPropagation;
use ascent::lattice::set::Set;
use ascent::Dual;
use either::Either;
use log::warn;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

fn asm_dom_set_with_self(strict: &Set<Address>, node: Address) -> Set<Address> {
    let mut result = strict.0.clone();
    result.insert(node);
    Set(result)
}

// The width-suffixed bswap builtin for an operand register: 32-bit for an E-prefixed or R8D..R15D form, 64-bit otherwise, since a bare __builtin_bswap is undefined at link.
fn bswap_builtin_for_reg(r: &str) -> &'static str {
    let u = r.to_ascii_uppercase();
    if u.starts_with('E') || u.ends_with('D') {
        "__builtin_bswap32"
    } else {
        "__builtin_bswap64"
    }
}

// Genuinely-packed SIMD mnemonics with no scalar lowering, whose presence fingerprints an auto-vectorized loop; excludes block moves, PXOR and all scalar FP ops so struct-copy recovery is untouched.
fn is_packed_alu_mnem(mnem: &str) -> bool {
    let m = mnem.to_ascii_uppercase();
    let m = m.strip_prefix('V').unwrap_or(&m);
    matches!(
        m,
        // packed integer add/sub (byte/word/dword/qword), with/without saturation
        "PADDB" | "PADDW" | "PADDD" | "PADDQ" | "PADDSB" | "PADDSW" | "PADDUSB" | "PADDUSW"
        | "PSUBB" | "PSUBW" | "PSUBD" | "PSUBQ" | "PSUBSB" | "PSUBSW" | "PSUBUSB" | "PSUBUSW"
        // packed integer multiply / multiply-add
        | "PMULLW" | "PMULLD" | "PMULHW" | "PMULHUW" | "PMULDQ" | "PMULUDQ" | "PMADDWD" | "PMADDUBSW"
        // packed integer min/max/abs/avg
        | "PMINSB" | "PMINSW" | "PMINSD" | "PMINUB" | "PMINUW" | "PMINUD"
        | "PMAXSB" | "PMAXSW" | "PMAXSD" | "PMAXUB" | "PMAXUW" | "PMAXUD"
        | "PABSB" | "PABSW" | "PABSD" | "PAVGB" | "PAVGW"
        // packed compare (eq/gt) -- byte/word/dword/qword
        | "PCMPEQB" | "PCMPEQW" | "PCMPEQD" | "PCMPEQQ"
        | "PCMPGTB" | "PCMPGTW" | "PCMPGTD" | "PCMPGTQ"
        // packed shuffle / unpack / pack / blend
        | "PSHUFB" | "PSHUFD" | "PSHUFW" | "PSHUFLW" | "PSHUFHW"
        | "PUNPCKLBW" | "PUNPCKLWD" | "PUNPCKLDQ" | "PUNPCKLQDQ"
        | "PUNPCKHBW" | "PUNPCKHWD" | "PUNPCKHDQ" | "PUNPCKHQDQ"
        | "PACKSSWB" | "PACKSSDW" | "PACKUSWB" | "PACKUSDW" | "PBLENDW" | "PBLENDVB"
        // movemask / packed-vector convert (note scalar forms are CVTSI2SS/CVTSS2SI etc.)
        | "PMOVMSKB" | "CVTDQ2PS" | "CVTDQ2PD" | "CVTPS2DQ" | "CVTTPS2DQ" | "CVTPD2DQ" | "CVTTPD2DQ"
        // packed float arithmetic (scalar forms are ADDSS/MULSS/...; these are the *PS/*PD lanes)
        | "ADDPS" | "ADDPD" | "SUBPS" | "SUBPD" | "MULPS" | "MULPD" | "DIVPS" | "DIVPD"
        | "MAXPS" | "MAXPD" | "MINPS" | "MINPD" | "HADDPS" | "HADDPD" | "HSUBPS" | "HSUBPD"
        // packed shifts by immediate/reg (vector lane shifts; scalar is SHL/SHR/SAR)
        | "PSLLW" | "PSLLD" | "PSLLQ" | "PSRLW" | "PSRLD" | "PSRLQ" | "PSRAW" | "PSRAD"
    )
}

fn is_addr32_gp_name(name: &str) -> bool {
    matches!(
        name,
        "EAX"
            | "EBX"
            | "ECX"
            | "EDX"
            | "ESI"
            | "EDI"
            | "EBP"
            | "R8D"
            | "R9D"
            | "R10D"
            | "R11D"
            | "R12D"
            | "R13D"
            | "R14D"
            | "R15D"
    )
}

fn is_no_address_register(name: &str) -> bool {
    name.is_empty() || name == "NONE"
}

fn is_unmodeled_segment(name: &str) -> bool {
    matches!(name, "FS" | "GS")
}

fn chunk_from_mnem(mnem: &str) -> MemoryChunk {
    let m = mnem.to_ascii_uppercase();
    if m.contains("MOVB") || m.ends_with('B') {
        MemoryChunk::MInt8Unsigned
    } else if m.contains("MOVW") || m.ends_with('W') {
        MemoryChunk::MInt16Unsigned
    } else if m.contains("MOVL") || m.ends_with('L') {
        MemoryChunk::MInt32
    } else if m.contains("MOVQ") || m.ends_with('Q') {
        MemoryChunk::MInt64
    } else if m.contains("MOVSS") {
        MemoryChunk::MFloat32
    } else if m.contains("MOVSD") {
        MemoryChunk::MAny64
    }
    // MOVD/VMOVD (not MOVDQA/MOVDQU) is a 32-bit GP<->XMM transfer, sized 32-bit so the float-bits shuffle does not over-read the source's upper half.
    else if m == "MOVD" || m == "VMOVD" {
        MemoryChunk::MInt32
    } else {
        MemoryChunk::MAny64
    }
}

// Like chunk_from_mnem but also handles the sign/zero-extend load spellings; Intel forms carry no source width, so signedness is set here and the 8-vs-16 narrowing comes from the operand width.
fn chunk_from_mnem_ext(mnem: &str) -> MemoryChunk {
    let m = mnem.to_ascii_uppercase();
    if m.contains("MOVZB") {
        MemoryChunk::MInt8Unsigned
    } else if m.contains("MOVSB") {
        MemoryChunk::MInt8Signed
    } else if m.contains("MOVZW") {
        MemoryChunk::MInt16Unsigned
    } else if m.contains("MOVSW") {
        MemoryChunk::MInt16Signed
    } else if m.contains("MOVSXD") {
        MemoryChunk::MInt32
    } else if m.contains("MOVZX") {
        MemoryChunk::MInt8Unsigned
    } else if m.contains("MOVSX") {
        MemoryChunk::MInt8Signed
    } else {
        chunk_from_mnem(mnem)
    }
}

// Narrow a mnemonic-derived chunk to the capstone operand byte width (B5); Intel syntax has no size suffix so chunk_from_mnem defaults to MAny64 and the operand width is authoritative. Preserves sign/float-ness; never widens.
fn refine_chunk_with_size(mc: MemoryChunk, size: usize) -> MemoryChunk {
    let signed = matches!(mc, MemoryChunk::MInt8Signed | MemoryChunk::MInt16Signed);
    match size {
        1 => {
            if signed {
                MemoryChunk::MInt8Signed
            } else {
                MemoryChunk::MInt8Unsigned
            }
        }
        2 => {
            if signed {
                MemoryChunk::MInt16Signed
            } else {
                MemoryChunk::MInt16Unsigned
            }
        }
        // A 4-byte access through a float register is a single (movss); MFloat64 narrows to MFloat32, never to an int chunk.
        4 => {
            if mc == MemoryChunk::MFloat32 || mc == MemoryChunk::MFloat64 {
                MemoryChunk::MFloat32
            } else {
                MemoryChunk::MInt32
            }
        }
        _ => mc,
    }
}

/// Maps SETcc TestCond (CF/ZF unsigned-style) to the float Condition it materializes (A->Cgt, AE->Cge, B->Clt, BE->Cle, E->Ceq, NE->Cnotcompf(Ceq)); returns None for parity/signed conditions a bare float setcc cannot express.
fn fcmp_setcc_condition(tc: TestCond, is_double: bool) -> Option<Condition> {
    let cmp = match tc {
        TestCond::CondA => Comparison::Cgt,
        TestCond::CondAe => Comparison::Cge,
        TestCond::CondB => Comparison::Clt,
        TestCond::CondBe => Comparison::Cle,
        TestCond::CondE => Comparison::Ceq,
        TestCond::CondNe => {
            return Some(if is_double {
                Condition::Cnotcompf(Comparison::Ceq)
            } else {
                Condition::Cnotcompfs(Comparison::Ceq)
            })
        }
        _ => return None,
    };
    Some(if is_double {
        Condition::Ccompf(cmp)
    } else {
        Condition::Ccompfs(cmp)
    })
}

fn recover_signed_divisor(magic: i64, total_shift: i64) -> Option<i64> {
    if magic <= 0 || total_shift < 32 {
        return None;
    }
    let num = 1u128 << total_shift as u32;
    let mag = magic as u128;
    let d = (num + (mag / 2)) / mag; // rounded division
    if d <= 1 || d > i64::MAX as u128 {
        return None;
    }
    // Verify: magic ~= 2^total_shift / d
    let check = num / d;
    if (check as i64 - magic).unsigned_abs() <= 1 {
        Some(d as i64)
    } else {
        None
    }
}

fn recover_signed_divisor_compensating(magic_signed_i32: i64, total_shift: i64) -> Option<i64> {
    if total_shift < 32 || total_shift > 63 {
        return None;
    }
    // magic_signed_i32 came from a 32-bit immediate; interpret as i32 then add 2^32.
    let m_low = (magic_signed_i32 as i32) as i64;
    if m_low >= 0 {
        // Non-compensating already handled by recover_signed_divisor.
        return None;
    }
    let effective = (m_low as i64).wrapping_add(1i64 << 32);
    if effective <= 0 {
        return None;
    }
    let mag = effective as u128;
    let num = 1u128 << total_shift as u32;
    let d = (num + (mag / 2)) / mag;
    if d <= 1 || d > i64::MAX as u128 {
        return None;
    }
    let check = num / d;
    let check_i = check as i64;
    if (check_i - effective).unsigned_abs() <= 1 {
        Some(d as i64)
    } else {
        None
    }
}

fn recover_unsigned_divisor(magic: i64, total_shift: i64) -> Option<i64> {
    if total_shift < 32 {
        return None;
    }
    let mag = magic as u64 as u128;
    if mag == 0 {
        return None;
    }
    let num = 1u128 << total_shift as u32;
    let d = (num + (mag / 2)) / mag;
    if d <= 1 || d > i64::MAX as u128 {
        return None;
    }
    // Verify: |2^total_shift / d - magic| <= 1 (the magic is rounded either way).
    let check = num / d;
    let mag_u64 = magic as u64;
    let check_u64 = check as u64;
    let diff = if check_u64 >= mag_u64 {
        check_u64 - mag_u64
    } else {
        mag_u64 - check_u64
    };
    if diff <= 1 {
        Some(d as i64)
    } else {
        None
    }
}

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct AsmPassProgram;

    relation arg_constrained_as_ptr(Node, RTLReg);
    relation base_ident_to_symbol(Ident, Symbol);
    relation block_in_function(Node, Address);
    relation call_return_reg(Node, RTLReg);
    relation emit_clight_stmt(Address, Node, ClightStmt);
    relation emit_function_return_type_candidate(Address, ClightType);
    relation emit_goto_target(Address, Node);

    relation emit_loop_body(Address, Node, Node);
    relation emit_loop_exit(Address, Node, Node, Condition, Arc<Vec<CsharpminorExpr>>, Node, Node);
    relation emit_switch_chain(Address, Node, RTLReg);
    relation func_param_struct_type_candidate(Address, usize, usize);
    relation global_struct_catalog(u64, usize, usize, usize);
    relation ident_to_symbol(Ident, Symbol);

    relation instr_in_function(Node, Address);
    relation is_callee_saved(Mreg);
    relation known_extern_signature(Symbol, usize, XType, Arc<Vec<XType>>);
    relation known_func_param_is_ptr(Symbol, usize);
    relation known_func_returns_long(Symbol);
    relation known_func_returns_ptr(Symbol);
    relation main_function(Address);
    relation reg_rtl(Node, Mreg, RTLReg);
    relation reg_xtl(Node, Mreg, RTLReg);
    relation stack_var(Address, Address, i64, RTLReg);
    relation string_data(String, String, usize);
    relation struct_id_to_canonical(usize, usize);

    relation block_in_function(Node, Address);
    relation instr_in_function(Node, Address);


    relation preg_of(Mreg, Preg);
    relation ireg_of(Preg, Ireg);

    preg_of(Mreg::AX, Preg::Ir(Ireg::RAX));
    preg_of(Mreg::BX, Preg::Ir(Ireg::RBX));
    preg_of(Mreg::CX, Preg::Ir(Ireg::RCX));
    preg_of(Mreg::DX, Preg::Ir(Ireg::RDX));
    preg_of(Mreg::SI, Preg::Ir(Ireg::RSI));
    preg_of(Mreg::DI, Preg::Ir(Ireg::RDI));
    preg_of(Mreg::BP, Preg::Ir(Ireg::RBP));
    preg_of(Mreg::R8, Preg::Ir(Ireg::R8));
    preg_of(Mreg::R9, Preg::Ir(Ireg::R9));
    preg_of(Mreg::R10, Preg::Ir(Ireg::R10));
    preg_of(Mreg::R11, Preg::Ir(Ireg::R11));
    preg_of(Mreg::R12, Preg::Ir(Ireg::R12));
    preg_of(Mreg::R13, Preg::Ir(Ireg::R13));
    preg_of(Mreg::R14, Preg::Ir(Ireg::R14));
    preg_of(Mreg::R15, Preg::Ir(Ireg::R15));

    preg_of(Mreg::X0, Preg::Fr(Freg::XMM0));
    preg_of(Mreg::X1, Preg::Fr(Freg::XMM1));
    preg_of(Mreg::X2, Preg::Fr(Freg::XMM2));
    preg_of(Mreg::X3, Preg::Fr(Freg::XMM3));
    preg_of(Mreg::X4, Preg::Fr(Freg::XMM4));
    preg_of(Mreg::X5, Preg::Fr(Freg::XMM5));
    preg_of(Mreg::X6, Preg::Fr(Freg::XMM6));
    preg_of(Mreg::X7, Preg::Fr(Freg::XMM7));
    preg_of(Mreg::X8, Preg::Fr(Freg::XMM8));
    preg_of(Mreg::X9, Preg::Fr(Freg::XMM9));
    preg_of(Mreg::X10, Preg::Fr(Freg::XMM10));
    preg_of(Mreg::X11, Preg::Fr(Freg::XMM11));
    preg_of(Mreg::X12, Preg::Fr(Freg::XMM12));
    preg_of(Mreg::X13, Preg::Fr(Freg::XMM13));
    preg_of(Mreg::X14, Preg::Fr(Freg::XMM14));
    preg_of(Mreg::X15, Preg::Fr(Freg::XMM15));

    preg_of(Mreg::FP0, Preg::ST0);


    ireg_of(Preg::Ir(Ireg::RAX), Ireg::RAX);
    ireg_of(Preg::Ir(Ireg::RBX), Ireg::RBX);
    ireg_of(Preg::Ir(Ireg::RCX), Ireg::RCX);
    ireg_of(Preg::Ir(Ireg::RDX), Ireg::RDX);
    ireg_of(Preg::Ir(Ireg::RSI), Ireg::RSI);
    ireg_of(Preg::Ir(Ireg::RDI), Ireg::RDI);
    ireg_of(Preg::Ir(Ireg::RBP), Ireg::RBP);
    ireg_of(Preg::Ir(Ireg::RSP), Ireg::RSP);
    ireg_of(Preg::Ir(Ireg::R8), Ireg::R8);
    ireg_of(Preg::Ir(Ireg::R9), Ireg::R9);
    ireg_of(Preg::Ir(Ireg::R10), Ireg::R10);
    ireg_of(Preg::Ir(Ireg::R11), Ireg::R11);
    ireg_of(Preg::Ir(Ireg::R12), Ireg::R12);
    ireg_of(Preg::Ir(Ireg::R13), Ireg::R13);
    ireg_of(Preg::Ir(Ireg::R14), Ireg::R14);
    ireg_of(Preg::Ir(Ireg::R15), Ireg::R15);

    relation symbols(Address, Symbol, Symbol);
    relation symbol_size(Address, usize, Symbol);
    relation builtins(Symbol);

    relation func_entry(Symbol, Address);

    relation next(Address, Address);
    relation op_register(Symbol, &'static str);
    relation op_immediate(Symbol, i64, usize);
    relation block_boundaries(Address, Address, Address);
    relation reg_use(Address, Mreg);
    relation reg_def(Address, Mreg);
    relation reg_def_used(Address, Mreg, Address);
    relation decoded_reg_use(Address, Mreg);
    relation decoded_reg_def(Address, Mreg);
    // Immutable snapshots of the decoder's register effects, populated in
    // dedicated seed inputs by the pass wrapper before this fixpoint starts.
    // Public identity outputs both avoid feedback from Mach-derived facts and
    // make the scheduled pipeline retain the rows and order RTL after Asm.
    relation asm_reg_use_seed(Address, Mreg);
    relation asm_reg_def_seed(Address, Mreg);
    relation asm_reg_use(Address, Mreg);
    relation asm_reg_def(Address, Mreg);
    asm_reg_use(addr, *reg) <-- asm_reg_use_seed(addr, reg);
    asm_reg_def(addr, *reg) <-- asm_reg_def_seed(addr, reg);

    relation stack_def(Address, Symbol, i64);
    relation stack_use(Address, Symbol, i64);
    relation stack_def_used(Address, Symbol, i64, Address, Symbol, i64);

    relation flags_and_jump_pair(Address, Address, &'static str);

    relation op_indirect(Symbol, &'static str, &'static str, &'static str, i64, i64, usize);

    relation code_in_block(Address, Address);

    relation block(Address);

    relation ddisasm_cfg_edge(Address, Address, Symbol);

    relation direct_call(Address, Address);

    relation ddisasm_function_entry(Address);

    relation direct_jump(Address, Address);

    // Relocation-authenticated import-pointer JMPs are resolved tail calls,
    // not unknown intra-function predecessors.  This relation is declared
    // here because the frame-safety proof consumes it before Mach lowering.
    relation is_extern_tailcall_jmp(Address);

    relation stack_base_move(Address, Symbol, Symbol);
    // Decoder-owned affine changes to the architectural stack pointer.  This
    // is deliberately distinct from reg_def: only these exact adjustments may
    // preserve an entry-frame coordinate across an SP write.
    relation adjusts_stack(Address, Symbol, i64);

    // INT/INT3 end a decoder basic block, but Windows checked-build INT 2Ch
    // assertions and INT3 debug breaks resume at the following instruction
    // after debugger/exception handling.
    // Preserve only this use-specific fallthrough for frame propagation; the
    // program CFG itself remains unchanged and still represents the trap.
    #[local] relation trap_fallthrough(Address, Address);
    trap_fallthrough(src, dst) <--
        next(src, dst),
        instruction(src, _, _, "INT3", _, _, _, _, _, _),
        instr_in_function(src, func),
        instr_in_function(dst, func);
    trap_fallthrough(src, dst) <--
        next(src, dst),
        instruction(src, _, _, "INT", vector, _, _, _, _, _),
        op_immediate(vector, value, _),
        if matches!(*value, 3 | 0x2c),
        instr_in_function(src, func),
        instr_in_function(dst, func);
    #[local] relation frame_cfg_step(Address, Address);
    frame_cfg_step(src, dst) <-- cfg_step(src, dst);
    frame_cfg_step(src, dst) <-- trap_fallthrough(src, dst);

    // A BP-relative access is frame based only when one particular RSP->RBP
    // copy reaches and dominates that access. A function-global "ever copied"
    // bit is unsound for conditional copies and later RBP clobbers.
    #[local] relation bp_func_entry_block(Address, Address);
    bp_func_entry_block(func, block) <--
        block_in_function(block, func),
        code_in_block(func, block);

    #[local] relation bp_block_next(Address, Address);
    bp_block_next(src_block, dst_block) <--
        ddisasm_cfg_edge(src, dst, edge_type),
        if *edge_type != "call" && *edge_type != "indirect" && *edge_type != "indirect_call",
        code_in_block(src, src_block),
        code_in_block(dst, dst_block);
    bp_block_next(src_block, dst_block) <--
        trap_fallthrough(src, dst),
        code_in_block(src, src_block),
        code_in_block(dst, dst_block);

    #[local] lattice bp_block_strict_dom_set(Address, Address, Dual<Set<Address>>);
    #[local] lattice bp_block_dom_set(Address, Address, Dual<Set<Address>>);
    bp_block_dom_set(*func, *entry, Dual(Set::singleton(*entry))) <--
        bp_func_entry_block(func, entry);
    bp_block_strict_dom_set(*func, *node, Dual(pred_doms.0.clone())) <--
        bp_block_dom_set(func, pred, pred_doms),
        bp_block_next(pred, node),
        block_in_function(node, func),
        !bp_func_entry_block(func, node);
    bp_block_dom_set(*func, *node, Dual(asm_dom_set_with_self(&strict.0, *node))) <--
        bp_block_strict_dom_set(func, node, strict),
        !bp_func_entry_block(func, node);

    // BP-only may-reaching defs over decoded instruction CFG. The reaching
    // value is available at a redefining instruction itself (which reads the
    // old value) but does not propagate past that clobber.
    #[local] relation bp_def_reaches(Address, Address, Address);
    bp_def_reaches(func, def, succ) <--
        asm_reg_def(def, ?&Mreg::BP),
        instr_in_function(def, func),
        frame_cfg_step(def, succ),
        instr_in_function(succ, func);
    bp_def_reaches(func, def, next_addr) <--
        bp_def_reaches(func, def, cur),
        !asm_reg_def(cur, Mreg::BP),
        frame_cfg_step(cur, next_addr),
        instr_in_function(next_addr, func);

    #[local] relation bp_copy_dominates(Address, Address, Address);
    bp_copy_dominates(func, copy, access) <--
        code_in_block(copy, block),
        code_in_block(access, block),
        instr_in_function(copy, func),
        if *copy < *access;
    bp_copy_dominates(func, copy, access) <--
        code_in_block(copy, copy_block),
        code_in_block(access, access_block),
        if *copy_block != *access_block,
        instr_in_function(copy, func),
        instr_in_function(access, func),
        bp_block_dom_set(func, access_block, doms),
        if doms.0.contains(copy_block);

    #[local] relation bp_competing_reaching_def(Address, Address, Address);
    bp_competing_reaching_def(func, copy, access) <--
        bp_def_reaches(func, copy, access),
        bp_def_reaches(func, other, access),
        if other != copy;

    // The architectural register family collapses EBP/RBP to Mreg::BP, but
    // address-size-overridden [ebp] is not a 64-bit frame access. Keep the raw
    // spelling in the proof so only an actual [rbp] operand can become a stack
    // claim; subregister bases continue through the ordinary pointer routes.
    #[local] relation exact_rbp_mem_access(Address);
    exact_rbp_mem_access(addr) <--
        instruction(addr, _, _, _, operand, _, _, _, _, _),
        op_indirect(operand, _, "RBP", _, _, _, _);
    exact_rbp_mem_access(addr) <--
        instruction(addr, _, _, _, _, operand, _, _, _, _),
        op_indirect(operand, _, "RBP", _, _, _, _);
    exact_rbp_mem_access(addr) <--
        instruction(addr, _, _, _, _, _, operand, _, _, _),
        op_indirect(operand, _, "RBP", _, _, _, _);
    exact_rbp_mem_access(addr) <--
        instruction(addr, _, _, _, _, _, _, operand, _, _),
        op_indirect(operand, _, "RBP", _, _, _, _);

    #[local] relation bp_rsp_value_reaches(Address, Address, Address);
    bp_rsp_value_reaches(access, func, copy) <--
        stack_base_move(copy, src, dst),
        if *src == "RSP" && *dst == "RBP",
        bp_def_reaches(func, copy, access),
        bp_copy_dominates(func, copy, access),
        !bp_competing_reaching_def(func, copy, access);

    #[local] relation bp_rsp_copy_reaches(Address, Address, Address);
    bp_rsp_copy_reaches(access, func, copy) <--
        bp_rsp_value_reaches(access, func, copy),
        exact_rbp_mem_access(access);

    // Entry-anchored RSP state, evaluated over the real instruction CFG.  The
    // previous stack_offset lattice walked `next` linearly, so it crossed
    // branches and treated arbitrary writes such as MOV/AND RSP as if they
    // preserved the frame. A flat constant-propagation lattice stores at most
    // one exact coordinate per (function, instruction); conflicting paths,
    // killed paths, and non-invariant loop edges become Top immediately. This
    // bounds both memory and fixpoint work independently of the path count.
    #[local] relation rsp_transfer_kill(Address);
    rsp_transfer_kill(addr) <--
        instruction(addr, _, _, "POP", dst, _, _, _, _, _),
        op_register(dst, dst_str),
        if matches!(*dst_str, "RSP" | "SP");
    rsp_transfer_kill(addr) <--
        asm_reg_def(addr, ?&Mreg::SP),
        !adjusts_stack(addr, "RSP", _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        // CALL's architectural push is undone before its fallthrough executes.
        if *mnem != "CALL";

    #[local] lattice rsp_state(Address, Address, ConstPropagation<i64>);

    rsp_state(func, func, ConstPropagation::Constant(0)) <--
        func_span(_, func, _);

    // Affine transfer over every CFG edge. A differing back-edge value joins
    // its header to Top just like a conflicting forward join.
    rsp_state(*func, *dst, next_state) <--
        rsp_state(func, src, state),
        frame_cfg_step(src, dst),
        instr_in_function(src, func),
        instr_in_function(dst, func),
        !rsp_transfer_kill(src),
        adjusts_stack(src, "RSP", delta),
        let next_state = match state {
            ConstPropagation::Constant(ofs) => ofs
                .checked_add(*delta)
                .map(ConstPropagation::Constant)
                .unwrap_or(ConstPropagation::Top),
            ConstPropagation::Bottom => ConstPropagation::Bottom,
            ConstPropagation::Top => ConstPropagation::Top,
        };

    // Identity transfer, including calls.
    rsp_state(*func, *dst, (*state).clone()) <--
        rsp_state(func, src, state),
        frame_cfg_step(src, dst),
        instr_in_function(src, func),
        instr_in_function(dst, func),
        !rsp_transfer_kill(src),
        !adjusts_stack(src, "RSP", _);

    // A non-affine write destroys the coordinate for every successor.
    rsp_state(*func, *dst, ConstPropagation::Top) <--
        rsp_state(func, src, _),
        frame_cfg_step(src, dst),
        instr_in_function(src, func),
        instr_in_function(dst, func),
        rsp_transfer_kill(src);

    relation rsp_frame_offset_at(Address, Address, i64);
    rsp_frame_offset_at(*func, *access, *ofs) <--
        rsp_state(func, access, state),
        if let ConstPropagation::Constant(ofs) = state;

    relation rsp_frame_at(Address, Address);
    rsp_frame_at(*access, *func) <--
        rsp_frame_offset_at(func, access, _);

    // A reaching/dominating RSP->RBP copy proves a frame base only if the copy
    // itself still has a unique entry-anchored RSP coordinate. Otherwise RBP
    // merely preserves an arbitrary stack-derived pointer and is unsupported.
    relation bp_frame_at(Address, Address);
    bp_frame_at(*access, *func) <--
        bp_rsp_copy_reaches(access, func, copy),
        rsp_frame_at(copy, func);

    #[local] relation invalid_rsp_derived_bp_at(Address, Address);
    invalid_rsp_derived_bp_at(*access, *func) <--
        bp_rsp_copy_reaches(access, func, copy),
        !rsp_frame_at(copy, func);

    #[local] relation proven_frame_base_at(Address, Mreg);
    proven_frame_base_at(access, Mreg::SP) <-- rsp_frame_at(access, _);
    proven_frame_base_at(access, Mreg::BP) <-- bp_frame_at(access, _);

    #[local] relation unproven_frame_base_at(Address, Mreg);
    unproven_frame_base_at(access, Mreg::SP) <--
        instr_in_function(access, func),
        !rsp_frame_at(access, func);
    unproven_frame_base_at(access, Mreg::BP) <--
        instr_in_function(access, func),
        !bp_frame_at(access, func);

    // Pointer-preserving fallback address for BP memory operands that are not
    // entitled to scalar stack-slot lowering. RBP remains an ordinary SSA
    // register when it is not a proved frame base. RSP is intentionally
    // excluded: the source IR has no sound way to name an arbitrary current
    // stack pointer after a non-affine or path-dependent write.
    #[local] relation raw_operand_at(Address, Symbol);
    raw_operand_at(addr, *operand) <--
        instruction(addr, _, _, _, operand, _, _, _, _, _);
    raw_operand_at(addr, *operand) <--
        instruction(addr, _, _, _, _, operand, _, _, _, _);
    raw_operand_at(addr, *operand) <--
        instruction(addr, _, _, _, _, _, operand, _, _, _);
    raw_operand_at(addr, *operand) <--
        instruction(addr, _, _, _, _, _, _, operand, _, _);

    // Decoder rows are authoritative. The fallback keeps hand-seeded tests on
    // the ordinary 64-bit addressing path when they omit decoder metadata.
    #[local] relation effective_address_size(Address, u8);
    effective_address_size(*addr, *size) <--
        instruction_address_size(addr, size);
    effective_address_size(*addr, 8) <--
        instruction(addr, _, _, _, _, _, _, _, _, _),
        !instruction_address_size(addr, _);

    #[local] relation addr32_memory_operand(Address, Symbol, &'static str, &'static str, &'static str, i64, i64);
    addr32_memory_operand(*addr, *operand, *segment, *base, *index, *scale, *disp) <--
        effective_address_size(addr, address_size),
        if *address_size == 4,
        raw_operand_at(addr, operand),
        op_indirect(operand, segment, base, index, scale, disp, _);

    #[local] relation unsupported_addr32_access(Address, Address, Symbol);

    // A 32-bit effective address is not the current 64-bit stack pointer, and
    // this IR deliberately has no expression for ESP's independently wrapped
    // value. Segment overrides and absolute/symbolic forms likewise need
    // provenance which cannot be represented by Aaddr32.
    unsupported_addr32_access(*func, *addr, "addr32-segment") <--
        instr_in_function(addr, func),
        addr32_memory_operand(addr, _, segment, _, _, _, _),
        if !is_no_address_register(segment);
    unsupported_addr32_access(*func, *addr, "addr32-register") <--
        instr_in_function(addr, func),
        addr32_memory_operand(addr, _, _, base, index, _, _),
        if (!is_no_address_register(base) && !is_addr32_gp_name(base))
            || (!is_no_address_register(index) && !is_addr32_gp_name(index));
    unsupported_addr32_access(*func, *addr, "addr32-pattern") <--
        instr_in_function(addr, func),
        addr32_memory_operand(addr, _, _, base, index, scale, _),
        if (is_no_address_register(base) && is_no_address_register(index))
            || !matches!(*scale, 1 | 2 | 4 | 8);

    // EBP is a valid addr32 scratch base, but a reaching RSP->RBP frame copy
    // means its low half is stack-derived. Do not silently reinterpret that
    // as either Ainstack or an ordinary pointer.
    unsupported_addr32_access(*func, *addr, "addr32-frame-ebp") <--
        instr_in_function(addr, func),
        addr32_memory_operand(addr, _, _, base, index, _, _),
        if *base == "EBP" || *index == "EBP",
        bp_rsp_value_reaches(addr, func, _);

    // A shared instruction has one node-global lowering.  If one owner sees
    // EBP as a reaching frame copy while another sees it as an ordinary
    // scratch register, neither interpretation is safe for that shared node;
    // reject it for every owner.
    #[local] relation shared_addr32_ebp_disagreement(Address);
    shared_addr32_ebp_disagreement(*addr) <--
        addr32_memory_operand(addr, _, _, base, index, _, _),
        if *base == "EBP" || *index == "EBP",
        instr_in_function(addr, frame_owner),
        instr_in_function(addr, other_owner),
        if frame_owner != other_owner,
        bp_rsp_value_reaches(addr, frame_owner, _),
        !bp_rsp_value_reaches(addr, other_owner, _);
    unsupported_addr32_access(*func, *addr, "addr32-shared-ebp") <--
        shared_addr32_ebp_disagreement(addr),
        instr_in_function(addr, func);

    // x86 normally has one explicit memory operand. If a decoder ever emits
    // more than one, selecting one by ordinal would be nondeterministic and
    // can attach the wrong effective address to a synthetic RTL node.
    unsupported_addr32_access(*func, *addr, "addr32-multiple-memory-operands") <--
        instr_in_function(addr, func),
        addr32_memory_operand(addr, first, _, _, _, _, _),
        addr32_memory_operand(addr, second, _, _, _, _, _),
        if first != second;

    #[local] relation unsafe_stack_access(Address, Address, Symbol);

    // Only these binary integer memory-source forms have the dedicated
    // three-node indexed-RSP lowering in RTL. A frame proof establishes the
    // coordinate, not support for every opcode or for RSP as index/destination.
    #[local] relation supported_proved_rsp_indexed(Address);
    supported_proved_rsp_indexed(addr) <--
        float_load_op(addr, op, _, addressing, args, dst, false),
        if args.len() == 2 && args[0] == Mreg::SP && args[1] != Mreg::SP,
        if *dst != Mreg::SP,
        if matches!(addressing, Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if matches!(op,
            Operation::Oadd | Operation::Oaddl |
            Operation::Osub | Operation::Osubl |
            Operation::Oand | Operation::Oandl |
            Operation::Oor | Operation::Oorl |
            Operation::Oxor | Operation::Oxorl);

    // Ordinary MOV loads/stores already have the established SP-indexed
    // two-node lowering in RTL.  Keep those on the proved path as well; the
    // fused-op allowlist above is deliberately narrower because it needs a
    // distinct load-then-op chain.
    supported_proved_rsp_indexed(addr) <--
        mach_inst(addr, ?MachInst::Mload(_, addressing, args, dst)),
        if args.len() == 2 && args[0] == Mreg::SP && args[1] != Mreg::SP,
        if *dst != Mreg::SP,
        if matches!(addressing, Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _));
    supported_proved_rsp_indexed(addr) <--
        mach_inst(addr, ?MachInst::Mstore(_, addressing, args, _)),
        if args.len() == 2 && args[0] == Mreg::SP && args[1] != Mreg::SP,
        if matches!(addressing, Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _));

    // FS/GS contribute a segment base that none of the Mach/RTL addressing
    // forms represent.  In particular, an operand such as fs:[rsp+N] is not
    // the ordinary stack/home cell at [rsp+N].  Reject every explicit FS/GS
    // memory operand at the shared structured-address boundary until segment
    // bases become first-class, rather than allowing a lowering rule that
    // ignores the segment column to scalarize it.
    unsafe_stack_access(*func, *addr, "unmodeled-segment") <--
        instr_in_function(addr, func),
        raw_operand_at(addr, operand),
        op_indirect(operand, segment, _, _, _, _, _),
        if is_unmodeled_segment(segment);

    unsafe_stack_access(*func, *addr, "rsp-coordinate-unknown") <--
        instr_in_function(addr, func),
        raw_operand_at(addr, operand),
        op_indirect(operand, _, "RSP", _, _, _, _),
        !rsp_frame_at(addr, func);

    unsafe_stack_access(*func, *addr, "rsp-indexed-lowering-missing") <--
        instr_in_function(addr, func),
        raw_operand_at(addr, operand),
        op_indirect(operand, _, "RSP", index, _, _, _),
        if *index != "NONE" && !index.is_empty(),
        rsp_frame_at(addr, func),
        !supported_proved_rsp_indexed(addr);

    // Proved indexed RSP accesses are not scalar slots, but RTL has a
    // dedicated lowering for them: it materializes the exact entry-anchored
    // stack base and then applies the index. The preceding rule still rejects
    // indexed accesses whose current RSP coordinate is path-dependent or
    // otherwise unknown.

    unsafe_stack_access(*func, *addr, "rbp-coordinate-unknown") <--
        invalid_rsp_derived_bp_at(addr, func);

    // Node-global IR cannot safely choose between different per-function
    // frame coordinates for a shared tail. Reject every owner of a shared
    // stack-memory instruction rather than emitting conflicting candidates.
    #[local] relation shared_stack_access(Address);
    shared_stack_access(*addr) <--
        instr_in_function(addr, first),
        instr_in_function(addr, second),
        if first != second,
        raw_operand_at(addr, operand),
        op_indirect(operand, _, base, _, _, _, _),
        if *base == "RSP" || *base == "RBP";
    unsafe_stack_access(*func, *addr, "shared-stack-node") <--
        shared_stack_access(addr),
        instr_in_function(addr, func);

    // An unresolved computed jump has no complete predecessor set. It can
    // enter a later stack access after an unobserved SP/BP mutation, so no
    // stack classification in that function is trustworthy.
    #[local] relation function_has_unresolved_indirect(Address);
    function_has_unresolved_indirect(*func) <--
        ddisasm_cfg_edge(src, _, edge_type),
        if *edge_type == "indirect",
        !is_extern_tailcall_jmp(src),
        instr_in_function(src, func);
    // An unresolved predecessor may mutate BP before reaching the access.
    // Even an apparently scratch EBP therefore lacks sufficient provenance.
    unsupported_addr32_access(*func, *addr, "addr32-unresolved-indirect") <--
        function_has_unresolved_indirect(func),
        instr_in_function(addr, func),
        addr32_memory_operand(addr, _, _, base, index, _, _),
        if *base == "EBP" || *index == "EBP";
    unsafe_stack_access(*func, *addr, "unresolved-indirect-control-flow") <--
        function_has_unresolved_indirect(func),
        instr_in_function(addr, func),
        raw_operand_at(addr, operand),
        op_indirect(operand, _, base, _, _, _, _),
        if *base == "RSP" || *base == "RBP";

    // Machine-readable trigger provenance.  The stable public suppression
    // reason remains one of the two adapter-facing codes below, while this
    // relation distinguishes cases that can gain a positive proof from cases
    // that are architecturally unrepresentable.
    relation unsupported_address_detail_seed(Address, Address, Symbol);
    unsupported_address_detail_seed(*func, *addr, *detail) <--
        unsafe_stack_access(func, addr, detail);
    unsupported_address_detail_seed(*func, *addr, *detail) <--
        unsupported_addr32_access(func, addr, detail);

    relation unsupported_stack_address_seed(Address, Address, Symbol);
    unsupported_stack_address_seed(*func, *addr, "unsupported-stack-address") <--
        unsafe_stack_access(func, addr, _);
    unsupported_stack_address_seed(*func, *addr, "unsupported-addr32-address") <--
        unsupported_addr32_access(func, addr, _);

    #[local] relation generic_bp_sp_address(Address, Symbol, Addressing, Arc<Vec<Mreg>>);
    generic_bp_sp_address(addr, *operand, Addressing::Aindexed(*disp), Arc::new(vec![Mreg::BP])) <--
        raw_operand_at(addr, operand),
        op_indirect(operand, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        unproven_frame_base_at(addr, Mreg::BP),
        !invalid_rsp_derived_bp_at(addr, _);
    generic_bp_sp_address(addr, *operand, addressing, Arc::new(args)) <--
        raw_operand_at(addr, operand),
        op_indirect(operand, _, base_str, idx_str, scale, disp, _),
        if *base_str == "RBP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        !invalid_rsp_derived_bp_at(addr, _),
        let index = Mreg::x86(*idx_str),
        let addressing = if *scale > 1 {
            Addressing::Aindexed2scaled(*scale, *disp)
        } else {
            Addressing::Aindexed2(*disp)
        },
        let args = vec![Mreg::BP, index];

    instr_in_function(addr, func)<--
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


    relation pcast(Address, Symbol, Symbol);
    pcast(addr, dst, src) <--
        instruction(addr, _, _, "CQO", src, dst, _, _, _, _);
    pcast(addr, dst, src) <--
        instruction(addr, _, _, "CDQ", src, dst, _, _, _, _);
    pcast(addr, dst, src) <--
        instruction(addr, _, _, "CWD", src, dst, _, _, _, _);


    relation cr8_operand(Symbol);
    cr8_operand(operand) <--
        op_register(operand, name),
        if *name == "CR8";

    relation pcr8_read(Address, Mreg);
    pcr8_read(addr, dst_reg) <--
        instruction(addr, _, _, "MOV", src, dst, _, _, _, _),
        cr8_operand(src),
        op_register(dst, dst_name),
        let dst_reg = Mreg::x86(dst_name),
        if dst_reg != Mreg::Unknown;

    relation pmov(Address, Symbol, Symbol);
    pmov(addr, dst, src)<--
        instruction(addr, _, _, "MOV", src, dst, _, _, _, _),
        !cr8_operand(src),
        !cr8_operand(dst);

    pmov(addr, dst, src)<--
        instruction(addr, _, _, "MOVZX", src, dst, _, _, _, _);

    pmov(addr, dst, src) <--
        instruction(addr, _, _, "MOVSX", src, dst, _, _, _, _);
    pmov(addr, dst, src) <--
        instruction(addr, _, _, "MOVSXD", src, dst, _, _, _, _);

    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVAPS", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVAPS", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVUPS", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVUPS", src, dst, _, _, _, _);
    // Packed-double reg moves (double counterpart of MOVAPS/MOVUPS): absent from pmov, a reg-reg movapd produced no mach_inst, leaving dst undefined and silently dropping the double dataflow through it (e.g. an XMM copy feeding fabs(p1-p2)).
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVAPD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVAPD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVUPD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVUPD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVDQA", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVDQA", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVDQU", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVDQU", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVQ", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVQ", src, dst, _, _, _, _);
    // MOVD is the 32-bit GP<->XMM transfer of the float-bits shuffle; absent from the move list and excluded from ALU lowering, it produced no instruction and the float value became 0.
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVSD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVSD", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVSS", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "VMOVSS", src, dst, _, _, _, _);
    pmov(addr, dst, src) <-- instruction(addr, _, _, "MOVABS", src, dst, _, _, _, _);


    relation pand(Address, Symbol, Symbol);
    pand(addr, dst, src)<--
        instruction(addr, _, _, "AND", src, dst, _, _, _, _);
    // Atomic RMW: capstone glues the lock prefix onto the mnemonic, so the bare-mnemonic rule never fires; the memory-dest lowering is identical to the non-atomic load-op-store.
    pand(addr, dst, src)<--
        instruction(addr, _, _, "LOCK AND", src, dst, _, _, _, _);

    relation psetcc(Address, TestCond, Symbol);
    psetcc(addr, TestCond::CondNp, dst)<--
        instruction(addr, _, _, "SETNP", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondP, dst)<--
        instruction(addr, _, _, "SETP", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondE, dst)<--
        instruction(addr, _, _, "SETE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondNe, dst)<--
        instruction(addr, _, _, "SETNE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondB, dst)<--
        instruction(addr, _, _, "SETB", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondBe, dst)<--
        instruction(addr, _, _, "SETBE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondAe, dst)<--
        instruction(addr, _, _, "SETAE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondA, dst)<--
        instruction(addr, _, _, "SETA", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondL, dst)<--
        instruction(addr, _, _, "SETL", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondLe, dst)<--
        instruction(addr, _, _, "SETLE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondGe, dst)<--
        instruction(addr, _, _, "SETGE", dst, _, _, _, _, _);

    psetcc(addr, TestCond::CondG, dst)<--
        instruction(addr, _, _, "SETG", dst, _, _, _, _, _);

    relation pxor(Address, Symbol, Symbol);
    pxor(addr, dst, src)<--
        instruction(addr, _, _, "XOR", src, dst, _, _, _, _);

    pxor(addr, dst, src)<--
        instruction(addr, _, _, "XORL", src, dst, _, _, _, _);
    pxor(addr, dst, src)<--
        instruction(addr, _, _, "LOCK XOR", src, dst, _, _, _, _);

    // Bit-test-and-modify with an immediate index, modelled as a masked bitwise op (BTC xor, BTS or, BTR and-not); BT is excluded since it writes no register and a fabricated def would corrupt dataflow.
    relation pbtc(Address, Symbol, Symbol);
    pbtc(addr, dst, src)<--
        instruction(addr, _, _, "BTC", src, dst, _, _, _, _);

    relation pbts(Address, Symbol, Symbol);
    pbts(addr, dst, src)<--
        instruction(addr, _, _, "BTS", src, dst, _, _, _, _);

    relation pbtr(Address, Symbol, Symbol);
    pbtr(addr, dst, src)<--
        instruction(addr, _, _, "BTR", src, dst, _, _, _, _);

    // Address of a genuinely-packed SIMD op: these produce no mach_inst, so this flag is the only surviving trace, plumbed downstream so structuring can recognize an auto-vectorized loop.
    relation packed_alu_addr(Address);
    packed_alu_addr(*addr) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if is_packed_alu_mnem(mnem);

    relation por(Address, Symbol, Symbol);
    por(addr, dst, src)<--
        instruction(addr, _, _, "OR", src, dst, _, _, _, _);
    por(addr, dst, src)<--
        instruction(addr, _, _, "LOCK OR", src, dst, _, _, _, _);

    relation pneg(Address, Symbol);
    pneg(addr, dst)<--
        instruction(addr, _, _, "NEG", dst, _, _, _, _, _);

    relation psub(Address, Symbol, Symbol);
    psub(addr, dst, src)<--
        instruction(addr, _, _, "SUB", src, dst, _, _, _, _);
    psub(addr, dst, src)<--
        instruction(addr, _, _, "LOCK SUB", src, dst, _, _, _, _);

    relation padd(Address, Symbol, Symbol);
    padd(addr, dst, src)<--
        instruction(addr, _, _, "ADD", src, dst, _, _, _, _);
    padd(addr, dst, src)<--
        instruction(addr, _, _, "LOCK ADD", src, dst, _, _, _, _);

    relation pjcc(Address, TestCond, Symbol);
    pjcc(addr, TestCond::CondNp, dst)<--
        instruction(addr, _, _, "JNP", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondP, dst)<--
        instruction(addr, _, _, "JP", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondE, dst)<--
        instruction(addr, _, _, "JE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondNe, dst)<--
        instruction(addr, _, _, "JNE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondB, dst)<--
        instruction(addr, _, _, "JB", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondBe, dst)<--
        instruction(addr, _, _, "JBE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondAe, dst)<--
        instruction(addr, _, _, "JAE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondA, dst)<--
        instruction(addr, _, _, "JA", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondL, dst)<--
        instruction(addr, _, _, "JL", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondLe, dst)<--
        instruction(addr, _, _, "JLE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondGe, dst)<--
        instruction(addr, _, _, "JGE", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondG, dst)<--
        instruction(addr, _, _, "JG", dst, _, _, _, _, _);

    relation pcmp(Address, Symbol, Symbol);
    pcmp(addr, dst, src)<--
        instruction(addr, _, _, "CMP", src, dst, _, _, _, _);

    #[local] relation pcmov(Address, TestCond, Symbol, Symbol);

    pcmov(addr, TestCond::CondE, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVE" || *inst == "CMOVZ";

    pcmov(addr, TestCond::CondNe, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVNE" || *inst == "CMOVNZ";

    pcmov(addr, TestCond::CondB, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVB" || *inst == "CMOVC" || *inst == "CMOVNAE";

    pcmov(addr, TestCond::CondAe, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVAE" || *inst == "CMOVNC" || *inst == "CMOVNB";

    pcmov(addr, TestCond::CondBe, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVBE" || *inst == "CMOVNA";

    pcmov(addr, TestCond::CondA, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVA" || *inst == "CMOVNBE";

    pcmov(addr, TestCond::CondL, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVL" || *inst == "CMOVNGE";

    pcmov(addr, TestCond::CondGe, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVGE" || *inst == "CMOVNL";

    pcmov(addr, TestCond::CondLe, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVLE" || *inst == "CMOVNG";

    pcmov(addr, TestCond::CondG, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVG" || *inst == "CMOVNLE";

    pcmov(addr, TestCond::CondP, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVP" || *inst == "CMOVPE";

    pcmov(addr, TestCond::CondNp, dst, src) <--
        instruction(addr, _, _, inst, src, dst, _, _, _, _),
        if *inst == "CMOVNP" || *inst == "CMOVPO";

    // CMOVS (sign flag set = negative) maps to CondL (same as JS)
    pcmov(addr, TestCond::CondL, dst, src) <--
        instruction(addr, _, _, "CMOVS", src, dst, _, _, _, _);

    // CMOVNS (sign flag clear = non-negative) maps to CondGe (same as JNS)
    pcmov(addr, TestCond::CondGe, dst, src) <--
        instruction(addr, _, _, "CMOVNS", src, dst, _, _, _, _);

    relation pmul(Address, Symbol);
    pmul(addr, dst)<--
        instruction(addr, _, _, "IMUL", dst, "0", _, _, _, _);
    pmul(addr, dst)<--
        instruction(addr, _, _, "MUL", dst, "0", _, _, _, _);

    relation pimul(Address, Symbol, Symbol);
    pimul(addr, dst, src)<--
        instruction(addr, _, _, "IMUL", src, dst, op3, _, _, _),
        if *dst != "0",
        if *op3 == "0";

    // 3-operand IMUL: imul dst, src, imm (dst = src * imm)
    relation pimul3(Address, Symbol, Symbol, Symbol);
    pimul3(addr, dst, src, imm)<--
        instruction(addr, _, _, "IMUL", src, dst, imm, _, _, _),
        if *dst != "0",
        if *imm != "0";

    relation ppush(Address, Symbol);
    ppush(addr, src)<--
        instruction(addr, _, _, "PUSH", src, _, _, _, _, _);

    relation ppop(Address, Symbol);
    ppop(addr, dst)<--
        instruction(addr, _, _, "POP", dst, _, _, _, _, _);

    relation pidiv(Address, Symbol, Symbol);
    pidiv(addr, src, dst)<--
        instruction(addr, _, _, "IDIV", src, dst, _, _, _, _);

    relation pudiv(Address, Symbol, Symbol);
    pudiv(addr, src, dst)<--
        instruction(addr, _, _, "DIV", src, dst, _, _, _, _);

    relation pdiv(Address, Symbol, Symbol);
    pdiv(addr, src, dst) <-- pidiv(addr, src, dst);
    pdiv(addr, src, dst) <-- pudiv(addr, src, dst);

    relation pnot(Address, Symbol);
    pnot(addr, dst)<--
        instruction(addr, _, _, "NOT", dst, _, _, _, _, _);

    relation psal(Address, Symbol, Symbol);
    psal(addr, dst, src)<--
        instruction(addr, _, _, "SAL", src, dst, _, _, _, _);
    psal(addr, dst, src)<--
        instruction(addr, _, _, "SHL", src, dst, _, _, _, _);

    relation psar(Address, Symbol, Symbol);
    psar(addr, dst, src)<--
        instruction(addr, _, _, "SAR", src, dst, _, _, _, _);

    relation pshr(Address, Symbol, Symbol);
    pshr(addr, dst, src)<--
        instruction(addr, _, _, "SHR", src, dst, _, _, _, _);

    relation pror(Address, Symbol, Symbol);
    pror(addr, dst, src)<--
        instruction(addr, _, _, "ROR", src, dst, _, _, _, _);

    relation prol(Address, Symbol, Symbol);
    prol(addr, dst, src)<--
        instruction(addr, _, _, "ROL", src, dst, _, _, _, _);

    relation pmovsd(Address, Symbol, Symbol);
    pmovsd(addr, dst, src)<--
        instruction(addr, _, _, "MOVSD", src, dst, _, _, _, _);
    pmovsd(addr, dst, src)<--
        instruction(addr, _, _, "VMOVSD", src, dst, _, _, _, _);

    relation ppxor(Address, Symbol, Symbol);
    ppxor(addr, dst, src)<--
        instruction(addr, _, _, "PXOR", src, dst, _, _, _, _);
    ppxor(addr, dst, src)<--
        instruction(addr, _, _, "VPXOR", src, dst, _, _, _, _);

    relation pcvtsi2sd(Address, Symbol, Symbol);
    pcvtsi2sd(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if mnemonic.starts_with("CVTSI2SD") || mnemonic.starts_with("VCVTSI2SD");

    relation pcvtsd2si(Address, Symbol, Symbol);
    pcvtsd2si(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if mnemonic.starts_with("CVTSD2SI") || mnemonic.starts_with("CVTTSD2SI")
           || mnemonic.starts_with("VCVTSD2SI") || mnemonic.starts_with("VCVTTSD2SI");

    relation paddsd(Address, Symbol, Symbol);
    paddsd(addr, dst, src)<--
        instruction(addr, _, _, "ADDSD", src, dst, _, _, _, _);

    relation psubsd(Address, Symbol, Symbol);
    psubsd(addr, dst, src)<--
        instruction(addr, _, _, "SUBSD", src, dst, _, _, _, _);

    relation pmulsd(Address, Symbol, Symbol);
    pmulsd(addr, dst, src)<--
        instruction(addr, _, _, "MULSD", src, dst, _, _, _, _);

    relation pdivsd(Address, Symbol, Symbol);
    pdivsd(addr, dst, src)<--
        instruction(addr, _, _, "DIVSD", src, dst, _, _, _, _);

    relation pucomisd(Address, Symbol, Symbol);
    pucomisd(addr, dst, src)<--
        instruction(addr, _, _, "UCOMISD", src, dst, _, _, _, _);
    pucomisd(addr, dst, src)<--
        instruction(addr, _, _, "COMISD", src, dst, _, _, _, _);
    pucomisd(addr, dst, src)<--
        instruction(addr, _, _, "VUCOMISD", src, dst, _, _, _, _);
    pucomisd(addr, dst, src)<--
        instruction(addr, _, _, "VCOMISD", src, dst, _, _, _, _);

    relation pucomiss(Address, Symbol, Symbol);
    pucomiss(addr, dst, src)<--
        instruction(addr, _, _, "UCOMISS", src, dst, _, _, _, _);
    pucomiss(addr, dst, src)<--
        instruction(addr, _, _, "COMISS", src, dst, _, _, _, _);
    pucomiss(addr, dst, src)<--
        instruction(addr, _, _, "VUCOMISS", src, dst, _, _, _, _);
    pucomiss(addr, dst, src)<--
        instruction(addr, _, _, "VCOMISS", src, dst, _, _, _, _);

    relation paddss(Address, Symbol, Symbol);
    paddss(addr, dst, src)<--
        instruction(addr, _, _, "ADDSS", src, dst, _, _, _, _);

    relation psubss(Address, Symbol, Symbol);
    psubss(addr, dst, src)<--
        instruction(addr, _, _, "SUBSS", src, dst, _, _, _, _);

    relation pmulss(Address, Symbol, Symbol);
    pmulss(addr, dst, src)<--
        instruction(addr, _, _, "MULSS", src, dst, _, _, _, _);

    relation pdivss(Address, Symbol, Symbol);
    pdivss(addr, dst, src)<--
        instruction(addr, _, _, "DIVSS", src, dst, _, _, _, _);

    relation pcvtsd2ss(Address, Symbol, Symbol);
    pcvtsd2ss(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if *mnemonic == "CVTSD2SS" || *mnemonic == "VCVTSD2SS";

    relation pcvtss2sd(Address, Symbol, Symbol);
    pcvtss2sd(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if *mnemonic == "CVTSS2SD" || *mnemonic == "VCVTSS2SD";

    relation pcvtsi2ss(Address, Symbol, Symbol);
    pcvtsi2ss(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if mnemonic.starts_with("CVTSI2SS") || mnemonic.starts_with("VCVTSI2SS");

    relation pcvtss2si(Address, Symbol, Symbol);
    pcvtss2si(addr, dst, src)<--
        instruction(addr, _, _, mnemonic, src, dst, _, _, _, _),
        if mnemonic.starts_with("CVTSS2SI") || mnemonic.starts_with("CVTTSS2SI")
           || mnemonic.starts_with("VCVTSS2SI") || mnemonic.starts_with("VCVTTSS2SI");

    relation pxorpd(Address, Symbol, Symbol);
    pxorpd(addr, dst, src)<--
        instruction(addr, _, _, "XORPD", src, dst, _, _, _, _);
    pxorpd(addr, dst, src)<--
        instruction(addr, _, _, "VXORPD", src, dst, _, _, _, _);

    relation pandpd(Address, Symbol, Symbol);
    pandpd(addr, dst, src)<--
        instruction(addr, _, _, "ANDPD", src, dst, _, _, _, _);
    pandpd(addr, dst, src)<--
        instruction(addr, _, _, "VANDPD", src, dst, _, _, _, _);

    relation pxorps(Address, Symbol, Symbol);
    pxorps(addr, dst, src)<--
        instruction(addr, _, _, "XORPS", src, dst, _, _, _, _);
    pxorps(addr, dst, src)<--
        instruction(addr, _, _, "VXORPS", src, dst, _, _, _, _);

    relation pandps(Address, Symbol, Symbol);
    pandps(addr, dst, src)<--
        instruction(addr, _, _, "ANDPS", src, dst, _, _, _, _);
    pandps(addr, dst, src)<--
        instruction(addr, _, _, "VANDPS", src, dst, _, _, _, _);

    relation pmaxsd(Address, Symbol, Symbol);
    pmaxsd(addr, dst, src)<--
        instruction(addr, _, _, "MAXSD", src, dst, _, _, _, _);
    pmaxsd(addr, dst, src)<--
        instruction(addr, _, _, "VMAXSD", src, dst, _, _, _, _);

    relation pminsd(Address, Symbol, Symbol);
    pminsd(addr, dst, src)<--
        instruction(addr, _, _, "MINSD", src, dst, _, _, _, _);
    pminsd(addr, dst, src)<--
        instruction(addr, _, _, "VMINSD", src, dst, _, _, _, _);

    relation pmovss(Address, Symbol, Symbol);
    pmovss(addr, dst, src)<--
        instruction(addr, _, _, "MOVSS", src, dst, _, _, _, _);
    pmovss(addr, dst, src)<--
        instruction(addr, _, _, "VMOVSS", src, dst, _, _, _, _);

    relation pswap(Address, Symbol);
    pswap(addr, dst)<--
        instruction(addr, _, _, "BSWAP", dst, _, _, _, _, _);

    relation plabel(Address, Symbol);


    relation phlt(Address);
    phlt(addr) <--
        instruction(addr, _, _, "HLT", _, _, _, _, _, _);

    relation pint2c(Address);
    pint2c(addr) <--
        instruction(addr, _, _, "INT", vector, _, _, _, _, _),
        op_immediate(vector, value, _),
        if *value == 0x2c;

    relation ptest(Address, Symbol, Symbol);
    ptest(addr, dst, src) <--
        instruction(addr, _, _, "TEST", src, dst, _, _, _, _);

    // TEST reg, reg is semantically CMP reg, 0: emit pcmp with a synthesized zero immediate
    pcmp(addr, dst, zero_sym), op_immediate(zero_sym, 0, 0) <--
        ptest(addr, dst, src),
        op_register(dst, r1),
        op_register(src, r2),
        if r1 == r2,
        let zero_sym = Box::leak(format!("__test_zero_{:x}", addr).into_boxed_str()) as &'static str;

    relation pbsr(Address, Symbol, Symbol);
    pbsr(addr, dst, src) <--
        instruction(addr, _, _, "BSR", src, dst, _, _, _, _);

    relation pbsf(Address, Symbol, Symbol);
    pbsf(addr, dst, src) <--
        instruction(addr, _, _, "BSF", src, dst, _, _, _, _);

    relation psqrt(Address, Symbol, Symbol);
    psqrt(addr, dst, src) <--
        instruction(addr, _, _, "SQRTSD", src, dst, _, _, _, _);
    psqrt(addr, dst, src) <--
        instruction(addr, _, _, "VSQRTSD", src, dst, _, _, _, _);

    relation psqrtss(Address, Symbol, Symbol);
    psqrtss(addr, dst, src) <--
        instruction(addr, _, _, "SQRTSS", src, dst, _, _, _, _);
    psqrtss(addr, dst, src) <--
        instruction(addr, _, _, "VSQRTSS", src, dst, _, _, _, _);

    relation psqrtsd(Address, Symbol, Symbol);
    psqrtsd(addr, res, a1) <--
        instruction(addr, _, _, "SQRTPD", a1, res, _, _, _, _);
    psqrtsd(addr, res, a1) <--
        instruction(addr, _, _, "VSQRTPD", a1, res, _, _, _, _);

    relation padc(Address, Symbol, Symbol);
    padc(addr, dst, src)<--
        instruction(addr, _, _, "ADC", src, dst, _, _, _, _);

    relation psbb(Address, Symbol, Symbol);
    psbb(addr, res, a1)<--
        instruction(addr, _, _, "SBB", a1, res, _, _, _, _);

    relation pjmp(Address, Symbol);
    pjmp(addr, dst)<--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondGe, dst)<--
        instruction(addr, _, _, "JNS", dst, _, _, _, _, _);

    pjcc(addr, TestCond::CondL, dst)<--
        instruction(addr, _, _, "JS", dst, _, _, _, _, _);

    relation plea(Address, Symbol, Symbol);
    plea(addr, dst, src)<--
        instruction(addr, _, _, "LEA", src, dst, _, _, _, _);

    pxor(addr, dst, src)<--
        instruction(addr, _, _, "PXOR", src, dst, _, _, _, _);

    relation unrefinedinstruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation symbol_table(Address, usize, Symbol, Symbol, Symbol, usize, Symbol, usize, Symbol);
    relation plt_entry(Address, Symbol);
    relation plt_block(Address, Symbol);
    relation code_in_refined_block(Address, Address);
    relation code_pointer_in_data(Address, Address);
    // A data slot holding a pointer to a DATA target, from ELF relocations or non-PIE section bytes; the slot is provably a pointer with initializer &target.
    relation pointer_in_data(Address, Address);
    // [start, end) ranges of allocated data sections (pushed by the loader), used to distinguish a genuine no-base absolute/scaled-index data address from a small arithmetic displacement (e.g. `lea 0x8(,%rsi,8)`).
    relation data_section_range(Address, Address);
    // Read-only addresses holding an IEEE-754 sign-clear mask with its width, the positive evidence that lets the reg-reg andps/andpd form lower to Oabsfs/Oabsf.
    relation fp_sign_clear_mask(Address, usize);
    relation pointer_to_external_symbol(Address, Symbol);
    relation global_symbol(Address, Symbol);
    // A global_symbol row that is a synthesized placeholder (SUB_/TABLE_), used to keep a rip-relative indirect call through a mutable function-pointer slot from resolving to a bogus direct call.
    relation synthesized_global_symbol(Address);
    relation arch_bit(i64);
    relation trim_instruction(Address);
    relation vla_alloca(Address, Mreg, Mreg);

    relation mach_inst(Address, MachInst);
    relation func_stacksz(Address, Address, Symbol, u64);
    relation func_span(Symbol, Address, Address);
    relation resolved_addr_to_symbol(Address, Ident, i64);

    relation type_size(Typ, u64);
    relation ireg_hold_type(String, Typ);
    relation reg_64(&'static str);
    relation reg_is_64(&'static str, bool);
    relation reg_cx(&'static str);
    relation reg_sp(&'static str);
    relation reg_bp(&'static str);
    relation reg_ip(&'static str);
    relation reg_xmm(&'static str);
    relation type_to_memchunk(Typ, MemoryChunk);
    relation instruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation instruction_address_size(Address, u8);
    relation rip_target_addr(Address, Address);
    relation abs_target_addr(Address, Address);
    relation is_external_function(Address);
    relation function_symbol(Address, Symbol);
    relation prev_instr(Address, Address);
    relation is_tail_call_jmp(Address);
    relation transl_store_inferred(Address, MemoryChunk, Addrmode, Arc<Vec<Mreg>>, Mreg);
    relation transl_store(Address, MemoryChunk, Addrmode, Arc<Vec<Mreg>>, Mreg);
    relation addrmode_needs_resolution(Address, Addrmode, Address);
    relation addr_requiring_symbol(Address);
    relation mach_imm_stack_init(Address, i64, i64, Typ);
    // Carries the size-narrowed MemoryChunk, not a Typ: Typ cannot represent MInt8/MInt16, so an immediate byte store over-widened to 4 bytes and mis-scaled its pointer arithmetic.
    relation mach_imm_indirect_store(Address, i64, MemoryChunk, Mreg, i64);
    // Memory-source arithmetic: load from [base+disp], then apply op with dst_reg.
    relation arith_load_op(Address, Operation, MemoryChunk, Mreg, i64, Mreg);
    // Memory-dest arithmetic with register source: load [base+disp], op with src_reg, store back.
    relation arith_store_reg(Address, Operation, MemoryChunk, Mreg, i64, Mreg);
    // Memory-dest arithmetic with immediate source: load [base+disp], op_imm, store back.
    relation arith_store_imm(Address, Operation, MemoryChunk, Mreg, i64);
    // Absolute addressing arith store: read-modify-write at global symbol address.
    relation arith_store_abs_reg(Address, Operation, MemoryChunk, Ident, i64, Mreg);
    relation arith_store_abs_imm(Address, Operation, MemoryChunk, Ident, i64);
    // Single source of truth for the no-linear-op side-effect class (fused mem-arith RMW, immediate indirect/absolute stores, mem-source loads), which linear_pass keeps self-canonical so a goto cannot thread past one.
    relation node_unlowered_side_effect(Address);
    node_unlowered_side_effect(addr) <-- mach_imm_indirect_store(addr, _, _, _, _);
    node_unlowered_side_effect(addr) <-- arith_load_op(addr, _, _, _, _, _);
    node_unlowered_side_effect(addr) <-- arith_store_reg(addr, _, _, _, _, _);
    node_unlowered_side_effect(addr) <-- arith_store_imm(addr, _, _, _, _);
    node_unlowered_side_effect(addr) <-- arith_store_abs_reg(addr, _, _, _, _, _);
    node_unlowered_side_effect(addr) <-- arith_store_abs_imm(addr, _, _, _, _);
    node_unlowered_side_effect(addr) <-- float_load_op(addr, _, _, _, _, _, _);
    node_unlowered_side_effect(addr) <-- stack_unary_load_op(addr, _, _, _, _);
    node_unlowered_side_effect(addr) <-- float_arith_stack_op(addr, _, _, _, _);
    relation transl_load_inferred(Address, MemoryChunk, Addrmode, Arc<Vec<Mreg>>, Mreg);
    relation transl_load(Address, MemoryChunk, Addrmode, Arc<Vec<Mreg>>, Mreg);
    relation expand_builtin_inline(Address, Symbol, Arc<Vec<BuiltinArg<Mreg>>>, BuiltinArg<Mreg>);
    relation expand_builtin_va_start_32(Address, Ireg);
    relation pallocframe(Address, i64);
    relation pallocframe_by_func(Symbol, Address, i64);

    type_size(Typ::Tint, 4);
    type_size(Typ::Tfloat, 8);
    type_size(Typ::Tsingle, 4);
    type_size(Typ::Tany32, 4);

    type_size(Typ::Tlong, 8) <-- arch_bit(64);
    type_size(Typ::Tany64, 8) <-- arch_bit(64);

    // 64-bit registers use Tany64 (not Tlong) to let downstream type inference determine the actual type from usage context.
    ireg_hold_type("RAX".to_string(), Typ::Tany64);
    ireg_hold_type("RBX".to_string(), Typ::Tany64);
    ireg_hold_type("RCX".to_string(), Typ::Tany64);
    ireg_hold_type("RDX".to_string(), Typ::Tany64);
    ireg_hold_type("RSI".to_string(), Typ::Tany64);
    ireg_hold_type("RDI".to_string(), Typ::Tany64);
    ireg_hold_type("RBP".to_string(), Typ::Tany64);
    ireg_hold_type("RSP".to_string(), Typ::Tany64);
    ireg_hold_type("R8".to_string(), Typ::Tany64);
    ireg_hold_type("R9".to_string(), Typ::Tany64);
    ireg_hold_type("R10".to_string(), Typ::Tany64);
    ireg_hold_type("R11".to_string(), Typ::Tany64);
    ireg_hold_type("R12".to_string(), Typ::Tany64);
    ireg_hold_type("R13".to_string(), Typ::Tany64);
    ireg_hold_type("R14".to_string(), Typ::Tany64);
    ireg_hold_type("R15".to_string(), Typ::Tany64);
    reg_64("RAX"); reg_64("RBX"); reg_64("RCX"); reg_64("RDX");
    reg_64("RSI"); reg_64("RDI"); reg_64("RBP"); reg_64("RSP");
    reg_64("R8");  reg_64("R9");  reg_64("R10"); reg_64("R11");
    reg_64("R12"); reg_64("R13"); reg_64("R14"); reg_64("R15");
    reg_is_64("RAX", true); reg_is_64("RBX", true); reg_is_64("RCX", true); reg_is_64("RDX", true);
    reg_is_64("RSI", true); reg_is_64("RDI", true); reg_is_64("RBP", true); reg_is_64("RSP", true);
    reg_is_64("R8", true);  reg_is_64("R9", true);  reg_is_64("R10", true); reg_is_64("R11", true);
    reg_is_64("R12", true); reg_is_64("R13", true); reg_is_64("R14", true); reg_is_64("R15", true);
    reg_is_64("EAX", false); reg_is_64("EBX", false); reg_is_64("ECX", false); reg_is_64("EDX", false);
    reg_is_64("ESI", false); reg_is_64("EDI", false); reg_is_64("EBP", false); reg_is_64("ESP", false);
    reg_is_64("R8D", false); reg_is_64("R9D", false); reg_is_64("R10D", false); reg_is_64("R11D", false);
    reg_is_64("R12D", false); reg_is_64("R13D", false); reg_is_64("R14D", false); reg_is_64("R15D", false);
    // 16- and 8-bit sub-registers are sub-64-bit: without these facts every reg_is_64 join silently fails for a byte operand, so the CMP+CMOV/SETcc Osel never forms and the value chain is DCE'd.
    reg_is_64("AX", false); reg_is_64("BX", false); reg_is_64("CX", false); reg_is_64("DX", false);
    reg_is_64("SI", false); reg_is_64("DI", false); reg_is_64("BP", false); reg_is_64("SP", false);
    reg_is_64("R8W", false); reg_is_64("R9W", false); reg_is_64("R10W", false); reg_is_64("R11W", false);
    reg_is_64("R12W", false); reg_is_64("R13W", false); reg_is_64("R14W", false); reg_is_64("R15W", false);
    reg_is_64("AL", false); reg_is_64("BL", false); reg_is_64("CL", false); reg_is_64("DL", false);
    reg_is_64("AH", false); reg_is_64("BH", false); reg_is_64("CH", false); reg_is_64("DH", false);
    reg_is_64("SIL", false); reg_is_64("DIL", false); reg_is_64("BPL", false); reg_is_64("SPL", false);
    reg_is_64("R8B", false); reg_is_64("R9B", false); reg_is_64("R10B", false); reg_is_64("R11B", false);
    reg_is_64("R12B", false); reg_is_64("R13B", false); reg_is_64("R14B", false); reg_is_64("R15B", false);
    reg_cx("RCX"); reg_cx("ECX"); reg_cx("CX"); reg_cx("CL"); reg_cx("CH");
    // Stack claims are valid only for 64-bit addressing. E*/word/byte spellings
    // are ordinary address-size/subregister operations in x86-64, not aliases
    // for frame analysis.
    reg_sp("RSP");
    reg_bp("RBP");
    reg_ip("RIP"); reg_ip("EIP"); reg_ip("IP");
    reg_xmm("XMM0"); reg_xmm("XMM1"); reg_xmm("XMM2"); reg_xmm("XMM3");
    reg_xmm("XMM4"); reg_xmm("XMM5"); reg_xmm("XMM6"); reg_xmm("XMM7");
    reg_xmm("XMM8"); reg_xmm("XMM9"); reg_xmm("XMM10"); reg_xmm("XMM11");
    reg_xmm("XMM12"); reg_xmm("XMM13"); reg_xmm("XMM14"); reg_xmm("XMM15");
    // YMM registers alias to their XMM counterparts for scalarization
    reg_xmm("YMM0"); reg_xmm("YMM1"); reg_xmm("YMM2"); reg_xmm("YMM3");
    reg_xmm("YMM4"); reg_xmm("YMM5"); reg_xmm("YMM6"); reg_xmm("YMM7");
    reg_xmm("YMM8"); reg_xmm("YMM9"); reg_xmm("YMM10"); reg_xmm("YMM11");
    reg_xmm("YMM12"); reg_xmm("YMM13"); reg_xmm("YMM14"); reg_xmm("YMM15");
    ireg_hold_type("EAX".to_string(), Typ::Tint);
    ireg_hold_type("EBX".to_string(), Typ::Tint);
    ireg_hold_type("ECX".to_string(), Typ::Tint);
    ireg_hold_type("EDX".to_string(), Typ::Tint);
    ireg_hold_type("ESI".to_string(), Typ::Tint);
    ireg_hold_type("EDI".to_string(), Typ::Tint);
    ireg_hold_type("EBP".to_string(), Typ::Tint);
    ireg_hold_type("ESP".to_string(), Typ::Tint);
    ireg_hold_type("R8D".to_string(), Typ::Tint);
    ireg_hold_type("R9D".to_string(), Typ::Tint);
    ireg_hold_type("R10D".to_string(), Typ::Tint);
    ireg_hold_type("R11D".to_string(), Typ::Tint);
    ireg_hold_type("R12D".to_string(), Typ::Tint);
    ireg_hold_type("R13D".to_string(), Typ::Tint);
    ireg_hold_type("R14D".to_string(), Typ::Tint);
    ireg_hold_type("R15D".to_string(), Typ::Tint);
    // XMM registers use Tany64, not Tfloat: floatness is a property of the INSTRUCTION, and gcc uses XMM as 16-byte block movers, so Tfloat made struct recovery emit double fields for pointer data.
    ireg_hold_type("XMM0".to_string(), Typ::Tany64);
    ireg_hold_type("XMM1".to_string(), Typ::Tany64);
    ireg_hold_type("XMM2".to_string(), Typ::Tany64);
    ireg_hold_type("XMM3".to_string(), Typ::Tany64);
    ireg_hold_type("XMM4".to_string(), Typ::Tany64);
    ireg_hold_type("XMM5".to_string(), Typ::Tany64);
    ireg_hold_type("XMM6".to_string(), Typ::Tany64);
    ireg_hold_type("XMM7".to_string(), Typ::Tany64);
    ireg_hold_type("XMM8".to_string(), Typ::Tany64);
    ireg_hold_type("XMM9".to_string(), Typ::Tany64);
    ireg_hold_type("XMM10".to_string(), Typ::Tany64);
    ireg_hold_type("XMM11".to_string(), Typ::Tany64);
    ireg_hold_type("XMM12".to_string(), Typ::Tany64);
    ireg_hold_type("XMM13".to_string(), Typ::Tany64);
    ireg_hold_type("XMM14".to_string(), Typ::Tany64);
    ireg_hold_type("XMM15".to_string(), Typ::Tany64);
    ireg_hold_type("FP0".to_string(), Typ::Tfloat);
    ireg_hold_type("FP1".to_string(), Typ::Tfloat);
    ireg_hold_type("FP2".to_string(), Typ::Tfloat);
    ireg_hold_type("FP3".to_string(), Typ::Tfloat);
    ireg_hold_type("FP4".to_string(), Typ::Tfloat);
    ireg_hold_type("FP5".to_string(), Typ::Tfloat);
    ireg_hold_type("FP6".to_string(), Typ::Tfloat);
    ireg_hold_type("FP7".to_string(), Typ::Tfloat);
    ireg_hold_type("AX".to_string(), Typ::Tint);
    ireg_hold_type("BX".to_string(), Typ::Tint);
    ireg_hold_type("CX".to_string(), Typ::Tint);
    ireg_hold_type("DX".to_string(), Typ::Tint);
    ireg_hold_type("SI".to_string(), Typ::Tint);
    ireg_hold_type("DI".to_string(), Typ::Tint);
    ireg_hold_type("BP".to_string(), Typ::Tint);
    ireg_hold_type("SP".to_string(), Typ::Tint);
    ireg_hold_type("R8W".to_string(), Typ::Tint);
    ireg_hold_type("R9W".to_string(), Typ::Tint);
    ireg_hold_type("R10W".to_string(), Typ::Tint);
    ireg_hold_type("R11W".to_string(), Typ::Tint);
    ireg_hold_type("R12W".to_string(), Typ::Tint);
    ireg_hold_type("R13W".to_string(), Typ::Tint);
    ireg_hold_type("R14W".to_string(), Typ::Tint);
    ireg_hold_type("R15W".to_string(), Typ::Tint);
    ireg_hold_type("AL".to_string(), Typ::Tint);
    ireg_hold_type("BL".to_string(), Typ::Tint);
    ireg_hold_type("CL".to_string(), Typ::Tint);
    ireg_hold_type("DL".to_string(), Typ::Tint);
    ireg_hold_type("SIL".to_string(), Typ::Tint);
    ireg_hold_type("DIL".to_string(), Typ::Tint);
    ireg_hold_type("BPL".to_string(), Typ::Tint);
    ireg_hold_type("SPL".to_string(), Typ::Tint);
    ireg_hold_type("R8B".to_string(), Typ::Tint);
    ireg_hold_type("R9B".to_string(), Typ::Tint);
    ireg_hold_type("R10B".to_string(), Typ::Tint);
    ireg_hold_type("R11B".to_string(), Typ::Tint);
    ireg_hold_type("R12B".to_string(), Typ::Tint);
    ireg_hold_type("R13B".to_string(), Typ::Tint);
    ireg_hold_type("R14B".to_string(), Typ::Tint);
    ireg_hold_type("R15B".to_string(), Typ::Tint);
    ireg_hold_type("AH".to_string(), Typ::Tint);
    ireg_hold_type("BH".to_string(), Typ::Tint);
    ireg_hold_type("CH".to_string(), Typ::Tint);
    ireg_hold_type("DH".to_string(), Typ::Tint);

    type_to_memchunk(Typ::Tint, MemoryChunk::MInt32);
    type_to_memchunk(Typ::Tlong, MemoryChunk::MInt64);
    type_to_memchunk(Typ::Tfloat, MemoryChunk::MFloat64);
    type_to_memchunk(Typ::Tsingle, MemoryChunk::MFloat32);
    type_to_memchunk(Typ::Tany32, MemoryChunk::MAny32);
    type_to_memchunk(Typ::Tany64, MemoryChunk::MAny64);

    resolved_addr_to_symbol(addr, ident, 0) <--
        symbols(addr, _, _),
        let ident = *addr as Ident;

    #[local] relation nonzero_symbol(Address);
    nonzero_symbol(addr) <--
        symbols(addr, _, _),
        if *addr != 0;

    // A nonzero symbol address that is a genuine reference target (data object or string label), not code; gates immediate->&symbol so an imm merely colliding with a code address stays an integer.
    #[local] relation immediate_nonzero_data_symbol(Address);
    immediate_nonzero_data_symbol(addr) <--
        nonzero_symbol(addr),
        !function_symbol(addr, _);

    resolved_addr_to_symbol(immediate_addr, ident, offset) <--
        addr_requiring_symbol(immediate_addr),
        func_span(_, start, end),
        if *immediate_addr >= *start,
        if *immediate_addr < *end,
        let ident = *start as Ident,
        let offset = (*immediate_addr - *start) as i64;

    resolved_addr_to_symbol(plt_addr, ident, 0) <--
        plt_entry(plt_addr, _),
        let ident = *plt_addr as Ident;


    rip_target_addr(addr, target) <--
        plea(addr, _, am),
        instruction(addr, size, _, _, _, _, _, _, _, _),
        op_indirect(am, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    rip_target_addr(addr, target) <--
        pmov(addr, _, am),
        instruction(addr, size, _, _, _, _, _, _, _, _),
        op_indirect(am, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    rip_target_addr(addr, target) <--
        pmov(addr, am, _),
        instruction(addr, size, _, _, _, _, _, _, _, _),
        op_indirect(am, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    // RIP-relative target for comiss/ucomiss/comisd/ucomisd with memory operand
    rip_target_addr(addr, target) <--
        pucomiss(addr, _, src),
        instruction(addr, size, _, _, _, _, _, _, _, _),
        op_indirect(src, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    rip_target_addr(addr, target) <--
        pucomisd(addr, _, src),
        instruction(addr, size, _, _, _, _, _, _, _, _),
        op_indirect(src, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    // Generic RIP-relative catch-all for instructions with a RIP-relative operand.
    rip_target_addr(addr, target) <--
        instruction(addr, size, _, _, op1, _, _, _, _, _),
        op_indirect(op1, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    rip_target_addr(addr, target) <--
        instruction(addr, size, _, _, _, op2, _, _, _, _),
        op_indirect(op2, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        let target = (*addr as i64 + *size as i64 + disp) as Address;

    // Absolute addressing (clang -fno-pie): base="NONE", no index, displacement IS the address
    abs_target_addr(addr, target) <--
        pmov(addr, _, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pmov(addr, am, _),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        plea(addr, _, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pucomiss(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pucomisd(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    // Absolute addressing for padd/psub (arith_store patterns)
    abs_target_addr(addr, target) <--
        padd(addr, dst, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        psub(addr, dst, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    // Absolute addressing for padd/psub memory-SOURCE direction (`add 0x404024,%eax`, 3.8c)
    abs_target_addr(addr, target) <--
        padd(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        psub(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pand(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        por(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pxor(addr, _, src),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    // Absolute addressing for the float/SSE memory-source op family (addsd/mulss/... 0x..., 3.8a)
    abs_target_addr(addr, target) <--
        float_mem_op_raw(addr, _, _, src, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    // Scaled-index absolute addressing (clang const-table load): the displacement must land in a real data section, since a small arithmetic displacement is an index computation, not a table base.
    abs_target_addr(addr, target) <--
        pmov(addr, _, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address,
        data_section_range(start, end),
        if target >= *start && target < *end;

    abs_target_addr(addr, target) <--
        pmov(addr, am, _),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address,
        data_section_range(start, end),
        if target >= *start && target < *end;

    abs_target_addr(addr, target) <--
        plea(addr, _, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address,
        data_section_range(start, end),
        if target >= *start && target < *end;

    resolved_addr_to_symbol(target, ident, 0) <--
        rip_target_addr(_, target),
        symbols(target, _, _),
        let ident = *target as Ident;

    resolved_addr_to_symbol(target, ident, offset) <--
        rip_target_addr(_, target),
        func_span(_, start, end),
        if *target >= *start && *target < *end,
        let ident = *start as Ident,
        let offset = (target - start) as i64;

    // Symbol resolution for absolute addressing targets
    resolved_addr_to_symbol(target, ident, 0) <--
        abs_target_addr(_, target),
        symbols(target, _, _),
        let ident = *target as Ident;

    resolved_addr_to_symbol(target, ident, offset) <--
        abs_target_addr(_, target),
        func_span(_, start, end),
        if *target >= *start && *target < *end,
        let ident = *start as Ident,
        let offset = (target - start) as i64;


    plt_entry(addr, name) <--
        symbol_table(addr, _, _, _, _, _, section_name, _, name),
        if *section_name == ".plt" || *section_name == ".plt.got";

    is_external_function(addr) <-- plt_entry(addr, _);
    is_external_function(addr) <-- plt_block(addr, _);

    is_external_function(addr) <--
        symbol_table(addr, _, sym_type, _, _, _, section_name, _, _),
        if *sym_type == "FUNC",
        if *section_name == ".init" || *section_name == ".fini";

    function_symbol(addr, name) <--
        symbol_table(addr, _size, _type, _binding, _section_type, _section_idx, _section_name, _name_idx, name),
        if *_binding == "GLOBAL" || *_binding == "LOCAL",
        if *_type == "FUNC",
        if *addr > 0;

    function_symbol(addr, name) <--
        plt_entry(addr, name);

    function_symbol(addr, name) <--
        plt_block(addr, name);

    global_symbol(addr, name) <--
        symbol_table(addr, _size, _type, _binding, _section_type, _section_idx, _section_name, _name_idx, name),
        if *_binding == "GLOBAL" || *_binding == "LOCAL",
        if *_type == "OBJECT",
        if *addr > 0;

    global_symbol(target_addr, name_sym), synthesized_global_symbol(target_addr) <--
        rip_target_addr(_, target_addr),
        if *target_addr > 0,
        !symbol_table(target_addr, _, _, _, _, _, _, _, _),
        !function_symbol(target_addr, _),
        !plt_entry(target_addr, _),
        !func_entry(_, target_addr),
        let name_string = format!("SUB_{:x}", target_addr),
        let name_sym = Box::leak(name_string.into_boxed_str()) as &'static str;

    // Synthesise a global symbol for absolute-addressed targets (clang -fno-pie const-table loads) so the table base resolves to Aglobal instead of `&0`.
    global_symbol(target_addr, name_sym), synthesized_global_symbol(target_addr) <--
        abs_target_addr(_, target_addr),
        if *target_addr > 0,
        !symbol_table(target_addr, _, _, _, _, _, _, _, _),
        !function_symbol(target_addr, _),
        !plt_entry(target_addr, _),
        !func_entry(_, target_addr),
        let name_string = format!("TABLE_{:x}", target_addr),
        let name_sym = Box::leak(name_string.into_boxed_str()) as &'static str;

    symbols(addr, name, name) <--
        function_symbol(addr, name);

    symbols(addr, name, name) <--
        global_symbol(addr, name);


    instruction(addr, size, mnemonic, dst, src1, src2, src3, src4, prefix, suffix) <--
        unrefinedinstruction(addr, size, mnemonic, dst, src1, src2, src3, src4, prefix, suffix),
        code_in_refined_block(addr, _),
        !trim_instruction(addr);

    mach_inst(addr, MachInst::Mbuiltin(
        "alloca".to_string(),
        vec![BuiltinArg::BA(*size_mreg)],
        BuiltinArg::BA(*result_mreg),
    )) <--
        vla_alloca(addr, size_mreg, result_mreg);

    mach_inst(addr, MachInst::Mcall(Either::Left(Mreg::x86(Ireg::from(reg_str))))) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_register(dst, reg_str);

    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Right(*imm_str)))) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_immediate(dst, imm_str, _),
        !symbols(*imm_str as Address, _, _);

    // Resolve `call *disp(%rip)` via rip_target_addr to the GOT slot, then to an extern symbol or named symbol, emitting a named direct call so the known extern signature applies (RIP has no GP-reg).
    relation rip_indirect_call_resolved(Address);

    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Left(name)))),
    rip_indirect_call_resolved(*addr) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_indirect(dst, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target),
        pointer_to_external_symbol(target, name);

    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Left(name)))),
    rip_indirect_call_resolved(*addr) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_indirect(dst, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target),
        !pointer_to_external_symbol(target, _),
        // Only a REAL named target may become a direct call; a synthesized SUB_/TABLE_ placeholder means a mutable function-pointer slot, so exclude it and fall through to the indirect path.
        !synthesized_global_symbol(*target),
        symbols(*target, name, _);

    // Register-indirect (`call *%rax`) and unresolved RIP-relative calls fall back to Mcall(Left(base)); resolvable RIP slots are handled above and excluded here so no call is dropped.
    mach_inst(addr, MachInst::Mcall(Either::Left(Mreg::x86(Ireg::from(base_str))))) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_indirect(dst, _, base_str, _, _, _, _),
        if base_str != &"NONE",
        !rip_indirect_call_resolved(*addr);


    prev_instr(next_addr, curr_addr) <-- next(curr_addr, next_addr);

    mach_inst(addr, MachInst::Mtailcall(Either::Left(Mreg::x86(Ireg::from(reg_str))))) <--
        padd(addr, rsp, imm),
        op_immediate(imm, _, _),
        op_register(rsp, "RSP"),
        next(addr, addr1),
        pjmp(addr1, dst),
        op_register(dst, reg_str);

    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Right(*imm_str)))) <--
        padd(addr, rsp, imm),
        op_immediate(imm, _, _),
        op_register(rsp, "RSP"),
        next(addr, addr1),
        pjmp(addr1, dst),
        op_immediate(dst, imm_str, _),
        // Require a real function-entry target: an `L_<addr>` label (e.g. shared epilogue) is not a tailcall and would fabricate a call to an undefined `L_<addr>()` (S5).
        func_entry(_, *imm_str as u64);


    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(sym)))), plabel(addr1, sym) <--
        padd(addr, rsp, imm),
        op_immediate(imm, _, _),
        op_register(rsp, "RSP"),
        next(addr, addr1),
        pjmp(addr1, sym),
        op_indirect(sym, _, _, _, _, _, _),
        // A RIP-relative PE/COFF import jump is already a fully resolved
        // named tailcall at the JMP node.  Converting the preceding stack
        // restore as well fabricates a second callee from the local operand
        // label (for example `_21`) and lets selection discard the relocation.
        !is_extern_tailcall_jmp(addr1);

    is_tail_call_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        func_entry(_, target_addr_addr),
        if *target_addr as u64 == *target_addr_addr,
        // ASM-4: a JMP to its own function's entry is an intra-function jump, not a tail call; excluded via structural CFG containment so it becomes Mgoto.
        !jmp_target_in_own_func(addr);

    is_tail_call_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        instr_in_function(addr, func),
        next(addr, next_addr),
        !instr_in_function(next_addr, func),
        // Exclude intra-function jumps (loop back-edges at function boundaries are not tail calls).
        !instr_in_function(*target_addr as u64, func),
        // Require a real function-entry target: an `L_<addr>` label (e.g. landing-pad continuation) is not a tailcall and lets the JMP fall through to Mgoto instead of an undefined `L_<addr>()` (S5).
        func_entry(_, *target_addr as u64);

    is_tail_call_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        instr_in_function(addr, func),
        func_entry(_, target_func),
        if *target_addr as u64 == *target_func,
        if *target_func != *func;

    is_tail_call_jmp(addr) <--
        instruction(addr, _, _, "JMP", _, _, _, _, _, _),
        plt_entry(addr, _);

    // Forwarder thunk (mov args; jmp <plt_stub>): the stub is not a func_entry, so classify a JMP-to-extern as its own tail-call kind or the body collapses to goto <sym> and the function is dropped.
    // PE import thunks end in jmp qword ptr [rip+IAT]; resolve the slot like the indirect CALL path so the thunk becomes a named tail call and inherits the extern signature.
    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(name)))),
    is_extern_tailcall_jmp(*addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_indirect(dst, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target),
        pointer_to_external_symbol(target, name);

    is_extern_tailcall_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        plt_block(*target_addr as u64, _);
    is_extern_tailcall_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        plt_entry(*target_addr as u64, _);
    is_extern_tailcall_jmp(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        is_external_function(*target_addr as u64);

    // A direct JMP landing back inside its own function is an intra-function jump, never a tail call; exposed so the mach-level restore-then-goto heuristic cannot fabricate an L_<addr>() call.
    relation jmp_target_in_own_func(Address);
    jmp_target_in_own_func(addr) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        instr_in_function(addr, func),
        instr_in_function(*target_addr as u64, func);

    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(sym)))) <--
        instruction(addr, _, _, "JMP", _, _, _, _, _, _),
        plt_entry(addr, sym);

    // Thunk JMP-to-PLT-stub: emit a named tail call to the extern (symbol form), mirroring the direct-PLT-entry rule above so the same extern signature applies. The stub's plt_block/plt_entry name IS the extern name.
    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(sym)))) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        plt_block(*target_addr as u64, sym);
    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(sym)))) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        plt_entry(*target_addr as u64, sym);
    // Thunk JMP to a non-PLT extern (.init/.fini FUNC): resolve the target address to its symbol name and emit the named tail call.
    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Left(sym)))) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        is_external_function(*target_addr as u64),
        !plt_block(*target_addr as u64, _),
        !plt_entry(*target_addr as u64, _),
        symbols(*target_addr as u64, sym, _);

    mach_inst(addr, MachInst::Mtailcall(Either::Right(Either::Right(*target_addr)))) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        op_immediate(dst, target_addr, _),
        is_tail_call_jmp(addr);

    // Mgoto fallback fires only for a genuine intra-function/structured jump: it must be neither an inter-function tail call (is_tail_call_jmp) nor a thunk JMP-to-extern (is_extern_tailcall_jmp), or the thunk body would collapse to `goto <sym>` and the function would be dropped.
    mach_inst(addr, MachInst::Mgoto(dst)) <--
        instruction(addr, _, _, "JMP", dst, _, _, _, _, _),
        !is_tail_call_jmp(addr),
        !is_extern_tailcall_jmp(addr);

    mach_inst(addr, MachInst::Mreturn) <--
        instruction(addr, _, _, "RET", _, _, _, _, _, _);

    mach_inst(addr, MachInst::Mbuiltin(
        "__readcr8".to_string(),
        vec![],
        BuiltinArg::BA(*dst_reg)
    )) <--
        pcr8_read(addr, dst_reg),
        builtins("__readcr8");

    // INT 2C is a returning software interrupt in the source control flow.
    // Keep it as an ordinary side-effecting builtin; LinearPass will retain
    // its sequential fallthrough rather than treating it as unreachable.
    mach_inst(addr, MachInst::Mbuiltin(
        "__int2c".to_string(),
        vec![],
        BuiltinArg::BAInt(0)
    )) <--
        pint2c(addr),
        builtins("__int2c");


    // No-result builtin: BAInt(0) is the canonical empty-result form (matching cminor_pass) so downstream does not synthesize a dst and render `var = __builtin_unreachable()`.
    mach_inst(addr, MachInst::Mbuiltin(
        "__builtin_unreachable".to_string(),
        vec![],
        BuiltinArg::BAInt(0)
    )) <--
        phlt(addr),
        builtins("__builtin_unreachable");


    // ireg_hold_type maps sub-registers to Tint (MInt32), so chunk must come from the capstone operand size (msize), not the register family; refine_chunk_with_size narrows and never widens.
    transl_store_inferred(addr, mc, addrmode, regs.clone(), src) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        !reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, idx_str, _scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        let regs = Arc::new(vec![*arg]),
        ireg_hold_type(src_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    // A movss/movsd store of an XMM register to memory is a FLOAT store, taking its chunk from the mnemonic, not the XMM default Tany64 that would type the field int and truncate every stored float.
    transl_store_inferred(addr, mc, addrmode, regs.clone(), src) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, idx_str, _scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        let regs = Arc::new(vec![*arg]),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // Indexed store (non-SP/non-BP base with index register)
    transl_store_inferred(addr, mc, addrmode, regs.clone(), src) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        !reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*arg, *idx_arg]),
        ireg_hold_type(src_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    // movss/movsd INDEXED store of an XMM register (float `out[i] = val`): float chunk from mnemonic.
    transl_store_inferred(addr, mc, addrmode, regs.clone(), src) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*arg, *idx_arg]),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // Scaled-index store with no base register: mov src, disp(,%idx,scale)
    transl_store_inferred(addr, mc, addrmode, regs.clone(), src) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: None,
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*idx_arg]),
        ireg_hold_type(src_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    transl_store(addr, mc, addrmode, regs, src) <--
        transl_store_inferred(addr, mc, addrmode, regs, src);

    transl_store(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(src, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(dst, _, r2, idx_str, _scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        let regs = Arc::new(vec![*arg]),
        !transl_store_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // Indexed store (mnemonic-based, non-SP/non-BP base with index register)
    transl_store(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(src, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*arg, *idx_arg]),
        !transl_store_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // Scaled-index store with no base register (mnemonic-based): mov src, disp(,%idx,scale)
    transl_store(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(src, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(dst, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: None,
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*idx_arg]),
        !transl_store_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    addrmode_needs_resolution(*addr, *am, target_addr) <--
        transl_store(addr, _, am, _, _),
        effective_address_size(addr, address_size),
        if *address_size != 4,
        if let Some(target_addr_i64) = addrmode_needs_symbol_resolution(am),
        let target_addr = target_addr_i64 as Address;

    addrmode_needs_resolution(*addr, *am, target_addr) <--
        transl_load(addr, _, am, _, _),
        effective_address_size(addr, address_size),
        if *address_size != 4,
        if let Some(target_addr_i64) = addrmode_needs_symbol_resolution(am),
        let target_addr = target_addr_i64 as Address;


    addr_requiring_symbol(target_addr) <--
        addrmode_needs_resolution(_, _, target_addr);

    addr_requiring_symbol(target_addr) <--
        rip_target_addr(_, target_addr);

    addr_requiring_symbol(target_addr) <--
        abs_target_addr(_, target_addr);
    mach_inst(addr, MachInst::Mstore(*memory_chunk, addressing, Arc::new(args), *src)) <--
        transl_store(addr, memory_chunk, addrmode, regs, src),
        effective_address_size(addr, address_size),
        addrmode_needs_resolution(addr, *addrmode, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let resolved = Some((*ident, *offset)),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(*addrmode, resolved, *address_size);

    mach_inst(addr, MachInst::Mstore(*memory_chunk, addressing, Arc::new(args), *src)) <--
        transl_store(addr, memory_chunk, addrmode, regs, src),
        effective_address_size(addr, address_size),
        !addrmode_needs_resolution(addr, *addrmode, _),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(*addrmode, None, *address_size);

    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, *typ)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        !reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RSP",
        if *idx == "NONE",
        if *msize != 1 && *msize != 2,
        rsp_frame_at(addr, _),
        ireg_hold_type(srcstr.to_string(), typ);

    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, *typ)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        !reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RBP",
        if *idx == "NONE",
        if *msize != 1 && *msize != 2,
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        ireg_hold_type(srcstr.to_string(), typ);

    // A movss/movsd store of an XMM register to a stack slot is a FLOAT store whose width is the access size, not the 64-bit hold type, so a float param spill is not widened to double.
    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, typ)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RSP",
        if *idx == "NONE",
        if *msize == 4 || *msize == 8,
        rsp_frame_at(addr, _),
        let typ = if *msize == 4 { Typ::Tsingle } else { Typ::Tfloat };

    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, typ)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RBP",
        if *idx == "NONE",
        if *msize == 4 || *msize == 8,
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        let typ = if *msize == 4 { Typ::Tsingle } else { Typ::Tfloat };

    // A wide XMM/YMM spill (msize 16 or 32) has no float semantics and no 128-bit Typ, so materialize a Tany64 Msetstack to keep the base-offset slot defined and tracked.
    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, Typ::Tany64)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RSP",
        if *idx == "NONE",
        if *msize > 8,
        rsp_frame_at(addr, _);

    mach_inst(addr, MachInst::Msetstack(src_reg, *disp, Typ::Tany64)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RBP",
        if *idx == "NONE",
        if *msize > 8,
        instr_in_function(addr, func),
        bp_frame_at(addr, func);

    // Typ has no 8/16-bit member, so sub-register frame-slot stores use Ainstack Mstore with the true byte width; rtl re-derives the Lsetstack row from Lstore(Ainstack,[]) so slot dataflow is unchanged.
    mach_inst(addr, MachInst::Mstore(mc, Addressing::Ainstack(*disp), Arc::new(vec![]), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RSP",
        if *idx == "NONE",
        if *msize == 1 || *msize == 2,
        rsp_frame_at(addr, _),
        ireg_hold_type(srcstr.to_string(), _),
        let mc = if *msize == 1 { MemoryChunk::MInt8Unsigned } else { MemoryChunk::MInt16Unsigned };

    mach_inst(addr, MachInst::Mstore(mc, Addressing::Ainstack(*disp), Arc::new(vec![]), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx, _, disp, msize),
        if *r2 == "RBP",
        if *idx == "NONE",
        if *msize == 1 || *msize == 2,
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        ireg_hold_type(srcstr.to_string(), _),
        let mc = if *msize == 1 { MemoryChunk::MInt8Unsigned } else { MemoryChunk::MInt16Unsigned };

    mach_imm_stack_init(addr, disp, imm_int, ty) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, base, idx, _, disp, sz),
        if *base == "RBP",
        if *idx == "NONE" || idx.is_empty(),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        let ty = if *sz <= 4 { Typ::Tint } else { Typ::Tany64 },
        let imm_int = *imm_sym as i64;

    // Keep the raw instruction displacement, matching Msetstack/Mgetstack and
    // disassembly stack def-use keys. Entry-frame offsets are derived only at
    // ABI boundaries, not in the local-slot identity.
    mach_imm_stack_init(addr, *disp, imm_int, ty) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, base, idx, _, disp, sz),
        if *base == "RSP",
        if *idx == "NONE" || idx.is_empty(),
        rsp_frame_at(addr, _),
        let ty = if *sz <= 4 { Typ::Tint } else { Typ::Tany64 },
        let imm_int = *imm_sym as i64;

    // RIP base excluded: Mreg::x86("RIP") is Unknown so the row could never lower; RIP-relative immediate global stores have no lifting route and are silently dropped (FIXPLAN 3.8c).
    mach_imm_indirect_store(addr, imm_int, mc, base_mreg, *disp) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, r2, _, _scale, disp, sz),
        if *r2 != "RBP" && *r2 != "RSP",
        if *r2 != "NONE" && !r2.is_empty(),
        !reg_ip(r2),
        let base_mreg = Mreg::x86(r2),
        let mc = refine_chunk_with_size(if *sz <= 4 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 }, *sz),
        let imm_int = *imm_sym as i64;

    mach_inst(addr, MachInst::Mgetstack(*disp, *typ, Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx, _, disp, _),
        if *r2 == "RSP",
        if *idx == "NONE",
        op_register(dst, dststr),
        rsp_frame_at(addr, _),
        ireg_hold_type(dststr.to_string(), typ);

    mach_inst(addr, MachInst::Mgetstack(*disp, *typ, Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx, _, disp, _),
        if *r2 == "RBP",
        if *idx == "NONE",
        op_register(dst, dststr),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        ireg_hold_type(dststr.to_string(), typ);

    // SP-relative indexed store: unlike an index-free store, this is an array
    // element write rather than an Msetstack scalar slot. Preserve both the
    // stack base and the index so RTL can expand it through Ainstack.
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        !reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        ireg_hold_type(srcstr.to_string(), typ),
        type_to_memchunk(typ, chunk),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(*chunk, *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::SP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::SP, *idx_arg])
        };

    // XMM scalar stores take their type from the memory access width, not the
    // register's Tany64 hold type.
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        reg_xmm(srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = if *msize == 4 {
            MemoryChunk::MFloat32
        } else if *msize == 8 {
            MemoryChunk::MFloat64
        } else {
            MemoryChunk::MAny64
        },
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::SP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::SP, *idx_arg])
        };

    // SP-relative indexed load: treat as Mload with Aindexed2scaled addressing
    mach_inst(addr, MachInst::Mload(*mc, addressing, Arc::new(args), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str == "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        op_register(dst, dststr),
        ireg_hold_type(dststr.to_string(), typ),
        type_to_memchunk(typ, mc),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::SP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::SP, *idx_arg])
        };

    // BP-relative indexed load with frame pointer: treat as Mload
    mach_inst(addr, MachInst::Mload(*mc, addressing, Arc::new(args), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str == "RBP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        op_register(dst, dststr),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        ireg_hold_type(dststr.to_string(), typ),
        type_to_memchunk(typ, mc),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-relative load when BP is not the frame pointer is a regular memory load, INDEX-FREE form only: without the index guard the index was dropped and element[0] read forever.
    mach_inst(addr, MachInst::Mload(mc, Addressing::Aindexed(*disp), Arc::new(vec![Mreg::BP]), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx_str, _, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str == "NONE" || idx_str.is_empty(),
        op_register(dst, dststr),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        ireg_hold_type(dststr.to_string(), typ),
        type_to_memchunk(typ, base_chunk),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        // For a sign/zero-extending GP load the accessed memory is the narrower SOURCE, so take the chunk from the mnemonic and operand width, not the 64-bit dst hold-type; float moves excluded.
        let mnem_upper = mnem.to_ascii_uppercase(),
        let mc = if (mnem_upper.contains("MOVZX") || mnem_upper.contains("MOVSX")
            || mnem_upper.contains("MOVZB") || mnem_upper.contains("MOVSB")
            || mnem_upper.contains("MOVZW") || mnem_upper.contains("MOVSW")) {
            refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize)
        } else {
            *base_chunk
        };

    // BP-base INDEXED load, BP not the frame pointer: preserve base+index (mirror the frame-pointer rule).
    mach_inst(addr, MachInst::Mload(mc, addressing, Arc::new(args), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx_str, scale, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str != "NONE" && !idx_str.is_empty(),
        op_register(dst, dststr),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        ireg_hold_type(dststr.to_string(), typ),
        type_to_memchunk(typ, base_chunk),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        // Sign/zero-extending GP load: width comes from the SOURCE operand, not the 64-bit dst hold-type; float moves have their own XMM width rules.
        let mnem_upper = mnem.to_ascii_uppercase(),
        let mc = if (mnem_upper.contains("MOVZX") || mnem_upper.contains("MOVSX")
            || mnem_upper.contains("MOVZB") || mnem_upper.contains("MOVSB")
            || mnem_upper.contains("MOVZW") || mnem_upper.contains("MOVSW")) {
            refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize)
        } else {
            *base_chunk
        },
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-relative load (mnemonic-based chunk) when BP is NOT the frame pointer
    mach_inst(addr, MachInst::Mload(mc, Addressing::Aindexed(*disp), Arc::new(vec![Mreg::BP]), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx_str, _, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str == "NONE" || idx_str.is_empty(),
        op_register(dst, dststr),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        !ireg_hold_type(dststr.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    // BP-base INDEXED load (mnemonic chunk), BP not the frame pointer: preserve base+index.
    mach_inst(addr, MachInst::Mload(mc, addressing, Arc::new(args), Mreg::x86(dststr))) <--
        pmov(addr, dst, src),
        op_indirect(src, _, r2, idx_str, scale, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str != "NONE" && !idx_str.is_empty(),
        op_register(dst, dststr),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        !ireg_hold_type(dststr.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-base INDEXED store with a frame pointer, the mirror of the indexed load: without it a runtime-indexed stack-array write produced no Mstore and DSE collapsed the loop body.
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if *r2 == "RBP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        ireg_hold_type(srcstr.to_string(), typ),
        type_to_memchunk(typ, chunk),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(*chunk, *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-base INDEXED store (mnemonic chunk) WITH frame pointer: src reg has no inferred hold type.
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if *r2 == "RBP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        !ireg_hold_type(srcstr.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-relative store when BP is NOT the frame pointer. INDEX-FREE form only (indexed form below).
    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aindexed(*disp), Arc::new(vec![Mreg::BP]), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, _, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        ireg_hold_type(srcstr.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    // BP-base INDEXED store, BP not the frame pointer: preserve base+index (else `out[i]=x` -> `out[0]=x`).
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str != "NONE" && !idx_str.is_empty(),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        ireg_hold_type(srcstr.to_string(), typ),
        type_to_memchunk(typ, chunk),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(*chunk, *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-relative store (mnemonic-based chunk) when BP is NOT the frame pointer. INDEX-FREE form only.
    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aindexed(*disp), Arc::new(vec![Mreg::BP]), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, _, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        !ireg_hold_type(srcstr.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // BP-base INDEXED store (mnemonic chunk), BP not the frame pointer: preserve base+index.
    mach_inst(addr, MachInst::Mstore(mc, addressing, Arc::new(args), src_reg)) <--
        pmov(addr, dst, src),
        op_register(src, srcstr),
        let src_reg = Mreg::x86(srcstr),
        op_indirect(dst, _, r2, idx_str, scale, disp, msize),
        if Mreg::x86(r2) == Mreg::BP,
        if *idx_str != "NONE" && !idx_str.is_empty(),
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        !ireg_hold_type(srcstr.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize),
        let (addressing, args) = if *scale > 1 {
            (Addressing::Aindexed2scaled(*scale, *disp), vec![Mreg::BP, *idx_arg])
        } else {
            (Addressing::Aindexed2(*disp), vec![Mreg::BP, *idx_arg])
        };

    // BP-relative immediate store when BP is NOT the frame pointer
    mach_imm_indirect_store(addr, imm_int, mc, Mreg::BP, *disp) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, r2, _, _scale, disp, sz),
        if Mreg::x86(r2) == Mreg::BP,
        instr_in_function(addr, func),
        !bp_frame_at(addr, func),
        let mc = refine_chunk_with_size(if *sz <= 4 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 }, *sz),
        let imm_int = *imm_sym as i64;

    // BP-relative INDEXED immediate store with a frame pointer is a genuine stack-array element write, unlike the index-free scalar slot init; this row only fires the trigger and excludes the no-index case.
    mach_imm_indirect_store(addr, imm_int, mc, Mreg::BP, *disp) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, r2, idx_str, _scale, disp, sz),
        if *r2 == "RBP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        instr_in_function(addr, func),
        bp_frame_at(addr, func),
        let mc = refine_chunk_with_size(if *sz <= 4 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 }, *sz),
        let imm_int = *imm_sym as i64;


    #[local] relation cmp_jcc_link(Address, Address);
    cmp_jcc_link(addr0, addr1) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    cmp_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pcmp(addr0, _, _);

    #[local] relation test_jcc_link(Address, Address);
    test_jcc_link(addr0, addr1) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    test_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        ptest(addr0, _, _);

    // Secondary flags consumer: several jccs can consume one compare's flags, so each safe secondary carries its own Mcond at its OWN address, or the losing jcc degrades to an unconditional branch.
    #[local] relation secondary_flags_consumer(Address, Address);
    secondary_flags_consumer(addr0, addr1) <--
        cmp_jcc_link(addr0, addr1),
        cmp_jcc_link(addr0, mid),
        if *mid < *addr1;
    secondary_flags_consumer(addr0, addr1) <--
        test_jcc_link(addr0, addr1),
        test_jcc_link(addr0, mid),
        if *mid < *addr1;

    // When the compared register is redefined between the compare and a secondary jcc, re-reading it would test the WRONG value, so that secondary keeps the legacy cmp-keyed behavior.
    #[local] relation flags_args_redefined_between(Address, Address);
    flags_args_redefined_between(addr0, addr1) <--
        secondary_flags_consumer(addr0, addr1),
        pcmp(addr0, r1, _),
        op_register(r1, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r),
        reg_def(d, mreg),
        if *addr0 < *d && *d < *addr1;
    flags_args_redefined_between(addr0, addr1) <--
        secondary_flags_consumer(addr0, addr1),
        pcmp(addr0, _, r2),
        op_register(r2, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r),
        reg_def(d, mreg),
        if *addr0 < *d && *d < *addr1;
    flags_args_redefined_between(addr0, addr1) <--
        secondary_flags_consumer(addr0, addr1),
        ptest(addr0, r1, _),
        op_register(r1, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r),
        reg_def(d, mreg),
        if *addr0 < *d && *d < *addr1;

    #[local] relation secondary_safe(Address, Address);
    secondary_safe(addr0, addr1) <--
        secondary_flags_consumer(addr0, addr1),
        !flags_args_redefined_between(addr0, addr1);

    // Where each (compare, jcc) pair's Mcond is keyed: primary (and unsafe secondary) -> the compare's address; safe secondary -> the jcc's own address.
    #[local] relation mcond_emit_addr(Address, Address, Address);
    mcond_emit_addr(addr0, addr1, addr0) <--
        cmp_jcc_link(addr0, addr1),
        !secondary_safe(addr0, addr1);
    mcond_emit_addr(addr0, addr1, addr0) <--
        test_jcc_link(addr0, addr1),
        !secondary_safe(addr0, addr1);
    mcond_emit_addr(addr0, addr1, addr1) <--
        secondary_safe(addr0, addr1);

    relation mcond_at_jcc(Address);
    mcond_at_jcc(addr1) <-- secondary_safe(_, addr1);

    // A secondary-consumer Mcond reads its compare's registers at the JCC's address, where the raw jcc has no operands, so bind the uses here or liveness never connects the compare's def.
    reg_use(addr1, mreg) <--
        secondary_safe(addr0, addr1),
        pcmp(addr0, r, _),
        op_register(r, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r);
    reg_use(addr1, mreg) <--
        secondary_safe(addr0, addr1),
        pcmp(addr0, _, r),
        op_register(r, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r);
    reg_use(addr1, mreg) <--
        secondary_safe(addr0, addr1),
        ptest(addr0, r, _),
        op_register(r, reg_str),
        ireg_of(preg_of_r, Ireg::from(reg_str)),
        preg_of(mreg, preg_of_r);

    #[local] relation cmp_cmov_link(Address, Address);
    cmp_cmov_link(addr0, addr1) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        pcmov(addr1, _, _, _);
    cmp_cmov_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pcmp(addr0, _, _),
        pcmov(addr1, _, _, _);

    // CMP/TEST -> 1 non-flag-clobbering instruction -> CMOV
    cmp_cmov_link(addr0, addr2) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pcmov(addr2, _, _, _);

    // CMP/TEST -> 2 non-flag-clobbering instructions -> CMOV
    cmp_cmov_link(addr0, addr3) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        pcmov(addr3, _, _, _);

    // NEG -> CMOV link: NEG sets flags (SF based on result), enabling CMOVS/CMOVNS.
    #[local] relation neg_cmov_link(Address, Address);
    neg_cmov_link(addr0, addr1) <--
        pneg(addr0, _),
        next(addr0, addr1),
        pcmov(addr1, _, _, _);

    // NEG -> 1 non-flag-clobbering instruction -> CMOV
    neg_cmov_link(addr0, addr2) <--
        pneg(addr0, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pcmov(addr2, _, _, _);

    #[local] relation fcmp_jcc_link(Address, Address);
    fcmp_jcc_link(addr0, addr1) <--
        pucomisd(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    fcmp_jcc_link(addr0, addr1) <--
        pucomiss(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    fcmp_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pucomisd(addr0, _, _);
    fcmp_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pucomiss(addr0, _, _);

    // TestCond -> Condition divergence: CondE/CondNe are ZF-based (signedness-agnostic), so they map to both signed and unsigned comparison conditions.
    #[local] relation testcond_to_cond(TestCond, Condition);
    #[local] relation testcond_to_cond_64(TestCond, Condition);

    testcond_to_cond(test_cond.clone(), condition_for_testcond_sized(*test_cond, false)) <--
        pjcc(_, test_cond, _);
    testcond_to_cond_64(test_cond.clone(), condition_for_testcond_sized(*test_cond, true)) <--
        pjcc(_, test_cond, _);

    testcond_to_cond(TestCond::CondE, Condition::Ccompu(Comparison::Ceq)) <--
        pjcc(_, test_cond, _), if let TestCond::CondE = test_cond;
    testcond_to_cond_64(TestCond::CondE, Condition::Ccomplu(Comparison::Ceq)) <--
        pjcc(_, test_cond, _), if let TestCond::CondE = test_cond;

    testcond_to_cond(TestCond::CondNe, Condition::Ccompu(Comparison::Cne)) <--
        pjcc(_, test_cond, _), if let TestCond::CondNe = test_cond;
    testcond_to_cond_64(TestCond::CondNe, Condition::Ccomplu(Comparison::Cne)) <--
        pjcc(_, test_cond, _), if let TestCond::CondNe = test_cond;


    mach_inst(emit_addr, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, imm_val, _),
        reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        testcond_to_cond_64(*test_cond, base_cond),
        let condition = match base_cond {
            Condition::Ccomp(cmp) | Condition::Ccompl(cmp) => Condition::Ccomplimm(*cmp, *imm_val),
            Condition::Ccompu(cmp) | Condition::Ccomplu(cmp) => Condition::Ccompluimm(*cmp, *imm_val),
            other => other.clone(),
        };

    mach_inst(emit_addr, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, imm_val, _),
        !reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        testcond_to_cond(*test_cond, base_cond),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(*cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(*cmp, *imm_val),
            other => other.clone(),
        };

    mach_inst(emit_addr, MachInst::Mcond(condition.clone(), Arc::new(vec![*arg1, *arg2]), lbl)) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        ireg_of(preg_of_r2, Ireg::from(reg_str2)),
        preg_of(arg1, preg_of_r1),
        preg_of(arg2, preg_of_r2),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        testcond_to_cond_64(*test_cond, condition);

    mach_inst(emit_addr, MachInst::Mcond(condition.clone(), Arc::new(vec![*arg1, *arg2]), lbl)) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        !reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        ireg_of(preg_of_r2, Ireg::from(reg_str2)),
        preg_of(arg1, preg_of_r1),
        preg_of(arg2, preg_of_r2),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        testcond_to_cond(*test_cond, condition);


    mach_inst(emit_addr, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        if reg_str1 == reg_str2,
        reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        let base_cond = condition_for_testcond_sized(*test_cond, true),
        let condition = match base_cond {
            Condition::Ccomp(cmp) | Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, 0),
            Condition::Ccompu(cmp) | Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, 0),
            other => other,
        };

    mach_inst(emit_addr, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        if reg_str1 == reg_str2,
        !reg_64(reg_str1),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, test_cond, lbl),
        let base_cond = condition_for_testcond_sized(*test_cond, false),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, 0),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, 0),
            other => other,
        };


    // ADD followed by a flags-reading jcc tests the arithmetic RESULT against zero, so emit the Mcond at the jcc's own address; restricted to ZF/signed conditions, since after an ADD carry is not a comparison.
    #[local] relation arith_result_reg(Address, Symbol, bool);
    arith_result_reg(addr, dst, is_64) <--
        padd(addr, dst, _),
        op_register(dst, dst_str),
        reg_is_64(dst_str, is_64);

    // SAR/SHR then a flags-reading jcc is the divide-by-power-of-2 loop latch; ZF/signed conditions only, since after a shift CF is the last bit shifted out, not a comparison.
    arith_result_reg(addr, dst, is_64) <--
        psar(addr, dst, _),
        op_register(dst, dst_str),
        reg_is_64(dst_str, is_64);
    arith_result_reg(addr, dst, is_64) <--
        pshr(addr, dst, _),
        op_register(dst, dst_str),
        reg_is_64(dst_str, is_64);

    // MSVC uses DEC/JNE for optimized loop latches and repeated DEC/JE for
    // compact switch chains.  Their value lowering lives below; retain the
    // same destination here so the branch can consume the updated result.
    arith_result_reg(addr, dst, is_64) <--
        instruction(addr, _, _, mnem, dst, _, _, _, _, _),
        if *mnem == "INC" || *mnem == "DEC",
        op_register(dst, dst_str),
        reg_is_64(dst_str, is_64);

    // A SUB-immediate before a flags-reading jcc is decrement-and-branch: when the decremented value is live, keep the SUB's value op and emit the Mcond at the jcc, or the -1 is dropped.
    arith_result_reg(addr, dst, is_64) <--
        sub_result_live(addr, dst_mreg),
        psub(addr, dst, n),
        op_immediate(n, _, _),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        reg_is_64(dst_str, is_64);

    // Instruction-level CFG successor: intra-block sequential adjacency plus real inter-block edges, excluding calls and indirect edges; a structural walk, never an address comparison.
    #[local] relation cfg_step(Address, Address);
    cfg_step(a, b) <--
        next(a, b),
        code_in_block(a, blk),
        code_in_block(b, blk);
    cfg_step(a, b) <--
        ddisasm_cfg_edge(a, b, edge_type),
        if *edge_type != "call" && *edge_type != "indirect" && *edge_type != "indirect_call";

    // Forward reaching-value of the SUB's destination over the CFG, killed at any redefinition; a node that both reads and writes R is still reached, since the kill blocks propagation past it.
    #[local] relation sub_dst_reaches(Address, Mreg, Address);
    sub_dst_reaches(sub_addr, dst_mreg, succ) <--
        psub(sub_addr, dst, n),
        op_immediate(n, _, _),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        cfg_step(sub_addr, succ);
    sub_dst_reaches(sub_addr, dst_mreg, nxt) <--
        sub_dst_reaches(sub_addr, dst_mreg, cur),
        !reg_def(cur, dst_mreg),
        cfg_step(cur, nxt);

    // sub_result_live: the decremented value is genuinely read downstream by a real value instruction, excluding the consuming jcc whose only reg_use is the synthetic relocated-condition row.
    #[local] relation sub_result_live(Address, Mreg);
    sub_result_live(sub_addr, dst_mreg) <--
        sub_dst_reaches(sub_addr, dst_mreg, u),
        if *u != *sub_addr,
        !pjcc(u, _, _),
        reg_use(u, dst_mreg);

    // arith_result_jcc(arith_addr, jcc_addr, dst_sym, is_64, test_cond): the arith result feeds the jcc.
    #[local] relation arith_result_jcc(Address, Address, Symbol, bool, TestCond);
    arith_result_jcc(addr0, addr1, dst, *is_64, *test_cond) <--
        arith_result_reg(addr0, dst, is_64),
        next(addr0, addr1),
        pjcc(addr1, test_cond, _),
        if arith_result_testcond_ok(*test_cond);
    arith_result_jcc(addr0, addr2, dst, *is_64, *test_cond) <--
        arith_result_reg(addr0, dst, is_64),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pjcc(addr2, test_cond, _),
        if arith_result_testcond_ok(*test_cond);
    // The disassembly CFG also links INC/DEC to a later Jcc across any number
    // of intervening flag-preserving instructions.  This is needed for MSVC
    // scheduling patterns where a MOV sits between the loop update and JNE.
    arith_result_jcc(addr0, addr1, dst, *is_64, *test_cond) <--
        arith_result_reg(addr0, dst, is_64),
        flags_and_jump_pair(addr0, addr1, _),
        pjcc(addr1, test_cond, _),
        if arith_result_testcond_ok(*test_cond);

    // Mcond at the jcc's OWN address: `if (R <cmp> 0) goto target`.
    mach_inst(addr1, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        arith_result_jcc(_, addr1, dst, is_64, test_cond),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(arg1, preg_of_dst),
        pjcc(addr1, _, lbl),
        let base_cond = condition_for_testcond_sized(*test_cond, *is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, 0),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, 0),
            other => other,
        };

    // The Mcond lives at the jcc address, so linear computes its fall-through as next(jcc).
    mcond_at_jcc(addr1) <-- arith_result_jcc(_, addr1, _, _, _);

    // The raw jcc has no register operands, so bind the result register's use at the jcc to connect the arith def to the relocated condition.
    reg_use(addr1, mreg) <--
        arith_result_jcc(_, addr1, dst, _, _),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(mreg, preg_of_dst);


    // osel_compare_site(cmov_addr, compare_addr): the compare/test whose flags this cmov consumes. rtl_pass resolves the cmov condition operands at compare_addr, since clang may reuse the compared register as a value holder between the compare and the cmov.
    relation osel_compare_site(Address, Address);

    // CMP-imm + CMOV -> Osel: anchored at CMOV address so dst_mreg def is visible to downstream uses (avoids DSE when there is a gap).
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(condition, typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *cmp_arg1]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pcmp(addr0, cmp_r1, cmp_r2),
        op_register(cmp_r1, cmp_reg_str),
        op_immediate(cmp_r2, imm_val, _),
        ireg_of(preg_of_cmp1, Ireg::from(cmp_reg_str)),
        preg_of(cmp_arg1, preg_of_cmp1),
        cmp_cmov_link(addr0, addr1),
        pcmov(addr1, cmov_cond, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        let neg_cond = negate_testcond(*cmov_cond),
        reg_is_64(cmp_reg_str, cmp_is_64),
        let base_cond = condition_for_testcond_sized(neg_cond, *cmp_is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            other => other,
        },
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // CMP-mem + CMOV -> Osel: the compare's first operand is a memory deref, so use its base register as the Osel condition operand; gated on cmp_cmov_link and a base-register form.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(condition, typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *cmp_arg1]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pcmp(addr0, cmp_r1, cmp_r2),
        op_indirect(cmp_r1, _, cmp_base_str, _, _, _, _),
        if *cmp_base_str != "NONE" && *cmp_base_str != "RIP" && !cmp_base_str.is_empty(),
        op_immediate(cmp_r2, imm_val, _),
        ireg_of(preg_of_cmp1, Ireg::from(cmp_base_str)),
        preg_of(cmp_arg1, preg_of_cmp1),
        cmp_cmov_link(addr0, addr1),
        pcmov(addr1, cmov_cond, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        let neg_cond = negate_testcond(*cmov_cond),
        reg_is_64(cmp_base_str, cmp_is_64),
        let base_cond = condition_for_testcond_sized(neg_cond, *cmp_is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            other => other,
        },
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // Backstop for a CMOV whose compare uses a RIP-relative operand with no usable register: emit a plain two-operand Osel so the cmov's def survives, gated on cmp_cmov_link and !osel_compare_site.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(condition, typ),
        Arc::new(vec![*dst_mreg, *src_mreg]),
        *dst_mreg
    )) <--
        pcmp(addr0, cmp_r1, _),
        op_indirect(cmp_r1, _, _, _, _, _, _),
        cmp_cmov_link(addr0, addr1),
        pcmov(addr1, cmov_cond, dst_sym, src_sym),
        !osel_compare_site(addr1, _),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        let neg_cond = negate_testcond(*cmov_cond),
        reg_is_64(dst_str, dst_is_64),
        let condition = condition_for_testcond_sized(neg_cond, *dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // CMP-reg + CMOV -> Osel: anchored at CMOV address.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(condition, typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *cmp_arg1, *cmp_arg2]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pcmp(addr0, cmp_r1, cmp_r2),
        op_register(cmp_r1, cmp_str1),
        op_register(cmp_r2, cmp_str2),
        ireg_of(preg_of_cmp1, Ireg::from(cmp_str1)),
        ireg_of(preg_of_cmp2, Ireg::from(cmp_str2)),
        preg_of(cmp_arg1, preg_of_cmp1),
        preg_of(cmp_arg2, preg_of_cmp2),
        cmp_cmov_link(addr0, addr1),
        pcmov(addr1, cmov_cond, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        let neg_cond = negate_testcond(*cmov_cond),
        reg_is_64(cmp_str1, cmp_is_64),
        let condition = condition_for_testcond_sized(neg_cond, *cmp_is_64),
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // NEG + CMOV -> Osel: anchored at CMOV address; NEG sets flags as `0 - operand`, condition compares NEG operand against 0; NEG reg is both input and negated output.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(condition, typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *neg_arg]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pneg(addr0, neg_r),
        op_register(neg_r, neg_reg_str),
        ireg_of(preg_of_neg, Ireg::from(neg_reg_str)),
        preg_of(neg_arg, preg_of_neg),
        neg_cmov_link(addr0, addr1),
        pcmov(addr1, cmov_cond, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        let neg_cond = negate_testcond(*cmov_cond),
        reg_is_64(neg_reg_str, neg_is_64),
        let base_cond = condition_for_testcond_sized(neg_cond, *neg_is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, 0),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, 0),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, 0),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, 0),
            other => other,
        },
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // TEST/AND $mask + CMOVE/CMOVNE -> Osel with a mask condition; the Osel condition selects the PRESERVE value, so it is the NEGATION of the cmov predicate (CMOVE gives Cmasknotzero).
    #[local] relation test_cmov_link(Address, Address);
    test_cmov_link(addr0, addr1) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        pcmov(addr1, _, _, _);
    test_cmov_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        ptest(addr0, _, _),
        pcmov(addr1, _, _, _);
    test_cmov_link(addr0, addr2) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pcmov(addr2, _, _, _);
    test_cmov_link(addr0, addr3) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        pcmov(addr3, _, _, _);

    #[local] relation and_cmov_link(Address, Address);
    and_cmov_link(addr0, addr1) <--
        pand(addr0, _, _),
        next(addr0, addr1),
        pcmov(addr1, _, _, _);
    and_cmov_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pand(addr0, _, _),
        pcmov(addr1, _, _, _);
    and_cmov_link(addr0, addr2) <--
        pand(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pcmov(addr2, _, _, _);

    // TEST $mask + CMOVE: Osel(Cmasknotzero(mask), dst, src, masked_reg).
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(Condition::Cmasknotzero(high8_mask_adjust(test_str, *mask_val)), typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *test_arg]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        ptest(addr0, test_r, mask_op),
        op_register(test_r, test_str),
        op_immediate(mask_op, mask_val, _),
        ireg_of(preg_of_test, Ireg::from(test_str)),
        preg_of(test_arg, preg_of_test),
        test_cmov_link(addr0, addr1),
        pcmov(addr1, TestCond::CondE, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // TEST $mask + CMOVNE: Osel(Cmaskzero(mask), dst, src, masked_reg).
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(Condition::Cmaskzero(high8_mask_adjust(test_str, *mask_val)), typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *test_arg]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        ptest(addr0, test_r, mask_op),
        op_register(test_r, test_str),
        op_immediate(mask_op, mask_val, _),
        ireg_of(preg_of_test, Ireg::from(test_str)),
        preg_of(test_arg, preg_of_test),
        test_cmov_link(addr0, addr1),
        pcmov(addr1, TestCond::CondNe, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // AND $mask + CMOVE: same as the TEST case, since masking is idempotent so resolving the post-AND value under Cmask* is still correct.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(Condition::Cmasknotzero(high8_mask_adjust(test_str, *mask_val)), typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *test_arg]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pand(addr0, test_r, mask_op),
        op_register(test_r, test_str),
        op_immediate(mask_op, mask_val, _),
        ireg_of(preg_of_test, Ireg::from(test_str)),
        preg_of(test_arg, preg_of_test),
        and_cmov_link(addr0, addr1),
        pcmov(addr1, TestCond::CondE, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };

    // AND $mask + CMOVNE.
    mach_inst(addr1, MachInst::Mop(
        Operation::Osel(Condition::Cmaskzero(high8_mask_adjust(test_str, *mask_val)), typ),
        Arc::new(vec![*dst_mreg, *src_mreg, *test_arg]),
        *dst_mreg
    )),
    osel_compare_site(*addr1, *addr0) <--
        pand(addr0, test_r, mask_op),
        op_register(test_r, test_str),
        op_immediate(mask_op, mask_val, _),
        ireg_of(preg_of_test, Ireg::from(test_str)),
        preg_of(test_arg, preg_of_test),
        and_cmov_link(addr0, addr1),
        pcmov(addr1, TestCond::CondNe, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        !reg_xmm(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(dst_mreg, preg_of_dst),
        preg_of(src_mreg, preg_of_src),
        reg_is_64(dst_str, dst_is_64),
        let typ = if *dst_is_64 { Typ::Tany64 } else { Typ::Tint };


    mach_inst(emit_addr, MachInst::Mcond(Condition::Cmaskzero(adj_mask), Arc::new(vec![*arg1]), lbl)) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, TestCond::CondE, lbl),
        let adj_mask = high8_mask_adjust(reg_str1, *mask_val);

    mach_inst(emit_addr, MachInst::Mcond(Condition::Cmasknotzero(adj_mask), Arc::new(vec![*arg1]), lbl)) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        mcond_emit_addr(addr0, addr1, emit_addr),
        pjcc(addr1, TestCond::CondNe, lbl),
        let adj_mask = high8_mask_adjust(reg_str1, *mask_val);


    #[local] relation sub_jcc_link(Address, Address);
    sub_jcc_link(addr0, addr1) <--
        psub(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    sub_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        psub(addr0, _, _);

    // SUB -> 1 non-flag-clobbering instruction -> JCC (e.g. SUB + MOV mem + JCC)
    sub_jcc_link(addr0, addr2) <--
        psub(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pjcc(addr2, _, _);

    // SUB -> 2 non-flag-clobbering instructions -> JCC
    sub_jcc_link(addr0, addr3) <--
        psub(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        pjcc(addr3, _, _);

    // Suppressed when arith_result_jcc covers this SUB: the branch is then emitted at the jcc's address and the SUB keeps its value op, so the decrement is not lost and the taken edge survives.
    mach_inst(addr0, MachInst::Mcond(condition, Arc::new(vec![*arg1]), lbl)) <--
        psub(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, imm_val, _),
        !arith_result_jcc(addr0, _, _, _, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        sub_jcc_link(addr0, addr1),
        pjcc(addr1, test_cond, lbl),
        reg_is_64(reg_str1, is_64),
        let base_cond = condition_for_testcond_sized(*test_cond, *is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            other => other,
        };

    mach_inst(addr0, MachInst::Mcond(condition, Arc::new(vec![*arg1, *arg2]), lbl)) <--
        psub(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        ireg_of(preg_of_r2, Ireg::from(reg_str2)),
        preg_of(arg1, preg_of_r1),
        preg_of(arg2, preg_of_r2),
        sub_jcc_link(addr0, addr1),
        pjcc(addr1, test_cond, lbl),
        reg_is_64(reg_str1, is_64),
        let condition = condition_for_testcond_sized(*test_cond, *is_64);


    #[local] relation and_jcc_link(Address, Address);
    and_jcc_link(addr0, addr1) <--
        pand(addr0, _, _),
        next(addr0, addr1),
        pjcc(addr1, _, _);
    and_jcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pand(addr0, _, _);

    // AND -> 1 non-flag-clobbering instruction -> JCC (e.g. AND + MOV mem + JCC)
    and_jcc_link(addr0, addr2) <--
        pand(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        pjcc(addr2, _, _);

    // AND -> 2 non-flag-clobbering instructions -> JCC
    and_jcc_link(addr0, addr3) <--
        pand(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        pjcc(addr3, _, _);

    mach_inst(addr0, MachInst::Mcond(Condition::Cmaskzero(adj_mask), Arc::new(vec![*arg1]), lbl)) <--
        pand(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        and_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondE, lbl),
        let adj_mask = high8_mask_adjust(reg_str1, *mask_val);

    mach_inst(addr0, MachInst::Mcond(Condition::Cmasknotzero(adj_mask), Arc::new(vec![*arg1]), lbl)) <--
        pand(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        and_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondNe, lbl),
        let adj_mask = high8_mask_adjust(reg_str1, *mask_val);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompimm(Comparison::Ceq, 0), Arc::new(vec![*arg1]), lbl)) <--
        pand(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        and_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondE, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompimm(Comparison::Cne, 0), Arc::new(vec![*arg1]), lbl)) <--
        pand(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        and_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondNe, lbl);

    // JCXZ/JECXZ/JRCXZ test (E/R)CX==0 directly with no preceding compare, so lift each as a self-contained Mcond at its own address; the CFG already creates both edges.
    mach_inst(addr, MachInst::Mcond(Condition::Ccompimm(Comparison::Ceq, 0), Arc::new(vec![Mreg::CX]), dst)) <--
        instruction(addr, _, _, mnem, dst, _, _, _, _, _),
        if *mnem == "JCXZ" || *mnem == "JECXZ";

    mach_inst(addr, MachInst::Mcond(Condition::Ccomplimm(Comparison::Ceq, 0), Arc::new(vec![Mreg::CX]), dst)) <--
        instruction(addr, _, _, "JRCXZ", dst, _, _, _, _, _);

    // ASM-5: JO/JNO lift to the opaque Coverflow/Cnotoverflow, emitted self-contained at the jump's own address so a preceding CMP is never folded into a false comparison.
    mach_inst(addr, MachInst::Mcond(Condition::Coverflow, Arc::new(vec![]), dst)) <--
        instruction(addr, _, _, "JO", dst, _, _, _, _, _);
    mach_inst(addr, MachInst::Mcond(Condition::Cnotoverflow, Arc::new(vec![]), dst)) <--
        instruction(addr, _, _, "JNO", dst, _, _, _, _, _);

    // ASM-5: LOOP/LOOPE/LOOPNE branch iff the decremented RCX != 0, which as a function of RCX at the instruction is exactly Ccomplimm(Cne, 1); the RCX write-back and ZF refinement are not modelled.
    mach_inst(addr, MachInst::Mcond(Condition::Ccomplimm(Comparison::Cne, 1), Arc::new(vec![Mreg::CX]), dst)) <--
        instruction(addr, _, _, mnem, dst, _, _, _, _, _),
        if *mnem == "LOOP" || *mnem == "LOOPE" || *mnem == "LOOPNE";

    // Every single-instruction conditional jump keys its Mcond at its OWN address, so mcond_at_jcc switches linear to the fall-through = next(addr) rule instead of the paired-compare skip.
    relation single_inst_cond_jump(Address);
    single_inst_cond_jump(addr) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem,
            "JCXZ" | "JECXZ" | "JRCXZ" | "JO" | "JNO" | "LOOP" | "LOOPE" | "LOOPNE");
    mcond_at_jcc(addr) <-- single_inst_cond_jump(addr);


    // SETcc fuses with the nearest preceding flag-setting compare, tolerating scheduled non-flag-clobbering instructions; the walk stops at any flag setter so a stale compare is never bound.
    #[local] relation cmp_setcc_link(Address, Address);
    cmp_setcc_link(addr0, addr1) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        psetcc(addr1, _, _);
    cmp_setcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pcmp(addr0, _, _),
        psetcc(addr1, _, _);
    cmp_setcc_link(addr0, addr2) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        psetcc(addr2, _, _);
    cmp_setcc_link(addr0, addr3) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        psetcc(addr3, _, _);

    #[local] relation test_setcc_link(Address, Address);
    test_setcc_link(addr0, addr1) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        psetcc(addr1, _, _);
    test_setcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        ptest(addr0, _, _),
        psetcc(addr1, _, _);
    test_setcc_link(addr0, addr2) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        psetcc(addr2, _, _);
    test_setcc_link(addr0, addr3) <--
        ptest(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        psetcc(addr3, _, _);

    #[local] relation sub_setcc_link(Address, Address);
    sub_setcc_link(addr0, addr1) <--
        psub(addr0, _, _),
        next(addr0, addr1),
        psetcc(addr1, _, _);
    sub_setcc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        psub(addr0, _, _),
        psetcc(addr1, _, _);
    sub_setcc_link(addr0, addr2) <--
        psub(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        psetcc(addr2, _, _);

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(condition),
        Arc::new(vec![*arg1]),
        *dst_mreg
    )) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, imm_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        cmp_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        reg_is_64(reg_str1, is_64),
        let base_cond = condition_for_testcond_sized(*test_cond, *is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            other => other,
        };

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(condition),
        Arc::new(vec![*arg1, *arg2]),
        *dst_mreg
    )) <--
        pcmp(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        ireg_of(preg_of_r2, Ireg::from(reg_str2)),
        preg_of(arg1, preg_of_r1),
        preg_of(arg2, preg_of_r2),
        cmp_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        reg_is_64(reg_str1, is_64),
        let condition = condition_for_testcond_sized(*test_cond, *is_64);

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(condition),
        Arc::new(vec![*arg1]),
        *dst_mreg
    )) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_register(r2, reg_str2),
        if reg_str1 == reg_str2,
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        test_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        reg_is_64(reg_str1, is_64),
        let base_cond = condition_for_testcond_sized(*test_cond, *is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, 0),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, 0),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, 0),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, 0),
            other => other,
        };

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(Condition::Cmaskzero(high8_mask_adjust(reg_str1, *mask_val))),
        Arc::new(vec![*arg1]),
        *dst_mreg
    )) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        test_setcc_link(addr0, addr_set),
        psetcc(addr_set, TestCond::CondE, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst);

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(Condition::Cmasknotzero(high8_mask_adjust(reg_str1, *mask_val))),
        Arc::new(vec![*arg1]),
        *dst_mreg
    )) <--
        ptest(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, mask_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        test_setcc_link(addr0, addr_set),
        psetcc(addr_set, TestCond::CondNe, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst);

    mach_inst(addr0, MachInst::Mop(
        Operation::Ocmp(condition),
        Arc::new(vec![*arg1]),
        *dst_mreg
    )) <--
        psub(addr0, r1, r2),
        op_register(r1, reg_str1),
        op_immediate(r2, imm_val, _),
        ireg_of(preg_of_r1, Ireg::from(reg_str1)),
        preg_of(arg1, preg_of_r1),
        sub_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        reg_is_64(reg_str1, is_64),
        let base_cond = condition_for_testcond_sized(*test_cond, *is_64),
        let condition = match base_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            other => other,
        };

    mach_inst(addr0, MachInst::Mop(
        Operation::Oshrximm(*shift_n),
        Arc::new(args.clone()),
        *arg_mreg
    )) <--
        ptest(addr0, test_r1, test_r2),
        op_register(test_r1, test_reg1),
        op_register(test_r2, test_reg2),
        if test_reg1 == test_reg2,
        next(addr0, addr1),
        plea(addr1, lea_dst_sym, lea_src_sym),
        op_register(lea_dst_sym, lea_dst_str),
        op_indirect(lea_src_sym, _, base_reg, _, _, disp, _),
        if base_reg == test_reg1,
        if lea_dst_str != test_reg1,
        next(addr1, addr2),
        pcmov(addr2, TestCond::CondL, cmov_dst_sym, cmov_src_sym),
        op_register(cmov_dst_sym, cmov_dst_str),
        op_register(cmov_src_sym, cmov_src_str),
        if cmov_dst_str == test_reg1,
        if cmov_src_str == lea_dst_str,
        next(addr2, addr3),
        psar(addr3, sar_dst_sym, sar_src_sym),
        op_register(sar_dst_sym, sar_dst_str),
        op_immediate(sar_src_sym, shift_n, _),
        if sar_dst_str == test_reg1,
        if *disp == (1i64 << *shift_n) - 1,
        ireg_of(preg_of_arg, Ireg::from(test_reg1)),
        preg_of(arg_mreg, preg_of_arg),
        let args = vec![*arg_mreg];

    mach_inst(addr0, MachInst::Mop(
        Operation::Oshrxlimm(*shift_n),
        Arc::new(args.clone()),
        *arg_mreg
    )) <--
        pcast(addr0, _, _),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "CQO",
        next(addr0, addr1),
        pshr(addr1, shr_dst_sym, shr_src_sym),
        op_register(shr_dst_sym, shr_dst_str),
        op_immediate(shr_src_sym, shr_amount, _),
        next(addr1, addr2),
        plea(addr2, lea_dst_sym, lea_src_sym),
        op_register(lea_dst_sym, lea_dst_str),
        op_indirect(lea_src_sym, _, lea_base, lea_index, _scale, lea_disp, _),
        if *lea_disp == 0,
        if lea_base == lea_dst_str,
        if lea_index == shr_dst_str,
        next(addr2, addr3),
        psar(addr3, sar_dst_sym, sar_src_sym),
        op_register(sar_dst_sym, sar_dst_str),
        op_immediate(sar_src_sym, shift_n, _),
        if sar_dst_str == lea_dst_str,
        if *shr_amount == 64 - *shift_n,
        ireg_of(preg_of_arg, Ireg::from(lea_dst_str)),
        preg_of(arg_mreg, preg_of_arg),
        let args = vec![*arg_mreg];

    // Reusable sub-relations for multi-instruction idiom detection

    // Unify SAR and SHR
    relation pshift_right(Address, Symbol, Symbol);
    pshift_right(addr, dst, src) <-- psar(addr, dst, src);
    pshift_right(addr, dst, src) <-- pshr(addr, dst, src);

    // 32-bit magic multiply: MOVSXD -> IMUL3 $MAGIC -> SAR/SHR
    #[local] relation div_magic_32(Address, Address, i64, i64, Ireg, Ireg);
    div_magic_32(addr0, addr2, *magic, *shift_n, Ireg::from(input_str), Ireg::from(imul_dst_str)) <--
        pmov(addr0, movsxd_dst, movsxd_src),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "MOVSXD",
        op_register(movsxd_dst, temp_str),
        op_register(movsxd_src, input_str),
        next(addr0, addr1),
        pimul3(addr1, imul_dst, imul_src, imul_imm),
        op_register(imul_dst, imul_dst_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_src_str) == Ireg::from(temp_str),
        op_immediate(imul_imm, magic, _),
        next(addr1, addr2),
        pshift_right(addr2, sar_dst, sar_src),
        op_register(sar_dst, sar_dst_str),
        if Ireg::from(sar_dst_str) == Ireg::from(imul_dst_str),
        op_immediate(sar_src, shift_n, _),
        if *shift_n >= 32;

    // GCC signed-div with COMPENSATING magic (e.g. -O1 n/7, n/11): movsxd; imul3 NEG_MAGIC; shr 0x20; add result,input; sar K; sar input,0x1f; sub result,input. Tuple: (movsxd_addr, imul_addr, magic_low32, total_shift, input_ireg, result_ireg).
    #[local] relation div_magic_32_compensating_gcc(Address, Address, i64, i64, Ireg, Ireg);
    div_magic_32_compensating_gcc(addr0, addr1, *magic, 32 + *post_shift, Ireg::from(input_str), Ireg::from(result_str)) <--
        pmov(addr0, movsxd_dst, movsxd_src),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "MOVSXD",
        op_register(movsxd_dst, result_str),
        op_register(movsxd_src, input_str),
        next(addr0, addr1),
        // imul3 result, result, NEG_MAGIC (in-place)
        pimul3(addr1, imul_dst, imul_src, imul_imm),
        op_register(imul_dst, imul_dst_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_dst_str) == Ireg::from(result_str),
        if Ireg::from(imul_src_str) == Ireg::from(result_str),
        op_immediate(imul_imm, magic, _),
        if (*magic as i32) < 0,
        next(addr1, addr2),
        // SHR result, 0x20 (logical)
        pshr(addr2, shr1_dst, shr1_src),
        op_register(shr1_dst, shr1_dst_str),
        if Ireg::from(shr1_dst_str) == Ireg::from(result_str),
        op_immediate(shr1_src, shr1_n, _),
        if *shr1_n == 32,
        next(addr2, addr3),
        // ADD result, input (compensation with ORIGINAL input register)
        padd(addr3, add1_dst, add1_src),
        op_register(add1_dst, add1_dst_str),
        op_register(add1_src, add1_src_str),
        if Ireg::from(add1_dst_str) == Ireg::from(result_str),
        if Ireg::from(add1_src_str) == Ireg::from(input_str),
        next(addr3, addr4),
        // SAR result, K (arithmetic post-shift)
        psar(addr4, sar_dst, sar_src),
        op_register(sar_dst, sar_dst_str),
        if Ireg::from(sar_dst_str) == Ireg::from(result_str),
        op_immediate(sar_src, post_shift, _),
        if *post_shift > 0 && *post_shift <= 31,
        next(addr4, addr5),
        // SAR input, 0x1f (sign extraction via arithmetic shift, clobbers input)
        psar(addr5, sarsign_dst, sarsign_src),
        op_register(sarsign_dst, sarsign_dst_str),
        if Ireg::from(sarsign_dst_str) == Ireg::from(input_str),
        op_immediate(sarsign_src, sign_shift, _),
        if *sign_shift == 31,
        next(addr5, addr6),
        // SUB result, input
        psub(addr6, sub_dst, sub_src),
        op_register(sub_dst, sub_dst_str),
        op_register(sub_src, sub_src_str),
        if Ireg::from(sub_dst_str) == Ireg::from(result_str),
        if Ireg::from(sub_src_str) == Ireg::from(input_str);

    mach_inst(movsxd_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)) <--
        div_magic_32_compensating_gcc(movsxd_addr, _imul_addr, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor_compensating(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, result_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        let addr = movsxd_addr;
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, addr);
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, addr);
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, addr);
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, addr);
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, addr);
    div_consumed(addr) <--
        div_magic_32_compensating_gcc(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, a5),
        next(a5, addr);

    // Clang signed-div with COMPENSATING magic (neg i32, e.g. n/7, n/11): movsxd; imul3 NEG_MAGIC; shr 0x20; add result,partial; mov sign,result; shr sign,0x1f; sar K; add result,sign. Tuple: (movsxd_addr, imul_addr, magic_low32_signed, total_shift, input_ireg, result_ireg).
    #[local] relation div_magic_32_compensating(Address, Address, i64, i64, Ireg, Ireg);
    div_magic_32_compensating(addr0, addr1, *magic, 32 + *post_shift, Ireg::from(input_str), Ireg::from(result_str)) <--
        pmov(addr0, movsxd_dst, movsxd_src),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "MOVSXD",
        op_register(movsxd_dst, result_str),
        op_register(movsxd_src, input_str),
        next(addr0, addr1),
        // imul3 partial, result, NEG_MAGIC
        pimul3(addr1, imul_dst, imul_src, imul_imm),
        op_register(imul_dst, partial_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_src_str) == Ireg::from(result_str),
        if Ireg::from(partial_str) != Ireg::from(result_str),
        op_immediate(imul_imm, magic, _),
        if (*magic as i32) < 0,
        next(addr1, addr2),
        // SHR partial, 0x20 (logical)
        pshr(addr2, shr1_dst, shr1_src),
        op_register(shr1_dst, shr1_dst_str),
        if Ireg::from(shr1_dst_str) == Ireg::from(partial_str),
        op_immediate(shr1_src, shr1_n, _),
        if *shr1_n == 32,
        next(addr2, addr3),
        // ADD result, partial (compensation)
        padd(addr3, add1_dst, add1_src),
        op_register(add1_dst, add1_dst_str),
        op_register(add1_src, add1_src_str),
        if Ireg::from(add1_dst_str) == Ireg::from(result_str),
        if Ireg::from(add1_src_str) == Ireg::from(partial_str),
        next(addr3, addr4),
        // MOV sign_reg, result
        pmov(addr4, mov_dst, mov_src),
        op_register(mov_dst, sign_reg_str),
        op_register(mov_src, mov_src_str),
        if Ireg::from(mov_src_str) == Ireg::from(result_str),
        if Ireg::from(sign_reg_str) != Ireg::from(result_str),
        next(addr4, addr5),
        // SHR sign_reg, 0x1f (logical)
        pshr(addr5, shrsign_dst, shrsign_src),
        op_register(shrsign_dst, shrsign_dst_str),
        if Ireg::from(shrsign_dst_str) == Ireg::from(sign_reg_str),
        op_immediate(shrsign_src, sign_shift, _),
        if *sign_shift == 31,
        next(addr5, addr6),
        // SAR result, K (arithmetic)
        psar(addr6, sar_dst, sar_src),
        op_register(sar_dst, sar_dst_str),
        if Ireg::from(sar_dst_str) == Ireg::from(result_str),
        op_immediate(sar_src, post_shift, _),
        if *post_shift > 0 && *post_shift <= 31,
        next(addr6, addr7),
        // ADD result, sign_reg
        padd(addr7, add2_dst, add2_src),
        op_register(add2_dst, add2_dst_str),
        op_register(add2_src, add2_src_str),
        if Ireg::from(add2_dst_str) == Ireg::from(result_str),
        if Ireg::from(add2_src_str) == Ireg::from(sign_reg_str);

    // Emit divide at the MOVSXD address: capstone reports reg_use=[input] and reg_def=[result] there, which exactly matches our Lop's I/O.
    mach_inst(movsxd_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)) <--
        div_magic_32_compensating(movsxd_addr, _imul_addr, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor_compensating(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, result_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    // Mark all 8 instructions of the compensating idiom as div-consumed.
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        let addr = movsxd_addr;
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, a5),
        next(a5, addr);
    div_consumed(addr) <--
        div_magic_32_compensating(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, a5),
        next(a5, a6),
        next(a6, addr);

    // Clang signed-div with separate partial reg (e.g. n/10): movsxd; imul3 POS_MAGIC partial!=result; mov sign,partial; shr sign,0x3f; sar partial,K; add partial,sign. Tuple: (movsxd_addr, imul_addr, magic, total_shift, result_ireg=movsxd dst, partial_ireg=imul dst).
    #[local] relation div_magic_32_signbias_partial(Address, Address, i64, i64, Ireg, Ireg);
    div_magic_32_signbias_partial(addr0, addr1, *magic, *post_shift, Ireg::from(result_str), Ireg::from(partial_str)) <--
        pmov(addr0, movsxd_dst, _movsxd_src),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "MOVSXD",
        op_register(movsxd_dst, result_str),
        next(addr0, addr1),
        // imul3 partial, result, POS_MAGIC
        pimul3(addr1, imul_dst, imul_src, imul_imm),
        op_register(imul_dst, partial_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_src_str) == Ireg::from(result_str),
        if Ireg::from(partial_str) != Ireg::from(result_str),
        op_immediate(imul_imm, magic, _),
        if *magic > 0 && *magic <= i32::MAX as i64,
        next(addr1, addr2),
        // MOV sign_reg, partial
        pmov(addr2, mov_dst, mov_src),
        op_register(mov_dst, sign_reg_str),
        op_register(mov_src, mov_src_str),
        if Ireg::from(mov_src_str) == Ireg::from(partial_str),
        if Ireg::from(sign_reg_str) != Ireg::from(partial_str),
        next(addr2, addr3),
        // SHR sign_reg, 0x3f
        pshr(addr3, shrsign_dst, shrsign_src),
        op_register(shrsign_dst, shrsign_dst_str),
        if Ireg::from(shrsign_dst_str) == Ireg::from(sign_reg_str),
        op_immediate(shrsign_src, sign_shift, _),
        if *sign_shift == 63,
        next(addr3, addr4),
        // SAR partial, K (arithmetic; total_shift = K)
        psar(addr4, sar_dst, sar_src),
        op_register(sar_dst, sar_dst_str),
        if Ireg::from(sar_dst_str) == Ireg::from(partial_str),
        op_immediate(sar_src, post_shift, _),
        if *post_shift >= 32 && *post_shift <= 63,
        next(addr4, addr5),
        // ADD partial, sign_reg
        padd(addr5, add_dst, add_src),
        op_register(add_dst, add_dst_str),
        op_register(add_src, add_src_str),
        if Ireg::from(add_dst_str) == Ireg::from(partial_str),
        if Ireg::from(add_src_str) == Ireg::from(sign_reg_str);

    // Emit divide at IMUL address: capstone reg_use=[result]/reg_def=[partial] matches Lop I/O; movsxd's Ocast32signed stays live.
    mach_inst(imul_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *partial_mreg)) <--
        div_magic_32_signbias_partial(_movsxd_addr, imul_addr, magic, total_shift, input_ireg, partial_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_partial, partial_ireg),
        preg_of(partial_mreg, preg_of_partial),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    // Consume IMUL through final ADD (imul, mov, shr, sar, add); MOVSXD's Ocast32signed Mop is preserved.
    div_consumed(addr) <--
        div_magic_32_signbias_partial(_, addr, _, _, _, _);
    div_consumed(addr) <--
        div_magic_32_signbias_partial(_, imul_addr, _, _, _, _),
        next(imul_addr, addr);
    div_consumed(addr) <--
        div_magic_32_signbias_partial(_, imul_addr, _, _, _, _),
        next(imul_addr, a1),
        next(a1, addr);
    div_consumed(addr) <--
        div_magic_32_signbias_partial(_, imul_addr, _, _, _, _),
        next(imul_addr, a1),
        next(a1, a2),
        next(a2, addr);
    div_consumed(addr) <--
        div_magic_32_signbias_partial(_, imul_addr, _, _, _, _),
        next(imul_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, addr);

    // Clang signed-div (small pos magic, e.g. n/3, n/5): movsxd; imul3 MAGIC; mov sign,result; shr sign,0x3f; shr result,0x20; add result,sign. Tuple: (movsxd_addr, imul_addr, magic, total_shift=32, input_ireg, result_ireg).
    #[local] relation div_magic_32_signbias(Address, Address, i64, i64, Ireg, Ireg);
    div_magic_32_signbias(addr0, addr1, *magic, 32, Ireg::from(input_str), Ireg::from(result_str)) <--
        pmov(addr0, movsxd_dst, movsxd_src),
        instruction(addr0, _, _, mnem0, _, _, _, _, _, _),
        if *mnem0 == "MOVSXD",
        op_register(movsxd_dst, result_str),
        op_register(movsxd_src, input_str),
        next(addr0, addr1),
        pimul3(addr1, imul_dst, imul_src, imul_imm),
        op_register(imul_dst, imul_dst_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_dst_str) == Ireg::from(result_str),
        if Ireg::from(imul_src_str) == Ireg::from(result_str),
        op_immediate(imul_imm, magic, _),
        if *magic > 0 && *magic <= i32::MAX as i64,
        next(addr1, addr2),
        // MOV sign_reg, result
        pmov(addr2, mov_dst, mov_src),
        op_register(mov_dst, sign_reg_str),
        op_register(mov_src, mov_src_str),
        if Ireg::from(mov_src_str) == Ireg::from(result_str),
        if Ireg::from(sign_reg_str) != Ireg::from(result_str),
        next(addr2, addr3),
        // SHR sign_reg, 0x3f (logical, must be SHR)
        pshr(addr3, shrsign_dst, shrsign_src),
        op_register(shrsign_dst, shrsign_dst_str),
        if Ireg::from(shrsign_dst_str) == Ireg::from(sign_reg_str),
        op_immediate(shrsign_src, sign_shift, _),
        if *sign_shift == 63,
        next(addr3, addr4),
        // SHR result, 0x20 (logical, must be SHR)
        pshr(addr4, shrres_dst, shrres_src),
        op_register(shrres_dst, shrres_dst_str),
        if Ireg::from(shrres_dst_str) == Ireg::from(result_str),
        op_immediate(shrres_src, res_shift, _),
        if *res_shift == 32,
        next(addr4, addr5),
        // ADD result_lo, sign_reg_lo
        padd(addr5, add_dst, add_src),
        op_register(add_dst, add_dst_str),
        op_register(add_src, add_src_str),
        if Ireg::from(add_dst_str) == Ireg::from(result_str),
        if Ireg::from(add_src_str) == Ireg::from(sign_reg_str);

    // Emit the divide Mop at the MOVSXD address (capstone reg_use covers the input).
    mach_inst(movsxd_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)) <--
        div_magic_32_signbias(movsxd_addr, _imul_addr, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, result_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    // Mark the entire idiom as div-consumed so the raw instructions are suppressed.
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        let addr = movsxd_addr;
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, addr);
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, addr);
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, addr);
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, addr);
    div_consumed(addr) <--
        div_magic_32_signbias(movsxd_addr, _, _, _, _, _),
        next(movsxd_addr, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, addr);

    // Magic multiply via MOV + 2-operand IMUL: `mov $MAGIC, tmp; imul reg, tmp`. Tuple: (mov_addr, shift_addr, magic, shift_n, input_ireg dividend src, result_ireg IMUL/SHR dst). IMUL is operand-symmetric: magic reg may be src or dst.
    #[local] relation div_magic_64(Address, Address, i64, i64, Ireg, Ireg);

    // Variant A: magic is the IMUL's src; magic-holding reg IS the IMUL's dst (and SHR target), i.e. it held the input before being clobbered.
    div_magic_64(addr0, addr2, *magic, *shift_n, Ireg::from(imul_dst_str), Ireg::from(imul_dst_str)) <--
        pmov(addr0, mov_dst, mov_src),
        op_register(mov_dst, temp_str),
        op_immediate(mov_src, magic, _),
        if *magic != 0,
        next(addr0, addr1),
        pimul(addr1, imul_dst, imul_src),
        op_register(imul_dst, imul_dst_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_src_str) == Ireg::from(temp_str),
        next(addr1, addr2),
        pshift_right(addr2, shr_dst, shr_src),
        op_register(shr_dst, shr_dst_str),
        if Ireg::from(shr_dst_str) == Ireg::from(imul_dst_str),
        op_immediate(shr_src, shift_n, _),
        if *shift_n >= 32;

    // Variant B (clang udiv): magic loaded into IMUL dst, input held in IMUL src. `mov ecx,edi; mov eax,MAGIC; imul rax,rcx; shr rax,k`.
    div_magic_64(addr0, addr2, *magic, *shift_n, Ireg::from(imul_src_str), Ireg::from(imul_dst_str)) <--
        pmov(addr0, mov_dst, mov_src),
        op_register(mov_dst, temp_str),
        op_immediate(mov_src, magic, _),
        if *magic != 0,
        next(addr0, addr1),
        pimul(addr1, imul_dst, imul_src),
        op_register(imul_dst, imul_dst_str),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_dst_str) == Ireg::from(temp_str),
        if Ireg::from(imul_src_str) != Ireg::from(temp_str),
        next(addr1, addr2),
        pshift_right(addr2, shr_dst, shr_src),
        op_register(shr_dst, shr_dst_str),
        if Ireg::from(shr_dst_str) == Ireg::from(imul_dst_str),
        op_immediate(shr_src, shift_n, _),
        if *shift_n >= 32;

    // Unsigned magic-divide with the magic pre-loaded in a register, bound to the IMUL by REGISTER IDENTITY rather than address order; the |2^k/d - magic| <= 1 recover step is the safety net.

    // reg_loads_magic: a register loaded with a 32-bit immediate, kept as its unsigned low-32 value since capstone may sign-extend it.
    #[local] relation reg_loads_magic(Ireg, i64);
    reg_loads_magic(Ireg::from(dst_str), (*magic as u32) as i64) <--
        pmov(mov_addr, mov_dst, mov_src),
        instruction(mov_addr, _, _, "MOV", _, _, _, _, _, _),
        op_register(mov_dst, dst_str),
        op_immediate(mov_src, magic, _);

    // imul_dst_unwritten_until: cur_addr is reached over the literal next-chain with nothing between writing dst_ireg, bounded to a short window; next is adjacency, not a reachability test.
    #[local] relation imul2_magic(Address, Ireg, Ireg, i64);
    imul2_magic(imul_addr, Ireg::from(imul_dst_str), Ireg::from(imul_src_str), *magic_masked) <--
        pimul(imul_addr, imul_dst, imul_src),
        op_register(imul_dst, imul_dst_str),
        reg_is_64(imul_dst_str, true),
        op_register(imul_src, imul_src_str),
        if Ireg::from(imul_src_str) != Ireg::from(imul_dst_str),
        reg_loads_magic(Ireg::from(imul_src_str), magic_masked);

    // Depth-bounded (<=4 hops): the SHR is at most a couple of interleaved ops past the IMUL.
    #[local] relation imul2_dst_unwritten(Address, Address, Ireg, usize);
    imul2_dst_unwritten(imul_addr, next_addr, *dst_ireg, 1) <--
        imul2_magic(imul_addr, dst_ireg, _, _),
        next(imul_addr, next_addr);
    imul2_dst_unwritten(imul_addr, next_addr, *dst_ireg, h+1) <--
        imul2_dst_unwritten(imul_addr, cur_addr, dst_ireg, h),
        if *h < 4,
        ireg_of(preg, dst_ireg),
        preg_of(dst_mreg, preg),
        !reg_def(cur_addr, dst_mreg),
        next(cur_addr, next_addr);

    // div_magic_2op: a recovered unsigned magic divide, where quot_ireg is the IMUL/SHR dst and dividend_ireg is the value the IMUL read.
    #[local] relation div_magic_2op(Address, Address, i64, Ireg, Ireg);

    // The SHR of the IMUL dst, k>=32, reached without an intervening overwrite of that register.
    #[local] relation imul2_shr(Address, Address, Ireg, i64);
    imul2_shr(imul_addr, shr_addr, *dst_ireg, *shift_n) <--
        imul2_magic(imul_addr, dst_ireg, _, _),
        next(imul_addr, shr_addr),
        pshr(shr_addr, shr_dst, shr_src),
        op_register(shr_dst, shr_dst_str),
        if Ireg::from(shr_dst_str) == *dst_ireg,
        op_immediate(shr_src, shift_n, _),
        if *shift_n >= 32;
    imul2_shr(imul_addr, shr_addr, *dst_ireg, *shift_n) <--
        imul2_dst_unwritten(imul_addr, cur_addr, dst_ireg, _),
        next(cur_addr, shr_addr),
        ireg_of(preg, dst_ireg),
        preg_of(dst_mreg, preg),
        !reg_def(cur_addr, dst_mreg),
        pshr(shr_addr, shr_dst, shr_src),
        op_register(shr_dst, shr_dst_str),
        if Ireg::from(shr_dst_str) == *dst_ireg,
        op_immediate(shr_src, shift_n, _),
        if *shift_n >= 32;

    // Dividend feeding the IMUL: SRC when the preceding instruction is a reg-to-reg copy into the IMUL dst, otherwise the IMUL dst already held it.
    #[local] relation imul2_dividend(Address, Ireg, Ireg);
    imul2_dividend(imul_addr, Ireg::from(src_str), *quot_ireg) <--
        imul2_magic(imul_addr, quot_ireg, _, _),
        prev_instr(imul_addr, prev_addr),
        pmov(prev_addr, mov_dst, mov_src),
        instruction(prev_addr, _, _, "MOV", _, _, _, _, _, _),
        op_register(mov_dst, mov_dst_str),
        if Ireg::from(mov_dst_str) == *quot_ireg,
        op_register(mov_src, src_str),
        if Ireg::from(src_str) != *quot_ireg;
    imul2_dividend(imul_addr, *quot_ireg, *quot_ireg) <--
        imul2_magic(imul_addr, quot_ireg, _, _),
        prev_instr(imul_addr, prev_addr),
        !magic_prev_mov_into_dst(imul_addr, prev_addr);

    // Helper: the instruction before the IMUL is a reg-to-reg `mov IMUL_DST, SRC`.
    #[local] relation magic_prev_mov_into_dst(Address, Address);
    magic_prev_mov_into_dst(imul_addr, prev_addr) <--
        imul2_magic(imul_addr, quot_ireg, _, _),
        prev_instr(imul_addr, prev_addr),
        pmov(prev_addr, mov_dst, mov_src),
        instruction(prev_addr, _, _, "MOV", _, _, _, _, _, _),
        op_register(mov_dst, mov_dst_str),
        if Ireg::from(mov_dst_str) == *quot_ireg,
        op_register(mov_src, src_str),
        if Ireg::from(src_str) != *quot_ireg;

    div_magic_2op(imul_addr, shr_addr, divisor, *dividend_ireg, *quot_ireg) <--
        imul2_magic(imul_addr, quot_ireg, _magic_reg, magic),
        imul2_shr(imul_addr, shr_addr, _, shift_n),
        imul2_dividend(imul_addr, dividend_ireg, _),
        let divisor_opt = recover_unsigned_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        if divisor >= 2 && divisor <= 1_000_000_000;

    // Emit Odivuimm at the IMUL address (capstone reg_use there covers the dividend register).
    mach_inst(imul_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *quot_mreg)) <--
        div_magic_2op(imul_addr, _shr_addr, divisor, dividend_ireg, quot_ireg),
        ireg_of(preg_of_input, dividend_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_quot, quot_ireg),
        preg_of(quot_mreg, preg_of_quot),
        let args = vec![*input_mreg],
        let op = Operation::Odivuimm(*divisor);

    // The dividend register is not necessarily a capstone operand of the IMUL, so register its use here for reaching-defs to bind the divide's input to its defining value.
    reg_use(imul_addr, *input_mreg) <--
        div_magic_2op(imul_addr, _shr_addr, _divisor, dividend_ireg, _quot_ireg),
        ireg_of(preg_of_input, dividend_ireg),
        preg_of(input_mreg, preg_of_input);

    // Suppress the raw IMUL and SHR; any intervening unrelated instruction stays, since it does not touch the dividend/quotient register.
    div_consumed(imul_addr) <-- div_magic_2op(imul_addr, _, _, _, _);
    div_consumed(shr_addr) <-- div_magic_2op(_, shr_addr, _, _, _);

    // Seed the modulo synthesis chain so an x % K next to the recovered x / K is also lifted; the endpoint is the SHR, where the quotient lands in quot_ireg.
    divide_q(shr_addr, *quot_ireg, *divisor, *dividend_ireg, false) <--
        div_magic_2op(_imul_addr, shr_addr, divisor, dividend_ireg, quot_ireg);

    // Sign correction: SAR/SHR $31/$63 + SUB (signed division fixup)
    #[local] relation sign_correction(Address, Address, Ireg, Ireg, i64);
    sign_correction(shift_addr, sub_addr, Ireg::from(sign_str), Ireg::from(sub_dst_str), *sign_shift) <--
        pshift_right(shift_addr, sign_dst, sign_src),
        op_register(sign_dst, sign_str),
        op_immediate(sign_src, sign_shift, _),
        if *sign_shift == 31 || *sign_shift == 63,
        next(shift_addr, sub_addr),
        psub(sub_addr, sub_dst, sub_src),
        op_register(sub_dst, sub_dst_str),
        op_register(sub_src, sub_src_str),
        if Ireg::from(sub_src_str) == Ireg::from(sign_str);

    // Marks a div_magic_32 chain whose quotient WAS lowered, gating the raw-instruction suppression so a shape the consumers do not fully match keeps its correct magic arithmetic.
    #[local] relation div_magic_32_lowered(Address);

    // Signed div (32-bit): MOVSXD -> IMUL3 -> SHR/SAR -> sign_correction
    mach_inst(addr0, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)),
    div_magic_32_lowered(addr0) <--
        div_magic_32(addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        next(shift_addr, sign_addr),
        sign_correction(sign_addr, _sub_addr, sign_ireg, sub_dst_ireg, sign_shift),
        if *sign_shift == 31,
        if sign_ireg == input_ireg,
        if sub_dst_ireg == result_ireg,
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, sub_dst_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    // Signed div (32-bit): MOVSXD -> IMUL3 -> SAR -> MOV copy,input -> SAR copy,31 -> SUB result,copy. GCC -O1 sign-extracts via a *copy* of the input (scheduling places copy inside sign-correction).
    mach_inst(addr0, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)),
    div_magic_32_lowered(addr0) <--
        div_magic_32(addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        // input_copy = input
        next(shift_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_dst, copy_str),
        op_register(mov_src, copy_src_str),
        if Ireg::from(copy_src_str) == *input_ireg,
        if Ireg::from(copy_str) != *input_ireg,
        next(mov_addr, sar_addr),
        sign_correction(sar_addr, _sub_addr, sign_ireg, sub_dst_ireg, 31),
        if *sign_ireg == Ireg::from(copy_str),
        if sub_dst_ireg == result_ireg,
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, sub_dst_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivimm(divisor);

    // Mark the input-copy MOV and the SAR after it as div_consumed too, so the raw mov/sar emissions are suppressed.
    div_consumed(mov_addr) <--
        div_magic_32(_, shift_addr, _, _, in_ireg, _),
        next(shift_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_dst, copy_str),
        op_register(mov_src, copy_src_str),
        if Ireg::from(copy_src_str) == *in_ireg,
        if Ireg::from(copy_str) != *in_ireg,
        next(mov_addr, sar_addr),
        sign_correction(sar_addr, _, sign_ireg, _, 31),
        if *sign_ireg == Ireg::from(copy_str);
    div_consumed(sar_addr) <--
        div_magic_32(_, shift_addr, _, _, in_ireg, _),
        next(shift_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_dst, copy_str),
        op_register(mov_src, copy_src_str),
        if Ireg::from(copy_src_str) == *in_ireg,
        if Ireg::from(copy_str) != *in_ireg,
        next(mov_addr, sar_addr),
        sign_correction(sar_addr, _, sign_ireg, _, 31),
        if *sign_ireg == Ireg::from(copy_str);
    div_consumed(sub_addr) <--
        div_magic_32(_, shift_addr, _, _, in_ireg, _),
        next(shift_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_dst, copy_str),
        op_register(mov_src, copy_src_str),
        if Ireg::from(copy_src_str) == *in_ireg,
        if Ireg::from(copy_str) != *in_ireg,
        next(mov_addr, sar_addr),
        sign_correction(sar_addr, sub_addr, sign_ireg, _, 31),
        if *sign_ireg == Ireg::from(copy_str);

    // Signed div (64-bit): MOV $MAGIC -> IMUL -> SAR -> sign_correction. Emit Mop at IMUL address (capstone reports reg_use of both IMUL operands there).
    mach_inst(imul_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)) <--
        div_magic_64(addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        next(addr0, imul_addr),
        next(shift_addr, sign_addr),
        sign_correction(sign_addr, _sub_addr, _sign_ireg, _sub_dst_ireg, 63),
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, result_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivlimm(divisor);

    // Unsigned div: MOV $MAGIC -> IMUL -> SHR (no sign correction).
    mach_inst(imul_addr, MachInst::Mop(op.clone(), Arc::new(args.clone()), *result_mreg)) <--
        div_magic_64(addr0, _shift_addr, magic, shift_n, input_ireg, result_ireg),
        next(addr0, imul_addr),
        // Unsigned: the SHR is logical (already in div_magic_64) and no sign_correction follows
        let divisor_opt = recover_unsigned_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, result_ireg),
        preg_of(result_mreg, preg_of_result),
        let args = vec![*input_mreg],
        let op = Operation::Odivuimm(divisor);


    // Mark div-magic-consumed instructions so generic rules skip them.
    #[local] relation div_consumed(Address);

    // 32-bit: MOVSXD(addr0) -> IMUL(addr1) -> SHIFT(addr2) -> SAR(addr3) -> SUB(addr4); MOVSXD also marked consumed since the Odivimm Mop is re-emitted there (suppress its raw Ocast32signed). Gated on div_magic_32_lowered so an unmatched magic shape (no Odivimm emitted) keeps its raw instructions.
    div_consumed(movsxd_addr) <--
        div_magic_32(movsxd_addr, _, _, _, _, _),
        div_magic_32_lowered(movsxd_addr);
    div_consumed(imul_addr) <--
        div_magic_32(movsxd_addr, _, _, _, _, _),
        div_magic_32_lowered(movsxd_addr),
        next(movsxd_addr, imul_addr);
    div_consumed(shift_addr) <--
        div_magic_32(movsxd_addr, shift_addr, _, _, _, _),
        div_magic_32_lowered(movsxd_addr);
    div_consumed(sar_addr) <--
        div_magic_32(movsxd_addr, shift_addr, _, _, _, _),
        div_magic_32_lowered(movsxd_addr),
        next(shift_addr, sar_addr);
    div_consumed(sub_addr) <--
        div_magic_32(movsxd_addr, shift_addr, _, _, _, _),
        div_magic_32_lowered(movsxd_addr),
        next(shift_addr, sar_addr),
        next(sar_addr, sub_addr);

    // 64-bit: MOV(addr0) -> IMUL(addr1) -> SHIFT(addr2) [-> SAR(addr3) -> SUB(addr4)]
    div_consumed(mov_addr) <--
        div_magic_64(mov_addr, _, _, _, _, _);
    div_consumed(imul_addr) <--
        div_magic_64(mov_addr, _, _, _, _, _),
        next(mov_addr, imul_addr);
    div_consumed(shift_addr) <--
        div_magic_64(_, shift_addr, _, _, _, _);
    div_consumed(sar_addr) <--
        div_magic_64(_, shift_addr, _, _, _, _),
        next(shift_addr, sar_addr);
    div_consumed(sub_addr) <--
        div_magic_64(_, shift_addr, _, _, _, _),
        next(shift_addr, sar_addr),
        next(sar_addr, sub_addr);

    // Magic-modulo recognition: x86-64 lowers `n % K` as q = magic-divide(n, K); t = q * K (1-3 LEA/ADD/IMUL); r = n - t. Synthesize one Omodimm/Omodlimm Mop at the SUB; upstream Odivimm/Odivlimm stays for sources using both q and r.

    // divide_q records each div_magic_* chain endpoint and the quotient reg there. Tuple: (end_addr last instr, q_ireg quotient reg, divisor K, input_ireg dividend, is_long false=Omodimm/true=Omodlimm).
    #[local] relation divide_q(Address, Ireg, i64, Ireg, bool);

    // div_magic_32 variant 1: MOVSXD->IMUL3->SAR->sign_correction(SAR31, SUB).
    divide_q(sub_addr, *result_ireg, divisor, *input_ireg, false) <--
        div_magic_32(_addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(shift_addr, sign_addr),
        sign_correction(sign_addr, sub_addr, sign_ireg, sub_dst_ireg, 31),
        if sign_ireg == input_ireg,
        if sub_dst_ireg == result_ireg;

    // div_magic_32 variant 2: MOVSXD->IMUL3->SAR->MOV(copy)->SAR->SUB.
    divide_q(sub_addr, *result_ireg, divisor, *input_ireg, false) <--
        div_magic_32(_addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(shift_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_dst, copy_str),
        op_register(mov_src, copy_src_str),
        if Ireg::from(*copy_src_str) == *input_ireg,
        if Ireg::from(*copy_str) != *input_ireg,
        next(mov_addr, sar_addr),
        sign_correction(sar_addr, sub_addr, sign_ireg, sub_dst_ireg, 31),
        if *sign_ireg == Ireg::from(*copy_str),
        if sub_dst_ireg == result_ireg;

    // div_magic_32_compensating_gcc: 7 instructions ending at the SUB.
    divide_q(addr6, *result_ireg, divisor, *input_ireg, false) <--
        div_magic_32_compensating_gcc(addr0, _, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor_compensating(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(addr0, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, a5),
        next(a5, addr6);

    // div_magic_32_compensating (clang): 8 instructions ending at the final ADD.
    divide_q(addr7, *result_ireg, divisor, *input_ireg, false) <--
        div_magic_32_compensating(addr0, _, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor_compensating(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(addr0, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, a5),
        next(a5, a6),
        next(a6, addr7);

    // div_magic_32_signbias_partial (clang n/10 partial-reg): 6 instrs (movsxd + 5), end at ADD; q in partial_ireg. The relation's 5th field is MOVSXD's dst; use MOVSXD's src for divide_q so the mod Mop's input is the original param, not the suppressed cast dst.
    divide_q(addr5, *partial_ireg, divisor, Ireg::from(*movsxd_src_str), false) <--
        div_magic_32_signbias_partial(movsxd_addr, imul_addr, magic, total_shift, _result_ireg, partial_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        pmov(movsxd_addr, _movsxd_dst, movsxd_src),
        op_register(movsxd_src, movsxd_src_str),
        next(imul_addr, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, addr5);

    // div_magic_32_signbias (clang small magic, n/3 etc.): 6 instrs ending at ADD.
    divide_q(addr5, *result_ireg, divisor, *input_ireg, false) <--
        div_magic_32_signbias(addr0, _, magic, total_shift, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *total_shift),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(addr0, a1),
        next(a1, a2),
        next(a2, a3),
        next(a3, a4),
        next(a4, addr5);

    // div_magic_64 (signed): MOV->IMUL->SHR/SAR->sign_correction(SAR63, SUB).
    divide_q(sub_addr, *result_ireg, divisor, *input_ireg, true) <--
        div_magic_64(_addr0, shift_addr, magic, shift_n, input_ireg, result_ireg),
        let divisor_opt = recover_signed_divisor(*magic, *shift_n),
        if divisor_opt.is_some(),
        let divisor = divisor_opt.unwrap(),
        next(shift_addr, sign_addr),
        sign_correction(sign_addr, sub_addr, _, _, 63);

    // q_factor(addr, ireg, factor, input_ireg, divisor, is_long): after addr, `ireg` holds factor * (input_ireg / divisor). Invariant carried only through LEA/ADD/IMUL3/MOV; other writers drop the entry.
    #[local] relation q_factor(Address, Ireg, i64, Ireg, i64, bool);

    // Bootstrap: divide endpoint contributes q_factor=1 for the q register.
    q_factor(end_addr, *q_ireg, 1, *input_ireg, *divisor, *is_long) <--
        divide_q(end_addr, q_ireg, divisor, input_ireg, is_long);

    // Generation: LEA dst, [r,r,scale] with disp=0  =>  dst = factor * (scale+1).
    q_factor(lea_addr, dst_ireg, new_factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, base_ireg, factor, input, divisor, is_long),
        next(prev_addr, lea_addr),
        plea(lea_addr, lea_dst, lea_src),
        op_indirect(lea_src, _, base_str, idx_str, scale, 0, _),
        if Ireg::from(*base_str) == *base_ireg,
        if Ireg::from(*idx_str) == *base_ireg,
        if *scale >= 1 && *scale <= 8,
        let new_factor = factor.checked_mul(scale + 1).unwrap_or(0),
        if new_factor != 0,
        op_register(lea_dst, dst_str),
        let dst_ireg = Ireg::from(*dst_str);

    // Generation: ADD dst, src (both regs hold q-factors)  =>  dst gets f_dst + f_src.
    q_factor(add_addr, dst_ireg, new_factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, dst_ireg, f_dst, input, divisor, is_long),
        q_factor(prev_addr, src_ireg, f_src, input, divisor, is_long),
        next(prev_addr, add_addr),
        padd(add_addr, dst_sym, src_sym),
        op_register(dst_sym, dst_str),
        op_register(src_sym, src_str),
        if Ireg::from(*dst_str) == *dst_ireg,
        if Ireg::from(*src_str) == *src_ireg,
        let new_factor = f_dst.checked_add(*f_src).unwrap_or(0),
        if new_factor != 0;

    // Generation: IMUL3 dst, src, imm  =>  dst = f_src * imm.
    q_factor(imul_addr, dst_ireg, new_factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, src_ireg, f_src, input, divisor, is_long),
        next(prev_addr, imul_addr),
        pimul3(imul_addr, imul_dst, imul_src, imul_imm),
        op_register(imul_src, src_str),
        if Ireg::from(*src_str) == *src_ireg,
        op_immediate(imul_imm, imm, _),
        let new_factor = f_src.checked_mul(*imm).unwrap_or(0),
        if new_factor != 0,
        op_register(imul_dst, dst_str),
        let dst_ireg = Ireg::from(*dst_str);

    // Generation: MOV dst, src (reg-to-reg copy)  =>  dst gets f_src.
    q_factor(mov_addr, dst_ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, src_ireg, factor, input, divisor, is_long),
        next(prev_addr, mov_addr),
        pmov(mov_addr, mov_dst, mov_src),
        op_register(mov_src, src_str),
        if Ireg::from(*src_str) == *src_ireg,
        op_register(mov_dst, dst_str),
        let dst_ireg = Ireg::from(*dst_str),
        if dst_ireg != *src_ireg;

    // Carry-forward: LEA/ADD/IMUL3/MOV (the four q*K participants) preserve q_factor when they don't overwrite ireg.
    q_factor(lea_addr, *ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, ireg, factor, input, divisor, is_long),
        next(prev_addr, lea_addr),
        plea(lea_addr, lea_dst, _),
        op_register(lea_dst, dst_str),
        if Ireg::from(*dst_str) != *ireg;

    q_factor(add_addr, *ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, ireg, factor, input, divisor, is_long),
        next(prev_addr, add_addr),
        padd(add_addr, add_dst, _),
        op_register(add_dst, dst_str),
        if Ireg::from(*dst_str) != *ireg;

    q_factor(imul_addr, *ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, ireg, factor, input, divisor, is_long),
        next(prev_addr, imul_addr),
        pimul3(imul_addr, imul_dst, _, _),
        op_register(imul_dst, dst_str),
        if Ireg::from(*dst_str) != *ireg;

    q_factor(mov_addr, *ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, ireg, factor, input, divisor, is_long),
        next(prev_addr, mov_addr),
        pmov(mov_addr, mov_dst, _),
        op_register(mov_dst, dst_str),
        if Ireg::from(*dst_str) != *ireg;

    // Memory-store MOV (`mov %reg, [mem]`) writes to memory, not to any register, so q_factor for every register is preserved.
    q_factor(mov_addr, *ireg, factor, *input, *divisor, *is_long) <--
        q_factor(prev_addr, ireg, factor, input, divisor, is_long),
        next(prev_addr, mov_addr),
        pmov(mov_addr, mov_dst, _),
        !op_register(mov_dst, _);

    // Modulo synthesis: SUB n, qK_reg where qK_reg holds factor=K * q. Tuple: (sub_addr, dst_mreg, input_mreg, divisor, is_long).
    #[local] relation mod_synth(Address, Mreg, Mreg, i64, bool);

    // Dividend value tracking: which regs hold input_ireg at sub_addr. Simple sub_dst==input_ireg misses MOVSXD-then-SUB cases (e.g. `movsxd rax,edi; sub eax,ecx` where EAX is a different Ireg from EDI).
    #[local] relation dividend_holder(Address, Ireg, Ireg, i64, bool);

    dividend_holder(*sub_addr, *input_ireg, *input_ireg, *divisor, *is_long) <--
        psub(sub_addr, _, _),
        divide_q(_, _, divisor, input_ireg, is_long);

    // low-32 alias of any MOVSXD/MOV; divide_q drives so the mov_src filter prunes pmov early and psub is crossed LAST, since leading with psub made every subtract multiply the cross-product (36s).
    dividend_holder(*sub_addr, low32_ireg, *input_ireg, *divisor, *is_long) <--
        divide_q(_, _, divisor, input_ireg, is_long),
        pmov(_mov_addr, mov_dst_sym, mov_src_sym),
        op_register(mov_src_sym, mov_src_str),
        if Ireg::from(*mov_src_str) == *input_ireg,
        op_register(mov_dst_sym, mov_dst_str),
        let low32_ireg = Ireg::from(reg_low32_alias(*mov_dst_str)),
        psub(sub_addr, _, _);

    // full-width MOVSXD dst (e.g., RAX itself when source is EDI).
    dividend_holder(*sub_addr, mov_dst_full_ireg, *input_ireg, *divisor, *is_long) <--
        divide_q(_, _, divisor, input_ireg, is_long),
        pmov(_mov_addr, mov_dst_sym, mov_src_sym),
        op_register(mov_src_sym, mov_src_str),
        if Ireg::from(*mov_src_str) == *input_ireg,
        op_register(mov_dst_sym, mov_dst_str),
        let mov_dst_full_ireg = Ireg::from(*mov_dst_str),
        psub(sub_addr, _, _);

    mod_synth(sub_addr, *result_mreg, *input_mreg, *divisor, *is_long) <--
        psub(sub_addr, sub_dst_sym, sub_src_sym),
        op_register(sub_src_sym, sub_src_str),
        op_register(sub_dst_sym, sub_dst_str),
        let sub_src_ireg = Ireg::from(*sub_src_str),
        let sub_dst_ireg = Ireg::from(*sub_dst_str),
        prev_instr(sub_addr, prev_addr),
        q_factor(prev_addr, sub_src_ireg, divisor, input_ireg, divisor_q, is_long),
        if *divisor == *divisor_q,
        dividend_holder(*sub_addr, sub_dst_ireg, input_ireg, divisor, is_long),
        ireg_of(preg_of_input, input_ireg),
        preg_of(input_mreg, preg_of_input),
        ireg_of(preg_of_result, &sub_dst_ireg),
        preg_of(result_mreg, preg_of_result);

    // Emit the modulus Mop at the SUB address.
    mach_inst(sub_addr, MachInst::Mop(op, Arc::new(args.clone()), *result_mreg)) <--
        mod_synth(sub_addr, result_mreg, input_mreg, divisor, is_long),
        let args = vec![*input_mreg],
        let op = if *is_long { Operation::Omodlimm(*divisor) } else { Operation::Omodimm(*divisor) };

    // The synthesized Omod reads a dividend register that is not a capstone operand of the SUB, so register the use here or RTL fabricates an undefined dividend (0 % K).
    reg_use(sub_addr, *input_mreg) <--
        mod_synth(sub_addr, _, input_mreg, _, _);

    // Suppress generic Mop emission for the SUB and the q*K chain back to (but excluding) the divide_q endpoint. Bounded search.
    div_consumed(sub_addr) <-- mod_synth(sub_addr, _, _, _, _);

    #[local] relation mod_chain_back(Address, Address, usize);

    mod_chain_back(sub_addr, sub_addr, 0) <--
        mod_synth(sub_addr, _, _, _, _);

    mod_chain_back(sub_addr, prev_addr, h+1) <--
        mod_chain_back(sub_addr, cur_addr, h),
        if *h < 6,
        prev_instr(cur_addr, prev_addr),
        !divide_q(prev_addr, _, _, _, _);

    div_consumed(addr) <-- mod_chain_back(_, addr, h), if *h > 0;

    // Two-branch JP+JNE float != on registers: match the pair on a shared resolved TARGET ADDRESS via direct_jump, since the disassembler may give the same target distinct operand symbol strings.
    mach_inst(addr0, MachInst::Mcond(
        Condition::Cnotcompf(Comparison::Ceq),
        Arc::new(vec![arg1, arg2]),
        lbl
    )) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        next(addr0, addr1),
        pjcc(addr1, TestCond::CondP, _lbl_p),
        direct_jump(addr1, tgt),
        next(addr1, addr2),
        pjcc(addr2, TestCond::CondNe, lbl),
        direct_jump(addr2, tgt);

    mach_inst(addr0, MachInst::Mcond(
        Condition::Cnotcompfs(Comparison::Ceq),
        Arc::new(vec![arg1, arg2]),
        lbl
    )) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        next(addr0, addr1),
        pjcc(addr1, TestCond::CondP, _lbl_p),
        direct_jump(addr1, tgt),
        next(addr1, addr2),
        pjcc(addr2, TestCond::CondNe, lbl),
        direct_jump(addr2, tgt);


    transl_load_inferred(addr, *chunk, addrmode, regs.clone(), dst) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, r2, idx_str, _scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        let regs = Arc::new(vec![*arg]),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        if !mnem_upper.contains("MOVZ") && !mnem_upper.contains("MOVS");

    // Indexed load (non-SP/non-BP base with index register)
    transl_load_inferred(addr, *chunk, addrmode, regs.clone(), dst) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, r2, idx_str, scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*arg, *idx_arg]),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        if !mnem_upper.contains("MOVZ") && !mnem_upper.contains("MOVS");

    // Scaled-index load with no base register: mov disp(,%idx,scale), dst.
    transl_load_inferred(addr, *chunk, addrmode, regs.clone(), dst) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: None,
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*idx_arg]),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        if !mnem_upper.contains("MOVZ") && !mnem_upper.contains("MOVS");

    transl_load(addr, mc, addrmode, regs, dst) <--
        transl_load_inferred(addr, mc, addrmode, regs, dst);

    transl_load(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(src, _, r2, idx_str, _scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        let regs = Arc::new(vec![*arg]),
        !transl_load_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    // Indexed load (mnemonic-based, non-SP/non-BP base with index register)
    transl_load(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(src, _, r2, idx_str, scale, disp, msize),
        if *r2 != "RBP" && *r2 != "RSP",
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_r2, Ireg::from(r2)),
        preg_of(arg, preg_of_r2),
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*arg, *idx_arg]),
        !transl_load_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    // Scaled-index load with no base register (mnemonic-based): mov disp(,%idx,scale), dst
    transl_load(addr, mc, addrmode, regs.clone(), dst_reg) <--
        pmov(addr, dst, src),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode{
            base: None,
            index: Some((Ireg::from(idx_str), *scale)),
            disp: Displacement::from(*disp),
        },
        ireg_of(preg_of_idx, Ireg::from(idx_str)),
        preg_of(idx_arg, preg_of_idx),
        let regs = Arc::new(vec![*idx_arg]),
        !transl_load_inferred(addr, _, addrmode, _, _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    mach_inst(addr, MachInst::Mload(*memory_chunk, addressing, Arc::new(args), *dst)) <--
        transl_load(addr, memory_chunk, addrmode, regs, dst),
        effective_address_size(addr, address_size),
        !addrmode_needs_resolution(addr, *addrmode, _),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(*addrmode, None, *address_size);

    // Resolved load: transl_load whose addrmode needs symbol resolution (e.g., absolute displacement)
    mach_inst(addr, MachInst::Mload(*memory_chunk, addressing, Arc::new(args), *src)) <--
        transl_load(addr, memory_chunk, addrmode, regs, src),
        effective_address_size(addr, address_size),
        addrmode_needs_resolution(addr, *addrmode, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let resolved = Some((*ident, *offset)),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(*addrmode, resolved, *address_size);

    mach_inst(addr, MachInst::Mload(*mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), dst)) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, mc);

    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), src)) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, base_str, _, _, disp, msize),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        ireg_hold_type(src_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    // Absolute addressing load: mov 0x40402c, %eax (base="NONE", disp is the address)
    mach_inst(addr, MachInst::Mload(*mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), dst)) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, mc);

    // Scaled-index absolute addressing load
    mach_inst(addr, MachInst::Mload(*mc, Addressing::Abasedscaled(*scale, *ident, *offset), Arc::new(vec![idx_mreg]), dst)) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let idx_mreg = Mreg::x86(*idx_str),
        ireg_hold_type(dst_str.to_string(), typ),
        type_to_memchunk(typ, mc);

    mach_inst(addr, MachInst::Mload(mc, Addressing::Abasedscaled(*scale, *ident, *offset), Arc::new(vec![idx_mreg]), dst)) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, scale, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let idx_mreg = Mreg::x86(*idx_str),
        !ireg_hold_type(dst_str.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    // Absolute addressing load (mnemonic-based fallback)
    mach_inst(addr, MachInst::Mload(mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), dst)) <--
        pmov(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, base_str, idx_str, _, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        !ireg_hold_type(dst_str.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem_ext(mnem), *msize);

    // Absolute addressing store: mov %eax, 0x40402c (base="NONE", disp is the address)
    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), src)) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, base_str, idx_str, _, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        ireg_hold_type(src_str.to_string(), typ),
        type_to_memchunk(typ, chunk),
        let mc = refine_chunk_with_size(*chunk, *msize);

    // Absolute addressing store (mnemonic-based fallback)
    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), src)) <--
        pmov(addr, dst, src_sym),
        op_register(src_sym, src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, base_str, idx_str, _, disp, msize),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        !ireg_hold_type(src_str.to_string(), _),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        let mc = refine_chunk_with_size(chunk_from_mnem(mnem), *msize);

    // Absolute addressing immediate store: movl $0x1, 0x40402c (base="NONE", disp is the address)
    mach_inst(addr, MachInst::Mop(Operation::Ointconst(imm_int), Arc::new(vec![]), Mreg::DI)) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, sz),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, _, _),
        let imm_int = *imm_sym as i64;

    mach_inst(addr, MachInst::Mstore(mc, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::DI)) <--
        pmov(addr, dst, src),
        op_immediate(src, _, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, sz),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        // 3.8d: carry the operand's true width; refine_chunk_with_size narrows 1/2-byte sizes and keeps 4/8 as int chunks.
        let mc = refine_chunk_with_size(if *sz <= 4 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 }, *sz);


    mach_inst(address, MachInst::Mop(Operation::Omove, Arc::new(args), *res)) <--
        pmov(address, rd, rs),
        !div_consumed(address),
        op_register(rd, res_str),
        ireg_of(preg_of_r, Ireg::from(res_str)),
        preg_of(res, preg_of_r),
        op_register(rs, arg_str),
        ireg_of(preg_of_rs, Ireg::from(arg_str)),
        preg_of(arg, preg_of_rs),
        let args = vec![*arg];

    mach_inst(address, MachInst::Mop(Operation::Omove, Arc::new(vec![src]), dst)) <--
        pmov(address, rd, rs),
        !div_consumed(address),
        op_register(rd, dst_str),
        reg_xmm(dst_str),
        op_register(rs, src_str),
        let dst = Mreg::x86(dst_str),
        let src = Mreg::x86(src_str);

    mach_inst(address, MachInst::Mop(Operation::Omove, Arc::new(vec![src]), dst)) <--
        pmov(address, rd, rs),
        !div_consumed(address),
        op_register(rd, dst_str),
        op_register(rs, src_str),
        reg_xmm(src_str),
        let dst = Mreg::x86(dst_str),
        let src = Mreg::x86(src_str);

    mach_inst(address, MachInst::Mop(Operation::Ointconst(imm_int), Arc::new(args), *res)) <--
        pmov(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, res_str),
        !reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Olongconst(imm_int), Arc::new(args), *res)) <--
        pmov(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, res_str),
        reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Ointconst(0), Arc::new(args), *res)) <--
        pxor(address, r, r0),
        if r == r0,
        op_register(r, res_str),
        !reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Olongconst(0), Arc::new(args), *res)) <--
        pxor(address, r, r0),
        if r == r0,
        op_register(r, res_str),
        reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        let args = vec![];

    // Divergence: XOR r,r can also be a genuine self-XOR (Asmgen.v: Oxor compiles to Pxorl_rr)
    mach_inst(address, MachInst::Mop(Operation::Oxor, Arc::new(args), *res)) <--
        pxor(address, r, r0),
        if r == r0,
        op_register(r, res_str),
        !reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        let args = vec![*res, *res];

    mach_inst(address, MachInst::Mop(Operation::Oxorl, Arc::new(args), *res)) <--
        pxor(address, r, r0),
        if r == r0,
        op_register(r, res_str),
        reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        let args = vec![*res, *res];

    mach_inst(address, MachInst::Mop(Operation::Oindirectsymbol(*imm_str as usize), Arc::new(args), *res)) <--
        pmov(address, r, symid),
        op_immediate(symid, imm_str, _),
        op_register(r, res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        if *imm_str != 0,
        symbols(*imm_str as Address, _, _),
        !function_symbol(*imm_str as Address, _),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Ointconst(*imm_str as i64), Arc::new(args), *res)) <--
        pmov(address, r, symid),
        !div_consumed(address),
        op_immediate(symid, imm_str, _),
        op_register(r, res_str),
        !reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Olongconst(*imm_str as i64), Arc::new(args), *res)) <--
        pmov(address, r, symid),
        !div_consumed(address),
        op_immediate(symid, imm_str, _),
        op_register(r, res_str),
        reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Ointconst(*imm_str), Arc::new(args), *res)) <--
        pmov(address, r, symid),
        !div_consumed(address),
        op_immediate(symid, imm_str, _),
        op_register(r, res_str),
        !reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        if *imm_str != 0,
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Olongconst(*imm_str), Arc::new(args), *res)) <--
        pmov(address, r, symid),
        !div_consumed(address),
        op_immediate(symid, imm_str, _),
        op_register(r, res_str),
        reg_64(res_str),
        let res_ireg = Ireg::from(res_str),
        ireg_of(preg_of_r, res_ireg),
        preg_of(res, preg_of_r),
        if *imm_str != 0,
        !immediate_nonzero_data_symbol(*imm_str as Address),
        let args = vec![];

    mach_inst(address, MachInst::Mop(cast_op, Arc::new(args.clone()), *res)) <--
        pmov(address, r, r1),
        !div_consumed(address),
        op_register(r1, r1_str),
        op_register(r, r_str),
        reg_is_64(r1_str, r1_is_64),
        reg_is_64(r_str, r_is_64),
        if r1_is_64 != r_is_64,
        ireg_of(preg_of_r1, Ireg::from(r1_str)),
        preg_of(a1, preg_of_r1),
        ireg_of(preg_of_res, Ireg::from(r_str)),
        preg_of(res, preg_of_res),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        let args = vec![*a1],
        let cast_op = if mnem_upper.starts_with("MOVZX") || mnem_upper.starts_with("MOVZB") {
            if is_reg_8(r1_str) {
                Operation::Ocast8unsigned
            } else {
                Operation::Ocast16unsigned
            }
        } else if mnem_upper.starts_with("MOVSXD") || mnem_upper.starts_with("CDQE") {
            Operation::Ocast32signed
        } else if mnem_upper.starts_with("MOVSX") || mnem_upper.starts_with("MOVSB") || mnem_upper.starts_with("MOVSW") {
            if is_reg_8(r1_str) {
                Operation::Ocast8signed
            } else {
                Operation::Ocast16signed
            }
        } else {
            Operation::Ocast8signed
        };

    // Sub-32-bit sign/zero extension via MOVSX/MOVZX from 8/16-bit source (prior rule only fires for reg_is_64 operands).
    mach_inst(address, MachInst::Mop(cast_op, Arc::new(args.clone()), *res)) <--
        pmov(address, r, r1),
        op_register(r1, r1_str),
        op_register(r, r_str),
        if is_reg_8(r1_str) || is_reg_16(r1_str),
        if !is_reg_8(r_str) && !is_reg_16(r_str),
        ireg_of(preg_of_r1, Ireg::from(r1_str)),
        preg_of(a1, preg_of_r1),
        ireg_of(preg_of_res, Ireg::from(r_str)),
        preg_of(res, preg_of_res),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        let args = vec![*a1],
        let cast_op = if mnem_upper.starts_with("MOVZX") || mnem_upper.starts_with("MOVZB") || mnem_upper.starts_with("MOVZW") {
            if is_reg_8(r1_str) { Operation::Ocast8unsigned } else { Operation::Ocast16unsigned }
        } else if mnem_upper.starts_with("MOVSX") || mnem_upper.starts_with("MOVSB") || mnem_upper.starts_with("MOVSW") {
            if is_reg_8(r1_str) { Operation::Ocast8signed } else { Operation::Ocast16signed }
        } else {
            if is_reg_8(r1_str) { Operation::Ocast8signed } else { Operation::Ocast16signed }
        };

    mach_inst(address, MachInst::Mop(Operation::Oneg, Arc::new(args), *res)) <--
        pneg(address, r),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        psub(address, r, r2),
        !div_consumed(address),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2],
        let op = if *r_is_64 { Operation::Osubl } else { Operation::Osub };

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        psbb(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2],
        let op = if *r_is_64 { Operation::Osubl } else { Operation::Osub };

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        padc(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2],
        let op = if *r_is_64 { Operation::Oaddl } else { Operation::Oadd };

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        psbb(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        !adc_carry_lifted(address),
        let args = vec![*res],
        let op = if *r_is_64 { Operation::Oaddlimm(-imm_int) } else { Operation::Oaddimm(-imm_int) };

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        padc(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        !adc_carry_lifted(address),
        let args = vec![*res],
        let op = if *r_is_64 { Operation::Oaddlimm(imm_int) } else { Operation::Oaddimm(imm_int) };

    // cmp + adc/sbb is branchless conditional-accumulate: model CF as the 0/1 result of the compare's carry condition so rtl_pass can expand it, instead of discarding CF and lowering a dead acc += imm.
    #[local] relation padc_or_sbb(Address);
    padc_or_sbb(addr) <-- padc(addr, _, _);
    padc_or_sbb(addr) <-- psbb(addr, _, _);

    #[local] relation cmp_adc_link(Address, Address);
    cmp_adc_link(addr0, addr1) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        padc_or_sbb(addr1);
    cmp_adc_link(addr0, addr1) <--
        flags_and_jump_pair(addr0, addr1, _),
        pcmp(addr0, _, _),
        padc_or_sbb(addr1);
    cmp_adc_link(addr0, addr2) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        padc_or_sbb(addr2);
    cmp_adc_link(addr0, addr3) <--
        pcmp(addr0, _, _),
        next(addr0, addr1),
        instruction(addr1, _, _, mnem1, _, _, _, _, _, _),
        if !is_flag_setting(mnem1),
        next(addr1, addr2),
        instruction(addr2, _, _, mnem2, _, _, _, _, _, _),
        if !is_flag_setting(mnem2),
        next(addr2, addr3),
        padc_or_sbb(addr3);

    // adc_carry_op: the carry-aware fact rtl_pass expands, carrying the compare's carry condition, the add-vs-subtract selector, the residual constant, and the accumulator width.
    relation adc_carry_op(Address, Address, Condition, bool, i64, bool, Mreg, Mreg);
    #[local] relation adc_carry_lifted(Address);
    adc_carry_lifted(adc_addr) <-- adc_carry_op(adc_addr, _, _, _, _, _, _, _);

    // adc $imm,%acc with carry from `cmp $cimm,%creg`: acc = acc + CF + imm, CF = (creg <u cimm).
    adc_carry_op(adc_addr, cmp_addr, cf_cond, false, imm_int, acc_is_64, acc_mreg, cmp_mreg) <--
        padc(adc_addr, acc_r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(acc_r, acc_str),
        reg_is_64(acc_str, acc_is_64),
        let acc_mreg = Mreg::x86(acc_str),
        cmp_adc_link(cmp_addr, adc_addr),
        pcmp(cmp_addr, cmp_r, cmp_imm),
        op_register(cmp_r, cmp_str),
        op_immediate(cmp_imm, cimm, _),
        reg_is_64(cmp_str, cmp_is_64),
        let cmp_mreg = Mreg::x86(cmp_str),
        let cf_cond = if *cmp_is_64 { Condition::Ccompluimm(Comparison::Clt, *cimm) } else { Condition::Ccompuimm(Comparison::Clt, *cimm) };

    // sbb $imm,%acc with carry from `cmp $cimm,%creg`: acc = acc - CF - imm, CF = (creg <u cimm).
    adc_carry_op(adc_addr, cmp_addr, cf_cond, true, -imm_int, acc_is_64, acc_mreg, cmp_mreg) <--
        psbb(adc_addr, acc_r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(acc_r, acc_str),
        reg_is_64(acc_str, acc_is_64),
        let acc_mreg = Mreg::x86(acc_str),
        cmp_adc_link(cmp_addr, adc_addr),
        pcmp(cmp_addr, cmp_r, cmp_imm),
        op_register(cmp_r, cmp_str),
        op_immediate(cmp_imm, cimm, _),
        reg_is_64(cmp_str, cmp_is_64),
        let cmp_mreg = Mreg::x86(cmp_str),
        let cf_cond = if *cmp_is_64 { Condition::Ccompluimm(Comparison::Clt, *cimm) } else { Condition::Ccompuimm(Comparison::Clt, *cimm) };

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        padd(address, r, r2),
        !div_consumed(address),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2],
        let op = if *r_is_64 { Operation::Oaddl } else { Operation::Oadd };

    // Divergence: ADD reg,reg can also be Olea(Aindexed2(0)); CompCert has no Oadd for x86, register addition is Olea(Aindexed2 0) compiling to LEA r,[r1+r2].
    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        padd(address, r, r2),
        !div_consumed(address),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2],
        let op = if *r_is_64 { Operation::Oleal(Addressing::Aindexed2(0)) } else { Operation::Olea(Addressing::Aindexed2(0)) };

    mach_inst(address, MachInst::Mop(Operation::Olea(addressing), Arc::new(empty_args), *res)) <--
        plea(address, dst_reg, src_addr),
        op_register(dst_reg, dst_str),
        op_indirect(src_addr, _, base_str, idx_str, _, offset, _),
        reg_sp(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rsp_frame_at(address, _),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        let addressing = Addressing::Ainstack(*offset),
        let empty_args = vec![];

    // A stack slot a register is SPILLED into is real storage, so a later read is a value RELOAD, never an address-of; emitting both makes the node ambiguous and it is dropped entirely.
    #[local] relation sp_slot_stored(Address, i64);
    sp_slot_stored(func, *offset) <--
        pmov(addr, dst, src),
        op_register(src, _),
        op_indirect(dst, _, base_str, idx_str, _, offset, _),
        reg_sp(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(addr, func);

    mach_inst(address, MachInst::Mop(Operation::Olea(addressing), Arc::new(empty_args), *res)) <--
        pmov(address, dst_reg, src_addr),
        op_register(dst_reg, dst_str),
        op_indirect(src_addr, _, base_str, idx_str, _, offset, _),
        reg_sp(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(address, func),
        rsp_frame_at(address, func),
        !sp_slot_stored(func, *offset),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        let addressing = Addressing::Ainstack(*offset),
        let empty_args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Olea(addressing), Arc::new(empty_args), *res)) <--
        pmov(address, dst_reg, src_reg),
        op_register(src_reg, src_str),
        reg_sp(src_str),
        op_register(dst_reg, dst_str),
        !reg_sp(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        rsp_frame_offset_at(_, address, rsp_ofs),
        let addressing = Addressing::Ainstack(*rsp_ofs),
        let empty_args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Oindirectsymbol(*ident as usize), Arc::new(empty_args), *res)) <--
        pmov(address, dst_reg, src_addr),
        op_register(dst_reg, dst_str),
        op_immediate(src_addr, symbol_addr, _),
        !reg_64(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        resolved_addr_to_symbol(*symbol_addr as Address, ident, _offset),
        !function_symbol(*symbol_addr as Address, _),
        if *ident != 0,
        let empty_args = vec![];

    mach_inst(address, MachInst::Mop(Operation::Oindirectsymbol(*ident as usize), Arc::new(empty_args), *res)) <--
        pmov(address, dst_reg, src_addr),
        op_register(dst_reg, dst_str),
        op_immediate(src_addr, symbol_addr, _),
        reg_64(dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        resolved_addr_to_symbol(*symbol_addr as Address, ident, _offset),
        !function_symbol(*symbol_addr as Address, _),
        if *ident != 0,
        let empty_args = vec![];

    mach_inst(addr, MachInst::Mcall(Either::Right(Either::Left(symbol_name)))) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_immediate(dst, imm_addr, _),
        symbols(*imm_addr as Address, symbol_name, _);

    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        padd(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res],
        let op = if *r_is_64 { Operation::Oaddlimm(imm_int) } else { Operation::Oaddimm(imm_int) };

    // INC reg / DEC reg: capstone emits these as their own mnemonics (not ADD/SUB), so lower them as add of immediate +/-1; without this they emit no Mach op and are silently dropped, corrupting dataflow (e.g. a setcc then inc idiom loses its +1).
    mach_inst(address, MachInst::Mop(op, Arc::new(vec![*res]), *res)) <--
        instruction(address, _, _, "INC", dst, _, _, _, _, _),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        reg_is_64(dst_str, is_64),
        let op = if *is_64 { Operation::Oaddlimm(1) } else { Operation::Oaddimm(1) };

    mach_inst(address, MachInst::Mop(op, Arc::new(vec![*res]), *res)) <--
        instruction(address, _, _, "DEC", dst, _, _, _, _, _),
        op_register(dst, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        reg_is_64(dst_str, is_64),
        let op = if *is_64 { Operation::Oaddlimm(-1) } else { Operation::Oaddimm(-1) };

    // base="NONE" excluded: Mreg::x86("NONE") is Unknown so the load leg cannot lower and would conflict with the Aglobal float_load_op route.
    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        padd(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        psub(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // ADD/SUB with memory source on SP-relative (stack parameter access)
    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        padd(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        psub(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // ADD/SUB with memory destination, register source: read-modify-write at [mem].
    arith_store_reg(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(src_str)) <--
        padd(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *src_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_reg(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(src_str)) <--
        psub(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *src_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_reg(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(src_str)) <--
        padd(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *src_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // AND/OR/XOR with memory destination: previously emitted arith_load_op (wrong direction), clobbering the source register and silently dropping the memory store (3.8c direction bug).
    arith_store_reg(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(src_str)) <--
        pand(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *src_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_reg(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(src_str)) <--
        por(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *src_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_reg(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(src_str)) <--
        pxor(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *src_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Atomic XADD: the store side is the load-op-store recovered as add %reg,(%mem); capstone's operand order is (reg, mem), so bind op_register on field4 and op_indirect on field5 directly.
    arith_store_reg(*addr, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(reg_str)) <--
        instruction(addr, _, _, mnem, reg_op, mem_op, _, _, _, _),
        if *mnem == "LOCK XADD" || *mnem == "XADD",
        op_register(reg_op, reg_str),
        reg_is_64(reg_str, reg_is_64),
        op_indirect(mem_op, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *reg_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *reg_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Atomic XCHG: the store half is a plain register-to-memory store routed through pmov so transl_store builds the addrmode; operand order is (reg, mem) like XADD.
    pmov(addr, mem_op, reg_op) <--
        instruction(addr, _, _, "XCHG", reg_op, mem_op, _, _, _, _),
        op_register(reg_op, _),
        op_indirect(mem_op, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str);

    // AND/OR/XOR with memory source: previously no rule matched so the instruction vanished (3.8c); mirrors padd/psub arith_load_op, SP/RIP excluded.
    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        pand(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        por(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        pxor(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Literal-SP forms use the same memory-source lowering.  Keep them
    // separate from the generic rules above, whose SP exclusion prevents an
    // unproved current stack coordinate from becoming an ordinary pointer.
    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        pand(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        por(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        pxor(address, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // INDEXED memory-source integer arithmetic, routed through float_load_op
    // whose lowering is operation-agnostic and takes a full Addressing, since
    // arith_load_op excludes indexed addressing.  Literal RSP is retained:
    // RTL lowers a proved frame coordinate through an Ainstack base and the
    // structured stack-safety boundary rejects an unproved one.
    float_load_op(*addr, op, chunk, addressing, Arc::new(vec![base_mreg, idx_mreg]), Mreg::x86(dst_str), false) <--
        padd(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let base_mreg = Mreg::x86(base_str),
        let idx_mreg = Mreg::x86(idx_str),
        let addressing = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) },
        let op = if *dst_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, addressing, Arc::new(vec![base_mreg, idx_mreg]), Mreg::x86(dst_str), false) <--
        psub(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let base_mreg = Mreg::x86(base_str),
        let idx_mreg = Mreg::x86(idx_str),
        let addressing = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) },
        let op = if *dst_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, addressing, Arc::new(vec![base_mreg, idx_mreg]), Mreg::x86(dst_str), false) <--
        pand(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let base_mreg = Mreg::x86(base_str),
        let idx_mreg = Mreg::x86(idx_str),
        let addressing = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) },
        let op = if *dst_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, addressing, Arc::new(vec![base_mreg, idx_mreg]), Mreg::x86(dst_str), false) <--
        por(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let base_mreg = Mreg::x86(base_str),
        let idx_mreg = Mreg::x86(idx_str),
        let addressing = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) },
        let op = if *dst_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, addressing, Arc::new(vec![base_mreg, idx_mreg]), Mreg::x86(dst_str), false) <--
        pxor(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let base_mreg = Mreg::x86(base_str),
        let idx_mreg = Mreg::x86(idx_str),
        let addressing = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) },
        let op = if *dst_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // RIP-relative integer memory-source arithmetic (3.8c): arith_load_op cannot express a global ident and excludes RIP, so routed through float_load_op whose rtl lowering is operation-agnostic.
    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        padd(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        psub(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        pand(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        por(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        pxor(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, _, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Absolute-addressed (clang -fno-pie) memory-source arith: base="NONE" with the address in disp; abs_target_addr rows exist for padd/psub/pand/por/pxor.
    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        padd(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        psub(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        pand(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        por(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    float_load_op(*addr, op, chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::x86(dst_str), false) <--
        pxor(addr, dst, src),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *dst_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // ADD/SUB immediate to memory: RIP excluded because Mreg::x86("RIP") is Unknown and would shadow the ident-keyed arith_store_abs_imm RIP route.
    arith_store_imm(*address, op, chunk, Mreg::x86(base_str), *disp) <--
        padd(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(imm_int) } else { Operation::Oaddlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_imm(*address, op, chunk, Mreg::x86(base_str), *disp) <--
        psub(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(-imm_int) } else { Operation::Oaddlimm(-imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    // AND/OR/XOR immediate to memory: previously no rule matched (reg-dest Oandimm requires op_register on dst) so the instruction vanished (3.8c family).
    arith_store_imm(*address, op, chunk, Mreg::x86(base_str), *disp) <--
        pand(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oandimm(imm_int) } else { Operation::Oandlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_imm(*address, op, chunk, Mreg::x86(base_str), *disp) <--
        por(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oorimm(imm_int) } else { Operation::Oorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_imm(*address, op, chunk, Mreg::x86(base_str), *disp) <--
        pxor(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oxorimm(imm_int) } else { Operation::Oxorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    // Absolute addressing arith: add reg, 0x40402c (read-modify-write at global)
    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        padd(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        psub(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Absolute addressing arith: addl $imm, 0x40402c (read-modify-write at global)
    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        padd(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(imm_int) } else { Operation::Oaddlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        psub(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(-imm_int) } else { Operation::Oaddlimm(-imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    // RIP-relative memory-dest arithmetic (3.8c): absolute-addressing rules require base="NONE"; the RIP form resolved nothing and the whole RMW vanished; arith_store_abs_* is ident-keyed so applies unchanged.
    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        padd(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oaddl } else { Operation::Oadd },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        psub(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Osubl } else { Operation::Osub },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        padd(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(imm_int) } else { Operation::Oaddlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        psub(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oaddimm(-imm_int) } else { Operation::Oaddlimm(-imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    // AND/OR/XOR global RMW (RIP-relative and -fno-pie absolute): previously no rule matched and the whole RMW vanished (3.8c); arith_store_abs_* is op-agnostic.
    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        pand(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        por(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        pxor(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        pand(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oandl } else { Operation::Oand },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        por(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oorl } else { Operation::Oor },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_reg(*address, op, chunk, *ident, *offset, Mreg::x86(src_str)) <--
        pxor(address, dst, src),
        op_register(src, src_str),
        reg_is_64(src_str, src_is_64),
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let op = if *src_is_64 { Operation::Oxorl } else { Operation::Oxor },
        let chunk = if *sz > 4 || *src_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        pand(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oandimm(imm_int) } else { Operation::Oandlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        por(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oorimm(imm_int) } else { Operation::Oorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        pxor(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, disp, sz),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oxorimm(imm_int) } else { Operation::Oxorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        pand(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oandimm(imm_int) } else { Operation::Oandlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        por(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oorimm(imm_int) } else { Operation::Oorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    arith_store_abs_imm(*address, op, chunk, *ident, *offset) <--
        pxor(address, dst, src),
        op_immediate(src, imm_str, _),
        let imm_int = *imm_str as i64,
        op_indirect(dst, _, base_str, idx, _, _, sz),
        if *idx == "NONE" || idx.is_empty(),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset),
        let is_32 = *sz <= 4,
        let op = if is_32 { Operation::Oxorimm(imm_int) } else { Operation::Oxorlimm(imm_int) },
        let chunk = if is_32 { MemoryChunk::MInt32 } else { MemoryChunk::MInt64 };

    // abs_target_addr for and/or/xor RMW destinations so they resolve to synthesized idents; mirrors the padd/psub dst-side rows (the src-side rows cover only the memory-source direction).
    abs_target_addr(addr, target) <--
        pand(addr, dst, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        por(addr, dst, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    abs_target_addr(addr, target) <--
        pxor(addr, dst, _),
        op_indirect(dst, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        let target = *disp as Address;

    mach_inst(address, MachInst::Mop(Operation::Omul, Arc::new(args), *res)) <--
        pimul(address, r, r2),
        !div_consumed(address),
        op_register(r, r_str),
        ireg_hold_type(r_str.to_string(), typ),
        if *typ == Typ::Tint,
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];


    mach_inst(address, MachInst::Mop(Operation::Omulimm(imm_int), Arc::new(args), *res)) <--
        pimul(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    // 3-operand IMUL (32-bit): imul dst, src, imm; suppressed inside div-by-constant.
    mach_inst(address, MachInst::Mop(Operation::Omulimm(imm_int), Arc::new(args), *res)) <--
        pimul3(address, dst, src, imm),
        !div_consumed(address),
        op_immediate(imm, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(dst, dst_str),
        ireg_hold_type(dst_str.to_string(), typ),
        if *typ == Typ::Tint,
        op_register(src, src_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(src_arg, preg_of_src),
        let args = vec![*src_arg];

    // 3-operand IMUL (64-bit): imul dst, src, imm; suppressed inside div-by-constant.
    mach_inst(address, MachInst::Mop(Operation::Omullimm(imm_int), Arc::new(args), *res)) <--
        pimul3(address, dst, src, imm),
        !div_consumed(address),
        op_immediate(imm, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(dst, dst_str),
        ireg_hold_type(dst_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        op_register(src, src_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(res, preg_of_dst),
        ireg_of(preg_of_src, Ireg::from(src_str)),
        preg_of(src_arg, preg_of_src),
        let args = vec![*src_arg];

    // A 3-operand IMUL writes dst = load(src) * imm. Keep it unary after the
    // load: unlike 2-operand IMUL, the incoming value of dst is not an input.
    #[local] relation imul3_mem_raw(Address, Operation, MemoryChunk, Symbol, Mreg);
    imul3_mem_raw(*address, op, chunk, *src, Mreg::x86(dst_str)) <--
        pimul3(address, dst, src, imm),
        !div_consumed(address),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, _, _, _, _, _),
        op_immediate(imm, imm_value, _),
        let op = if *dst_is_64 { Operation::Omullimm(*imm_value) } else { Operation::Omulimm(*imm_value) },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Non-stack register-base and indexed sources use the generic load+op
    // path. `true` selects its write-only/unary form, so old dst is not kept.
    float_load_op(*addr, op.clone(), *chunk, addressing, Arc::new(args), *dst, true) <--
        imul3_mem_raw(addr, op, chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE" && !base_str.is_empty(),
        if *base_str != "RBP" && *base_str != "RSP",
        !reg_ip(base_str),
        let has_idx = *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode {
            base: Some(Ireg::from(base_str)),
            index: if has_idx { Some((Ireg::from(idx_str), *scale)) } else { None },
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    // RIP-relative IMUL sources use the same resolved-global unary path.
    float_load_op(*addr, op.clone(), *chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), *dst, true) <--
        imul3_mem_raw(addr, op, chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    // A proved BP/SP scalar source is lowered through stack_unary_load_op so
    // the canonical stack-slot SSA value is used instead of a raw frame load.
    stack_unary_load_op(*addr, op.clone(), Mreg::x86(base_str), *disp, *dst) <--
        imul3_mem_raw(addr, op, _chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP" || *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let base = Mreg::x86(*base_str),
        proven_frame_base_at(addr, base);

    // If BP/RSP is not proved to name this function's frame (or the operand is
    // indexed), retain an explicit pointer load plus unary multiply.  The old
    // shortcut emitted no RTL at all when no stack variable could be proved.
    float_load_op(*addr, op.clone(), *chunk, addressing.clone(), args.clone(), *dst, true) <--
        imul3_mem_raw(addr, op, chunk, src, dst),
        generic_bp_sp_address(addr, src, addressing, args);

    // IMUL with a SIMPLE memory source loads the value and multiplies via arith_load_op; the legacy rules treated the memory operand as its base register's value and dropped the load.
    arith_load_op(*address, op, chunk, Mreg::x86(base_str), *disp, Mreg::x86(dst_str)) <--
        pimul(address, dst, src),
        !div_consumed(address),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_sp(base_str),
        !reg_ip(base_str),
        let op = if *dst_is_64 { Operation::Omull } else { Operation::Omul },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // IMUL with SP-relative memory source (stack parameter / spilled local).
    arith_load_op(*address, op, chunk, Mreg::SP, *disp, Mreg::x86(dst_str)) <--
        pimul(address, dst, src),
        !div_consumed(address),
        op_register(dst, dst_str),
        reg_is_64(dst_str, dst_is_64),
        op_indirect(src, _, base_str, idx, _, disp, _),
        if *idx == "NONE" || idx.is_empty(),
        reg_sp(base_str),
        let op = if *dst_is_64 { Operation::Omull } else { Operation::Omul },
        let chunk = if *dst_is_64 { MemoryChunk::MInt64 } else { MemoryChunk::MInt32 };

    // Legacy fallback for an INDEXED memory source: keep the base-register approximation, since there is no arith_load_op indexed form yet.
    mach_inst(address, MachInst::Mop(Operation::Omul, Arc::new(args), *res)) <--
        pimul(address, r, mem),
        !div_consumed(address),
        op_register(r, r_str),
        ireg_hold_type(r_str.to_string(), typ),
        if *typ == Typ::Tint,
        op_indirect(mem, _, base_str, idx, _, disp, _),
        if *idx != "NONE" && !idx.is_empty(),
        if !base_str.is_empty() && *base_str != "NONE",
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        let args = vec![*res, *base_arg];

    mach_inst(address, MachInst::Mop(Operation::Omull, Arc::new(args), *res)) <--
        pimul(address, r, mem),
        !div_consumed(address),
        op_register(r, r_str),
        ireg_hold_type(r_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        op_indirect(mem, _, base_str, idx, _, disp, _),
        if *idx != "NONE" && !idx.is_empty(),
        if !base_str.is_empty() && *base_str != "NONE",
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        let args = vec![*res, *base_arg];

    mach_inst(address, MachInst::Mop(Operation::Omulhs, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omulhu, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("MUL") && !mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omullhs, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omullhu, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("MUL") && !mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 32-bit signed div: CDQ before IDIV -> Odiv (result in AX)
    mach_inst(address, MachInst::Mop(Operation::Odiv, Arc::new(args), Mreg::AX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        pcast(prevaddr, _, _),
        instruction(prevaddr, _, _, "CDQ", _, _, _, _, _, _),
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 32-bit unsigned div: xor EDX,EDX before DIV -> Odivu (result in AX)
    mach_inst(address, MachInst::Mop(Operation::Odivu, Arc::new(args), Mreg::AX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        pxor(prevaddr, rd, rs),
        op_register(rd, rd_str),
        op_register(rs, rs_str),
        if *rd_str == *rs_str,
        if *rd_str == "EDX" || *rd_str == "RDX",
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 32-bit signed mod: CDQ before IDIV -> Omod (result in DX)
    mach_inst(address, MachInst::Mop(Operation::Omod, Arc::new(args), Mreg::DX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        pcast(prevaddr, _, _),
        instruction(prevaddr, _, _, "CDQ", _, _, _, _, _, _),
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 32-bit unsigned mod: xor EDX,EDX before DIV -> Omodu (result in DX)
    mach_inst(address, MachInst::Mop(Operation::Omodu, Arc::new(args), Mreg::DX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        pxor(prevaddr, rd, rs),
        op_register(rd, rd_str),
        op_register(rs, rs_str),
        if *rd_str == *rs_str,
        if *rd_str == "EDX" || *rd_str == "RDX",
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // Fallback: IDIV without CDQ predecessor, treat as signed 32-bit
    mach_inst(address, MachInst::Mop(Operation::Odiv, Arc::new(args), Mreg::AX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pcast(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omod, Arc::new(args), Mreg::DX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pcast(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // Fallback: DIV without xor RDX predecessor, treat as unsigned 32-bit
    mach_inst(address, MachInst::Mop(Operation::Odivu, Arc::new(args), Mreg::AX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pxor(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omodu, Arc::new(args), Mreg::DX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pxor(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tint,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // Standalone CDQ/CQO sign-extension is a real definition of DX when no IDIV consumes it (the manual signed modulo-by-power-of-two idiom), so emit the arithmetic-shift sign replication.
    mach_inst(addr, MachInst::Mop(Operation::Oshrimm(31), Arc::new(vec![Mreg::AX]), Mreg::DX)) <--
        instruction(addr, _, _, "CDQ", _, _, _, _, _, _),
        prev_instr(next, addr),
        !pidiv(next, _, _);

    mach_inst(addr, MachInst::Mop(Operation::Oshrlimm(63), Arc::new(vec![Mreg::AX]), Mreg::DX)) <--
        instruction(addr, _, _, "CQO", _, _, _, _, _, _),
        prev_instr(next, addr),
        !pidiv(next, _, _);


    mach_inst(address, MachInst::Mop(Operation::Oand, Arc::new(args), *res)) <--
        pand(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oandimm(imm_int), Arc::new(args), *res)) <--
        pand(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oor, Arc::new(args), *res)) <--
        por(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oorimm(imm_int), Arc::new(args), *res)) <--
        por(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oxor, Arc::new(args), *res)) <--
        pxor(address, r, r2),
        if r!=r2,
        op_register(r, r_str),
        op_register(r2, r2_str),
        !reg_64(r_str),
        !reg_64(r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oxorimm(imm_int), Arc::new(args), *res)) <--
        pxor(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        !reg_64(r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Onot, Arc::new(args), *res)) <--
        pnot(address, r),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshl, Arc::new(args), *res)) <--
        psal(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshlimm(imm_int), Arc::new(args), *res)) <--
        psal(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshr, Arc::new(args), *res)) <--
        psar(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshrimm(imm_int), Arc::new(args), *res)) <--
        psar(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        // A 32-bit shift masks its count to 5 bits, so an immediate >= 32 can only be a 64-bit shift; suppress the 32-bit candidate so the mulhi product keeps full width instead of truncating to 0.
        if imm_int < 32,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshru, Arc::new(args), *res)) <--
        pshr(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshruimm(imm_int), Arc::new(args), *res)) <--
        pshr(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        // As with Oshrimm: a 32-bit logical shift masks its count to 5 bits, so imm >= 32 is a 64-bit shift and the 32-bit candidate is suppressed.
        if imm_int < 32,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Ororimm(imm_int), Arc::new(args), *res)) <--
        pror(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Ororimm(32 - imm_int), Arc::new(args), *res)) <--
        prol(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        !reg_64(r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Ororlimm(64 - imm_int), Arc::new(args), *res)) <--
        prol(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_64(r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshldimm(imm_int), Arc::new(args), *res)) <--
        psal(address, r, r2),
        op_indirect(_, r, _, r_str, _, n, _),
        let imm_int = *n as i64,
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    // A stack-slot address-of with a 64-bit dest is a 64-bit pointer -> Oleal; the previous 32-bit Olea truncated the high bits of the x86-64 stack address.
    mach_inst(address, MachInst::Mop(Operation::Oleal(addr), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        reg_sp(*r_str),
        if *index_str == "NONE" || index_str.is_empty(),
        rsp_frame_at(address, _),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        let addr = Addressing::Ainstack(*disp);

    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        !reg_sp(*r_str), !reg_bp(*r_str),
        if *index_str == "NONE" || index_str.is_empty(),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(arg, preg_of_r),
        let args = vec![*arg],
        let addr = Addressing::Aindexed(*disp);


    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        !reg_sp(*base_str), !reg_bp(*base_str), !reg_ip(*base_str),
        if *scale == 1,
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*base_arg, *index_arg],
        let addr = Addressing::Aindexed2(*disp);

    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        !reg_sp(*base_str), !reg_bp(*base_str), !reg_ip(*base_str),
        if *scale > 1,
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*base_arg, *index_arg],
        let addr = Addressing::Aindexed2scaled(*scale, *disp);

    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*index_arg],
        let addr = Addressing::Ascaled(*scale, *disp);

    // A scalar RBP LEA without a use-specific frame proof is ordinary pointer
    // arithmetic. The RTL stack shortcut is deliberately unavailable in this
    // case, so retain the base operand instead of dropping the instruction.
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, _, disp, _),
        if *base_str == "RBP",
        if *index_str == "NONE" || index_str.is_empty(),
        instr_in_function(address, func),
        !bp_frame_at(address, func),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        let args = vec![*base_arg],
        let addr = Addressing::Aindexed(*disp);

    // Once RSP no longer has a proved entry-frame coordinate, an RSP-relative
    // LEA is ordinary pointer arithmetic.  Keep the live SP operand instead of
    // fabricating a zero-argument Ainstack address.
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, "RSP", index_str, _, disp, _),
        if *index_str == "NONE" || index_str.is_empty(),
        unproven_frame_base_at(address, Mreg::SP),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        let args = vec![Mreg::SP],
        let addr = Addressing::Aindexed(*disp);

    // 64-bit LEA with a BP base and index when BP is not the frame pointer: without it the index is dropped and a frame-relative &local is fabricated for an argv-indexed pointer.
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        reg_bp(*base_str),
        instr_in_function(address, func),
        !bp_frame_at(address, func),
        op_register(r, res_str),
        reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*base_arg, *index_arg],
        let addr = if *scale > 1 { Addressing::Aindexed2scaled(*scale, *disp) } else { Addressing::Aindexed2(*disp) };

    mach_inst(address, MachInst::Mop(Operation::Oindirectsymbol(*ident as usize), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, _offset),
        op_register(r, res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res);

    mach_inst(address, MachInst::Mop(Operation::Olongconst(*target_addr as i64), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, _, _, disp, _),
        reg_ip(*base_str),
        rip_target_addr(address, target_addr),
        !resolved_addr_to_symbol(target_addr, _, _),
        op_register(r, res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res);

    // Absolute addressing LEA: lea 0x40402c, %rax (base="NONE", no index, disp is the address)
    mach_inst(address, MachInst::Mop(Operation::Oindirectsymbol(*ident as usize), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        resolved_addr_to_symbol(target_addr, ident, _offset),
        op_register(r, res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res);

    mach_inst(address, MachInst::Mop(Operation::Olongconst(*target_addr as i64), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(address, target_addr),
        !resolved_addr_to_symbol(target_addr, _, _),
        op_register(r, res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res);


    mach_inst(address, MachInst::Mop(cast_op, Arc::new(args.clone()), *res)) <--
        pmov(address, r, r1),
        !div_consumed(address),
        op_register(r1, r1_str),
        op_register(r, r_str),
        reg_is_64(r1_str, r1_is_64),
        reg_is_64(r_str, r_is_64),
        if r1_is_64 != r_is_64,
        ireg_of(preg_of_r1, Ireg::from(r1_str)),
        preg_of(a1, preg_of_r1),
        ireg_of(preg_of_res, Ireg::from(r_str)),
        preg_of(res, preg_of_res),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        let mnem_upper = mnem.to_ascii_uppercase(),
        let args = vec![*a1],
        let cast_op = if mnem_upper.starts_with("MOVSXD") || mnem_upper.starts_with("CDQE") {
            Operation::Ocast32signed
        } else if !*r1_is_64 && *r_is_64 {
            Operation::Ocast32unsigned
        } else {
            Operation::Olowlong
        };

    mach_inst(address, MachInst::Mop(Operation::Onegl, Arc::new(args), *res)) <--
        pneg(address, r),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];


    mach_inst(address, MachInst::Mop(op, Arc::new(args), *res)) <--
        psub(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_is_64(r_str, r_is_64),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res],
        let op = if *r_is_64 { Operation::Oaddlimm(-imm_int) } else { Operation::Oaddimm(-imm_int) };

    mach_inst(address, MachInst::Mop(Operation::Omull, Arc::new(args), *res)) <--
        pimul(address, r, r2),
        !div_consumed(address),
        op_register(r, r_str),
        ireg_hold_type(r_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omullimm(imm_int), Arc::new(args), *res)) <--
        pimul(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Omullhs, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX ,*a2];

    mach_inst(address, MachInst::Mop(Operation::Omullhu, Arc::new(args), Mreg::DX)) <--
        pmul(address, r2),
        instruction(address, _, _, mnem, _, _, _, _, _, _),
        if mnem.contains("MUL") && !mnem.contains("IMUL"),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX ,*a2];

    // 64-bit signed div: CQO before IDIV -> Odivl (result in AX)
    mach_inst(address, MachInst::Mop(Operation::Odivl, Arc::new(args), Mreg::AX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        pcast(prevaddr, _, _),
        instruction(prevaddr, _, _, "CQO", _, _, _, _, _, _),
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 64-bit unsigned div: xor RDX,RDX before DIV -> Odivlu (result in AX)
    mach_inst(address, MachInst::Mop(Operation::Odivlu, Arc::new(args), Mreg::AX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        pxor(prevaddr, rd, rs),
        op_register(rd, rd_str),
        op_register(rs, rs_str),
        if *rd_str == *rs_str,
        if *rd_str == "EDX" || *rd_str == "RDX",
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 64-bit signed mod: CQO before IDIV -> Omodl (result in DX)
    mach_inst(address, MachInst::Mop(Operation::Omodl, Arc::new(args), Mreg::DX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        pcast(prevaddr, _, _),
        instruction(prevaddr, _, _, "CQO", _, _, _, _, _, _),
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // 64-bit unsigned mod: xor RDX,RDX before DIV -> Omodlu (result in DX)
    mach_inst(address, MachInst::Mop(Operation::Omodlu, Arc::new(args), Mreg::DX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        pxor(prevaddr, rd, rs),
        op_register(rd, rd_str),
        op_register(rs, rs_str),
        if *rd_str == *rs_str,
        if *rd_str == "EDX" || *rd_str == "RDX",
        op_register(r2, r2_str),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // Fallback: IDIV without CQO predecessor, treat as signed 64-bit
    mach_inst(address, MachInst::Mop(Operation::Odivl, Arc::new(args), Mreg::AX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pcast(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omodl, Arc::new(args), Mreg::DX)) <--
        pidiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pcast(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    // Fallback: DIV without xor RDX predecessor, treat as unsigned 64-bit
    mach_inst(address, MachInst::Mop(Operation::Odivlu, Arc::new(args), Mreg::AX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pxor(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Omodlu, Arc::new(args), Mreg::DX)) <--
        pudiv(address, r2, _),
        prev_instr(address, prevaddr),
        !pxor(prevaddr, _, _),
        op_register(r2, r2_str),
        ireg_hold_type(r2_str.to_string(), typ),
        if *typ == Typ::Tlong || *typ == Typ::Tany64,
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![Mreg::AX, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oandl, Arc::new(args), *res)) <--
        pand(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oandlimm(imm_int), Arc::new(args), *res)) <--
        pand(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oorl, Arc::new(args), *res)) <--
        por(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oorlimm(imm_int), Arc::new(args), *res)) <--
        por(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oxorl, Arc::new(args), *res)) <--
        pxor(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_64(r_str),
        reg_64(r2_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        ireg_of(preg_of_r2, Ireg::from(r2_str)),
        preg_of(a2, preg_of_r2),
        let args = vec![*res, *a2];

    mach_inst(address, MachInst::Mop(Operation::Oxorlimm(imm_int), Arc::new(args), *res)) <--
        pxor(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        reg_64(r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    // Bit-test-and-modify immediate forms lower to masked bitwise ops, with the bit index masked to the operand width and the mask built unsigned so bit 63 does not overflow the shift.
    mach_inst(address, MachInst::Mop(Operation::Oxorlimm(mask), Arc::new(args), *res)) <--
        pbtc(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        reg_64(r_str),
        let mask = (1u64 << ((*imm_str as u32) & 63)) as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oxorimm(mask), Arc::new(args), *res)) <--
        pbtc(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        !reg_64(r_str),
        let mask = (1u32 << ((*imm_str as u32) & 31)) as i32 as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oorlimm(mask), Arc::new(args), *res)) <--
        pbts(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        reg_64(r_str),
        let mask = (1u64 << ((*imm_str as u32) & 63)) as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oorimm(mask), Arc::new(args), *res)) <--
        pbts(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        !reg_64(r_str),
        let mask = (1u32 << ((*imm_str as u32) & 31)) as i32 as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oandlimm(mask), Arc::new(args), *res)) <--
        pbtr(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        reg_64(r_str),
        let mask = (!(1u64 << ((*imm_str as u32) & 63))) as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oandimm(mask), Arc::new(args), *res)) <--
        pbtr(address, r, n),
        op_immediate(n, imm_str, _),
        op_register(r, r_str),
        !reg_64(r_str),
        let mask = (!(1u32 << ((*imm_str as u32) & 31))) as i32 as i64,
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Onotl, Arc::new(args), *res)) <--
        pnot(address, r),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshll, Arc::new(args), *res)) <--
        psal(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshllimm(imm_int), Arc::new(args), *res)) <--
        psal(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshrl, Arc::new(args), *res)) <--
        psar(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshrlimm(imm_int), Arc::new(args), *res)) <--
        psar(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Oshrlu, Arc::new(args), *res)) <--
        pshr(address, r, cxreg),
        op_register(cxreg, cxreg_str),
        reg_cx(cxreg_str),
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res, Mreg::CX];

    mach_inst(address, MachInst::Mop(Operation::Oshrluimm(imm_int), Arc::new(args), *res)) <--
        pshr(address, r, n),
        !div_consumed(address),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    mach_inst(address, MachInst::Mop(Operation::Ororlimm(imm_int), Arc::new(args), *res)) <--
        pror(address, r, n),
        op_immediate(n, imm_str, _),
        let imm_int = *imm_str as i64,
        op_register(r, r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(res, preg_of_r),
        let args = vec![*res];

    // Mirror of the 64-bit rule: a stack-slot address-of with a 32-bit dest is Olea, since the previous Oleal over-promoted a genuine 32-bit leal result to long.
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        reg_sp(*r_str),
        if *index_str == "NONE" || index_str.is_empty(),
        rsp_frame_at(address, _),
        let addr = Addressing::Ainstack(*disp);

    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(vec![Mreg::SP]), *res)) <--
        plea(address, r, am),
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        op_indirect(am, _, "RSP", index_str, _, disp, _),
        if *index_str == "NONE" || index_str.is_empty(),
        unproven_frame_base_at(address, Mreg::SP),
        let addr = Addressing::Aindexed(*disp);

    // A 32-bit LEA used as integer arithmetic is Olea (Xint), not Oleal: mis-promoting to long breaks a downstream unsigned 32-bit comparison, which then renders as a signed long compare.
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        if *index_str == "NONE" || index_str.is_empty(),
        !reg_sp(*r_str), !reg_bp(*r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(arg, preg_of_r),
        let args = vec![*arg],
        let addr = Addressing::Aindexed(*disp);

    mach_inst(address, MachInst::Mop(Operation::Oleal(addr), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        padd(nextaddress, r_same, delta),
        next(address, nextaddress),
        op_register(r, res_str),
        op_register(r_same, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        reg_sp(*r_str),
        if *index_str == "NONE" || index_str.is_empty(),
        rsp_frame_at(address, _),
        let addr = Addressing::Ainstack(*disp);

    mach_inst(address, MachInst::Mop(Operation::Oleal(addr), Arc::new(vec![]), *res)) <--
        plea(address, r, am),
        padd(nextaddress, r_same, delta),
        next(address, nextaddress),
        op_register(r, res_str),
        op_register(r_same, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        op_indirect(am, _, r_str, index_str, scale, disp, _),
        if *index_str == "NONE" || index_str.is_empty(),
        !reg_sp(*r_str), !reg_bp(*r_str),
        ireg_of(preg_of_r, Ireg::from(r_str)),
        preg_of(arg, preg_of_r),
        let args = vec![*arg],
        let addr = Addressing::Aindexed(*disp);

    // 32-bit LEA with base + index (scale=1) -> Olea(Aindexed2)
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        !reg_sp(*base_str), !reg_bp(*base_str), !reg_ip(*base_str),
        if *scale == 1,
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*base_arg, *index_arg],
        let addr = Addressing::Aindexed2(*disp);

    // 32-bit LEA with base + index (scale>1) -> Olea(Aindexed2scaled)
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        !reg_sp(*base_str), !reg_bp(*base_str), !reg_ip(*base_str),
        if *scale > 1,
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_base, Ireg::from(base_str)),
        preg_of(base_arg, preg_of_base),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*base_arg, *index_arg],
        let addr = Addressing::Aindexed2scaled(*scale, *disp);

    // 32-bit LEA with no base, just index -> Olea(Ascaled)
    mach_inst(address, MachInst::Mop(Operation::Olea(addr), Arc::new(args), *res)) <--
        plea(address, r, am),
        op_indirect(am, _, base_str, index_str, scale, disp, _),
        if *index_str != "NONE" && !index_str.is_empty(),
        if *base_str == "NONE" || base_str.is_empty(),
        op_register(r, res_str),
        !reg_64(res_str),
        ireg_of(preg_of_res, Ireg::from(res_str)),
        preg_of(res, preg_of_res),
        ireg_of(preg_of_index, Ireg::from(index_str)),
        preg_of(index_arg, preg_of_index),
        let args = vec![*index_arg],
        let addr = Addressing::Ascaled(*scale, *disp);

    mach_inst(address, MachInst::Mop(Operation::Ointconst(0), Arc::new(vec![]), res)) <--
        ppxor(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        if *r_str == *r2_str,
        reg_xmm(r_str),
        let res = Mreg::x86(r_str);

    mach_inst(address, MachInst::Mop(Operation::Ointconst(0), Arc::new(vec![]), res)) <--
        pxorpd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        if *r_str == *r2_str,
        reg_xmm(r_str),
        let res = Mreg::x86(r_str);

    mach_inst(address, MachInst::Mop(Operation::Ointconst(0), Arc::new(vec![]), res)) <--
        pxorps(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        if *r_str == *r2_str,
        reg_xmm(r_str),
        let res = Mreg::x86(r_str);

    mach_inst(address, MachInst::Mop(Operation::Omove, Arc::new(args), res)) <--
        pmovsd(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_xmm(arg_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(addr, MachInst::Mload(MemoryChunk::MFloat64, addressing, Arc::new(args), dst)) <--
        pmovsd(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        reg_xmm(dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, r2, _, _scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    mach_inst(addr, MachInst::Mstore(MemoryChunk::MFloat64, addressing, Arc::new(args), src)) <--
        pmovsd(addr, dst, src_sym),
        op_register(src_sym, src_str),
        reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, _, _scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    mach_inst(address, MachInst::Mop(Operation::Ofloatoflong, Arc::new(args), res)) <--
        pcvtsi2sd(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_64(arg_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Ofloatofint, Arc::new(args), res)) <--
        pcvtsi2sd(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_is_64(arg_str, false),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Olongoffloat, Arc::new(args), res)) <--
        pcvtsd2si(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(arg_str),
        reg_64(res_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Ointoffloat, Arc::new(args), res)) <--
        pcvtsd2si(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(arg_str),
        reg_is_64(res_str, false),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Oaddf, Arc::new(args), res)) <--
        paddsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Osubf, Arc::new(args), res)) <--
        psubsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Omulf, Arc::new(args), res)) <--
        pmulsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Odivf, Arc::new(args), res)) <--
        pdivsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];


    mach_inst(address, MachInst::Mop(Operation::Oaddfs, Arc::new(args), res)) <--
        paddss(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Osubfs, Arc::new(args), res)) <--
        psubss(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Omulfs, Arc::new(args), res)) <--
        pmulss(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Odivfs, Arc::new(args), res)) <--
        pdivss(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];


    mach_inst(address, MachInst::Mop(Operation::Oaddf, Arc::new(args), res)) <--
        instruction(address, _, _, "VADDSD", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Osubf, Arc::new(args), res)) <--
        instruction(address, _, _, "VSUBSD", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Omulf, Arc::new(args), res)) <--
        instruction(address, _, _, "VMULSD", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Odivf, Arc::new(args), res)) <--
        instruction(address, _, _, "VDIVSD", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Oaddfs, Arc::new(args), res)) <--
        instruction(address, _, _, "VADDSS", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Osubfs, Arc::new(args), res)) <--
        instruction(address, _, _, "VSUBSS", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Omulfs, Arc::new(args), res)) <--
        instruction(address, _, _, "VMULSS", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];

    mach_inst(address, MachInst::Mop(Operation::Odivfs, Arc::new(args), res)) <--
        instruction(address, _, _, "VDIVSS", src1, dst, src2, _, _, _),
        op_register(dst, dst_str),
        op_register(src1, src1_str),
        op_register(src2, src2_str),
        reg_xmm(dst_str),
        let res = Mreg::x86(dst_str),
        let a1 = Mreg::x86(src1_str),
        let a2 = Mreg::x86(src2_str),
        let args = vec![a1, a2];


    mach_inst(address, MachInst::Mop(Operation::Osingleoffloat, Arc::new(args), res)) <--
        pcvtsd2ss(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Ofloatofsingle, Arc::new(args), res)) <--
        pcvtss2sd(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];


    // A float/SSE op with a MEMORY source matches no reg-reg rule and would be dropped entirely, so capture it as a fused load+op lowered in rtl_pass; BP/SP and RIP bases are excluded.
    relation float_mem_op_raw(Address, Operation, MemoryChunk, Symbol, Mreg, bool);

    float_mem_op_raw(addr, Operation::Oaddfs, MemoryChunk::MFloat32, *src, Mreg::x86(d), false) <--
        paddss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Osubfs, MemoryChunk::MFloat32, *src, Mreg::x86(d), false) <--
        psubss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Omulfs, MemoryChunk::MFloat32, *src, Mreg::x86(d), false) <--
        pmulss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Odivfs, MemoryChunk::MFloat32, *src, Mreg::x86(d), false) <--
        pdivss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Oaddf, MemoryChunk::MFloat64, *src, Mreg::x86(d), false) <--
        paddsd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Osubf, MemoryChunk::MFloat64, *src, Mreg::x86(d), false) <--
        psubsd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Omulf, MemoryChunk::MFloat64, *src, Mreg::x86(d), false) <--
        pmulsd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Odivf, MemoryChunk::MFloat64, *src, Mreg::x86(d), false) <--
        pdivsd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Ofloatofsingle, MemoryChunk::MFloat32, *src, Mreg::x86(d), true) <--
        pcvtss2sd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);
    float_mem_op_raw(addr, Operation::Osingleoffloat, MemoryChunk::MFloat64, *src, Mreg::x86(d), true) <--
        pcvtsd2ss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _);

    relation float_load_op(Address, Operation, MemoryChunk, Addressing, Arc<Vec<Mreg>>, Mreg, bool);

    float_load_op(*addr, op.clone(), *chunk, addressing, Arc::new(args), *dst, *is_unary) <--
        float_mem_op_raw(addr, op, chunk, src, dst, is_unary),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE" && !base_str.is_empty(),
        if *base_str != "RBP" && *base_str != "RSP",
        !reg_ip(base_str),
        let has_idx = *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode {
            base: Some(Ireg::from(base_str)),
            index: if has_idx { Some((Ireg::from(idx_str), *scale)) } else { None },
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    // A scalar BP/SP source without a use-specific frame proof remains an
    // ordinary pointer load. Indexed BP/SP operands are likewise pointer/array
    // accesses rather than scalar stack slots.
    float_load_op(*addr, op.clone(), *chunk, addressing.clone(), args.clone(), *dst, *is_unary) <--
        float_mem_op_raw(addr, op, chunk, src, dst, is_unary),
        generic_bp_sp_address(addr, src, addressing, args);

    // RIP-relative float source (3.8a): dominant PIE shape for float constants; previously excluded so the op vanished entirely (function collapsed to `return 0`); resolved to synthesized global symbol via Aglobal.
    float_load_op(*addr, op.clone(), *chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), *dst, *is_unary) <--
        float_mem_op_raw(addr, op, chunk, src, dst, is_unary),
        op_indirect(src, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    // Absolute-addressed source (clang -fno-pie constant pool): base="NONE", disp is the address.
    float_load_op(*addr, op.clone(), *chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), *dst, *is_unary) <--
        float_mem_op_raw(addr, op, chunk, src, dst, is_unary),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);


    // cvtsi2ss/sd with a MEMORY source matches no register-source rule and would be dropped, so capture it as a unary conversion whose integer source width comes from the memory operand size.
    relation cvtsi2_mem_raw(Address, Operation, MemoryChunk, Symbol, Mreg);

    cvtsi2_mem_raw(addr, op, chunk, *src, Mreg::x86(d)) <--
        pcvtsi2sd(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _),
        op_indirect(src, _, _, _, _, _, sz),
        let (op, chunk) = if *sz == 8 { (Operation::Ofloatoflong, MemoryChunk::MInt64) }
                          else { (Operation::Ofloatofint, MemoryChunk::MInt32) };
    cvtsi2_mem_raw(addr, op, chunk, *src, Mreg::x86(d)) <--
        pcvtsi2ss(addr, dsym, src), op_register(dsym, d), reg_xmm(d), !op_register(src, _),
        op_indirect(src, _, _, _, _, _, sz),
        let (op, chunk) = if *sz == 8 { (Operation::Osingleoflong, MemoryChunk::MInt64) }
                          else { (Operation::Osingleofint, MemoryChunk::MInt32) };

    // Register-base and indexed source reuse float_load_op's chunk-agnostic unary load+op lowering; BP/SP and RIP are excluded here exactly as in the float_mem_op_raw derivation.
    float_load_op(*addr, op.clone(), *chunk, addressing, Arc::new(args), *dst, true) <--
        cvtsi2_mem_raw(addr, op, chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE" && !base_str.is_empty(),
        if *base_str != "RBP" && *base_str != "RSP",
        !reg_ip(base_str),
        let has_idx = *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode {
            base: Some(Ireg::from(base_str)),
            index: if has_idx { Some((Ireg::from(idx_str), *scale)) } else { None },
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    float_load_op(*addr, op.clone(), *chunk, addressing.clone(), args.clone(), *dst, true) <--
        cvtsi2_mem_raw(addr, op, chunk, src, dst),
        generic_bp_sp_address(addr, src, addressing, args);

    // RIP-relative integer source (a PIE int constant in .rodata) resolves to a global symbol like the float RIP path; it was previously excluded and the conversion vanished.
    float_load_op(*addr, op.clone(), *chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), *dst, true) <--
        cvtsi2_mem_raw(addr, op, chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, _, _, _),
        reg_ip(*base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    // A unary op reading a BP/SP-relative SCALAR stack slot must use the
    // slot-variable model, not a raw Iload that would re-materialize it as
    // *(&local + k); an indexed access is a stack array.
    relation stack_unary_load_op(Address, Operation, Mreg, i64, Mreg);
    stack_unary_load_op(*addr, op.clone(), Mreg::x86(base_str), *disp, *dst) <--
        cvtsi2_mem_raw(addr, op, _chunk, src, dst),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP" || *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let base = Mreg::x86(*base_str),
        proven_frame_base_at(addr, base);

    // Unary scalar SSE conversions use the same proved slot path as CVTSI.
    stack_unary_load_op(*addr, op.clone(), Mreg::x86(base_str), *disp, *dst) <--
        float_mem_op_raw(addr, op, _chunk, src, dst, true),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP" || *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let base = Mreg::x86(*base_str),
        proven_frame_base_at(addr, base);

    // BINARY float arith reading a spilled float SCALAR slot, which float_load_op excludes: read it through the slot's canonical SSA reg as a binary in-RMW op; scalar slots only.
    relation float_arith_stack_op(Address, Operation, Mreg, i64, Mreg);
    float_arith_stack_op(*addr, op.clone(), Mreg::x86(base_str), *disp, *dst) <--
        float_mem_op_raw(addr, op, _chunk, src, dst, false),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP" || *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        let base = Mreg::x86(*base_str),
        proven_frame_base_at(addr, base);

    mach_inst(address, MachInst::Mop(Operation::Osingleofint, Arc::new(args), res)) <--
        pcvtsi2ss(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_is_64(arg_str, false),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Osingleoflong, Arc::new(args), res)) <--
        pcvtsi2ss(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_64(arg_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Ointofsingle, Arc::new(args), res)) <--
        pcvtss2si(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(arg_str),
        reg_is_64(res_str, false),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(address, MachInst::Mop(Operation::Olongofsingle, Arc::new(args), res)) <--
        pcvtss2si(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(arg_str),
        reg_64(res_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];


    mach_inst(address, MachInst::Mop(Operation::Onegf, Arc::new(args), res)) <--
        pxorpd(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_indirect(r2, _, _, _, _, _, _),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    mach_inst(address, MachInst::Mop(Operation::Oabsf, Arc::new(args), res)) <--
        pandpd(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_indirect(r2, _, _, _, _, _, _),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    mach_inst(address, MachInst::Mop(Operation::Onegfs, Arc::new(args), res)) <--
        pxorps(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_indirect(r2, _, _, _, _, _, _),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    mach_inst(address, MachInst::Mop(Operation::Oabsfs, Arc::new(args), res)) <--
        pandps(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_indirect(r2, _, _, _, _, _, _),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    // Reg-reg fabs/fabsf: recovered from positive evidence that the loaded constant is the IEEE-754 sign-clear mask plus a structural CFG reaching-def proving the mask register still holds it.

    // A movss/movsd loading a 4-/8-byte sign-clear mask from RIP-relative .rodata into an XMM register; the width distinction keeps a movss from validating a double mask.
    relation mask_load(Address, Mreg, usize);
    mask_load(load_addr, dst_reg, 4) <--
        instruction(load_addr, _, _, mnem, _, dst, _, _, _, _),
        if *mnem == "MOVSS" || *mnem == "VMOVSS",
        op_register(dst, dst_str),
        reg_xmm(dst_str),
        rip_target_addr(load_addr, target),
        fp_sign_clear_mask(target, 4),
        let dst_reg = Mreg::x86(dst_str);
    mask_load(load_addr, dst_reg, 8) <--
        instruction(load_addr, _, _, mnem, _, dst, _, _, _, _),
        if *mnem == "MOVSD" || *mnem == "VMOVSD",
        op_register(dst, dst_str),
        reg_xmm(dst_str),
        rip_target_addr(load_addr, target),
        fp_sign_clear_mask(target, 8),
        let dst_reg = Mreg::x86(dst_str);

    // Reaching-def lattice for the mask register as monotone CFG propagation: defined at the load, flowing over next edges, killed by any reg_def; structural, never an address comparison.
    relation mask_reg_in(Address, Mreg, usize);
    relation mask_reg_out(Address, Mreg, usize);
    mask_reg_out(load_addr, reg, w) <-- mask_load(load_addr, reg, w);
    mask_reg_in(succ, reg, w) <--
        mask_reg_out(addr, reg, w),
        next(addr, succ);
    mask_reg_out(addr, reg, w) <--
        mask_reg_in(addr, reg, w),
        !reg_def(addr, reg);

    // andps/andpd where the mask register provably holds the sign-clear mask IS fabsf/fabs; r is the accumulator and r2 the mask register, matched by width.
    mach_inst(address, MachInst::Mop(Operation::Oabsfs, Arc::new(args), res)) <--
        pandps(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_register(r2, r2_str),
        reg_xmm(r2_str),
        let mask_reg = Mreg::x86(r2_str),
        mask_reg_in(address, mask_reg, 4),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    mach_inst(address, MachInst::Mop(Operation::Oabsf, Arc::new(args), res)) <--
        pandpd(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_register(r2, r2_str),
        reg_xmm(r2_str),
        let mask_reg = Mreg::x86(r2_str),
        mask_reg_in(address, mask_reg, 8),
        let res = Mreg::x86(r_str),
        let args = vec![res];

    // ANDPS/ANDPD is commutative, so gcc may put the mask in the DESTINATION register; without this symmetric form the andps is dropped and the value subtree is DCE'd.
    mach_inst(address, MachInst::Mop(Operation::Oabsfs, Arc::new(args), res)) <--
        pandps(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_register(r2, r2_str),
        reg_xmm(r2_str),
        let mask_reg = Mreg::x86(r_str),
        mask_reg_in(address, mask_reg, 4),
        let res = Mreg::x86(r_str),
        let value = Mreg::x86(r2_str),
        let args = vec![value];

    mach_inst(address, MachInst::Mop(Operation::Oabsf, Arc::new(args), res)) <--
        pandpd(address, r, r2),
        op_register(r, r_str),
        reg_xmm(r_str),
        op_register(r2, r2_str),
        reg_xmm(r2_str),
        let mask_reg = Mreg::x86(r_str),
        mask_reg_in(address, mask_reg, 8),
        let res = Mreg::x86(r_str),
        let value = Mreg::x86(r2_str),
        let args = vec![value];


    mach_inst(address, MachInst::Mop(Operation::Omaxf, Arc::new(args), res)) <--
        pmaxsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];

    mach_inst(address, MachInst::Mop(Operation::Ominf, Arc::new(args), res)) <--
        pminsd(address, r, r2),
        op_register(r, r_str),
        op_register(r2, r2_str),
        reg_xmm(r_str),
        let res = Mreg::x86(r_str),
        let a2 = Mreg::x86(r2_str),
        let args = vec![res, a2];


    mach_inst(addr0, MachInst::Mcond(Condition::Ccompf(Comparison::Cgt), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondA, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompf(Comparison::Cge), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondAe, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompf(Comparison::Clt), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondB, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompf(Comparison::Cle), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondBe, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompf(Comparison::Ceq), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondE, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Cnotcompf(Comparison::Ceq), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondNe, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompfs(Comparison::Cgt), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondA, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompfs(Comparison::Cge), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondAe, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompfs(Comparison::Clt), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondB, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompfs(Comparison::Cle), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondBe, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Ccompfs(Comparison::Ceq), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondE, lbl);

    mach_inst(addr0, MachInst::Mcond(Condition::Cnotcompfs(Comparison::Ceq), Arc::new(vec![arg1, arg2]), lbl)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, TestCond::CondNe, lbl);

    // comiss with a RIP-relative operand loads the constant into FP0 (x87 ST0, safe scratch); fp0_loaded_at records it so a synthetic reg_def connects the load to the consumer's reg_use.
    #[local] relation fp0_loaded_at(Address);

    reg_def(addr0, Mreg::FP0) <-- fp0_loaded_at(addr0);

    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat32, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        op_indirect(r2, _, base_str, _, _, _, _),
        reg_ip(*base_str),
        rip_target_addr(addr0, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat64, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        op_indirect(r2, _, base_str, _, _, _, _),
        reg_ip(*base_str),
        rip_target_addr(addr0, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    // comiss/ucomiss with absolute addressing memory operand
    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat32, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr0, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat64, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "NONE" || base_str.is_empty(),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp > 0,
        abs_target_addr(addr0, target_addr),
        resolved_addr_to_symbol(target_addr, ident, offset);

    // Register-base memory operand for comisd/ucomiss: loads into FP0, mirroring RIP/absolute paths; BP/SP excluded per slot-variable model.
    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat32, addressing, Arc::new(args), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        if *base_str != "RBP" && *base_str != "RSP",
        let has_idx = *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode {
            base: Some(Ireg::from(base_str)),
            index: if has_idx { Some((Ireg::from(idx_str), *scale)) } else { None },
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr0, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat64, addressing, Arc::new(args), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE" && !base_str.is_empty(),
        !reg_ip(base_str),
        if *base_str != "RBP" && *base_str != "RSP",
        let has_idx = *idx_str != "NONE" && !idx_str.is_empty(),
        let addrmode = Addrmode {
            base: Some(Ireg::from(base_str)),
            index: if has_idx { Some((Ireg::from(idx_str), *scale)) } else { None },
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr0, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    // An unproved RBP is an ordinary SSA pointer, not a stack slot. Preserve
    // the compare's memory read through the same generic BP addressing helper
    // used by the other fused load operations.
    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat32, addressing.clone(), args.clone(), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        generic_bp_sp_address(addr0, r2, addressing, args);

    mach_inst(addr0, MachInst::Mload(MemoryChunk::MFloat64, addressing.clone(), args.clone(), Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        generic_bp_sp_address(addr0, r2, addressing, args);

    // BP-relative float compare operand: read the spilled slot into FP0 via Mgetstack, the same slot-variable path every other BP/SP float access uses; scalar slots only.
    mach_inst(addr0, MachInst::Mgetstack(*disp, Typ::Tsingle, Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(addr0, func),
        bp_frame_at(addr0, func);

    mach_inst(addr0, MachInst::Mgetstack(*disp, Typ::Tfloat, Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "RBP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(addr0, func),
        bp_frame_at(addr0, func);

    // SP-relative variant of the BP slot-load, for -O2/-O3 frame-pointer-less
    // float spills, guarded by the use-specific CFG-safe RSP proof.
    mach_inst(addr0, MachInst::Mgetstack(*disp, Typ::Tsingle, Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomiss(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        rsp_frame_at(addr0, _);

    mach_inst(addr0, MachInst::Mgetstack(*disp, Typ::Tfloat, Mreg::FP0)),
    fp0_loaded_at(*addr0) <--
        pucomisd(addr0, _r1, r2),
        op_indirect(r2, _, base_str, idx_str, _, disp, _),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        rsp_frame_at(addr0, _);

    // pucomiss/pucomisd with a memory operand: (compare addr, is_double, register operand).
    #[local] relation fcmp_mem_arg(Address, bool, Mreg);
    fcmp_mem_arg(addr0, false, Mreg::x86(r1_str)) <--
        pucomiss(addr0, r1, r2), op_register(r1, r1_str), !op_register(r2, _), reg_xmm(r1_str);
    fcmp_mem_arg(addr0, true, Mreg::x86(r1_str)) <--
        pucomisd(addr0, r1, r2), op_register(r1, r1_str), !op_register(r2, _), reg_xmm(r1_str);

    // Mem-operand float compare + jcc relocated to the jcc's address (CF-5c): the FP0 Mload occupies the compare's address and the candidate pick would drop an Mcond keyed there.
    mach_inst(addr1, MachInst::Mcond(condition.clone(), Arc::new(vec![*arg1, Mreg::FP0]), lbl)),
    mcond_at_jcc(*addr1),
    reg_use(*addr1, *arg1),
    reg_use(*addr1, Mreg::FP0) <--
        fcmp_mem_arg(addr0, is_double, arg1),
        fp0_loaded_at(*addr0),
        fcmp_jcc_link(addr0, addr1),
        pjcc(addr1, test_cond, lbl),
        if let Some(condition) = fcmp_setcc_condition(*test_cond, *is_double);

    // Two-branch JP+JNE float != with a MEMORY operand: mirror the register rule but read the loaded FP0, keyed at the JNE, gated on fp0_loaded_at so FP0 is defined before the read.
    mach_inst(addr2, MachInst::Mcond(Condition::Cnotcompfs(Comparison::Ceq), Arc::new(vec![*arg1, Mreg::FP0]), lbl)),
    mcond_at_jcc(*addr2),
    reg_use(*addr2, *arg1),
    reg_use(*addr2, Mreg::FP0) <--
        fcmp_mem_arg(addr0, false, arg1),
        fp0_loaded_at(*addr0),
        next(addr0, addr1),
        pjcc(addr1, TestCond::CondP, _lbl_p),
        direct_jump(addr1, tgt),
        next(addr1, addr2),
        pjcc(addr2, TestCond::CondNe, lbl),
        direct_jump(addr2, tgt);

    mach_inst(addr2, MachInst::Mcond(Condition::Cnotcompf(Comparison::Ceq), Arc::new(vec![*arg1, Mreg::FP0]), lbl)),
    mcond_at_jcc(*addr2),
    reg_use(*addr2, *arg1),
    reg_use(*addr2, Mreg::FP0) <--
        fcmp_mem_arg(addr0, true, arg1),
        fp0_loaded_at(*addr0),
        next(addr0, addr1),
        pjcc(addr1, TestCond::CondP, _lbl_p),
        direct_jump(addr1, tgt),
        next(addr1, addr2),
        pjcc(addr2, TestCond::CondNe, lbl),
        direct_jump(addr2, tgt);

    // Float compare feeding SETcc (3.8b): had no fcmp->setcc link so neither instruction lifted (function collapsed to `return 0`); mirrors cmp_setcc_link to produce Mop(Ocmp(Ccompf*/Cnotcompf*)) writing the setcc destination.
    #[local] relation fcmp_setcc_link(Address, Address);
    fcmp_setcc_link(addr0, addr1) <--
        pucomisd(addr0, _, _),
        next(addr0, addr1),
        psetcc(addr1, _, _);
    fcmp_setcc_link(addr0, addr1) <--
        pucomiss(addr0, _, _),
        next(addr0, addr1),
        psetcc(addr1, _, _);

    // Reg-reg Ocmp keyed at compare's address: int cmp_setcc design parity; setcc node bridges via linear's branch-to-next default.
    mach_inst(addr0, MachInst::Mop(Operation::Ocmp(condition.clone()), Arc::new(vec![arg1, arg2]), *dst_mreg)) <--
        pucomisd(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        if let Some(condition) = fcmp_setcc_condition(*test_cond, true);

    mach_inst(addr0, MachInst::Mop(Operation::Ocmp(condition.clone()), Arc::new(vec![arg1, arg2]), *dst_mreg)) <--
        pucomiss(addr0, r1, r2),
        op_register(r1, r1_str),
        op_register(r2, r2_str),
        reg_xmm(r1_str),
        let arg1 = Mreg::x86(r1_str),
        let arg2 = Mreg::x86(r2_str),
        fcmp_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        if let Some(condition) = fcmp_setcc_condition(*test_cond, false);

    // Mem-operand Ocmp keyed at SETcc's address: FP0 Mload occupies compare's address causing the same single-instruction-per-node collision as the relocated Mcond.
    mach_inst(addr_set, MachInst::Mop(Operation::Ocmp(condition.clone()), Arc::new(vec![*arg1, Mreg::FP0]), *dst_mreg)),
    reg_use(*addr_set, *arg1),
    reg_use(*addr_set, Mreg::FP0) <--
        fcmp_mem_arg(addr0, is_double, arg1),
        fp0_loaded_at(*addr0),
        fcmp_setcc_link(addr0, addr_set),
        psetcc(addr_set, test_cond, dst_sym),
        op_register(dst_sym, dst_str),
        ireg_of(preg_of_dst, Ireg::from(dst_str)),
        preg_of(dst_mreg, preg_of_dst),
        if let Some(condition) = fcmp_setcc_condition(*test_cond, *is_double);


    mach_inst(address, MachInst::Mop(Operation::Omove, Arc::new(args), res)) <--
        pmovss(address, rd, rs),
        op_register(rd, res_str),
        op_register(rs, arg_str),
        reg_xmm(res_str),
        reg_xmm(arg_str),
        let res = Mreg::x86(res_str),
        let arg = Mreg::x86(arg_str),
        let args = vec![arg];

    mach_inst(addr, MachInst::Mload(MemoryChunk::MFloat32, addressing, Arc::new(args), dst)) <--
        pmovss(addr, dst_sym, src),
        op_register(dst_sym, dst_str),
        reg_xmm(dst_str),
        let dst = Mreg::x86(dst_str),
        op_indirect(src, _, r2, _, _scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    mach_inst(addr, MachInst::Mstore(MemoryChunk::MFloat32, addressing, Arc::new(args), src)) <--
        pmovss(addr, dst, src_sym),
        op_register(src_sym, src_str),
        reg_xmm(src_str),
        let src = Mreg::x86(src_str),
        op_indirect(dst, _, r2, _, _scale, disp, _),
        if *r2 != "RBP" && *r2 != "RSP",
        let addrmode = Addrmode{
            base: Some(Ireg::from(r2)),
            index: None,
            disp: Displacement::from(*disp),
        },
        effective_address_size(addr, address_size),
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, *address_size);

    expand_builtin_inline(addr, bswap_name, Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))]), BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))) <--
        pswap(addr, res),
        op_register(res, r_str),
        let bswap_name = bswap_builtin_for_reg(r_str);

    expand_builtin_inline(
        addr,
        bswap_name,
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pmov(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str),
        pswap(addr1, res),
        next(addr, addr1),
        if a1 != res,
        let bswap_name = bswap_builtin_for_reg(a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_clz",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsr(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str),
        pxor(addr1, res, n),
        op_immediate(n, imm_str, _),
        if *imm_str == 31,
        next(addr, addr1);

    expand_builtin_inline(
        addr,
        "__builtin_clzl",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsr(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str),
        pxor(addr1, res, n),
        op_immediate(n, imm_str, _),
        if *imm_str == 31,
        next(addr, addr1);

    expand_builtin_inline(
        addr,
        "__builtin_clzl",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsr(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str),
        pxor(addr1, res, n),
        op_immediate(n, imm_str, _),
        if *imm_str == 63,
        next(addr, addr1);

    expand_builtin_inline(
        addr,
        "__builtin_clzll",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsr(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str),
        pxor(addr1, res, n),
        op_immediate(n, imm_str, _),
        if *imm_str == 63,
        next(addr, addr1);

    expand_builtin_inline(
        addr,
        "__builtin_ctz",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsf(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_ctzl",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsf(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_ctzl",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsf(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_ctzll",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(r_str)))
    ) <--
        pbsf(addr, res, a1),
        op_register(res, r_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_ctzll",
        Arc::new(vec![
            BuiltinArg::BA(Mreg::x86(Ireg::from(ah))),
            BuiltinArg::BA(Mreg::x86(Ireg::from(al)))
        ]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        ptest(addr, al, *al),
        op_register(ah, ah_str),
        op_register(al, al_str),
        next(addr, addr1),
        pjcc(addr1, TestCond::CondE, lbl1),
        next(addr1, addr2),
        pbsf(addr2, res, al),
        op_register(res, res_str),
        next(addr2, addr3),
        pjmp(addr3, lbl2),
        next(addr3, addr4),
        plabel(addr4, lbl1),
        next(addr4, addr5),
        next(addr5, addr6),
        next(addr6, addr7),
        pbsf(addr5, res, ah),
        padd(addr6, res, n),
        op_immediate(n, imm_str, _),
        if *imm_str == 32,
        plabel(addr7, lbl2);

    expand_builtin_inline(
        addr,
        "__builtin_fsqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrt(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_sqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrt(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_fsqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrtsd(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_sqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrtsd(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_fsqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrtss(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr,
        "__builtin_sqrt",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(a1)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(res_str)))
    ) <--
        psqrtss(addr, res, a1),
        op_register(res, res_str),
        op_register(a1, a1_str);

    expand_builtin_inline(
        addr1,
        "__builtin_negl",
        Arc::new(vec![
            BuiltinArg::BA(Mreg::x86(Ireg::from(ah_str))),
            BuiltinArg::BA(Mreg::x86(Ireg::from(al_str)))
        ]),
        BuiltinArg::BA(Mreg::x86(Ireg::from(rh_str)))
    ) <--
        pneg(addr1, rl),
        next(addr1, addr2),
        padc(addr2, rh, _),
        next(addr2, addr3),
        pneg(addr3, rh),
        op_register(rh, rh_str),
        op_register(rl, rl_str),
        op_register(ah, ah_str),
        op_register(al, al_str);


    expand_builtin_inline(
        addr,
        "__builtin_va_start",
        Arc::new(vec![BuiltinArg::BA(Mreg::x86(Ireg::from(*a)))]),
        BuiltinArg::BA(Mreg::x86(Ireg::Unknown))
    ) <--
        expand_builtin_va_start_32(addr, a);

    expand_builtin_va_start_32(addr1, Ireg::from(reg2)) <--
        plea(addr1, rax, mem_addr),
        op_register(rax, "RAX"),
        op_indirect(mem_addr, "RSP", _, _, 1, ofs, _ ),
        next(addr1, addr2),
        pmov(addr2, r, rax),
        op_indirect(r, reg1, reg2, reg3, scale, disp, _),
        op_register(rax, "RAX");

    mach_inst(addr, MachInst::Mbuiltin((*name).to_string(), args.to_vec(), (*res).clone())) <--
        expand_builtin_inline(addr, name, args, res),
        builtins(*name);

    mach_inst(addr, MachInst::Mlabel((*lbl).to_string())) <--
        symbols(addr, lbl, _);


    pallocframe(addr, sz) <--
        psub(addr, rsp_sym, sz_sym),
        op_register(rsp_sym, rsp_str),
        reg_sp(rsp_str),
        op_immediate(sz_sym, sz, _);

    // Clang stack alignment: push of scratch register at function entry instead of sub $8, %rsp
    #[local] relation push_align_alloc(Symbol, Address, i64);

    push_align_alloc(func_name, start_addr, 8) <--
        func_span(func_name, start_addr, end_addr),
        ppush(start_addr, sym),
        op_register(sym, reg_str),
        !is_callee_saved(Mreg::x86(*reg_str)),
        !reg_sp(reg_str);

    push_align_alloc(func_name, push_addr, 8) <--
        func_span(func_name, start_addr, end_addr),
        next(start_addr, push_addr),
        if push_addr < end_addr,
        ppush(push_addr, sym),
        op_register(sym, reg_str),
        !is_callee_saved(Mreg::x86(*reg_str)),
        !reg_sp(reg_str);

    push_align_alloc(func_name, push_addr, 8) <--
        func_span(func_name, start_addr, end_addr),
        next(start_addr, mid_addr),
        next(mid_addr, push_addr),
        if push_addr < end_addr,
        ppush(push_addr, sym),
        op_register(sym, reg_str),
        !is_callee_saved(Mreg::x86(*reg_str)),
        !reg_sp(reg_str);

    pallocframe_by_func(func_name, alloc_address, stack_size) <--
        func_span(func_name, start_addr, end_addr),
        pallocframe(alloc_address, stack_size),
        if alloc_address >= start_addr,
        if alloc_address < end_addr;

    func_stacksz(start_addr, end_addr, *func_name, stack_size_u64) <--
        func_span(func_name, start_addr, end_addr),
        agg stack_size = aggregators::sum(sz) in pallocframe_by_func(func_name, _, sz),
        let stack_size_u64 = stack_size.max(0) as u64;

    // Functions with no sub RSP but with alignment pushes: use push-based stack size
    func_stacksz(start_addr, end_addr, *func_name, push_size_u64) <--
        func_span(func_name, start_addr, end_addr),
        !pallocframe_by_func(func_name, _, _),
        agg push_size = aggregators::sum(sz) in push_align_alloc(func_name, _, sz),
        let push_size_u64 = push_size.max(0) as u64;

    // Functions with no sub RSP and no alignment pushes: zero stack size
    func_stacksz(start_addr, end_addr, *func_name, 0u64) <--
        func_span(func_name, start_addr, end_addr),
        !pallocframe_by_func(func_name, _, _),
        !push_align_alloc(func_name, _, _);


    relation is_arg_reg(Mreg);

    // Emit signed+unsigned chunk variants for byte/word; type inference narrows later

    // Load: byte signedness variants
    mach_inst(addr, MachInst::Mload(MemoryChunk::MInt8Signed, addressing.clone(), args.clone(), *dst)) <--
        mach_inst(addr, ?MachInst::Mload(MemoryChunk::MInt8Unsigned, addressing, args, dst));

    mach_inst(addr, MachInst::Mload(MemoryChunk::MInt8Unsigned, addressing.clone(), args.clone(), *dst)) <--
        mach_inst(addr, ?MachInst::Mload(MemoryChunk::MInt8Signed, addressing, args, dst));

    // Load: word signedness variants
    mach_inst(addr, MachInst::Mload(MemoryChunk::MInt16Signed, addressing.clone(), args.clone(), *dst)) <--
        mach_inst(addr, ?MachInst::Mload(MemoryChunk::MInt16Unsigned, addressing, args, dst));

    mach_inst(addr, MachInst::Mload(MemoryChunk::MInt16Unsigned, addressing.clone(), args.clone(), *dst)) <--
        mach_inst(addr, ?MachInst::Mload(MemoryChunk::MInt16Signed, addressing, args, dst));

    // Store: byte signedness variants
    mach_inst(addr, MachInst::Mstore(MemoryChunk::MInt8Signed, addressing.clone(), args.clone(), *src)) <--
        mach_inst(addr, ?MachInst::Mstore(MemoryChunk::MInt8Unsigned, addressing, args, src));

    mach_inst(addr, MachInst::Mstore(MemoryChunk::MInt8Unsigned, addressing.clone(), args.clone(), *src)) <--
        mach_inst(addr, ?MachInst::Mstore(MemoryChunk::MInt8Signed, addressing, args, src));

    // Store: word signedness variants
    mach_inst(addr, MachInst::Mstore(MemoryChunk::MInt16Signed, addressing.clone(), args.clone(), *src)) <--
        mach_inst(addr, ?MachInst::Mstore(MemoryChunk::MInt16Unsigned, addressing, args, src));

    mach_inst(addr, MachInst::Mstore(MemoryChunk::MInt16Unsigned, addressing.clone(), args.clone(), *src)) <--
        mach_inst(addr, ?MachInst::Mstore(MemoryChunk::MInt16Signed, addressing, args, src));

    // Diagnostic safety net: an instruction with a capstone reg_def but no lowering still KILLS the def it overwrote in the reaching-def lattice, so collect such sites for the wrapper to warn about.
    relation unhandled_reg_def_site(Address, &'static str);
    unhandled_reg_def_site(addr, mnemonic) <--
        reg_def(addr, _),
        unrefinedinstruction(addr, _, mnemonic, _, _, _, _, _, _, _),
        !mach_inst(addr, _),
        !trim_instruction(addr),
        // SETcc's Ocmp is correctly keyed at the fused CMP's address, so its own address legitimately has a reg_def and no mach_inst; exclude it so the diagnostic only fires for an unlowered mnemonic.
        !psetcc(addr, _, _);

}

fn addr32_modes_by_instruction(
    db: &DecompileDB,
) -> BTreeMap<Address, (Addressing, Arc<Vec<Mreg>>)> {
    let address_sizes: BTreeMap<Address, u8> = db
        .rel_iter::<(Address, u8)>("instruction_address_size")
        .copied()
        .collect();
    let indirect: BTreeMap<Symbol, (&'static str, &'static str, &'static str, i64, i64)> = db
        .rel_iter::<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )>("op_indirect")
        .map(|(operand, segment, base, index, scale, disp, _)| {
            (*operand, (*segment, *base, *index, *scale, *disp))
        })
        .collect();

    let mut result = BTreeMap::new();
    for (addr, _, _, _, op1, op2, op3, op4, _, _) in db.rel_iter::<(
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
    {
        if address_sizes.get(addr) != Some(&4) {
            continue;
        }
        let memory: Vec<_> = [*op1, *op2, *op3, *op4]
            .into_iter()
            .filter_map(|operand| indirect.get(&operand).copied())
            .collect();
        let [(segment, base, index, scale, disp)] = memory.as_slice() else {
            continue;
        };
        if !is_no_address_register(segment)
            || (!is_no_address_register(base) && !is_addr32_gp_name(base))
            || (!is_no_address_register(index) && !is_addr32_gp_name(index))
            || (is_no_address_register(base) && is_no_address_register(index))
        {
            continue;
        }
        let addrmode = Addrmode {
            base: (!is_no_address_register(base)).then(|| Ireg::from(*base)),
            index: (!is_no_address_register(index)).then(|| (Ireg::from(*index), *scale)),
            disp: Displacement::Const(*disp),
        };
        if let Ok((addressing, args)) = transl_addressing_rev_sized(addrmode, None, 4) {
            result.insert(*addr, (addressing, Arc::new(args)));
        }
    }
    result
}

// Normalize before LinearPass sees Mach. This is load-bearing for EBP: the
// generic decoder register family is RBP, and an unnormalized Ainstack would
// be scalarized into a local slot before RTL could recover the real addr32
// expression. The final RTL pass repeats the wrapper check for direct/fused
// producers which bypass Mach.
fn normalize_addr32_asm_outputs(db: &mut DecompileDB) {
    let modes = addr32_modes_by_instruction(db);
    if modes.is_empty() {
        return;
    }

    let mach: ascent::boxcar::Vec<(Address, MachInst)> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .map(|(addr, inst)| {
            let Some((addressing, args)) = modes.get(addr) else {
                return (*addr, inst.clone());
            };
            let normalized = match inst {
                MachInst::Mload(chunk, _, _, dst) => {
                    MachInst::Mload(*chunk, addressing.clone(), args.clone(), *dst)
                }
                MachInst::Mstore(chunk, _, _, src) => {
                    MachInst::Mstore(*chunk, addressing.clone(), args.clone(), *src)
                }
                MachInst::Mop(Operation::Olea(_) | Operation::Oleal(_), _, dst) => {
                    let op = if matches!(inst, MachInst::Mop(Operation::Oleal(_), _, _)) {
                        Operation::Oleal(addressing.clone())
                    } else {
                        Operation::Olea(addressing.clone())
                    };
                    MachInst::Mop(op, args.clone(), *dst)
                }
                _ => inst.clone(),
            };
            (*addr, normalized)
        })
        .collect();
    db.rel_set("mach_inst", mach);

    let float_loads: ascent::boxcar::Vec<(
        Address,
        Operation,
        MemoryChunk,
        Addressing,
        Arc<Vec<Mreg>>,
        Mreg,
        bool,
    )> = db
        .rel_iter::<(
            Address,
            Operation,
            MemoryChunk,
            Addressing,
            Arc<Vec<Mreg>>,
            Mreg,
            bool,
        )>("float_load_op")
        .map(|(addr, op, chunk, addressing, args, dst, unary)| {
            let (addressing, args) = modes
                .get(addr)
                .map(|(addressing, args)| (addressing.clone(), args.clone()))
                .unwrap_or_else(|| (addressing.clone(), args.clone()));
            (*addr, op.clone(), *chunk, addressing, args, *dst, *unary)
        })
        .collect();
    db.rel_set("float_load_op", float_loads);
}

// Segment-relative memory cannot be represented by Mach addressing.  The
// declarative seed above makes the site structured-unsupported; removing its
// Mach interpretation here also prevents a direct FS/GS stack MOV from first
// becoming Mgetstack/Msetstack (and later an ordinary scalar/home-slot move).
fn suppress_unmodeled_segment_mach_outputs(db: &mut DecompileDB) {
    let segmented_operands: BTreeSet<Symbol> = db
        .rel_iter::<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )>("op_indirect")
        .filter_map(|(operand, segment, ..)| is_unmodeled_segment(segment).then_some(*operand))
        .collect();
    if segmented_operands.is_empty() {
        return;
    }

    let segmented_addresses: BTreeSet<Address> = db
        .rel_iter::<(
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
        .filter_map(|(address, _, _, _, op1, op2, op3, op4, _, _)| {
            [*op1, *op2, *op3, *op4]
                .into_iter()
                .any(|operand| segmented_operands.contains(&operand))
                .then_some(*address)
        })
        .collect();

    let mach: ascent::boxcar::Vec<(Address, MachInst)> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| !segmented_addresses.contains(address))
        .cloned()
        .collect();
    db.rel_set("mach_inst", mach);
}

pub struct AsmPass;

impl IRPass for AsmPass {
    fn name(&self) -> &'static str {
        "asm"
    }

    fn run(&self, db: &mut DecompileDB) {
        // AArch64 is lowered by aarch64_asm_pass; the two are mutually exclusive, since the architectures share mnemonic spellings with different operand counts, order and semantics.
        if db.abi().arch == crate::abi::Arch::Aarch64 {
            return;
        }

        // Seed from the decoder-owned immutable relations, never from the
        // mutable public reg_use/reg_def outputs.  Otherwise a second run on
        // the same DB promotes the first run's synthesized uses to raw facts.
        let decoded_reg_uses = db
            .rel_iter::<(Address, Mreg)>("decoded_reg_use")
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        let decoded_reg_defs = db
            .rel_iter::<(Address, Mreg)>("decoded_reg_def")
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        db.rel_set("asm_reg_use_seed", decoded_reg_uses);
        db.rel_set("asm_reg_def_seed", decoded_reg_defs);

        // These are pass outputs, not another source of decoded facts.  A
        // reused DecompileDB can still contain the previous AsmPass snapshot;
        // swapping that snapshot into the fresh program would union stale
        // rows with the new seed-derived output.
        db.rel_set("asm_reg_use", ascent::boxcar::Vec::<(Address, Mreg)>::new());
        db.rel_set("asm_reg_def", ascent::boxcar::Vec::<(Address, Mreg)>::new());
        db.rel_set(
            "unsupported_stack_address_seed",
            ascent::boxcar::Vec::<(Address, Address, Symbol)>::new(),
        );

        let mut prog = AsmPassProgram::default();
        prog.swap_db_fields(db);
        prog.run();

        // Safety diagnostic: warn once per mnemonic for any instruction that wrote a register but produced no mach_inst, since its def still kills the value it overwrote in the reaching-def lattice.
        {
            use std::collections::BTreeMap;
            let mut per_mnem: BTreeMap<&'static str, (usize, Address)> = BTreeMap::new();
            for (addr, mnem) in prog.unhandled_reg_def_site.iter() {
                let entry = per_mnem.entry(*mnem).or_insert((0usize, *addr));
                entry.0 += 1;
                if *addr < entry.1 {
                    entry.1 = *addr;
                }
            }
            for (mnem, (count, first_addr)) in per_mnem {
                warn!(
                    "asm lowering: instruction '{}' writes a register but produced no mach_inst \
                     ({} site(s), first at {:#x}); its dropped def still kills the value it \
                     overwrote in reaching-defs -- a lowering for '{}' is likely missing.",
                    mnem, count, first_addr, mnem
                );
            }
        }

        if db.measure_rule_times {
            let pass_name = "AsmPassProgram".to_string();
            let summary = format!(
                "=== Rule Times ===\n{}\n\n=== Relation Sizes ===\n{}",
                prog.scc_times_summary(),
                prog.relation_sizes_summary(),
            );
            eprintln!(
                "[RULE_TIMES] pass {}\n{}\n[/RULE_TIMES]",
                pass_name, summary
            );
            db.rule_time_reports.push((pass_name, summary));
        }

        prog.swap_db_fields(db);
        normalize_addr32_asm_outputs(db);
        suppress_unmodeled_segment_mach_outputs(db);
    }

    declare_io_from!(AsmPassProgram);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressingError {
    NeedsSymbolResolution {
        address: i64,
    },
    UnknownRegister {
        register: Ireg,
    },
    UnsupportedPattern {
        addrmode: Addrmode,
    },
    #[allow(dead_code)]
    InvalidRegisterCombination {
        base: Option<Ireg>,
        index: Option<(Ireg, i64)>,
    },
}

impl std::fmt::Display for AddressingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddressingError::NeedsSymbolResolution { address } => {
                write!(f, "Address 0x{:x} needs symbol resolution", address)
            }
            AddressingError::UnknownRegister { register } => {
                write!(f, "Unknown register: {:?}", register)
            }
            AddressingError::UnsupportedPattern { addrmode } => {
                write!(f, "Unsupported addrmode pattern: {:?}", addrmode)
            }
            AddressingError::InvalidRegisterCombination { base, index } => {
                write!(
                    f,
                    "Invalid register combination: base={:?}, index={:?}",
                    base, index
                )
            }
        }
    }
}

#[inline]

// Returns true for valid general-purpose registers (excludes Unknown and RIP).
fn is_valid_gpr(r: Ireg) -> bool {
    !matches!(r, Ireg::Unknown | Ireg::RIP)
}

// Map a 64-bit GPR name to its 32-bit alias (RAX->EAX); pass through 32-bit/unknown. Used by mod-synth dividend-holder check to equate EAX with EDI after `MOVSXD rax, edi`.
pub fn reg_low32_alias(s: &str) -> &str {
    match s {
        "RAX" => "EAX",
        "RBX" => "EBX",
        "RCX" => "ECX",
        "RDX" => "EDX",
        "RDI" => "EDI",
        "RSI" => "ESI",
        "RBP" => "EBP",
        "RSP" => "ESP",
        "R8" => "R8D",
        "R9" => "R9D",
        "R10" => "R10D",
        "R11" => "R11D",
        "R12" => "R12D",
        "R13" => "R13D",
        "R14" => "R14D",
        "R15" => "R15D",
        other => other,
    }
}

// Check if register name is an 8-bit GPR.
pub fn is_reg_8(name: &str) -> bool {
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
            | "SPL"
            | "BPL"
            | "R8B"
            | "R9B"
            | "R10B"
            | "R11B"
            | "R12B"
            | "R13B"
            | "R14B"
            | "R15B"
    )
}

// True for the legacy high-8 sub-registers AH/BH/CH/DH, which collapse to their parent Ireg, so a mask test through one must use M<<8 against the parent to stay equivalent.
pub fn is_reg_high8(name: &str) -> bool {
    matches!(name, "AH" | "BH" | "CH" | "DH")
}

// Adjust a TEST/AND mask immediate for a high-8 register operand, shifting it into the parent register's bit positions; other registers keep the mask unchanged.
pub fn high8_mask_adjust(reg_name: &str, mask: i64) -> i64 {
    if is_reg_high8(reg_name) {
        mask << 8
    } else {
        mask
    }
}

// Check if register name is a 16-bit GPR.
pub fn is_reg_16(name: &str) -> bool {
    matches!(
        name,
        "AX" | "BX"
            | "CX"
            | "DX"
            | "SI"
            | "DI"
            | "SP"
            | "BP"
            | "R8W"
            | "R9W"
            | "R10W"
            | "R11W"
            | "R12W"
            | "R13W"
            | "R14W"
            | "R15W"
    )
}

// Conditions for which a jcc after an add tests the signed RESULT against zero exactly: ZF and the signed conditions qualify, unsigned CF-based ones do not, since CF is carry-out.
fn arith_result_testcond_ok(c: TestCond) -> bool {
    matches!(
        c,
        TestCond::CondE
            | TestCond::CondNe
            | TestCond::CondL
            | TestCond::CondLe
            | TestCond::CondG
            | TestCond::CondGe
    )
}

// Returns true if the mnemonic modifies CPU flags; conservative (true when uncertain); used by gap-bridging rules to detect flag-safe instructions between CMP/TEST and CMOV.
fn is_flag_setting(mnem: &str) -> bool {
    !matches!(
        mnem,
        "MOV"
            | "MOVZX"
            | "MOVSX"
            | "MOVSXD"
            | "MOVABS"
            | "LEA"
            | "NOP"
            | "ENDBR64"
            | "ENDBR32"
            | "PUSH"
            | "POP"
            | "XCHG"
            | "BSWAP"
            | "VZEROUPPER"
            | "VZEROALL"
            | "MOVAPS"
            | "MOVUPS"
            | "MOVAPD"
            | "MOVUPD"
            | "VMOVAPS"
            | "VMOVUPS"
            | "VMOVAPD"
            | "VMOVUPD"
            | "MOVSS"
            | "MOVSD"
            | "MOVDQA"
            | "MOVDQU"
            | "VMOVSS"
            | "VMOVSD"
            | "VMOVDQA"
            | "VMOVDQU"
            | "MOVQ"
            | "MOVD"
            | "MOVHPS"
            | "MOVLPS"
            | "VMOVQ"
            | "VMOVD"
            | "CMOVE"
            | "CMOVNE"
            | "CMOVZ"
            | "CMOVNZ"
            | "CMOVB"
            | "CMOVC"
            | "CMOVNAE"
            | "CMOVAE"
            | "CMOVNC"
            | "CMOVNB"
            | "CMOVBE"
            | "CMOVNA"
            | "CMOVA"
            | "CMOVNBE"
            | "CMOVL"
            | "CMOVNGE"
            | "CMOVGE"
            | "CMOVNL"
            | "CMOVLE"
            | "CMOVNG"
            | "CMOVG"
            | "CMOVNLE"
            | "CMOVS"
            | "CMOVNS"
            | "CMOVP"
            | "CMOVPE"
            | "CMOVNP"
            | "CMOVPO"
    )
}

// Returns the absolute address if the addrmode requires symbol resolution (RIP-relative or absolute).
pub fn addrmode_needs_symbol_resolution(am: &Addrmode) -> Option<i64> {
    match am {
        Addrmode {
            base: Some(Ireg::RIP),
            index: None,
            disp: Displacement::Const(n),
        } if *n != 0 => Some(*n),

        Addrmode {
            base: None,
            index: None,
            disp: Displacement::Const(n),
        } if *n != 0 => Some(*n),

        // Scaled index with no base: absolute displacement needs symbol resolution
        Addrmode {
            base: None,
            index: Some(_),
            disp: Displacement::Const(n),
        } if *n != 0 => Some(*n),

        _ => None,
    }
}

// Reverse of CompCert's transl_addressing (see x86/Asmgen.v).
pub fn transl_addressing_rev(
    am: Addrmode,
    resolved_symbol: Option<(Ident, i64)>,
) -> Result<(Addressing, Vec<Mreg>), String> {
    transl_addressing_rev_inner(am, resolved_symbol).map_err(|e| e.to_string())
}

/// Reverse an x86 address mode while preserving the architectural address
/// size. In long mode, a 0x67 override computes the effective address in
/// 32-bit arithmetic and zero-extends it; treating EBP as RBP here would turn
/// a scratch register into Ainstack and lose those semantics before RTL.
pub fn transl_addressing_rev_sized(
    am: Addrmode,
    resolved_symbol: Option<(Ident, i64)>,
    address_size: u8,
) -> Result<(Addressing, Vec<Mreg>), String> {
    if address_size != 4 {
        return transl_addressing_rev(am, resolved_symbol);
    }

    let Addrmode { base, index, disp } = am;
    let Displacement::Const(disp) = disp else {
        return Err("addr32 symbolic displacement is unsupported".to_string());
    };
    if resolved_symbol.is_some() {
        return Err("addr32 symbol resolution is unsupported".to_string());
    }
    let valid = |reg: Ireg| is_valid_gpr(reg) && reg != Ireg::RSP;
    let (inner, args) = match (base, index) {
        (Some(base), None) if valid(base) => (Addressing::Aindexed(disp), vec![base.into()]),
        (Some(base), Some((index, 1))) if valid(base) && valid(index) => {
            (Addressing::Aindexed2(disp), vec![base.into(), index.into()])
        }
        (Some(base), Some((index, scale)))
            if valid(base) && valid(index) && matches!(scale, 2 | 4 | 8) =>
        {
            (
                Addressing::Aindexed2scaled(scale, disp),
                vec![base.into(), index.into()],
            )
        }
        (None, Some((index, scale))) if valid(index) && matches!(scale, 1 | 2 | 4 | 8) => {
            (Addressing::Ascaled(scale, disp), vec![index.into()])
        }
        _ => return Err(format!("unsupported addr32 address mode: {am:?}")),
    };
    Ok((Addressing::Aaddr32(Box::new(inner)), args))
}

fn transl_addressing_rev_inner(
    am: Addrmode,
    resolved_symbol: Option<(Ident, i64)>,
) -> Result<(Addressing, Vec<Mreg>), AddressingError> {
    match am {
        Addrmode {
            base: Some(Ireg::RSP),
            index: None,
            disp: Displacement::Const(n),
        } => Ok((Addressing::Ainstack(n), vec![])),

        Addrmode {
            base: Some(Ireg::RBP),
            index: None,
            disp: Displacement::Const(n),
        } => Ok((Addressing::Ainstack(n), vec![])),

        Addrmode {
            base: None,
            index: None,
            disp: Displacement::Symbol { ident, ofs },
        } => Ok((Addressing::Aglobal(ident, ofs), vec![])),

        Addrmode {
            base: Some(Ireg::RIP),
            index: None,
            disp: Displacement::Symbol { ident, ofs },
        } => Ok((Addressing::Aglobal(ident, ofs), vec![])),

        Addrmode {
            base: Some(Ireg::RIP),
            index: None,
            disp: Displacement::Const(n),
        } => {
            if let Some((ident, offset)) = resolved_symbol {
                Ok((Addressing::Aglobal(ident, offset), vec![]))
            } else {
                Err(AddressingError::NeedsSymbolResolution { address: n })
            }
        }

        Addrmode {
            base: None,
            index: None,
            disp: Displacement::Const(n),
        } if n != 0 => {
            if let Some((ident, offset)) = resolved_symbol {
                Ok((Addressing::Aglobal(ident, offset), vec![]))
            } else {
                Err(AddressingError::NeedsSymbolResolution { address: n })
            }
        }

        Addrmode {
            base: None,
            index: None,
            disp: Displacement::Const(0),
        } => {
            if let Some((ident, offset)) = resolved_symbol {
                Ok((Addressing::Aglobal(ident, offset), vec![]))
            } else {
                Err(AddressingError::UnsupportedPattern { addrmode: am })
            }
        }

        Addrmode {
            base: Some(r1),
            index: None,
            disp: Displacement::Const(n),
        } if is_valid_gpr(r1) && r1 != Ireg::RSP => Ok((Addressing::Aindexed(n), vec![r1.into()])),

        Addrmode {
            base: Some(r1),
            index: Some((r2, 1)),
            disp: Displacement::Const(n),
        } if is_valid_gpr(r1) && is_valid_gpr(r2) => {
            Ok((Addressing::Aindexed2(n), vec![r1.into(), r2.into()]))
        }

        Addrmode {
            base: None,
            index: Some((r1, sc)),
            disp: Displacement::Const(n),
        } if is_valid_gpr(r1) => {
            if let Some((ident, offset)) = resolved_symbol {
                Ok((Addressing::Abasedscaled(sc, ident, offset), vec![r1.into()]))
            } else {
                Ok((Addressing::Ascaled(sc, n), vec![r1.into()]))
            }
        }

        Addrmode {
            base: Some(r1),
            index: Some((r2, sc)),
            disp: Displacement::Const(n),
        } if is_valid_gpr(r1) && is_valid_gpr(r2) && sc != 1 => Ok((
            Addressing::Aindexed2scaled(sc, n),
            vec![r1.into(), r2.into()],
        )),

        Addrmode {
            base: Some(r1),
            index: None,
            disp: Displacement::Symbol { ident, ofs },
        } if is_valid_gpr(r1) => Ok((Addressing::Abased(ident, ofs), vec![r1.into()])),

        Addrmode {
            base: None,
            index: Some((r1, sc)),
            disp: Displacement::Symbol { ident, ofs },
        } if is_valid_gpr(r1) => Ok((Addressing::Abasedscaled(sc, ident, ofs), vec![r1.into()])),

        Addrmode {
            base: Some(r1),
            index: Some((r2, sc)),
            disp: Displacement::Symbol { ident, ofs },
        } if is_valid_gpr(r1) && is_valid_gpr(r2) => {
            if sc == 1 {
                log::debug!(
                    "Addressing base({:?}) + index({:?}) + symbol({}, {}): using Abased, index discarded",
                    r1, r2, ident, ofs
                );
                Ok((Addressing::Abased(ident, ofs), vec![r1.into()]))
            } else {
                log::warn!(
                    "Unsupported addressing: base({:?}) + {}*index({:?}) + symbol({}, {}). \
                     No CompCert mode can represent this pattern.",
                    r1,
                    sc,
                    r2,
                    ident,
                    ofs
                );
                Err(AddressingError::UnsupportedPattern { addrmode: am })
            }
        }

        Addrmode {
            base: Some(r1),
            index: Some((Ireg::Unknown, _)),
            disp: Displacement::Const(n),
        } if is_valid_gpr(r1) => Ok((Addressing::Aindexed(n), vec![r1.into()])),

        Addrmode {
            base: Some(Ireg::RIP),
            index: Some((r1, sc)),
            disp,
        } if is_valid_gpr(r1) => match disp {
            Displacement::Symbol { ident, ofs } => {
                if sc == 1 {
                    Ok((Addressing::Abased(ident, ofs), vec![r1.into()]))
                } else {
                    Ok((Addressing::Abasedscaled(sc, ident, ofs), vec![r1.into()]))
                }
            }
            Displacement::Const(n) => {
                if let Some((ident, offset)) = resolved_symbol {
                    if sc == 1 {
                        Ok((Addressing::Abased(ident, offset), vec![r1.into()]))
                    } else {
                        Ok((Addressing::Abasedscaled(sc, ident, offset), vec![r1.into()]))
                    }
                } else {
                    if sc == 1 {
                        Ok((Addressing::Aindexed(n), vec![r1.into()]))
                    } else {
                        Ok((Addressing::Ascaled(sc, n), vec![r1.into()]))
                    }
                }
            }
        },

        Addrmode {
            base: Some(Ireg::Unknown),
            ..
        } => Err(AddressingError::UnknownRegister {
            register: Ireg::Unknown,
        }),

        _ => Err(AddressingError::UnsupportedPattern { addrmode: am }),
    }
}

#[cfg(test)]
mod privileged_instruction_tests {
    use super::*;
    use crate::decompile::passes::linear_pass::LinearPass;

    const CR8_ADDR: Address = 0x1000;
    const INT2C_ADDR: Address = 0x1003;
    const INT2D_ADDR: Address = 0x1005;
    const RET_ADDR: Address = 0x1007;
    const NONE: Symbol = "privileged_test_none";

    fn on_pipeline_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("privileged-instruction-test".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(test)
            .expect("spawn privileged instruction test")
            .join()
            .expect("privileged instruction test panicked");
    }

    fn instruction(
        addr: Address,
        size: usize,
        mnemonic: &'static str,
        op1: Symbol,
        op2: Symbol,
    ) -> (
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
    ) {
        (addr, size, "", mnemonic, op1, op2, NONE, NONE, 0, 0)
    }

    #[test]
    fn cr8_and_int2c_use_only_the_dedicated_builtin_lowerings() {
        on_pipeline_stack(|| {
        const CR8_OP: Symbol = "privileged_test_cr8";
        const RAX_OP: Symbol = "privileged_test_rax";
        const INT2C_OP: Symbol = "privileged_test_int2c";
        const INT2D_OP: Symbol = "privileged_test_int2d";

        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        db.rel_push("builtins", ("__readcr8",));
        db.rel_push("builtins", ("__int2c",));
        db.rel_push("op_register", (CR8_OP, "CR8"));
        db.rel_push("op_register", (RAX_OP, "RAX"));
        db.rel_push("op_immediate", (INT2C_OP, 0x2c_i64, 0_usize));
        db.rel_push("op_immediate", (INT2D_OP, 0x2d_i64, 0_usize));
        db.rel_push("instruction", instruction(CR8_ADDR, 3, "MOV", CR8_OP, RAX_OP));
        db.rel_push("instruction", instruction(INT2C_ADDR, 2, "INT", INT2C_OP, NONE));
        db.rel_push("instruction", instruction(INT2D_ADDR, 2, "INT", INT2D_OP, NONE));

        AsmPass.run(&mut db);

        assert!(!db
            .rel_iter::<(Address, Symbol, Symbol)>("pmov")
            .any(|(addr, _, _)| *addr == CR8_ADDR));
        assert_eq!(
            db.rel_iter::<(Address, MachInst)>("mach_inst")
                .filter(|(addr, _)| *addr == CR8_ADDR)
                .map(|(_, inst)| inst.clone())
                .collect::<Vec<_>>(),
            vec![MachInst::Mbuiltin(
                "__readcr8".to_string(),
                vec![],
                BuiltinArg::BA(Mreg::AX),
            )]
        );
        assert_eq!(
            db.rel_iter::<(Address, MachInst)>("mach_inst")
                .filter(|(addr, _)| *addr == INT2C_ADDR)
                .map(|(_, inst)| inst.clone())
                .collect::<Vec<_>>(),
            vec![MachInst::Mbuiltin(
                "__int2c".to_string(),
                vec![],
                BuiltinArg::BAInt(0),
            )]
        );
        assert!(!db
            .rel_iter::<(Address, MachInst)>("mach_inst")
            .any(|(addr, _)| *addr == INT2D_ADDR));
        });
    }

    #[test]
    fn int2c_builtin_retains_its_sequential_fallthrough() {
        on_pipeline_stack(|| {
        let mut db = DecompileDB::default();
        db.rel_push(
            "linear_inst",
            (
                INT2C_ADDR,
                LinearInst::Lbuiltin("__int2c".to_string(), vec![], BuiltinArg::BAInt(0)),
            ),
        );
        db.rel_push("linear_inst", (RET_ADDR, LinearInst::Lreturn));
        db.rel_push("next", (INT2C_ADDR, RET_ADDR));

        LinearPass.run(&mut db);

        assert!(db
            .rel_iter::<(Node, LTLInst)>("ltl_inst")
            .any(|(node, inst)| {
                *node == INT2C_ADDR
                    && inst
                        == &LTLInst::Lbuiltin(
                            "__int2c".to_string(),
                            vec![],
                            BuiltinArg::BAInt(0),
                        )
            }));
        assert!(db
            .rel_iter::<(Node, Node)>("ltl_fallthrough")
            .any(|edge| *edge == (INT2C_ADDR, RET_ADDR)));
        });
    }
}
