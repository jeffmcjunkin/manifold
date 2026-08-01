use crate::decompile::elevator::DecompileDB;
use crate::run_pass;

use crate::decompile::passes::asm_pass::transl_addressing_rev_sized;
use crate::decompile::passes::cminor_pass::*;
use crate::decompile::passes::csh_pass::*;
use crate::decompile::passes::pass::IRPass;
use crate::mreg::Mreg;
use crate::util::DEFAULT_VAR;
use crate::x86::asm::{Ireg, TestCond};
use crate::x86::mach::X86Mreg;
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use ascent::ascent_par;
use ascent::lattice::set::Set;
use ascent::Dual;
use either::Either;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

const ENDBR64_LEN: u64 = 4;

// Strict-dominator set plus the node itself, mirroring structuring_pass's dom_set helper. Used by the block-dominator lattice in RTLPassProgram so dominance is maintained as O(blocks * dom-depth) lattice state rather than an O(blocks^2) pairwise/path-avoiding workspace.
fn dom_set_with_self(strict: &Set<Address>, n: Address) -> Set<Address> {
    let mut s = strict.0.clone();
    s.insert(n);
    Set(s)
}

// ABI-1 (3.2e): MOVSX/MOVSXD/MOVZX -> MemoryChunk; returns None for non-extending mnemonics so plain MOV (which also covers callee-save reloads) never fires; (unsigned, 4) rejected because 32-bit writes zero-extend implicitly.
fn extending_load_chunk(mnem: &str, size: usize) -> Option<MemoryChunk> {
    let m = mnem.to_ascii_uppercase();
    let signed = if m == "MOVZX" {
        false
    } else if m == "MOVSX" || m == "MOVSXD" {
        true
    } else {
        return None;
    };
    match (signed, size) {
        (true, 4) => Some(MemoryChunk::MInt32),
        (true, 2) => Some(MemoryChunk::MInt16Signed),
        (true, 1) => Some(MemoryChunk::MInt8Signed),
        (false, 2) => Some(MemoryChunk::MInt16Unsigned),
        (false, 1) => Some(MemoryChunk::MInt8Unsigned),
        _ => None,
    }
}

// Scalar replacement of a mutable Win64 home cell must retain the explicit
// sign/zero extension performed by the load.  The ordinary stack path carries
// this in MemoryChunk; once the backing memory is replaced by one canonical
// scalar, make that conversion explicit instead of degrading it to Omove.
fn extending_load_operation(mnem: &str, size: usize) -> Option<Operation> {
    match extending_load_chunk(mnem, size)? {
        MemoryChunk::MInt8Signed => Some(Operation::Ocast8signed),
        MemoryChunk::MInt8Unsigned => Some(Operation::Ocast8unsigned),
        MemoryChunk::MInt16Signed => Some(Operation::Ocast16signed),
        MemoryChunk::MInt16Unsigned => Some(Operation::Ocast16unsigned),
        MemoryChunk::MInt32 => Some(Operation::Ocast32signed),
        _ => None,
    }
}

fn is_x86_64_gp_register_name(name: &str) -> bool {
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

// Operand decoding collapses subregister spellings into one Mreg family. That
// is useful for value flow, but unsafe for frame analysis: every E*-based
// memory operand has 32-bit address-size semantics. In particular, a proven
// RAX copy of RSP does not make a later [eax] access stack-relative. Keep the
// raw spelling boundary closed over the complete 64-bit GP register set.
fn is_valid_stack_operand_base_name(name: &str) -> bool {
    is_x86_64_gp_register_name(name)
}

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct RTLPassProgram;

    relation arch_bit(i64);
    relation arg_constrained_as_ptr(Node, RTLReg);
    // RSP<->RBP register moves from disassembly; use-specific proofs below
    // decide whether an individual BP-relative access is frame based.
    relation stack_base_move(Address, Symbol, Symbol);
    relation block_in_function(Node, Address);
    relation emit_clight_stmt(Address, Node, ClightStmt);
    relation emit_goto_target(Address, Node);
    relation emit_loop_body(Address, Node, Node);
    relation emit_loop_exit(Address, Node, Node, Condition, Arc<Vec<CsharpminorExpr>>, Node, Node);
    relation emit_switch_chain(Address, Node, RTLReg);
    relation func_param_struct_type_candidate(Address, usize, usize);
    relation func_span(Symbol, Address, Address);
    relation global_struct_catalog(u64, usize, usize, usize);
    relation instr_in_function(Node, Address);
    relation is_external_function(Address);
    relation known_extern_signature(Symbol, usize, XType, Arc<Vec<XType>>);
    relation known_varargs_function(Symbol, usize);

    relation known_func_param_is_ptr(Symbol, usize);
    relation known_func_returns_long(Symbol);
    relation known_func_returns_ptr(Symbol);
    relation main_function(Address);
    relation reg_def_used(Address, Mreg, Address);
    relation asm_reg_use(Address, Mreg);
    relation asm_reg_def(Address, Mreg);
    // CFG-safe entry-frame proof produced by AsmPass.  Every BP/SP shortcut in
    // this pass must be use-specific rather than relying on a function-wide or
    // linear-fallthrough approximation.
    relation rsp_frame_at(Address, Address);
    relation rsp_frame_offset_at(Address, Address, i64);
    relation bp_frame_at(Address, Address);
    relation unsupported_stack_address_seed(Address, Address, Symbol);
    relation unsupported_address_detail_seed(Address, Address, Symbol);
    relation unsupported_address_detail(Address, Address, Symbol);
    unsupported_address_detail(*func, *access, *detail) <--
        unsupported_address_detail_seed(func, access, detail);
    relation unsupported_stack_address(Address, Address, Symbol);
    unsupported_stack_address(*func, *access, *reason) <--
        unsupported_stack_address_seed(func, access, reason);
    relation string_data(String, String, usize);
    relation struct_id_to_canonical(usize, usize);
    relation symbol_resolved_addr(Symbol, Address);

    relation block(Address);
    relation block_boundaries(Address, Address, Address);
    relation block_in_function(Node, Address);
    relation code_in_block(Address, Address);
    relation ddisasm_cfg_edge(Address, Address, Symbol);
    relation emit_var_type_candidate(RTLReg, XType);
    relation instr_in_function(Node, Address);
    relation instruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation instruction_address_size(Address, u8);
    #[local] relation effective_address_size(Address, u8);
    effective_address_size(*addr, *size) <-- instruction_address_size(addr, size);
    effective_address_size(*addr, 8) <--
        instruction(addr, _, _, _, _, _, _, _, _, _),
        !instruction_address_size(addr, _);
    relation ireg_hold_type(String, Typ);
    relation is_not_ptr(RTLReg);
    relation is_ptr(RTLReg);
    relation is_char_ptr(RTLReg);
    relation known_extern_signature(Symbol, usize, XType, Arc<Vec<XType>>);
    relation known_varargs_function(Symbol, usize);

    relation ltl_inst(Node, LTLInst);
    relation ltl_succ(Node, Node);
    relation rtl_next(Node, Node);
    relation mach_imm_indirect_store(Address, i64, MemoryChunk, Mreg, i64);
    relation mach_imm_stack_init(Address, i64, i64, Typ);
    relation arith_load_op(Address, Operation, MemoryChunk, Mreg, i64, Mreg);
    relation float_load_op(Address, Operation, MemoryChunk, Addressing, Arc<Vec<Mreg>>, Mreg, bool);
    // stack_unary_load_op: a write-only unary op reading a spilled SCALAR
    // slot; it consumes the slot's canonical SSA reg directly.
    relation stack_unary_load_op(Address, Operation, Mreg, i64, Mreg);
    // float_arith_stack_op: a binary float arith reading a spilled float SCALAR slot in place, with the slot operand read through its canonical SSA reg rather than a raw frame Iload.
    relation float_arith_stack_op(Address, Operation, Mreg, i64, Mreg);
    relation arith_store_reg(Address, Operation, MemoryChunk, Mreg, i64, Mreg);
    relation arith_store_imm(Address, Operation, MemoryChunk, Mreg, i64);
    relation arith_store_abs_reg(Address, Operation, MemoryChunk, Ident, i64, Mreg);
    relation arith_store_abs_imm(Address, Operation, MemoryChunk, Ident, i64);
    // adc_carry_op: a branchless cmp;adc/sbb conditional-accumulate, expanded here into a 0/1 carry compute plus an accumulator update, since the carry flag has no machine register.
    relation adc_carry_op(Address, Address, Condition, bool, i64, bool, Mreg, Mreg);
    relation flags_and_jump_pair(Address, Address, &'static str);
    relation next(Address, Address);
    relation op_immediate(Symbol, i64, usize);
    relation op_produces_data(Node, RTLReg);
    relation op_produces_ptr(Node, RTLReg);
    relation padd(Address, Symbol, Symbol);
    relation pcmp(Address, Symbol, Symbol);
    // osel_compare_site(cmov_addr, compare_addr): from asm_pass; the compare whose flags this cmov consumes, so condition operands resolve at the compare site rather than the (possibly register-reusing) cmov site.
    relation osel_compare_site(Address, Address);
    relation pdiv(Address, Symbol, Symbol);
    relation pidiv(Address, Symbol, Symbol);
    relation pudiv(Address, Symbol, Symbol);
    relation pjcc(Address, TestCond, Symbol);
    relation plabel(Address, Symbol);
    relation plea(Address, Symbol, Symbol);
    // 3.8 R3 inputs: MOV operand rows plus the asm-side RIP/global resolution, needed for the plain RIP-relative immediate global store, which has no mach/ltl route.
    relation pmov(Address, Symbol, Symbol);
    relation rip_target_addr(Address, Address);
    relation resolved_addr_to_symbol(Address, Ident, i64);
    relation psub(Address, Symbol, Symbol);
    relation reg_def(Address, Mreg);
    relation reg_def_used(Address, Mreg, Address);
    relation reg_use(Address, Mreg);
    relation stack_def(Address, Symbol, i64);
    relation decoded_memory_write_operand(Address, Symbol);
    relation trim_instruction(Address);
    relation stack_def_used(Address, Symbol, i64, Address, Symbol, i64);
    relation symbols(Address, Symbol, Symbol);

    relation base_addr_usage(Node, RTLReg, i64);
    relation op_produces_int(Node, RTLReg);
    relation op_produces_long(Node, RTLReg);
    relation op_produces_float(Node, RTLReg);
    relation op_produces_single(Node, RTLReg);
    relation op_produces_bool(Node, RTLReg);
    relation op_produces_int8signed(Node, RTLReg);
    relation op_produces_int8unsigned(Node, RTLReg);
    relation op_produces_int16signed(Node, RTLReg);
    relation op_produces_int16unsigned(Node, RTLReg);
    relation is_int(RTLReg);
    relation is_long(RTLReg);
    relation is_float(RTLReg);
    relation is_single(RTLReg);
    relation is_bool(RTLReg);
    relation is_int8signed(RTLReg);
    relation is_int8unsigned(RTLReg);
    relation is_int16signed(RTLReg);
    relation is_int16unsigned(RTLReg);
    relation is_signed(RTLReg);
    relation is_unsigned(RTLReg);
    relation must_be_ptr(RTLReg);
    relation has_ptr_type(RTLReg);
    relation has_char_ptr_type(RTLReg);
    relation has_float_type(RTLReg);
    relation has_single_type(RTLReg);
    relation has_long_type(RTLReg);
    relation has_int_type(RTLReg);
    relation has_subint_type(RTLReg);


    relation plt_block(Address, Symbol);
    relation plt_entry(Address, Symbol);
    relation func_stacksz(Address, Address, Symbol, u64);
    relation call_has_known_signature(Node, Symbol, usize, XType);
    relation function_entry_count(Address, Node, i64);

    relation rtl_inst_candidate(Node, RTLInst);
    relation emit_function(Address, Symbol, Node);
    relation rtl_succ_candidate(Node, Node);

    relation rtl_edge_negated(Node, Node);
    // Memory-indirect call (CALL with op_indirect operand that is a true memory load); tuple is (call_addr, base_reg_str, idx_reg_str, scale, disp).
    relation call_through_memory(Address, &'static str, &'static str, i64, i64);
    // Addresses with a real LTL operation (not just a label/branch fallback)
    relation has_ltl_op(Node);
    relation is_arg_reg(Mreg);
    relation is_xmm_arg_reg(Mreg);
    relation is_float_arg_reg(Mreg);
    relation is_caller_saved(Mreg);
    relation abi_int_arg_position(Mreg, usize);
    relation abi_float_arg_position(Mreg, usize);
    relation abi_shared_arg_slots(bool);
    relation abi_first_stack_arg_position(usize);
    relation abi_outgoing_stack_base(i64);
    relation abi_incoming_sp_stack_base(i64);
    relation abi_incoming_bp_stack_base(i64);
    relation abi_stack_slot_size(i64);
    relation ltl_inst_uses_mreg(Node, Mreg);
    relation ltl_inst_reads_addr_base(Node, Mreg);
    relation ltl_builtin_unconverted(Node, Symbol, Arc<Vec<BuiltinArg<Mreg>>>, BuiltinArg<Mreg>);
    relation is_call_or_tailcall(Node);
    relation is_call_clobbered(Node, Mreg);
    relation call_args_collected_candidate(Node, Args);
    relation op_arg_mapping(Node, usize, RTLReg);
    relation op_args_collected(Node, Args);
    relation load_arg_mapping(Node, usize, RTLReg);
    relation load_args_collected(Node, Args);
    relation store_arg_mapping(Node, usize, RTLReg);
    relation store_args_collected(Node, Args);
    relation cond_arg_mapping(Node, usize, RTLReg);
    relation cond_args_collected(Node, Args);
    relation global_var_ref(usize);
    relation global_load_chunk(Ident, MemoryChunk);
    relation emit_global_is_ptr(Ident);
    relation emit_global_is_char_ptr(Ident);
    relation global_addr_reg(Ident, RTLReg);
    relation base_ident_to_symbol(Ident, Symbol);
    relation ident_to_symbol(Ident, Symbol);
    relation comparison_operand(Node, Condition, RTLReg);
    relation rtl_reg_used_in_func(Address, RTLReg);
    relation func_has_div_instr(Address);
    relation temp_cmp_reg(Address, RTLReg);
    relation temp_cmp_mreg_args(Address, usize, Mreg);
    relation temp_cmp_args_rtl(Address, usize, RTLReg);
    // A memory CMP materializes its operand in a root-owned temporary, then a
    // following JCC/SETcc consumes that value at another decoded node.  Keep
    // the exact root, consumer, temporary and shared function owner so the
    // post-fixed-point safety filter can remove only the dependent candidate
    // when the memory root is rejected.
    relation cmp_memory_temp_consumer(Address, Address, RTLReg, Address);
    relation call_returns_value(Address, Mreg);
    relation ax_value_addr(Address, Address);
    relation func_ax_def(Address, Address, RTLReg);
    relation stack_mem_add_imm(Address, i64, i64, usize);
    relation stack_mem_sub_imm(Address, i64, i64, usize);

    // SP-indexed load/store detection: loads/stores with 2 mregs where mregs[0] == SP
    // Retained after the Datalog fixed point: the imperative completeness
    // classifier must see the structural witnesses before synthetic CFG
    // chains are either accepted or rejected atomically.
    relation sp_indexed_load(Node);
    relation sp_indexed_store(Node);
    relation sp_indexed_load_complete(Node);
    relation sp_indexed_store_complete(Node);

    // BP-frame-pointer-indexed load/store detection (mregs[0] == BP in a framed function), expanded like the SP-indexed case, or the base collapses to the whole-frame Olea(Ainstack(0)) and walks off the frame.
    relation bp_indexed_load(Node);
    relation bp_indexed_store(Node);
    relation bp_indexed_load_complete(Node);
    relation bp_indexed_store_complete(Node);

    // The synthetic Olea(Ainstack(ofs)) base from an SP/BP-indexed expansion, driving a per-node stack_xtl and same-offset alias so it resolves to the array's named local instead of a bare frame integer.
    // The raw synthetic base is always emitted so an indexed operation never
    // consumes an undefined RTL register. Normalized coordinates are optional
    // evidence used only to alias bases whose SP/BP provenance is proved.
    #[local] relation indexed_synth_stack_base(Address, Node, Node, Mreg, i64, RTLReg);
    #[local] relation indexed_synth_stack_coord(Address, Node, i64, RTLReg);
    // Selected stack-address base, keyed to its normalized entry-SP
    // coordinate.  Struct recovery consumes this exact provenance instead of
    // combining raw offsets and widths aggregated from unrelated accesses.
    relation normalized_stack_lea_base(Address, Node, RTLReg, i64);
    #[local] relation bp_base_at(Address, Node, i64);
    #[local] relation stack_claim_base_at(Address, Node, Mreg);

    // sp_synth_skip_to_ltl(start, func, dst): start has no ltl_inst; walking next within func reaches dst with one. Function-bounded; used ONLY by SP-indexed-load/store synth so it doesn't affect ltl_fallthrough.
    #[local] relation sp_synth_skip_to_ltl(Address, Address, Node);

    // synth_only_addr: addr has no ltl_inst (fused into a synth chain upstream) yet exists in rtl as synth, bridging the predecessor-side edge without resurrecting the broken global skip-over.
    #[local] relation synth_only_addr(Address);


    rtl_next(src, dst) <-- ltl_succ(src, dst);

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lop(_, args, _)),
        for mreg in args.iter();

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        for mreg in args.iter();

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, args, _)),
        for mreg in args.iter();

    ltl_inst_uses_mreg(addr, src) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, _, src));

    ltl_inst_uses_mreg(addr, reg) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(reg)));

    ltl_inst_uses_mreg(addr, reg) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(Either::Left(reg)));

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lbuiltin(_, args, _)),
        for arg in args.iter(),
        if let BuiltinArg::BA(mreg) = arg;

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lbuiltin(_, args, _)),
        for arg in args.iter(),
        if matches!(arg, BuiltinArg::BASplitLong(_, _) | BuiltinArg::BAAddPtr(_, _)),
        let mregs = extract_builtin_arg_regs(arg),
        for mreg in mregs.iter();

    ltl_inst_uses_mreg(addr, mreg) <--
        ltl_inst(addr, ?LTLInst::Lcond(_, args, _, _)),
        for mreg in args.iter();

    ltl_inst_uses_mreg(addr, arg) <--
        ltl_inst(addr, ?LTLInst::Ljumptable(arg, _));

    ltl_inst_uses_mreg(addr, Mreg::AX) <--
        ltl_inst(addr, LTLInst::Lreturn);

    ltl_inst_uses_mreg(addr, src) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, _, _, _));


    is_call_or_tailcall(*addr) <-- ltl_inst(addr, ?LTLInst::Lcall(_));
    is_call_or_tailcall(*addr) <-- ltl_inst(addr, ?LTLInst::Ltailcall(_));
    is_call_or_tailcall(*addr) <--
        ltl_inst(addr, ?LTLInst::Lbranch(Either::Right(target))),
        emit_function(_, _, target);

    is_call_clobbered(n, r) <-- is_call_or_tailcall(n), is_caller_saved(r);


    call_args_collected_candidate(call_addr, args) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        agg args = build_call_args(pos, reg) in call_arg_mapping(call_addr, pos, reg);

    // Memory-indirect call detection: op_indirect on a CALL means callee is loaded from memory (register-direct CALLs use op_register).
    call_through_memory(*addr, base_str, idx_str, *scale, *disp) <--
        instruction(addr, _, _, "CALL", dst, _, _, _, _, _),
        op_indirect(dst, _, base_str, idx_str, scale, disp, _),
        if *base_str != "NONE";


    // osel_cond_pos: a cmov condition operand that is the register the cmov writes, where clang's reuse makes the value at the compare differ from the cmov-site shadow; non-reuse cmovs are unmarked.
    relation osel_cond_pos(Node, usize, RTLReg);
    osel_cond_pos(addr, pos, cmp_rtl) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Osel(_, _), mregs, dst_reg)),
        osel_compare_site(addr, cmp_addr),
        lop_overwrites_input(addr, dst_reg),
        lop_overwrite_use_id(addr, *dst_reg, src_xtl),
        xtl_canonical(src_xtl, shadow_rtl),
        for (pos, arg) in mregs.iter().enumerate(),
        if pos >= 2,
        if arg == dst_reg,
        reg_rtl(cmp_addr, *arg, cmp_rtl),
        if cmp_rtl != shadow_rtl;

    // cmov_reuse_node(addr): cmov whose flag register was reused as a value holder. The structuring select-to-if lift targets exactly these (dispatch-style selects), leaving other branchless selects for their own recovery passes (jump tables, closed-form switches, plain ternaries).
    relation cmov_reuse_node(Node);
    cmov_reuse_node(addr) <-- osel_cond_pos(addr, _, _);

    op_arg_mapping(addr, pos, cmp_rtl) <--
        osel_cond_pos(addr, pos, cmp_rtl);

    // An Osel's condition operand reflects the compared register's value AT THE COMPARE, so resolve it at osel_compare_site; resolving at the cmov invents a never-defined reg and its producer is dropped.
    #[local] relation osel_cond_at_compare(Node, usize, RTLReg);
    osel_cond_at_compare(addr, pos, cmp_rtl) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Osel(_, _), mregs, dst_reg)),
        osel_compare_site(addr, cmp_addr),
        for (pos, arg) in mregs.iter().enumerate(),
        if pos >= 2,
        if arg != dst_reg,
        reg_rtl(cmp_addr, *arg, cmp_rtl);

    op_arg_mapping(addr, pos, cmp_rtl) <--
        osel_cond_at_compare(addr, pos, cmp_rtl);

    op_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lop(_, mregs, dst_reg)),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg != dst_reg,
        !osel_cond_at_compare(addr, pos, _),
        reg_rtl(addr, *arg, arg_rtl);

    op_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lop(_, mregs, dst_reg)),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg == dst_reg,
        !lop_overwrites_input(addr, *dst_reg),
        reg_rtl(addr, *arg, arg_rtl);

    op_arg_mapping(addr, pos, arg_rtl_use) <--
        ltl_inst(addr, ?LTLInst::Lop(_, mregs, dst_reg)),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg == dst_reg,
        !osel_cond_pos(addr, pos, _),
        lop_overwrite_use_id(addr, *dst_reg, src_xtl),
        xtl_canonical(src_xtl, arg_rtl_use);


    op_args_collected(addr, args) <--
        ltl_inst(addr, ?LTLInst::Lop(_, mregs, _)),
        if !mregs.is_empty(),
        agg args = build_call_args(pos, reg) in op_arg_mapping(addr, pos, reg);

    op_args_collected(addr, Arc::new(vec![])) <--
        ltl_inst(addr, ?LTLInst::Lop(_, mregs, _)),
        if !mregs.is_empty(),
        !op_arg_mapping(addr, _, _);


    // Detect SP-indexed loads (2 mregs where mregs[0] is SP)
    sp_indexed_load(addr) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::SP,
        indexed_stack_operand(addr, Mreg::SP, _, _),
        rsp_frame_at(addr, _);

    // Detect SP-indexed stores (2 mregs where mregs[0] is SP)
    sp_indexed_store(addr) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::SP,
        indexed_stack_operand(addr, Mreg::SP, _, _),
        rsp_frame_at(addr, _);

    // Detect BP-frame-pointer-indexed loads (2 mregs where mregs[0] is BP) in a framed function.
    bp_indexed_load(addr) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::BP,
        real_addr_in_func(addr, func_start),
        indexed_stack_operand(addr, Mreg::BP, _, _),
        bp_base_at(func_start, addr, _);

    // Detect BP-frame-pointer-indexed stores (2 mregs where mregs[0] is BP) in a framed function.
    bp_indexed_store(addr) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::BP,
        real_addr_in_func(addr, func_start),
        indexed_stack_operand(addr, Mreg::BP, _, _),
        bp_base_at(func_start, addr, _);

    // Address-arg position whose register is NOT the load's destination: its value is the register's ordinary canonical at this node.
    load_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg != dst_reg,
        reg_rtl(addr, *arg, arg_rtl);

    // Address-arg position whose register IS reused as the load's destination:
    // use the canonical value reaching the read side.  The node-local
    // load-overwrite shadow is only an SSA separator; treating it as the
    // address loses a real prior definition such as `mov rdx, [home]; mov
    // rdx, [rdx]` before a call.
    load_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg == dst_reg,
        reaching_use_rtl(addr, *dst_reg, arg_rtl);

    load_args_collected(addr, args) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, _)),
        if mregs.len() > 1,
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        agg args = build_call_args(pos, reg) in load_arg_mapping(addr, pos, reg);

    load_args_collected(addr, Arc::new(vec![])) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, _)),
        if mregs.len() > 1,
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        !load_arg_mapping(addr, _, _);


    store_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        !sp_indexed_store(addr),
        !bp_indexed_store(addr),
        for (pos, arg) in mregs.iter().enumerate(),
        reg_rtl(addr, *arg, arg_rtl);

    store_args_collected(addr, args) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if !mregs.is_empty(),
        !sp_indexed_store(addr),
        !bp_indexed_store(addr),
        agg args = build_call_args(pos, reg) in store_arg_mapping(addr, pos, reg);

    store_args_collected(addr, Arc::new(vec![])) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if !mregs.is_empty(),
        !sp_indexed_store(addr),
        !bp_indexed_store(addr),
        !store_arg_mapping(addr, _, _);


    cond_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lcond(_, mregs, _, _)),
        for (pos, arg) in mregs.iter().enumerate(),
        reg_rtl(addr, *arg, arg_rtl);

    cond_args_collected(addr, args) <--
        ltl_inst(addr, ?LTLInst::Lcond(_, mregs, _, _)),
        if !mregs.is_empty(),
        agg args = build_call_args(pos, reg) in cond_arg_mapping(addr, pos, reg);

    cond_args_collected(addr, Arc::new(vec![])) <--
        ltl_inst(addr, ?LTLInst::Lcond(_, mregs, _, _)),
        if !mregs.is_empty(),
        !cond_arg_mapping(addr, _, _);

    call_args_collected_candidate(call_addr, args) <--
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        agg args = build_call_args(pos, reg) in call_arg_mapping(call_addr, pos, reg);

    call_args_collected_candidate(call_addr, Arc::new(vec![])) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        !call_arg_mapping(call_addr, _, _);

    call_args_collected_candidate(call_addr, Arc::new(vec![])) <--
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        !call_arg_mapping(call_addr, _, _);


    ident_to_symbol(ident, *name) <-- base_ident_to_symbol(ident, name);

    base_ident_to_symbol(ident, *name) <--
        symbols(addr, name, _),
        !plt_block(addr, _),
        !plt_entry(addr, _),
        let ident = *addr as Ident;

    base_ident_to_symbol(ident, clean_name) <--
        plt_entry(addr, sym),
        let ident = *addr as Ident,
        let clean_name = strip_version_suffix(sym);

    base_ident_to_symbol(ident, clean_name) <--
        plt_block(addr, sym),
        let ident = *addr as Ident,
        let clean_name = strip_version_suffix(sym);

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lop(Operation::Oindirectsymbol(ident), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lload(_, Addressing::Aglobal(ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lstore(_, Addressing::Aglobal(ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lload(_, Addressing::Abased(ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lstore(_, Addressing::Abased(ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lload(_, Addressing::Abasedscaled(_, ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lstore(_, Addressing::Abasedscaled(_, ident, _), _, _));

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Lcall(Either::Right(Either::Right(name_sym)))),
        ident_to_symbol(ident, name),
        if name == name_sym;

    global_var_ref(ident) <--
        ltl_inst(_, ?LTLInst::Ltailcall(Either::Right(Either::Right(name_sym)))),
        ident_to_symbol(ident, name),
        if name == name_sym;

    // Track memory chunk type used when loading from each global (for rodata constant inlining)
    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lload(chunk, Addressing::Aglobal(ident, _), _, _));

    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lload(chunk, Addressing::Abased(ident, _), _, _));

    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lload(chunk, Addressing::Abasedscaled(_, ident, _), _, _));

    // Store chunks via direct addressing (also inform global type inference)
    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lstore(chunk, Addressing::Aglobal(ident, _), _, _));

    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lstore(chunk, Addressing::Abased(ident, _), _, _));

    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(_, ?LTLInst::Lstore(chunk, Addressing::Abasedscaled(_, ident, _), _, _));

    // Track RTL register holding address of a global (from Oindirectsymbol, used in PIC binaries)
    global_addr_reg(*ident, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(ident), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    // Propagate global address register through aliases
    global_addr_reg(*ident, b) <-- global_addr_reg(ident, a), alias_edge(a, b);
    global_addr_reg(*ident, a) <-- global_addr_reg(ident, b), alias_edge(a, b);

    // Drive off the load instruction and match global_addr_reg BY the bound base_rtl; leading with global_addr_reg scanned ltl_inst as an O(n*m) cross-product (26s). Clause order is semantics-neutral.
    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, Addressing::Aindexed(0), args, _)),
        if !args.is_empty(),
        reg_rtl(node, args[0], base_rtl),
        global_addr_reg(ident, base_rtl);

    // When a global address register is used as base for a store at offset 0, track the chunk
    global_load_chunk(*ident, chunk.clone()) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, Addressing::Aindexed(0), args, _)),
        if !args.is_empty(),
        reg_rtl(node, args[0], base_rtl),
        global_addr_reg(ident, base_rtl);

    // Note: emit_global_is_ptr/emit_global_is_char_ptr are computed in type_pass.rs (more precise).

    is_ptr(reg) <--
        op_produces_ptr(_, reg);

    is_not_ptr(reg) <--
        op_produces_data(_, reg);


    comparison_operand(addr, cond.clone(), rtl_arg) <--
        ltl_inst(addr, ?LTLInst::Lcond(cond, mregs, _, _)),
        for mreg in mregs.iter(),
        reg_rtl(addr, *mreg, rtl_arg);


    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lop(_, mregs, _)),
        for mreg in mregs.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        !lop_overwrites_input(head_addr, mreg),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lop(_, mregs, _)),
        for mreg in mregs.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        !lop_overwrites_input(head_addr, mreg),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(head_addr, mreg, id) <--
        ltl_inst(head_addr, ?LTLInst::Lbuiltin(_, args, _)),
        for builtin_arg in args,
        if let BuiltinArg::BA(mreg) = builtin_arg,
        reg_def_used(defaddr, *mreg, *head_addr),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, id);

    reg_xtl(head_addr, mreg, id) <--
        ltl_inst(head_addr, ?LTLInst::Lbuiltin(_, args, _)),
        for builtin_arg in args,
        if let BuiltinArg::BA(mreg) = builtin_arg,
        reg_def_used(defaddr, *mreg, *head_addr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, id),
        reg_xtl(defaddr, *mreg, id);

    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lload(_, _, mregs, _)),
        for mreg in mregs.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        !load_overwrites_base(head_addr, mreg),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lload(_, _, mregs, _)),
        for mreg in mregs.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        !load_overwrites_base(head_addr, mreg),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(addr, Mreg::x86(reg_str.to_string()), arg_id) <--
        pcmp(addr, sym1, sym2),
        op_register(sym1, reg_str),
        reg_def_used(addr, Mreg::x86(reg_str.to_string()), *addr),
        reg_xtl(*addr, Mreg::x86(reg_str.to_string()), arg_id);

    reg_xtl(addr, Mreg::x86(reg_str.to_string()), arg_id) <--
        pcmp(addr, sym1, sym2),
        op_register(sym2, reg_str),
        reg_def_used(addr, Mreg::x86(reg_str.to_string()), *addr),
        reg_xtl(*addr, Mreg::x86(reg_str.to_string()), arg_id);

    reg_xtl(head_addr, *src, srcid) <--
        ltl_inst(head_addr, ?LTLInst::Lstore(mc, adrs, args, src)),
        reg_def_used(defaddr, *src, *head_addr),
        !load_overwrites_base(defaddr, src),
        reg_xtl(defaddr, *src, srcid);

    reg_xtl(head_addr, *src, srcid) <--
        ltl_inst(head_addr, ?LTLInst::Lstore(mc, adrs, args, src)),
        reg_def_used(defaddr, *src, *head_addr),
        load_overwrites_base(defaddr, src),
        is_def(defaddr, srcid),
        reg_xtl(defaddr, *src, srcid);

    reg_xtl(head_addr, *src, srcid) <--
        ltl_inst(head_addr, ?LTLInst::Lsetstack(src, _, _, _)),
        reg_def_used(defaddr, *src, *head_addr),
        !load_overwrites_base(defaddr, src),
        reg_xtl(defaddr, *src, srcid);

    reg_xtl(head_addr, *src, srcid) <--
        ltl_inst(head_addr, ?LTLInst::Lsetstack(src, _, _, _)),
        reg_def_used(defaddr, *src, *head_addr),
        load_overwrites_base(defaddr, src),
        is_def(defaddr, srcid),
        reg_xtl(defaddr, *src, srcid);


    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lcond(_, args, _, _)),
        for mreg in args.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(head_addr, mreg, arg_id) <--
        ltl_inst(head_addr, ?LTLInst::Lcond(_, args, _, _)),
        for mreg in args.iter(),
        reg_def_used(defaddr, *mreg, *head_addr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *mreg, arg_id);

    reg_xtl(addr, arg, arg_id) <--
        ltl_inst(addr, ?LTLInst::Ljumptable(arg, _)),
        reg_def_used(defaddr, *arg, *addr),
        !load_overwrites_base(defaddr, arg),
        reg_xtl(defaddr, *arg, arg_id);

    reg_xtl(addr, arg, arg_id) <--
        ltl_inst(addr, ?LTLInst::Ljumptable(arg, _)),
        reg_def_used(defaddr, *arg, *addr),
        load_overwrites_base(defaddr, arg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *arg, arg_id);


    reg_xtl(addr, mreg, callee_id) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(mreg))),
        reg_def_used(defaddr, *mreg, *addr),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, callee_id);

    reg_xtl(addr, mreg, callee_id) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(mreg))),
        reg_def_used(defaddr, *mreg, *addr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, callee_id),
        reg_xtl(defaddr, *mreg, callee_id);

    reg_xtl(addr, mreg, callee_id) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(Either::Left(mreg))),
        reg_def_used(defaddr, *mreg, *addr),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, *mreg, callee_id);

    reg_xtl(addr, mreg, callee_id) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(Either::Left(mreg))),
        reg_def_used(defaddr, *mreg, *addr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, callee_id),
        reg_xtl(defaddr, *mreg, callee_id);


    rtl_inst_candidate(addr, inst) <--
        mach_imm_stack_init(addr, ofs, imm_val, typ),
        real_addr_in_func(addr, func_start),
        stack_var(func_start, addr, ofs, rtl_reg),
        let op = match typ {
            Typ::Tint => Operation::Ointconst(*imm_val),
            Typ::Tlong | Typ::Tany64 => Operation::Olongconst(*imm_val),
            _ => Operation::Ointconst(*imm_val),
        },
        let inst = RTLInst::Iop(op, Arc::new(vec![]), *rtl_reg);

    rtl_inst_candidate(addr, inst) <--
        mach_imm_stack_init(addr, ofs, imm_val, typ),
        real_addr_in_func(addr, func_start),
        !stack_xtl(func_start, addr, ofs, _),
        let rtl_reg = fresh_stack_cell_reg(*addr),
        let op = match typ {
            Typ::Tint => Operation::Ointconst(*imm_val),
            Typ::Tlong | Typ::Tany64 => Operation::Olongconst(*imm_val),
            _ => Operation::Ointconst(*imm_val),
        },
        let inst = RTLInst::Iop(op, Arc::new(vec![]), rtl_reg);

    reg_xtl(addr, base_mreg, arg_id) <--
        mach_imm_indirect_store(addr, _, _, base_mreg, _),
        reg_def_used(defaddr, *base_mreg, *addr),
        !load_overwrites_base(defaddr, base_mreg),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(addr, base_mreg, arg_id) <--
        mach_imm_indirect_store(addr, _, _, base_mreg, _),
        reg_def_used(defaddr, *base_mreg, *addr),
        load_overwrites_base(defaddr, base_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *base_mreg, arg_id);

    rtl_succ_candidate(src, dst) <-- rtl_next(src, dst), !rtl_edge_negated(src, dst);

    // Classify one raw CMP memory operand as generic pointer memory. Narrow
    // EBP/ESP spellings, indexed forms, and scalar RBP/RSP accesses lacking a
    // use-specific stack proof all stay on this route instead of being dropped
    // between the old stack and non-stack shortcuts.
    #[local] relation cmp_mem_operand(Node, Symbol, &'static str, &'static str);
    cmp_mem_operand(addr, *sym, *base_str, *idx_str) <--
        pcmp(addr, _, sym),
        op_indirect(sym, _, base_str, idx_str, _, _, _);
    cmp_mem_operand(addr, *sym, *base_str, *idx_str) <--
        pcmp(addr, sym, _),
        op_indirect(sym, _, base_str, idx_str, _, _, _);

    #[local] relation cmp_generic_mem_operand(Node, Symbol);
    cmp_generic_mem_operand(addr, *sym) <--
        cmp_mem_operand(addr, sym, base_str, _),
        if *base_str != "RBP" && *base_str != "RSP";
    cmp_generic_mem_operand(addr, *sym) <--
        cmp_mem_operand(addr, sym, _, idx_str),
        if *idx_str != "NONE" && !idx_str.is_empty();
    cmp_generic_mem_operand(addr, *sym) <--
        cmp_mem_operand(addr, sym, base_str, idx_str),
        if *base_str == "RBP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        real_addr_in_func(addr, func_start),
        !bp_base_at(func_start, addr, _);
    cmp_generic_mem_operand(addr, *sym) <--
        cmp_mem_operand(addr, sym, base_str, idx_str),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        real_addr_in_func(addr, func_start),
        !sp_entry_ofs(func_start, addr, _);

    // Generic-memory cmps fold into an Icond at the jcc address (not the cmp),
    // so the cmp -> jcc edge must remain to reach the Icond.
    relation cmp_has_non_stack_mem(Address);
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, _, sym),
        cmp_generic_mem_operand(addr, sym);
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, sym, _),
        cmp_generic_mem_operand(addr, sym);

    // cmp+jcc fold: when fused into one Icond AT the cmp addr (BP-relative mem / Lcond->Icond), kill the cmp->jcc edge and replace with edges to Icond's true target and fallthrough; else liveness misses cross-cmp use and DSE nops the loop. Must NOT fire for non-stack-mem cmps (Icond at jcc_addr).
    rtl_edge_negated(addr, *jcc_addr) <--
        pcmp(addr, _, _),
        next(addr, jcc_addr),
        pjcc(jcc_addr, _, _),
        !cmp_has_non_stack_mem(*addr);
    rtl_succ_candidate(*addr, *target_addr) <--
        pcmp(addr, _, _),
        next(addr, jcc_addr),
        pjcc(jcc_addr, _, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        !cmp_has_non_stack_mem(*addr);
    rtl_succ_candidate(*addr, *fallthrough) <--
        pcmp(addr, _, _),
        next(addr, jcc_addr),
        pjcc(jcc_addr, _, _),
        next(jcc_addr, fallthrough),
        !cmp_has_non_stack_mem(*addr);

    // Dual of the cmp+jcc fold for the non-stack-mem case: rebuild the fused Icond's CFG edges from its embedded targets, or it becomes an island and DSE deletes the live def feeding its use.
    #[local] relation icond_at_jcc(Node, Node, Node);
    icond_at_jcc(*jcc_addr, *tgt, *fth) <--
        pjcc(jcc_addr, _, _),
        rtl_inst_candidate(jcc_addr, ?RTLInst::Icond(_, _, Either::Right(tgt), Either::Right(fth)));
    rtl_succ_candidate(*jcc_addr, *tgt) <--
        icond_at_jcc(jcc_addr, tgt, _),
        !rtl_edge_negated(*jcc_addr, *tgt);
    rtl_succ_candidate(*jcc_addr, *fth) <--
        icond_at_jcc(jcc_addr, _, fth),
        !rtl_edge_negated(*jcc_addr, *fth);
    rtl_succ_candidate(*cmp_addr, *jcc_addr) <--
        icond_at_jcc(jcc_addr, _, _),
        next(cmp_addr, jcc_addr),
        !rtl_edge_negated(*cmp_addr, *jcc_addr);

    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lop(_, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lload(_, _, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lstore(_, _, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lcall(_));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Ltailcall(_));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lcond(_, _, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lbuiltin(_, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lsetstack(_, _, _, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Ljumptable(_, _));
    has_ltl_op(addr) <-- ltl_inst(addr, ?LTLInst::Lreturn);

    rtl_inst_candidate(addr, iop_inst) <--
        mach_imm_indirect_store(addr, imm_val, mc, _base_mreg, _disp),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let op = match mc {
            MemoryChunk::MInt64 => Operation::Olongconst(*imm_val),
            _ => Operation::Ointconst(*imm_val),
        },
        let iop_inst = RTLInst::Iop(op, Arc::new(vec![]), fresh_reg);

    // An immediate store carries an index register that mach_imm_indirect_store drops, so detect it from the raw asm operands; this relation gates the base-only Istore off and routes to the indexed path.
    #[local] relation imm_indirect_store_indexed(Address, i64, MemoryChunk, &'static str, &'static str, i64, i64);
    imm_indirect_store_indexed(*addr, imm_int, mc, base_str, idx_str, *scale, *disp) <--
        mach_imm_indirect_store(addr, _, _, _, _),
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, base_str, idx_str, scale, disp, sz),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if !is_rip(base_str),
        let base_mreg = Mreg::x86(*base_str),
        // Exact RSP stays excluded, but address-size-overridden ESP is an
        // ordinary pointer base and must retain the generic indexed route.
        if *base_str != "RSP",
        let mc = match *sz {
            1 => MemoryChunk::MInt8Unsigned,
            2 => MemoryChunk::MInt16Unsigned,
            8 => MemoryChunk::MInt64,
            _ => MemoryChunk::MInt32,
        },
        let imm_int = *imm_sym as i64;

    #[local] relation imm_indirect_store_has_index(Address);
    imm_indirect_store_has_index(*addr) <-- imm_indirect_store_indexed(addr, _, _, _, _, _, _);

    // BP-frame-pointer indexed immediate store, replaced below by the Olea(Ainstack(disp))+inner-0 expansion; its membership check uses the stable block_in_function inputs, not the growing instr_in_function.
    #[local] relation real_addr_in_func(Address, Address);
    real_addr_in_func(addr, func) <--
        block_in_function(block, func),
        code_in_block(addr, block);
    #[local] relation imm_indirect_store_bp_indexed(Address, i64, MemoryChunk, &'static str, i64, i64);
    imm_indirect_store_bp_indexed(*addr, *imm_val, mc.clone(), *idx_str, *scale, *disp) <--
        imm_indirect_store_indexed(addr, imm_val, mc, base_str, idx_str, scale, disp),
        if *base_str == "RBP",
        real_addr_in_func(addr, func_start),
        bp_base_at(func_start, *addr, _);

    // Base-only Istore: only when there is no index to preserve (indexed case handled below).
    rtl_inst_candidate(synthetic_addr, istore_inst) <--
        mach_imm_indirect_store(addr, _imm_val, mc, base_mreg, disp),
        !imm_indirect_store_has_index(*addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let synthetic_addr = *addr | (1u64 << 62),
        let addressing = Addressing::Aindexed(*disp),
        let istore_inst = RTLInst::Istore(mc.clone(), addressing, Arc::new(vec![*base_rtl]), fresh_reg);

    // Indexed immediate store mirroring the indexed-load and SP-indexed-store address shapes so the + idx*scale addend survives; the BP-frame-pointer base is handled by the Olea(Ainstack) chain below.
    rtl_inst_candidate(synthetic_addr, istore_inst) <--
        imm_indirect_store_indexed(addr, _imm_val, mc, base_str, idx_str, scale, disp),
        !imm_indirect_store_bp_indexed(*addr, _, _, _, _, _),
        let base_mreg = Mreg::x86(*base_str),
        let idx_mreg = Mreg::x86(*idx_str),
        reg_rtl(addr, base_mreg, base_rtl),
        reg_rtl(addr, idx_mreg, idx_rtl),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let synthetic_addr = *addr | (1u64 << 62),
        let addressing = if *scale > 1 {
            Addressing::Aindexed2scaled(*scale, *disp)
        } else {
            Addressing::Aindexed2(*disp)
        },
        let istore_inst = RTLInst::Istore(mc.clone(), addressing, Arc::new(vec![*base_rtl, *idx_rtl]), fresh_reg);

    // BP-frame-pointer indexed immediate store as a 3-node chain (const, Olea(Ainstack(disp)), indexed Istore), with indexed_synth_stack_base resolving the base to the array's named local.
    rtl_inst_candidate(synth1, lea_inst), op_produces_ptr(synth1, bp_addr_rtl),
    indexed_synth_stack_base(func_start, synth1, *addr, Mreg::BP, *disp, bp_addr_rtl) <--
        imm_indirect_store_bp_indexed(addr, _imm_val, _mc, _idx_str, _scale, disp),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let bp_addr_rtl = fresh_xtl_reg(synth1, Mreg::BP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*disp)), Arc::new(vec![]), bp_addr_rtl);

    rtl_inst_candidate(synth2, istore_inst) <--
        imm_indirect_store_bp_indexed(addr, _imm_val, mc, idx_str, scale, _disp),
        let idx_mreg = Mreg::x86(*idx_str),
        reg_rtl(addr, idx_mreg, idx_rtl),
        let synth1 = *addr | (1u64 << 62),
        let bp_addr_rtl = fresh_xtl_reg(synth1, Mreg::BP) | FRESH_NS_SP_BASE,
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let synth2 = *addr | (1u64 << 63),
        let addressing = if *scale > 1 {
            Addressing::Aindexed2scaled(*scale, 0)
        } else {
            Addressing::Aindexed2(0)
        },
        let istore_inst = RTLInst::Istore(mc.clone(), addressing, Arc::new(vec![bp_addr_rtl, *idx_rtl]), fresh_reg);

    // Thread the index register's value web so the index operand chains to its def site instead of staying a fresh unbound register.
    reg_xtl(addr, idx_mreg, arg_id) <--
        imm_indirect_store_indexed(addr, _, _, _, idx_str, _, _),
        let idx_mreg = Mreg::x86(*idx_str),
        reg_def_used(defaddr, idx_mreg, *addr),
        !load_overwrites_base(defaddr, &idx_mreg),
        reg_xtl(defaddr, idx_mreg, arg_id);

    rtl_edge_negated(addr, next) <--
        mach_imm_indirect_store(addr, _, _, _, _),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        mach_imm_indirect_store(addr, _, _, _, _),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    // synth1 -> next for the non-BP-indexed case; the BP-frame-pointer indexed case inserts synth2 (the real Istore) between synth1 and next instead.
    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        mach_imm_indirect_store(addr, _, _, _, _),
        !imm_indirect_store_bp_indexed(*addr, _, _, _, _, _),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    // BP-frame-pointer indexed immediate store: chain synth1 (Olea) -> synth2 (Istore) -> next.
    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start),
    instr_in_function(synth2, func_start) <--
        imm_indirect_store_bp_indexed(addr, _, _, _, _, _),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        imm_indirect_store_bp_indexed(addr, _, _, _, _, _),
        instr_in_function(addr, func_start),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    // 3.8 R3: a RIP-relative immediate global store has no lifting route, so detect it from asm operand rows plus RIP resolution and lower ident-keyed as Ointconst then Istore(Aglobal).
    #[local] relation mach_imm_global_store(Address, i64, MemoryChunk, Ident, i64);
    mach_imm_global_store(*addr, imm_int, mc, *ident, *ofs) <--
        pmov(addr, dst, src),
        op_immediate(src, imm_sym, _),
        op_indirect(dst, _, base_str, idx_str, _, _, sz),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_rip(base_str),
        rip_target_addr(addr, target_addr),
        resolved_addr_to_symbol(target_addr, ident, ofs),
        let imm_int = *imm_sym as i64,
        let mc = match *sz {
            1 => MemoryChunk::MInt8Unsigned,
            2 => MemoryChunk::MInt16Unsigned,
            8 => MemoryChunk::MInt64,
            _ => MemoryChunk::MInt32,
        };

    rtl_inst_candidate(addr, iop_inst) <--
        mach_imm_global_store(addr, imm_val, mc, _ident, _ofs),
        !has_ltl_op(addr),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let op = if matches!(mc, MemoryChunk::MInt64) {
            Operation::Olongconst(*imm_val)
        } else {
            Operation::Ointconst(*imm_val)
        },
        let iop_inst = RTLInst::Iop(op, Arc::new(vec![]), fresh_reg);

    rtl_inst_candidate(synthetic_addr, istore_inst) <--
        mach_imm_global_store(addr, _imm_val, mc, ident, ofs),
        !has_ltl_op(addr),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::DI),
        let synthetic_addr = *addr | (1u64 << 62),
        let istore_inst = RTLInst::Istore(*mc, Addressing::Aglobal(*ident, *ofs), Arc::new(vec![]), fresh_reg);

    rtl_edge_negated(addr, next) <--
        mach_imm_global_store(addr, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        mach_imm_global_store(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        mach_imm_global_store(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    synth_only_addr(addr) <--
        mach_imm_global_store(addr, _, _, _, _),
        !has_ltl_op(addr);

    // reg_xtl for arith addresses chains from the reg_def_used def site when no real ltl op exists; for loads overwriting their base propagate only def_id, or use_id leaks and DSE wipes the load chain.
    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_load_op(addr, _, _, base_mreg, _, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        !load_overwrites_base(defaddr, base_mreg),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_load_op(addr, _, _, base_mreg, _, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        load_overwrites_base(defaddr, base_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *dst_mreg, arg_id) <--
        arith_load_op(addr, _, _, _, _, dst_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        !load_overwrites_base(defaddr, dst_mreg),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    reg_xtl(*addr, *dst_mreg, arg_id) <--
        arith_load_op(addr, _, _, _, _, dst_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        load_overwrites_base(defaddr, dst_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_store_reg(addr, _, _, base_mreg, _, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        !load_overwrites_base(defaddr, base_mreg),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_store_reg(addr, _, _, base_mreg, _, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        load_overwrites_base(defaddr, base_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *src_mreg, arg_id) <--
        arith_store_reg(addr, _, _, _, _, src_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *src_mreg, *addr),
        !load_overwrites_base(defaddr, src_mreg),
        reg_xtl(defaddr, *src_mreg, arg_id);

    reg_xtl(*addr, *src_mreg, arg_id) <--
        arith_store_reg(addr, _, _, _, _, src_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *src_mreg, *addr),
        load_overwrites_base(defaddr, src_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *src_mreg, arg_id);

    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_store_imm(addr, _, _, base_mreg, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        !load_overwrites_base(defaddr, base_mreg),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *base_mreg, arg_id) <--
        arith_store_imm(addr, _, _, base_mreg, _),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *base_mreg, *addr),
        load_overwrites_base(defaddr, base_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *base_mreg, arg_id);

    reg_xtl(*addr, *src_mreg, arg_id) <--
        arith_store_abs_reg(addr, _, _, _, _, src_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *src_mreg, *addr),
        !load_overwrites_base(defaddr, src_mreg),
        reg_xtl(defaddr, *src_mreg, arg_id);

    reg_xtl(*addr, *src_mreg, arg_id) <--
        arith_store_abs_reg(addr, _, _, _, _, src_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *src_mreg, *addr),
        load_overwrites_base(defaddr, src_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *src_mreg, arg_id);

    // 3.8 R1: reg_xtl chains for FUSED float ops, mirroring the arith chains; without them the addressing regs stay fresh and a param-fed addsd reads an unbound register.
    reg_xtl(*addr, *m, arg_id) <--
        float_load_arg(addr, _, m),
        reg_def_used(defaddr, *m, *addr),
        !load_overwrites_base(defaddr, m),
        reg_xtl(defaddr, *m, arg_id);

    reg_xtl(*addr, *m, arg_id) <--
        float_load_arg(addr, _, m),
        reg_def_used(defaddr, *m, *addr),
        load_overwrites_base(defaddr, m),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *m, arg_id);

    // Binary (RMW) form only: dst is read+written (`dst = dst OP loaded`); the unary conversion form writes a fresh dst and must not chain prior values in.
    reg_xtl(*addr, *dst_mreg, arg_id) <--
        float_load_op(addr, _, _, _, _, dst_mreg, false),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        !load_overwrites_base(defaddr, dst_mreg),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    reg_xtl(*addr, *dst_mreg, arg_id) <--
        float_load_op(addr, _, _, _, _, dst_mreg, false),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        load_overwrites_base(defaddr, dst_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    // A proved immutable /homeparams arithmetic read consumes the incoming
    // register value directly.  Keep the real node as the CFG anchor and put
    // the binary operation at the standard synthetic successor.
    #[local] relation arith_load_uses_home_param(Node);
    arith_load_uses_home_param(addr) <--
        win64_home_arith_read(addr, _, _, _);

    rtl_inst_candidate(addr, nop) <--
        arith_load_uses_home_param(addr),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, _base_mreg, _disp, dst_mreg),
        win64_home_arith_read(addr, func_start, param_mreg, _),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let param_rtl = fresh_xtl_reg(*func_start, *param_mreg),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, param_rtl]), *dst_rtl);

    // arith_load_op on a THREADED spilled slot reads the slot's canonical SSA reg directly: a real frame Iload would re-materialize the slot as *(&var_0 + k) garbage, so the load node carries an Inop.
    rtl_inst_candidate(addr, nop) <--
        arith_load_op(addr, _op, _chunk, base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        stack_claim_base_at(func_start, addr, *base_mreg),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        stack_claim_base_at(func_start, addr, *base_mreg),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, *stack_rtl]), *dst_rtl);

    // A concrete read of an incoming stack parameter and its ABI slot ordinal
    // (zero is the first stack-passed argument). Keeping the access address in
    // the key avoids compacting sparse parameters or conflating raw SP
    // displacements observed at different prologue depths.
    relation stack_param_access(Node, Address, i64, usize);
    relation stack_param_ordinal(Address, usize);

    // ABI-1: memory-source arithmetic reading an INCOMING stack argument binds to the function's synthetic stack-param reg, so the body references pN instead of a fresh-RTEMP fallback.
    #[local] relation arith_load_uses_stack_param(Node);
    arith_load_uses_stack_param(addr) <--
        arith_load_op(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, *base_mreg),
        stack_param_access(addr, func_start, disp, _);

    // Keep the original node as a transparent Inop anchor while the stack-param arithmetic lives at the synthetic successor, or rtl_optimize's liveness cannot carry the RMW destination across it.
    rtl_inst_candidate(addr, nop) <--
        arith_load_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, *base_mreg),
        stack_param_access(addr, func_start, disp, idx),
        !arith_load_uses_stack_var(addr),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, param_reg]), *dst_rtl);

    // TR-5: the load's memory chunk supplies TYPE evidence for stack-passed
    // positions, which previously had no evidence path and defaulted to Xany64.
    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        arith_load_op(addr, _, chunk, base_mreg, disp, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, *base_mreg),
        stack_param_access(addr, func_start, disp, idx),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let xt = match chunk {
            MemoryChunk::MInt32 => XType::Xint,
            MemoryChunk::MInt64 => XType::Xlong,
            MemoryChunk::MFloat64 => XType::Xfloat,
            MemoryChunk::MFloat32 => XType::Xsingle,
            MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
            | MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => XType::Xint,
            _ => XType::Xany64,
        };

    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, Addressing::Aindexed(disp), args, _)),
        for base in args.iter(),
        if *base == Mreg::BP || *base == Mreg::SP,
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let xt = match chunk {
            MemoryChunk::MInt32 => XType::Xint,
            MemoryChunk::MInt64 => XType::Xlong,
            MemoryChunk::MFloat64 => XType::Xfloat,
            MemoryChunk::MFloat32 => XType::Xsingle,
            MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
            | MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => XType::Xint,
            _ => XType::Xany64,
        };

    // Fallback (non-BP base or BP without matching stack_var/stack-param): use a fresh RTEMP, same as before.
    #[local] relation arith_load_uses_stack_var(Node);
    arith_load_uses_stack_var(addr) <--
        arith_load_op(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        stack_claim_base_at(func_start, addr, *base_mreg);

    rtl_inst_candidate(addr, load_inst) <--
        arith_load_op(addr, _op, chunk, base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_load_uses_stack_var(addr),
        !arith_load_uses_stack_param(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, _base_mreg, _disp, dst_mreg),
        !has_ltl_op(addr),
        !arith_load_uses_home_param(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        !arith_load_uses_stack_var(addr),
        !arith_load_uses_stack_param(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, temp]), *dst_rtl);

    rtl_edge_negated(addr, next) <--
        arith_load_op(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        arith_load_op(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        arith_load_op(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    #[local] relation stack_unary_uses_stack_param(Node);
    stack_unary_uses_stack_param(addr) <--
        stack_unary_load_op(addr, _, _, disp, _),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, _);

    // stack_unary_load_op: a local slot is already live in stack_rtl; an
    // incoming slot instead consumes the ABI-positioned synthetic parameter.
    // In both cases the real node remains an Inop anchor for the synth edge.
    rtl_inst_candidate(addr, nop) <--
        stack_unary_load_op(addr, _op, _base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        !stack_unary_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(addr, nop) <--
        stack_unary_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    // Unary op at synth: dst = op(slot). The destination is write-only, so
    // handle both an existing canonical SSA reg and the fresh case.
    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        !stack_unary_uses_stack_param(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), *dst_rtl);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        !stack_unary_uses_stack_param(addr),
        !reg_xtl(addr, *dst_mreg, _),
        let synthetic_addr = *addr | (1u64 << 62),
        let dst_rtl = fresh_xtl_reg(*addr, *dst_mreg),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), dst_rtl);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![param_reg]), *dst_rtl);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        !reg_xtl(addr, *dst_mreg, _),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let dst_rtl = fresh_xtl_reg(*addr, *dst_mreg),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![param_reg]), dst_rtl);

    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, _dst_mreg),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let xt = match op {
            Operation::Ofloatoflong | Operation::Osingleoflong | Operation::Omullimm(_) => XType::Xlong,
            _ => XType::Xint,
        };

    rtl_edge_negated(addr, next) <--
        stack_unary_load_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        stack_unary_load_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        stack_unary_load_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    synth_only_addr(addr) <-- stack_unary_load_op(addr, _, _, _, _), !has_ltl_op(addr);

    // float_arith_stack_op: the slot value is live in stack_rtl, so the binary op at the synthetic address consumes it as the second operand; the read+written dst threads its incoming value in via reg_xtl.
    reg_xtl(*addr, *dst_mreg, arg_id) <--
        float_arith_stack_op(addr, _, _, _, dst_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        !load_overwrites_base(defaddr, dst_mreg),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    reg_xtl(*addr, *dst_mreg, arg_id) <--
        float_arith_stack_op(addr, _, _, _, dst_mreg),
        !has_ltl_op(addr),
        reg_def_used(defaddr, *dst_mreg, *addr),
        load_overwrites_base(defaddr, dst_mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, *dst_mreg, arg_id);

    // A fused float RMW can read an incoming stack parameter like the integer path; kept distinct from a callee-owned spill, which binds to stack_var's local value web instead.
    #[local] relation float_arith_uses_stack_param(Node);
    float_arith_uses_stack_param(addr) <--
        float_arith_stack_op(addr, _, _, disp, _),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, _),
        !stack_var(func_start, addr, disp, _);

    rtl_inst_candidate(addr, nop) <--
        float_arith_stack_op(addr, _op, _base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        !float_arith_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    // Binary op at synth: dst = dst OP slot, where dst is read+written and already has a canonical SSA reg threaded in by the reg_xtl chain above.
    rtl_inst_candidate(synthetic_addr, op_inst) <--
        float_arith_stack_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        !float_arith_uses_stack_param(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, *stack_rtl]), *dst_rtl);

    // Incoming stack-param form: retain the real address as an Inop anchor so optimizer liveness crosses it, then consume the parameter at the same synthetic successor.
    rtl_inst_candidate(addr, nop) <--
        float_arith_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        float_arith_stack_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        !stack_var(func_start, addr, disp, _),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, param_reg]), *dst_rtl);

    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        float_arith_stack_op(addr, op, _base_mreg, disp, _dst_mreg),
        instr_in_function(addr, func_start),
        stack_param_access(addr, func_start, disp, idx),
        !stack_var(func_start, addr, disp, _),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let xt = if crate::x86::types::is_single_operation(op) {
            XType::Xsingle
        } else {
            XType::Xfloat
        };

    rtl_edge_negated(addr, next) <--
        float_arith_stack_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        float_arith_stack_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        float_arith_stack_op(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    synth_only_addr(addr) <-- float_arith_stack_op(addr, _, _, _, _), !has_ltl_op(addr);

    // An indexed fused memory operation rooted at a proved RSP coordinate
    // needs one more node than the generic float_load_op expansion: materialize
    // the stack base, load through base+index*scale, then apply the operation.
    // Integer ADD/SUB/AND/OR/XOR use float_load_op too because that relation is
    // the existing operation-agnostic carrier for a full Addressing value.
    // Retained so the post-fixed-point selector can distinguish one coherent
    // lowering from zero or multiple reaching-value combinations.
    relation sp_indexed_fused_load(Node, Operation, MemoryChunk, i64, i64, Mreg, Mreg);
    sp_indexed_fused_load(*addr, op.clone(), *chunk, *scale, *disp, args[1], *dst) <--
        float_load_op(addr, op, chunk, addressing, args, dst, false),
        if args.len() == 2 && args[0] == Mreg::SP,
        if let Addressing::Aindexed2scaled(scale, disp) = addressing,
        indexed_stack_operand(addr, Mreg::SP, raw_disp, _),
        if *raw_disp == *disp;
    sp_indexed_fused_load(*addr, op.clone(), *chunk, 1, *disp, args[1], *dst) <--
        float_load_op(addr, op, chunk, addressing, args, dst, false),
        if args.len() == 2 && args[0] == Mreg::SP,
        if let Addressing::Aindexed2(disp) = addressing,
        indexed_stack_operand(addr, Mreg::SP, raw_disp, _),
        if *raw_disp == *disp;

    // A unique dominating full-stride write is concrete aggregate-phase
    // evidence.  Reads, later instructions, and writes on another branch must
    // not manufacture a stack-object base for an indexed field access.
    // Restrict the dominance workspace to the exact write/access shapes that
    // consume it below.  The previous rules first emitted every ordered
    // instruction pair in a block and every pair across dominated blocks,
    // which is quadratic in a long function even when it has no fused stack
    // access at all.
    #[local] relation sp_indexed_anchor_write_dominates(Address, Address, Address);
    sp_indexed_anchor_write_dominates(*other, *addr, *func_start) <--
        sp_indexed_fused_load(addr, _, _, _, _, _, _),
        direct_stack_operand(other, Mreg::SP, _, _),
        decoded_memory_write_operand(other, _),
        real_addr_in_func(other, func_start),
        real_addr_in_func(addr, func_start),
        code_in_block(other, block),
        code_in_block(addr, block),
        if *other < *addr;
    sp_indexed_anchor_write_dominates(*other, *addr, *func_start) <--
        sp_indexed_fused_load(addr, _, _, _, _, _, _),
        direct_stack_operand(other, Mreg::SP, _, _),
        decoded_memory_write_operand(other, _),
        real_addr_in_func(other, func_start),
        real_addr_in_func(addr, func_start),
        code_in_block(other, other_block),
        code_in_block(addr, addr_block),
        if other_block != addr_block,
        block_dom_set(func_start, addr_block, doms),
        if doms.0.contains(other_block);

    // A source aggregate need not be initialized by one write per record.
    // GCC commonly emits one scalar write per field, while Clang may cover
    // several records with one vector write.  Merge only adjacent/overlapping
    // non-home writes that dominate this fused access; a span covering two or
    // more whole strides gives the same independent phase evidence as the
    // exact full-stride write below without treating a lone scalar local as an
    // aggregate anchor.
    #[local] relation sp_indexed_fused_init_span(Node, Address, i64, i64);
    sp_indexed_fused_init_span(*addr, *func_start, *write_start, *write_end) <--
        sp_indexed_fused_load(addr, _, _, _, _, _, _),
        normalized_stack_write_range(write_node, func_start, _, _, write_start, write_end),
        stack_write_dominates_node(func_start, write_node, addr),
        !win64_home_spill_candidate(write_node, func_start, _, _);
    sp_indexed_fused_init_span(*addr, *func_start, *span_start, *right_end) <--
        sp_indexed_fused_init_span(addr, func_start, span_start, left_end),
        sp_indexed_fused_init_span(addr, func_start, right_start, right_end),
        if *right_start <= *left_end,
        if *right_end > *left_end;

    #[local] relation sp_indexed_fused_anchor_candidate(Node, i64);
    sp_indexed_fused_anchor_candidate(*addr, candidate_disp) <--
        sp_indexed_fused_load(addr, _, chunk, scale, disp, _, _),
        if *scale > 1,
        real_addr_in_func(addr, func_start),
        rsp_frame_at(addr, func_start),
        rsp_frame_offset_at(func_start, addr, sp_offset),
        direct_stack_operand(other, Mreg::SP, other_disp, other_size),
        decoded_memory_write_operand(other, _),
        real_addr_in_func(other, func_start),
        !win64_home_spill_candidate(other, func_start, _, _),
        sp_indexed_anchor_write_dominates(other, addr, func_start),
        rsp_frame_at(other, func_start),
        rsp_frame_offset_at(func_start, other, other_sp_offset),
        if *other_size as i64 == *scale,
        if let Some(target_coord) = sp_offset.checked_add(*disp),
        if let Some(candidate_coord) = other_sp_offset.checked_add(*other_disp),
        if let Some(field_delta) = target_coord.checked_sub(candidate_coord),
        if field_delta > 0,
        if field_delta + (chunk_size_bits(chunk) as i64 / 8) <= *scale,
        if let Some(candidate_disp) = candidate_coord.checked_sub(*sp_offset);
    sp_indexed_fused_anchor_candidate(*addr, candidate_disp) <--
        sp_indexed_fused_load(addr, _, chunk, scale, disp, _, _),
        if *scale > 1,
        real_addr_in_func(addr, func_start),
        rsp_frame_at(addr, func_start),
        rsp_frame_offset_at(func_start, addr, sp_offset),
        sp_indexed_fused_init_span(addr, func_start, span_start, span_end),
        let span_len = *span_end - *span_start,
        if span_len >= 2 * *scale,
        if span_len % *scale == 0,
        if let Some(target_coord) = sp_offset.checked_add(*disp),
        let field_delta = target_coord - *span_start,
        if field_delta > 0,
        if field_delta + (chunk_size_bits(chunk) as i64 / 8) <= *scale,
        if let Some(candidate_disp) = span_start.checked_sub(*sp_offset);

    #[local] relation sp_indexed_fused_anchor_ambiguous(Node);
    sp_indexed_fused_anchor_ambiguous(addr) <--
        sp_indexed_fused_anchor_candidate(addr, first),
        sp_indexed_fused_anchor_candidate(addr, second),
        if first != second;

    #[local] relation sp_indexed_fused_anchor(Node, i64);
    sp_indexed_fused_anchor(addr, *candidate) <--
        sp_indexed_fused_anchor_candidate(addr, candidate),
        !sp_indexed_fused_anchor_ambiguous(addr);

    #[local] relation sp_indexed_fused_stack_addr(Node, Address, i64, i64);
    sp_indexed_fused_stack_addr(*addr, *func_start, *base_disp, field_disp) <--
        sp_indexed_fused_load(addr, _, _, _, disp, _, _),
        real_addr_in_func(addr, func_start),
        rsp_frame_at(addr, func_start),
        rsp_frame_offset_at(func_start, addr, _),
        sp_indexed_fused_anchor(addr, base_disp),
        if let Some(field_disp) = disp.checked_sub(*base_disp);
    sp_indexed_fused_stack_addr(*addr, *func_start, *disp, 0) <--
        sp_indexed_fused_load(addr, _, _, _, disp, _, _),
        real_addr_in_func(addr, func_start),
        rsp_frame_at(addr, func_start),
        rsp_frame_offset_at(func_start, addr, _),
        !sp_indexed_fused_anchor(addr, _);

    // One witness owns the entire three-node expansion. Incomplete or
    // rejected sites retain their natural fallthrough and cannot leave a load
    // or arithmetic node consuming an undefined synthetic temporary.
    #[local] relation sp_indexed_fused_lowering(Node, Address, Operation, MemoryChunk, i64, i64, i64, Mreg, RTLReg, Mreg, RTLReg);
    sp_indexed_fused_lowering(addr, *func_start, op.clone(), *chunk, *scale, *base_disp, *field_disp, *index, *index_rtl, *dst, *dst_rtl) <--
        sp_indexed_fused_load(addr, op, chunk, scale, _, index, dst),
        sp_indexed_fused_stack_addr(addr, func_start, base_disp, field_disp),
        real_addr_in_func(addr, func_start),
        !has_ltl_op(addr),
        reaching_use_rtl(addr, *index, index_rtl),
        reaching_use_rtl(addr, *dst, dst_rtl),
        !unsupported_stack_address_seed(func_start, addr, _);

    relation sp_indexed_fused_complete(Node);
    sp_indexed_fused_complete(addr) <--
        sp_indexed_fused_lowering(addr, _, _, _, _, _, _, _, _, _, _);

    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, sp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *addr, Mreg::SP, *base_disp, sp_addr_rtl) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, base_disp, _, _, _, _, _),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*base_disp)), Arc::new(vec![]), sp_addr_rtl);

    rtl_inst_candidate(synth1, load_inst) <--
        sp_indexed_fused_lowering(addr, _, _, chunk, scale, _, field_disp, _, index_rtl, _, _),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synth1 = *addr | (1u64 << 62),
        let addressing = if *scale > 1 {
            Addressing::Aindexed2scaled(*scale, *field_disp)
        } else {
            Addressing::Aindexed2(*field_disp)
        },
        let load_inst = RTLInst::Iload(*chunk, addressing, Arc::new(vec![sp_addr_rtl, *index_rtl]), temp);

    rtl_inst_candidate(synth2, op_inst) <--
        sp_indexed_fused_lowering(addr, _, op, _, _, _, _, _, _, _, dst_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synth2 = *addr | (1u64 << 63),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, temp]), *dst_rtl);

    rtl_edge_negated(addr, next) <--
        sp_indexed_fused_lowering(addr, _, _, _, _, _, _, _, _, _, _),
        next(addr, next);

    rtl_succ_candidate(addr, synth1) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        let synth1 = *addr | (1u64 << 62);

    rtl_succ_candidate(synth1, synth2) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    relation sp_indexed_fused_member(Node, Address);
    sp_indexed_fused_member(synth1, *func_start) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        let synth1 = *addr | (1u64 << 62);
    sp_indexed_fused_member(synth2, *func_start) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    // If a non-lowered bookkeeping instruction follows the fused operation,
    // bridge from the actual final node (bit 63).  The generic one-synth tail
    // bridge below must not start at bit 62 and bypass this arithmetic node.
    rtl_succ_candidate(synth2, dst) <--
        sp_indexed_fused_lowering(addr, func_start, _, _, _, _, _, _, _, _, _),
        next(addr, mid),
        !ltl_inst(*mid, _),
        !synth_only_addr(*mid),
        real_addr_in_func(*mid, func_start),
        sp_synth_skip_to_ltl(*mid, *func_start, dst),
        let synth2 = *addr | (1u64 << 63);

    // float_load_op: an SSE op with a memory source produces no mach/ltl op, so lower it as a float load into a fresh temp plus the op at a synthetic address; fires only when nothing else lowered it.
    #[local] relation float_load_arg(Node, usize, Mreg);
    float_load_arg(addr, pos, *m) <--
        float_load_op(addr, _, _, _, args, _, _),
        !has_ltl_op(addr),
        for (pos, m) in args.iter().enumerate();

    #[local] relation float_load_arg_rtl(Node, usize, RTLReg);
    float_load_arg_rtl(addr, pos, rtl) <--
        float_load_arg(addr, pos, m),
        reg_rtl(addr, *m, rtl);

    #[local] relation float_load_args_collected(Node, Arc<Vec<RTLReg>>);
    float_load_args_collected(addr, args) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        agg args = build_call_args(pos, reg) in float_load_arg_rtl(addr, pos, reg);
    float_load_args_collected(addr, Arc::new(vec![])) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        !float_load_arg(addr, _, _);

    rtl_inst_candidate(addr, load_inst) <--
        float_load_op(addr, _, chunk, addressing, _, _, _),
        !has_ltl_op(addr),
        float_load_args_collected(addr, arg_rtls),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, addressing.clone(), arg_rtls.clone(), temp);

    // Binary op (mul/add/sub/div): dst = dst OP loaded; dst is read+written like arith_load_op.
    rtl_inst_candidate(synthetic_addr, op_inst) <--
        float_load_op(addr, op, _, _, _, dst, false),
        !has_ltl_op(addr),
        reg_rtl(addr, *dst, dst_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, temp]), *dst_rtl);

    // Unary conversion (cvtss2sd/cvtsd2ss): dst = convert(loaded); dst is write-only so it may be a fresh def; handle both the existing-canonical and fresh-reg cases.
    rtl_inst_candidate(synthetic_addr, op_inst) <--
        float_load_op(addr, op, _, _, _, dst, true),
        !has_ltl_op(addr),
        reg_rtl(addr, *dst, dst_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp]), *dst_rtl);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        float_load_op(addr, op, _, _, _, dst, true),
        !has_ltl_op(addr),
        !reg_xtl(addr, *dst, _),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let synthetic_addr = *addr | (1u64 << 62),
        let dst_rtl = fresh_xtl_reg(*addr, *dst),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp]), dst_rtl);

    rtl_edge_negated(addr, next) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(addr, func_start) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next), instr_in_function(synthetic_addr, func_start) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next);

    synth_only_addr(addr) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr);
    synth_only_addr(addr) <-- sp_indexed_fused_lowering(addr, _, _, _, _, _, _, _, _, _, _);

    // adc_carry_op expansion: materialize the carry as the 0/1 value of the compare's carry condition at addr, then update the accumulator at synthetic addresses, reading the compared reg at cmp_addr.
    #[local] relation adc_carry_needs_k(Address);
    adc_carry_needs_k(adc_addr) <-- adc_carry_op(adc_addr, _, _, _, k, _, _, _), if *k != 0;

    // addr: t = Ocmp(cf_cond, compared_reg-at-compare). One register arg (immediate baked into cf_cond).
    rtl_inst_candidate(adc_addr, cmp_inst) <--
        adc_carry_op(adc_addr, cmp_addr, cf_cond, _, _, _, _, cmp_mreg),
        reg_rtl(cmp_addr, *cmp_mreg, cmp_rtl),
        let temp = fresh_xtl_reg(*adc_addr, Mreg::x86("RTEMP")),
        let cmp_inst = RTLInst::Iop(Operation::Ocmp(cf_cond.clone()), Arc::new(vec![*cmp_rtl]), temp);

    // synth1: acc = acc (+/-) t  (subtract when sbb, since sbb subtracts the carry).
    rtl_inst_candidate(synth1, op_inst) <--
        adc_carry_op(adc_addr, _, _, is_sub, _, acc_is_64, acc_mreg, _),
        reg_rtl(adc_addr, *acc_mreg, acc_rtl),
        let temp = fresh_xtl_reg(*adc_addr, Mreg::x86("RTEMP")),
        let synth1 = *adc_addr | (1u64 << 62),
        let combine = match (*is_sub, *acc_is_64) {
            (false, false) => Operation::Oadd,
            (false, true)  => Operation::Oaddl,
            (true, false)  => Operation::Osub,
            (true, true)   => Operation::Osubl,
        },
        let op_inst = RTLInst::Iop(combine, Arc::new(vec![*acc_rtl, temp]), *acc_rtl);

    // synth2 (only when a residual constant remains): acc = acc + k.
    rtl_inst_candidate(synth2, op_inst) <--
        adc_carry_op(adc_addr, _, _, _, k, acc_is_64, acc_mreg, _),
        if *k != 0,
        reg_rtl(adc_addr, *acc_mreg, acc_rtl),
        let synth2 = *adc_addr | (1u64 << 63),
        let addk = if *acc_is_64 { Operation::Oaddlimm(*k) } else { Operation::Oaddimm(*k) },
        let op_inst = RTLInst::Iop(addk, Arc::new(vec![*acc_rtl]), *acc_rtl);

    rtl_edge_negated(adc_addr, next) <--
        adc_carry_op(adc_addr, _, _, _, _, _, _, _),
        next(adc_addr, next);

    rtl_succ_candidate(adc_addr, synth1), instr_in_function(adc_addr, func_start) <--
        adc_carry_op(adc_addr, _, _, _, _, _, _, _),
        instr_in_function(adc_addr, func_start),
        let synth1 = *adc_addr | (1u64 << 62);

    // K==0: synth1 -> next directly (two-node chain).
    rtl_succ_candidate(synth1, next), instr_in_function(synth1, func_start) <--
        adc_carry_op(adc_addr, _, _, _, _, _, _, _),
        !adc_carry_needs_k(adc_addr),
        instr_in_function(adc_addr, func_start),
        let synth1 = *adc_addr | (1u64 << 62),
        next(adc_addr, next);

    // K!=0: synth1 -> synth2 -> next (three-node chain).
    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start) <--
        adc_carry_op(adc_addr, _, _, _, _, _, _, _),
        adc_carry_needs_k(adc_addr),
        instr_in_function(adc_addr, func_start),
        let synth1 = *adc_addr | (1u64 << 62),
        let synth2 = *adc_addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        adc_carry_op(adc_addr, _, _, _, _, _, _, _),
        adc_carry_needs_k(adc_addr),
        instr_in_function(adc_addr, func_start),
        let synth2 = *adc_addr | (1u64 << 63),
        next(adc_addr, next);

    synth_only_addr(addr) <-- adc_carry_op(addr, _, _, _, _, _, _, _);

    // arith_store_reg: memory-dest arithmetic with a register source; a
    // BP/SP-relative RMW operates IN-PLACE on the slot's stack_rtl so the
    // update threads to later reads of the same slot.
    #[local] relation arith_store_reg_uses_stack_param(Node, Address, i64, usize);
    arith_store_reg_uses_stack_param(addr, func_start, *disp, *ordinal) <--
        arith_store_reg(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        incoming_stack_slot(addr, func_start, base_mreg, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    // Seed a real mutable local from the incoming parameter. The existing
    // three-node RMW chain then updates that local and later stack def-use
    // reloads observe the updated value.
    stack_xtl(func_start, addr, *disp, stack_rtl) <--
        arith_store_reg_uses_stack_param(addr, func_start, disp, _),
        let stack_rtl = fresh_stack_cell_reg(*addr);

    #[local] relation arith_store_reg_uses_stack_var(Node);
    arith_store_reg_uses_stack_var(addr) <--
        arith_store_reg(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        stack_claim_base_at(func_start, addr, *base_mreg);

    // Stack-slot in-place RMW: the load/op/store triple collapses to a single in-place `slot = slot op src` writing the slot's stack_rtl (matches the Lsetstack model), no explicit memory load/store; the edge stays addr->next (no synthetic chain), see the guarded succ rules.
    rtl_inst_candidate(addr, op_inst) <--
        arith_store_reg(addr, op, _chunk, base_mreg, disp, src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *src_mreg, src_rtl),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        stack_claim_base_at(func_start, addr, *base_mreg),
        !arith_store_reg_uses_stack_param(addr, _, _, _),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl, *src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, init_inst) <--
        arith_store_reg_uses_stack_param(addr, func_start, disp, ordinal),
        stack_var(func_start, addr, disp, stack_rtl),
        let param_rtl = fresh_stack_param_reg(*func_start, *ordinal),
        let init_inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), *stack_rtl);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_reg_uses_stack_param(addr, func_start, disp, _),
        arith_store_reg(addr, op, _, _, _, src_mreg),
        reg_rtl(addr, *src_mreg, src_rtl),
        stack_var(func_start, addr, disp, stack_rtl),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl, *src_rtl]), *stack_rtl);

    rtl_inst_candidate(synth2, nop) <--
        arith_store_reg_uses_stack_param(addr, _, _, _),
        let synth2 = *addr | (1u64 << 63),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(addr, load_inst) <--
        arith_store_reg(addr, _op, chunk, base_mreg, disp, _src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_store_reg_uses_stack_var(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_reg(addr, op, _chunk, _base_mreg, _disp, src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *src_mreg, src_rtl),
        !arith_store_reg_uses_stack_var(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp, *src_rtl]), temp2);

    rtl_inst_candidate(synth2, store_inst) <--
        arith_store_reg(addr, _op, chunk, base_mreg, disp, _src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_store_reg_uses_stack_var(addr),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth2 = *addr | (1u64 << 63),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp2);

    // Stack-var case keeps the same addr->synth1->synth2->next chain (CFG must not depend on the value/stack layer); the two synthetic nodes carry Inop since the in-place op at addr does the work.
    rtl_inst_candidate(synth1, nop) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_reg_uses_stack_var(addr),
        !arith_store_reg_uses_stack_param(addr, _, _, _),
        let synth1 = *addr | (1u64 << 62),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synth2, nop) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_reg_uses_stack_var(addr),
        !arith_store_reg_uses_stack_param(addr, _, _, _),
        let synth2 = *addr | (1u64 << 63),
        let nop = RTLInst::Inop;

    rtl_edge_negated(addr, next) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synth1), instr_in_function(addr, func_start) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62);

    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    // arith_store_imm: memory-dest arithmetic with immediate; only fires
    // without a real ltl op. BP/SP-relative read-modify-write operates
    // IN-PLACE on the slot's stack_rtl so the updated value threads to later
    // reads of the same slot (see arith_store_reg).
    #[local] relation arith_store_imm_uses_stack_param(Node, Address, i64, usize);
    arith_store_imm_uses_stack_param(addr, func_start, *disp, *ordinal) <--
        arith_store_imm(addr, _, _, base_mreg, disp),
        !has_ltl_op(addr),
        incoming_stack_slot(addr, func_start, base_mreg, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    stack_xtl(func_start, addr, *disp, stack_rtl) <--
        arith_store_imm_uses_stack_param(addr, func_start, disp, _),
        let stack_rtl = fresh_stack_cell_reg(*addr);

    #[local] relation arith_store_imm_uses_stack_var(Node);
    arith_store_imm_uses_stack_var(addr) <--
        arith_store_imm(addr, _, _, base_mreg, disp),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        stack_claim_base_at(func_start, addr, *base_mreg);

    // Stack-slot in-place RMW: the load/op/store triple collapses to one in-place op writing the slot's stack_rtl, so the edge stays addr->next with no synthetic chain.
    rtl_inst_candidate(addr, op_inst) <--
        arith_store_imm(addr, op, _chunk, base_mreg, disp),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        stack_claim_base_at(func_start, addr, *base_mreg),
        !arith_store_imm_uses_stack_param(addr, _, _, _),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, init_inst) <--
        arith_store_imm_uses_stack_param(addr, func_start, disp, ordinal),
        stack_var(func_start, addr, disp, stack_rtl),
        let param_rtl = fresh_stack_param_reg(*func_start, *ordinal),
        let init_inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), *stack_rtl);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_imm_uses_stack_param(addr, func_start, disp, _),
        arith_store_imm(addr, op, _, _, _),
        stack_var(func_start, addr, disp, stack_rtl),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), *stack_rtl);

    rtl_inst_candidate(synth2, nop) <--
        arith_store_imm_uses_stack_param(addr, _, _, _),
        let synth2 = *addr | (1u64 << 63),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(addr, load_inst) <--
        arith_store_imm(addr, _op, chunk, base_mreg, disp),
        !has_ltl_op(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_store_imm_uses_stack_var(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_imm(addr, op, _chunk, _base_mreg, _disp),
        !has_ltl_op(addr),
        !arith_store_imm_uses_stack_var(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp]), temp2);

    rtl_inst_candidate(synth2, store_inst) <--
        arith_store_imm(addr, _op, chunk, base_mreg, disp),
        !has_ltl_op(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_store_imm_uses_stack_var(addr),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth2 = *addr | (1u64 << 63),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp2);

    // Stack-var case keeps the same addr->synth1->synth2->next chain (CFG must not depend on the value/stack layer, else the RTL successor SCC absorbs stack_var and breaks stratification); the two synthetic nodes carry Inop since the in-place op at addr already does the work.
    rtl_inst_candidate(synth1, nop) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_imm_uses_stack_var(addr),
        !arith_store_imm_uses_stack_param(addr, _, _, _),
        let synth1 = *addr | (1u64 << 62),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synth2, nop) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_imm_uses_stack_var(addr),
        !arith_store_imm_uses_stack_param(addr, _, _, _),
        let synth2 = *addr | (1u64 << 63),
        let nop = RTLInst::Inop;

    rtl_edge_negated(addr, next) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synth1), instr_in_function(addr, func_start) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62);

    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    // arith_store_abs_reg: read-modify-write at absolute (global) address with register source.
    rtl_inst_candidate(addr, load_inst) <--
        arith_store_abs_reg(addr, _op, chunk, ident, offset, _src_mreg),
        !has_ltl_op(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), temp);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_abs_reg(addr, op, _chunk, _ident, _offset, src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *src_mreg, src_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp, *src_rtl]), temp2);

    rtl_inst_candidate(synth2, store_inst) <--
        arith_store_abs_reg(addr, _op, chunk, ident, offset, _src_mreg),
        !has_ltl_op(addr),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth2 = *addr | (1u64 << 63),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), temp2);

    rtl_edge_negated(addr, next) <--
        arith_store_abs_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synth1), instr_in_function(addr, func_start) <--
        arith_store_abs_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62);

    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start) <--
        arith_store_abs_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        arith_store_abs_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    // arith_store_abs_imm: read-modify-write at absolute (global) address with immediate source.
    rtl_inst_candidate(addr, load_inst) <--
        arith_store_abs_imm(addr, _op, chunk, ident, offset),
        !has_ltl_op(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), temp);

    rtl_inst_candidate(synth1, op_inst) <--
        arith_store_abs_imm(addr, op, _chunk, _ident, _offset),
        !has_ltl_op(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth1 = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![temp]), temp2);

    rtl_inst_candidate(synth2, store_inst) <--
        arith_store_abs_imm(addr, _op, chunk, ident, offset),
        !has_ltl_op(addr),
        let temp2 = fresh_xtl_reg(*addr | (1u64 << 62), Mreg::x86("RTEMP")),
        let synth2 = *addr | (1u64 << 63),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aglobal(*ident, *offset), Arc::new(vec![]), temp2);

    rtl_edge_negated(addr, next) <--
        arith_store_abs_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        next(addr, next);

    rtl_succ_candidate(addr, synth1), instr_in_function(addr, func_start) <--
        arith_store_abs_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62);

    rtl_succ_candidate(synth1, synth2), instr_in_function(synth1, func_start) <--
        arith_store_abs_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth1 = *addr | (1u64 << 62),
        let synth2 = *addr | (1u64 << 63);

    rtl_succ_candidate(synth2, next), instr_in_function(synth2, func_start) <--
        arith_store_abs_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        let synth2 = *addr | (1u64 << 63),
        next(addr, next);

    reg_xtl(addr, dst, fresh_dst_id), is_def(addr, fresh_dst_id) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(_, _ofs, _, dst)),
        let fresh_dst_id = fresh_xtl_reg(*addr, *dst);

    ltl_inst(addr, LTLInst::Lsetstack(*src, slot, *ofs, *typ)) <--
        ltl_inst(addr, ?LTLInst::Lstore(mc, Addressing::Ainstack(ofs), args, src)),
        if args.len() == 0,
        let slot = Slot::Local,
        instruction(addr, _, _, _, src_reg, _, _, _, _, _),
        op_register(src_reg, src_reg_str),
        ireg_hold_type(src_reg_str.to_string(), typ);


    rtl_reg_used_in_func(func_start, *reg_rtl) <--
        reg_rtl(addr, mreg, reg_rtl),
        instr_in_function(addr, func_start);


    // A homed parameter reload can carry both Mgetstack and the historical
    // zero-argument Olea(Ainstack) interpretation at one address. Once the
    // home spill proves this is a value reload, suppress only that shadow LEA
    // so candidate selection cannot replace the parameter with &stack[ofs].
    #[local] relation home_reload_shadow_lea(Node);
    home_reload_shadow_lea(addr) <--
        win64_home_reload(addr, _, _, _),
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, _)),
        if mregs.is_empty(),
        if matches!(op, Operation::Olea(Addressing::Ainstack(_)));

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if mregs.is_empty(),
        !home_reload_shadow_lea(addr),
        reg_rtl(addr, *dst_reg, dst_rtl),
        let inst = RTLInst::Iop(op.clone(), Arc::new(vec![]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if mregs.is_empty(),
        !home_reload_shadow_lea(addr),
        !reg_xtl(addr, *dst_reg, _),
        let dst_rtl = fresh_xtl_reg(*addr, *dst_reg),
        let inst = RTLInst::Iop(op.clone(), Arc::new(vec![]), dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if !mregs.is_empty(),
        !lop_overwrites_input(addr, dst_reg),
        !div_result_dead(addr, dst_reg),
        reg_rtl(addr, *dst_reg, dst_rtl),
        op_args_collected(addr, args),
        let inst = RTLInst::Iop(op.clone(), args.clone(), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if !mregs.is_empty(),
        !reg_xtl(addr, *dst_reg, _),
        let dst_rtl = fresh_xtl_reg(*addr, *dst_reg),
        op_args_collected(addr, args),
        let inst = RTLInst::Iop(op.clone(), args.clone(), dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if !mregs.is_empty(),
        lop_overwrites_input(addr, dst_reg),
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        op_args_collected(addr, args),
        let inst = RTLInst::Iop(op.clone(), args.clone(), *dst_rtl);

    // Lop with dst shadowing an input mreg, restricted to Osel (preserve semantics: cmov fall-through, abs idiom); other self-mod ops (e.g. `add reg,imm` dst==src) need the original emission for SSA coalescing.
    relation lop_overwrites_input(Node, Mreg);
    lop_overwrites_input(addr, *dst_reg) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Osel(_, _), mregs, dst_reg)),
        for src in mregs.iter(),
        if src == dst_reg;

    // Osel reads dst as the false-branch preserve value; capstone doesn't always mark cmov dst readable, so add the use here to drive reg_def_used for the prior-def site.
    reg_use(addr, *dst_reg) <--
        lop_overwrites_input(addr, dst_reg);

    // Use-side xtl id for shadow position (mirrors load_overwrite_use_id below): Osel dst-position arg uses this canonical, leaving fresh def_id for the dst itself.
    relation lop_overwrite_use_id(Node, Mreg, RTLReg);
    lop_overwrite_use_id(addr, *dst_reg, use_id) <--
        lop_overwrites_input(addr, dst_reg),
        let use_id = fresh_xtl_reg(*addr, *dst_reg) | (1u64 << 62);

    reg_xtl(addr, *dst_reg, use_id) <--
        lop_overwrite_use_id(addr, dst_reg, use_id);

    reg_def_site(addr, Mreg::AX) <--
        pdiv(addr, _, _);

    reg_def_site(addr, Mreg::DX) <--
        pdiv(addr, _, _);

    // A div defines both AX and DX but the two Mops collide at one address, so mark the never-consumed output dead; only when the OTHER is consumed, so a div with both or neither used keeps both.
    relation div_result_dead(Node, Mreg);
    div_result_dead(addr, Mreg::AX) <--
        pdiv(addr, _, _),
        !reg_def_used(addr, Mreg::AX, _),
        reg_def_used(addr, Mreg::DX, _);
    div_result_dead(addr, Mreg::DX) <--
        pdiv(addr, _, _),
        !reg_def_used(addr, Mreg::DX, _),
        reg_def_used(addr, Mreg::AX, _);

    func_has_div_instr(func_start) <--
        instr_in_function(addr, func_start),
        pdiv(addr, _, _);


    // Signed division (IDIV): quotient in AX, remainder in DX
    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odiv, args.clone(), *quot_rtl)) <--
        pidiv(addr, _, divisor_sym),
        op_register(divisor_sym, divisor_reg_str),
        let divisor_mreg = Mreg::x86(divisor_reg_str.to_string()),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        reg_def_used(div_def_addr, divisor_mreg, addr),
        reg_rtl(div_def_addr, divisor_mreg, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(synthetic_addr, RTLInst::Iop(Operation::Omod, args.clone(), *rem_rtl)) <--
        pidiv(addr, _, divisor_sym),
        op_register(divisor_sym, divisor_reg_str),
        let divisor_mreg = Mreg::x86(divisor_reg_str.to_string()),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        reg_def_used(div_def_addr, divisor_mreg, addr),
        reg_rtl(div_def_addr, divisor_mreg, divisor_rtl),
        reg_rtl(addr, Mreg::DX, rem_rtl),
        let synthetic_addr = *addr | (1u64 << 62),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    // Unsigned division (DIV): quotient in AX, remainder in DX
    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odivu, args.clone(), *quot_rtl)) <--
        pudiv(addr, _, divisor_sym),
        op_register(divisor_sym, divisor_reg_str),
        let divisor_mreg = Mreg::x86(divisor_reg_str.to_string()),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        reg_def_used(div_def_addr, divisor_mreg, addr),
        reg_rtl(div_def_addr, divisor_mreg, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(synthetic_addr, RTLInst::Iop(Operation::Omodu, args.clone(), *rem_rtl)) <--
        pudiv(addr, _, divisor_sym),
        op_register(divisor_sym, divisor_reg_str),
        let divisor_mreg = Mreg::x86(divisor_reg_str.to_string()),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        reg_def_used(div_def_addr, divisor_mreg, addr),
        reg_rtl(div_def_addr, divisor_mreg, divisor_rtl),
        reg_rtl(addr, Mreg::DX, rem_rtl),
        let synthetic_addr = *addr | (1u64 << 62),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_edge_negated(addr, next_addr) <--
        pdiv(addr, _, divisor_sym),
        op_register(divisor_sym, _),
        next(addr, next_addr);

    rtl_succ_candidate(addr, synthetic_addr), instr_in_function(synthetic_addr, func_start) <--
        pdiv(addr, _, divisor_sym),
        op_register(divisor_sym, _),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62);

    rtl_succ_candidate(synthetic_addr, next_addr), instr_in_function(synthetic_addr, func_start) <--
        pdiv(addr, _, divisor_sym),
        op_register(divisor_sym, _),
        instr_in_function(addr, func_start),
        let synthetic_addr = *addr | (1u64 << 62),
        next(addr, next_addr);

    // A memory-divisor div has no ltl_inst and thus no reg_use(AX), so synthesize the AX dividend-use; the divisor is the SRC field of pidiv/pudiv, the same field the mach div rules read.
    reg_use(addr, Mreg::AX) <--
        pdiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, _, _, _, _, _, _);

    // Stack-divisor div emits quotient and remainder as COMPETING candidates at the SAME node, so selection keeps only the used result and a dead quotient cannot overwrite the dividend AX.
    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odiv, args.clone(), *quot_rtl)) <--
        pidiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if *reg2 == "RBP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::BP),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Omod, args.clone(), *rem_rtl)) <--
        pidiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if *reg2 == "RBP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::BP),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::DX, rem_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odivu, args.clone(), *quot_rtl)) <--
        pudiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if *reg2 == "RBP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::BP),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Omodu, args.clone(), *rem_rtl)) <--
        pudiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if *reg2 == "RBP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::BP),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::DX, rem_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.is_empty(),
        reg_rtl(addr, *dst_reg, dst_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() == 1,
        !win64_home_reload(addr, _, _, _),
        let arg_mreg = mregs[0],
        if arg_mreg != *dst_reg,
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, arg_mreg, arg_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![*arg_rtl]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() == 1,
        !win64_home_reload(addr, _, _, _),
        let arg_mreg = mregs[0],
        if arg_mreg == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, *dst_reg, arg_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![*arg_rtl]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        win64_home_reload(addr, func_start, param_mreg, _),
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        if mregs.len() == 1 && mregs[0] != *dst_reg,
        reg_rtl(addr, *dst_reg, dst_rtl),
        let param_rtl = fresh_xtl_reg(*func_start, *param_mreg),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        win64_home_reload(addr, func_start, param_mreg, _),
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        if mregs.len() == 1 && mregs[0] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        let param_rtl = fresh_xtl_reg(*func_start, *param_mreg),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), *dst_rtl);

    // dst_rtl binds via the load's def id, not reg_rtl(addr, dst_reg), which is multi-valued when the load reuses its destination as an address operand and previously dropped the load.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() > 1,
        if *addressing != Addressing::Aindexed(0),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        load_args_collected(addr, args),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), args.clone(), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() > 1,
        if *addressing == Addressing::Aindexed(0),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reg_rtl(addr, mregs[0], base_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![*base_rtl]), *dst_rtl);

    // SP-indexed load: expand [SP, idx] to Olea(Ainstack(ofs)) + Iload(Aindexed2scaled(scale, 0)).
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, sp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *addr, Mreg::SP, *ofs, sp_addr_rtl) <--
        sp_indexed_load(addr),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, addressing, mregs, _)),
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), sp_addr_rtl);

    // No collision: destination and index have distinct value webs.
    rtl_inst_candidate(synth, load_inst) <--
        sp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        if mregs[1] != *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        sp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2(ofs) = addressing,
        if mregs[1] != *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2(0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *dst_rtl);

    // Collision: the load overwrites its index register, so use the load's
    // fresh def for dst and the unique value reaching the read for the index.
    // Canonicalizing the node-local shadow directly is ambiguous after alias
    // closure: it names both the undefined shadow and the real incoming web.
    rtl_inst_candidate(synth, load_inst) <--
        sp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        if mregs[1] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, *dst_reg, idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        sp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        if mregs[1] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, *dst_reg, idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2(0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *dst_rtl);

    // SP-indexed load: edge rewiring; redirect addr's successors through the synthetic node
    rtl_edge_negated(addr, next) <--
        sp_indexed_load(addr),
        ltl_succ(addr, next);

    rtl_succ_candidate(addr, synth) <--
        sp_indexed_load(addr),
        let synth = *addr | (1u64 << 62);

    // Place synth Iload's instr_in_function via addr's func membership (not ltl_succ): when raw fallthrough lacks ltl_inst (fused `add mem,reg`), ltl_succ is missing and the synth was being dropped from func.inst.
    instr_in_function(synth, func_start) <--
        sp_indexed_load(addr),
        instr_in_function(addr, func_start),
        let synth = *addr | (1u64 << 62);

    // Synth -> next via ltl_succ when available (normal case).
    rtl_succ_candidate(synth, next) <--
        sp_indexed_load(addr),
        ltl_succ(addr, next),
        let synth = *addr | (1u64 << 62);

    // Synth -> next via function-bounded skip walk when raw next has no ltl_inst (ltl_succ missing); scoped replacement for the global next_skips_to_ltl, applied only to SP-indexed-load synth nodes.
    rtl_succ_candidate(synth, dst) <--
        sp_indexed_load(addr),
        instr_in_function(addr, func_start),
        next(addr, mid),
        !ltl_inst(*mid, _),
        instr_in_function(*mid, func_start),
        sp_synth_skip_to_ltl(*mid, *func_start, dst),
        let synth = *addr | (1u64 << 62);

    // SP-indexed store: Aindexed2scaled/Aindexed2 with [SP, idx] -> expand similarly
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, sp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *addr, Mreg::SP, *ofs, sp_addr_rtl) <--
        sp_indexed_store(addr),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, addressing, mregs, _)),
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), sp_addr_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        sp_indexed_store(addr),
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        reaching_use_rtl(addr, *src_reg, src_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *src_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        sp_indexed_store(addr),
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if let Addressing::Aindexed2(ofs) = addressing,
        reaching_use_rtl(addr, *src_reg, src_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2(0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *src_rtl);

    // SP-indexed store: edge rewiring
    rtl_edge_negated(addr, next) <--
        sp_indexed_store(addr),
        ltl_succ(addr, next);

    rtl_succ_candidate(addr, synth) <--
        sp_indexed_store(addr),
        let synth = *addr | (1u64 << 62);

    // Place the synthetic Istore's instr_in_function directly off addr's function membership (parallel to the SP-load case above).
    instr_in_function(synth, func_start) <--
        sp_indexed_store(addr),
        instr_in_function(addr, func_start),
        let synth = *addr | (1u64 << 62);

    // Synth -> next via ltl_succ when available (normal case).
    rtl_succ_candidate(synth, next) <--
        sp_indexed_store(addr),
        ltl_succ(addr, next),
        let synth = *addr | (1u64 << 62);

    // Synth -> next via the function-bounded skip walk when the store's raw next address has no ltl_inst (parallel to the SP-load case).
    rtl_succ_candidate(synth, dst) <--
        sp_indexed_store(addr),
        instr_in_function(addr, func_start),
        next(addr, mid),
        !ltl_inst(*mid, _),
        instr_in_function(*mid, func_start),
        sp_synth_skip_to_ltl(*mid, *func_start, dst),
        let synth = *addr | (1u64 << 62);

    // BP-frame-pointer-indexed load/store, the structural mirror of the SP-indexed expansion: expand [BP, idx] to Olea(Ainstack(disp)) plus an indexed-at-0 access, gated on a use-specific frame-pointer proof.
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, bp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *addr, Mreg::BP, *ofs, bp_addr_rtl) <--
        bp_indexed_load(addr),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, addressing, _, _)),
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), bp_addr_rtl);

    // No-collision: dst register differs from the index register, so reg_rtl gives both directly.
    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        if mregs[1] != *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        if mregs[1] != *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2(0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    // Collision where the load overwrites its own index register: the dst is
    // the fresh def and the index is the unique incoming reaching value.
    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        if mregs[1] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, *dst_reg, idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        if mregs[1] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        reaching_use_rtl(addr, *dst_reg, idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2(0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    // BP-indexed load: edge rewiring (mirror SP-indexed load).
    rtl_edge_negated(addr, next) <--
        bp_indexed_load(addr),
        ltl_succ(addr, next);

    rtl_succ_candidate(addr, synth) <--
        bp_indexed_load(addr),
        let synth = *addr | (1u64 << 62);

    instr_in_function(synth, func_start) <--
        bp_indexed_load(addr),
        instr_in_function(addr, func_start),
        let synth = *addr | (1u64 << 62);

    rtl_succ_candidate(synth, next) <--
        bp_indexed_load(addr),
        ltl_succ(addr, next),
        let synth = *addr | (1u64 << 62);

    rtl_succ_candidate(synth, dst) <--
        bp_indexed_load(addr),
        instr_in_function(addr, func_start),
        next(addr, mid),
        !ltl_inst(*mid, _),
        instr_in_function(*mid, func_start),
        sp_synth_skip_to_ltl(*mid, *func_start, dst),
        let synth = *addr | (1u64 << 62);

    // BP-indexed store: expand [BP, idx] to Olea(Ainstack(disp)) + Istore(Aindexed2scaled(scale, 0)).
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, bp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *addr, Mreg::BP, *ofs, bp_addr_rtl) <--
        bp_indexed_store(addr),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, addressing, _, _)),
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), bp_addr_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        bp_indexed_store(addr),
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        reaching_use_rtl(addr, *src_reg, src_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *src_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        bp_indexed_store(addr),
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        reaching_use_rtl(addr, *src_reg, src_rtl),
        reaching_use_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2(0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *src_rtl);

    // BP-indexed store: edge rewiring (mirror SP-indexed store).
    rtl_edge_negated(addr, next) <--
        bp_indexed_store(addr),
        ltl_succ(addr, next);

    rtl_succ_candidate(addr, synth) <--
        bp_indexed_store(addr),
        let synth = *addr | (1u64 << 62);

    instr_in_function(synth, func_start) <--
        bp_indexed_store(addr),
        instr_in_function(addr, func_start),
        let synth = *addr | (1u64 << 62);

    rtl_succ_candidate(synth, next) <--
        bp_indexed_store(addr),
        ltl_succ(addr, next),
        let synth = *addr | (1u64 << 62);

    rtl_succ_candidate(synth, dst) <--
        bp_indexed_store(addr),
        instr_in_function(addr, func_start),
        next(addr, mid),
        !ltl_inst(*mid, _),
        instr_in_function(*mid, func_start),
        sp_synth_skip_to_ltl(*mid, *func_start, dst),
        let synth = *addr | (1u64 << 62);

    // Give each SP/BP-indexed synthetic Olea(Ainstack(ofs)) a per-node
    // stack_xtl so cminor resolves it to Eaddrof(&local), not a frame integer.
    stack_xtl(func_start, addr, raw_ofs, stack_cell) <--
        indexed_synth_stack_base(func_start, addr, _, _, raw_ofs, _),
        let stack_cell = fresh_stack_cell_reg(*addr);

    indexed_synth_stack_coord(func_start, addr, normalized_ofs, stack_cell) <--
        indexed_synth_stack_base(func_start, addr, origin, base_kind, raw_ofs, _),
        if *base_kind == Mreg::SP,
        sp_entry_ofs(func_start, origin, sp_ofs),
        let normalized_ofs = sp_ofs.0 + *raw_ofs,
        let stack_cell = fresh_stack_cell_reg(*addr);
    indexed_synth_stack_coord(func_start, addr, normalized_ofs, stack_cell) <--
        indexed_synth_stack_base(func_start, addr, origin, base_kind, raw_ofs, _),
        if *base_kind == Mreg::BP,
        bp_base_at(func_start, origin, bp_base),
        let normalized_ofs = *bp_base + *raw_ofs,
        let stack_cell = fresh_stack_cell_reg(*addr);

    normalized_stack_lea_base(func_start, addr, *emitted_base, normalized_ofs) <--
        indexed_synth_stack_base(func_start, addr, origin, base_kind, raw_ofs, emitted_base),
        if *base_kind == Mreg::SP,
        sp_entry_ofs(func_start, origin, sp_ofs),
        let normalized_ofs = sp_ofs.0 + *raw_ofs;
    normalized_stack_lea_base(func_start, addr, *emitted_base, normalized_ofs) <--
        indexed_synth_stack_base(func_start, addr, origin, base_kind, raw_ofs, emitted_base),
        if *base_kind == Mreg::BP,
        bp_base_at(func_start, origin, bp_base),
        let normalized_ofs = *bp_base + *raw_ofs;

    // Preserve the old explicit stack-LEA coverage while requiring the same
    // decoder-owned base and normalized coordinate proof as ordinary stack
    // accesses.  Oleal is intentionally excluded: 32-bit address semantics
    // are not equivalent to a Win64 frame pointer.
    normalized_stack_lea_base(func_start, node, *emitted_base, normalized_ofs) <--
        rtl_inst_candidate(node, ?RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), _, emitted_base)),
        instr_in_function(node, func_start),
        direct_stack_operand(node, decoded_base, decoded_disp, _),
        sp_based_mem_at(node, func_start, proven_base, base_ofs),
        if decoded_base == proven_base,
        let normalized_ofs = *base_ofs + *decoded_disp;

    // Normalize ordinary SP/BP stack variables before aliasing an indexed
    // synthetic base. Equal raw displacements at different SP depths (or on
    // different base registers) are not the same cell.
    #[local] relation normalized_stack_xtl(Address, i64, RTLReg);
    normalized_stack_xtl(func_start, normalized_ofs, *stack_reg) <--
        stack_xtl(func_start, addr, raw_ofs, stack_reg),
        direct_stack_operand(addr, Mreg::SP, raw_ofs, _),
        sp_entry_ofs(func_start, addr, sp_ofs),
        let normalized_ofs = sp_ofs.0 + *raw_ofs;
    normalized_stack_xtl(func_start, normalized_ofs, *stack_reg) <--
        stack_xtl(func_start, addr, raw_ofs, stack_reg),
        direct_stack_operand(addr, Mreg::BP, raw_ofs, _),
        bp_base_at(func_start, addr, bp_base),
        let normalized_ofs = *bp_base + *raw_ofs;
    normalized_stack_xtl(func_start, *normalized_ofs, *base_reg) <--
        indexed_synth_stack_coord(func_start, _, normalized_ofs, base_reg);

    alias_edge(base_reg, other_reg) <--
        indexed_synth_stack_coord(func_start, _, normalized_ofs, base_reg),
        normalized_stack_xtl(func_start, normalized_ofs, other_reg),
        if base_reg != other_reg;

    // Function-bounded skip walk: from a non-ltl-inst addr, walk `next` within the func until hitting one with ltl_inst. Only consumed by SP-indexed synth rules above.
    sp_synth_skip_to_ltl(start, func, dst) <--
        instr_in_function(start, func),
        next(start, dst),
        instr_in_function(dst, func),
        ltl_inst(*dst, _);

    sp_synth_skip_to_ltl(start, func, dst) <--
        instr_in_function(start, func),
        next(start, dst),
        instr_in_function(dst, func),
        synth_only_addr(*dst);

    sp_synth_skip_to_ltl(start, func, dst) <--
        instr_in_function(start, func),
        next(start, mid),
        !ltl_inst(*mid, _),
        !synth_only_addr(*mid),
        instr_in_function(*mid, func),
        sp_synth_skip_to_ltl(*mid, *func, dst);

    // synth_only_addr members: addresses with RTL synth chains but no ltl_inst.
    synth_only_addr(addr) <-- arith_load_op(addr, _, _, _, _, _), !has_ltl_op(addr);
    synth_only_addr(addr) <-- arith_store_reg(addr, _, _, _, _, _), !has_ltl_op(addr);
    synth_only_addr(addr) <-- arith_store_imm(addr, _, _, _, _), !has_ltl_op(addr);
    synth_only_addr(addr) <-- arith_store_abs_reg(addr, _, _, _, _, _), !has_ltl_op(addr);
    synth_only_addr(addr) <-- arith_store_abs_imm(addr, _, _, _, _), !has_ltl_op(addr);
    synth_only_addr(addr) <-- mach_imm_indirect_store(addr, _, _, _, _);
    synth_only_addr(addr) <-- mach_imm_stack_init(addr, _, _, _);

    // Bridge real ltl_inst's natural fallthrough into the first synth-only addr in a contiguous synth-only run; without this, ltl_fallthrough (which needs ltl_inst at next) drops the edge and erases the function body. Bounded by instr_in_function on both ends to never cross function boundaries.
    rtl_succ_candidate(prev, addr) <--
        ltl_inst(prev, prev_inst),
        if crate::decompile::passes::linear_pass::has_fallthrough(&prev_inst, *prev),
        instr_in_function(prev, func),
        next(prev, addr),
        !ltl_inst(*addr, _),
        synth_only_addr(*addr),
        instr_in_function(*addr, func);

    // Tail-bridge: when a synth-only addr's next() is neither ltl_inst nor synth-only, walk past it to the next live in-func ltl_inst and add a parallel edge to the post-skip target.
    rtl_succ_candidate(synth_last, dst) <--
        synth_only_addr(addr),
        instr_in_function(addr, func),
        next(addr, mid),
        !ltl_inst(*mid, _),
        !synth_only_addr(*mid),
        instr_in_function(*mid, func),
        sp_synth_skip_to_ltl(*mid, *func, dst),
        let synth_last = (*addr | (1u64 << 62)) | (1u64 << 63);

    // Some synth-only producers emit only a single synth (bit 62), not the double (62+63); tail is `addr | (1<<62)`, emit bridging edge from that synth too.
    rtl_succ_candidate(synth_last, dst) <--
        synth_only_addr(addr),
        instr_in_function(addr, func),
        next(addr, mid),
        !ltl_inst(*mid, _),
        !synth_only_addr(*mid),
        instr_in_function(*mid, func),
        sp_synth_skip_to_ltl(*mid, *func, dst),
        let synth_last = *addr | (1u64 << 62);

    // CMP+JCC for [mem]+imm with a non-stack base; stack-based forms are handled by the stack_var rules, and the non-BP/SP restriction avoids competing candidates.

    temp_cmp_reg(addr, fresh_reg) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, _, _),
        op_indirect(src, _, base_str, idx_str, _, _, _),
        cmp_generic_mem_operand(*addr, *src),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::x86("RTEMP"));

    temp_cmp_mreg_args(addr, pos, mreg) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, _, _),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        cmp_generic_mem_operand(*addr, *src),
        effective_address_size(addr, address_size),
        let res = transl_generic_cmp_addressing_rev_sized(
            base_str,
            idx_str,
            *scale,
            *disp,
            *address_size,
        ),
        if res.is_ok(),
        let (_, mreg_vec) = res.unwrap(),
        for (pos, mreg) in mreg_vec.into_iter().enumerate();

    temp_cmp_args_rtl(addr, pos, rtl_reg) <--
        temp_cmp_mreg_args(addr, pos, mreg),
        reg_rtl(addr, mreg, rtl_reg);

    ltl_inst_uses_mreg(addr, mreg) <--
        temp_cmp_mreg_args(addr, _, mreg);

    // Propagate reg_xtl into CMP address from prior def of regs CMP uses; CMP has no LTL/MachInst entry (fused into Icond at RTL synth), so ltl_inst-based propagation doesn't fire and temp_cmp_args_rtl can't resolve base/index regs.
    reg_xtl(cmp_addr, mreg, arg_id) <--
        temp_cmp_mreg_args(cmp_addr, _, mreg),
        reg_def_used(defaddr, mreg, cmp_addr),
        !load_overwrites_base(defaddr, mreg),
        reg_xtl(defaddr, mreg, arg_id);

    reg_xtl(cmp_addr, mreg, arg_id) <--
        temp_cmp_mreg_args(cmp_addr, _, mreg),
        reg_def_used(defaddr, mreg, cmp_addr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, arg_id),
        reg_xtl(defaddr, mreg, arg_id);

    rtl_inst_candidate(addr, RTLInst::Iload(chunk, addressing.clone(), mreg_args, *fresh_reg)) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, _, _),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        cmp_generic_mem_operand(*addr, *src),
        temp_cmp_reg(addr, fresh_reg),
        effective_address_size(addr, address_size),
        let res = transl_generic_cmp_addressing_rev_sized(
            base_str,
            idx_str,
            *scale,
            *disp,
            *address_size,
        ),
        if res.is_ok(),
        let (addressing, _) = res.unwrap(),
        agg mreg_args = build_call_args(pos, rtl_reg) in temp_cmp_args_rtl(addr, pos, rtl_reg),
        if !mreg_args.is_empty(),
        let chunk = if mnem.ends_with("L") { MemoryChunk::MInt32 }
                   else if mnem.ends_with("Q") { MemoryChunk::MInt64 }
                   else { MemoryChunk::MInt32 };

    // cmp imm, [global] with a RIP-relative operand has no base/index GPR, so the mreg-gated Iload never fires; resolve the RIP target to an Aglobal and define the fresh reg the Icond consumes.
    rtl_inst_candidate(addr, RTLInst::Iload(MemoryChunk::MInt32, Addressing::Aglobal(target_addr, 0), Arc::new(vec![]), *fresh_reg)) <--
        instruction(addr, size, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if is_rip(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        temp_cmp_reg(addr, fresh_reg),
        let target_addr = (*addr as i64 + *size as i64 + *disp) as Ident;

    // Icond from cmp[mem],imm placed at JCC (CMP/JCC sequential); false target is JCC's fallthrough (not next_addr=JCC) to avoid self-loop on false branch. Mirrors patterns at 1766/1782/1834.
    rtl_inst_candidate(*next_addr, RTLInst::Icond(cond, args.clone(), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, imm_sym, _),
        op_indirect(src, _, base_str, idx_str, _, _, _),
        cmp_generic_mem_operand(*addr, *src),
        let imm_val = *imm_sym as i64,
        next(addr, next_addr),
        pjcc(next_addr, test_cond, lbl),
        temp_cmp_reg(addr, fresh_reg),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(next_addr, my_func_start),
        if target_func_start == my_func_start,
        next(next_addr, fallthrough),
        // jcc-taken branch fires on the identity predicate of the test (not its logical inverse); mirrors asm_pass.rs.
        let base = crate::x86::types::condition_for_testcond(*test_cond),
        let cond = match base {
             Condition::Ccomp(c)  => Condition::Ccompimm(c, imm_val),
             Condition::Ccompu(c) => Condition::Ccompuimm(c, imm_val),
             _ => Condition::Ccomp(Comparison::Unknown)
        },
        let args = Arc::new(vec![*fresh_reg]);

    temp_cmp_reg(addr, fresh_reg) <--
        pcmp(addr, r1, r2),
        op_register(r1, _),
        op_indirect(r2, _, _, _, _, _, _),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::x86("RTEMP"));

    temp_cmp_mreg_args(addr, pos, mreg) <--
        pcmp(addr, r1, r2),
        op_register(r1, _),
        op_indirect(r2, _, base_str, idx_str, scale, disp, _),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        effective_address_size(addr, address_size),
        let res = transl_addressing_rev_sized(addrmode, None, *address_size),
        if res.is_ok(),
        let (_, mreg_vec) = res.unwrap(),
        for (pos, mreg) in mreg_vec.into_iter().enumerate();

    rtl_inst_candidate(addr, RTLInst::Iload(chunk, addressing.clone(), mreg_args, *fresh_reg)) <--
        pcmp(addr, r1, r2),
        op_register(r1, _),
        op_indirect(r2, _, base_str, idx_str, scale, disp, sz),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        effective_address_size(addr, address_size),
        let res = transl_addressing_rev_sized(addrmode, None, *address_size),
        if res.is_ok(),
        let (addressing, _) = res.unwrap(),
        agg mreg_args = build_call_args(pos, rtl_reg) in temp_cmp_args_rtl(addr, pos, rtl_reg),
        let chunk = match *sz { 1 => MemoryChunk::MInt8Unsigned, 2 => MemoryChunk::MInt16Unsigned, 8 => MemoryChunk::MInt64, _ => MemoryChunk::MInt32 };

    rtl_inst_candidate(*jcc_addr, RTLInst::Icond(cond, Arc::new(vec![*reg_rtl, *fresh_reg]), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        pcmp(addr, r1, r2),
        op_register(r1, reg_str),
        op_indirect(r2, _, _, _, _, disp, _),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let mreg = Mreg::x86(reg_str),
        reg_rtl(addr, mreg, reg_rtl),
        next(addr, jcc_addr),
        pjcc(jcc_addr, test_cond, lbl),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(jcc_addr, my_func_start),
        if target_func_start == my_func_start,
        next(jcc_addr, fallthrough),
        let cond = condition_for_testcond(*test_cond);

    rtl_inst_candidate(*jcc_addr, RTLInst::Icond(cond, Arc::new(vec![*reg_rtl, *fresh_reg]), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        pcmp(addr, r1, r2),
        op_register(r1, reg_str),
        op_indirect(r2, _, _, _, _, disp, _),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let mreg = Mreg::x86(reg_str),
        reg_rtl(addr, mreg, reg_rtl),
        flags_and_jump_pair(addr, jcc_addr, _),
        pjcc(jcc_addr, test_cond, lbl),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(jcc_addr, my_func_start),
        if target_func_start == my_func_start,
        next(jcc_addr, fallthrough),
        let cond = condition_for_testcond(*test_cond);

    temp_cmp_reg(addr, fresh_reg) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, _, _, _, _, _),
        op_register(r2, _),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::x86("RTEMP"));

    temp_cmp_mreg_args(addr, pos, mreg) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, base_str, idx_str, scale, disp, _),
        op_register(r2, _),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        effective_address_size(addr, address_size),
        let res = transl_addressing_rev_sized(addrmode, None, *address_size),
        if res.is_ok(),
        let (_, mreg_vec) = res.unwrap(),
        for (pos, mreg) in mreg_vec.into_iter().enumerate();

    rtl_inst_candidate(addr, RTLInst::Iload(chunk, addressing.clone(), mreg_args, *fresh_reg)) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, base_str, idx_str, scale, disp, sz),
        op_register(r2, _),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        effective_address_size(addr, address_size),
        let res = transl_addressing_rev_sized(addrmode, None, *address_size),
        if res.is_ok(),
        let (addressing, _) = res.unwrap(),
        agg mreg_args = build_call_args(pos, rtl_reg) in temp_cmp_args_rtl(addr, pos, rtl_reg),
        let chunk = match *sz { 1 => MemoryChunk::MInt8Unsigned, 2 => MemoryChunk::MInt16Unsigned, 8 => MemoryChunk::MInt64, _ => MemoryChunk::MInt32 };

    rtl_inst_candidate(*jcc_addr, RTLInst::Icond(cond, Arc::new(vec![*fresh_reg, *reg_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, _, _, _, disp, _),
        op_register(r2, reg_str),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let mreg = Mreg::x86(reg_str),
        reg_rtl(addr, mreg, reg_rtl),
        next(addr, jcc_addr),
        pjcc(jcc_addr, test_cond, lbl),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(jcc_addr, my_func_start),
        if target_func_start == my_func_start,
        next(jcc_addr, fallthrough),
        let cond = condition_for_testcond(*test_cond);

    rtl_inst_candidate(*jcc_addr, RTLInst::Icond(cond, Arc::new(vec![*fresh_reg, *reg_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, _, _, _, disp, _),
        op_register(r2, reg_str),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let mreg = Mreg::x86(reg_str),
        reg_rtl(addr, mreg, reg_rtl),
        flags_and_jump_pair(addr, jcc_addr, _),
        pjcc(jcc_addr, test_cond, lbl),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(jcc_addr, my_func_start),
        if target_func_start == my_func_start,
        next(jcc_addr, fallthrough),
        let cond = condition_for_testcond(*test_cond);

    rtl_inst_candidate(*jcc_addr, RTLInst::Icond(cond, args.clone(), Either::Right(*target_addr), Either::Right(*fallthrough))) <--
        instruction(addr, _, mnem, dst, src, _, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_indirect(dst, _, _, _, _, _, _),
        op_immediate(src, imm_sym, _),
        let imm_val = *imm_sym as i64,
        flags_and_jump_pair(addr, jcc_addr, _),
        pjcc(jcc_addr, test_cond, lbl),
        temp_cmp_reg(addr, fresh_reg),
        symbol_resolved_addr(*lbl, target_addr),
        instr_in_function(target_addr, target_func_start),
        instr_in_function(jcc_addr, my_func_start),
        if target_func_start == my_func_start,
        next(jcc_addr, fallthrough),
        let cond = match test_cond {
             TestCond::CondG  => Condition::Ccompimm(Comparison::Cgt, imm_val),
             TestCond::CondL  => Condition::Ccompimm(Comparison::Clt, imm_val),
             TestCond::CondGe => Condition::Ccompimm(Comparison::Cge, imm_val),
             TestCond::CondLe => Condition::Ccompimm(Comparison::Cle, imm_val),
             TestCond::CondE  => Condition::Ccompimm(Comparison::Ceq, imm_val),
             TestCond::CondNe => Condition::Ccompimm(Comparison::Cne, imm_val),
             TestCond::CondA  => Condition::Ccompuimm(Comparison::Cgt, imm_val),
             TestCond::CondB  => Condition::Ccompuimm(Comparison::Clt, imm_val),
             TestCond::CondAe => Condition::Ccompuimm(Comparison::Cge, imm_val),
             TestCond::CondBe => Condition::Ccompuimm(Comparison::Cle, imm_val),
             _ => Condition::Ccomp(Comparison::Unknown)
        },
        let args = Arc::new(vec![*fresh_reg]);

    // CMP+SETcc fusion where one CMP operand is genuine memory: emit the boolean as an Ocmp into the SETcc's destination consuming the already-loaded fresh_reg; gated !stack_var so named slots stay on the stack path.
    relation setcc_testcond(Address, TestCond);
    setcc_testcond(addr, tc) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if let Some(tc) = setcc_mnem_testcond(mnem);

    // Cross-node ownership for the exact CMP consumers emitted above/below.
    // Candidate-specific filtering later still verifies that the candidate
    // actually reads `fresh_reg`, so an unrelated lowering at the same JCC or
    // SETcc node remains independent.
    cmp_memory_temp_consumer(*addr, *consumer, *fresh_reg, *function) <--
        temp_cmp_reg(addr, fresh_reg),
        cmp_has_non_stack_mem(addr),
        next(addr, consumer),
        pjcc(consumer, _, _),
        instr_in_function(addr, function),
        instr_in_function(consumer, function);
    cmp_memory_temp_consumer(*addr, *consumer, *fresh_reg, *function) <--
        temp_cmp_reg(addr, fresh_reg),
        cmp_has_non_stack_mem(addr),
        flags_and_jump_pair(addr, consumer, _),
        pjcc(consumer, _, _),
        instr_in_function(addr, function),
        instr_in_function(consumer, function);
    cmp_memory_temp_consumer(*addr, *consumer, *fresh_reg, *function) <--
        temp_cmp_reg(addr, fresh_reg),
        cmp_has_non_stack_mem(addr),
        next(addr, consumer),
        setcc_testcond(consumer, _),
        instr_in_function(addr, function),
        instr_in_function(consumer, function);

    // pcmp(r1=indirect mem, r2=register), SETcc immediately follows: Ocmp([loaded_mem, reg], cond).
    rtl_inst_candidate(*setcc_addr, RTLInst::Iop(Operation::Ocmp(cond), Arc::new(vec![*fresh_reg, *reg_rtl]), *dst_rtl)) <--
        pcmp(addr, r1, r2),
        op_indirect(r1, _, _, _, _, disp, _),
        op_register(r2, reg_str),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let cmp_mreg = Mreg::x86(reg_str),
        reg_rtl(addr, cmp_mreg, reg_rtl),
        next(addr, setcc_addr),
        setcc_testcond(setcc_addr, test_cond),
        instruction(setcc_addr, _, _, _, dst_sym, _, _, _, _, _),
        op_register(dst_sym, dst_str),
        reg_rtl(setcc_addr, Mreg::x86(dst_str), dst_rtl),
        let cond = condition_for_testcond(*test_cond);

    // pcmp(r1=register, r2=indirect mem), SETcc immediately follows: Ocmp([reg, loaded_mem], cond).
    rtl_inst_candidate(*setcc_addr, RTLInst::Iop(Operation::Ocmp(cond), Arc::new(vec![*reg_rtl, *fresh_reg]), *dst_rtl)) <--
        pcmp(addr, r1, r2),
        op_register(r1, reg_str),
        op_indirect(r2, _, _, _, _, disp, _),
        instr_in_function(addr, cmp_func),
        !stack_var(cmp_func, addr, *disp, _),
        temp_cmp_reg(addr, fresh_reg),
        let cmp_mreg = Mreg::x86(reg_str),
        reg_rtl(addr, cmp_mreg, reg_rtl),
        next(addr, setcc_addr),
        setcc_testcond(setcc_addr, test_cond),
        instruction(setcc_addr, _, _, _, dst_sym, _, _, _, _, _),
        op_register(dst_sym, dst_str),
        reg_rtl(setcc_addr, Mreg::x86(dst_str), dst_rtl),
        let cond = condition_for_testcond(*test_cond);

    // cmp [mem],imm with a non-stack base followed by SETcc: Ocmp([loaded_mem], Ccompimm), with the Iload produced by the CMP-mem-imm rules above and the immediate baked into the condition.
    rtl_inst_candidate(*setcc_addr, RTLInst::Iop(Operation::Ocmp(cond), Arc::new(vec![*fresh_reg]), *dst_rtl)) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_indirect(dst, _, base_str, idx_str, _, _, _),
        op_immediate(src, imm_sym, _),
        cmp_generic_mem_operand(*addr, *dst),
        temp_cmp_reg(addr, fresh_reg),
        let imm_val = *imm_sym as i64,
        next(addr, setcc_addr),
        setcc_testcond(setcc_addr, test_cond),
        instruction(setcc_addr, _, _, _, dst_sym, _, _, _, _, _),
        op_register(dst_sym, dst_str),
        reg_rtl(setcc_addr, Mreg::x86(dst_str), dst_rtl),
        let cond = match condition_for_testcond(*test_cond) {
             Condition::Ccomp(c)  => Condition::Ccompimm(c, imm_val),
             Condition::Ccompu(c) => Condition::Ccompuimm(c, imm_val),
             other => other,
        };

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if !mregs.is_empty(),
        !win64_home_spill(addr, _, _, _),
        !sp_indexed_store(addr),
        !bp_indexed_store(addr),
        reg_rtl(addr, *src_reg, src_rtl),
        store_args_collected(addr, args),
        let inst = RTLInst::Istore(*chunk, addressing.clone(), args.clone(), *src_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if mregs.is_empty(),
        !win64_home_spill(addr, _, _, _),
        reg_rtl(addr, *src_reg, src_rtl),
        let inst = RTLInst::Istore(*chunk, addressing.clone(), Arc::new(vec![]), *src_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Left(mreg) = callee,
        !call_through_memory(addr, _, _, _, _),
        reg_rtl(addr, *mreg, callee_rtl),
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(*callee_rtl), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Left(mreg) = callee,
        !call_through_memory(addr, _, _, _, _),
        reg_rtl(addr, *mreg, callee_rtl),
        call_args_collected_candidate(addr, args),
        !call_return_reg(addr, _),
        next(addr, next_addr),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(*callee_rtl), args.clone(), None, *next_addr);

    // Fallback: indirect Lcall with missing reg_rtl; create fresh RTL reg for callee target.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Left(mreg) = callee,
        !call_through_memory(addr, _, _, _, _),
        !reg_rtl(addr, *mreg, _),
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        let fresh_callee = fresh_xtl_reg(*addr, *mreg),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(fresh_callee), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Left(mreg) = callee,
        !call_through_memory(addr, _, _, _, _),
        !reg_rtl(addr, *mreg, _),
        call_args_collected_candidate(addr, args),
        !call_return_reg(addr, _),
        next(addr, next_addr),
        let fresh_callee = fresh_xtl_reg(*addr, *mreg),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(fresh_callee), args.clone(), None, *next_addr);

    // Memory-indirect call: Mcall(Left(base_mreg)) loses disp/idx; re-emit Icall with a fresh RTL reg callee. The fresh reg is inlined as Eload by cshminor so the C renders as `(*(disp(base)))(args)`, not a direct call to base.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(_))),
        call_through_memory(addr, _, _, _, _),
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RCALL_TGT")),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(temp), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(_))),
        call_through_memory(addr, _, _, _, _),
        call_args_collected_candidate(addr, args),
        !call_return_reg(addr, _),
        next(addr, next_addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RCALL_TGT")),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Left(temp), args.clone(), None, *next_addr);

    // Surface the mem-indirect call's load addressing for cshminor Eload rendering: base->Aindexed(disp); base+idx->Aindexed2(disp); base+idx*scale->Aindexed2scaled(scale,disp). Temp reg matches the Icall callee above.
    relation call_through_memory_load(Node, RTLReg, MemoryChunk, Addressing, Args);

    call_through_memory_load(*addr, temp, MemoryChunk::MInt64, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl])) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(_))),
        call_through_memory(addr, base_str, idx_str, _scale, disp),
        if *idx_str == "NONE" || idx_str.is_empty(),
        let base_mreg = Mreg::x86(*base_str),
        reg_rtl(addr, base_mreg, base_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RCALL_TGT"));

    call_through_memory_load(*addr, temp, MemoryChunk::MInt64, Addressing::Aindexed2(*disp), Arc::new(vec![*base_rtl, *idx_rtl])) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(_))),
        call_through_memory(addr, base_str, idx_str, scale, disp),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *scale <= 1,
        let base_mreg = Mreg::x86(*base_str),
        let idx_mreg = Mreg::x86(*idx_str),
        reg_rtl(addr, base_mreg, base_rtl),
        reg_rtl(addr, idx_mreg, idx_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RCALL_TGT"));

    call_through_memory_load(*addr, temp, MemoryChunk::MInt64, Addressing::Aindexed2scaled(*scale, *disp), Arc::new(vec![*base_rtl, *idx_rtl])) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Left(_))),
        call_through_memory(addr, base_str, idx_str, scale, disp),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if *scale > 1,
        let base_mreg = Mreg::x86(*base_str),
        let idx_mreg = Mreg::x86(*idx_str),
        reg_rtl(addr, base_mreg, base_rtl),
        reg_rtl(addr, idx_mreg, idx_rtl),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RCALL_TGT"));

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        emit_function_signature_candidate(target_addr, sig),
        let inst = RTLInst::Icall(Some(sig.clone()), Either::Right(target.clone()), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        !call_return_reg(addr, _),
        next(addr, next_addr),
        emit_function_signature_candidate(target_addr, sig),
        let inst = RTLInst::Icall(Some(sig.clone()), Either::Right(target.clone()), args.clone(), None, *next_addr);


    reg_xtl(call_addr, x0, ret_rtl), call_return_reg(call_addr, ret_rtl) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        next(call_addr, next_addr),
        let x0 = Mreg::X0,
        ltl_inst_uses_mreg(next_addr, x0),
        !call_returns_value(call_addr, _),
        let ret_rtl = fresh_xtl_reg(*call_addr, x0);

    reg_xtl(call_addr, mreg, ret_rtl), call_return_reg(call_addr, ret_rtl) <--
        call_has_known_signature(call_addr, _name, _count, ret_type),
        if *ret_type != XType::Xvoid,
        !call_returns_value(call_addr, _),
        let mreg = if matches!(ret_type, XType::Xfloat | XType::Xsingle) {
            Mreg::X0
        } else {
            Mreg::AX
        },
        let ret_rtl = fresh_xtl_reg(*call_addr, mreg);

    reg_def_site(addr, *dst) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, _, dst));

    reg_def_site(addr, *dst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, dst));

    reg_def_site(addr, *dst) <--
        ltl_inst(addr, ?LTLInst::Lop(_, _, dst));

    reg_def_site(*addr, *mreg) <--
        reg_def(addr, mreg),
        if *mreg != Mreg::SP,
        !ltl_inst(addr, _),
        !trim_instruction(addr);


    reg_def_site(*call_addr, *reg) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        is_caller_saved(reg),
        reg_def_used(*call_addr, *reg, _);


    return_val_used(*call_addr, 0u64, Mreg::AX, *use_addr, 0i64) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        reg_def_used(*call_addr, Mreg::AX, use_addr),
        instr_in_function(call_addr, func_start),
        instr_in_function(use_addr, func_start);

    call_returns_value(*call_addr, Mreg::AX) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        reg_def_used(*call_addr, Mreg::AX, use_addr),
        instr_in_function(call_addr, func_start),
        instr_in_function(use_addr, func_start);

    // XMM0 return detection GATED on the callee actually returning float, or a bare call's caller-saved XMM0 clobber mints a spurious X0 return that poisons the real int/ptr return.
    call_returns_value(*call_addr, Mreg::X0) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        reg_def_used(*call_addr, Mreg::X0, use_addr),
        instr_in_function(call_addr, func_start),
        instr_in_function(use_addr, func_start),
        x0_value_write(call_addr);

    reg_xtl(call_addr, mreg, ret_rtl), call_return_reg(call_addr, ret_rtl) <--
        call_returns_value(call_addr, mreg),
        let ret_rtl = fresh_xtl_reg(*call_addr, *mreg);

    reg_xtl(addr, mreg, id), is_def(addr, id) <--
        reg_def_site(addr, mreg),
        let id = fresh_xtl_reg(*addr, *mreg);

    // Lbranch addresses reg_def butno ltl op
    reg_def_site(*addr, *mreg) <--
        arith_load_op(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        reg_def(addr, mreg),
        if *mreg != Mreg::SP,
        !trim_instruction(addr);

    reg_def_site(*addr, *mreg) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        reg_def(addr, mreg),
        if *mreg != Mreg::SP,
        !trim_instruction(addr);

    reg_def_site(*addr, *mreg) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        reg_def(addr, mreg),
        if *mreg != Mreg::SP,
        !trim_instruction(addr);

    // 3.8 R1: fused float op's register write (e.g. the X0 of `addsd k(%rip),%xmm0`), mirroring the arith legs above so the def exists even when a label/branch ltl_inst coexists at the address.
    reg_def_site(*addr, *mreg) <--
        float_load_op(addr, _, _, _, _, _, _),
        !has_ltl_op(addr),
        reg_def(addr, mreg),
        if *mreg != Mreg::SP,
        !trim_instruction(addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Left(mreg) = callee,
        reg_rtl(addr, *mreg, callee_rtl),
        call_args_collected_candidate(addr, args),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Left(*callee_rtl), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Left(mreg) = callee,
        reg_rtl(addr, *mreg, callee_rtl),
        !call_args_collected_candidate(addr, _),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Left(*callee_rtl), Arc::new(vec![]));

    // Fallback: indirect Ltailcall with missing reg_rtl (mirrors Lcall fallback above).
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Left(mreg) = callee,
        !reg_rtl(addr, *mreg, _),
        call_args_collected_candidate(addr, args),
        let fresh_callee = fresh_xtl_reg(*addr, *mreg),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Itailcall(Some(inferred_sig), Either::Left(fresh_callee), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Left(mreg) = callee,
        !reg_rtl(addr, *mreg, _),
        !call_args_collected_candidate(addr, _),
        let fresh_callee = fresh_xtl_reg(*addr, *mreg),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Left(fresh_callee), Arc::new(vec![]));

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        emit_function_signature_candidate(target_addr, sig),
        let inst = RTLInst::Itailcall(Some(sig.clone()), Either::Right(target.clone()), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        !call_args_collected_candidate(addr, _),
        emit_function_signature_candidate(target_addr, sig),
        let inst = RTLInst::Itailcall(Some(sig.clone()), Either::Right(target.clone()), Arc::new(vec![]));

    // The extern signature table applies to a by-name target ONLY when the binary does not define that name; the twin rule below covers the gated symbol via inference.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        !emit_function(_, symbol, _),
        known_extern_signature(symbol, _, ret_type, known_params),
        let sig = Signature { sig_args: known_params.clone(), sig_res: *ret_type, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(sig), Either::Right(target.clone()), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        !known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Itailcall(Some(inferred_sig), Either::Right(target.clone()), args.clone());

    // Hole-closer twin of the rule above: a locally-defined symbol that ALSO has a table entry (so `!known_extern_signature` is false) still gets a signature, inferred from the binary's own call-site args rather than the corpus table.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        emit_function(_, symbol, _),
        known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Itailcall(Some(inferred_sig), Either::Right(target.clone()), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_args_collected_candidate(addr, _),
        !emit_function(_, symbol, _),
        known_extern_signature(symbol, _, ret_type, _known_params),
        let sig = Signature { sig_args: Arc::new(vec![]), sig_res: *ret_type, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(sig), Either::Right(target.clone()), Arc::new(vec![]));

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_args_collected_candidate(addr, _),
        !known_extern_signature(symbol, _, _, _),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Right(target.clone()), Arc::new(vec![]));

    // Hole-closer twin: locally-defined symbol that also has a table entry -> default sig (matching the no-args fallback above) instead of the corpus table.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_args_collected_candidate(addr, _),
        emit_function(_, symbol, _),
        known_extern_signature(symbol, _, _, _),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Right(target.clone()), Arc::new(vec![]));

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        !emit_function_signature_candidate(target_addr, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Itailcall(Some(inferred_sig), Either::Right(target.clone()), args.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        !call_args_collected_candidate(addr, _),
        !emit_function_signature_candidate(target_addr, _),
        let default_sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xint, sig_cc: CallConv::default() },
        let inst = RTLInst::Itailcall(Some(default_sig), Either::Right(target.clone()), Arc::new(vec![]));


    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        !emit_function_signature_candidate(target_addr, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Left(target_addr) = target,
        call_args_collected_candidate(addr, args),
        !call_return_reg(addr, _),
        next(addr, next_addr),
        !emit_function_signature_candidate(target_addr, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), None, *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        !emit_function(_, symbol, _),
        known_extern_signature(symbol, _, ret_type, known_params),
        let sig = Signature { sig_args: known_params.clone(), sig_res: *ret_type, sig_cc: CallConv::default() },
        let inst = RTLInst::Icall(Some(sig), Either::Right(target.clone()), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        !known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), Some(*final_ret), *next_addr);

    // Hole-closer twin: locally-defined symbol that also has a table entry -> inferred sig.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        call_args_collected_candidate(addr, args),
        call_return_reg(addr, _ret_rtl),
        reg_rtl(addr, Mreg::AX, final_ret),
        next(addr, next_addr),
        emit_function(_, symbol, _),
        known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, true),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), Some(*final_ret), *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_return_reg(addr, _),
        call_args_collected_candidate(addr, args),
        next(addr, next_addr),
        !emit_function(_, symbol, _),
        known_extern_signature(symbol, _, _, known_params),
        let sig = Signature { sig_args: known_params.clone(), sig_res: XType::Xvoid, sig_cc: CallConv::default() },
        let inst = RTLInst::Icall(Some(sig), Either::Right(target.clone()), args.clone(), None, *next_addr);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_return_reg(addr, _),
        call_args_collected_candidate(addr, args),
        next(addr, next_addr),
        !known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), None, *next_addr);

    // Hole-closer twin: locally-defined symbol that also has a table entry -> inferred sig.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcall(callee)),
        if let Either::Right(target) = callee,
        if let Either::Right(symbol) = target,
        !call_return_reg(addr, _),
        call_args_collected_candidate(addr, args),
        next(addr, next_addr),
        emit_function(_, symbol, _),
        known_extern_signature(symbol, _, _, _),
        let inferred_sig = crate::decompile::passes::rtl_pass::infer_signature_from_args(&args, false),
        let inst = RTLInst::Icall(Some(inferred_sig), Either::Right(target.clone()), args.clone(), None, *next_addr);

    ltl_builtin_unconverted(addr, name_sym, Arc::new(args.clone()), result.clone()) <--
        ltl_inst(addr, ?LTLInst::Lbuiltin(name, args, result)),
        let name_sym = Box::leak(name.clone().into_boxed_str()) as Symbol;

    // LTL builtin -> RTL Ibuiltin: map each Mreg arg to its RTLReg via reg_rtl.
    rtl_inst_candidate(addr, inst) <--
        ltl_builtin_unconverted(addr, name, args, result),
        agg pairs = collect_builtin_reg_pairs(mreg, r) in reg_rtl(addr, mreg, r),
        let inst = build_builtin_inst(*addr, name, args, result, &pairs);

    // Builtin at a node with no reg_rtl entries: convert with an empty map.
    rtl_inst_candidate(addr, inst) <--
        ltl_builtin_unconverted(addr, name, args, result),
        !reg_rtl(addr, _, _),
        let inst = build_builtin_inst(*addr, name, args, result, &[]);


    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lbranch(target)),
        // Suppress Ibranch when R3 imm-global-store owns the address: the synth chain already routes to next(addr), and a competing Ibranch orphans the synthetic Istore, breaking every guard arm's return-tail shape.
        !mach_imm_global_store(addr, _, _, _, _),
        let inst = RTLInst::Ibranch(target.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcond(cond, mregs, target_true, target_false)),
        if mregs.len() > 0,
        !counter_branch_split_addr(addr),
        cond_args_collected(addr, args),
        let inst = RTLInst::Icond(cond.clone(), args.clone(), target_true.clone(), target_false.clone());

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lcond(cond, mregs, target_true, target_false)),
        if mregs.is_empty(),
        !counter_branch_split_addr(addr),
        let inst = RTLInst::Icond(cond.clone(), Arc::new(vec![]), target_true.clone(), target_false.clone());

    // Decrement-and-branch (sub $1,%reg; jcc): split so the Iop decrement stays at addr and the Icond moves to the synthetic successor, shifting its immediate by the addend since it now runs post-decrement.
    relation counter_branch_split_addr(Address);
    counter_branch_split_addr(*addr) <--
        counter_branch_op_addend(addr, dst, _),
        ltl_inst(addr, ?LTLInst::Lcond(cond, cond_mregs, _, _)),
        if counter_branch_shifted_cond(cond, 0).is_some(),
        if cond_mregs.len() == 1 && &cond_mregs[0] == dst;

    // An in-place immediate add (the persisted decrement) writing reg from reg, 32- or 64-bit; the 32-bit case was previously unmatched, dropping the decrement and freezing the counter.
    relation counter_branch_op_addend(Address, Mreg, i64);
    counter_branch_op_addend(*addr, *dst, *k) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Oaddimm(k), op_mregs, dst)),
        if op_mregs.len() == 1 && &op_mregs[0] == dst;
    counter_branch_op_addend(*addr, *dst, *k) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Oaddlimm(k), op_mregs, dst)),
        if op_mregs.len() == 1 && &op_mregs[0] == dst;

    rtl_inst_candidate(synth_addr, icond) <--
        counter_branch_op_addend(addr, dst, k),
        ltl_inst(addr, ?LTLInst::Lcond(cond, cond_mregs, target_true, target_false)),
        if cond_mregs.len() == 1 && &cond_mregs[0] == dst,
        if let Some(new_cond) = counter_branch_shifted_cond(cond, *k),
        cond_args_collected(addr, args),
        let synth_addr = *addr | (1u64 << 62),
        let icond = RTLInst::Icond(new_cond, args.clone(), target_true.clone(), target_false.clone());

    // Edge surgery for the split: route addr -> synth -> the branch's two targets, removing addr's direct branch edges.
    rtl_edge_negated(addr, dst) <--
        counter_branch_split_addr(addr),
        rtl_next(addr, dst);
    rtl_succ_candidate(addr, synth_addr), instr_in_function(synth_addr, func_start) <--
        counter_branch_split_addr(addr),
        instr_in_function(addr, func_start),
        let synth_addr = *addr | (1u64 << 62);
    rtl_succ_candidate(synth_addr, dst), instr_in_function(synth_addr, func_start) <--
        counter_branch_split_addr(addr),
        instr_in_function(addr, func_start),
        rtl_next(addr, dst),
        let synth_addr = *addr | (1u64 << 62);

    // A CMP with a spilled-scalar stack slot folds into one Icond at the cmp address reading the slot's canonical SSA reg; both BP and SP bases, since -O2 spills loop bounds SP-relative.
    rtl_inst_candidate(addr, inst) <--
        pcmp(addr, sym1, sym2),
        op_register(sym1, reg_str),
        let reg1 = Mreg::x86(reg_str.to_string()),
        op_indirect(sym2, reg_1, reg_2, reg_3, mult, disp, sz),
        if *reg_2 == "RBP" || *reg_2 == "RSP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2)),
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *sz),
        stack_var(func_start, addr, *disp, arg_rtl),
        reg_rtl(addr, reg1, reg_rtl2),
        let inst = RTLInst::Icond(cond, Arc::new(vec![*reg_rtl2, *arg_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough));

    rtl_inst_candidate(addr, inst) <--
        pcmp(addr, sym1, sym2),
        op_indirect(sym1, reg_1, reg_2, reg_3, mult, disp, sz),
        if *reg_2 == "RBP" || *reg_2 == "RSP",
        op_register(sym2, reg_str),
        let reg1 = Mreg::x86(reg_str.to_string()),
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2)),
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *sz),
        stack_var(func_start, addr, *disp, arg_rtl),
        reg_rtl(addr, reg1, reg_rtl2),
        let inst = RTLInst::Icond(cond, Arc::new(vec![*arg_rtl, *reg_rtl2]), Either::Right(*target_addr), Either::Right(*fallthrough));

    rtl_inst_candidate(addr, inst) <--
        pcmp(addr, sym1, sym2),
        op_indirect(sym1, reg_1a, reg_2a, reg_3a, mult_a, disp_a, sz_a),
        if *reg_2a == "RBP" || *reg_2a == "RSP",
        op_indirect(sym2, reg_1, reg_2, reg_3, mult, disp, sz),
        if *reg_2 == "RBP" || *reg_2 == "RSP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2a)),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2)),
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *sz),
        stack_var(func_start, addr, *disp_a, arg_rtl1),
        stack_var(func_start, addr, *disp, arg_rtl2),
        let inst = RTLInst::Icond(cond, Arc::new(vec![*arg_rtl1, *arg_rtl2]), Either::Right(*target_addr), Either::Right(*fallthrough));

    rtl_inst_candidate(addr, inst) <--
        pcmp(addr, sym1, sym2),
        op_indirect(sym1, reg_1, reg_2, reg_3, mult, disp, sz),
        if *reg_2 == "RBP" || *reg_2 == "RSP",
        op_immediate(sym2, imm_val, _),
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2)),
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *sz),
        let cond_with_imm = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            _ => sized_cond.clone(),
        },
        stack_var(func_start, addr, *disp, arg_rtl),
        let inst = RTLInst::Icond(cond_with_imm, Arc::new(vec![*arg_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough));


    rtl_inst_candidate(addr, inst) <--
        pcmp(addr, sym1, sym2),
        op_immediate(sym1, imm_val, _),
        op_indirect(sym2, reg_1, reg_2, reg_3, mult, disp, sz),
        if *reg_2 == "RBP" || *reg_2 == "RSP",
        instr_in_function(addr, func_start),
        stack_claim_base_at(func_start, addr, Mreg::x86(*reg_2)),
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *sz),
        let cond_with_imm = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm_val),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm_val),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm_val),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm_val),
            _ => sized_cond.clone(),
        },
        stack_var(func_start, addr, *disp, arg_rtl),
        let inst = RTLInst::Icond(cond_with_imm, Arc::new(vec![*arg_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough));


    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Ljumptable(arg_reg, targets)),
        reg_rtl(addr, *arg_reg, arg_rtl),
        let inst = RTLInst::Ijumptable(*arg_rtl, Arc::new(targets.clone()));


    // Direct def-use: the return value's defining instruction reaches the return.
    relation ax_value_direct(Address, Address);
    ax_value_direct(ret_addr, def_addr) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        reg_def_used(def_addr, Mreg::AX, ret_addr);

    // Availability-based void detection: any AX def reaching a return looks like a return value, so detect genuine void functions as those with a return reachable on a path setting no real value.

    // A return-value AX def: an AX def-site NOT consumed by a non-return use, excluding compare-operand loads and intermediate computations while keeping mov-imm/arithmetic/used-call results; the dead-call-then-return shape is resolved by the control-flow availability below.
    relation ltl_is_return(Address);
    ltl_is_return(addr) <--
        ltl_inst(addr, ?LTLInst::Lreturn);

    // A flag-only compare use: Lcond reads its operands purely to set flags and branch (no dst Mreg, no store), so a value merely tested by a loop-exit / null-check cmp is NOT redefined nor stored and must still count as a return value.
    #[local] relation flag_only_compare_use(Address);
    flag_only_compare_use(addr) <-- ltl_inst(addr, ?LTLInst::Lcond(_, _, _, _));

    relation ax_def_consumed_nonret(Address);
    ax_def_consumed_nonret(def_addr) <--
        reg_def_used(def_addr, Mreg::AX, use_addr),
        !ltl_is_return(use_addr),
        !flag_only_compare_use(use_addr);

    // Base return-value AX def: an AX def-site whose only consumer is the return; split out so the void-callee analysis can build on it without a negation cycle with the final ax_retval_def (which subtracts void-callee calls from this base).
    relation ax_retval_def_base(Address);
    ax_retval_def_base(def_addr) <--
        reg_def_site(def_addr, Mreg::AX),
        !ax_def_consumed_nonret(def_addr);

    // Void-callee return-value exclusion: a call whose callee leaves no value in AX is a dead clobber, so a caller whose only value source is that clobber is itself void.

    // A function genuinely produces a return value when a real AX/X0 value source reaches one of its returns; structural, so it can gate the void-callee exclusion below.
    relation func_produces_retval(Address);
    // (a) a non-call return-value AX def reaches a return (mov-imm / arithmetic / load result).
    func_produces_retval(func_start) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        ax_retval_def_base(def_addr),
        !is_call_or_tailcall(def_addr),
        reg_def_used(def_addr, Mreg::AX, ret_addr);
    // (b) a float/double X0 VALUE def reaches a return (call X0 clobbers are events, not values - see x0_value_write).
    func_produces_retval(func_start) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        x0_value_write(def_addr);
    // (c) the function forwards a local callee's result and that callee itself produces a value (positive recursion; keeps the SCC negation-free and thus stratifiable).
    func_produces_retval(func_start) <--
        instr_in_function(call_addr, func_start),
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(callee)))),
        ax_retval_def_base(call_addr),
        reg_def_used(call_addr, Mreg::AX, ret_addr),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        func_produces_retval(callee);
    // (d) the function forwards an extern/indirect callee's result; the callee return type is unknown here, so conservatively assume a value (never over-void unknown callees).
    relation local_addr_call(Address);
    local_addr_call(call_addr) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(_))));
    local_addr_call(call_addr) <--
        ltl_inst(call_addr, ?LTLInst::Ltailcall(Either::Right(Either::Left(_))));
    func_produces_retval(func_start) <--
        instr_in_function(call_addr, func_start),
        ax_retval_def_base(call_addr),
        is_call_or_tailcall(call_addr),
        !local_addr_call(call_addr),
        reg_def_used(call_addr, Mreg::AX, ret_addr),
        ltl_inst(ret_addr, ?LTLInst::Lreturn);

    // A call site whose callee is a discovered local function that produces no return value.
    relation void_callee_call(Address);
    void_callee_call(call_addr) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(callee)))),
        func_stacksz(callee, _, _, _),
        !func_produces_retval(callee);

    relation ax_retval_def(Address);
    ax_retval_def(def_addr) <--
        ax_retval_def_base(def_addr),
        !void_callee_call(def_addr);

    relation block_has_ax_retval_def(Address);
    block_has_ax_retval_def(blk) <--
        ax_retval_def(def_addr),
        code_in_block(def_addr, blk);

    // Forward may-undef over the block CFG: a block entered on some path with no return-value AX def set yet, seeded at each discovered function's entry block.
    relation ax_retval_undef_at_block(Address);
    ax_retval_undef_at_block(entry_blk) <--
        func_stacksz(func_start, _, _, _),
        code_in_block(func_start, entry_blk);
    ax_retval_undef_at_block(succ_blk) <--
        ax_retval_undef_at_block(blk),
        !block_has_ax_retval_def(blk),
        asm_block_next(blk, succ_node),
        code_in_block(succ_node, succ_blk);

    // A function has a void return path if some return is reached with no return-value AX def available (undef at the return's block entry and its own block sets none).
    relation func_void_avail(Address);
    func_void_avail(func_start) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        code_in_block(ret_addr, ret_blk),
        ax_retval_undef_at_block(ret_blk),
        !block_has_ax_retval_def(ret_blk),
        // Do not declare void when a genuine AX value def structurally reaches a return: a value read by a non-return use must still be returned, and func_ax_def_reaches_return stratifies here.
        !func_ax_def_reaches_return(func_start);

    // Gate the return-value binding on the function not being void: a void function has no ax_value_addr, so it gets no function_return_point_reg / has-return and its returns lower to the void Ireturn path; identical to the established void rendering.
    ax_value_addr(ret_addr, def_addr) <--
        ax_value_direct(ret_addr, def_addr),
        instr_in_function(ret_addr, vfunc),
        !func_void_avail(vfunc);

    func_ax_def(func_start, addr, rtl_reg) <--
        instr_in_function(addr, func_start),
        reg_def_site(addr, Mreg::AX),
        reg_rtl(addr, Mreg::AX, rtl_reg);

    // Recovery when the return def-use is broken: bind an AX def that STRUCTURALLY reaches it, replacing the entry-order magnitude proxy that matched defs on disjoint branches.
    relation ax_recovered_cand(Address, Address, Address);
    ax_recovered_cand(func_start, ret_addr, def_addr) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        !ax_value_direct(ret_addr, _),
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        func_ax_def(func_start, def_addr, _);

    // Several defs can reach one return, so keep one deterministic representative (max def address); this is a tie-break among structurally-proven candidates, not a reachability decision.
    relation ax_recovered_killed(Address, Address, Address);
    ax_recovered_killed(func_start, ret_addr, def_addr) <--
        ax_recovered_cand(func_start, ret_addr, def_addr),
        ax_recovered_cand(func_start, ret_addr, other_def),
        if *other_def > *def_addr;

    relation ax_recovered(Address, Address, Address);
    ax_recovered(func_start, ret_addr, def_addr) <--
        ax_recovered_cand(func_start, ret_addr, def_addr),
        !ax_recovered_killed(func_start, ret_addr, def_addr);

    ax_value_addr(ret_addr, def_addr) <--
        ax_recovered(_, ret_addr, def_addr),
        instr_in_function(ret_addr, vfunc),
        !func_void_avail(vfunc);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lreturn),
        !func_returns_float_at(addr),
        ax_value_addr(addr, axaddr),
        reg_rtl(addr, Mreg::AX, ret_rtl),
        let inst = RTLInst::Ireturn(*ret_rtl);

    // Float/double return: bind Ireturn to the real XMM0 def (mirror of the AX rule above).
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lreturn),
        x0_value_addr(addr, x0addr),
        reg_rtl(addr, Mreg::X0, ret_rtl),
        let inst = RTLInst::Ireturn(*ret_rtl);

    // Recovery return: the return node has no propagated AX rtl reg, so build Ireturn over the recovered def's reg_rtl directly (consistent with function_return_point_reg).
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lreturn),
        ax_recovered(_, addr, def_addr),
        !reg_rtl(addr, Mreg::AX, _),
        reg_rtl(def_addr, Mreg::AX, ret_rtl),
        let inst = RTLInst::Ireturn(*ret_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lreturn),
        !ax_value_addr(addr, _),
        !x0_value_addr(addr, _),
        let void_reg = crate::decompile::passes::rtl_pass::fresh_xtl_reg(*addr, Mreg::AX),
        let inst = RTLInst::Ireturn(void_reg);

    // An incoming stack-argument load binds its dst to the function's
    // synthetic stack-param reg at the actual ABI ordinal; otherwise the
    // Lgetstack fallback fabricates an uninitialized local.
    #[local] relation lgetstack_is_stack_param(Node);
    lgetstack_is_stack_param(addr) <--
        stack_param_load(addr, func_start, disp, _, _);

    rtl_inst_candidate(addr, inst) <--
        stack_param_load(addr, func_start, _disp, idx, _),
        ltl_inst(addr, ?LTLInst::Lgetstack(_slot, _ofs, _typ, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_reg]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        stack_param_load(addr, func_start, _disp, idx, _),
        ltl_inst(addr, ?LTLInst::Lgetstack(_slot, _ofs, _typ, dst)),
        !reg_rtl(addr, dst, _),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_reg]), fresh_dst);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, ofs, rtl_reg),
        !lgetstack_is_stack_param(addr),
        !win64_home_reload(addr, _, _, _),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*rtl_reg]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        instr_in_function(addr, func_start),
        !stack_var(func_start, addr, ofs, _),
        !lgetstack_is_stack_param(addr),
        !win64_home_reload(addr, _, _, _),
        let fresh_src = fresh_stack_cell_reg(*addr),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![fresh_src]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        !reg_rtl(addr, dst, _),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, ofs, rtl_reg),
        !lgetstack_is_stack_param(addr),
        !win64_home_reload(addr, _, _, _),
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*rtl_reg]), fresh_dst);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        !reg_rtl(addr, dst, _),
        instr_in_function(addr, func_start),
        !stack_var(func_start, addr, ofs, _),
        !lgetstack_is_stack_param(addr),
        !win64_home_reload(addr, _, _, _),
        let fresh_src = fresh_stack_cell_reg(*addr),
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![fresh_src]), fresh_dst);

    rtl_inst_candidate(addr, inst) <--
        win64_home_reload(addr, func_start, param_mreg, _),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        let param_rtl = fresh_xtl_reg(*func_start, *param_mreg),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        win64_home_reload(addr, func_start, param_mreg, _),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, dst)),
        !reg_rtl(addr, dst, _),
        let param_rtl = fresh_xtl_reg(*func_start, *param_mreg),
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_rtl]), fresh_dst);

    // /homeparams stores are ABI bookkeeping. Their reloads above read the
    // entry parameter directly, so retaining a local-slot assignment would
    // fabricate an address-valued frame object.
    rtl_inst_candidate(addr, nop) <--
        win64_home_spill(addr, _, _, _),
        let nop = RTLInst::Inop;

    // Read the spill SOURCE id at the DEF site, but bind the load's def id at a load_overwrites_base site, or the ADDRESS is spilled instead of the loaded value and the load's real def is DSE'd.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        reg_def_used(defaddr, *src, *addr),
        !load_overwrites_base(defaddr, src),
        reg_rtl(defaddr, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        reg_def_used(defaddr, *src, *addr),
        load_overwrites_base(defaddr, src),
        is_def(defaddr, def_xtl),
        reg_xtl(defaddr, *src, def_xtl),
        xtl_canonical(def_xtl, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        !reg_def_used(_, src, addr),
        is_arg_reg(src),
        func_param_validated(func_start, src),
        reg_rtl(func_start, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        reg_rtl(addr, *src, src_rtl),
        !stack_var(func_start, addr, ofs, _),
        let fresh_stack = fresh_stack_cell_reg(*addr),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), fresh_stack);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        !reg_def_used(_, src, addr),
        is_arg_reg(src),
        !func_param_validated(func_start, src),
        reg_rtl(addr, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        !win64_home_spill(addr, _, _, _),
        instr_in_function(addr, func_start),
        !reg_def_used(_, src, addr),
        !is_arg_reg(src),
        reg_rtl(addr, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);


    rtl_inst_candidate(addr, inst), reg_xtl(addr, dst_mreg, dst_rtl), op_produces_ptr(addr, dst_rtl) <--
        plea(addr, dst_sym, src_addr),
        op_register(dst_sym, dst_str),
        op_indirect(src_addr, _, base_str, _, _scale, disp, _),
        if *base_str == "RBP",
        real_addr_in_func(addr, func_start),
        bp_base_at(func_start, addr, _),
        let dst_mreg = Mreg::x86(dst_str.to_string()),
        let dst_rtl = fresh_xtl_reg(*addr, dst_mreg),
        let inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*disp)), Arc::new(vec![]), dst_rtl);

    ltl_inst(addr, ltl) <--
        plea(addr, dst_sym, src_addr),
        op_register(dst_sym, dst_str),
        op_indirect(src_addr, _, base_str, _, _scale, disp, _),
        if *base_str == "RBP",
        real_addr_in_func(addr, func_start),
        bp_base_at(func_start, addr, _),
        let dst_mreg = Mreg::x86(dst_str.to_string()),
        let ltl = LTLInst::Lop(Operation::Olea(Addressing::Ainstack(*disp)), Arc::new(vec![]), dst_mreg);


    stack_mem_add_imm(addr, disp, imm_val, sz) <--
        padd(addr, dst_sym, src_sym),
        op_indirect(dst_sym, _, base_str, _, _, disp, sz),
        if *base_str == "RBP",
        real_addr_in_func(addr, func_start),
        bp_base_at(func_start, addr, _),
        op_immediate(src_sym, imm_val, _);

    stack_mem_sub_imm(addr, disp, imm_val, sz) <--
        psub(addr, dst_sym, src_sym),
        op_indirect(dst_sym, _, base_str, _, _, disp, sz),
        if *base_str == "RBP",
        real_addr_in_func(addr, func_start),
        bp_base_at(func_start, addr, _),
        op_immediate(src_sym, imm_val, _);

    stack_xtl(func_start, addr, disp, rtl_reg) <--
        stack_mem_add_imm(addr, disp, _, _),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_stack_cell_reg(*addr);

    stack_xtl(func_start, addr, disp, rtl_reg) <--
        stack_mem_sub_imm(addr, disp, _, _),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_stack_cell_reg(*addr);

    stack_xtl(func_start, addr, ofs, rtl_reg) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(_, _, ofs, _)),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_stack_cell_reg(*addr);

    rtl_inst_candidate(addr, inst) <--
        stack_mem_add_imm(addr, disp, imm_val, sz),
        if *sz == 8,
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        let inst = RTLInst::Iop(Operation::Oaddlimm(*imm_val), Arc::new(vec![*stack_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        stack_mem_add_imm(addr, disp, imm_val, sz),
        if *sz == 4,
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        let inst = RTLInst::Iop(Operation::Oaddimm(*imm_val), Arc::new(vec![*stack_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        stack_mem_sub_imm(addr, disp, imm_val, sz),
        if *sz == 8,
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        let inst = RTLInst::Iop(Operation::Oaddlimm(-*imm_val), Arc::new(vec![*stack_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        stack_mem_sub_imm(addr, disp, imm_val, sz),
        if *sz == 4,
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        let inst = RTLInst::Iop(Operation::Oaddimm(-*imm_val), Arc::new(vec![*stack_rtl]), *stack_rtl);


    relation reg_xtl(Node, Mreg, RTLReg);
    relation reg_def_site(Node, Mreg);
    relation is_def(Node, RTLReg);
    relation reg_rtl(Node, Mreg, RTLReg);

    relation alias_edge(RTLReg, RTLReg);

    #[local] lattice xtl_canonical_lat(RTLReg, ascent::Dual<RTLReg>);

    // Seed: each id maps to itself
    xtl_canonical_lat(a, ascent::Dual(*a)) <-- reg_xtl(_, _, a);
    xtl_canonical_lat(a, ascent::Dual(*a)) <-- is_def(_, a);
    xtl_canonical_lat(a, ascent::Dual(*a)) <-- stack_xtl(_, _, _, a);
    xtl_canonical_lat(a, ascent::Dual(*a)) <-- alias_edge(a, _);
    xtl_canonical_lat(b, ascent::Dual(*b)) <-- alias_edge(_, b);

    // Propagate min canonical through alias edges
    xtl_canonical_lat(b, *canonical) <-- alias_edge(a, b), xtl_canonical_lat(a, canonical);
    xtl_canonical_lat(a, *canonical) <-- alias_edge(a, b), xtl_canonical_lat(b, canonical);

    // Project lattice to regular relation
    relation xtl_canonical(RTLReg, RTLReg);
    xtl_canonical(id, canonical.0) <-- xtl_canonical_lat(id, canonical);


    #[local] relation param_reg_used_early(Address, Mreg);

    relation return_val_used(Address, Address, Mreg, Address, i64);
    relation call_return_reg(Node, RTLReg);


    relation asm_effective_def(Address, Mreg);
    asm_effective_def(addr, reg) <-- reg_def(addr, reg), !trim_instruction(addr);
    asm_effective_def(addr, Mreg::AX) <--
        unrefinedinstruction(addr, _, _, "CALL", _, _, _, _, _, _);

    asm_effective_def(addr, reg) <--
        unrefinedinstruction(addr, _, _, "CALL", _, _, _, _, _, _),
        is_caller_saved(reg);

    // Lbuiltin results are effective defs even at trimmed addresses (e.g. VLA alloca replaces trimmed MOV RSP)
    asm_effective_def(addr, *dst_reg) <--
        ltl_inst(addr, ?LTLInst::Lbuiltin(_, _, BuiltinArg::BA(dst_reg))),
        if *dst_reg != Mreg::Unknown;

    // A def-EVENT: every register write INCLUDING trimmed instructions, so older defs are not spliced through a trimmed site; value-consuming rules additionally require asm_effective_def.
    relation asm_def_kill(Address, Mreg);
    asm_def_kill(addr, reg) <-- reg_def(addr, reg);
    asm_def_kill(addr, reg) <-- asm_effective_def(addr, reg);

    reg_use(addr, Mreg::AX) <--
        ltl_inst(addr, ?LTLInst::Lreturn);

    // Float (XMM0) returns are unseen by the AX-keyed machinery, so a function returns float iff some return is reached by an in-function X0 def and no AX value reaches any return.

    // X0 VALUE writes: an Lop/Lload/Lgetstack producing X0, or a call whose callee actually returns float; a bare call's caller-saved X0 clobber is an EVENT, not a value.
    #[local] relation x0_value_write(Address);
    x0_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lop(_, _, Mreg::X0));
    x0_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lload(_, _, _, Mreg::X0));
    x0_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, Mreg::X0));
    x0_value_write(addr) <--
        ltl_inst(addr, ?LTLInst::Lcall(Either::Right(Either::Left(callee)))),
        func_returns_float(callee);
    x0_value_write(addr) <--
        external_call_site(addr, _, name),
        known_extern_signature(name, _, ret, _),
        if matches!(ret, XType::Xfloat | XType::Xsingle);
    // 3.8 R1: a FUSED float op writing X0 is a genuine X0 VALUE write; it has no Lop/Lload, so func_returns_float stayed false and DCE collapsed the body to return 0.
    x0_value_write(addr) <--
        float_load_op(addr, _, _, _, _, Mreg::X0, _),
        !has_ltl_op(addr);
    x0_value_write(addr) <--
        float_arith_stack_op(addr, _, _, _, Mreg::X0),
        !has_ltl_op(addr);

    relation func_x0_def_reaches_return(Address);
    func_x0_def_reaches_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        x0_value_write(def_addr);

    // An AX def-site reaching a return means an int/ptr/bool result in AX, not a float return; structural (reg_def_site only) so it stays OUT of the reg_def_used/reg_use SCC.
    #[local] relation ax_value_write(Address);
    ax_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lop(_, _, Mreg::AX));
    ax_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lload(_, _, _, Mreg::AX));
    ax_value_write(addr) <-- ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, Mreg::AX));
    // A fused integer memory operation has no Lop/Lload, but its destination
    // is still a genuine AX value definition. Without this evidence a value
    // that is both consumed in the body and returned makes func_void_avail
    // misclassify the function as void.
    ax_value_write(addr) <--
        float_load_op(addr, _, _, _, _, Mreg::AX, _),
        !has_ltl_op(addr),
        !unsupported_stack_address_seed(_, addr, _);

    // An AX def that is an internal float->int conversion is a SCRATCH integer, not the return value, so excluding it stops it lighting func_ax_def_reaches_return and suppressing func_returns_float.
    #[local] relation ax_conv_to_int_def(Address);
    ax_conv_to_int_def(addr) <-- ltl_inst(addr, ?LTLInst::Lop(Operation::Ointofsingle, _, Mreg::AX));
    ax_conv_to_int_def(addr) <-- ltl_inst(addr, ?LTLInst::Lop(Operation::Ointoffloat, _, Mreg::AX));

    // An AX def that is a verbatim COPY of an XMM value is gcc's ABI/NaN-canonicalization bitcast holding raw float bits, not an int return, so it is likewise disqualified.
    #[local] relation ax_float_shuffle_def(Address);
    ax_float_shuffle_def(addr) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Omove, args, Mreg::AX)),
        if args.len() == 1,
        if is_float_mreg(&args[0]);

    // Positive, NON-recursive X0-value-reaches-return signal using only direct X0 writes, deliberately excluding the recursive Lcall arm so the negation below stays stratified.
    #[local] relation x0_value_write_direct(Address);
    x0_value_write_direct(addr) <-- ltl_inst(addr, ?LTLInst::Lop(_, _, Mreg::X0));
    x0_value_write_direct(addr) <-- ltl_inst(addr, ?LTLInst::Lload(_, _, _, Mreg::X0));
    x0_value_write_direct(addr) <-- ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, Mreg::X0));
    x0_value_write_direct(addr) <--
        float_load_op(addr, _, _, _, _, Mreg::X0, _),
        !has_ltl_op(addr);
    x0_value_write_direct(addr) <--
        float_arith_stack_op(addr, _, _, _, Mreg::X0),
        !has_ltl_op(addr);
    relation func_x0_value_reaches_return(Address);
    func_x0_value_reaches_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        x0_value_write_direct(def_addr);

    // An AX def reaching a return that is DISQUALIFIED as the int return: an internal float->int conversion in a function that genuinely returns a float in X0.
    relation ax_return_def_disqualified(Address, Address);
    ax_return_def_disqualified(func_start, *def_addr) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_conv_to_int_def(def_addr),
        func_x0_value_reaches_return(func_start);

    // The AX reaching the ret is a float-bits bitcast, not the int return, so a double-returning function whose epilogue shuffles X0 through AX is still classified float.
    ax_return_def_disqualified(func_start, *def_addr) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_float_shuffle_def(def_addr),
        func_x0_value_reaches_return(func_start);

    // The return's OWN block computes a float VALUE: an X0 value def is the block-last X0 def at the return, a structural in-block dominance fact, not an address comparison.
    #[local] relation func_x0_inblock_return(Address);
    func_x0_inblock_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        block_last_def(ret_addr, x0_def, Mreg::X0),
        x0_value_write_direct(x0_def);

    // The return's OWN block computes an int/ptr VALUE: an ax_value_write def is the block-last AX def at the return.
    #[local] relation func_ax_inblock_return(Address);
    func_ax_inblock_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        block_last_def(ret_addr, ax_def, Mreg::AX),
        ax_value_write(ax_def);

    // Disqualify a reaching AX def when the return's block computes a float and no int value, so a stale cross-block loop counter cannot suppress a float-accumulator return.
    ax_return_def_disqualified(func_start, *def_addr) <--
        func_x0_inblock_return(func_start),
        !func_ax_inblock_return(func_start),
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_value_write(def_addr);

    // The block-last AX reaching the ret is an ADDRESS when its value is read as a load/store base; built SCC-safe from block_last_def and gated on X0 genuinely carrying a float to the ret.
    ltl_inst_reads_addr_base(addr, base_mreg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, lddr, args, _)),
        if matches!(lddr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0];
    ltl_inst_reads_addr_base(addr, base_mreg) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, sddr, args, _)),
        if matches!(sddr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0];

    #[local] relation ax_def_used_as_addr_base(Address);
    ax_def_used_as_addr_base(*def_addr) <--
        ax_value_write(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_reads_addr_base(use_addr, Mreg::AX),
        block_last_def(use_addr, def_addr, Mreg::AX);
    ax_def_used_as_addr_base(*def_addr) <--
        ax_value_write(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_reads_addr_base(use_addr, Mreg::AX),
        code_in_block(use_addr, blk),
        instr_in_function(use_addr, func),
        !block_last_def(use_addr, _, Mreg::AX),
        reaching_def_in(func, blk, Mreg::AX, s),
        if s.0.contains(def_addr);

    ax_return_def_disqualified(func_start, *def_addr) <--
        func_x0_value_reaches_return(func_start),
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_def_used_as_addr_base(def_addr);

    // A call's AX result genuinely live in the body, consumed by a real instruction operand; built from ltl_inst_uses_mreg plus block_last_def so it stays out of the func_returns_float SCC.
    #[local] relation ax_call_result_used_in_body(Address);
    ax_call_result_used_in_body(*def_addr) <--
        is_call_or_tailcall(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_uses_mreg(use_addr, Mreg::AX),
        !ltl_is_return(use_addr),
        !flag_only_compare_use(use_addr),
        block_last_def(use_addr, def_addr, Mreg::AX);
    ax_call_result_used_in_body(*def_addr) <--
        is_call_or_tailcall(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_uses_mreg(use_addr, Mreg::AX),
        !ltl_is_return(use_addr),
        !flag_only_compare_use(use_addr),
        code_in_block(use_addr, blk),
        instr_in_function(use_addr, func),
        !block_last_def(use_addr, _, Mreg::AX),
        reaching_def_in(func, blk, Mreg::AX, s),
        if s.0.contains(def_addr);

    // A call result consumed ONLY by a flag-only compare is bound as a returned value so the compare operand resolves, but kept SEPARATE so it does not feed func_ax_def_reaches_return or void detection.
    #[local] relation ax_call_result_used_in_cond(Address);
    ax_call_result_used_in_cond(*def_addr) <--
        is_call_or_tailcall(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_uses_mreg(use_addr, Mreg::AX),
        flag_only_compare_use(use_addr),
        block_last_def(use_addr, def_addr, Mreg::AX);
    ax_call_result_used_in_cond(*def_addr) <--
        is_call_or_tailcall(def_addr),
        asm_effective_def(def_addr, Mreg::AX),
        ltl_inst_uses_mreg(use_addr, Mreg::AX),
        flag_only_compare_use(use_addr),
        code_in_block(use_addr, blk),
        instr_in_function(use_addr, func),
        !block_last_def(use_addr, _, Mreg::AX),
        reaching_def_in(func, blk, Mreg::AX, s),
        if s.0.contains(def_addr);

    call_returns_value(*call_addr, Mreg::AX) <--
        ax_call_result_used_in_cond(call_addr);

    relation func_ax_def_reaches_return(Address);
    func_ax_def_reaches_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_value_write(def_addr),
        !ax_return_def_disqualified(func_start, def_addr);
    func_ax_def_reaches_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ax_call_result_used_in_body(def_addr);

    // The X0 def consumed by an X0-to-memory store (intra-block last def, else lattice membership at the store's block) - same def-reaches-point shape as def_reaches_return/def_reaches_call.
    #[local] relation x0_store_consumed_def(Address, Address);
    x0_store_consumed_def(*st_addr, *def_addr) <--
        ltl_inst(st_addr, ?LTLInst::Lstore(_, _, _, st_src)),
        if *st_src == Mreg::X0,
        block_last_def(st_addr, def_addr, Mreg::X0),
        asm_effective_def(def_addr, Mreg::X0);
    x0_store_consumed_def(*st_addr, *def_addr) <--
        ltl_inst(st_addr, ?LTLInst::Lstore(_, _, _, st_src)),
        if *st_src == Mreg::X0,
        code_in_block(st_addr, blk),
        instr_in_function(st_addr, func),
        !block_last_def(st_addr, _, Mreg::X0),
        reaching_def_in(func, blk, Mreg::X0, s),
        for def_addr in s.0.iter();
    // Stack-slot store form: SIMD struct copies lift as Lgetstack(X0)/Lsetstack(X0) pairs, so the Lstore-only match missed them and the X0 temp was misread as a float return value the moment no AX value reached the return.
    x0_store_consumed_def(*st_addr, *def_addr) <--
        ltl_inst(st_addr, ?LTLInst::Lsetstack(st_src, _, _, _)),
        if *st_src == Mreg::X0,
        block_last_def(st_addr, def_addr, Mreg::X0),
        asm_effective_def(def_addr, Mreg::X0);
    x0_store_consumed_def(*st_addr, *def_addr) <--
        ltl_inst(st_addr, ?LTLInst::Lsetstack(st_src, _, _, _)),
        if *st_src == Mreg::X0,
        code_in_block(st_addr, blk),
        instr_in_function(st_addr, func),
        !block_last_def(st_addr, _, Mreg::X0),
        reaching_def_in(func, blk, Mreg::X0, s),
        for def_addr in s.0.iter();

    // An X0 arriving at a return that was also stored to memory is a computed temp, not a float return; the SAME def feeds both, replacing an order proxy that matched unrelated stores.
    relation func_x0_stored_before_return(Address);
    func_x0_stored_before_return(func_start) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        x0_store_consumed_def(st_addr, def_addr),
        instr_in_function(st_addr, func_start);

    relation func_returns_float(Address);
    func_returns_float(func_start) <--
        func_x0_def_reaches_return(func_start),
        !func_ax_def_reaches_return(func_start),
        !func_x0_stored_before_return(func_start);

    // Per-return-node projection of func_returns_float, used to suppress the AX Ireturn and return-point binding since the value is in X0.
    relation func_returns_float_at(Address);
    func_returns_float_at(ret_addr) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        instr_in_function(ret_addr, func_start),
        func_returns_float(func_start);

    // X0 return reg_use, gated to float-returning functions so int functions are unaffected.
    reg_use(addr, Mreg::X0) <--
        ltl_inst(addr, ?LTLInst::Lreturn),
        instr_in_function(addr, func_start),
        func_returns_float(func_start);

    relation x0_value_direct(Address, Address);
    x0_value_direct(ret_addr, def_addr) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        reg_def_used(def_addr, Mreg::X0, ret_addr);

    relation x0_value_addr(Address, Address);
    x0_value_addr(ret_addr, def_addr) <--
        x0_value_direct(ret_addr, def_addr);

    relation block_last_insn(Address, Address);
    block_last_insn(block_start, last_insn) <--
        block_boundaries(block_start, last_insn, _);

    relation asm_block_next(Address, Address);
    asm_block_next(src_block, dst) <--
        ddisasm_cfg_edge(src, dst, edge_type),
        if *edge_type != "call" && *edge_type != "indirect" && *edge_type != "indirect_call",
        code_in_block(src, src_block);

    // block_last_def: the last def-EVENT for r strictly before u in u's block, chaining on asm_def_kill so a trimmed def stops older defs; value consumers must also check asm_effective_def.
    relation block_last_def(Address, Address, Mreg);

    block_last_def(next_addr, def_addr, reg) <--
        asm_def_kill(def_addr, reg),
        next(def_addr, next_addr),
        code_in_block(def_addr, blk),
        code_in_block(next_addr, blk);

    block_last_def(next_addr, def_addr, reg) <--
        block_last_def(curr_addr, def_addr, reg),
        !asm_def_kill(curr_addr, reg),
        next(curr_addr, next_addr),
        code_in_block(curr_addr, blk),
        code_in_block(next_addr, blk);

    relation last_def_in_block(Address, Address, Mreg);

    last_def_in_block(block, last_insn, reg) <--
        block_last_insn(block, last_insn),
        asm_effective_def(last_insn, reg);

    last_def_in_block(block, def_addr, reg) <--
        block_last_insn(block, last_insn),
        block_last_def(last_insn, def_addr, reg),
        asm_effective_def(def_addr, reg),
        !asm_def_kill(last_insn, reg);

    relation asm_defined_in_block(Address, Mreg);
    asm_defined_in_block(block, reg) <--
        asm_def_kill(ea, reg),
        code_in_block(ea, block);

    relation live_var_def(Address, Mreg, Address);
    live_var_def(block, reg, def_addr) <--
        last_def_in_block(block, def_addr, reg);

    relation live_var_used(Address, Mreg, Address);
    live_var_used(block, reg, use_addr) <--
        reg_use(use_addr, reg),
        !trim_instruction(use_addr),
        code_in_block(use_addr, block),
        !block_last_def(use_addr, _, reg);

    relation live_var_at_block_end(Address, Address, Mreg);

    live_var_at_block_end(prev_block, use_block, reg) <--
        live_var_used(use_block, reg, _),
        asm_block_next(prev_block, use_block);

    live_var_at_block_end(prev_block, use_block, reg) <--
        live_var_at_block_end(mid_block, use_block, reg),
        !asm_defined_in_block(mid_block, reg),
        asm_block_next(prev_block, mid_block);

    reg_def_used(def_addr, reg, use_addr) <--
        reg_use(use_addr, reg),
        !trim_instruction(use_addr),
        block_last_def(use_addr, def_addr, reg),
        asm_effective_def(def_addr, reg);

    reg_def_used(def_addr, reg, use_addr) <--
        live_var_at_block_end(def_block, use_block, reg),
        live_var_def(def_block, reg, def_addr),
        live_var_used(use_block, reg, use_addr);

    // Raw decoded-instruction reaching defs, bounded to the x86 GP register
    // file used by frame/copied-SP provenance. Every dependency below comes
    // from the AsmPass snapshots, so this chain cannot join RTL's augmented
    // reg_use/reg_def_used SCC.
    #[local] relation raw_stack_base_reg(Mreg);
    raw_stack_base_reg(Mreg::AX); raw_stack_base_reg(Mreg::BX);
    raw_stack_base_reg(Mreg::CX); raw_stack_base_reg(Mreg::DX);
    raw_stack_base_reg(Mreg::SI); raw_stack_base_reg(Mreg::DI);
    raw_stack_base_reg(Mreg::BP); raw_stack_base_reg(Mreg::SP);
    raw_stack_base_reg(Mreg::R8); raw_stack_base_reg(Mreg::R9);
    raw_stack_base_reg(Mreg::R10); raw_stack_base_reg(Mreg::R11);
    raw_stack_base_reg(Mreg::R12); raw_stack_base_reg(Mreg::R13);
    raw_stack_base_reg(Mreg::R14); raw_stack_base_reg(Mreg::R15);

    // Coordinate provenance is killed by a decoded write or an ABI call
    // clobber.  Only decoded writes seed a new coordinate below: a call kill
    // prevents an older copied-SP value from flowing through volatile R10/R11
    // without pretending the call produced a new stack base.
    #[local] relation raw_reg_kill(Address, Mreg);
    raw_reg_kill(addr, reg) <-- asm_reg_def(addr, reg), raw_stack_base_reg(reg);
    raw_reg_kill(addr, reg) <--
        unrefinedinstruction(addr, _, _, "CALL", _, _, _, _, _, _),
        is_caller_saved(reg),
        raw_stack_base_reg(reg);

    #[local] relation raw_block_kill_reaches(Address, Mreg);
    raw_block_kill_reaches(next_addr, *reg) <--
        raw_reg_kill(kill_addr, reg),
        !asm_reg_def(kill_addr, reg),
        next(kill_addr, next_addr),
        code_in_block(kill_addr, block),
        code_in_block(next_addr, block);
    raw_block_kill_reaches(next_addr, reg) <--
        raw_block_kill_reaches(cur_addr, reg),
        !asm_reg_def(cur_addr, reg),
        next(cur_addr, next_addr),
        code_in_block(cur_addr, block),
        code_in_block(next_addr, block);

    #[local] relation raw_block_last_def(Address, Address, Mreg);
    raw_block_last_def(next_addr, def_addr, *reg) <--
        asm_reg_def(def_addr, reg),
        raw_stack_base_reg(reg),
        next(def_addr, next_addr),
        code_in_block(def_addr, block),
        code_in_block(next_addr, block);
    raw_block_last_def(next_addr, def_addr, reg) <--
        raw_block_last_def(cur_addr, def_addr, reg),
        !raw_reg_kill(cur_addr, reg),
        next(cur_addr, next_addr),
        code_in_block(cur_addr, block),
        code_in_block(next_addr, block);

    #[local] relation raw_last_def_in_block(Address, Address, Mreg);
    raw_last_def_in_block(block, last_addr, *reg) <--
        block_last_insn(block, last_addr),
        asm_reg_def(last_addr, reg),
        raw_stack_base_reg(reg);
    raw_last_def_in_block(block, def_addr, reg) <--
        block_last_insn(block, last_addr),
        raw_block_last_def(last_addr, def_addr, reg),
        !raw_reg_kill(last_addr, reg);

    #[local] relation raw_reg_defined_in_block(Address, Mreg);
    raw_reg_defined_in_block(block, *reg) <--
        raw_reg_kill(addr, reg),
        raw_stack_base_reg(reg),
        code_in_block(addr, block);

    #[local] relation raw_live_var_def(Address, Mreg, Address);
    raw_live_var_def(block, reg, def_addr) <--
        raw_last_def_in_block(block, def_addr, reg);

    #[local] relation raw_live_var_used(Address, Mreg, Address);
    raw_live_var_used(block, *reg, use_addr) <--
        asm_reg_use(use_addr, reg),
        raw_stack_base_reg(reg),
        code_in_block(use_addr, block),
        !raw_block_last_def(use_addr, _, reg),
        !raw_block_kill_reaches(use_addr, reg);

    #[local] relation raw_live_var_at_block_end(Address, Address, Mreg);
    raw_live_var_at_block_end(prev_block, use_block, reg) <--
        raw_live_var_used(use_block, reg, _),
        asm_block_next(prev_block, use_block);
    raw_live_var_at_block_end(prev_block, use_block, reg) <--
        raw_live_var_at_block_end(mid_block, use_block, reg),
        !raw_reg_defined_in_block(mid_block, reg),
        asm_block_next(prev_block, mid_block);

    #[local] relation raw_reg_def_used(Address, Mreg, Address);
    raw_reg_def_used(def_addr, reg, use_addr) <--
        asm_reg_use(use_addr, reg),
        raw_stack_base_reg(reg),
        raw_block_last_def(use_addr, def_addr, reg),
        asm_reg_def(def_addr, reg);
    raw_reg_def_used(def_addr, reg, use_addr) <--
        raw_live_var_at_block_end(def_block, use_block, reg),
        raw_live_var_def(def_block, reg, def_addr),
        raw_live_var_used(use_block, reg, use_addr);

    param_reg_used_early(*func_start, *mreg) <--
        arg_reg_param_live_at(func_start, use_addr, mreg),
        is_arg_reg(mreg),
        reg_use(use_addr, mreg),
        !trim_instruction(use_addr);

    reg_def_used(*func_start, *mreg, *use_addr) <--
        param_reg_used_early(func_start, mreg),
        arg_reg_param_live_at(func_start, use_addr, mreg),
        is_arg_reg(mreg),
        reg_use(use_addr, mreg),
        !trim_instruction(use_addr);

    // (is_early_arg_use deleted: the entry-BFS-distance window is replaced by param_still_live / instr_in_function partitioning in the reg_xtl binding rules; distance must never decide which def binds a use.)

    // Float (XMM) parameter early reaching-def mirroring the integer edge, since is_arg_reg holds only int regs; gated on is_float_arg_reg to stay out of the validation negation cycle.
    #[local] relation float_param_reg_used_early(Address, Mreg);
    float_param_reg_used_early(*func_start, *mreg) <--
        arg_reg_param_live_at(func_start, use_addr, mreg),
        is_float_arg_reg(mreg),
        reg_use(use_addr, mreg),
        !trim_instruction(use_addr);

    reg_def_used(*func_start, *mreg, *use_addr) <--
        float_param_reg_used_early(func_start, mreg),
        arg_reg_param_live_at(func_start, use_addr, mreg),
        is_float_arg_reg(mreg),
        reg_use(use_addr, mreg),
        !trim_instruction(use_addr);

    relation stack_xtl(Address, Address, i64, RTLReg);
    relation stack_var(Address, Address, i64, RTLReg);

    stack_var(func_start, use_addr, ofs, canonical.0) <--
        stack_xtl(func_start, use_addr, ofs, xtl_id),
        xtl_canonical_lat(xtl_id, canonical);

    relation unrefinedinstruction(Address, usize, &'static str, &'static str, Symbol, Symbol, Symbol, Symbol, usize, usize);
    relation symbol_table(Address, usize, Symbol, Symbol, Symbol, usize, Symbol, usize, Symbol);
    relation op_indirect(Symbol, &'static str, &'static str, &'static str, i64, i64, usize);
    relation op_register(Symbol, &'static str);

    relation global_symbol(Address, Symbol);
    global_symbol(addr, name) <--
        symbol_table(addr, _size, _type, _binding, _section_type, _section_idx, _section_name, _name_idx, name),
        if *_binding == "GLOBAL" || *_binding == "LOCAL",
        if *_type == "OBJECT",
        if *addr > 0;

    relation rip_relative_access(Address, Size, i64);
    rip_relative_access(addr, size, disp) <--
        unrefinedinstruction(addr, size, _, _, op1, _, _, _, _, _),
        op_indirect(op1, _, base_reg, _, disp, _, _),
        if is_rip(base_reg);

    rip_relative_access(addr, size, disp) <--
        unrefinedinstruction(addr, size, _, _, _, op2, _, _, _, _),
        op_indirect(op2, _, base_reg, _, disp, _, _),
        if is_rip(base_reg);

    rip_relative_access(addr, size, disp) <--
        unrefinedinstruction(addr, size, _, _, op1, _, _, _, _, _),
        op_indirect(op1, _, base_reg, _, _, disp, _),
        if is_rip(base_reg);

    rip_relative_access(addr, size, disp) <--
        unrefinedinstruction(addr, size, _, _, _, op2, _, _, _, _),
        op_indirect(op2, _, base_reg, _, _, disp, _),
        if is_rip(base_reg);

    global_symbol(target_addr, name_sym) <--
        rip_relative_access(addr, size, disp),
        let target_addr = (*addr as i64 + *size as i64 + *disp) as Address,
        !symbol_table(target_addr, _, _, _, _, _, _, _, _),
        !plt_entry(target_addr, _),
        !func_span(_, target_addr, _),
        let name_string = format!("SUB_{:x}", target_addr),
        let name_sym = Box::leak(name_string.into_boxed_str()) as &'static str;

    symbols(addr, name, name) <--
        global_symbol(addr, name);

    relation stack_var_chunk(Address, i64, MemoryChunk);

    relation ltl_use(Node, Mreg);
    relation param_alias_source(Address, Mreg);

    reg_rtl(node, mreg, *canonical) <--
        reg_xtl(node, mreg, xtl_id),
        xtl_canonical(xtl_id, canonical);

    relation load_overwrites_base(Node, Mreg);
    load_overwrites_base(addr, *dst_reg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, dst_reg)),
        for src in args.iter(),
        if src == dst_reg;

    relation load_overwrite_use_id(Node, Mreg, RTLReg);
    load_overwrite_use_id(addr, *src, use_id) <--
        load_overwrites_base(addr, src),
        let use_id = fresh_xtl_reg(*addr, *src) | (1u64 << 62);

    reg_xtl(addr, *src, use_id) <--
        load_overwrite_use_id(addr, src, use_id);

    // Canonical value reaching a read operand.  reg_rtl intentionally also
    // contains the node-local fresh use id; joining on it directly can create
    // a Cartesian product of defined and undefined webs for indexed memory
    // operands (especially when one register is both index and store source).
    // Retained for the post-fixed-point canonical-home selector as well as
    // consumed inside this program.  Making this transient would force that
    // selector back onto ambiguous historical reg_rtl candidates.
    relation reaching_use_rtl(Node, Mreg, RTLReg);
    // A real prior definition contributes exactly its destination id.  The
    // other reg_xtl rows at that node are read-side ids and must not compete
    // with the value written by the instruction.
    reaching_use_rtl(use_addr, *mreg, *canonical) <--
        reg_def_used(def_addr, mreg, use_addr),
        if def_addr != use_addr,
        is_def(def_addr, def_id),
        reg_xtl(def_addr, *mreg, def_id),
        xtl_canonical(def_id, canonical);

    // Function-entry pseudo definitions are ABI live-ins, not instructions.
    // Require the ABI value to be live at this exact use; reg_def_used can
    // otherwise retain a stale func_start edge after a real definition.
    relation abi_livein_reaches_use(Address, Mreg, Address);
    abi_livein_reaches_use(*func_start, *mreg, *use_addr) <--
        reg_def_used(func_start, mreg, use_addr),
        func_param_validated(func_start, mreg),
        arg_reg_param_live_at(func_start, use_addr, mreg);
    abi_livein_reaches_use(*func_start, *mreg, *use_addr) <--
        reg_def_used(func_start, mreg, use_addr),
        func_float_param_validated(func_start, mreg, _),
        arg_reg_param_live_at(func_start, use_addr, mreg);
    // The first iteration of a loop still reads the incoming integer argument
    // even when the ordinary MUST-live walk is tainted by its back edge.
    abi_livein_reaches_use(*func_start, *mreg, *use_addr) <--
        reg_def_used(func_start, mreg, use_addr),
        arg_reg_param_live_at_fwd(func_start, use_addr, mreg),
        !arg_reg_param_live_at(func_start, use_addr, mreg);

    reaching_use_rtl(use_addr, *mreg, *canonical) <--
        abi_livein_reaches_use(func_start, mreg, use_addr),
        let param_id = fresh_xtl_reg(*func_start, *mreg),
        xtl_canonical(param_id, canonical);

    // A fused indexed read/modify/write can have a loop-carried reaching edge
    // from the instruction to itself.  That edge carries the previous
    // iteration's exact destination definition.  Ordinary load collisions do
    // not share this rule; the post-fixed-point classifier rejects them.
    reaching_use_rtl(node, *dst, *canonical) <--
        sp_indexed_fused_load(node, _, _, _, _, _, dst),
        reg_def_used(node, dst, node),
        is_def(node, def_id),
        reg_xtl(node, *dst, def_id),
        xtl_canonical(def_id, canonical);

    // Merge use-side IDs only at the same node; defs kill values so def/use IDs must not alias (alias_edge handles def-to-use).
    alias_edge(id1, id2) <--
        reg_xtl(node, mreg, id1),
        reg_xtl(node, mreg, id2),
        if id1 != id2,
        !load_overwrites_base(node, mreg),
        !lop_overwrites_input(node, mreg),
        !is_def(node, id1),
        !is_def(node, id2);

    // Block-level dominator computation for back-edge detection; entry block of each function is the block containing the function start address.
    #[local] relation func_entry_block(Address, Address);
    func_entry_block(func, entry_block) <--
        block_in_function(entry_block, func),
        code_in_block(func, entry_block);

    // Block-dominator lattice (Cooper-Harvey-Kennedy) stored as Dual<Set> so the join is intersection: O(blocks * depth) rather than the old O(blocks^2) workspace, with no pairwise relation.
    #[local] lattice block_strict_dom_set(Address, Address, Dual<Set<Address>>);
    lattice block_dom_set(Address, Address, Dual<Set<Address>>);

    block_dom_set(*func, *entry_block, Dual(Set::singleton(*entry_block))) <--
        func_entry_block(func, entry_block);

    block_strict_dom_set(*func, *n, Dual(p_doms.0.clone())) <--
        block_dom_set(func, p, p_doms),
        asm_block_next(p, n),
        block_in_function(n, func),
        !func_entry_block(func, n);

    block_dom_set(*func, *n, Dual(dom_set_with_self(&strict.0, *n))) <--
        block_strict_dom_set(func, n, strict),
        !func_entry_block(func, n);

    // Back-edge: ltl_succ (src, dst) where dst's block dominates src's block in the same function.
    #[local] relation is_loop_back_edge(Address, Address);
    is_loop_back_edge(src, dst) <--
        rtl_next(src, dst),
        code_in_block(src, src_block),
        code_in_block(dst, dst_block),
        if *src_block != *dst_block,
        instr_in_function(src, func),
        block_dom_set(func, src_block, doms),
        if doms.0.contains(dst_block);
    // Intra-block back-edge: dst <= src within the same block (backward jump).
    is_loop_back_edge(src, dst) <--
        rtl_next(src, dst),
        code_in_block(src, blk),
        code_in_block(dst, blk),
        if *dst <= *src;

    // Forward MAY-reaching-defs at each block ENTRY (join is union), restricted to arg/return registers to bound cost; the lattice replacement for the instr_min_order proxies, exact at any distance.
    #[local] relation reg_of_interest(Mreg);
    reg_of_interest(r) <-- is_arg_reg(r);
    reg_of_interest(r) <-- is_xmm_arg_reg(r);
    reg_of_interest(Mreg::AX);
    reg_of_interest(Mreg::X0);

    #[local] lattice reaching_def_in(Address, Address, Mreg, Set<Address>);

    // gen: the def of r leaving block blk (its in-block last def) reaches every successor's entry and kills any incoming value of r along that block.
    reaching_def_in(*func, *succ, *r, Set::singleton(*d)) <--
        reg_of_interest(r),
        last_def_in_block(blk, d, r),
        block_in_function(blk, func),
        asm_block_next(blk, succ),
        instr_in_function(succ, func);

    // pass-through: blk does not define r, so its incoming reaching set flows unchanged to successors.
    reaching_def_in(*func, *succ, *r, in_set.clone()) <--
        reaching_def_in(func, blk, r, in_set),
        !asm_defined_in_block(blk, r),
        asm_block_next(blk, succ),
        instr_in_function(succ, func);

    // def_addr reaches ret_addr intra-block via block_last_def, else via membership in the reaching set at the return's block; the structural replacement for the def_order <= ret_order proxy.
    relation def_reaches_return(Address, Address, Mreg);
    def_reaches_return(*ret_addr, *def_addr, *r) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        reg_of_interest(r),
        block_last_def(ret_addr, def_addr, r),
        asm_effective_def(def_addr, r);
    def_reaches_return(*ret_addr, *def_addr, *r) <--
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        reg_of_interest(r),
        code_in_block(ret_addr, blk),
        instr_in_function(ret_addr, func),
        !block_last_def(ret_addr, _, r),
        reaching_def_in(func, blk, r, s),
        for def_addr in s.0.iter();

    // def_addr reaches call_addr the same way, mirroring def_reaches_return; reaching_def_in already excludes any def clobbered before the call, so no distance window or ordering is needed.
    relation def_reaches_call(Address, Address, Mreg);
    def_reaches_call(*call_addr, *def_addr, *r) <--
        is_call_instruction(call_addr),
        reg_of_interest(r),
        block_last_def(call_addr, def_addr, r),
        asm_effective_def(def_addr, r);
    def_reaches_call(*call_addr, *def_addr, *r) <--
        is_call_instruction(call_addr),
        reg_of_interest(r),
        code_in_block(call_addr, blk),
        instr_in_function(call_addr, func),
        !block_last_def(call_addr, _, r),
        reaching_def_in(func, blk, r, s),
        for def_addr in s.0.iter();

    // Forward reachability: transitive closure of rtl_next excluding back-edges, scoped per function.
    #[local] relation forward_reachable(Address, Address, Address);
    forward_reachable(func, src, dst) <--
        rtl_next(src, dst),
        instr_in_function(src, func),
        instr_in_function(dst, func),
        !is_loop_back_edge(src, dst);
    forward_reachable(func, src, dst) <--
        forward_reachable(func, src, mid),
        rtl_next(mid, dst),
        instr_in_function(dst, func),
        !is_loop_back_edge(mid, dst);

    relation has_intervening_def(Address, Address, Mreg);
    // Reachability checks precede instr_in_function so the path constraint prunes candidate mids before the multi-valued lookup; a pure clause reorder deriving the same set (8.9s + 18.3s).
    has_intervening_def(def_addr, use_addr, mreg) <--
        reg_def_used(def_addr, mreg, use_addr),
        reg_def_site(mid_addr, mreg),
        if *mid_addr != *def_addr && *mid_addr != *use_addr,
        instr_in_function(def_addr, func),
        forward_reachable(func, def_addr, mid_addr),
        forward_reachable(func, mid_addr, use_addr),
        instr_in_function(mid_addr, func);

    has_intervening_def(def_addr, use_addr, mreg) <--
        reg_def_used(def_addr, mreg, use_addr),
        is_call_clobbered(mid_addr, mreg),
        if *mid_addr != *def_addr && *mid_addr != *use_addr,
        instr_in_function(def_addr, func),
        forward_reachable(func, def_addr, mid_addr),
        forward_reachable(func, mid_addr, use_addr),
        instr_in_function(mid_addr, func);

    // param_live_block is MUST-live: the param reaches the block on ALL paths with no intervening def, unlike the earlier MAY-reach that bound params at joins where they no longer hold.
    relation param_block_reach(Address, Address, Mreg);
    relation param_def_tainted(Address, Address, Mreg);
    relation param_live_block(Address, Address, Mreg);

    // The successor guard confines propagation to the function's own blocks, since asm_block_next is the raw CFG and can cross function boundaries, leaking a param program-wide.
    param_block_reach(func_start, entry_block, mreg) <--
        func_param_validated(func_start, mreg),
        code_in_block(func_start, entry_block);
    param_block_reach(func_start, succ_block, mreg) <--
        param_block_reach(func_start, curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);

    // def-tainted: some path entry->block passes through a reachable block that defines (clobbers, including a CALL of a caller-saved reg) mreg, so on that path mreg no longer holds the param.
    param_def_tainted(func_start, succ_block, mreg) <--
        param_block_reach(func_start, curr_block, mreg),
        asm_defined_in_block(curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);
    param_def_tainted(func_start, succ_block, mreg) <--
        param_def_tainted(func_start, curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);

    param_live_block(func_start, block, mreg) <--
        param_block_reach(func_start, block, mreg),
        !param_def_tainted(func_start, block, mreg);

    // arg_reg_param_live_at is MUST: the arg register still holds the incoming caller value at addr; seeded structurally on is_arg_reg to stay out of the param-validation SCC, distance-independent.
    #[local] relation arg_reg_param_reach(Address, Address, Mreg);
    #[local] relation arg_reg_param_tainted(Address, Address, Mreg);
    #[local] relation arg_reg_param_live_block(Address, Address, Mreg);
    relation arg_reg_param_live_at(Address, Address, Mreg);

    // Only seed/propagate arg regs actually used in the function; arg_reg_param_live_at is consumed only at reg uses, so unused arg regs would propagate through the whole CFG for nothing.
    #[local] relation arg_reg_used_somewhere(Address, Mreg);
    arg_reg_used_somewhere(*func, *r) <-- is_arg_reg(r), reg_use(addr, r), instr_in_function(addr, func);
    arg_reg_used_somewhere(*func, *r) <-- is_float_arg_reg(r), reg_use(addr, r), instr_in_function(addr, func);

    // A direct call to a known-signature callee implicitly READS its first sig_count ABI argument registers, seeded here so a forwarded parameter is recognized as live; confined to arg_reg_used_somewhere.
    arg_reg_used_somewhere(*func, *arg_reg) <--
        call_known_sig_gp_position(call_addr, pos),
        instr_in_function(call_addr, func),
        is_arg_reg(arg_reg),
        abi_int_arg_position(arg_reg, pos);

    // Calls do not list their ABI argument registers as operands, so seed the fixed set in call-containing functions; forwarding_corroborated still requires real evidence before materializing an argument.
    arg_reg_used_somewhere(*func, *arg_reg) <--
        is_call_instruction(call_addr),
        instr_in_function(call_addr, func),
        is_arg_reg(arg_reg);
    arg_reg_used_somewhere(*func, *arg_reg) <--
        is_call_instruction(call_addr),
        instr_in_function(call_addr, func),
        is_float_arg_reg(arg_reg);

    arg_reg_param_reach(*func, *entry_block, *r) <--
        arg_reg_used_somewhere(func, r), func_entry_block(func, entry_block);
    arg_reg_param_reach(*func, *succ, *r) <--
        arg_reg_param_reach(func, blk, r),
        asm_block_next(blk, succ),
        instr_in_function(succ, func);

    arg_reg_param_tainted(*func, *succ, *r) <--
        arg_reg_param_reach(func, blk, r),
        asm_defined_in_block(blk, r),
        asm_block_next(blk, succ),
        instr_in_function(succ, func);
    arg_reg_param_tainted(*func, *succ, *r) <--
        arg_reg_param_tainted(func, blk, r),
        asm_block_next(blk, succ),
        instr_in_function(succ, func);

    arg_reg_param_live_block(*func, *blk, *r) <--
        arg_reg_param_reach(func, blk, r),
        !arg_reg_param_tainted(func, blk, r);

    arg_reg_param_live_at(*func, *addr, *r) <--
        arg_reg_param_live_block(func, blk, r),
        code_in_block(addr, blk),
        instr_in_function(addr, func),
        !block_last_def(addr, _, r);

    // Loop-induction param entry value: recompute the taint over FORWARD edges only, since tainting via the back-edge drops the param's first-iteration value and the walker is read uninitialized.
    #[local] relation block_back_edge(Address, Address);
    block_back_edge(blk, succ) <--
        asm_block_next(blk, succ),
        instr_in_function(blk, func),
        block_dom_set(func, blk, doms),
        if doms.0.contains(succ);

    #[local] relation arg_reg_param_tainted_fwd(Address, Address, Mreg);
    arg_reg_param_tainted_fwd(*func, *succ, *r) <--
        arg_reg_param_reach(func, blk, r),
        asm_defined_in_block(blk, r),
        asm_block_next(blk, succ),
        !block_back_edge(blk, succ),
        instr_in_function(succ, func);
    arg_reg_param_tainted_fwd(*func, *succ, *r) <--
        arg_reg_param_tainted_fwd(func, blk, r),
        asm_block_next(blk, succ),
        !block_back_edge(blk, succ),
        instr_in_function(succ, func);

    #[local] relation arg_reg_param_live_block_fwd(Address, Address, Mreg);
    arg_reg_param_live_block_fwd(*func, *blk, *r) <--
        arg_reg_param_reach(func, blk, r),
        !arg_reg_param_tainted_fwd(func, blk, r);

    #[local] relation arg_reg_param_live_at_fwd(Address, Address, Mreg);
    arg_reg_param_live_at_fwd(*func, *addr, *r) <--
        arg_reg_param_live_block_fwd(func, blk, r),
        code_in_block(addr, blk),
        instr_in_function(addr, func),
        !block_last_def(addr, _, r);

    reg_def_used(*func_start, *mreg, *use_addr) <--
        arg_reg_param_live_at_fwd(func_start, use_addr, mreg),
        is_arg_reg(mreg),
        reg_use(use_addr, mreg),
        !trim_instruction(use_addr),
        // Only the genuine loop delta: a use the MUST-live path drops because the param is tainted by its own back-edge; for non-loop uses arg_reg_param_live_at already binds the param.
        !arg_reg_param_live_at(func_start, use_addr, mreg);

    alias_edge(def_id, use_id) <--
        reg_def_used(defaddr, mreg, useaddr),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, mreg, def_id),
        load_overwrite_use_id(useaddr, mreg, use_id),
        if def_id != use_id,
        !has_intervening_def(defaddr, useaddr, mreg);

    alias_edge(def_id, use_id) <--
        reg_def_used(defaddr, mreg, useaddr),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, mreg, def_id),
        lop_overwrite_use_id(useaddr, mreg, use_id),
        if def_id != use_id,
        !has_intervening_def(defaddr, useaddr, mreg);

    // Param case of the Osel preserve binding: bind the preserve use-id to the function's param reg when dst still holds the incoming caller value, since an entry pseudo-def carries no is_def.
    alias_edge(param_id, use_id) <--
        lop_overwrite_use_id(node, mreg, use_id),
        is_arg_reg(mreg),
        arg_reg_param_live_at(func_start, node, mreg),
        reg_xtl(func_start, mreg, param_id),
        if param_id != use_id;

    // CMP/TEST+SETCC fusion: capstone still tags the SETCC as the reg_def, so alias the two def ids or that phantom shadows the Ocmp's def and its consumer reads an uninitialized register.
    alias_edge(lop_def_id, setcc_def_id) <--
        ltl_inst(lop_addr, ?LTLInst::Lop(Operation::Ocmp(_), _, dst_mreg)),
        next(lop_addr, setcc_addr),
        reg_def(setcc_addr, *dst_mreg),
        !ltl_inst(setcc_addr, _),
        !trim_instruction(setcc_addr),
        is_def(lop_addr, lop_def_id),
        reg_xtl(lop_addr, *dst_mreg, lop_def_id),
        is_def(setcc_addr, setcc_def_id),
        reg_xtl(setcc_addr, *dst_mreg, setcc_def_id),
        if lop_def_id != setcc_def_id;

    alias_edge(def_id, use_id) <--
        reg_def_used(defaddr, mreg, useaddr),
        load_overwrites_base(defaddr, mreg),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, mreg, def_id),
        reg_xtl(useaddr, mreg, use_id),
        if def_id != use_id,
        !has_intervening_def(defaddr, useaddr, mreg),
        !load_overwrites_base(useaddr, mreg);

    alias_edge(def_id, use_id) <--
        reg_def_used(defaddr, mreg, useaddr),
        !load_overwrites_base(defaddr, mreg),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, mreg, def_id),
        reg_xtl(useaddr, mreg, use_id),
        if def_id != use_id,
        !has_intervening_def(defaddr, useaddr, mreg),
        !load_overwrites_base(useaddr, mreg);

    // Value-phi at a shared return: alias every def that STRUCTURALLY reaches the return to the return's AX use-id, so all arms' values coalesce and each survives DSE on its own arm.
    alias_edge(def_id, use_id) <--
        func_produces_retval(func),
        instr_in_function(ret_addr, func),
        def_reaches_return(ret_addr, def_addr, Mreg::AX),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        reg_xtl(ret_addr, Mreg::AX, use_id),
        !is_def(ret_addr, use_id),
        is_def(def_addr, def_id),
        reg_xtl(def_addr, Mreg::AX, def_id),
        if def_id != use_id;

    relation param_still_live(Address, Address, Mreg);
    param_still_live(func_start, use_addr, mreg) <--
        func_param_validated(func_start, mreg),
        instr_in_function(use_addr, func_start),
        code_in_block(use_addr, block),
        param_live_block(func_start, block, mreg),
        !block_last_def(use_addr, _, mreg);

    // Float-param liveness mirroring param_live_block for XMM args, so an XMM-param use reachable def-free from entry binds to the func_start param reg instead of a fresh undefined one.
    relation float_param_block_reach(Address, Address, Mreg);
    relation float_param_def_tainted(Address, Address, Mreg);
    relation float_param_live_block(Address, Address, Mreg);

    float_param_block_reach(func_start, entry_block, mreg) <--
        func_float_param_validated(func_start, mreg, _),
        code_in_block(func_start, entry_block);
    float_param_block_reach(func_start, succ_block, mreg) <--
        float_param_block_reach(func_start, curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);

    float_param_def_tainted(func_start, succ_block, mreg) <--
        float_param_block_reach(func_start, curr_block, mreg),
        asm_defined_in_block(curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);
    float_param_def_tainted(func_start, succ_block, mreg) <--
        float_param_def_tainted(func_start, curr_block, mreg),
        asm_block_next(curr_block, succ_block),
        instr_in_function(succ_block, func_start);

    float_param_live_block(func_start, block, mreg) <--
        float_param_block_reach(func_start, block, mreg),
        !float_param_def_tainted(func_start, block, mreg);

    relation float_param_still_live(Address, Address, Mreg);
    float_param_still_live(func_start, use_addr, mreg) <--
        func_float_param_validated(func_start, mreg, _),
        instr_in_function(use_addr, func_start),
        code_in_block(use_addr, block),
        float_param_live_block(func_start, block, mreg),
        !block_last_def(use_addr, _, mreg);

    // Thread the float param's func_start reg to an early XMM-param use with no in-function def (mirrors the integer func_param_validated reg_xtl rule), else the use got a fresh undefined xtl id (self-undef fabs / undefined compare operand).
    reg_xtl(*use_addr, *mreg, param_id) <--
        func_float_param_validated(func_start, mreg, _),
        reg_xtl(func_start, mreg, param_id),
        float_param_still_live(func_start, use_addr, mreg),
        reg_use(use_addr, mreg),
        !reg_def_site(use_addr, mreg);

    // 3.8 R1: a fused float RMW both reads and writes its dst, so the threading rule is blocked by its own reg_def_site; alias the param web onto the op's ids under the 2-address model.
    alias_edge(param_id, op_id) <--
        float_load_op(addr, _, _, _, _, mreg, false),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        func_float_param_validated(func_start, mreg, _),
        float_param_still_live(func_start, addr, mreg),
        reg_xtl(func_start, mreg, param_id),
        reg_xtl(addr, *mreg, op_id),
        if param_id != op_id;

    // The same two-address fused RMW can target an integer argument register.
    // Its own def blocks ordinary entry-parameter threading, so explicitly
    // join the incoming value to the operation web just as for XMM arguments.
    alias_edge(param_id, op_id) <--
        float_load_op(addr, _, _, _, _, mreg, false),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        func_param_validated(func_start, mreg),
        param_still_live(func_start, addr, mreg),
        reg_xtl(func_start, mreg, param_id),
        reg_xtl(addr, *mreg, op_id),
        if param_id != op_id;

    param_alias_source(useaddr, mreg) <-- arg_reg_has_external_def(useaddr, mreg);
    param_alias_source(useaddr, mreg) <-- arg_reg_used_no_def(useaddr, mreg);
    param_alias_source(useaddr, mreg) <-- arg_reg_used_very_early(useaddr, mreg);

    param_alias_source(use_addr, mreg) <--
        param_still_live(func_start, use_addr, mreg),
        reg_xtl(use_addr, mreg, _),
        !asm_effective_def(use_addr, mreg);

    alias_edge(def_id, use_id) <--
        param_alias_source(useaddr, mreg),
        instr_in_function(useaddr, func_start),
        func_param_validated(func_start, mreg),
        reg_xtl(func_start, mreg, def_id),
        reg_xtl(useaddr, mreg, use_id),
        if def_id != use_id;

    // Param-bind an arg-reg use over an external def edge whenever the reg still holds the incoming value; the old entry-distance window was a magnitude proxy that dropped deep clean uses.
    reg_xtl(*useaddr, *reg, param_id) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        instr_in_function(useaddr, func_start),
        is_arg_reg(reg),
        !instr_in_function(defaddr, func_start),
        func_arg_reg_used(func_start, reg),
        reg_xtl(func_start, reg, param_id),
        param_still_live(func_start, useaddr, reg);

    reg_xtl(*use_addr, *mreg, param_id) <--
        func_param_validated(func_start, mreg),
        reg_xtl(func_start, mreg, param_id),
        param_still_live(func_start, use_addr, mreg),
        reg_use(use_addr, mreg),
        !reg_def_site(use_addr, mreg);

    reg_xtl(*useaddr, *reg, def_id) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        !is_arg_reg(reg),
        !return_val_used(defaddr, _, reg, useaddr, _),
        !load_overwrites_base(useaddr, reg),
        !has_intervening_def(defaddr, useaddr, reg),
        reg_xtl(defaddr, reg, def_id),
        is_def(defaddr, def_id);

    reg_xtl(*useaddr, *reg, rtl_reg) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        !is_arg_reg(reg),
        !return_val_used(defaddr, _, reg, useaddr, _),
        !load_overwrites_base(useaddr, reg),
        has_intervening_def(defaddr, useaddr, reg),
        let rtl_reg = fresh_xtl_reg(*useaddr, *reg);

    reg_xtl(node, Mreg::AX, rtl_reg) <--
        call_return_reg(node, rtl_reg);

    reg_xtl(*use_addr, *mreg, ret_rtl) <--
        return_val_used(call_addr, _, mreg, use_addr, _),
        call_return_reg(call_addr, ret_rtl);


    reg_xtl(node, Mreg::AX, fresh) <--
        ltl_inst(node, ?LTLInst::Lreturn),
        let fresh = fresh_xtl_reg(*node, Mreg::AX);

    // External-def edge whose use is NOT param-bindable (the reg no longer holds the caller value there): fresh id, reconnected to its def by alias_edge downstream. Exact complement of the param-bind rule above, replacing the "use is far from entry" (!is_early_arg_use) distance proxy.
    reg_xtl(*useaddr, *reg, rtl_reg) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        is_arg_reg(reg),
        instr_in_function(useaddr, func_start),
        !instr_in_function(defaddr, func_start),
        func_arg_reg_used(func_start, reg),
        !param_still_live(func_start, useaddr, reg),
        let rtl_reg = fresh_xtl_reg(*useaddr, *reg);

    reg_xtl(*useaddr, *reg, rtl_reg) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        is_arg_reg(reg),
        instr_in_function(useaddr, func_start),
        !instr_in_function(defaddr, func_start),
        !func_arg_reg_used(func_start, reg),
        let rtl_reg = fresh_xtl_reg(*useaddr, *reg);

    // Internal-def edge to an arg-reg use: always a fresh id (the def-to-use connection is made by alias_edge / xtl canonicalization). The old is_early_arg_use condition merely split this one semantic case across two window variants.
    reg_xtl(*useaddr, *reg, rtl_reg) <--
        reg_def_used(defaddr, reg, useaddr),
        if *reg != Mreg::SP,
        is_arg_reg(reg),
        instr_in_function(useaddr, func_start),
        instr_in_function(defaddr, func_start),
        let rtl_reg = fresh_xtl_reg(*useaddr, *reg);

    reg_def_site(defaddr, dst_reg) <--
        ltl_inst(defaddr, ?LTLInst::Lop(_, _, dst_reg));

    reg_def_site(defaddr, *dst_reg) <--
        ltl_inst(defaddr, ?LTLInst::Lload(_, _, _, dst_reg));

    reg_def_site(defaddr, *dst_reg) <--
        ltl_inst(defaddr, ?LTLInst::Lgetstack(_, _, _, dst_reg));

    reg_def_site(defaddr, *dst_reg) <--
        ltl_inst(defaddr, ?LTLInst::Lbuiltin(_, _, BuiltinArg::BA(dst_reg))),
        if *dst_reg != Mreg::Unknown;

    reg_xtl(defaddr, dst_reg, rtl_reg), is_def(defaddr, rtl_reg) <--
        reg_def_site(defaddr, dst_reg),
        let rtl_reg = fresh_xtl_reg(*defaddr, *dst_reg);

    ltl_use(addr, src_reg) <--
        ltl_inst(addr, ?LTLInst::Lop(_, srcs, _)),
        for src_reg in srcs.iter();

    ltl_use(addr, arg_reg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        for arg_reg in args.iter();

    ltl_use(addr, src_reg) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, _, src_reg));

    ltl_use(addr, arg_reg) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, args, _)),
        for arg_reg in args.iter();

    ltl_use(addr, arg_reg) <--
        ltl_inst(addr, ?LTLInst::Lcond(_, args, _, _)),
        for arg_reg in args.iter();

    reg_xtl(addr, *reg, fresh_id) <--
        ltl_use(addr, reg),
        if *reg != Mreg::SP,
        let fresh_id = fresh_xtl_reg(*addr, *reg);

    ltl_use(addr, src) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, _, _, _));

    ltl_use(addr, reg) <--
        ltl_inst(addr, ?LTLInst::Ljumptable(reg, _));

    ltl_use(addr, reg) <--
        ltl_inst(addr, ?LTLInst::Lcall(tgt)),
        if let Either::Left(reg) = tgt;

    ltl_use(addr, reg) <--
        ltl_inst(addr, ?LTLInst::Ltailcall(tgt)),
        if let Either::Left(reg) = tgt;

    ltl_use(addr, arg_reg) <--
        ltl_inst(addr, ?LTLInst::Lbuiltin(_, args, _)),
        for arg in args.iter(),
        if let BuiltinArg::BA(arg_reg) = arg;


    stack_xtl(func_start, addr, ofs, rtlreg) <--
        mach_imm_stack_init(addr, ofs, _, _),
        instr_in_function(addr, func_start),
        let rtlreg = fresh_stack_cell_reg(*addr);

    // Propagate stack_xtl through def-use chains, joining on def offset and emitting use offset.
    stack_xtl(func_start, use_addr, use_ofs, rtlreg) <--
        stack_xtl(func_start, def_addr, def_ofs, rtlreg),
        instr_in_function(def_addr, func_start),
        stack_def_used(def_addr, _, def_ofs, use_addr, _, use_ofs),
        instr_in_function(use_addr, func_start);

    stack_xtl(func_start, addr, ofs, rtlreg) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, _, _)),
        instr_in_function(addr, func_start),
        !stack_def_used(_, _, _, addr, _, ofs),
        let rtlreg = fresh_stack_cell_reg(*addr);

    stack_xtl(func_start, addr, ofs, rtlreg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Ainstack(ofs), _, _)),
        instr_in_function(addr, func_start),
        !stack_def_used(_, _, _, addr, _, ofs),
        let rtlreg = fresh_stack_cell_reg(*addr);

    stack_xtl(func_start, addr, ofs, rtlreg) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Olea(Addressing::Ainstack(ofs)), _, _)),
        instr_in_function(addr, func_start),
        let rtlreg = fresh_stack_cell_reg(*addr);


    alias_edge(rtlreg1, rtlreg2) <--
        stack_xtl(func_start, use_addr, ofs, rtlreg1),
        stack_xtl(func_start, use_addr, ofs, rtlreg2),
        if rtlreg1 != rtlreg2;

    // Bucket E: a reload of an address-taken slot has no reaching def, so alias its stack_xtl with the escaping slot's and both resolve to the SAME local the callee wrote through &local.
    #[local] relation slot_addr_escaped(Address, Address, i64);

    // reg_holds_slot_addr: a register holding &slot(ofs), base case the Olea(Ainstack(ofs)) and transitive case a register copy, so the escape is seen even when the LEA does not write the arg register.
    #[local] relation reg_holds_slot_addr(Address, Address, i64, Mreg);
    reg_holds_slot_addr(func_start, lea_addr, *ofs, *dst_reg) <--
        ltl_inst(lea_addr, ?LTLInst::Lop(Operation::Olea(Addressing::Ainstack(ofs)), _, dst_reg)),
        instr_in_function(lea_addr, func_start);
    reg_holds_slot_addr(func_start, mov_addr, ofs, dst_reg) <--
        reg_holds_slot_addr(func_start, src_def, ofs, src_reg),
        reg_def_used(src_def, src_reg, mov_addr),
        ltl_inst(mov_addr, ?LTLInst::Lop(Operation::Omove, args, dst_reg)),
        if args.len() == 1 && args[0] == *src_reg;

    slot_addr_escaped(func_start, lea_addr, *ofs) <--
        ltl_inst(lea_addr, ?LTLInst::Lop(Operation::Olea(Addressing::Ainstack(ofs)), _, _)),
        instr_in_function(lea_addr, func_start),
        reg_holds_slot_addr(func_start, def_addr, *ofs, dst_reg),
        arg_setup_candidate(def_addr, dst_reg, _);

    alias_edge(reload_rtl, lea_rtl) <--
        ltl_inst(reload_addr, ?LTLInst::Lgetstack(_slot, ofs, _typ, _dst)),
        instr_in_function(reload_addr, func_start),
        !stack_def_used(_, _, _, reload_addr, _, ofs),
        slot_addr_escaped(func_start, lea_addr, ofs),
        if *lea_addr != *reload_addr,
        stack_xtl(func_start, reload_addr, ofs, reload_rtl),
        stack_xtl(func_start, lea_addr, ofs, lea_rtl),
        if reload_rtl != lea_rtl;

    // Whole-slot unification for an escaped address-taken slot: every access at that frame offset denotes one memory cell, so unify the LEA's address reg with every stack_xtl reg at the same offset.
    alias_edge(lea_rtl, other_rtl) <--
        slot_addr_escaped(func_start, lea_addr, ofs),
        stack_xtl(func_start, lea_addr, ofs, lea_rtl),
        stack_xtl(func_start, _, ofs, other_rtl),
        if lea_rtl != other_rtl;

    // Bucket E parity for synthetic-only stack reads: export the slot's single escaped canonical so a fresh-RTEMP Iload resolves deterministically instead of via the multi-valued stack_local_at.
    // Retain the escaping LEA's origin node as well as its raw displacement.
    // The post-fixed-point Win64 home selector may replace one particular LEA
    // with a node-keyed canonical home address.  Another, post-prologue LEA in
    // the same function can legitimately use the same raw displacement for a
    // different local, so filtering only by (function, offset) is unsound.
    relation slot_escaped_origin(Address, Node, i64, RTLReg);
    slot_escaped_origin(func_start, *lea_addr, ofs, canonical) <--
        slot_addr_escaped(func_start, lea_addr, ofs),
        stack_var(func_start, lea_addr, ofs, canonical);

    relation slot_escaped_canonical(Address, i64, RTLReg);
    slot_escaped_canonical(func, ofs, m) <--
        slot_escaped_origin(func, _, ofs, _),
        agg m = ascent::aggregators::min(c) in slot_escaped_origin(func, _, ofs, c);

    stack_var_chunk(*func, ofs, chunk) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, Addressing::Ainstack(ofs), _, _)),
        instr_in_function(node, func);

    stack_var_chunk(*func, ofs, chunk) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, Addressing::Ainstack(ofs), _, _)),
        instr_in_function(node, func);

    stack_var_chunk(*func, ofs, chunk) <--
        ltl_inst(node, ?LTLInst::Lgetstack(_, ofs, typ, _)),
        let chunk = typ_to_chunk(*typ),
        instr_in_function(node, func);

    stack_var_chunk(*func, ofs, chunk) <--
        ltl_inst(node, ?LTLInst::Lsetstack(_, _, ofs, typ)),
        let chunk = typ_to_chunk(*typ),
        instr_in_function(node, func);

    // An immediate-to-stack store has no ltl_inst, so record its width here like any other stack access, or an 8-byte zero-init out-param falls back to int and truncates to 4 bytes.
    stack_var_chunk(*func, ofs, chunk) <--
        mach_imm_stack_init(addr, ofs, _, typ),
        instr_in_function(addr, func),
        let chunk = typ_to_chunk(*typ);


    relation plt_function(Address, Symbol);

    plt_function(addr, clean_name_sym) <--
        plt_entry(addr, raw_name),
        let clean_name = raw_name.split('@').next().unwrap_or(raw_name),
        let clean_name_sym = Box::leak(clean_name.to_string().into_boxed_str()) as Symbol;

    plt_function(addr, clean_name_sym) <--
        plt_block(addr, raw_name),
        let clean_name = raw_name.split('@').next().unwrap_or(raw_name),
        let clean_name_sym = Box::leak(clean_name.to_string().into_boxed_str()) as Symbol;

    relation external_call_site(Node, Address, Symbol);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        plt_function(target, name);


    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        let offset_target = target + ENDBR64_LEN,
        plt_function(offset_target, name);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        if *target >= ENDBR64_LEN,
        let offset_target = target - ENDBR64_LEN,
        plt_function(offset_target, name);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Ltailcall(Either::Right(Either::Left(target)))),
        plt_function(target, name);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Ltailcall(Either::Right(Either::Left(target)))),
        let offset_target = target + ENDBR64_LEN,
        plt_function(offset_target, name);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Ltailcall(Either::Right(Either::Left(target)))),
        if *target >= ENDBR64_LEN,
        let offset_target = target - ENDBR64_LEN,
        plt_function(offset_target, name);

    external_call_site(call_site, target, *name) <--
        ltl_inst(call_site, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        base_ident_to_symbol(target_id, name),
        if *target as usize == *target_id,
        // Local def wins: a binary's own function sharing a tabled libc name is not an extern call.
        !emit_function(_, name, _),
        known_extern_signature(name, _, _, _);

    relation call_site(Node, Symbol);
    relation call_arg(Node, usize, RTLReg);

    call_site(node, *name) <--
        external_call_site(node, _, name);

    call_site(call_addr, *name) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        emit_function(target, name, _);

    call_arg(node, pos, reg) <--
        call_arg_mapping(node, pos, reg);


    ident_to_symbol(id, *name) <-- base_ident_to_symbol(id, name);
    ident_to_symbol(target_id, *name) <--
        external_call_site(_, target, name),
        let target_id = *target as usize;


    relation resolved_extern_signature(Symbol, usize, XType, Arc<Vec<XType>>);

    resolved_extern_signature(*name, *count, *ret, params.clone()) <--
        plt_function(_, name),
        known_extern_signature(name, count, ret, params);

    call_has_known_signature(call_site, *name, *param_count, *ret_type) <--
        external_call_site(call_site, _, name),
        known_extern_signature(name, param_count, ret_type, _);

    // Also match symbol-based calls (Right(Right(name))) against known signatures
    call_has_known_signature(call_site, *name, *param_count, *ret_type) <--
        ltl_inst(call_site, ?LTLInst::Lcall(Either::Right(Either::Right(name)))),
        !emit_function(_, name, _),
        known_extern_signature(name, param_count, ret_type, _);

    call_has_known_signature(call_site, *name, *param_count, *ret_type) <--
        ltl_inst(call_site, ?LTLInst::Ltailcall(Either::Right(Either::Right(name)))),
        !emit_function(_, name, _),
        known_extern_signature(name, param_count, ret_type, _);

    // Count of GP-class params in a known signature: float args consume no integer position, so gating forwarded-param evidence on this count stops a pure-float callee fabricating a phantom int param.
    #[local] relation known_sig_gp_param_count(Symbol, usize);
    known_sig_gp_param_count(*name, gp) <--
        known_extern_signature(name, _, _, params),
        let gp = params.iter().filter(|t| !matches!(**t, XType::Xfloat | XType::Xsingle)).count();

    #[local] relation known_sig_gp_position(Symbol, usize);
    known_sig_gp_position(name, pos) <--
        known_sig_gp_param_count(name, gp),
        !abi_shared_arg_slots(true),
        abi_int_arg_position(_, pos),
        if *pos < *gp;
    known_sig_gp_position(*name, pos) <--
        known_extern_signature(name, _, _, params),
        abi_shared_arg_slots(true),
        abi_int_arg_position(_, pos),
        if *pos < params.len(),
        if !matches!(params[*pos], XType::Xfloat | XType::Xsingle);

    relation call_known_sig_gp_position(Node, usize);
    call_known_sig_gp_position(*call_site, *pos) <--
        call_has_known_signature(call_site, name, _, _),
        known_sig_gp_position(name, pos);

    relation unknown_extern(Symbol);

    unknown_extern(*name) <--
        plt_function(_, name),
        !known_extern_signature(name, _, _, _);

    relation function_entry_dist(Address, Node, i64);
    relation call_arg_mapping(Node, usize, RTLReg);
    relation call_arg_position_allowed(Node, usize);

    relation call_target_func(Node, Address);

    call_target_func(call_addr, *target) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(target))));

    call_target_func(call_addr, *target) <--
        ltl_inst(call_addr, ?LTLInst::Ltailcall(Either::Right(Either::Left(target))));

    relation func_param_position_type(Address, usize, XType);

    func_param_position_type(func_start, 0, xtype) <--
        func_has_param_at_position(func_start, 0),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::DI, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 1, xtype) <--
        func_has_param_at_position(func_start, 1),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::SI, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 2, xtype) <--
        func_has_param_at_position(func_start, 2),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::DX, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 3, xtype) <--
        func_has_param_at_position(func_start, 3),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::CX, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 4, xtype) <--
        func_has_param_at_position(func_start, 4),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::R8, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 5, xtype) <--
        func_has_param_at_position(func_start, 5),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::R9, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);

    func_param_position_type(func_start, 0, XType::Xint) <--
        func_has_param_at_position(func_start, 0),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::DI, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, 1, XType::Xint) <--
        func_has_param_at_position(func_start, 1),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::SI, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, 2, XType::Xint) <--
        func_has_param_at_position(func_start, 2),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::DX, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, 3, XType::Xint) <--
        func_has_param_at_position(func_start, 3),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::CX, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, 4, XType::Xint) <--
        func_has_param_at_position(func_start, 4),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::R8, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, 5, XType::Xint) <--
        func_has_param_at_position(func_start, 5),
        !abi_shared_arg_slots(true),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::R9, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    // Windows shared-slot GP parameter typing.
    func_param_position_type(func_start, *pos, xtype) <--
        abi_shared_arg_slots(true),
        func_param_validated(func_start, mreg),
        abi_int_arg_position(mreg, pos),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, mreg, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *pos, XType::Xint) <--
        abi_shared_arg_slots(true),
        func_param_validated(func_start, mreg),
        abi_int_arg_position(mreg, pos),
        func_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, mreg, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    // Integer-only param count, used as offset for SysV float param positions.
    relation emit_function_int_param_count(Address, usize);

    emit_function_int_param_count(target_addr, count) <--
        !abi_shared_arg_slots(true),
        func_param_validated(target_addr, _),
        agg count = ascent::aggregators::count() in func_param_validated(target_addr, _);

    emit_function_int_param_count(target_addr, 0) <--
        !abi_shared_arg_slots(true),
        func_stacksz(target_addr, _, _, _),
        !func_param_validated(target_addr, _);

    // Float param position types appended after integer params
    func_param_position_type(func_start, *int_count + 0, xtype) <--
        func_float_param_validated(func_start, Mreg::X0, 0),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X0, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 0, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X0, 0),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X0, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 1, xtype) <--
        func_float_param_validated(func_start, Mreg::X1, 1),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X1, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 1, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X1, 1),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X1, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 2, xtype) <--
        func_float_param_validated(func_start, Mreg::X2, 2),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X2, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 2, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X2, 2),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X2, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 3, xtype) <--
        func_float_param_validated(func_start, Mreg::X3, 3),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X3, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 3, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X3, 3),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X3, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 4, xtype) <--
        func_float_param_validated(func_start, Mreg::X4, 4),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X4, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 4, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X4, 4),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X4, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 5, xtype) <--
        func_float_param_validated(func_start, Mreg::X5, 5),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X5, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 5, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X5, 5),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X5, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 6, xtype) <--
        func_float_param_validated(func_start, Mreg::X6, 6),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X6, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 6, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X6, 6),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X6, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    func_param_position_type(func_start, *int_count + 7, xtype) <--
        func_float_param_validated(func_start, Mreg::X7, 7),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X7, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *int_count + 7, XType::Xfloat) <--
        func_float_param_validated(func_start, Mreg::X7, 7),
        emit_function_int_param_count(func_start, int_count),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, Mreg::X7, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    // Windows floating-point registers use the same source-language ordinal as their GP counterparts rather than an appended XMM sequence.
    func_param_position_type(func_start, *pos, xtype) <--
        abi_shared_arg_slots(true),
        func_float_param_validated(func_start, mreg, pos),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, mreg, rtl_reg),
        emit_var_type_candidate(rtl_reg, xtype);
    func_param_position_type(func_start, *pos, XType::Xfloat) <--
        abi_shared_arg_slots(true),
        func_float_param_validated(func_start, mreg, pos),
        func_float_arg_used_undefined(func_start, rtl_reg),
        reg_xtl(func_start, mreg, rtl_reg),
        !emit_var_type_candidate(rtl_reg, _);

    // U1: route inferred-pointer evidence into the SIGNATURE param types, not just the decl, or the callee's int signature makes every call site cast the pointer arg to (int).
    func_param_position_type(func_start, 0, XType::Xptr) <--
        func_has_param_at_position(func_start, 0),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::DI);
    func_param_position_type(func_start, 1, XType::Xptr) <--
        func_has_param_at_position(func_start, 1),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::SI);
    func_param_position_type(func_start, 2, XType::Xptr) <--
        func_has_param_at_position(func_start, 2),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::DX);
    func_param_position_type(func_start, 3, XType::Xptr) <--
        func_has_param_at_position(func_start, 3),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::CX);
    func_param_position_type(func_start, 4, XType::Xptr) <--
        func_has_param_at_position(func_start, 4),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::R8);
    func_param_position_type(func_start, 5, XType::Xptr) <--
        func_has_param_at_position(func_start, 5),
        !abi_shared_arg_slots(true),
        param_inferred_type_is_pointer(func_start, Mreg::R9);

    func_param_position_type(func_start, *pos, XType::Xptr) <--
        abi_shared_arg_slots(true),
        func_param_validated(func_start, mreg),
        abi_int_arg_position(mreg, pos),
        param_inferred_type_is_pointer(func_start, mreg);

    relation func_param_types(Address, Arc<Vec<XType>>);

    func_param_types(func_start, param_types) <--
        func_param_position_type(func_start, _, _),
        agg param_types = build_xtype_vec(pos, xtype) in func_param_position_type(func_start, pos, xtype);

    // For internal functions matching known externs, use known sig to avoid false variadic params.
    relation func_has_known_extern_sig(Address);

    // Only suppress inferred sigs for variadic functions; non-variadic may be custom implementations.
    func_has_known_extern_sig(func_start) <--
        emit_function(func_start, name, _),
        known_varargs_function(name, _);

    relation emit_function_signature_candidate(Address, Signature);

    // When a known varargs extern signature exists, prefer it over inferred signature.
    emit_function_signature_candidate(func_start, sig) <--
        emit_function(func_start, name, _),
        known_varargs_function(name, _),
        known_extern_signature(name, _, ret_type, known_params),
        let sig = Signature { sig_args: known_params.clone(), sig_res: *ret_type, sig_cc: CallConv::default() };

    emit_function_signature_candidate(func_start, sig) <--
        emit_function(func_start, _, _),
        !func_has_known_extern_sig(func_start),
        func_param_types(func_start, param_types),
        emit_function_return_type_xtype_candidate(func_start, ret_type),
        let sig = Signature { sig_args: param_types.clone(), sig_res: *ret_type, sig_cc: CallConv::default() };

    emit_function_signature_candidate(func_start, sig) <--
        emit_function(func_start, _, _),
        !func_has_known_extern_sig(func_start),
        func_param_types(func_start, param_types),
        !emit_function_return_type_xtype_candidate(func_start, _),
        let sig = Signature { sig_args: param_types.clone(), sig_res: XType::Xvoid, sig_cc: CallConv::default() };

    emit_function_signature_candidate(func_start, sig) <--
        emit_function(func_start, _, _),
        !func_has_known_extern_sig(func_start),
        !func_param_types(func_start, _),
        emit_function_return_type_xtype_candidate(func_start, ret_type),
        let sig = Signature { sig_args: Arc::new(vec![]), sig_res: *ret_type, sig_cc: CallConv::default() };

    emit_function_signature_candidate(func_start, sig) <--
        emit_function(func_start, _, _),
        !func_has_known_extern_sig(func_start),
        !func_param_types(func_start, _),
        !emit_function_return_type_xtype_candidate(func_start, _),
        let sig = Signature { sig_args: Arc::new(vec![]), sig_res: XType::Xvoid, sig_cc: CallConv::default() };


    call_arg_position_allowed(call_addr, pos) <--
        call_has_arg_at_position(call_addr, pos),
        !call_has_known_signature(call_addr, _, _, _);

    call_arg_position_allowed(call_addr, pos) <--
        call_has_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, _, _),
        known_varargs_function(name, _);

    call_arg_position_allowed(call_addr, pos) <--
        call_has_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, sig_arg_count, _),
        !known_varargs_function(name, _),
        !abi_shared_arg_slots(true),
        if pos < sig_arg_count;

    call_arg_position_allowed(call_addr, pos) <--
        call_has_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, _, _),
        !known_varargs_function(name, _),
        known_extern_signature(name, _, _, arg_types),
        abi_shared_arg_slots(true),
        abi_first_stack_arg_position(first_stack),
        if *pos < arg_types.len(),
        if *pos >= *first_stack || !matches!(arg_types[*pos], XType::Xfloat | XType::Xsingle);

    call_arg_mapping(*call_addr, pos, rtl_reg) <--
        call_arg_setup_detected(defaddr, dst_reg, call_addr),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, dst_reg, def_id),
        xtl_canonical(def_id, rtl_reg),
        abi_int_arg_position(dst_reg, pos),
        call_arg_position_allowed(call_addr, pos),
        instr_in_function(defaddr, _func_start);

    call_arg_mapping(*call_addr, pos, rtl_reg) <--
        tailcall_arg_setup_detected(defaddr, dst_reg, call_addr),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, dst_reg, def_id),
        xtl_canonical(def_id, rtl_reg),
        abi_int_arg_position(dst_reg, pos),
        call_arg_position_allowed(call_addr, pos),
        instr_in_function(defaddr, _func_start);

    // If a higher explicit setup proves the call's arity, every lower ABI
    // position exists.  Bind a lower register that still holds this function's
    // incoming parameter directly to that parameter instead of letting the
    // positional aggregator synthesize an uninitialized placeholder.  This is
    // the common VS2013 tail-wrapper shape `f(p0, 0)` where only RDX is written
    // immediately before the COFF import jump.
    call_arg_mapping(*call_addr, pos, canonical) <--
        call_has_arg_at_position(call_addr, pos),
        call_arg_position_allowed(call_addr, pos),
        abi_int_arg_position(mreg, pos),
        instr_in_function(call_addr, func_start),
        arg_reg_param_live_at(func_start, call_addr, mreg),
        reg_xtl(func_start, mreg, incoming),
        xtl_canonical(incoming, canonical);

    // R3: outgoing STACK arguments anchored in the ENTRY frame so they reconcile against the call's RSP; an approximation that may under-recover an arg stored in an earlier block but never fabricates one for a local.
    #[local] relation og_arg_store(Address, i64, RTLReg, Address);
    og_arg_store(*st_addr, abs_slot, *src_rtl, *func) <--
        ltl_inst(st_addr, ?LTLInst::Lstore(_, Addressing::Aindexed(ofs), args, src)),
        !win64_home_spill_candidate(st_addr, _, _, _),
        if args.len() == 1 && args[0] == Mreg::SP,
        if *ofs >= 0,
        instr_in_function(st_addr, func),
        direct_stack_operand(st_addr, Mreg::SP, ofs, _),
        !stack_var(func, st_addr, *ofs, _),
        sp_entry_ofs(func, st_addr, sp_st),
        let abs_slot = *ofs + sp_st.0,
        reg_rtl(st_addr, *src, src_rtl);

    // Simple [rsp+disp] = reg stores lift through Lsetstack, so recover their entry-anchored slot; these shapes are WEAK and need callee-body corroboration, while the Lstore and PUSH shapes stay strong.
    #[local] relation og_arg_store_weak(Address, i64, RTLReg);

    og_arg_store(*st_addr, abs_slot, *src_rtl, *func),
    og_arg_store_weak(*st_addr, abs_slot, *src_rtl) <--
        ltl_inst(st_addr, ?LTLInst::Lsetstack(src, _, ofs, _)),
        !win64_home_spill_candidate(st_addr, _, _, _),
        pmov(st_addr, dst_sym, _),
        op_indirect(dst_sym, _, base_str, idx_str, _, disp, _),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *disp == *ofs,
        instr_in_function(st_addr, func),
        sp_entry_ofs(func, st_addr, sp_st),
        let abs_slot = *disp + sp_st.0,
        reg_rtl(st_addr, *src, src_rtl);

    // Immediate stack stores have no LTL instruction, and mach_imm_stack_init's slot omits absorbed callee-save pushes, so re-anchor the raw displacement with this pass's CFG-derived SP state.
    og_arg_store(*st_addr, abs_slot, *src_rtl, *func),
    og_arg_store_weak(*st_addr, abs_slot, *src_rtl) <--
        mach_imm_stack_init(st_addr, stack_ofs, _, _),
        pmov(st_addr, dst_sym, _),
        op_indirect(dst_sym, _, base_str, idx_str, _, disp, _),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        instr_in_function(st_addr, func),
        sp_entry_ofs(func, st_addr, sp_st),
        let abs_slot = *disp + sp_st.0,
        stack_var(func, st_addr, stack_ofs, src_rtl);

    // The store to abs_slot reaches addr within the block with no intervening re-store, the same kill-on-redef chain as block_last_def; the trailing flag carries weak/strong provenance.
    #[local] relation og_store_reaches(Address, i64, RTLReg, bool);
    og_store_reaches(*next_addr, *abs_slot, *src_rtl, true) <--
        og_arg_store_weak(st_addr, abs_slot, src_rtl),
        next(st_addr, next_addr),
        code_in_block(st_addr, blk),
        code_in_block(next_addr, blk);
    og_store_reaches(*next_addr, *abs_slot, *src_rtl, false) <--
        og_arg_store(st_addr, abs_slot, src_rtl, _),
        !og_arg_store_weak(st_addr, *abs_slot, *src_rtl),
        next(st_addr, next_addr),
        code_in_block(st_addr, blk),
        code_in_block(next_addr, blk);
    og_store_reaches(*next_addr, *abs_slot, *src_rtl, *weak) <--
        og_store_reaches(curr_addr, abs_slot, src_rtl, weak),
        !og_arg_store(curr_addr, *abs_slot, _, _),
        next(curr_addr, next_addr),
        code_in_block(curr_addr, blk),
        code_in_block(next_addr, blk);

    // At a call, map the structurally reaching store using the ABI's first stack position, shadow-space base, and slot width, emitting ungated position fill and an arity-gated value mapping.
    #[local] relation call_outgoing_stack_arg(Node, usize, RTLReg);
    call_outgoing_stack_arg(*call_addr, pos, *src_rtl) <--
        is_call_instruction(call_addr),
        og_store_reaches(call_addr, abs_slot, src_rtl, ?false),
        instr_in_function(call_addr, func),
        sp_entry_ofs(func, call_addr, sp_call),
        let k = *abs_slot - sp_call.0,
        abi_outgoing_stack_base(stack_base),
        abi_first_stack_arg_position(first_pos),
        abi_stack_slot_size(slot_size),
        if k >= *stack_base && (k - *stack_base) % *slot_size == 0,
        let pos = *first_pos + ((k - *stack_base) / *slot_size) as usize;

    // A weak store materializes an argument only where the callee's body proves it reads a stack parameter at that ordinal; with no callee body the slot stays a local, the conservative reading.
    call_outgoing_stack_arg(*call_addr, pos, *src_rtl) <--
        is_call_instruction(call_addr),
        og_store_reaches(call_addr, abs_slot, src_rtl, ?true),
        instr_in_function(call_addr, func),
        sp_entry_ofs(func, call_addr, sp_call),
        let k = *abs_slot - sp_call.0,
        abi_outgoing_stack_base(stack_base),
        abi_first_stack_arg_position(first_pos),
        abi_stack_slot_size(slot_size),
        if k >= *stack_base && (k - *stack_base) % *slot_size == 0,
        let pos = *first_pos + ((k - *stack_base) / *slot_size) as usize,
        call_target_func(call_addr, callee),
        emit_function_stack_param_count(callee, stack_params),
        if pos < *first_pos + *stack_params;

    call_has_arg_evidence(call_addr, pos) <--
        call_outgoing_stack_arg(call_addr, pos, _);

    call_arg_mapping(*call_addr, *pos, *src_rtl) <--
        call_outgoing_stack_arg(call_addr, pos, src_rtl),
        call_arg_position_allowed(call_addr, pos);

    // R3b: outgoing stack args marshaled via PUSH, recovered into the same og_arg_store relation; requiring an in-FUNCTION def of the pushed register separates an argument push from a prologue callee-save.
    #[local] relation push_arg_reg(Address, Mreg);
    push_arg_reg(*addr, Mreg::x86(reg_str)) <--
        instruction(addr, _, _, "PUSH", operand, _, _, _, _, _),
        op_register(operand, reg_str);

    // Track pushed registers in the reaching-def lattice so a cross-block arg value resolves, since a call ends a basic block; only the few registers actually pushed, so the lattice stays bounded.
    reg_of_interest(r) <-- push_arg_reg(_, r);

    // The def supplying the pushed value: intra-block last def (block_last_def), else the cross-block may-reaching def at the push's block entry (with no in-block shadow). Mirrors def_reaches_call / arg_reg_block_reaching_def.
    #[local] relation push_arg_def(Address, Mreg, Address);
    push_arg_def(*push_addr, *mreg, *def_addr) <--
        push_arg_reg(push_addr, mreg),
        block_last_def(push_addr, def_addr, mreg);
    push_arg_def(*push_addr, *mreg, *def_addr) <--
        push_arg_reg(push_addr, mreg),
        !block_last_def(push_addr, _, mreg),
        instr_in_function(push_addr, func),
        code_in_block(push_addr, blk),
        reaching_def_in(func, blk, mreg, defs),
        for def_addr in defs.iter();

    og_arg_store(*push_addr, abs_slot, *src_rtl, *func) <--
        push_arg_def(push_addr, mreg, def_addr),
        asm_effective_def(def_addr, *mreg),
        reg_rtl(def_addr, *mreg, src_rtl),
        instr_in_function(push_addr, func),
        sp_entry_ofs(func, push_addr, sp_push),
        abi_stack_slot_size(slot_size),
        let abs_slot = sp_push.0 - *slot_size;

    // R3c: an outgoing stack arg marshaled as push $imm, synthesized as a fresh const reg spliced onto every CFG edge into the call so the constant dominates it.
    #[local] relation push_arg_imm(Address, i64);
    push_arg_imm(*addr, *imm) <--
        instruction(addr, _, _, "PUSH", operand, _, _, _, _, _),
        op_immediate(operand, imm, _);

    // The call this immediate push set up an arg for is its own block's terminator; uses a DIRECT Lcall check, since is_call_instruction depends on emit_function and would form an unstratifiable cycle.
    #[local] relation is_direct_call_insn(Node);
    is_direct_call_insn(addr) <-- ltl_inst(addr, ?LTLInst::Lcall(_));
    is_direct_call_insn(addr) <-- ltl_inst(addr, ?LTLInst::Ltailcall(_));

    #[local] relation imm_push_call(Address, i64, Node);
    imm_push_call(*push_addr, *imm, *call_addr) <--
        push_arg_imm(push_addr, imm),
        code_in_block(push_addr, blk),
        code_in_block(call_addr, blk),
        is_direct_call_insn(call_addr),
        if push_addr < call_addr;

    og_arg_store(*push_addr, abs_slot, fresh_reg, *func) <--
        imm_push_call(push_addr, _imm, _call),
        instr_in_function(push_addr, func),
        sp_entry_ofs(func, push_addr, sp_push),
        abi_stack_slot_size(slot_size),
        let abs_slot = sp_push.0 - *slot_size,
        let fresh_reg = fresh_xtl_reg(*push_addr, Mreg::AX);

    // Define the constant at a synthetic node and splice it before the call (dominates it).
    rtl_inst_candidate(synth, inst), instr_in_function(synth, *func) <--
        imm_push_call(push_addr, imm, _call),
        instr_in_function(push_addr, func),
        let synth = *push_addr | (1u64 << 62),
        let fresh_reg = fresh_xtl_reg(*push_addr, Mreg::AX),
        let inst = RTLInst::Iop(Operation::Olongconst(*imm), Arc::new(vec![]), fresh_reg);

    rtl_edge_negated(pred, *call_addr) <--
        imm_push_call(_push_addr, _imm, call_addr),
        rtl_next(pred, call_addr);
    rtl_succ_candidate(*pred, synth), instr_in_function(synth, *func) <--
        imm_push_call(push_addr, _imm, call_addr),
        rtl_next(pred, call_addr),
        instr_in_function(push_addr, func),
        let synth = *push_addr | (1u64 << 62);
    rtl_succ_candidate(synth, *call_addr) <--
        imm_push_call(push_addr, _imm, call_addr),
        let synth = *push_addr | (1u64 << 62);


    relation arg_setup_candidate(Node, Mreg, Node);
    relation call_clobbers_arg_reg(Node, Mreg, Node);
    relation is_call_instruction(Node);

    is_call_instruction(addr) <-- ltl_inst(addr, ?LTLInst::Lcall(_));
    is_call_instruction(addr) <-- ltl_inst(addr, ?LTLInst::Ltailcall(_));
    is_call_instruction(addr) <--
        ltl_inst(addr, ?LTLInst::Lbranch(Either::Right(target))),
        emit_function(_, _, target);

    lattice call_backward_reach(Node, Node, ascent::Dual<i64>);

    call_backward_reach(call_addr, call_addr, ascent::Dual(0)) <--
        is_call_instruction(call_addr);

    call_backward_reach(call_addr, prev, ascent::Dual(steps.0 + 1)) <--
        call_backward_reach(call_addr, curr, steps),
        if steps.0 < 128,
        next(prev, curr),
        instr_in_function(call_addr, func_start),
        instr_in_function(prev, func_start);

    call_backward_reach(call_addr, prev, ascent::Dual(steps.0 + 1)) <--
        call_backward_reach(call_addr, curr, steps),
        if steps.0 < 128,
        ltl_succ(prev, curr),
        instr_in_function(call_addr, func_start),
        instr_in_function(prev, func_start);

    // Split from arg_setup_candidate so corroboration counting doesn't depend on its own gated output.
    relation arg_setup_candidate_real(Node, Mreg, Node);

    arg_setup_candidate_real(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lop(_, _, dst_reg)),
        is_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    arg_setup_candidate_real(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lgetstack(_slot, _ofs, _typ, dst_reg)),
        is_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    arg_setup_candidate_real(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lload(_, _, _, dst_reg)),
        is_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    // c19: veto crediting an arg position a body-analysed callee provably does not take, since every call site of a 0-param callee carries the same leftover regs and cross-site agreement is not evidence.
    #[local] relation callee_takes_no_arg_at(Node, Mreg);
    callee_takes_no_arg_at(call_addr, mreg) <--
        arg_setup_candidate_real(_, mreg, call_addr),
        call_target_func(call_addr, callee),
        func_stacksz(callee, _, _, _),
        !func_param_validated(callee, mreg);

    arg_setup_candidate(defaddr, mreg, call_addr) <--
        arg_setup_candidate_real(defaddr, mreg, call_addr),
        !callee_takes_no_arg_at(call_addr, mreg);

    // Fix 1: an ABI arg register that is a validated parameter and is not redefined before a resolved call still holds the incoming value, so forward it as a call arg.
    relation call_has_resolved_callee(Node);
    call_has_resolved_callee(call_addr) <-- call_target_func(call_addr, _);
    call_has_resolved_callee(call_addr) <-- call_has_known_signature(call_addr, _, _, _);

    // arg_reg_param_live_at gates the forward on MUST-liveness, or every resolved call receives a copy of every register including long-dead ones; fabrication is prevented downstream.
    #[local] relation forwarded_param_candidate(Node, Mreg, Node);
    forwarded_param_candidate(func_start, mreg, call_addr) <--
        is_arg_reg(mreg),
        is_call_instruction(call_addr),
        instr_in_function(call_addr, func_start),
        call_has_resolved_callee(call_addr),
        arg_reg_param_live_at(func_start, call_addr, mreg);

    // Corroboration gate: a forwarded still-live param materializes only when the callee validates the position or a second call site agrees, preventing arity fabricated from leftover state.
    #[local] relation forwarding_corroborated(Node, Mreg);

    // Callee-side, internal: the callee's own body validates the register as one of its params.
    forwarding_corroborated(call_addr, mreg) <--
        forwarded_param_candidate(_, mreg, call_addr),
        call_target_func(call_addr, callee),
        func_param_validated(callee, mreg);

    // Callee-side known signature: the GP-class param count covers the GP position, since pure-float params occupy XMM and must not corroborate a forwarded GP register.
    forwarding_corroborated(call_addr, mreg) <--
        forwarded_param_candidate(_, mreg, call_addr),
        call_known_sig_gp_position(call_addr, pos),
        abi_int_arg_position(mreg, pos);

    // Cross-site agreement: another distinct call site of the same callee also carries this position.
    #[local] relation callee_arg_reg_site(Address, Mreg, Node);
    callee_arg_reg_site(callee, mreg, call_addr) <--
        call_target_func(call_addr, callee),
        arg_setup_candidate_real(_, mreg, call_addr);
    callee_arg_reg_site(callee, mreg, call_addr) <--
        call_target_func(call_addr, callee),
        forwarded_param_candidate(_, mreg, call_addr);

    forwarding_corroborated(call_addr, mreg) <--
        forwarded_param_candidate(_, mreg, call_addr),
        call_target_func(call_addr, callee),
        callee_arg_reg_site(callee, mreg, other_site),
        if *other_site != *call_addr;

    arg_setup_candidate(func_start, mreg, call_addr) <--
        forwarded_param_candidate(func_start, mreg, call_addr),
        forwarding_corroborated(call_addr, mreg);

    // Float analog of the pass-through forward: an incoming XMM param still live at a resolved call is forwarded as that call's float argument, under the same corroboration discipline as the integer forward.
    #[local] relation float_forwarded_param_candidate(Node, Mreg, Node);
    float_forwarded_param_candidate(func_start, mreg, call_addr) <--
        is_xmm_arg_reg(mreg),
        is_call_instruction(call_addr),
        instr_in_function(call_addr, func_start),
        call_has_resolved_callee(call_addr),
        arg_reg_param_live_at(func_start, call_addr, mreg);

    #[local] relation float_forwarding_corroborated(Node, Mreg);
    float_forwarding_corroborated(call_addr, mreg) <--
        float_forwarded_param_candidate(_, mreg, call_addr),
        call_target_func(call_addr, callee),
        func_float_param_validated(callee, mreg, _);
    float_forwarding_corroborated(call_addr, mreg) <--
        float_forwarded_param_candidate(_, mreg, call_addr),
        call_has_known_signature(call_addr, name, _, _),
        !known_varargs_function(name, _),
        known_extern_signature(name, _, _, arg_types),
        abi_float_arg_position(mreg, pos),
        !abi_shared_arg_slots(true),
        if arg_types.iter().filter(|t| matches!(t, XType::Xfloat | XType::Xsingle)).count() > *pos;
    float_forwarding_corroborated(call_addr, mreg) <--
        float_forwarded_param_candidate(_, mreg, call_addr),
        call_has_known_signature(call_addr, name, _, _),
        !known_varargs_function(name, _),
        known_extern_signature(name, _, _, arg_types),
        abi_float_arg_position(mreg, pos),
        abi_shared_arg_slots(true),
        if *pos < arg_types.len(),
        if matches!(arg_types[*pos], XType::Xfloat | XType::Xsingle);
    #[local] relation callee_float_arg_reg_site(Address, Mreg, Node);
    callee_float_arg_reg_site(callee, mreg, call_addr) <--
        call_target_func(call_addr, callee),
        float_arg_setup_candidate(_, mreg, call_addr);
    callee_float_arg_reg_site(callee, mreg, call_addr) <--
        call_target_func(call_addr, callee),
        float_forwarded_param_candidate(_, mreg, call_addr);
    float_forwarding_corroborated(call_addr, mreg) <--
        float_forwarded_param_candidate(_, mreg, call_addr),
        call_target_func(call_addr, callee),
        callee_float_arg_reg_site(callee, mreg, other_site),
        if *other_site != *call_addr;

    float_arg_setup_candidate(func_start, mreg, call_addr) <--
        float_forwarded_param_candidate(func_start, mreg, call_addr),
        float_forwarding_corroborated(call_addr, mreg);

    // The old order-based between-call clobber suppression is removed: arg_setup_candidate now comes from def_reaches_call, which already excludes any def clobbered before the call.

    // In-block reaching def of an arg register at a call is THE value the call receives by intra-block dominance; restricted to setup candidates so it never overrides the live-in param fallback.
    relation arg_reg_block_reaching_def(Node, Mreg, Node);
    arg_reg_block_reaching_def(call_addr, mreg, def_addr) <--
        is_call_instruction(call_addr),
        block_last_def(call_addr, def_addr, mreg),
        asm_effective_def(def_addr, mreg),
        is_arg_reg(mreg),
        arg_setup_candidate(def_addr, mreg, call_addr);

    // Same intra-block dominance for XMM/float arg regs: arg_reg_block_reaching_def was only populated for integer regs, so the float clobber suppression below never fired and a back-edge-reachable later XMM def outranked the correct in-block def (A2 dropped float arg).
    arg_reg_block_reaching_def(call_addr, mreg, def_addr) <--
        is_call_instruction(call_addr),
        block_last_def(call_addr, def_addr, mreg),
        asm_effective_def(def_addr, mreg),
        is_xmm_arg_reg(mreg),
        float_arg_setup_candidate(def_addr, mreg, call_addr);

    // With an in-block reaching def, every other candidate can only reach the call around a back-edge that the in-block def kills, so suppress them and respect intra-block dominance.
    call_clobbers_arg_reg(other_def, mreg, call_addr) <--
        arg_reg_block_reaching_def(call_addr, mreg, in_block_def),
        arg_setup_candidate(other_def, mreg, call_addr),
        if *other_def != *in_block_def;

    call_clobbers_arg_reg(other_def, mreg, call_addr) <--
        arg_reg_block_reaching_def(call_addr, mreg, in_block_def),
        float_arg_setup_candidate(other_def, mreg, call_addr),
        if *other_def != *in_block_def;

    // Among multiple structurally-reaching arg setups, keep one deterministic representative (max def address); this picks a canonical member of a proven set, not reachability.
    call_clobbers_arg_reg(defaddr, mreg, call_addr) <--
        arg_setup_candidate(defaddr, mreg, call_addr),
        !arg_reg_block_reaching_def(call_addr, mreg, defaddr),
        arg_setup_candidate(other_def, mreg, call_addr),
        if *other_def > *defaddr;

    call_clobbers_arg_reg(defaddr, mreg, call_addr) <--
        float_arg_setup_candidate(defaddr, mreg, call_addr),
        !arg_reg_block_reaching_def(call_addr, mreg, defaddr),
        float_arg_setup_candidate(other_def, mreg, call_addr),
        if *other_def > *defaddr;

    relation call_arg_setup_detected(Node, Mreg, Node);

    call_arg_setup_detected(defaddr, mreg, call_addr) <--
        arg_setup_candidate(defaddr, mreg, call_addr),
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        !call_clobbers_arg_reg(defaddr, mreg, call_addr);

    relation tailcall_arg_setup_detected(Node, Mreg, Node);

    tailcall_arg_setup_detected(defaddr, mreg, call_addr) <--
        arg_setup_candidate(defaddr, mreg, call_addr),
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        !call_clobbers_arg_reg(defaddr, mreg, call_addr);

    tailcall_arg_setup_detected(defaddr, mreg, call_addr) <--
        arg_setup_candidate(defaddr, mreg, call_addr),
        ltl_inst(call_addr, ?LTLInst::Lbranch(Either::Right(target))),
        emit_function(_, _, target),
        !call_clobbers_arg_reg(defaddr, mreg, call_addr);

    // Call arg positions use max-evidence approach instead of requiring consecutive regs
    relation call_has_arg_evidence(Node, usize);

    call_has_arg_evidence(call_addr, pos) <--
        call_arg_setup_detected(_, mreg, call_addr),
        abi_int_arg_position(mreg, pos);
    call_has_arg_evidence(call_addr, pos) <--
        tailcall_arg_setup_detected(_, mreg, call_addr),
        abi_int_arg_position(mreg, pos);

    relation call_max_arg_position(Node, usize);

    call_max_arg_position(call_addr, max_pos) <--
        call_has_arg_evidence(call_addr, _),
        agg max_pos = ascent::aggregators::max(pos) in call_has_arg_evidence(call_addr, pos);

    relation call_has_arg_at_position(Node, usize);

    // ABI position-fill: arguments occupy contiguous ordinal slots including stack positions, with the positive recursion downstream of the aggregate.
    call_has_arg_at_position(call_addr, *max_pos) <--
        call_max_arg_position(call_addr, max_pos);
    call_has_arg_at_position(call_addr, p_lower) <--
        call_has_arg_at_position(call_addr, p),
        if *p >= 1,
        let p_lower = *p - 1;

    // Float arg setup detection at call sites (XMM0-XMM7, independent from integer args)
    relation float_arg_setup_candidate(Node, Mreg, Node);

    float_arg_setup_candidate(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lop(_, _, dst_reg)),
        is_xmm_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    float_arg_setup_candidate(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lgetstack(_slot, _ofs, _typ, dst_reg)),
        is_xmm_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    float_arg_setup_candidate(*defaddr, *dst_reg, *call_addr) <--
        ltl_inst(defaddr, ?LTLInst::Lload(_, _, _, dst_reg)),
        is_xmm_arg_reg(dst_reg),
        def_reaches_call(call_addr, defaddr, dst_reg);

    relation call_float_arg_setup_detected(Node, Mreg, Node);

    call_float_arg_setup_detected(defaddr, mreg, call_addr) <--
        float_arg_setup_candidate(defaddr, mreg, call_addr),
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        !call_clobbers_arg_reg(defaddr, mreg, call_addr);

    call_float_arg_setup_detected(defaddr, mreg, call_addr) <--
        float_arg_setup_candidate(defaddr, mreg, call_addr),
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        !call_clobbers_arg_reg(defaddr, mreg, call_addr);

    relation call_has_float_arg_evidence(Node, usize);

    call_has_float_arg_evidence(call_addr, pos) <--
        call_float_arg_setup_detected(_, mreg, call_addr),
        abi_float_arg_position(mreg, pos);

    relation call_float_arg_max_position(Node, usize);

    call_float_arg_max_position(call_addr, max_pos) <--
        call_has_float_arg_evidence(call_addr, _),
        agg max_pos = ascent::aggregators::max(pos) in call_has_float_arg_evidence(call_addr, pos);

    relation call_has_float_arg_at_position(Node, usize);

    // SysV XMM registers form their own compact sequence, so evidence for Xk fills X0..Xk; Windows shares slots with GP arguments, so only direct evidence is valid there.
    call_has_float_arg_at_position(call_addr, pos) <--
        call_has_float_arg_evidence(call_addr, pos),
        abi_shared_arg_slots(true);
    call_has_float_arg_at_position(call_addr, *max_pos) <--
        call_float_arg_max_position(call_addr, max_pos),
        !abi_shared_arg_slots(true);
    call_has_float_arg_at_position(call_addr, lower) <--
        call_has_float_arg_at_position(call_addr, pos),
        !abi_shared_arg_slots(true),
        if *pos > 0,
        let lower = *pos - 1;

    // Gate XMM/float-arg positions by the callee signature, mirroring the integer path; its remaining job is class filtering for stray same-function XMM defs that reach the call without being args.
    relation call_float_arg_position_allowed(Node, usize);

    // No known signature, direct internal target whose inferred param count covers this ordinal.
    call_float_arg_position_allowed(call_addr, pos) <--
        call_has_float_arg_at_position(call_addr, pos),
        !call_has_known_signature(call_addr, _, _, _),
        call_target_func(call_addr, target),
        func_has_param_at_position(target, pos);

    // No known signature and the direct target has no recovered param info: stay permissive (mirror of the integer rule) rather than dropping real float args.
    call_float_arg_position_allowed(call_addr, pos) <--
        call_has_float_arg_at_position(call_addr, pos),
        !call_has_known_signature(call_addr, _, _, _),
        call_target_func(call_addr, target),
        !func_has_param_at_position(target, _);

    // Known varargs function (printf-family with %f, etc.): allow any float position.
    call_float_arg_position_allowed(call_addr, pos) <--
        call_has_float_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, _, _),
        known_varargs_function(name, _);

    // Known non-varargs signature: allow float position k only if the signature declares more than k float-valued (Xfloat/Xsingle) parameters. Pointer args (e.g. char*) travel in integer regs and do not count toward XMM positions.
    call_float_arg_position_allowed(call_addr, pos) <--
        call_has_float_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, _, _),
        !known_varargs_function(name, _),
        known_extern_signature(name, _, _, arg_types),
        !abi_shared_arg_slots(true),
        if arg_types.iter().filter(|t| matches!(t, XType::Xfloat | XType::Xsingle)).count() > *pos;

    // Windows x64 uses the source-language ordinal directly for XMM0..3.
    call_float_arg_position_allowed(call_addr, pos) <--
        call_has_float_arg_at_position(call_addr, pos),
        call_has_known_signature(call_addr, name, _, _),
        !known_varargs_function(name, _),
        known_extern_signature(name, _, _, arg_types),
        abi_shared_arg_slots(true),
        if *pos < arg_types.len(),
        if matches!(arg_types[*pos], XType::Xfloat | XType::Xsingle);

    relation call_float_arg_mapping(Node, usize, RTLReg);

    call_float_arg_mapping(*call_addr, pos, rtl_reg) <--
        call_float_arg_setup_detected(defaddr, dst_reg, call_addr),
        is_def(defaddr, def_id),
        reg_xtl(defaddr, dst_reg, def_id),
        xtl_canonical(def_id, rtl_reg),
        abi_float_arg_position(dst_reg, pos),
        call_float_arg_position_allowed(call_addr, pos);

    // Windows has one ordinal argument sequence, so feed the selected XMM value into the ordinary positional mapping instead of an independent SysV float vector.
    call_arg_mapping(call_addr, pos, reg) <--
        abi_shared_arg_slots(true),
        call_float_arg_mapping(call_addr, pos, reg);

    relation call_float_args_collected(Node, Args);

    call_float_args_collected(call_addr, args) <--
        !abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        agg args = build_call_args(pos, reg) in call_float_arg_mapping(call_addr, pos, reg);

    call_float_args_collected(call_addr, args) <--
        !abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        agg args = build_call_args(pos, reg) in call_float_arg_mapping(call_addr, pos, reg);

    call_float_args_collected(call_addr, Arc::new(vec![])) <--
        !abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Lcall(_)),
        !call_float_arg_mapping(call_addr, _, _);

    call_float_args_collected(call_addr, Arc::new(vec![])) <--
        !abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_)),
        !call_float_arg_mapping(call_addr, _, _);

    call_float_args_collected(call_addr, Arc::new(vec![])) <--
        abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Lcall(_));
    call_float_args_collected(call_addr, Arc::new(vec![])) <--
        abi_shared_arg_slots(true),
        ltl_inst(call_addr, ?LTLInst::Ltailcall(_));


    // function_entry_dist: entry-BFS distance capped at FUNC_ARG_DIST, surviving ONLY as prologue-locality pattern bounds, never as a dataflow or reachability decision.
    function_entry_dist(addr, node, 0) <-- emit_function(addr, _, node);

    function_entry_dist(func, next_node, new_dist) <--
        function_entry_dist(func, curr_node, dist),
        next(curr_node, next_node),
        instr_in_function(next_node, func),
        if *dist < FUNC_ARG_DIST,
        let new_dist = dist + 1;


    function_entry_count(func, node, min_dist) <--
        function_entry_dist(func, node, _),
        agg min_dist = ascent::aggregators::min(d) in function_entry_dist(func, node, d);

    // instr_order_in_func / instr_min_order deleted: the BFS-order proxy is fully replaced by def_reaches_return / def_reaches_call / reaching_def_in, since order must never decide reachability.

    relation arg_reg_used_early_in_func(Address, Address, Mreg);
    relation func_float_arg_used_undefined(Address, RTLReg);
    relation emit_function_float_param(Address, RTLReg);

    func_float_arg_used_undefined(func_start, id), reg_xtl(func_start, mreg, id) <--
        func_float_param_validated(func_start, mreg, _),
        let id = fresh_xtl_reg(*func_start, *mreg);

    emit_function_float_param(func_start, *canonical) <--
        func_float_arg_used_undefined(func_start, v),
        xtl_canonical(v, canonical);

    // A PUSH's operand read is stack bookkeeping, not a data use, so it must not turn a live-from-entry arg register into parameter evidence; a push never survives lifting to anchor body dataflow.
    #[local] relation push_operand_use(Address);
    push_operand_use(addr) <--
        instruction(addr, _, _, "PUSH", _, _, _, _, _, _);

    // An arg register used where it still holds the incoming caller value IS a parameter use, exactly; this one structural rule replaces the six entry-distance window variants.
    arg_reg_used_early_in_func(func_start, addr, *reg) <--
        arg_reg_param_live_at(func_start, addr, reg),
        is_arg_reg(reg),
        reg_use(addr, reg),
        !push_operand_use(addr);

    relation arg_reg_has_external_def(Address, Mreg);

    arg_reg_has_external_def(func_start, *mreg) <--
        is_arg_reg(mreg),
        arg_reg_param_live_at(func_start, use_addr, mreg),
        reg_use(use_addr, mreg),
        !push_operand_use(use_addr);

    relation arg_reg_used_very_early(Address, Mreg);
    // (Dead internal-def helpers removed: arg_reg_used_very_early no longer references them; arg_reg_param_live_at already excludes a register that has an internal def before the use.)

    arg_reg_used_very_early(func_start, *mreg) <--
        is_arg_reg(mreg),
        arg_reg_param_live_at(func_start, early_addr, mreg),
        reg_use(early_addr, mreg),
        !push_operand_use(early_addr);

    relation arg_reg_used_no_def(Address, Mreg);

    // A self-XOR is a zeroing idiom that defines without reading, so exclude its spurious operand or the zeroed register is wrongly credited as a parameter.
    #[local] relation ltl_self_xor_zero(Address, Mreg);
    ltl_self_xor_zero(addr, *dst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, args, dst)),
        if matches!(op, Operation::Oxor | Operation::Oxorl),
        if !args.is_empty() && args.iter().all(|a| a == dst);

    arg_reg_used_no_def(func_start, mreg) <--
        is_arg_reg(mreg),
        arg_reg_param_live_at(func_start, use_addr, mreg),
        ltl_inst_uses_mreg(use_addr, mreg),
        !ltl_self_xor_zero(use_addr, mreg);

    relation arg_reg_spilled_to_stack(Address, Mreg);

    // An arg register spilled while it still holds the incoming caller value is a parameter spill; the three store shapes are kept, with the distance window replaced by arg_reg_param_live_at.
    arg_reg_spilled_to_stack(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, _slot, _ofs, _typ)),
        is_arg_reg(src),
        arg_reg_param_live_at(func_start, addr, src);

    arg_reg_spilled_to_stack(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, Addressing::Ainstack(_), _, src)),
        is_arg_reg(src),
        arg_reg_param_live_at(func_start, addr, src);

    arg_reg_spilled_to_stack(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, Addressing::Aindexed(ofs), args, src)),
        if *ofs < 0,
        for arg in args.iter(),
        if *arg == Mreg::BP,
        direct_stack_operand(addr, Mreg::BP, ofs, _),
        bp_base_at(func_start, addr, _),
        is_arg_reg(src),
        arg_reg_param_live_at(func_start, addr, src);

    relation arg_reg_copied_early(Address, Mreg, Mreg);
    // Same copies with the copy site kept, so the consumption check can follow the copy's OWN def-use edge instead of a distance window.
    relation arg_reg_copy_site(Address, Address, Mreg, Mreg);
    relation arg_reg_used_via_copy(Address, Mreg);

    // An arg register move-copied to a non-arg register while it still holds the caller value: the source is a parameter forwarded through the copy. Window + external/no-def variants -> arg_reg_param_live_at.
    arg_reg_copy_site(func_start, *addr, *src, *dst) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Omove, srcs, dst)),
        if srcs.len() == 1,
        for src in srcs.iter(),
        is_arg_reg(src),
        !is_arg_reg(dst),
        if *dst != Mreg::SP && *dst != Mreg::BP,
        arg_reg_param_live_at(func_start, addr, src);

    // Frameless (-O2 FPO) functions forward a parameter into rbp as scratch, so gate the dst==BP exclusion on whether the function ever establishes a frame pointer or every pointer param held in rbp loses its pointer evidence.
    arg_reg_copy_site(func_start, *addr, *src, *dst) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Omove, srcs, dst)),
        if srcs.len() == 1,
        for src in srcs.iter(),
        is_arg_reg(src),
        if *dst == Mreg::BP,
        arg_reg_param_live_at(func_start, addr, src),
        !func_ever_sets_frame_pointer(func_start);

    arg_reg_copied_early(func_start, src, dst) <--
        arg_reg_copy_site(func_start, _, src, dst);

    // The copied value is actually consumed (the copy's own def reaches a use), replacing a distance-512 window that could credit a use of a different, later def of dst.
    arg_reg_used_via_copy(func_start, *arg_reg) <--
        arg_reg_copy_site(func_start, copy_addr, arg_reg, dst_reg),
        reg_def_used(copy_addr, dst_reg, use_addr),
        instr_in_function(use_addr, func_start);

    relation func_arg_reg_used(Address, Mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_used_early_in_func(func_start, _, mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_has_external_def(func_start, mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_used_no_def(func_start, mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_used_very_early(func_start, mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_spilled_to_stack(func_start, mreg);

    func_arg_reg_used(func_start, mreg) <--
        arg_reg_used_via_copy(func_start, mreg);

    // SysV variadic prologue: the 8 movaps stores spill the XMM register save area, not real float params; detect an XMM arg reg stack-stored near entry with no in-function def chain.
    relation xmm_arg_spilled_in_prologue(Address, Mreg);

    // XMM arg reg stored to a BP/SP-relative stack slot while it holds the incoming caller value = variadic XMM save-area spill. Window + external/no-def variants -> arg_reg_param_live_at.
    xmm_arg_spilled_in_prologue(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, Addressing::Aindexed(_), args, src)),
        is_xmm_arg_reg(src),
        for arg in args.iter(),
        if *arg == Mreg::BP || *arg == Mreg::SP,
        stack_claim_base_at(func_start, addr, *arg),
        arg_reg_param_live_at(func_start, addr, src);

    // setstack form.
    xmm_arg_spilled_in_prologue(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, _slot, _ofs, _typ)),
        is_xmm_arg_reg(src),
        arg_reg_param_live_at(func_start, addr, src);

    // Lstore (Ainstack) form.
    xmm_arg_spilled_in_prologue(func_start, *src) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, Addressing::Ainstack(_), _, src)),
        is_xmm_arg_reg(src),
        arg_reg_param_live_at(func_start, addr, src);

    // Function has the canonical variadic-prologue XMM register save (all 8 XMMs spilled); real functions never spill all 8 XMM arg regs at entry.
    relation func_has_variadic_xmm_prologue(Address);
    func_has_variadic_xmm_prologue(func_start) <--
        !abi_shared_arg_slots(true),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X0),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X1),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X2),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X3),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X4),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X5),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X6),
        xmm_arg_spilled_in_prologue(func_start, Mreg::X7);

    relation func_float_param_used(Address, Mreg);

    // (Dead helper xmm_arg_use_has_internal_def removed: func_float_param_used now uses arg_reg_param_live_at, which already excludes an XMM register with an internal def.)

    // An XMM arg register used while it still holds the incoming caller value is a float parameter (excluding the variadic XMM save-area prologue). Window + internal/external-def variants -> arg_reg_param_live_at (covers X0-X7 by register value, subsumes the no-internal-def condition).
    func_float_param_used(func_start, *mreg) <--
        is_float_arg_reg(mreg),
        arg_reg_param_live_at(func_start, addr, mreg),
        ltl_inst_uses_mreg(addr, mreg),
        !func_has_variadic_xmm_prologue(func_start);

    // 3.8 R1: the XMM read of a FUSED float op (float_load_op binary form reads its dst) is float-param evidence too; it has no ltl_inst, so ltl_inst_uses_mreg above never sees it and a function whose ONLY float-param use is fused (`addsd k(%rip),%xmm0; ret`) validated no float params at all.
    func_float_param_used(func_start, *mreg) <--
        is_float_arg_reg(mreg),
        arg_reg_param_live_at(func_start, addr, mreg),
        float_load_op(addr, _, _, _, _, mreg, false),
        !has_ltl_op(addr),
        !func_has_variadic_xmm_prologue(func_start);

    // Positional chain: X_k validates only if X_{k-1} did, so a stray param-live use cannot fabricate k+1 float params. Known limit (DF-3): an unused lower-position sibling cannot validate here.
    relation func_float_param_validated(Address, Mreg, usize);

    func_float_param_validated(func_start, mreg, pos) <--
        abi_shared_arg_slots(true),
        func_float_param_used(func_start, mreg),
        abi_float_arg_position(mreg, pos);

    func_float_param_validated(func_start, mreg, pos) <--
        !abi_shared_arg_slots(true),
        func_float_param_used(func_start, mreg),
        abi_float_arg_position(mreg, pos),
        if *pos == 0;
    func_float_param_validated(func_start, mreg, pos) <--
        !abi_shared_arg_slots(true),
        func_float_param_used(func_start, mreg),
        abi_float_arg_position(mreg, pos),
        if *pos > 0,
        let previous = *pos - 1,
        func_float_param_validated(func_start, _, previous);

    relation emit_function_float_param_count(Address, usize);

    emit_function_float_param_count(func_start, count) <--
        func_float_param_validated(func_start, _, _),
        agg count = ascent::aggregators::count() in func_float_param_validated(func_start, _, _);

    emit_function_float_param_count(func_start, 0) <--
        emit_function(func_start, _, _),
        !func_float_param_validated(func_start, _, _);


    #[local] relation incoming_stack_slot(Node, Address, Mreg, i64, usize);

    // ABI-1: consume AsmPass's CFG-safe entry-anchored SP coordinate.  Keep the
    // historical local shape so the downstream ABI formulas stay explicit,
    // but do not recompute or extend the proof in RTL.
    #[local] lattice sp_entry_ofs(Address, Address, Dual<i64>);
    sp_entry_ofs(*func_start, *addr, Dual(*ofs)) <--
        rsp_frame_offset_at(func_start, addr, ofs);

    // A frame-pointer base is use-specific: the defining MOV must reach and
    // dominate this use, and its copy-site SP coordinate must still be known.
    // This rejects conditional copies and uses after an RBP clobber while
    // correctly normalizing copies made after pushes/frame allocation.
    //
    // Dominance is a query, not an all-pairs output.  Every relational
    // consumer below already requires a raw reaching-def edge, except the
    // fused-stack initializer (a normalized write queried at one fused use).
    // The post-fixed-point indexed-operand guard additionally needs decoded
    // definitions and call kills at raw SP/BP-indexed sites.  Build that
    // demand from decoder snapshots only: depending on RTL-derived indexed
    // relations here would pull bp_base_at back into its own aggregate SCC.
    // Materializing only this union avoids O(instructions^2) rows while
    // retaining every observable lookup.
    #[local] relation indexed_dominance_use(Node);
    indexed_dominance_use(node) <--
        indexed_stack_operand(node, base, _, _),
        if matches!(*base, Mreg::SP | Mreg::BP);

    #[local] relation indexed_dominance_operand(Node, Mreg);
    indexed_dominance_operand(node, *reg) <--
        indexed_dominance_use(node),
        asm_reg_use(node, reg);

    #[local] relation reg_def_dominance_query(Address, Node, Node);
    reg_def_dominance_query(*func_start, *def_addr, *use_addr) <--
        raw_reg_def_used(def_addr, _, use_addr),
        real_addr_in_func(def_addr, func_start),
        real_addr_in_func(use_addr, func_start);
    reg_def_dominance_query(*func_start, *call_addr, *use_addr) <--
        instruction(call_addr, _, _, "CALL", _, _, _, _, _, _),
        indexed_dominance_use(use_addr),
        real_addr_in_func(call_addr, func_start),
        real_addr_in_func(use_addr, func_start);
    // Decoder definitions cover ordinary integer, SIMD, and the original
    // writes at addresses later replaced by Lbuiltin.  Pair only matching
    // operands at the small set of indexed sites retained by the imperative
    // ambiguity classifier.
    reg_def_dominance_query(*func_start, *def_addr, *use_addr) <--
        asm_reg_def(def_addr, reg),
        indexed_dominance_operand(use_addr, reg),
        real_addr_in_func(def_addr, func_start),
        real_addr_in_func(use_addr, func_start);
    // The relational stack-initializer consumer needs normalized direct
    // writes only at fused indexed uses.  Raw direct writes paired with every
    // structural SP/BP indexed site are an independent, conservative demand
    // superset; the consumer still applies the exact normalization/fused
    // predicates after dominance is established.
    reg_def_dominance_query(*func_start, *write_addr, *use_addr) <--
        direct_stack_operand(write_addr, _, _, _),
        decoded_memory_write_operand(write_addr, _),
        indexed_dominance_use(use_addr),
        real_addr_in_func(write_addr, func_start),
        real_addr_in_func(use_addr, func_start);

    relation reg_def_dominates_use(Address, Node, Node);
    reg_def_dominates_use(func_start, def_addr, use_addr) <--
        reg_def_dominance_query(func_start, def_addr, use_addr),
        code_in_block(def_addr, block),
        code_in_block(use_addr, block),
        if *def_addr < *use_addr;
    reg_def_dominates_use(func_start, def_addr, use_addr) <--
        reg_def_dominance_query(func_start, def_addr, use_addr),
        code_in_block(def_addr, def_block),
        code_in_block(use_addr, use_block),
        if *def_block != *use_block,
        block_dom_set(func_start, use_block, doms),
        if doms.0.contains(def_block);

    #[local] relation competing_reaching_reg_def(Node, Mreg, Node);
    competing_reaching_reg_def(def_addr, *reg, use_addr) <--
        raw_reg_def_used(def_addr, reg, use_addr),
        raw_reg_def_used(other_def, reg, use_addr),
        if other_def != def_addr;

    bp_base_at(*func_start, *use_addr, sp_ofs.0) <--
        stack_base_move(copy_addr, src, dst),
        if *src == "RSP" && *dst == "RBP",
        bp_frame_at(use_addr, func_start),
        raw_reg_def_used(*copy_addr, ?&Mreg::BP, use_addr),
        reg_def_dominates_use(func_start, copy_addr, use_addr),
        !competing_reaching_reg_def(copy_addr, Mreg::BP, use_addr),
        sp_entry_ofs(func_start, copy_addr, sp_ofs);

    // Shared gate for the historical BP/SP scalar shortcuts. Exact raw base
    // spelling is required for both registers, and BP additionally needs the
    // use-specific reaching/dominating frame-copy proof.
    stack_claim_base_at(func_start, addr, Mreg::SP) <--
        real_addr_in_func(addr, func_start),
        direct_stack_operand(addr, Mreg::SP, _, _),
        rsp_frame_at(addr, func_start);
    stack_claim_base_at(func_start, addr, Mreg::BP) <--
        direct_stack_operand(addr, Mreg::BP, _, _),
        bp_base_at(func_start, addr, _);

    // A direct MOV copy of SP can be used as a stable base for the compiler's
    // Win64 /homeparams stores. Record the SP entry offset at the copy site;
    // later SP adjustments must not change the copied base's coordinate.
    #[local] relation sp_alias_def(Address, Node, Mreg, i64);

    sp_alias_def(func_start, *copy_addr, alias, sp_ofs.0) <--
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if *src_str == "RSP",
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let alias = Mreg::x86(*dst_str),
        if alias != Mreg::SP && alias != Mreg::BP,
        real_addr_in_func(copy_addr, func_start),
        sp_entry_ofs(func_start, copy_addr, sp_ofs);

    // A real LEA can copy SP plus a displacement. Use the raw instruction
    // operand here: its lowered Ainstack displacement may already include the
    // tracked SP offset, while MOV SP copies carry no raw displacement.
    sp_alias_def(func_start, *copy_addr, alias, base_ofs) <--
        plea(copy_addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let alias = Mreg::x86(*dst_str),
        if alias != Mreg::SP && alias != Mreg::BP,
        op_indirect(src, _, base_str, idx_str, _, raw_ofs, _),
        if *base_str == "RSP",
        if *idx_str == "NONE" || idx_str.is_empty(),
        real_addr_in_func(copy_addr, func_start),
        sp_entry_ofs(func_start, copy_addr, sp_ofs),
        let base_ofs = sp_ofs.0 + *raw_ofs;

    // Preserve coordinates through further value-preserving 64-bit copies.
    sp_alias_def(func_start, *copy_addr, dst_reg, *base_ofs) <--
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if is_x86_64_gp_register_name(src_str),
        let src_reg = Mreg::x86(*src_str),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        if dst_reg != Mreg::SP && dst_reg != Mreg::BP,
        real_addr_in_func(copy_addr, func_start),
        sp_base_alias_at(func_start, copy_addr, src_reg, base_ofs);

    // Preserve an entry-SP coordinate through LEA k(alias), but only when the
    // source alias is proved at this exact copy site. The 64-bit destination
    // guard excludes truncating E* forms.
    sp_alias_def(func_start, *copy_addr, dst_reg, base_ofs) <--
        plea(copy_addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        if dst_reg != Mreg::SP && dst_reg != Mreg::BP,
        op_indirect(src, _, base_str, idx_str, _, raw_ofs, _),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_x86_64_gp_register_name(base_str),
        let src_reg = Mreg::x86(*base_str),
        real_addr_in_func(copy_addr, func_start),
        sp_base_alias_at(func_start, copy_addr, src_reg, src_base_ofs),
        let base_ofs = *src_base_ofs + *raw_ofs;

    // A compiler may advance or retreat a copied stack base in place before
    // using it.  The ADD/SUB instruction both reads and defines the register,
    // so the use-site proof below names the reaching pre-definition
    // coordinate while this row records the post-definition coordinate.  The
    // 64-bit register-name guard rejects truncating E* writes.
    sp_alias_def(func_start, *add_addr, alias, base_ofs) <--
        padd(add_addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let alias = Mreg::x86(*dst_str),
        if alias != Mreg::SP && alias != Mreg::BP,
        op_immediate(src, delta, _),
        real_addr_in_func(add_addr, func_start),
        sp_base_alias_at(func_start, add_addr, alias, prior_ofs),
        let base_ofs = *prior_ofs + *delta;

    sp_alias_def(func_start, *sub_addr, alias, base_ofs) <--
        psub(sub_addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let alias = Mreg::x86(*dst_str),
        if alias != Mreg::SP && alias != Mreg::BP,
        op_immediate(src, delta, _),
        real_addr_in_func(sub_addr, func_start),
        sp_base_alias_at(func_start, sub_addr, alias, prior_ofs),
        let base_ofs = *prior_ofs - *delta;

    // May-alias provenance is use-specific rather than a function-wide
    // "ever copied RSP" bit.  A later overwrite therefore kills the alias
    // before a call, return, or unrelated pointer use, while a conditional
    // RSP copy still remains a possible reaching definition.  The raw chain
    // deliberately does not require a known affine SP coordinate.
    #[local] relation sp_may_alias_def(Address, Node, Mreg);
    sp_may_alias_def(func_start, *def_addr, *alias) <--
        sp_alias_def(func_start, def_addr, alias, _);
    sp_may_alias_def(func_start, *copy_addr, alias) <--
        pmov(copy_addr, dst, src),
        op_register(src, "RSP"),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let alias = Mreg::x86(*dst_str),
        if alias != Mreg::SP && alias != Mreg::BP,
        real_addr_in_func(copy_addr, func_start);
    sp_may_alias_def(func_start, *copy_addr, dst_reg) <--
        sp_may_alias_def(func_start, source_def, src_reg),
        raw_reg_def_used(*source_def, *src_reg, copy_addr),
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if is_x86_64_gp_register_name(src_str),
        let seen_src = Mreg::x86(*src_str),
        if seen_src == *src_reg,
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        if dst_reg != Mreg::SP && dst_reg != Mreg::BP,
        real_addr_in_func(copy_addr, func_start);

    #[local] relation sp_may_alias_at(Address, Node, Mreg);
    sp_may_alias_at(func_start, *use_addr, *alias) <--
        sp_may_alias_def(func_start, def_addr, alias),
        raw_reg_def_used(*def_addr, *alias, use_addr),
        real_addr_in_func(use_addr, func_start);

    relation sp_base_alias_at(Address, Node, Mreg, i64);
    sp_base_alias_at(func_start, use_addr, *alias, *base_ofs) <--
        sp_alias_def(func_start, def_addr, alias, base_ofs),
        real_addr_in_func(use_addr, func_start),
        raw_reg_def_used(*def_addr, *alias, use_addr),
        reg_def_dominates_use(func_start, def_addr, use_addr),
        !competing_reaching_reg_def(def_addr, *alias, use_addr);

    #[local] relation sp_based_mem_at(Node, Address, Mreg, i64);
    sp_based_mem_at(addr, func_start, Mreg::SP, sp_ofs.0) <--
        real_addr_in_func(addr, func_start),
        sp_entry_ofs(func_start, addr, sp_ofs);
    sp_based_mem_at(addr, func_start, Mreg::BP, *base_ofs) <--
        bp_base_at(func_start, addr, base_ofs);
    sp_based_mem_at(addr, func_start, *alias, *base_ofs) <--
        sp_base_alias_at(func_start, addr, alias, base_ofs);

    #[local] relation abi_home_arg_position(Mreg, usize);
    abi_home_arg_position(*reg, *pos) <-- abi_int_arg_position(reg, pos);
    abi_home_arg_position(*reg, *pos) <-- abi_float_arg_position(reg, pos);

    // A non-indexed SP (or proven copied-SP) access normalized to one of the
    // four Win64 home cells. This relation is deliberately independent of
    // whether the access is a read or write.
    #[local] relation win64_home_cell(Node, Address, Mreg, i64, usize);
    win64_home_cell(addr, func_start, *base_reg, *disp, *pos) <--
        abi_shared_arg_slots(true),
        direct_stack_operand(addr, base_reg, disp, _),
        sp_based_mem_at(addr, func_start, seen_base, base_ofs),
        if *seen_base == *base_reg,
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack,
        abi_incoming_sp_stack_base(incoming_base),
        abi_outgoing_stack_base(outgoing_base),
        abi_stack_slot_size(slot_size),
        let home_base = *incoming_base - *outgoing_base,
        if *base_ofs + *disp == home_base + (*pos as i64 * *slot_size);

    // Proven materializations of the address of one exact home cell.  This is
    // deliberately stricter than general copied-SP provenance: only a
    // non-indexed LEA into a 64-bit destination starts the proof, after which
    // exact 64-bit register copies may carry it to a use.  In particular,
    // passing RSP itself (or an ambiguous may-alias) is not evidence for
    // &home[pos].
    #[local] relation win64_home_exact_addr_def(Address, Node, Mreg, usize);
    #[local] relation win64_home_exact_addr_at(Address, Node, Mreg, usize);
    #[local] relation win64_home_exact_addr_call(Address, Node, Mreg, usize);
    #[local] relation win64_home_exact_addr_copy_use(Address, Node, Mreg, usize);
    #[local] relation win64_home_addr_origin_def(Address, Node, Node, Mreg, usize);
    #[local] relation win64_home_addr_origin_at(Address, Node, Node, Mreg, usize);
    #[local] relation win64_home_lea_address_taken(Node, Address, usize);

    win64_home_exact_addr_def(func_start, *addr, dst_reg, *pos) <--
        win64_home_cell(addr, func_start, base_reg, raw_disp, pos),
        plea(addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        op_indirect(src, _, base_str, idx_str, _, asm_disp, _),
        if is_x86_64_gp_register_name(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp;

    win64_home_exact_addr_def(func_start, *copy_addr, dst_reg, *pos) <--
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if is_x86_64_gp_register_name(src_str),
        let src_reg = Mreg::x86(*src_str),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        real_addr_in_func(copy_addr, func_start),
        win64_home_exact_addr_at(func_start, copy_addr, src_reg, pos);

    win64_home_exact_addr_at(func_start, use_addr, *reg, *pos) <--
        win64_home_exact_addr_def(func_start, def_addr, reg, pos),
        real_addr_in_func(use_addr, func_start),
        raw_reg_def_used(*def_addr, *reg, use_addr),
        reg_def_dominates_use(func_start, def_addr, use_addr),
        !competing_reaching_reg_def(def_addr, *reg, use_addr);

    win64_home_exact_addr_call(func_start, *call_addr, *reg, *pos) <--
        win64_home_exact_addr_def(func_start, def_addr, reg, pos),
        arg_setup_candidate(def_addr, reg, call_addr),
        real_addr_in_func(call_addr, func_start);

    // The sole value-propagating non-call use admitted for an exact home
    // address is a full-width register MOV.  In particular, an ADD/SUB or a
    // zero-displacement LEA may still be numerically using the address and
    // must not become exempt merely because it also creates an SP alias.
    win64_home_exact_addr_copy_use(func_start, *copy_addr, *src_reg, *pos) <--
        win64_home_exact_addr_at(func_start, copy_addr, src_reg, pos),
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if is_x86_64_gp_register_name(src_str),
        if Mreg::x86(*src_str) == *src_reg,
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str);

    win64_home_addr_origin_def(func_start, *lea_addr, *lea_addr, *reg, *pos) <--
        win64_home_exact_addr_def(func_start, lea_addr, reg, pos),
        plea(lea_addr, _, _);
    win64_home_addr_origin_def(func_start, *origin, *copy_addr, dst_reg, *pos) <--
        win64_home_addr_origin_at(func_start, origin, copy_addr, src_reg, pos),
        pmov(copy_addr, dst, src),
        instruction(copy_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ"),
        op_register(src, src_str),
        if is_x86_64_gp_register_name(src_str),
        let seen_src = Mreg::x86(*src_str),
        if seen_src == *src_reg,
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str);
    win64_home_addr_origin_at(func_start, *origin, use_addr, *reg, *pos) <--
        win64_home_addr_origin_def(func_start, origin, def_addr, reg, pos),
        real_addr_in_func(use_addr, func_start),
        raw_reg_def_used(*def_addr, *reg, use_addr),
        reg_def_dominates_use(func_start, def_addr, use_addr),
        !competing_reaching_reg_def(def_addr, *reg, use_addr);
    win64_home_lea_address_taken(*origin, func_start, *pos) <--
        win64_home_addr_origin_def(func_start, origin, def_addr, reg, pos),
        arg_setup_candidate(def_addr, reg, call_addr),
        real_addr_in_func(call_addr, func_start);

    // Exported because the post-fixed-point canonical-storage selector must
    // compare this closed set with its supported scalar shapes.
    relation win64_home_overlap(Node, Address, usize);
    win64_home_overlap(addr, func_start, *pos) <--
        abi_shared_arg_slots(true),
        direct_stack_operand(addr, base_reg, disp, mem_size),
        if *mem_size > 0,
        sp_based_mem_at(addr, func_start, seen_base, base_ofs),
        if *seen_base == *base_reg,
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack,
        abi_incoming_sp_stack_base(incoming_base),
        abi_outgoing_stack_base(outgoing_base),
        abi_stack_slot_size(slot_size),
        let home_base = *incoming_base - *outgoing_base,
        let slot_start = home_base + (*pos as i64 * *slot_size),
        let access_start = *base_ofs + *disp,
        let access_end = access_start + *mem_size as i64,
        if access_start < slot_start + *slot_size && access_end > slot_start;

    // Positive /homeparams candidate. Requiring the source base to denote
    // entry SP itself (offset zero) excludes outgoing argument stores after a
    // frame allocation that happen to reuse the same entry-frame coordinate.
    #[local] relation win64_home_move_class(Node, usize);
    // 0 = GP integer move, 1 = scalar single, 2 = scalar double. VEX and
    // legacy encodings share a semantic class, but integer/scalar classes do
    // not: equal widths alone cannot justify replacing the reload's value.
    win64_home_move_class(addr, 0) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVQ");
    win64_home_move_class(addr, 1) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOVSS" | "VMOVSS");
    win64_home_move_class(addr, 2) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "MOVSD" | "VMOVSD");

    relation win64_home_spill_candidate(Node, Address, Mreg, usize);
    win64_home_spill_candidate(addr, func_start, src_reg, *pos) <--
        win64_home_move_class(addr, move_class),
        pmov(addr, dst, src),
        op_indirect(dst, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        abi_stack_slot_size(slot_size),
        if *mem_size > 0 && (*mem_size as i64) <= *slot_size,
        op_register(src, src_str),
        let src_reg = Mreg::x86(*src_str),
        if (*move_class == 0 && !is_float_mreg(&src_reg))
            || (*move_class != 0 && is_float_mreg(&src_reg)),
        abi_home_arg_position(src_reg, pos),
        win64_home_cell(addr, func_start, base_reg, disp, pos),
        if *base_reg == Mreg::x86(base_str),
        sp_based_mem_at(addr, func_start, base_reg, base_ofs),
        if *base_ofs == 0,
        arg_reg_param_live_at(func_start, addr, src_reg);

    // The candidate spill must dominate a reload; address ordering alone is
    // not a control-flow proof. Within one basic block, instruction order is
    // sufficient. Across blocks, use the existing dominator lattice.
    #[local] relation home_spill_dominates_reload(Node, Node, Address, usize);
    home_spill_dominates_reload(spill_addr, reload_addr, func_start, *pos) <--
        win64_home_spill_candidate(spill_addr, func_start, _, pos),
        win64_home_cell(reload_addr, func_start, _, _, pos),
        code_in_block(spill_addr, block),
        code_in_block(reload_addr, block),
        if *spill_addr < *reload_addr;
    home_spill_dominates_reload(spill_addr, reload_addr, func_start, *pos) <--
        win64_home_spill_candidate(spill_addr, func_start, _, pos),
        win64_home_cell(reload_addr, func_start, _, _, pos),
        code_in_block(spill_addr, spill_block),
        code_in_block(reload_addr, reload_block),
        if *spill_block != *reload_block,
        block_dom_set(func_start, reload_block, doms),
        if doms.0.contains(spill_block);

    // A width-matched integer arithmetic read of an exactly homed register is
    // as immutable as a MOV reload.  VS2013 commonly consumes /homeparams
    // slots directly (`xor eax,[rsp+24]`, for example), so treating every
    // non-MOV read as mutation needlessly rejects otherwise lossless TUs.
    #[local] relation win64_home_arith_read_candidate(Node, Address, Mreg, usize);
    win64_home_arith_read_candidate(addr, func_start, *param_reg, *pos) <--
        arith_load_op(addr, _, chunk, base_reg, disp, _),
        win64_home_cell(addr, func_start, seen_base, seen_disp, pos),
        if *seen_base == *base_reg && *seen_disp == *disp,
        home_spill_dominates_reload(spill_addr, addr, func_start, pos),
        win64_home_spill_candidate(spill_addr, func_start, param_reg, pos),
        win64_home_move_class(spill_addr, move_class),
        if *move_class == 0,
        pmov(spill_addr, spill_dst, _),
        op_indirect(spill_dst, _, _, spill_idx, _, _, spill_size),
        if *spill_idx == "NONE" || spill_idx.is_empty(),
        if chunk_size_bits(chunk) as usize == *spill_size * 8;

    // A simple reload cell has a must-executed initial home spill, but is not
    // yet necessarily foldable: mutation/address escape checks below decide
    // whether it remains real storage.
    relation win64_home_reload_cell(Node, Address, Mreg, usize);
    win64_home_reload_cell(addr, func_start, *param_reg, *pos) <--
        win64_home_move_class(addr, reload_class),
        pmov(addr, dst, src),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        if (*reload_class == 0 && !is_float_mreg(&dst_reg))
            || (*reload_class != 0 && is_float_mreg(&dst_reg)),
        op_indirect(src, _, base_str, idx_str, _, disp, reload_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        win64_home_cell(addr, func_start, base_reg, disp, pos),
        if *base_reg == Mreg::x86(base_str),
        home_spill_dominates_reload(spill_addr, addr, func_start, pos),
        win64_home_spill_candidate(spill_addr, func_start, param_reg, pos),
        win64_home_move_class(spill_addr, spill_class),
        if *spill_class == *reload_class,
        pmov(spill_addr, spill_dst, _),
        op_indirect(spill_dst, _, _, spill_idx, _, _, spill_size),
        if *spill_idx == "NONE" || spill_idx.is_empty(),
        if *spill_size == *reload_size;

    // Any non-initial write, escaped address, unproved read, or competing
    // initial spill makes the home cell real mutable storage. This is a
    // deliberately function-wide conservative veto: it may retain a local,
    // but it cannot fold away externally observable mutation.
    #[local] relation win64_home_slot_unsafe(Address, usize);
    // Any classified access that is neither the exact initial spill nor a
    // proved reload is conservatively a mutation, escape, or unsupported
    // access. This catches arithmetic through copied-SP aliases, which the
    // literal-SP stack_def/stack_use relations intentionally do not record.
    win64_home_slot_unsafe(func_start, *pos) <--
        win64_home_overlap(addr, func_start, pos),
        !win64_home_spill_candidate(addr, func_start, _, pos),
        !win64_home_reload_cell(addr, func_start, _, pos),
        !win64_home_arith_read_candidate(addr, func_start, _, pos);
    win64_home_slot_unsafe(func_start, *pos) <--
        indexed_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_based_mem_at(addr, func_start, seen_base, _),
        if *seen_base == *base_reg,
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        indexed_stack_operand(addr, Mreg::SP, _, _),
        real_addr_in_func(addr, func_start),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    // A conditional/non-dominating RSP->RBP copy gives an indexed BP access
    // no usable coordinate. If the function ever establishes a frame pointer,
    // conservatively retain all home cells rather than treating that access as
    // unrelated pointer memory.
    win64_home_slot_unsafe(func_start, *pos) <--
        indexed_stack_operand(addr, Mreg::BP, _, _),
        real_addr_in_func(addr, func_start),
        func_ever_sets_frame_pointer(func_start),
        !bp_base_at(func_start, addr, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        direct_stack_operand(addr, Mreg::SP, _, _),
        real_addr_in_func(addr, func_start),
        !sp_entry_ofs(func_start, addr, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        direct_stack_operand(addr, Mreg::BP, _, _),
        real_addr_in_func(addr, func_start),
        !bp_base_at(func_start, addr, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    // If a possibly reaching stack alias lacks one proved coordinate at a
    // memory use, a conditional copy may have changed the base. Retain every
    // home cell rather than folding through that ambiguous access.
    win64_home_slot_unsafe(func_start, *pos) <--
        direct_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_may_alias_at(func_start, addr, base_reg),
        !sp_base_alias_at(func_start, addr, base_reg, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        indexed_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_may_alias_at(func_start, addr, base_reg),
        !sp_base_alias_at(func_start, addr, base_reg, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    // A decoded memory operand with unknown width cannot prove that it stays
    // within one home cell, even when its base coordinate is known.
    win64_home_slot_unsafe(func_start, *pos) <--
        direct_stack_operand(addr, base_reg, _, mem_size),
        if *mem_size == 0,
        real_addr_in_func(addr, func_start),
        sp_based_mem_at(addr, func_start, seen_base, _),
        if *seen_base == *base_reg,
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        indexed_stack_operand(addr, base_reg, _, mem_size),
        if *mem_size == 0,
        real_addr_in_func(addr, func_start),
        sp_based_mem_at(addr, func_start, seen_base, _),
        if *seen_base == *base_reg,
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    // A non-memory use reached by a possible SP alias may escape the frame,
    // including a branch where only one predecessor copied RSP. A later
    // unrelated overwrite no longer poisons the whole function.
    win64_home_slot_unsafe(func_start, *pos) <--
        sp_may_alias_at(func_start, addr, alias),
        real_addr_in_func(addr, func_start),
        asm_reg_use(addr, alias),
        !direct_stack_operand(addr, alias, _, _),
        !indexed_stack_operand(addr, alias, _, _),
        !sp_alias_def(func_start, addr, _, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    // Capstone does not consistently report the ABI's implicit argument and
    // return-register uses on CALL/RET. Use the existing exact may-reaching
    // def lattice so only an alias definition that reaches the boundary is an
    // escape.
    win64_home_slot_unsafe(func_start, *pos) <--
        abi_shared_arg_slots(true),
        sp_may_alias_def(func_start, alias_def, alias),
        abi_int_arg_position(alias, _),
        def_reaches_call(call_addr, alias_def, alias),
        real_addr_in_func(call_addr, func_start),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        abi_shared_arg_slots(true),
        sp_may_alias_def(func_start, alias_def, ?&Mreg::AX),
        def_reaches_return(ret_addr, alias_def, ?&Mreg::AX),
        real_addr_in_func(ret_addr, func_start),
        instruction(ret_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "RET" | "RETF" | "RETFQ"),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        win64_home_spill_candidate(first_addr, func_start, first_reg, pos),
        win64_home_spill_candidate(other_addr, func_start, other_reg, pos),
        if first_addr != other_addr || first_reg != other_reg;

    // Even though an affine copied-SP update can be proved precisely, keep it
    // out of the immutable /homeparams fold.  The post-pass scalar selector
    // can still canonicalize all of its exact accesses to one mutable local.
    win64_home_slot_unsafe(func_start, *pos) <--
        sp_alias_def(func_start, adjust_addr, alias, _),
        padd(adjust_addr, _, _),
        direct_stack_operand(use_addr, alias, _, _),
        sp_base_alias_at(func_start, use_addr, alias, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;
    win64_home_slot_unsafe(func_start, *pos) <--
        sp_alias_def(func_start, adjust_addr, alias, _),
        psub(adjust_addr, _, _),
        direct_stack_operand(use_addr, alias, _, _),
        sp_base_alias_at(func_start, use_addr, alias, _),
        abi_home_arg_position(_, pos),
        abi_first_stack_arg_position(first_stack),
        if *pos < *first_stack;

    // Unsafe /homeparams cells remain ordinary mutable storage.  Keep their
    // identity and every normalized access as exported evidence, but do not
    // feed these unsafe-dependent rows back into stack_xtl or RTL selection in
    // this fixed point: doing so would pull the raw escape proof into the
    // register/call SCC.  RTLPass::run materializes these rows immediately
    // after the fixed point instead.
    relation win64_home_storage(Address, usize, RTLReg);
    win64_home_storage(func_start, *pos, slot_rtl) <--
        win64_home_spill_candidate(_, func_start, _, pos),
        win64_home_slot_unsafe(func_start, pos),
        let slot_rtl = fresh_home_slot_reg(*func_start, *pos);

    // Exact backing type for a selected canonical home cell.  This is a rule
    // output (rather than an imperative-only side effect) so scheduled stages
    // retain it and TypePass can treat the storage signature as authoritative.
    relation win64_home_slot_type(RTLReg, XType);
    win64_home_slot_type(*slot, xtype) <--
        win64_home_storage(func_start, pos, slot),
        win64_home_storage_signature(func_start, pos, move_class, width),
        if let Some(xtype) = home_move_xtype(*move_class, *width);

    relation win64_unsafe_home_access(Node, Address, Mreg, i64, usize, i64);
    win64_unsafe_home_access(addr, func_start, *base_reg, *raw_disp, *pos, entry_ofs) <--
        win64_home_cell(addr, func_start, base_reg, raw_disp, pos),
        win64_home_spill_candidate(_, func_start, _, pos),
        win64_home_slot_unsafe(func_start, pos),
        abi_incoming_sp_stack_base(incoming_base),
        abi_outgoing_stack_base(outgoing_base),
        abi_stack_slot_size(slot_size),
        let entry_ofs = *incoming_base - *outgoing_base + (*pos as i64 * *slot_size);

    // Closed set of scalar-storage shapes.  A cell is canonicalized only when
    // every possibly overlapping access is represented by exactly one of
    // these rows and all moves agree with the initial spill's class and width.
    // The selector below additionally requires an unambiguous RTL source or
    // destination for every row before changing any instruction.
    #[local] relation win64_home_storage_signature(Address, usize, usize, usize);
    win64_home_storage_signature(func_start, *pos, *move_class, *mem_size) <--
        win64_home_spill_candidate(addr, func_start, _, pos),
        win64_home_move_class(addr, move_class),
        pmov(addr, dst, _),
        op_indirect(dst, _, _, idx_str, _, _, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *mem_size > 0;

    // A single source parameter cannot have two different scalar home-store
    // encodings.  Keep ambiguous slots as ordinary inference evidence rather
    // than publishing an arbitrary source type from relation iteration order.
    #[local] relation win64_home_storage_signature_conflict(Address, usize);
    win64_home_storage_signature_conflict(func_start, pos) <--
        win64_home_storage_signature(func_start, pos, first_class, first_width),
        win64_home_storage_signature(func_start, pos, other_class, other_width),
        if first_class != other_class || first_width != other_width;

    relation win64_home_scalar_store(Node, Address, Mreg, i64, usize, i64, Mreg, usize, usize);
    win64_home_scalar_store(addr, func_start, *base_reg, *raw_disp, *pos, *entry_ofs, src_reg, *move_class, *mem_size) <--
        win64_unsafe_home_access(addr, func_start, base_reg, raw_disp, pos, entry_ofs),
        win64_home_storage_signature(func_start, pos, move_class, width),
        win64_home_move_class(addr, seen_class),
        if *seen_class == *move_class,
        pmov(addr, dst, src),
        op_indirect(dst, _, base_str, idx_str, _, asm_disp, mem_size),
        if is_x86_64_gp_register_name(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp,
        if *mem_size == *width,
        op_register(src, src_str),
        let src_reg = Mreg::x86(*src_str),
        if (*move_class == 0 && !is_float_mreg(&src_reg))
            || (*move_class != 0 && is_float_mreg(&src_reg));

    relation win64_home_scalar_load(Node, Address, Mreg, i64, usize, i64, Mreg, usize, usize);
    win64_home_scalar_load(addr, func_start, *base_reg, *raw_disp, *pos, *entry_ofs, dst_reg, *move_class, *mem_size) <--
        win64_unsafe_home_access(addr, func_start, base_reg, raw_disp, pos, entry_ofs),
        win64_home_storage_signature(func_start, pos, move_class, width),
        win64_home_move_class(addr, seen_class),
        if *seen_class == *move_class,
        pmov(addr, dst, src),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        op_indirect(src, _, base_str, idx_str, _, asm_disp, mem_size),
        if is_x86_64_gp_register_name(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp,
        if *mem_size == *width,
        if (*move_class == 0 && !is_float_mreg(&dst_reg))
            || (*move_class != 0 && is_float_mreg(&dst_reg));

    // MOVSX/MOVSXD/MOVZX are scalar reads too, but they are intentionally not
    // members of win64_home_move_class: treating them as plain MOV reloads
    // would erase their signedness.  `win64_unsafe_home_access` is derived from
    // `win64_home_cell`, so this operand is already proved to begin at byte zero
    // of the canonical cell.  A narrower integer read therefore observes
    // exactly the low bits of the locked full-cell value on little-endian x86;
    // publish the decoded sign/zero cast explicitly.  Nonzero-offset overlaps,
    // wider reads, floating cells, and every non-extending opcode remain on the
    // structured unsupported path.
    relation win64_home_scalar_extend_load(
        Node, Address, Mreg, i64, usize, i64, Mreg, usize, Operation
    );
    win64_home_scalar_extend_load(
        addr, func_start, *base_reg, *raw_disp, *pos, *entry_ofs,
        dst_reg, *width, op
    ) <--
        win64_unsafe_home_access(
            addr, func_start, base_reg, raw_disp, pos, entry_ofs
        ),
        win64_home_storage_signature(func_start, pos, move_class, width),
        if *move_class == 0,
        pmov(addr, dst, src),
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        if !is_float_mreg(&dst_reg),
        op_indirect(src, _, base_str, idx_str, _, asm_disp, mem_size),
        if is_x86_64_gp_register_name(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp,
        if *mem_size <= *width,
        if let Some(op) = extending_load_operation(mnem, *mem_size);

    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_extend_load(
            addr, func_start, _, _, pos, _, dst_reg, _, op
        ),
        win64_home_storage(func_start, pos, slot),
        is_def(addr, def_id),
        reg_xtl(addr, dst_reg, def_id),
        xtl_canonical(def_id, destination),
        let inst = RTLInst::Iop(
            op.clone(), Arc::new(vec![*slot]), *destination
        );

    relation win64_home_scalar_lea(Node, Address, Mreg, i64, usize, i64, Mreg);
    win64_home_scalar_lea(addr, func_start, *base_reg, *raw_disp, *pos, *entry_ofs, dst_reg) <--
        win64_unsafe_home_access(addr, func_start, base_reg, raw_disp, pos, entry_ofs),
        plea(addr, dst, src),
        op_register(dst, dst_str),
        if is_x86_64_gp_register_name(dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        op_indirect(src, _, base_str, idx_str, _, asm_disp, _),
        if is_x86_64_gp_register_name(base_str),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp;

    // A mutable /homeparams cell may be consumed directly by an integer
    // ALU/CMP instruction. Preserve that as a scalar operation over the
    // canonical cell, using the same root/synthetic-node shape as the
    // immutable home-parameter path. Partial-cell arithmetic remains outside
    // this closed shape set.
    relation win64_home_scalar_arith_read(
        Node, Address, Mreg, i64, usize, i64, Mreg, usize, Operation
    );
    win64_home_scalar_arith_read(
        addr, func_start, *base_reg, *raw_disp, *pos, *entry_ofs,
        *dst_reg, *width, op.clone()
    ) <--
        win64_unsafe_home_access(
            addr, func_start, base_reg, raw_disp, pos, entry_ofs
        ),
        win64_home_storage_signature(func_start, pos, move_class, width),
        if *move_class == 0,
        arith_load_op(addr, op, chunk, seen_base, seen_disp, dst_reg),
        if *seen_base == *base_reg && *seen_disp == *raw_disp,
        if chunk_size_bits(chunk) as usize == *width * 8;

    rtl_inst_candidate(addr, RTLInst::Inop) <--
        win64_home_scalar_arith_read(addr, _, _, _, _, _, _, _, _);
    rtl_inst_candidate(synthetic, inst) <--
        win64_home_scalar_arith_read(
            addr, func_start, _, _, pos, _, dst_reg, _, op
        ),
        win64_home_storage(func_start, pos, slot),
        reaching_use_rtl(addr, dst_reg, destination),
        let synthetic = *addr | (1u64 << 62),
        let inst = RTLInst::Iop(
            op.clone(), Arc::new(vec![*destination, *slot]), *destination
        );

    // VS2013 checked builds frequently compare a reassigned /homeparams cell
    // directly against a register or immediate.  The ordinary stack-CMP
    // lowering already expresses the branch as one Icond; publish an exact
    // candidate over the canonical mutable cell instead of suppressing the
    // entire function.  The selector below still requires every access in the
    // cell to have one closed scalar shape before retaining this candidate.
    relation win64_home_scalar_cmp_read(
        Node, Address, Symbol, Mreg, i64, usize, i64, usize
    );
    win64_home_scalar_cmp_read(
        addr, func_start, *mem, *base_reg, *raw_disp, *pos, *entry_ofs, *width
    ) <--
        win64_unsafe_home_access(
            addr, func_start, base_reg, raw_disp, pos, entry_ofs
        ),
        win64_home_storage_signature(func_start, pos, move_class, width),
        if *move_class == 0,
        pcmp(addr, mem, _),
        op_indirect(mem, _, base_str, idx_str, _, asm_disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp,
        if *mem_size == *width;
    win64_home_scalar_cmp_read(
        addr, func_start, *mem, *base_reg, *raw_disp, *pos, *entry_ofs, *width
    ) <--
        win64_unsafe_home_access(
            addr, func_start, base_reg, raw_disp, pos, entry_ofs
        ),
        win64_home_storage_signature(func_start, pos, move_class, width),
        if *move_class == 0,
        pcmp(addr, _, mem),
        op_indirect(mem, _, base_str, idx_str, _, asm_disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == *base_reg && *asm_disp == *raw_disp,
        if *mem_size == *width;

    // Keep the branch fusion explicitly adjacent.  A scheduled CMP -> Jcc
    // pair needs a snapshot at CMP and the Icond at Jcc so intervening
    // flag-preserving operations still execute on both paths; moving the
    // branch to CMP would silently skip those operations.
    #[local] relation win64_home_cmp_jcc_consumer(
        Node, Address, Node, TestCond, Address, Address
    );
    win64_home_cmp_jcc_consumer(
        addr, func_start, jcc_addr, *testcond, *target_addr, *fallthrough
    ) <--
        next(addr, jcc_addr),
        pjcc(jcc_addr, testcond, target_sym),
        symbol_resolved_addr(*target_sym, target_addr),
        next(jcc_addr, fallthrough),
        // Function ranges are immutable Asm inputs.  Carry the owner through
        // this projected relation so a backward edge is accepted exactly like
        // a forward edge, while cross-function targets/fallthroughs cannot be
        // fused into this function's canonical home-cell branch.
        real_addr_in_func(addr, func_start),
        real_addr_in_func(jcc_addr, func_start),
        real_addr_in_func(target_addr, func_start),
        real_addr_in_func(fallthrough, func_start);

    // SETcc is a value-producing consumer of the same comparison flags.  Its
    // destination is defined at the SETcc address, while the fused RTL
    // operation remains keyed at CMP (the same convention used by Asm's
    // register-only cmp_setcc_link).  Start with the exact adjacent shape;
    // scheduled SETcc chains need an explicit flag-provenance relation rather
    // than an unconstrained instruction walk.
    #[local] relation win64_home_cmp_setcc_consumer(Node, Node, TestCond, RTLReg);
    win64_home_cmp_setcc_consumer(addr, setcc_addr, *test_cond, destination) <--
        next(addr, setcc_addr),
        setcc_testcond(setcc_addr, test_cond),
        instruction(setcc_addr, _, _, _, dst, _, _, _, _, _),
        op_register(dst, dst_str),
        let dst_reg = Mreg::x86(*dst_str),
        is_def(setcc_addr, def_id),
        reg_xtl(setcc_addr, dst_reg, def_id),
        xtl_canonical(def_id, destination),
        instr_in_function(addr, func_start),
        instr_in_function(setcc_addr, func_start);

    // The fused value is now defined at CMP, so its semantic successor is the
    // instruction after the consumed adjacent SETcc.  This is explicit rather
    // than relying on whether Linear happened to give the otherwise-unlowered
    // memory CMP an LTL fallthrough edge.
    rtl_edge_negated(addr, setcc_addr) <--
        win64_home_cmp_setcc_consumer(addr, setcc_addr, _, _);
    rtl_succ_candidate(addr, fallthrough) <--
        win64_home_cmp_setcc_consumer(addr, setcc_addr, _, _),
        next(setcc_addr, fallthrough);

    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, mem, other),
        op_register(other, reg_str),
        let other_reg = Mreg::x86(*reg_str),
        reaching_use_rtl(addr, other_reg, other_rtl),
        win64_home_cmp_jcc_consumer(
            addr, func_start, jcc_addr, testcond, target_addr, fallthrough
        ),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let inst = RTLInst::Icond(cond, Arc::new(vec![*slot, *other_rtl]), Either::Right(*target_addr), Either::Right(*fallthrough));
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, other, mem),
        op_register(other, reg_str),
        let other_reg = Mreg::x86(*reg_str),
        reaching_use_rtl(addr, other_reg, other_rtl),
        win64_home_cmp_jcc_consumer(
            addr, func_start, jcc_addr, testcond, target_addr, fallthrough
        ),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let inst = RTLInst::Icond(cond, Arc::new(vec![*other_rtl, *slot]), Either::Right(*target_addr), Either::Right(*fallthrough));
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, mem, other),
        op_immediate(other, imm, _),
        win64_home_cmp_jcc_consumer(
            addr, func_start, jcc_addr, testcond, target_addr, fallthrough
        ),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let cond = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm),
            other => other,
        },
        let inst = RTLInst::Icond(cond, Arc::new(vec![*slot]), Either::Right(*target_addr), Either::Right(*fallthrough));
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, other, mem),
        op_immediate(other, imm, _),
        win64_home_cmp_jcc_consumer(
            addr, func_start, jcc_addr, testcond, target_addr, fallthrough
        ),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let cond = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm),
            other => other,
        },
        let inst = RTLInst::Icond(cond, Arc::new(vec![*slot]), Either::Right(*target_addr), Either::Right(*fallthrough));

    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, mem, other),
        op_register(other, reg_str),
        let other_reg = Mreg::x86(*reg_str),
        reaching_use_rtl(addr, other_reg, other_rtl),
        win64_home_cmp_setcc_consumer(addr, setcc_addr, testcond, destination),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let inst = RTLInst::Iop(
            Operation::Ocmp(cond), Arc::new(vec![*slot, *other_rtl]), *destination
        );
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, other, mem),
        op_register(other, reg_str),
        let other_reg = Mreg::x86(*reg_str),
        reaching_use_rtl(addr, other_reg, other_rtl),
        win64_home_cmp_setcc_consumer(addr, setcc_addr, testcond, destination),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let inst = RTLInst::Iop(
            Operation::Ocmp(cond), Arc::new(vec![*other_rtl, *slot]), *destination
        );
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, mem, other),
        op_immediate(other, imm, _),
        win64_home_cmp_setcc_consumer(addr, setcc_addr, testcond, destination),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let cond = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm),
            other => other,
        },
        let inst = RTLInst::Iop(
            Operation::Ocmp(cond), Arc::new(vec![*slot]), *destination
        );
    rtl_inst_candidate(addr, inst) <--
        win64_home_scalar_cmp_read(addr, func_start, mem, _, _, pos, _, width),
        win64_home_storage(func_start, pos, slot),
        pcmp(addr, other, mem),
        op_immediate(other, imm, _),
        win64_home_cmp_setcc_consumer(addr, setcc_addr, testcond, destination),
        let raw_cond = crate::x86::types::condition_for_testcond(*testcond),
        let sized_cond = crate::decompile::passes::rtl_pass::adjust_condition_size(raw_cond, *width),
        let cond = match sized_cond {
            Condition::Ccomp(cmp) => Condition::Ccompimm(cmp, *imm),
            Condition::Ccompu(cmp) => Condition::Ccompuimm(cmp, *imm),
            Condition::Ccompl(cmp) => Condition::Ccomplimm(cmp, *imm),
            Condition::Ccomplu(cmp) => Condition::Ccompluimm(cmp, *imm),
            other => other,
        },
        let inst = RTLInst::Iop(
            Operation::Ocmp(cond), Arc::new(vec![*slot]), *destination
        );

    #[local] relation win64_home_scalar_access(Node, Address, usize);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_store(addr, func, _, _, pos, _, _, _, _);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_load(addr, func, _, _, pos, _, _, _, _);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_extend_load(addr, func, _, _, pos, _, _, _, _);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_lea(addr, func, _, _, pos, _, _),
        // A bare/numerically-used LEA is not enough: replacing RSP+k with
        // &scalar changes the meaning of subsequent integer arithmetic.  At
        // least one exact derived address must be consumed as a call argument.
        win64_home_lea_address_taken(addr, func, pos);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_arith_read(addr, func, _, _, pos, _, _, _, _);
    win64_home_scalar_access(addr, func, pos) <--
        win64_home_scalar_cmp_read(addr, func, _, _, _, pos, _, _);

    unsupported_stack_address(*func, *addr, "unsupported-stack-address") <--
        // A storage signature exists for every ordinary /homeparams spill,
        // including safe immutable spill/reload pairs.  Only cells already
        // proven mutable/unsafe need scalar replacement; otherwise their raw
        // overlaps are intentionally consumed by the entry-parameter fold.
        win64_home_storage(func, pos, _),
        win64_home_overlap(addr, func, pos),
        !win64_home_scalar_access(addr, func, pos);
    unsupported_address_detail(*func, *addr, "home-cell-shape-unrepresentable") <--
        win64_home_storage(func, pos, _),
        win64_home_overlap(addr, func, pos),
        !win64_home_scalar_access(addr, func, pos);

    // Reasons that make scalar replacement incomplete or ambiguous.  These
    // rows are separate from win64_home_slot_unsafe: "unsafe" merely means the
    // immutable parameter fold is invalid, while this relation means even a
    // mutable canonical scalar cannot represent every possible access.
    relation win64_home_canonical_veto(Address, usize);
    win64_home_canonical_veto(func_start, *pos) <--
        win64_home_storage_signature(func_start, pos, _, _),
        win64_home_overlap(addr, func_start, pos),
        !win64_home_scalar_access(addr, func_start, pos);
    win64_home_canonical_veto(func_start, *pos) <--
        win64_home_storage_signature(func_start, pos, _, _),
        win64_unsafe_home_access(addr, func_start, _, _, pos, _),
        !win64_home_scalar_access(addr, func_start, pos);
    win64_home_canonical_veto(func_start, *pos) <--
        win64_home_scalar_lea(_, func_start, _, _, pos, _, _),
        win64_home_storage_signature(func_start, pos, _, width),
        // An escaped home address exposes the ABI's whole eight-byte cell.
        // A narrower scalar would not provide safe backing for an opaque
        // callee, so retain the original stack storage instead.
        if *width != 8;

    win64_home_canonical_veto(func_start, *pos) <--
        indexed_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_based_mem_at(addr, func_start, seen_base, _),
        if *seen_base == *base_reg,
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        indexed_stack_operand(addr, Mreg::SP, _, _),
        real_addr_in_func(addr, func_start),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        indexed_stack_operand(addr, Mreg::BP, _, _),
        real_addr_in_func(addr, func_start),
        func_ever_sets_frame_pointer(func_start),
        !bp_base_at(func_start, addr, _),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        direct_stack_operand(addr, Mreg::SP, _, _),
        real_addr_in_func(addr, func_start),
        !sp_entry_ofs(func_start, addr, _),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        direct_stack_operand(addr, Mreg::BP, _, _),
        real_addr_in_func(addr, func_start),
        !bp_base_at(func_start, addr, _),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        direct_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_may_alias_at(func_start, addr, base_reg),
        !sp_base_alias_at(func_start, addr, base_reg, _),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        indexed_stack_operand(addr, base_reg, _, _),
        real_addr_in_func(addr, func_start),
        sp_may_alias_at(func_start, addr, base_reg),
        !sp_base_alias_at(func_start, addr, base_reg, _),
        win64_home_storage_signature(func_start, pos, _, _);

    // A non-memory use of a stack-base alias is admissible only when it is one
    // exact 64-bit MOV step in the address web or the proven consumption of
    // that web as a call argument.  Merely proving that the value is &home[pos]
    // cannot make arithmetic, comparisons, or returns safe scalar rewrites.
    win64_home_canonical_veto(func_start, *pos) <--
        sp_may_alias_at(func_start, addr, alias),
        real_addr_in_func(addr, func_start),
        asm_reg_use(addr, alias),
        !direct_stack_operand(addr, alias, _, _),
        !indexed_stack_operand(addr, alias, _, _),
        !sp_alias_def(func_start, addr, _, _),
        win64_home_storage_signature(func_start, pos, _, _),
        !win64_home_exact_addr_copy_use(func_start, addr, alias, pos),
        !win64_home_exact_addr_call(func_start, addr, alias, pos);
    // A proved exact-home address web receives the stricter policy even when
    // the instruction also happens to create another affine SP alias.  This
    // closes the ADD/LEA exemption above without disabling the independently
    // supported copied-SP affine storage pattern.
    win64_home_canonical_veto(func_start, *pos) <--
        win64_home_exact_addr_at(func_start, addr, alias, pos),
        asm_reg_use(addr, alias),
        !direct_stack_operand(addr, alias, _, _),
        !indexed_stack_operand(addr, alias, _, _),
        win64_home_storage_signature(func_start, pos, _, _),
        !win64_home_exact_addr_copy_use(func_start, addr, alias, pos),
        !win64_home_exact_addr_call(func_start, addr, alias, pos);
    win64_home_canonical_veto(func_start, *pos) <--
        abi_shared_arg_slots(true),
        sp_may_alias_def(func_start, alias_def, alias),
        abi_int_arg_position(alias, _),
        def_reaches_call(call_addr, alias_def, alias),
        real_addr_in_func(call_addr, func_start),
        win64_home_storage_signature(func_start, pos, _, _),
        !win64_home_exact_addr_at(func_start, call_addr, alias, pos),
        !win64_home_exact_addr_call(func_start, call_addr, alias, pos);
    win64_home_canonical_veto(func_start, *pos) <--
        abi_shared_arg_slots(true),
        sp_may_alias_def(func_start, alias_def, ?&Mreg::AX),
        def_reaches_return(ret_addr, alias_def, ?&Mreg::AX),
        real_addr_in_func(ret_addr, func_start),
        instruction(ret_addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "RET" | "RETF" | "RETFQ"),
        win64_home_storage_signature(func_start, pos, _, _);
    win64_home_canonical_veto(func_start, *pos) <--
        win64_home_spill_candidate(first_addr, func_start, first_reg, pos),
        win64_home_spill_candidate(other_addr, func_start, other_reg, pos),
        if first_addr != other_addr || first_reg != other_reg;

    // Node-keyed address evidence and function+slot escape evidence are
    // intentionally not keyed by a raw stack displacement.  RTLPass::run
    // filters these provisional rows to the cells it actually rewrites.
    relation win64_home_address(Node, RTLReg);
    win64_home_address(addr, slot) <--
        win64_home_scalar_lea(addr, func_start, _, _, pos, _, _),
        win64_home_storage(func_start, pos, slot);

    relation win64_home_escaped(Address, RTLReg);
    win64_home_escaped(func_start, slot) <--
        win64_home_scalar_lea(_, func_start, _, _, pos, _, _),
        win64_home_storage(func_start, pos, slot);

    // Only a nonescaping, immutable home cell with proved reloads may collapse
    // to the incoming SSA parameter. Unsafe cells fall through to the ordinary
    // Lsetstack/Lgetstack local-storage rules.
    relation win64_home_spill(Node, Address, Mreg, usize);
    win64_home_spill(addr, func_start, *param_reg, *pos) <--
        win64_home_spill_candidate(addr, func_start, param_reg, pos),
        !win64_home_slot_unsafe(func_start, pos);

    relation win64_home_reload(Node, Address, Mreg, usize);
    win64_home_reload(addr, func_start, *param_reg, *pos) <--
        win64_home_reload_cell(addr, func_start, param_reg, pos),
        !win64_home_slot_unsafe(func_start, pos);

    relation win64_home_arith_read(Node, Address, Mreg, usize);
    win64_home_arith_read(addr, func_start, param_reg, pos) <--
        win64_home_arith_read_candidate(addr, func_start, param_reg, pos),
        !win64_home_slot_unsafe(func_start, pos);

    // BP-based arg detection requires an actual frame pointer: in frameless functions RBP is a callee-saved scratch (often a struct pointer), so a positive-offset BP load is a field deref, not an incoming arg. With a frame pointer, caller args start at BP+16 (BP+0=saved RBP, BP+8=return address).
    #[local] relation func_ever_sets_frame_pointer(Address);
    func_ever_sets_frame_pointer(func_start) <--
        stack_base_move(addr, src, dst),
        if *src == "RSP" && *dst == "RBP",
        instr_in_function(addr, func_start);

    // Enumerate concrete non-indexed memory operands so copied-SP aliases as
    // well as literal SP/BP bases have a bound displacement. Consumers that
    // infer incoming slots explicitly restrict this relation back to SP/BP.
    #[local] relation direct_stack_operand(Node, Mreg, i64, usize);
    direct_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, operand, _, _, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    direct_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, operand, _, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    direct_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, _, operand, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    direct_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, _, _, operand, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);

    #[local] relation indexed_stack_operand(Node, Mreg, i64, usize);
    indexed_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, operand, _, _, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    indexed_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, operand, _, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    indexed_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, _, operand, _, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);
    indexed_stack_operand(*addr, base, *disp, *mem_size) <--
        instruction(addr, _, _, _, _, _, _, operand, _, _),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str != "NONE" && !idx_str.is_empty(),
        if is_valid_stack_operand_base_name(base_str),
        let base = Mreg::x86(*base_str);

    // Normalize a raw SP displacement to its function-entry coordinate, then
    // compute the real ABI ordinal. A missing SP-state row or an unaligned
    // address conservatively produces no parameter claim.
    incoming_stack_slot(addr, func_start, Mreg::SP, *disp, ordinal) <--
        direct_stack_operand(addr, Mreg::SP, disp, _),
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        sp_entry_ofs(func_start, addr, sp_ofs),
        abi_incoming_sp_stack_base(stack_base),
        abi_stack_slot_size(slot_size),
        let entry_disp = *disp + sp_ofs.0,
        if entry_disp >= *stack_base && (entry_disp - *stack_base) % *slot_size == 0,
        let ordinal = ((entry_disp - *stack_base) / *slot_size) as usize,
        if ordinal < 64;

    // A BP displacement is an incoming slot only when a dominating, reaching
    // frame-pointer copy gives this use an entry-SP coordinate.
    incoming_stack_slot(addr, func_start, Mreg::BP, *disp, ordinal) <--
        direct_stack_operand(addr, Mreg::BP, disp, _),
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        bp_base_at(func_start, addr, bp_base),
        abi_incoming_sp_stack_base(stack_base),
        abi_stack_slot_size(slot_size),
        let entry_disp = *bp_base + *disp,
        if entry_disp >= *stack_base && (entry_disp - *stack_base) % *slot_size == 0,
        let ordinal = ((entry_disp - *stack_base) / *slot_size) as usize,
        if ordinal < 64;

    // stack_def_used is keyed by one raw displacement, so it misses partial
    // writes such as a byte store at entry-SP+41 before a dword read at +40.
    // Attach each decoded definition and incoming read to full byte ranges,
    // normalized through the same proved SP/BP/copied-SP coordinate used by
    // ABI recovery.
    #[local] relation decoded_stack_write(Node, Mreg, i64, usize);
    decoded_stack_write(addr, base, *disp, *mem_size) <--
        decoded_memory_write_operand(addr, operand),
        op_indirect(operand, _, base_str, idx_str, _, disp, mem_size),
        if *idx_str == "NONE" || idx_str.is_empty(),
        if is_x86_64_gp_register_name(base_str),
        if *mem_size > 0,
        let base = Mreg::x86(*base_str);

    relation normalized_stack_write_range(Node, Address, Mreg, i64, i64, i64);
    normalized_stack_write_range(addr, func_start, *base, *disp, range_start, range_end) <--
        decoded_stack_write(addr, base, disp, mem_size),
        sp_based_mem_at(addr, func_start, proven_base, base_ofs),
        if *proven_base == *base,
        let range_start = *base_ofs + *disp,
        let range_end = range_start + *mem_size as i64;

    // Exact control-flow witness exported for consumers that use a decoded
    // stack write as object-initialization evidence.  Address order alone is
    // insufficient across branches and loops.
    relation stack_write_dominates_node(Address, Node, Node);
    stack_write_dominates_node(*func_start, *write_node, *use_node) <--
        normalized_stack_write_range(write_node, func_start, _, _, _, _),
        real_addr_in_func(use_node, func_start),
        reg_def_dominates_use(func_start, write_node, use_node);

    #[local] relation stack_read_site(Node);
    stack_read_site(addr) <-- ltl_inst(addr, ?LTLInst::Lload(_, _, _, _));
    stack_read_site(addr) <-- ltl_inst(addr, ?LTLInst::Lgetstack(_, _, _, _));
    stack_read_site(addr) <-- arith_load_op(addr, _, _, _, _, _);
    stack_read_site(addr) <-- float_arith_stack_op(addr, _, _, _, _);
    stack_read_site(addr) <-- stack_unary_load_op(addr, _, _, _, _);
    stack_read_site(addr) <-- arith_store_reg(addr, _, _, _, _, _);
    stack_read_site(addr) <-- arith_store_imm(addr, _, _, _, _);

    #[local] relation incoming_stack_read_range(Node, Address, i64, i64);
    incoming_stack_read_range(addr, func_start, range_start, range_end) <--
        stack_read_site(addr),
        incoming_stack_slot(addr, func_start, base, disp, _),
        direct_stack_operand(addr, seen_base, seen_disp, mem_size),
        if *seen_base == *base && *seen_disp == *disp && *mem_size > 0,
        sp_based_mem_at(addr, func_start, proven_base, base_ofs),
        if *proven_base == *base,
        let range_start = *base_ofs + *disp,
        let range_end = range_start + *mem_size as i64;

    // May-reaching instruction predecessors are deliberately conservative:
    // any executable path carrying a partially overlapping write makes
    // pristine ABI parameter recovery unsound.
    #[local] relation stack_cfg_step(Node, Node);
    stack_cfg_step(src, dst) <--
        next(src, dst),
        code_in_block(src, block),
        code_in_block(dst, block);
    stack_cfg_step(src, dst) <--
        ddisasm_cfg_edge(src, dst, edge_type),
        if *edge_type != "call" && *edge_type != "indirect" && *edge_type != "indirect_call";

    #[local] relation stack_read_predecessor(Node, Address, Node);
    stack_read_predecessor(read_addr, func_start, pred) <--
        incoming_stack_read_range(read_addr, func_start, _, _),
        stack_cfg_step(pred, read_addr),
        instr_in_function(pred, func_start);
    stack_read_predecessor(read_addr, func_start, pred) <--
        stack_read_predecessor(read_addr, func_start, cur),
        stack_cfg_step(pred, cur),
        instr_in_function(pred, func_start);

    relation stack_param_partial_write(Node, Node);
    stack_param_partial_write(read_addr, write_addr) <--
        incoming_stack_read_range(read_addr, func_start, read_start, read_end),
        stack_read_predecessor(read_addr, func_start, write_addr),
        if *write_addr != *read_addr,
        normalized_stack_write_range(write_addr, func_start, _, _, write_start, write_end),
        if *write_start < *read_end && *read_start < *write_end,
        // A same-start write that covers the complete read is already modeled
        // by stack_def_used. Everything else is an unmodeled partial overlap.
        if *write_start != *read_start || *write_end < *read_end;

    unsupported_stack_address(*func_start, *read_addr, "unsupported-stack-address") <--
        incoming_stack_read_range(read_addr, func_start, _, _),
        stack_param_partial_write(read_addr, _);
    unsupported_address_detail(*func_start, *read_addr, "stack-parameter-partial-write") <--
        incoming_stack_read_range(read_addr, func_start, _, _),
        stack_param_partial_write(read_addr, _);

    stack_param_access(addr, func_start, *disp, *ordinal) <--
        incoming_stack_slot(addr, func_start, base, disp, ordinal),
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Aindexed(ofs), args, _)),
        if *ofs == *disp,
        for arg in args.iter(),
        if *arg == *base,
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    stack_param_access(addr, func_start, *disp, *ordinal) <--
        incoming_stack_slot(addr, func_start, base, disp, ordinal),
        arith_load_op(addr, _, _, op_base, ofs, _),
        if *op_base == *base && *ofs == *disp,
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    stack_param_access(addr, func_start, *disp, *ordinal) <--
        incoming_stack_slot(addr, func_start, base, disp, ordinal),
        float_arith_stack_op(addr, _, op_base, ofs, _),
        if *op_base == *base && *ofs == *disp,
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    stack_param_access(addr, func_start, *disp, *ordinal) <--
        incoming_stack_slot(addr, func_start, base, disp, ordinal),
        stack_unary_load_op(addr, _, op_base, ofs, _),
        if *op_base == *base && *ofs == *disp,
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _);

    // A memory-destination RMW is also a read of the incoming parameter when
    // no earlier stack definition reaches it. Its lowering seeds a mutable
    // local from this ABI-positioned parameter before applying the update.
    stack_param_access(addr, func_start, *disp, *ordinal) <--
        arith_store_reg_uses_stack_param(addr, func_start, disp, ordinal);
    stack_param_access(addr, func_start, *disp, *ordinal) <--
        arith_store_imm_uses_stack_param(addr, func_start, disp, ordinal);

    #[local] relation stack_param_update_chunk(Address, usize, MemoryChunk);
    stack_param_update_chunk(func_start, *ordinal, *chunk) <--
        arith_store_reg_uses_stack_param(addr, func_start, _, ordinal),
        arith_store_reg(addr, _, chunk, _, _, _);
    stack_param_update_chunk(func_start, *ordinal, *chunk) <--
        arith_store_imm_uses_stack_param(addr, func_start, _, ordinal),
        arith_store_imm(addr, _, chunk, _, _);

    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        stack_param_update_chunk(func_start, ordinal, chunk),
        let param_reg = fresh_stack_param_reg(*func_start, *ordinal),
        let xt = match chunk {
            MemoryChunk::MInt32 => XType::Xint,
            MemoryChunk::MInt64 => XType::Xlong,
            MemoryChunk::MFloat64 => XType::Xfloat,
            MemoryChunk::MFloat32 => XType::Xsingle,
            MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned
            | MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => XType::Xint,
            _ => XType::Xany64,
        };

    // Sign/zero-extending stack-arg loads bypass Lload, so the capstone operand
    // pins width and signedness. Plain MOV is included only for Win64, where
    // the entry-anchored slot and reaching-def veto distinguish it from locals.
    #[local] relation stack_param_load(Node, Address, i64, usize, MemoryChunk);

    // SP-relative extending load.
    stack_param_load(addr, *func_start, *disp, *ordinal, mc) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, _, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, msize),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *base_str == "RSP",
        incoming_stack_slot(addr, func_start, Mreg::SP, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _),
        if let Some(mc) = extending_load_chunk(mnem, *msize);

    // A normal Win64 stack argument is read with plain MOV, not an extending load; entry-anchored offsets distinguish it from a local, and the reaching stack-def veto excludes an overwritten slot.
    stack_param_load(addr, *func_start, *disp, *ordinal, mc) <--
        abi_shared_arg_slots(true),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, typ, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVSS" | "MOVSD" | "VMOVSS" | "VMOVSD"),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *base_str == "RSP",
        incoming_stack_slot(addr, func_start, Mreg::SP, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _),
        let mc = typ_to_chunk(*typ);

    stack_param_load(addr, *func_start, *disp, *ordinal, mc) <--
        abi_shared_arg_slots(true),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, typ, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVSS" | "MOVSD" | "VMOVSS" | "VMOVSD"),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *base_str == "RBP",
        incoming_stack_slot(addr, func_start, Mreg::BP, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _),
        let mc = typ_to_chunk(*typ);

    // BP-relative extending load: incoming_stack_slot carries the explicit
    // use-specific frame-pointer proof (frameless RBP remains ordinary data).
    stack_param_load(addr, *func_start, *disp, *ordinal, mc) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, _, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, msize),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if *base_str == "RBP",
        incoming_stack_slot(addr, func_start, Mreg::BP, disp, ordinal),
        !stack_def_used(_, _, _, addr, _, disp),
        !stack_param_partial_write(addr, _),
        if let Some(mc) = extending_load_chunk(mnem, *msize);

    stack_param_access(addr, func_start, *disp, *ordinal) <--
        stack_param_load(addr, func_start, disp, ordinal, _);

    stack_param_ordinal(func_start, *ordinal) <--
        stack_param_access(_, func_start, _, ordinal);

    // TR-5 companion: extension width/signedness pins the sub-int XType for the synthetic stack-param reg (more precise than the arith/Lload chunk rules).
    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        stack_param_load(_, func_start, _disp, idx, chunk),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let xt = match chunk {
            MemoryChunk::MInt32 => XType::Xint,
            MemoryChunk::MInt16Signed => XType::Xint16signed,
            MemoryChunk::MInt16Unsigned => XType::Xint16unsigned,
            MemoryChunk::MInt8Signed => XType::Xint8signed,
            MemoryChunk::MInt8Unsigned => XType::Xint8unsigned,
            _ => XType::Xany64,
        };

    relation emit_function_stack_param_count(Address, usize);

    emit_function_stack_param_count(func_start, count) <--
        stack_param_ordinal(func_start, _),
        agg max_ordinal = ascent::aggregators::max(ordinal) in stack_param_ordinal(func_start, ordinal),
        let count = max_ordinal + 1;

    emit_function_stack_param_count(func_start, 0) <--
        emit_function(func_start, _, _),
        !stack_param_ordinal(func_start, _);


    relation dx_has_non_div_use(Address);

    // DX used (non-div) while it still holds the incoming caller value = DX is a genuine parameter, not just the high half of a division. Window -> arg_reg_param_live_at.
    dx_has_non_div_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::DX),
        ltl_inst(addr, ?LTLInst::Lop(_, args, _)),
        for arg in args.iter(),
        if *arg == Mreg::DX,
        !pdiv(addr, _, _);

    dx_has_non_div_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::DX),
        ltl_inst(addr, ?LTLInst::Lstore(_, _, _, src)),
        if *src == Mreg::DX,
        !pdiv(addr, _, _);

    dx_has_non_div_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::DX),
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        for arg in args.iter(),
        if *arg == Mreg::DX;

    dx_has_non_div_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::DX),
        ltl_inst(addr, ?LTLInst::Lcond(_, args, _, _)),
        for arg in args.iter(),
        if *arg == Mreg::DX;

    dx_has_non_div_use(func_start) <--
        arg_reg_copied_early(func_start, ?&Mreg::DX, _);

    relation dx_used_only_in_div(Address);

    dx_used_only_in_div(func_start) <--
        func_arg_reg_used(func_start, Mreg::DX),
        !arg_reg_spilled_to_stack(func_start, Mreg::DX),
        !dx_has_non_div_use(func_start),
        func_has_div_instr(func_start);

    // CX filter: suppress CX as param evidence when it's only used as shift/rotate count
    relation func_has_shift_instr(Address);

    func_has_shift_instr(func_start) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lop(op, args, _)),
        for arg in args.iter(),
        if *arg == Mreg::CX,
        if is_shift_or_rotate_op(op);

    relation cx_has_non_shift_use(Address);

    // CX used (non-shift) while it still holds the incoming caller value = CX is a genuine parameter, not just a shift count. Window -> arg_reg_param_live_at (CX holds the caller value at the use).
    cx_has_non_shift_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::CX),
        ltl_inst(addr, ?LTLInst::Lop(op, args, _)),
        for arg in args.iter(),
        if *arg == Mreg::CX,
        if !is_shift_or_rotate_op(op);

    cx_has_non_shift_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::CX),
        ltl_inst(addr, ?LTLInst::Lstore(_, _, _, src)),
        if *src == Mreg::CX;

    cx_has_non_shift_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::CX),
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        for arg in args.iter(),
        if *arg == Mreg::CX;

    cx_has_non_shift_use(func_start) <--
        arg_reg_param_live_at(func_start, addr, ?&Mreg::CX),
        ltl_inst(addr, ?LTLInst::Lcond(_, args, _, _)),
        for arg in args.iter(),
        if *arg == Mreg::CX;

    cx_has_non_shift_use(func_start) <--
        arg_reg_copied_early(func_start, ?&Mreg::CX, _);

    relation cx_used_only_in_shift(Address);

    cx_used_only_in_shift(func_start) <--
        func_arg_reg_used(func_start, Mreg::CX),
        !arg_reg_spilled_to_stack(func_start, Mreg::CX),
        !cx_has_non_shift_use(func_start),
        func_has_shift_instr(func_start);

    // The old scratch-register filter is removed: func_arg_reg_used now derives from arg_reg_param_live_at, which already excludes a defined-first scratch register.

    // Func param positions use max-evidence: all positions up to max are filled per ABI
    relation func_has_param_evidence(Address, usize);
    #[local] relation func_gp_param_evidence(Address, usize);
    #[local] relation func_float_param_evidence_at(Address, usize);

    func_gp_param_evidence(func_start, pos) <--
        func_arg_reg_used(func_start, mreg),
        abi_int_arg_position(mreg, pos),
        if *mreg != Mreg::DX && *mreg != Mreg::CX;
    func_gp_param_evidence(func_start, pos) <--
        func_arg_reg_used(func_start, Mreg::DX),
        abi_int_arg_position(?&Mreg::DX, pos),
        !dx_used_only_in_div(func_start);
    func_gp_param_evidence(func_start, pos) <--
        func_arg_reg_used(func_start, Mreg::CX),
        abi_int_arg_position(?&Mreg::CX, pos),
        !cx_used_only_in_shift(func_start);
    // A proved VS x64 /homeparams store is positive definition-side evidence
    // even when the source parameter is otherwise unused.  Its highest shared
    // ABI ordinal also proves all lower source positions exist.
    func_gp_param_evidence(func_start, pos) <--
        win64_home_spill_candidate(_, func_start, param_reg, pos),
        abi_int_arg_position(param_reg, pos);

    func_has_param_evidence(func_start, pos) <--
        func_gp_param_evidence(func_start, pos);
    // On Windows a used XMM register occupies that source-language ordinal in the same sequence as GP args.
    func_float_param_evidence_at(func_start, pos) <--
        abi_shared_arg_slots(true),
        func_float_param_used(func_start, mreg),
        abi_float_arg_position(mreg, pos);
    func_has_param_evidence(func_start, pos) <--
        func_float_param_evidence_at(func_start, pos);

    // A Win64 stack argument proves every lower shared ABI position exists even
    // when the function never reads those lower values. Feed its absolute
    // position into the same max-position ladder so sparse sixth/eighth accesses
    // produce six/eight-parameter prototypes rather than only their tail width.
    func_has_param_evidence(func_start, absolute_pos) <--
        abi_shared_arg_slots(true),
        stack_param_ordinal(func_start, ordinal),
        abi_first_stack_arg_position(first_stack),
        let absolute_pos = *first_stack + *ordinal;

    // Forwarded-only parameter evidence: an arg register still holding the caller value at a call the callee consumes as a parameter IS a parameter, closing the pass-through chicken-and-egg.
    func_has_param_evidence(func_start, pos) <--
        is_call_instruction(call_addr),
        instr_in_function(call_addr, func_start),
        call_target_func(call_addr, callee),
        is_arg_reg(arg_reg),
        arg_reg_param_live_at(func_start, call_addr, arg_reg),
        func_arg_reg_used(callee, arg_reg),
        abi_int_arg_position(arg_reg, pos);

    // Companion for a KNOWN-SIGNATURE extern callee, which has no body to corroborate: a live pass-through arg register within the extern's fixed param count IS a parameter of this function.
    func_has_param_evidence(func_start, pos) <--
        forwarded_param_candidate(func_start, arg_reg, call_addr),
        call_known_sig_gp_position(call_addr, pos),
        abi_int_arg_position(arg_reg, pos);

    relation func_max_param_position(Address, usize);

    func_max_param_position(func_start, max_pos) <--
        func_has_param_evidence(func_start, _),
        agg max_pos = ascent::aggregators::max(pos) in func_has_param_evidence(func_start, pos);

    relation func_has_param_at_position(Address, usize);

    func_has_param_at_position(func_start, *max_pos) <--
        func_max_param_position(func_start, max_pos);
    func_has_param_at_position(func_start, lower) <--
        func_has_param_at_position(func_start, pos),
        if *pos > 0,
        let lower = *pos - 1;

    relation func_param_validated(Address, Mreg);

    // SysV's GP sequence is independent and compact; Windows fills an unknown lower shared slot with the GP-class fallback but never where positive XMM evidence identifies the ordinal as float.
    func_param_validated(func_start, mreg) <--
        !abi_shared_arg_slots(true),
        abi_int_arg_position(mreg, pos),
        func_has_param_at_position(func_start, pos);
    func_param_validated(func_start, mreg) <--
        abi_shared_arg_slots(true),
        abi_int_arg_position(mreg, pos),
        func_has_param_at_position(func_start, pos),
        !func_float_param_evidence_at(func_start, pos);

    relation func_arg_used_undefined(Address, RTLReg);

    func_arg_used_undefined(func_start, id), reg_xtl(func_start, mreg, id) <--
        func_param_validated(func_start, mreg),
        let id = fresh_xtl_reg(*func_start, *mreg);


    relation param_used_in_deref(Address, Mreg);

    // Only check base register (args[0]) for pointer not index
    param_used_in_deref(func_start, *arg_reg) <--
        func_param_validated(func_start, arg_reg),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        if !args.is_empty() && args[0] == *arg_reg;

    param_used_in_deref(func_start, *arg_reg) <--
        func_param_validated(func_start, arg_reg),
        arg_reg_copied_early(func_start, arg_reg, copy_dst),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        if !args.is_empty() && args[0] == *copy_dst;

    param_used_in_deref(func_start, *arg_reg) <--
        func_param_validated(func_start, arg_reg),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, _, args, _)),
        if !args.is_empty() && args[0] == *arg_reg;

    param_used_in_deref(func_start, *arg_reg) <--
        func_param_validated(func_start, arg_reg),
        arg_reg_copied_early(func_start, arg_reg, copy_dst),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, _, args, _)),
        if !args.is_empty() && args[0] == *copy_dst;

    // The value-web canonicals carrying a parameter's INCOMING value: at -O0 the prologue spills the arg and each reload is a fresh value, so track the spill slot and add every reload's canonical.
    relation param_value_canon(Address, Mreg, RTLReg);

    param_value_canon(func_start, *arg_reg, canon) <--
        func_param_validated(func_start, arg_reg),
        reg_xtl(func_start, arg_reg, raw_v),
        xtl_canonical(raw_v, canon);

    param_value_canon(func_start, *arg_reg, reload_canon) <--
        func_param_validated(func_start, arg_reg),
        instr_in_function(spill, func_start),
        ltl_inst(spill, ?LTLInst::Lsetstack(src, _, ofs, _)),
        if src == arg_reg,
        arg_reg_param_live_at(func_start, spill, arg_reg),
        instr_in_function(reload, func_start),
        ltl_inst(reload, ?LTLInst::Lgetstack(_, ofs2, _, dst)),
        if ofs2 == ofs,
        reg_rtl(reload, *dst, reload_canon);

    // Value-web deref evidence: any register sharing a param value canonical used as a load/store BASE proves the param is a pointer, subsuming the -O0 spill/reload case.
    param_used_in_deref(func_start, *arg_reg) <--
        param_value_canon(func_start, arg_reg, canon),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, _, args, _)),
        if !args.is_empty(),
        reg_rtl(addr, args[0], canon);

    param_used_in_deref(func_start, *arg_reg) <--
        param_value_canon(func_start, arg_reg, canon),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, _, args, _)),
        if !args.is_empty(),
        reg_rtl(addr, args[0], canon);

    // Value-web address-arithmetic evidence: the param is the base of a lea/add whose RESULT is itself a memory base, requiring base_addr_usage so a cmov/clamp lea does not over-pointerize a scalar.
    param_used_in_deref(func_start, *arg_reg) <--
        param_value_canon(func_start, arg_reg, canon),
        instr_in_function(lea_addr, func_start),
        ltl_inst(lea_addr, ?LTLInst::Lop(op, srcs, dst)),
        if matches!(op, Operation::Olea(_) | Operation::Oleal(_)),
        if !srcs.is_empty(),
        reg_rtl(lea_addr, srcs[0], canon),
        reg_rtl(lea_addr, *dst, result_canon),
        base_addr_usage(_, result_canon, _);

    // param_used_in_lea removed: a bare LEA base is not pointer evidence; the genuine LEA-base-then-deref case is covered by param_used_in_deref, gated on base_addr_usage of the result.

    relation param_used_in_null_check(Address, Mreg);

    param_used_in_null_check(func_start, *arg_reg) <--
        func_param_validated(func_start, arg_reg),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lop(op, srcs, _)),
        if is_null_comparison_op(op),
        for reg in srcs.iter(),
        if *reg == *arg_reg;

    relation param_type_evidence_pointer(Address, Mreg);

    param_type_evidence_pointer(func_start, mreg) <--
        param_used_in_deref(func_start, mreg);

    // param_used_in_lea is intentionally NOT a pointer signal: gcc emits LEA for plain integer arithmetic on a scalar param, and treating it as evidence declared count/length params void* and made their arithmetic wild.

    param_type_evidence_pointer(func_start, mreg) <--
        param_used_in_null_check(func_start, mreg);

    // U1: a parameter forwarded into a callee argument slot known to take a pointer is itself a pointer, the only path that saves a thin forwarder whose value-web is_ptr is cancelled by competing int evidence.
    relation callee_ptr_param_at(Node, usize);

    callee_ptr_param_at(call_node, *arg_idx) <--
        call_site(call_node, func_name),
        call_arg(call_node, arg_idx, _),
        known_func_param_is_ptr(func_name, arg_idx);

    callee_ptr_param_at(call_node, *arg_idx) <--
        call_site(call_node, func_name),
        call_arg(call_node, arg_idx, _),
        !emit_function(_, func_name, _),
        known_extern_signature(func_name, _, _, params),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xptr | XType::Xcharptr);

    // Symbol-form libc calls are not covered by call_site, so read the callee pointer arg directly off the symbol name.
    callee_ptr_param_at(call_node, *arg_idx) <--
        ltl_inst(call_node, ?LTLInst::Lcall(Either::Right(Either::Right(name)))),
        !emit_function(_, name, _),
        call_arg(call_node, arg_idx, _),
        known_func_param_is_ptr(name, arg_idx);

    callee_ptr_param_at(call_node, *arg_idx) <--
        ltl_inst(call_node, ?LTLInst::Lcall(Either::Right(Either::Right(name)))),
        !emit_function(_, name, _),
        call_arg(call_node, arg_idx, _),
        known_extern_signature(name, _, _, params),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xptr | XType::Xcharptr);

    // The param's incoming value (same canonical xtl reg via the entry copy) is passed as a pointer-typed call argument somewhere in the body.
    param_type_evidence_pointer(func_start, mreg) <--
        func_param_validated(func_start, mreg),
        reg_xtl(func_start, mreg, raw_v),
        xtl_canonical(raw_v, canon),
        instr_in_function(call_node, func_start),
        call_arg(call_node, pos, arg_reg),
        xtl_canonical(arg_reg, canon),
        callee_ptr_param_at(call_node, pos);

    relation param_inferred_type_is_pointer(Address, Mreg);

    param_inferred_type_is_pointer(func_start, mreg) <--
        func_param_validated(func_start, mreg),
        param_type_evidence_pointer(func_start, mreg);

    relation emit_function_param_candidate(Address, RTLReg);
    relation emit_function_param_is_pointer_candidate(Address, RTLReg);
    relation emit_function_return(Address, RTLReg);
    relation emit_function_has_return_candidate(Address);
    relation emit_function_void_candidate(Address);
    relation emit_function_return_type_candidate(Address, ClightType);

    emit_function(func_start, *name, func_start) <--
        func_stacksz(func_start, _func_end, name, stacksize);

    emit_function_param_candidate(func_start, *canonical) <--
        func_stacksz(func_start, _func_end, name, stacksize),
        func_arg_used_undefined(func_start, v),
        xtl_canonical(v, canonical);

    emit_function_param_candidate(func_start, *canonical) <--
        func_float_arg_used_undefined(func_start, v),
        xtl_canonical(v, canonical);

    // VS2013 /homeparams is definition-side signature evidence even when the
    // source never reads the parameter again.  Keeping the proved incoming
    // register in the prototype lets the same compiler switch reproduce the
    // otherwise-dead home store; the exact spill width below prevents an R8B
    // parameter from being widened to a full R8 store.
    emit_function_param_candidate(func_start, *canonical) <--
        win64_home_spill_candidate(_, func_start, param_reg, _),
        reg_xtl(func_start, *param_reg, raw_param),
        xtl_canonical(raw_param, canonical);

    emit_function_param_type_candidate(func_start, *canonical, xtype) <--
        win64_home_spill_candidate(_, func_start, param_reg, pos),
        win64_home_storage_signature(func_start, pos, move_class, width),
        !win64_home_storage_signature_conflict(func_start, pos),
        reg_xtl(func_start, *param_reg, raw_param),
        xtl_canonical(raw_param, canonical),
        if let Some(xtype) = home_move_xtype(*move_class, *width);

    emit_function_param_is_pointer_candidate(func_start, *canonical) <--
        emit_function_param_candidate(func_start, canonical),
        func_arg_used_undefined(func_start, raw_v),
        xtl_canonical(raw_v, canonical),
        reg_xtl(func_start, mreg, raw_v),
        param_inferred_type_is_pointer(func_start, mreg);


    relation function_return_point_reg(Address, Node, RTLReg);

    // Gated !func_returns_float: a float return has no AX return value, so exclude the AX point entirely or a stale address def is selected as THE return reg and defeats the float candidate.
    function_return_point_reg(func_start, addr, rtl_reg) <--
        emit_function(func_start, _, entry_node),
        instr_in_function(addr, func_start),
        ltl_inst(addr, LTLInst::Lreturn),
        !func_returns_float(func_start),
        ax_value_addr(addr, defaddr),
        reg_rtl(defaddr, Mreg::AX, rtl_reg);

    // Float-return point reg: bind the return to the real XMM0 def (mirror of the AX rule).
    function_return_point_reg(func_start, addr, rtl_reg) <--
        emit_function(func_start, _, entry_node),
        instr_in_function(addr, func_start),
        ltl_inst(addr, LTLInst::Lreturn),
        x0_value_addr(addr, defaddr),
        reg_rtl(defaddr, Mreg::X0, rtl_reg);

    emit_function_return(func_start, min_ret) <--
        function_return_point_reg(func_start, _, _),
        agg min_ret = ascent::aggregators::min(ret) in function_return_point_reg(func_start, _, ret);

    emit_function_has_return_candidate(addr) <--
        emit_function_return(addr, _);

    // Whole-function noreturn recovery: an emit_function with no Lreturn and no tailcall cannot return; computed here rather than linear_pass, where emit_function is still empty and the rule is dead.
    relation function_has_lreturn(Address);
    function_has_lreturn(func) <--
        emit_function(func, _, _),
        instr_in_function(addr, func),
        ltl_inst(addr, LTLInst::Lreturn);

    relation function_has_tailcall(Address);
    function_has_tailcall(func) <--
        emit_function(func, _, _),
        instr_in_function(addr, func),
        ltl_inst(addr, ?LTLInst::Ltailcall(_));

    relation function_noreturn(Address);
    function_noreturn(func) <--
        emit_function(func, _, _),
        !function_has_lreturn(func),
        !function_has_tailcall(func);

    relation emit_function_param_type_candidate(Address, RTLReg, XType);

    emit_function_param_type_candidate(func_start, param_rtl, xtype) <--
        emit_function_param_candidate(func_start, param_rtl),
        emit_var_type_candidate(param_rtl, xtype);

    emit_function_param_type_candidate(func_start, param_rtl, XType::Xfloat) <--
        emit_function_float_param(func_start, param_rtl),
        !emit_var_type_candidate(param_rtl, _);

    // A parameter with pointer evidence gets a pointer type candidate, since its own pre-copy value web often carries only int-width candidates; Xptr lifts but never downgrades a more refined type.
    emit_function_param_type_candidate(func_start, param_rtl, XType::Xptr) <--
        emit_function_param_is_pointer_candidate(func_start, param_rtl);

    relation emit_function_return_type_xtype_candidate(Address, XType);

    emit_function_return_type_xtype_candidate(func_start, xtype) <--
        emit_function_return(func_start, ret_rtl),
        emit_var_type_candidate(ret_rtl, xtype);

    // A partial AX zero idiom carries return-width evidence that the generic
    // Oxor -> integer-constant lowering intentionally loses.  This is the
    // exact VS2013 BOOLEAN pattern (`xor al,al`) and is
    // tied to a proved definition reaching RET, not merely to an instruction
    // somewhere in the function.
    emit_function_return_type_xtype_candidate(func_start, XType::Xint8unsigned) <--
        abi_shared_arg_slots(true),
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        ax_value_addr(ret_addr, zero_addr),
        instruction(zero_addr, _, _, "XOR", left, right, _, _, _, _),
        op_register(left, "AL"),
        op_register(right, "AL");

    // Emit Xlong for every 64-bit/pointer-typed return point, or an early-exit return 0 (Xint, priority 4) outranks a bare Xany64 (3) and truncates a genuine 64-bit return.
    emit_function_return_type_xtype_candidate(func_start, XType::Xlong) <--
        function_return_point_reg(func_start, _, ret_rtl),
        emit_var_type_candidate(ret_rtl, xt),
        if matches!(xt,
            XType::Xany64 | XType::Xlong | XType::Xlongunsigned
            | XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr
            | XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    // Same anti-truncation guard keyed on the POINTER CLASSIFICATION, for a returned pointer whose pointer-ness was severed to a bare Xint candidate; an int-returning function never classifies its return reg is_ptr.
    emit_function_return_type_xtype_candidate(func_start, XType::Xlong) <--
        function_return_point_reg(func_start, _, ret_rtl),
        is_ptr(ret_rtl);

    // BUG2: track which frame slots hold a pointer, then type any return reg reloading such a slot is_ptr, since an -O0 pointer return is spilled and its reload's value web is severed from the producing def.
    relation call_returns_ptr_value(Address);
    call_returns_ptr_value(call_addr) <--
        call_has_known_signature(call_addr, _, _, ret),
        if matches!(ret, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr
            | XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));
    call_returns_ptr_value(call_addr) <--
        external_call_site(call_addr, _, name),
        known_func_returns_ptr(name);

    relation stack_slot_stored_ptr(Address, i64);
    // (i) a stored value already classified is_ptr in rtl (an Olea result, internal ptr-returning callee, or deref-base value web).
    stack_slot_stored_ptr(func_start, *ofs) <--
        instr_in_function(spill, func_start),
        ltl_inst(spill, ?LTLInst::Lsetstack(src, _, ofs, _)),
        reg_def_used(defaddr, *src, *spill),
        reg_rtl(defaddr, *src, src_rtl),
        is_ptr(src_rtl);
    // (ii) the AX result of a pointer-returning call, keyed on the call signature directly since by-name externs have no rtl is_ptr.
    stack_slot_stored_ptr(func_start, *ofs) <--
        instr_in_function(spill, func_start),
        ltl_inst(spill, ?LTLInst::Lsetstack(src, _, ofs, _)),
        if *src == Mreg::AX,
        reg_def_used(call_addr, Mreg::AX, *spill),
        call_returns_ptr_value(call_addr);
    // (iii) the spill of an inferred-pointer parameter (the returned-output-array-param case).
    stack_slot_stored_ptr(func_start, *ofs) <--
        instr_in_function(spill, func_start),
        ltl_inst(spill, ?LTLInst::Lsetstack(src, _, ofs, _)),
        param_inferred_type_is_pointer(func_start, *src),
        arg_reg_param_live_at(func_start, spill, *src);

    is_ptr(ret_rtl) <--
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        ax_value_addr(ret_addr, defaddr),
        ltl_inst(defaddr, ?LTLInst::Lgetstack(_, ofs, _, Mreg::AX)),
        stack_slot_stored_ptr(func_start, *ofs),
        reg_rtl(defaddr, Mreg::AX, ret_rtl);

    // 3.8 R1: float return type for fused-op-fed X0 returns, whose def has no ltl Lop; derive the class from the op that structurally reaches the return (*fs results single, the rest double).
    emit_function_return_type_xtype_candidate(func_start, xt) <--
        func_returns_float(func_start),
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        float_load_op(def_addr, op, _, _, _, Mreg::X0, _),
        !has_ltl_op(def_addr),
        let xt = if matches!(op, Operation::Oaddfs | Operation::Osubfs | Operation::Omulfs | Operation::Odivfs | Operation::Osingleoffloat) { XType::Xsingle } else { XType::Xfloat };

    // (B) Float return-type class for an X0 return fed by a PLAIN reg-reg float op, tracing back through Omove copies; single iff the producing op is single-precision, gated on func_returns_float.
    #[local] relation float_value_class(Address, Mreg, bool);
    float_value_class(*addr, *dst, true) <--
        ltl_inst(addr, ?LTLInst::Lop(op, _, dst)),
        if crate::x86::types::is_single_operation(op);
    float_value_class(*addr, *dst, false) <--
        ltl_inst(addr, ?LTLInst::Lop(op, _, dst)),
        if crate::x86::types::is_float_operation(op);
    float_value_class(*addr, dst, single) <--
        float_load_op(addr, op, _, _, _, dst, _),
        let single = matches!(op, Operation::Oaddfs | Operation::Osubfs | Operation::Omulfs | Operation::Odivfs | Operation::Osingleoffloat | Operation::Osingleofint | Operation::Osingleoflong);
    float_value_class(*addr, *dst, single) <--
        float_arith_stack_op(addr, op, _, _, dst),
        let single = crate::x86::types::is_single_operation(op);
    float_value_class(*addr, *dst, single) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Omove, src_args, dst)),
        if src_args.len() == 1,
        let src = src_args[0],
        reg_def_used(src_def, src, *addr),
        float_value_class(src_def, src, single);

    emit_function_return_type_xtype_candidate(func_start, xt) <--
        func_returns_float(func_start),
        instr_in_function(ret_addr, func_start),
        ltl_inst(ret_addr, ?LTLInst::Lreturn),
        def_reaches_return(ret_addr, def_addr, Mreg::X0),
        float_value_class(def_addr, Mreg::X0, single),
        let xt = if *single { XType::Xsingle } else { XType::Xfloat };

    emit_function_return_type_xtype_candidate(callee_func, xtype) <--
        ltl_inst(call_addr, ?LTLInst::Lcall(Either::Right(Either::Left(callee_func)))),
        call_return_reg(call_addr, ret_rtl),
        emit_var_type_candidate(ret_rtl, xtype),
        if *xtype != XType::Xvoid,
        !emit_function_return(callee_func, _);

    // U3: tail-call-only forwarders have zero Lreturn, so the ladder types them Xvoid and a value-returning caller emits an invalid use of void; their true return type is the tail-call target's.
    relation func_has_lreturn(Address);
    func_has_lreturn(func_start) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lreturn);

    relation func_tailcalls_local(Address, Address);
    func_tailcalls_local(func_start, target) <--
        emit_function(func_start, _, _),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Ltailcall(Either::Right(Either::Left(target))));

    // (A) Body: synthesize a return reg for a tail-call-only function whose target returns a value, so clight_pass lowers it as ret = target(); return ret; recursing up the tail-call chain.
    emit_function_return(func_start, ret_rtl) <--
        func_tailcalls_local(func_start, target),
        !func_has_lreturn(func_start),
        emit_function_return(target, _),
        let ret_rtl = fresh_xtl_reg(*func_start, Mreg::AX);

    // (B) Decl type: inherit the tail-call target's non-void return type by positive recursion within the candidate relation.
    emit_function_return_type_xtype_candidate(func_start, xtype) <--
        func_tailcalls_local(func_start, target),
        !func_has_lreturn(func_start),
        emit_function_return_type_xtype_candidate(target, xtype),
        if *xtype != XType::Xvoid;

    emit_function_return_type_candidate(func_start, clight_ty) <--
        emit_function_return_type_xtype_candidate(func_start, xtype),
        let clight_ty = clight_type_from_xtype(xtype);


    relation called_address(Address);

    called_address(target) <--
        ltl_inst(_caller, ?LTLInst::Lcall(Either::Right(Either::Left(target)))),
        instr_in_function(target, _);

    called_address(target2) <--
        ltl_inst(_caller2, ?LTLInst::Ltailcall(Either::Right(Either::Left(target2)))),
        instr_in_function(target2, _);

    emit_function(func_entry, *sym_name, func_entry) <--
        called_address(func_entry),
        !func_stacksz(func_entry, _, _, _),
        !plt_entry(func_entry, _),
        !plt_block(func_entry, _),
        symbols(func_entry, sym_name, _);

    emit_function(func_entry, func_name, func_entry) <--
        called_address(func_entry),
        !func_stacksz(func_entry, _, _, _),
        !plt_entry(func_entry, _),
        !plt_block(func_entry, _),
        !symbols(func_entry, _, _),
        let func_name = Box::leak(format!("FUN_{:x}", func_entry).into_boxed_str()) as Symbol;

    emit_function(addr, *sym_name, addr) <--
        symbols(addr, sym_name, _),
        instr_in_function(addr, addr),
        !func_stacksz(addr, _, _, _),
        !called_address(addr),
        !plt_entry(addr, _),
        !plt_block(addr, _);

    emit_function_void_candidate(undiscov_addr2) <--
        called_address(undiscov_addr2),
        !func_stacksz(undiscov_addr2, _, _, _),
        instr_in_function(ret_point2, undiscov_addr2),
        ltl_inst(ret_point2, ?LTLInst::Lreturn),
        !reg_rtl(ret_point2, Mreg::AX, _);

    // A function has a return value if any return point resolves an AX value (direct def-use or recovered last-def); ax_value_addr is keyed on the return node.
    relation func_has_ax_return_value(Address);
    func_has_ax_return_value(func_start) <--
        instr_in_function(ret_point, func_start),
        ltl_inst(ret_point, ?LTLInst::Lreturn),
        ax_value_addr(ret_point, _);

    // Float/double returns place the result in XMM0 (X0), which the AX-keyed machinery misses; treat an X0 def reaching a return as a return value so they are not misclassified as void.
    func_has_ax_return_value(func_start) <--
        instr_in_function(ret_point, func_start),
        def_reaches_return(ret_point, _, Mreg::X0);

    // Genuine void: a function whose return points have no AX/X0 value source at all; without this the void fallback fabricates an undefined AX reg typed as `long`, emitting `return <uninitialized var>`.
    emit_function_void_candidate(func_start) <--
        func_stacksz(func_start, _, _, _),
        instr_in_function(ret_point, func_start),
        ltl_inst(ret_point, ?LTLInst::Lreturn),
        !func_has_ax_return_value(func_start);


    relation emit_function_param_count_candidate(Address, usize);

    // Total param count includes both register and stack-passed params
    emit_function_param_count_candidate(target_addr, reg_count + *stack_count) <--
        emit_function_param_candidate(target_addr, _),
        agg reg_count = ascent::aggregators::count() in emit_function_param_candidate(target_addr, _),
        emit_function_stack_param_count(target_addr, stack_count);

    emit_function_param_count_candidate(target_addr, *stack_count) <--
        emit_function(target_addr, _, _),
        !emit_function_param_candidate(target_addr, _),
        emit_function_stack_param_count(target_addr, stack_count);


    is_ptr(reg) <-- op_produces_ptr(_, reg);
    is_ptr(reg) <-- base_addr_usage(_, reg, _);
    is_not_ptr(reg) <-- op_produces_data(_, reg);

    is_not_ptr(reg) <--
        comparison_operand(_, cond, reg),
        if matches!(cond, Condition::Ccomp(_) | Condition::Ccompu(_)
            | Condition::Ccompimm(_, _) | Condition::Ccompuimm(_, _)),
        if !is_null_comparison_cond(cond);

    // USE-SIDE 64-bit width floor (c16): operands of a genuine 64-bit comparison are 64-bit, but contribute is_long ONLY, since a 64-bit comparison may compare two pointers.
    is_long(reg) <--
        comparison_operand(_, cond, reg),
        if matches!(cond, Condition::Ccompl(_) | Condition::Ccomplu(_)
            | Condition::Ccomplimm(_, _) | Condition::Ccompluimm(_, _));

    must_be_ptr(reg) <-- arg_constrained_as_ptr(_, reg);
    must_be_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        !emit_function(_, func_name, _),
        known_func_returns_ptr(func_name);
    must_be_ptr(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);


    arg_constrained_as_ptr(node, arg_reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_func_param_is_ptr(func_name, arg_idx);

    is_ptr(reg) <-- arg_constrained_as_ptr(_, reg);

    is_char_ptr(arg_reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        !emit_function(_, func_name, _),
        known_extern_signature(func_name, _, _, params),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xcharptr);

    // ABI 64-bit width floor (c16): an argument passed to an extern parameter known to be 64-bit is itself 64-bit; Xany64 is excluded as too ambiguous and would over-widen narrow args.
    is_long(arg_reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        !emit_function(_, func_name, _),
        known_extern_signature(func_name, _, _, params),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xlong | XType::Xlongunsigned);

    is_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        !emit_function(_, func_name, _),
        known_func_returns_ptr(func_name);

    // (A) A by-NAME pointer-returning extern call result is a pointer: the call_site-gated rules cover only by-address PLT calls, so a by-name malloc/realloc result never got is_ptr and its return truncated.
    is_ptr(ret_rtl) <--
        call_returns_ptr_value(call_addr),
        reg_rtl(call_addr, Mreg::AX, ret_rtl);

    is_char_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        !emit_function(_, func_name, _),
        known_extern_signature(func_name, _, ret_type, _),
        if matches!(ret_type, XType::Xcharptr);

    // (A) By-NAME char-pointer extern call result refines is_ptr to char*, keyed on the signature directly via reg_rtl(call_addr, AX); Xcharptr outranks the bare Xptr above.
    is_char_ptr(ret_rtl) <--
        call_has_known_signature(call_addr, name, _, ret_type),
        !emit_function(_, name, _),
        if matches!(ret_type, XType::Xcharptr),
        reg_rtl(call_addr, Mreg::AX, ret_rtl);

    // ABI 64-bit width floor (c16): the result of an extern call known to return a 64-bit value is 64-bit, from known_func_returns_long or an explicit Xlong in the full signature.
    is_long(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        !emit_function(_, func_name, _),
        known_func_returns_long(func_name);

    is_long(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        !emit_function(_, func_name, _),
        known_extern_signature(func_name, _, ret_type, _),
        if matches!(ret_type, XType::Xlong | XType::Xlongunsigned);

    is_ptr(b) <--
        is_ptr(a),
        alias_edge(a, b),
        !is_not_ptr(b);

    is_ptr(a) <--
        is_ptr(b),
        alias_edge(a, b),
        !is_not_ptr(a);

    is_not_ptr(b) <--
        is_not_ptr(a),
        alias_edge(a, b),
        !must_be_ptr(b);

    is_not_ptr(a) <--
        is_not_ptr(b),
        alias_edge(a, b),
        !must_be_ptr(a);

    is_char_ptr(b) <--
        is_char_ptr(a),
        alias_edge(a, b),
        !is_not_ptr(b);

    is_char_ptr(a) <--
        is_char_ptr(b),
        alias_edge(a, b),
        !is_not_ptr(a);

    // 64-bit width propagates across value-identity copies (c16): alias_edge connects same-value same-width webs, so this only completes Xint -> Xlong while ptr/float/subint still dominate.
    is_long(b) <-- is_long(a), alias_edge(a, b);
    is_long(a) <-- is_long(b), alias_edge(a, b);


    is_int(reg) <-- op_produces_int(_, reg);
    is_long(reg) <-- op_produces_long(_, reg);
    is_float(reg) <-- op_produces_float(_, reg);
    is_single(reg) <-- op_produces_single(_, reg);
    is_bool(reg) <-- op_produces_bool(_, reg);
    is_int8signed(reg) <-- op_produces_int8signed(_, reg);
    is_int8unsigned(reg) <-- op_produces_int8unsigned(_, reg);
    is_int16signed(reg) <-- op_produces_int16signed(_, reg);
    is_int16unsigned(reg) <-- op_produces_int16unsigned(_, reg);


    op_produces_data(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if (is_int_operation(op) || is_long_operation(op)) && !is_pointer_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);


    op_produces_int8signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast8signed, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int8unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast8unsigned, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int16signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast16signed, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int16unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast16unsigned, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast32signed, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocast32unsigned, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_int_operation(op) && !matches!(op,
            Operation::Ocast8signed | Operation::Ocast8unsigned |
            Operation::Ocast16signed | Operation::Ocast16unsigned),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_long_operation(op) && !matches!(op,
            Operation::Ocast32signed | Operation::Ocast32unsigned |
            Operation::Oleal(_)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    // USE-SIDE 64-bit width floor (c16): an operand of a genuine 64-bit arithmetic op is 64-bit, excluding width-changing casts, Omakelong, Oleal and the float-input conversions.
    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, mregs, _)),
        if is_long_operation(op) && !matches!(op,
            Operation::Ocast32signed | Operation::Ocast32unsigned |
            Operation::Omakelong | Operation::Oleal(_) |
            Operation::Olongoffloat | Operation::Olongofsingle),
        for mreg in mregs.iter(),
        reg_rtl(node, *mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_float_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_single_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_bool(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_boolean_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    // cmov/Olea un-pointerize: a LEA result is a pointer ONLY from use-based evidence, since gcc emits LEA for plain integer arithmetic; Oindirectsymbol stays unconditionally a pointer.
    op_produces_ptr(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ointconst(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Olongconst(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ofloatconst(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osingleconst(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ointoffloat, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ointofsingle, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Olongoffloat, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Olongofsingle, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ofloatofint, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ofloatoflong, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ofloatofsingle, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osingleoffloat, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osingleofint, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osingleoflong, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Omakelong, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Olowlong, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ohighlong, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);


    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt32 | MemoryChunk::MAny32),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt64 | MemoryChunk::MAny64),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MFloat64),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MFloat32),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_int8signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt8Signed),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_int8unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt8Unsigned | MemoryChunk::MBool),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_int16signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt16Signed),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_int16unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt16Unsigned),
        reg_rtl(node, *src_mreg, rtl_reg);

    op_produces_int(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt32 | MemoryChunk::MAny32),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_long(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt64 | MemoryChunk::MAny64),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_float(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MFloat64),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_single(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MFloat32),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int8signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt8Signed),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int8unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt8Unsigned | MemoryChunk::MBool),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int16signed(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt16Signed),
        reg_rtl(node, *dst_mreg, rtl_reg);

    op_produces_int16unsigned(node, rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MInt16Unsigned),
        reg_rtl(node, *dst_mreg, rtl_reg);


    is_unsigned(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_unsigned_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    is_signed(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if is_signed_operation(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    is_unsigned(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, args, _)),
        if is_unsigned_operation(op),
        for mreg in args.iter(),
        reg_rtl(node, *mreg, rtl_reg);

    is_signed(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, args, _)),
        if is_signed_operation(op),
        for mreg in args.iter(),
        reg_rtl(node, *mreg, rtl_reg);


    relation zero_int_const_reg(RTLReg);

    zero_int_const_reg(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ointconst(0), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    zero_int_const_reg(b) <-- zero_int_const_reg(a), alias_edge(a, b);
    zero_int_const_reg(a) <-- zero_int_const_reg(b), alias_edge(a, b);

    relation has_float_int_zero_conflict(RTLReg);
    has_float_int_zero_conflict(reg) <-- zero_int_const_reg(reg), is_int(reg), is_float(reg);
    has_float_int_zero_conflict(reg) <-- zero_int_const_reg(reg), is_long(reg), is_float(reg);
    has_float_int_zero_conflict(reg) <-- zero_int_const_reg(reg), is_int(reg), is_single(reg);
    has_float_int_zero_conflict(reg) <-- zero_int_const_reg(reg), is_long(reg), is_single(reg);


    relation has_type_conflict(RTLReg);
    has_type_conflict(reg) <-- is_ptr(reg), is_not_ptr(reg);
    relation has_float_ptr_conflict(RTLReg);
    has_float_ptr_conflict(reg) <-- is_float(reg), is_ptr(reg);
    has_float_ptr_conflict(reg) <-- is_float(reg), must_be_ptr(reg);
    has_float_ptr_conflict(reg) <-- is_single(reg), is_ptr(reg);
    has_float_ptr_conflict(reg) <-- is_single(reg), must_be_ptr(reg);
    has_long_type(reg) <-- has_type_conflict(reg);
    has_long_type(reg) <-- has_float_ptr_conflict(reg);

    has_ptr_type(reg) <-- is_ptr(reg), !is_not_ptr(reg);
    has_ptr_type(reg) <-- must_be_ptr(reg), !has_type_conflict(reg);
    has_char_ptr_type(reg) <-- is_char_ptr(reg), !is_not_ptr(reg);
    has_char_ptr_type(reg) <-- is_char_ptr(reg), must_be_ptr(reg), !has_type_conflict(reg);
    has_float_type(reg) <-- is_float(reg), !has_float_int_zero_conflict(reg);
    has_single_type(reg) <-- is_single(reg), !has_float_int_zero_conflict(reg);
    has_long_type(reg) <-- is_long(reg);
    has_int_type(reg) <-- is_int(reg);
    has_subint_type(reg) <-- is_int8signed(reg);
    has_subint_type(reg) <-- is_int8unsigned(reg);
    has_subint_type(reg) <-- is_int16signed(reg);
    has_subint_type(reg) <-- is_int16unsigned(reg);

    emit_var_type_candidate(reg, XType::Xcharptr) <--
        has_char_ptr_type(reg);

    emit_var_type_candidate(reg, XType::Xptr) <--
        has_ptr_type(reg),
        !has_char_ptr_type(reg);

    emit_var_type_candidate(reg, XType::Xsingle) <--
        has_single_type(reg),
        !has_ptr_type(reg),
        !has_float_ptr_conflict(reg);

    emit_var_type_candidate(reg, XType::Xfloat) <--
        has_float_type(reg),
        !has_ptr_type(reg),
        !has_single_type(reg),
        !has_float_ptr_conflict(reg);

    emit_var_type_candidate(reg, XType::Xint8signed) <--
        is_int8signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg);

    emit_var_type_candidate(reg, XType::Xint8unsigned) <--
        is_int8unsigned(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !is_int8signed(reg);

    emit_var_type_candidate(reg, XType::Xint16signed) <--
        is_int16signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg);

    emit_var_type_candidate(reg, XType::Xint16unsigned) <--
        is_int16unsigned(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !is_int16signed(reg);

    emit_var_type_candidate(reg, XType::Xbool) <--
        is_bool(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_long_type(reg),
        !has_int_type(reg),
        !has_subint_type(reg);


    emit_var_type_candidate(reg, XType::Xlongunsigned) <--
        has_long_type(reg),
        is_unsigned(reg),
        !is_signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg);

    emit_var_type_candidate(reg, XType::Xlong) <--
        has_long_type(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg),
        !is_unsigned(reg);

    emit_var_type_candidate(reg, XType::Xlong) <--
        has_long_type(reg),
        is_unsigned(reg),
        is_signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg);

    emit_var_type_candidate(reg, XType::Xintunsigned) <--
        has_int_type(reg),
        !has_long_type(reg),
        is_unsigned(reg),
        !is_signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg);

    emit_var_type_candidate(reg, XType::Xint) <--
        has_int_type(reg),
        !has_long_type(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg),
        !is_unsigned(reg);

    emit_var_type_candidate(reg, XType::Xint) <--
        has_int_type(reg),
        !has_long_type(reg),
        is_unsigned(reg),
        is_signed(reg),
        !has_ptr_type(reg),
        !has_float_type(reg),
        !has_single_type(reg),
        !has_subint_type(reg);

    relation single_def_const(RTLReg, Constant);

    single_def_const(dst_rtl, cst) <--
        rtl_inst_candidate(_, ?RTLInst::Iop(op, args, dst_rtl)),
        if args.is_empty(),
        if let Some(cst) = crate::decompile::passes::cminor_pass::constant_from_operation(op);

    #[local] relation var_used(RTLReg);

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Iop(_, args, _)),
        for reg in args.iter();

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Iload(_, _, args, _)),
        for reg in args.iter();

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Istore(_, _, args, _)),
        for reg in args.iter();
    var_used(*src) <--
        rtl_inst_candidate(_, ?RTLInst::Istore(_, _, _, src));

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Icall(_, callee, _, _, _)),
        if let Either::Left(reg) = callee;
    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Icall(_, _, args, _, _)),
        for reg in args.iter();

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Itailcall(_, callee, _)),
        if let Either::Left(reg) = callee;
    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Itailcall(_, _, args)),
        for reg in args.iter();

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Icond(_, args, _, _)),
        for reg in args.iter();

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Ijumptable(reg, _));

    var_used(*reg) <--
        rtl_inst_candidate(_, ?RTLInst::Ireturn(reg));

    relation dead_def(RTLReg);
    dead_def(*dst) <--
        rtl_inst_candidate(_, ?RTLInst::Icall(_, _, _, Some(dst), _)),
        !var_used(*dst);

}

pub(crate) const FUNC_ARG_DIST: i64 = 256;

#[inline]
fn is_shift_or_rotate_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Oshl
            | Operation::Oshr
            | Operation::Oshru
            | Operation::Oshll
            | Operation::Oshrl
            | Operation::Oshrlu
    )
}

// True for a floating-point register; x86's XMM file and AArch64's V file both answer this, so it is asked of the register file rather than one architecture's names.
fn is_float_mreg(r: &Mreg) -> bool {
    match r {
        Mreg::X86(x) => x.is_xmm(),
        Mreg::A64(a) => a.is_vec(),
        Mreg::Unknown => false,
    }
}

#[inline]
pub fn infer_signature_from_args(args: &[RTLReg], has_return: bool) -> Signature {
    let sig_args: Vec<XType> = args.iter().map(|_| XType::Xlong).collect();
    let sig_res = if has_return {
        XType::Xint
    } else {
        XType::Xvoid
    };
    Signature {
        sig_args: Arc::new(sig_args),
        sig_res,
        sig_cc: CallConv::default(),
    }
}

// Dense register id packed into the low MREG_DISCRIMINANT_BITS: x86 0..=33 is fixed (query.rs hardcodes 33 for Unknown) and AArch64's 65 follow at 34..=98, which is why the field is 7 bits.
fn mreg_discriminant(reg: Mreg) -> u64 {
    match reg {
        Mreg::Unknown => 33,
        Mreg::X86(x) => match x {
            X86Mreg::AX => 0,
            X86Mreg::BX => 1,
            X86Mreg::CX => 2,
            X86Mreg::DX => 3,
            X86Mreg::SI => 4,
            X86Mreg::DI => 5,
            X86Mreg::BP => 6,
            X86Mreg::R8 => 7,
            X86Mreg::R9 => 8,
            X86Mreg::R10 => 9,
            X86Mreg::R11 => 10,
            X86Mreg::R12 => 11,
            X86Mreg::R13 => 12,
            X86Mreg::R14 => 13,
            X86Mreg::R15 => 14,
            X86Mreg::X0 => 15,
            X86Mreg::X1 => 16,
            X86Mreg::X2 => 17,
            X86Mreg::X3 => 18,
            X86Mreg::X4 => 19,
            X86Mreg::X5 => 20,
            X86Mreg::X6 => 21,
            X86Mreg::X7 => 22,
            X86Mreg::X8 => 23,
            X86Mreg::X9 => 24,
            X86Mreg::X10 => 25,
            X86Mreg::X11 => 26,
            X86Mreg::X12 => 27,
            X86Mreg::X13 => 28,
            X86Mreg::X14 => 29,
            X86Mreg::X15 => 30,
            X86Mreg::FP0 => 31,
            X86Mreg::SP => 32,
            X86Mreg::Unknown => 33,
        },
        Mreg::A64(a) => 34 + a.index(),
    }
}

pub fn build_call_args<'a>(
    inp: impl Iterator<Item = (&'a usize, &'a RTLReg)>,
) -> impl Iterator<Item = Args> {
    let mut pairs: Vec<(usize, RTLReg)> = inp.map(|(pos, reg)| (*pos, *reg)).collect();
    pairs.sort();
    pairs.dedup_by_key(|(pos, _)| *pos);

    // Scatter by position into a dense vector padded with DEFAULT_VAR so a missing leading arg leaves a sentinel hole instead of left-shifting later args into its slot.
    let len = pairs.last().map(|(pos, _)| *pos + 1).unwrap_or(0);
    let mut args: Vec<RTLReg> = vec![DEFAULT_VAR as RTLReg; len];
    for (pos, reg) in pairs {
        args[pos] = reg;
    }
    std::iter::once(Arc::new(args))
}

/// Build an Addrmode for a CMP/Iload operand with optional base, index, and scale; falls back to base-only when the index register is absent.
pub fn build_cmp_addrmode(base_str: &str, idx_str: &str, scale: i64, disp: i64) -> Addrmode {
    let has_base = base_str != "NONE" && !base_str.is_empty();
    let has_idx = idx_str != "NONE" && !idx_str.is_empty();
    let base = if has_base {
        Some(Ireg::from(base_str))
    } else {
        None
    };
    let index = if has_idx {
        Some((Ireg::from(idx_str), scale))
    } else {
        None
    };
    Addrmode {
        base,
        index,
        disp: Displacement::from(disp),
    }
}

/// Translate a CMP operand already classified as generic pointer memory.
///
/// CompCert's ordinary reverse translator intentionally maps scalar RBP/RSP
/// address modes to `Ainstack` with no register arguments.  That is correct
/// only after a use-specific frame proof.  On the generic CMP route those
/// same architectural registers are scratch pointers, so retaining the
/// `Ainstack` shortcut would leave the cross-node JCC/SETcc consumer reading a
/// temporary for which no pointer-based root load can be formed.  Addr32 keeps
/// using the shared sized translator because its explicit zero-extension is
/// part of the address semantics.
pub fn transl_generic_cmp_addressing_rev_sized(
    base_str: &str,
    idx_str: &str,
    scale: i64,
    disp: i64,
    address_size: u8,
) -> Result<(Addressing, Vec<Mreg>), String> {
    let has_index = idx_str != "NONE" && !idx_str.is_empty();
    if address_size != 4
        && !has_index
        && matches!(base_str, "RBP" | "RSP")
    {
        return Ok((
            Addressing::Aindexed(disp),
            vec![Mreg::x86(base_str)],
        ));
    }

    transl_addressing_rev_sized(
        build_cmp_addrmode(base_str, idx_str, scale, disp),
        None,
        address_size,
    )
}

// Width of the machine-register field in fresh_xtl_reg: 7, not 6, since AArch64 pushes the id range to 98; query.rs::is_fabricated_operand decodes it and must use the same width.
pub(crate) const MREG_DISCRIMINANT_BITS: u32 = 7;
pub(crate) const MREG_DISCRIMINANT_MASK: u64 = (1 << MREG_DISCRIMINANT_BITS) - 1;

pub(crate) fn fresh_xtl_reg(node: Node, reg: Mreg) -> RTLReg {
    let mreg_id = mreg_discriminant(reg);
    (1u64 << 63) | (node << MREG_DISCRIMINANT_BITS) | (mreg_id & MREG_DISCRIMINANT_MASK)
}

// SETcc mnemonic -> the TestCond it materializes as a 0/1 boolean, used by the CMP+SETcc memory-operand Ocmp fusion to recover the dropped boolean.
pub(crate) fn setcc_mnem_testcond(mnem: &str) -> Option<TestCond> {
    Some(match mnem {
        "SETE" | "SETZ" => TestCond::CondE,
        "SETNE" | "SETNZ" => TestCond::CondNe,
        "SETL" => TestCond::CondL,
        "SETLE" => TestCond::CondLe,
        "SETG" => TestCond::CondG,
        "SETGE" => TestCond::CondGe,
        "SETB" | "SETC" => TestCond::CondB,
        "SETBE" => TestCond::CondBe,
        "SETA" => TestCond::CondA,
        "SETAE" | "SETNC" => TestCond::CondAe,
        _ => return None,
    })
}

pub(crate) const FRESH_NS_REG_DST: u64 = 1 << 55;
pub(crate) const FRESH_NS_SP_BASE: u64 = 1 << 56;
pub(crate) const FRESH_NS_STACK_PARAM: u64 = 1 << 57;
pub(crate) const FRESH_NS_HOME_SLOT: u64 = 1 << 58;

// Stack cells share the fresh-register node encoding so ownership and
// statement-order consumers can still recover their originating node, but use
// a reserved low tag rather than masquerading as a real BP value.  All real
// x86/AArch64 mreg discriminants are <= 98, leaving 127 collision-free.
const STACK_CELL_DISCRIMINANT: u64 = MREG_DISCRIMINANT_MASK;

pub(crate) fn fresh_stack_cell_reg(node: Node) -> RTLReg {
    (1u64 << 63) | (node << MREG_DISCRIMINANT_BITS) | STACK_CELL_DISCRIMINANT
}

pub(crate) fn fresh_stack_param_reg(func_addr: Node, stack_idx: usize) -> RTLReg {
    (1u64 << 63) | FRESH_NS_STACK_PARAM | (func_addr << 6) | ((stack_idx as u64) & 0x3F)
}

pub(crate) fn fresh_home_slot_reg(func_addr: Node, home_pos: usize) -> RTLReg {
    debug_assert!(home_pos < 4);
    (1u64 << 63) | FRESH_NS_HOME_SLOT | (func_addr << 2) | ((home_pos as u64) & 0x3)
}

pub fn convert_builtin_arg(
    node: Node,
    arg: &BuiltinArg<Mreg>,
    reg_rtl_map: &HashMap<(Node, Mreg), RTLReg>,
) -> BuiltinArg<RTLReg> {
    match arg {
        BuiltinArg::BA(mreg) => BuiltinArg::BA(
            reg_rtl_map
                .get(&(node, *mreg))
                .copied()
                .unwrap_or(DEFAULT_VAR as u64),
        ),
        BuiltinArg::BAInt(z) => BuiltinArg::BAInt(*z),
        BuiltinArg::BALong(z) => BuiltinArg::BALong(*z),
        BuiltinArg::BAFloat(f) => BuiltinArg::BAFloat(*f),
        BuiltinArg::BASingle(f) => BuiltinArg::BASingle(*f),
        BuiltinArg::BALoadStack(chunk, ptrofs) => BuiltinArg::BALoadStack(*chunk, *ptrofs),
        BuiltinArg::BAAddrStack(ptrofs) => BuiltinArg::BAAddrStack(*ptrofs),
        BuiltinArg::BALoadGlobal(chunk, ident, ptrofs) => {
            BuiltinArg::BALoadGlobal(*chunk, ident.clone(), *ptrofs)
        }
        BuiltinArg::BAAddrGlobal(ident, ptrofs) => BuiltinArg::BAAddrGlobal(ident.clone(), *ptrofs),
        BuiltinArg::BASplitLong(a, b) => BuiltinArg::BASplitLong(
            Box::new(convert_builtin_arg(node, a, reg_rtl_map)),
            Box::new(convert_builtin_arg(node, b, reg_rtl_map)),
        ),
        BuiltinArg::BAAddPtr(a, b) => BuiltinArg::BAAddPtr(
            Box::new(convert_builtin_arg(node, a, reg_rtl_map)),
            Box::new(convert_builtin_arg(node, b, reg_rtl_map)),
        ),
    }
}

pub fn convert_ltl_builtins_to_rtl(
    builtin_data: &Vec<(Node, Symbol, Arc<Vec<BuiltinArg<Mreg>>>, BuiltinArg<Mreg>)>,
    reg_rtl_map: &HashMap<(Node, Mreg), RTLReg>,
) -> Vec<(Node, RTLInst)> {
    builtin_data
        .iter()
        .map(|(addr, name, args, result)| {
            let args_rtl: Vec<BuiltinArg<RTLReg>> = args
                .iter()
                .map(|arg| convert_builtin_arg(*addr, arg, reg_rtl_map))
                .collect();
            let result_rtl = convert_builtin_arg(*addr, result, reg_rtl_map);
            let inst = RTLInst::Ibuiltin(name.to_string(), args_rtl, result_rtl);
            (*addr, inst)
        })
        .collect()
}

// Ascent aggregator: collect a node's reg_rtl entries into sorted (Mreg, RTLReg) pairs, keeping the minimum RTLReg per Mreg (matches the imperative builder).
pub fn collect_builtin_reg_pairs<'a>(
    inp: impl Iterator<Item = (&'a Mreg, &'a RTLReg)>,
) -> impl Iterator<Item = Arc<Vec<(Mreg, RTLReg)>>> {
    let mut groups: HashMap<Mreg, RTLReg> = HashMap::new();
    for (m, r) in inp {
        groups
            .entry(*m)
            .and_modify(|cur| {
                if *r < *cur {
                    *cur = *r;
                }
            })
            .or_insert(*r);
    }
    let mut pairs: Vec<(Mreg, RTLReg)> = groups.into_iter().collect();
    pairs.sort_by_key(|(m, r)| (mreg_discriminant(*m), *r));
    std::iter::once(Arc::new(pairs))
}

// Convert one LTL builtin to RTLInst::Ibuiltin using a per-node Mreg->RTLReg map.
pub fn build_builtin_inst(
    node: Node,
    name: &Symbol,
    args: &Arc<Vec<BuiltinArg<Mreg>>>,
    result: &BuiltinArg<Mreg>,
    pairs: &[(Mreg, RTLReg)],
) -> RTLInst {
    let reg_rtl_map: HashMap<(Node, Mreg), RTLReg> =
        pairs.iter().map(|&(m, r)| ((node, m), r)).collect();
    let args_rtl: Vec<BuiltinArg<RTLReg>> = args
        .iter()
        .map(|arg| convert_builtin_arg(node, arg, &reg_rtl_map))
        .collect();
    let result_rtl = convert_builtin_arg(node, result, &reg_rtl_map);
    RTLInst::Ibuiltin(name.to_string(), args_rtl, result_rtl)
}

pub(crate) fn adjust_condition_size(cond: Condition, size: usize) -> Condition {
    match (cond, size) {
        (Condition::Ccompl(c), 4) => Condition::Ccomp(c),
        (Condition::Ccomplu(c), 4) => Condition::Ccompu(c),
        (Condition::Ccomp(c), 8) => Condition::Ccompl(c),
        (Condition::Ccompu(c), 8) => Condition::Ccomplu(c),
        (c, _) => c,
    }
}

// Counter-branch split helper: only Ceq/Cne are shift-invariant under addition, plus the sub $1;jae/jb carry form, where unsigned >= 1 is exactly != 0 and so is an equality in disguise.
pub(crate) fn counter_branch_shifted_cond(cond: &Condition, k: i64) -> Option<Condition> {
    let is_eq = |cmp: &Comparison| matches!(cmp, Comparison::Ceq | Comparison::Cne);
    let carry_to_eq = |cmp: &Comparison| match cmp {
        Comparison::Cge => Some(Comparison::Cne),
        Comparison::Clt => Some(Comparison::Ceq),
        _ => None,
    };
    match cond {
        Condition::Ccompimm(cmp, m) if is_eq(cmp) => Some(Condition::Ccompimm(*cmp, *m + k)),
        Condition::Ccompuimm(cmp, m) if is_eq(cmp) => Some(Condition::Ccompuimm(*cmp, *m + k)),
        Condition::Ccomplimm(cmp, m) if is_eq(cmp) => Some(Condition::Ccomplimm(*cmp, *m + k)),
        Condition::Ccompluimm(cmp, m) if is_eq(cmp) => Some(Condition::Ccompluimm(*cmp, *m + k)),
        Condition::Ccompuimm(cmp, 1) => carry_to_eq(cmp).map(|eq| Condition::Ccompuimm(eq, k)),
        Condition::Ccompluimm(cmp, 1) => carry_to_eq(cmp).map(|eq| Condition::Ccompluimm(eq, k)),
        _ => None,
    }
}

pub(crate) fn extract_builtin_arg_regs(arg: &BuiltinArg<Mreg>) -> Vec<Mreg> {
    match arg {
        BuiltinArg::BA(reg) => vec![*reg],
        BuiltinArg::BASplitLong(a, b) => {
            let mut regs = extract_builtin_arg_regs(a.as_ref());
            regs.extend(extract_builtin_arg_regs(b.as_ref()));
            regs
        }
        BuiltinArg::BAAddPtr(a, b) => {
            let mut regs = extract_builtin_arg_regs(a.as_ref());
            regs.extend(extract_builtin_arg_regs(b.as_ref()));
            regs
        }
        _ => vec![],
    }
}

pub fn is_rip(reg: &str) -> bool {
    reg.to_uppercase().ends_with("IP")
}

pub fn typ_to_chunk(typ: Typ) -> MemoryChunk {
    match typ {
        Typ::Tint => MemoryChunk::MInt32,
        Typ::Tlong => MemoryChunk::MInt64,
        Typ::Tfloat => MemoryChunk::MFloat64,
        Typ::Tsingle => MemoryChunk::MFloat32,
        Typ::Tany32 => MemoryChunk::MAny32,
        Typ::Tany64 => MemoryChunk::MAny64,
        Typ::Unknown => MemoryChunk::Unknown,
    }
}

#[inline]
pub fn is_null_comparison_cond(cond: &Condition) -> bool {
    matches!(
        cond,
        Condition::Ccompimm(Comparison::Ceq, 0)
            | Condition::Ccompuimm(Comparison::Ceq, 0)
            | Condition::Ccompimm(Comparison::Cne, 0)
            | Condition::Ccompuimm(Comparison::Cne, 0)
            | Condition::Ccomplimm(Comparison::Ceq, 0)
            | Condition::Ccompluimm(Comparison::Ceq, 0)
            | Condition::Ccomplimm(Comparison::Cne, 0)
            | Condition::Ccompluimm(Comparison::Cne, 0)
    )
}

pub fn is_null_comparison_op(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Ocmp(Condition::Ccompimm(Comparison::Ceq, 0))
            | Operation::Ocmp(Condition::Ccompuimm(Comparison::Ceq, 0))
            | Operation::Ocmp(Condition::Ccompimm(Comparison::Cne, 0))
            | Operation::Ocmp(Condition::Ccompuimm(Comparison::Cne, 0))
            | Operation::Ocmp(Condition::Ccomplimm(Comparison::Ceq, 0))
            | Operation::Ocmp(Condition::Ccompluimm(Comparison::Ceq, 0))
            | Operation::Ocmp(Condition::Ccomplimm(Comparison::Cne, 0))
            | Operation::Ocmp(Condition::Ccompluimm(Comparison::Cne, 0))
    )
}

pub fn chunk_size_bits(chunk: &MemoryChunk) -> u8 {
    match chunk {
        MemoryChunk::MBool | MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned => 8,
        MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => 16,
        MemoryChunk::MInt32 | MemoryChunk::MAny32 | MemoryChunk::MFloat32 => 32,
        MemoryChunk::MInt64 | MemoryChunk::MAny64 | MemoryChunk::MFloat64 => 64,
        MemoryChunk::Unknown => 64,
    }
}

pub(crate) fn build_xtype_vec<'a>(
    inp: impl Iterator<Item = (&'a usize, &'a XType)>,
) -> impl Iterator<Item = Arc<Vec<XType>>> {
    use crate::decompile::passes::clight_pass::xtype_refine_priority;
    // Exactly one xtype per position, keeping the most-refined candidate: without the dedup two candidates produced two vec entries, shifting every later param and emitting a scalar signature for a pointer.
    let mut by_pos: std::collections::BTreeMap<usize, XType> = std::collections::BTreeMap::new();
    for (pos, xt) in inp {
        by_pos
            .entry(*pos)
            .and_modify(|cur| {
                if xtype_refine_priority(xt) > xtype_refine_priority(cur) {
                    *cur = *xt;
                }
            })
            .or_insert(*xt);
    }
    let args: Vec<XType> = by_pos.into_values().collect();
    std::iter::once(Arc::new(args))
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum UnsafeHomeShape {
    Store {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        peer: Mreg,
        move_class: usize,
        width: usize,
    },
    Load {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        peer: Mreg,
        move_class: usize,
        width: usize,
    },
    ExtendLoad {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        peer: Mreg,
        width: usize,
        op: Operation,
    },
    Lea {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        peer: Mreg,
    },
    ArithRead {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        peer: Mreg,
        width: usize,
        op: Operation,
    },
    Compare {
        base: Mreg,
        raw_ofs: i64,
        entry_ofs: i64,
        width: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum UnsafeHomeRewrite {
    Store { source: RTLReg, slot: RTLReg },
    Load { slot: RTLReg, destination: RTLReg },
    ExtendLoad {
        slot: RTLReg,
        destination: RTLReg,
        op: Operation,
    },
    Lea { entry_ofs: i64, destination: RTLReg },
    ArithRead {
        slot: RTLReg,
        destination: RTLReg,
        op: Operation,
    },
    Compare { inst: RTLInst },
}

fn home_move_xtype(move_class: usize, width: usize) -> Option<XType> {
    match (move_class, width) {
        (0, 1) => Some(XType::Xint8unsigned),
        (0, 2) => Some(XType::Xint16unsigned),
        (0, 4) => Some(XType::Xint),
        (0, 8) => Some(XType::Xany64),
        (1, 4) => Some(XType::Xsingle),
        (2, 8) => Some(XType::Xfloat),
        _ => None,
    }
}

// Reassert exact canonical-home backing types after a type-producing pass.
// These synthetic storage IDs are deliberately not value-web peers: copying a
// source's narrower/pointer refinement onto the addressable cell changes its
// width or class and can make final C declarations disagree with the rewritten
// full-cell loads/stores.
pub(crate) fn enforce_win64_home_slot_types(db: &mut DecompileDB) {
    let locked: BTreeMap<RTLReg, XType> = db
        .rel_iter::<(RTLReg, XType)>("win64_home_slot_type")
        .map(|(reg, xtype)| (*reg, *xtype))
        .collect();
    if locked.is_empty() {
        return;
    }

    let mut filtered: BTreeSet<(RTLReg, XType)> = db
        .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
        .filter(|(reg, xtype)| locked.get(reg).map_or(true, |exact| *exact == *xtype))
        .copied()
        .collect();
    filtered.extend(locked);
    db.rel_set(
        "emit_var_type_candidate",
        filtered.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

fn collect_builtin_uses(arg: &BuiltinArg<RTLReg>, used: &mut BTreeSet<RTLReg>) {
    match arg {
        BuiltinArg::BA(reg) => {
            used.insert(*reg);
        }
        BuiltinArg::BASplitLong(left, right) | BuiltinArg::BAAddPtr(left, right) => {
            collect_builtin_uses(left, used);
            collect_builtin_uses(right, used);
        }
        _ => {}
    }
}

fn collect_rtl_uses(inst: &RTLInst, used: &mut BTreeSet<RTLReg>) {
    match inst {
        RTLInst::Inop | RTLInst::Ibranch(_) => {}
        RTLInst::Iop(_, args, _) | RTLInst::Iload(_, _, args, _) => {
            used.extend(args.iter().copied());
        }
        RTLInst::Istore(_, _, args, source) => {
            used.extend(args.iter().copied());
            used.insert(*source);
        }
        RTLInst::Icall(_, callee, args, _, _) | RTLInst::Itailcall(_, callee, args) => {
            if let Either::Left(reg) = callee {
                used.insert(*reg);
            }
            used.extend(args.iter().copied());
        }
        RTLInst::Ibuiltin(_, args, _) => {
            for arg in args {
                collect_builtin_uses(arg, used);
            }
        }
        RTLInst::Icond(_, args, _, _) => {
            used.extend(args.iter().copied());
        }
        RTLInst::Ijumptable(reg, _) | RTLInst::Ireturn(reg) => {
            used.insert(*reg);
        }
    }
}

const SYNTHETIC_NODE_MASK: u64 = (1u64 << 62) | (1u64 << 63);

fn real_instruction_for_node(node: Node, address_sizes: &BTreeMap<Address, u8>) -> Option<Address> {
    if address_sizes.contains_key(&node) {
        return Some(node);
    }
    let real = node & !SYNTHETIC_NODE_MASK;
    address_sizes.contains_key(&real).then_some(real)
}

fn normalize_addr32_addressing(addressing: &Addressing, args: &Args) -> Option<Addressing> {
    let (inner, already_wrapped) = match addressing {
        Addressing::Aaddr32(inner) => (inner.as_ref(), true),
        other => (other, false),
    };
    let expected_args = match inner {
        Addressing::Aindexed(_) | Addressing::Ascaled(_, _) => 1,
        Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _) => 2,
        // Stack, symbolic, unknown, and recursively wrapped modes have no
        // sound 32-bit effective-address interpretation here.
        Addressing::Aglobal(_, _)
        | Addressing::Abased(_, _)
        | Addressing::Abasedscaled(_, _, _)
        | Addressing::Ainstack(_)
        | Addressing::Aaddr32(_)
        | Addressing::Unknown => return None,
    };
    if args.len() != expected_args {
        return None;
    }
    Some(if already_wrapped {
        addressing.clone()
    } else {
        Addressing::Aaddr32(Box::new(addressing.clone()))
    })
}

// Final structural guard for every path which can materialize a memory
// operation without passing through Mach (fused arithmetic/compare, immediate
// stores, and memory-indirect calls). Synthetic nodes inherit the addr-size of
// their real instruction only after the cleared address is proven to exist.
fn normalize_addr32_rtl_outputs(db: &mut DecompileDB) {
    if db.abi().arch != crate::abi::Arch::X86_64 {
        return;
    }
    let address_sizes: BTreeMap<Address, u8> = db
        .rel_iter::<(Address, u8)>("instruction_address_size")
        .copied()
        .collect();
    if !address_sizes.values().any(|size| *size == 4) {
        return;
    }
    let mnemonics: BTreeMap<Address, &'static str> = db
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
        .map(|(addr, _, _, mnemonic, _, _, _, _, _, _)| (*addr, *mnemonic))
        .collect();
    let indirect_operands: BTreeSet<Symbol> = db
        .rel_iter::<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )>("op_indirect")
        .map(|(operand, ..)| *operand)
        .collect();
    let addr32_memory_reals: BTreeSet<Address> = db
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
        .filter_map(|(addr, _, _, mnemonic, op1, op2, op3, op4, _, _)| {
            (address_sizes.get(addr) == Some(&4)
                && *mnemonic != "NOP"
                && [*op1, *op2, *op3, *op4]
                    .iter()
                    .any(|operand| indirect_operands.contains(operand)))
            .then_some(*addr)
        })
        .collect();
    let mut owners: BTreeMap<Address, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        if let Some(real) = real_instruction_for_node(*node, &address_sizes) {
            owners.entry(real).or_default().insert(*function);
        }
    }

    let mut invalid_reals = BTreeSet::new();
    let mut handled_reals = BTreeSet::new();
    let mut candidates = Vec::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate") {
        let Some(real) = real_instruction_for_node(*node, &address_sizes) else {
            candidates.push((*node, inst.clone()));
            continue;
        };
        if address_sizes.get(&real) != Some(&4) {
            candidates.push((*node, inst.clone()));
            continue;
        }

        let mut address_bearing = false;
        let normalized = match inst {
            RTLInst::Iload(chunk, addressing, args, destination) => {
                address_bearing = true;
                normalize_addr32_addressing(addressing, args).map(|addressing| {
                    RTLInst::Iload(*chunk, addressing, args.clone(), *destination)
                })
            }
            RTLInst::Istore(chunk, addressing, args, source) => {
                address_bearing = true;
                normalize_addr32_addressing(addressing, args)
                    .map(|addressing| RTLInst::Istore(*chunk, addressing, args.clone(), *source))
            }
            RTLInst::Iop(Operation::Olea(addressing), args, destination)
                if mnemonics.get(&real) == Some(&"LEA") =>
            {
                address_bearing = true;
                normalize_addr32_addressing(addressing, args).map(|addressing| {
                    RTLInst::Iop(Operation::Olea(addressing), args.clone(), *destination)
                })
            }
            RTLInst::Iop(Operation::Oleal(addressing), args, destination)
                if mnemonics.get(&real) == Some(&"LEA") =>
            {
                address_bearing = true;
                normalize_addr32_addressing(addressing, args).map(|addressing| {
                    RTLInst::Iop(Operation::Oleal(addressing), args.clone(), *destination)
                })
            }
            _ => Some(inst.clone()),
        };
        if let Some(inst) = normalized {
            if address_bearing {
                handled_reals.insert(real);
            }
            candidates.push((*node, inst));
        } else {
            invalid_reals.insert(real);
        }
    }
    candidates.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    candidates.dedup();
    db.rel_set(
        "rtl_inst_candidate",
        candidates.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let mut indirect_loads = Vec::new();
    for (node, target, chunk, addressing, args) in
        db.rel_iter::<(Node, RTLReg, MemoryChunk, Addressing, Args)>("call_through_memory_load")
    {
        let Some(real) = real_instruction_for_node(*node, &address_sizes) else {
            indirect_loads.push((*node, *target, *chunk, addressing.clone(), args.clone()));
            continue;
        };
        if address_sizes.get(&real) != Some(&4) {
            indirect_loads.push((*node, *target, *chunk, addressing.clone(), args.clone()));
            continue;
        }
        if let Some(addressing) = normalize_addr32_addressing(addressing, args) {
            handled_reals.insert(real);
            indirect_loads.push((*node, *target, *chunk, addressing, args.clone()));
        } else {
            invalid_reals.insert(real);
        }
    }
    indirect_loads.sort_by_cached_key(|row| format!("{row:?}"));
    indirect_loads.dedup();
    db.rel_set(
        "call_through_memory_load",
        indirect_loads
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    // A decoded addr32 memory operand must never disappear merely because no
    // lowering rule recognized its mnemonic. Require one normalized
    // address-bearing artifact or an already structured unsupported reason.
    let existing_reasons: BTreeSet<(Address, Address, Symbol)> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .copied()
        .collect();
    for real in addr32_memory_reals {
        if handled_reals.contains(&real) {
            continue;
        }
        let fully_diagnosed = owners.get(&real).is_some_and(|functions| {
            !functions.is_empty()
                && functions.iter().all(|function| {
                    existing_reasons
                        .iter()
                        .any(|(owner, access, _)| owner == function && *access == real)
                })
        });
        if !fully_diagnosed {
            invalid_reals.insert(real);
        }
    }

    // RTL is a candidate relation: one decoded instruction can have several
    // speculative lowerings.  A rejected alternative must not poison the
    // whole instruction when another address-bearing candidate normalized
    // successfully.  The final safety filter only needs a structured reason
    // when no sound addr32 interpretation survived.
    invalid_reals.retain(|real| !handled_reals.contains(real));

    // Addr32 is an arithmetic address construction, not evidence that either
    // input register is a native pointer/structure base.
    let ptr_rows: ascent::boxcar::Vec<(Node, RTLReg)> = db
        .rel_iter::<(Node, RTLReg)>("op_produces_ptr")
        .filter(
            |(node, _)| match real_instruction_for_node(*node, &address_sizes) {
                Some(real) => {
                    address_sizes.get(&real) != Some(&4) || mnemonics.get(&real) != Some(&"LEA")
                }
                None => true,
            },
        )
        .copied()
        .collect();
    db.rel_set("op_produces_ptr", ptr_rows);

    if !invalid_reals.is_empty() {
        let mut reasons = existing_reasons;
        let mut details: BTreeSet<(Address, Address, Symbol)> = db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
            .copied()
            .collect();
        for real in invalid_reals {
            if let Some(functions) = owners.get(&real) {
                for function in functions {
                    reasons.insert((*function, real, "unsupported-addr32-address"));
                    details.insert((*function, real, "addr32-lowering-incomplete"));
                }
            }
        }
        db.rel_set(
            "unsupported_stack_address",
            reasons.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "unsupported_address_detail",
            details.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
    }
}

// The complete fused witness depends on register/dataflow facts, so negating
// it inside the generic float-load rules would put call/parameter aggregates
// in a recursive stratum.  Derive both positive alternatives, then choose the
// stack-aware three-node form atomically after the fixed point.
fn indexed_real_reaching_origins(
    db: &DecompileDB,
) -> BTreeMap<(Node, Mreg), BTreeSet<Node>> {
    let abi_liveins: BTreeSet<(Address, Mreg, Address)> = db
        .rel_iter::<(Address, Mreg, Address)>("abi_livein_reaches_use")
        .copied()
        .collect();
    let mut origins: BTreeMap<(Node, Mreg), BTreeSet<Node>> = BTreeMap::new();
    for (definition, mreg, usage) in
        db.rel_iter::<(Address, Mreg, Address)>("reg_def_used")
    {
        // ABI pseudo-defs and a site's loop-carried self edge are two-address
        // inputs, not independent real producers.  Do not infer ABI status
        // from address equality: function-start can also be a real def.
        if *definition == *usage
            || abi_liveins.contains(&(*definition, *mreg, *usage))
        {
            continue;
        }
        origins
            .entry((*usage, *mreg))
            .or_default()
            .insert(*definition);
    }
    origins
}

fn ambiguous_indexed_reaching_operands(db: &DecompileDB) -> BTreeSet<(Node, Mreg)> {
    let origins = indexed_real_reaching_origins(db);
    let dominating: BTreeSet<(Node, Node)> = db
        .rel_iter::<(Address, Node, Node)>("reg_def_dominates_use")
        .map(|(_, definition, usage)| (*definition, *usage))
        .collect();
    origins
        .into_iter()
        .filter_map(|(operand, definitions)| {
            let ambiguous = definitions.len() > 1
                || definitions.iter().next().is_some_and(|definition| {
                    !dominating.contains(&(*definition, operand.0))
                });
            ambiguous.then_some(operand)
        })
        .collect()
}

fn final_xtl_canonical_map(db: &DecompileDB) -> BTreeMap<RTLReg, RTLReg> {
    // xtl_canonical is the regular projection of a decreasing Dual lattice.
    // Ascent retains every value emitted while that lattice converges, so a
    // relation consumer can otherwise observe historical representatives as
    // competing values.  Collapse each original RTL id to its own final
    // (minimum) representative.  Do not collapse by (node, mreg): that would
    // erase the original value-web identity.  CFG-join ambiguity is guarded
    // separately by retained reaching-definition provenance below.
    let mut final_canonical: BTreeMap<RTLReg, RTLReg> = BTreeMap::new();
    for (id, canonical) in db.rel_iter::<(RTLReg, RTLReg)>("xtl_canonical") {
        final_canonical
            .entry(*id)
            .and_modify(|current| *current = (*current).min(*canonical))
            .or_insert(*canonical);
    }
    final_canonical
}

fn canonicalize_indexed_stack_rtl_values(db: &mut DecompileDB) {
    let final_canonical = final_xtl_canonical_map(db);
    let rewrite = |value: RTLReg| final_canonical.get(&value).copied().unwrap_or(value);

    let mut reaching: Vec<(Node, Mreg, RTLReg)> = db
        .rel_iter::<(Node, Mreg, RTLReg)>("reaching_use_rtl")
        .map(|(node, mreg, value)| (*node, *mreg, rewrite(*value)))
        .collect();
    reaching.sort_unstable();
    reaching.dedup();
    db.rel_set(
        "reaching_use_rtl",
        reaching.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let mut structural_sites: BTreeSet<Node> = BTreeSet::new();
    for relation in [
        "sp_indexed_load",
        "sp_indexed_store",
        "bp_indexed_load",
        "bp_indexed_store",
    ] {
        structural_sites.extend(
            db.rel_iter::<(Node,)>(relation)
                .map(|(node,)| *node),
        );
    }
    structural_sites.extend(
        db.rel_iter::<(Node, Operation, MemoryChunk, i64, i64, Mreg, Mreg)>(
            "sp_indexed_fused_load",
        )
        .map(|(node, ..)| *node),
    );
    if structural_sites.is_empty() {
        return;
    }

    let rewrite_args = |args: &Arc<Vec<RTLReg>>| {
        Arc::new(args.iter().map(|value| rewrite(*value)).collect::<Vec<_>>())
    };
    let mut candidates: Vec<(Node, RTLInst)> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .map(|(node, inst)| {
            let rewritten = if structural_sites.contains(&(*node & !SYNTHETIC_NODE_MASK)) {
                match inst {
                    RTLInst::Iop(op, args, destination) => {
                        RTLInst::Iop(op.clone(), rewrite_args(args), rewrite(*destination))
                    }
                    RTLInst::Iload(chunk, addressing, args, destination) => RTLInst::Iload(
                        *chunk,
                        addressing.clone(),
                        rewrite_args(args),
                        rewrite(*destination),
                    ),
                    RTLInst::Istore(chunk, addressing, args, source) => RTLInst::Istore(
                        *chunk,
                        addressing.clone(),
                        rewrite_args(args),
                        rewrite(*source),
                    ),
                    _ => inst.clone(),
                }
            } else {
                inst.clone()
            };
            (*node, rewritten)
        })
        .collect();
    candidates.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    candidates.dedup();
    db.rel_set(
        "rtl_inst_candidate",
        candidates.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

fn select_sp_indexed_fused_lowerings(db: &mut DecompileDB) {
    let structural_rows: Vec<(Node, Mreg, Mreg)> = db
        .rel_iter::<(Node, Operation, MemoryChunk, i64, i64, Mreg, Mreg)>(
            "sp_indexed_fused_load",
        )
        .map(|(node, _, _, _, _, index, destination)| (*node, *index, *destination))
        .collect();
    let structural_sites: BTreeSet<Node> = structural_rows
        .iter()
        .map(|(node, _, _)| *node)
        .collect();
    if structural_sites.is_empty() {
        return;
    }
    let final_canonical = final_xtl_canonical_map(db);
    let fused_temp_for = |real: Node| {
        let raw = fresh_xtl_reg(real, Mreg::x86("RTEMP"));
        final_canonical.get(&raw).copied().unwrap_or(raw)
    };
    let witnessed_sites: BTreeSet<Node> = db
        .rel_iter::<(Node,)>("sp_indexed_fused_complete")
        .map(|(node,)| *node)
        .collect();
    let ambiguous_operands = ambiguous_indexed_reaching_operands(db);
    // Canonical value webs deliberately alias branch definitions at their
    // join.  Preserve the reaching-definition provenance independently so
    // final-canonical candidate dedup cannot turn a MAY-join into a unique
    // lowering.  ABI live-ins and self-loop edges were removed above because
    // those are inputs to the same two-address web, not competing producers.
    let ambiguous_reaching_sites: BTreeSet<Node> = structural_rows
        .iter()
        .filter_map(|(node, index, destination)| {
            [*index, *destination]
                .into_iter()
                .any(|mreg| ambiguous_operands.contains(&(*node, mreg)))
                .then_some(*node)
        })
        .collect();
    let decoded_next: BTreeSet<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("next")
        .copied()
        .collect();

    #[derive(Clone, Copy, Default)]
    struct FusedShapeCounts {
        expected: [usize; 3],
        retained: [usize; 3],
        unexpected_nodes: usize,
    }

    // Count the candidate set that the selector will actually retain, not
    // merely the three expected shapes.  Known generic alternatives can add a
    // root Iload, its exact decoded-fallthrough Ibranch, a synth1 Iop, or a
    // non-RTEMP synth1 Iload; those are removed below only after the fused
    // proof succeeds.  Every other extra candidate remains a veto so the
    // supposedly atomic chain cannot become ambiguous.
    let mut shape_counts: BTreeMap<Node, FusedShapeCounts> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate") {
        let real = *node & !SYNTHETIC_NODE_MASK;
        if !structural_sites.contains(&real) {
            continue;
        }
        let counts = shape_counts.entry(real).or_default();
        let fused_temp = fused_temp_for(real);
        let (slot, expected, retained) = if *node == real {
            let exact_fallthrough_branch = matches!(
                inst,
                RTLInst::Ibranch(Either::Right(target))
                    if decoded_next.contains(&(real, *target))
            );
            (
                0,
                matches!(
                    inst,
                    RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), _, _)
                ),
                !matches!(inst, RTLInst::Iload(..)) && !exact_fallthrough_branch,
            )
        } else if *node == (real | (1u64 << 62)) {
            (
                1,
                matches!(inst, RTLInst::Iload(_, _, _, destination)
                    if *destination == fused_temp),
                match inst {
                    RTLInst::Iop(..) => false,
                    RTLInst::Iload(_, _, _, destination) => *destination == fused_temp,
                    _ => true,
                },
            )
        } else if *node == (real | (1u64 << 63)) {
            (
                2,
                matches!(inst, RTLInst::Iop(_, args, destination)
                    if args.len() == 2
                        && args[0] == *destination
                        && args[1] == fused_temp),
                true,
            )
        } else {
            counts.unexpected_nodes += 1;
            continue;
        };
        if expected {
            counts.expected[slot] += 1;
        }
        if retained {
            counts.retained[slot] += 1;
        }
    }
    let sites: BTreeSet<Node> = structural_sites
        .iter()
        .filter(|site| {
            let counts = shape_counts.get(site).copied().unwrap_or_default();
            witnessed_sites.contains(site)
                && !ambiguous_reaching_sites.contains(site)
                && counts.expected == [1, 1, 1]
                && counts.retained == [1, 1, 1]
                && counts.unexpected_nodes == 0
        })
        .copied()
        .collect();
    let rejected: BTreeSet<Node> = structural_sites.difference(&sites).copied().collect();

    db.rel_set(
        "sp_indexed_fused_complete",
        sites
            .iter()
            .copied()
            .map(|node| (node,))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    if !rejected.is_empty() {
        let mut reasons: BTreeSet<(Address, Address, Symbol)> = db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .copied()
            .collect();
        for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
            if rejected.contains(node) {
                reasons.insert((*function, *node, "unsupported-stack-address"));
            }
        }
        db.rel_set(
            "unsupported_stack_address",
            reasons.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
    }
    if sites.is_empty() {
        return;
    }

    let mut candidates: Vec<(Node, RTLInst)> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, inst)| {
            let real = *node & !SYNTHETIC_NODE_MASK;
            if !sites.contains(&real) {
                return true;
            }
            if *node == real {
                let exact_fallthrough_branch = matches!(
                    inst,
                    RTLInst::Ibranch(Either::Right(target))
                        if decoded_next.contains(&(real, *target))
                );
                return !matches!(inst, RTLInst::Iload(..)) && !exact_fallthrough_branch;
            }
            if *node == (real | (1u64 << 62)) {
                let fused_temp = fused_temp_for(real);
                return match inst {
                    RTLInst::Iop(..) => false,
                    RTLInst::Iload(_, _, _, destination) => *destination == fused_temp,
                    _ => true,
                };
            }
            true
        })
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    candidates.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    candidates.dedup();
    db.rel_set(
        "rtl_inst_candidate",
        candidates.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let mut edges: Vec<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("rtl_succ_candidate")
        .filter(|(source, destination)| {
            let real = *source & !SYNTHETIC_NODE_MASK;
            !sites.contains(&real)
                || *source != (real | (1u64 << 62))
                || *destination == (real | (1u64 << 63))
        })
        .copied()
        .collect();
    edges.sort_unstable();
    edges.dedup();
    db.rel_set(
        "rtl_succ_candidate",
        edges.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

// Synthetic membership feeds later passes, but keeping it out of the main
// fixed point prevents the complete-lowering witness from becoming recursive
// with function/call aggregates through instr_in_function.
fn materialize_sp_indexed_fused_members(db: &mut DecompileDB) {
    let mut rows: Vec<(Node, Address)> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .copied()
        .collect();
    rows.extend(
        db.rel_iter::<(Node, Address)>("sp_indexed_fused_member")
            .copied(),
    );
    rows.sort_unstable();
    rows.dedup();
    db.rel_set(
        "instr_in_function",
        rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

fn classify_indexed_stack_lowerings(db: &mut DecompileDB) {
    // Candidate generation and register dataflow share a positive SCC.  Keep
    // this completeness decision after its fixed point: driving CFG rules from
    // a Datalog witness would pull the aggregated call/signature relations
    // into that SCC.  A valid ordinary indexed expansion has exactly one
    // candidate at its synthetic memory node, of the expected load/store kind.
    let mut candidate_counts: BTreeMap<Node, (usize, usize, usize)> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate") {
        let counts = candidate_counts.entry(*node).or_default();
        counts.0 += 1;
        match inst {
            RTLInst::Iload(..) => counts.1 += 1,
            RTLInst::Istore(..) => counts.2 += 1,
            _ => {}
        }
    }

    // A load whose destination is also an address operand cannot represent a
    // loop-carried self definition with one ordinary two-node lowering: the
    // same RTL value would have to be both the pre-load index and post-load
    // destination.  A literal-entry ABI collision and a loop-carried collision
    // both appear as def==use here; neither has a distinct pre-load namespace,
    // so reject both atomically.
    let self_defs: BTreeSet<(Node, Mreg)> = db
        .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
        .filter_map(|(definition, mreg, usage)| {
            (*definition == *usage).then_some((*definition, *mreg))
        })
        .collect();
    let recursive_load_collisions: BTreeSet<Node> = db
        .rel_iter::<(Node, Mreg)>("load_overwrites_base")
        .filter_map(|(node, mreg)| self_defs.contains(&(*node, *mreg)).then_some(*node))
        .collect();

    let ambiguous_operands = ambiguous_indexed_reaching_operands(db);
    let ordinary_sites: BTreeSet<Node> = [
        "sp_indexed_load",
        "sp_indexed_store",
        "bp_indexed_load",
        "bp_indexed_store",
    ]
    .into_iter()
    .flat_map(|relation| db.rel_iter::<(Node,)>(relation).map(|(node,)| *node))
    .collect();
    let mut ambiguous_reaching_sites = BTreeSet::new();
    for (node, inst) in db.rel_iter::<(Node, LTLInst)>("ltl_inst") {
        if !ordinary_sites.contains(node) {
            continue;
        }
        let operands: Vec<Mreg> = match inst {
            LTLInst::Lload(_, _, args, _) if args.len() > 1 => vec![args[1]],
            LTLInst::Lstore(_, _, args, source) if args.len() > 1 => {
                vec![args[1], *source]
            }
            _ => Vec::new(),
        };
        if operands
            .into_iter()
            .any(|mreg| ambiguous_operands.contains(&(*node, mreg)))
        {
            ambiguous_reaching_sites.insert(*node);
        }
    }

    let mut incomplete = BTreeSet::new();
    let accepted_fused: BTreeSet<Node> = db
        .rel_iter::<(Node,)>("sp_indexed_fused_complete")
        .map(|(node,)| *node)
        .collect();
    for (structural_relation, complete_relation, expect_load) in [
        ("sp_indexed_load", "sp_indexed_load_complete", true),
        ("sp_indexed_store", "sp_indexed_store_complete", false),
        ("bp_indexed_load", "bp_indexed_load_complete", true),
        ("bp_indexed_store", "bp_indexed_store_complete", false),
    ] {
        let structural: BTreeSet<Node> = db
            .rel_iter::<(Node,)>(structural_relation)
            .map(|(node,)| *node)
            .collect();
        let mut complete = BTreeSet::new();
        for node in structural {
            // The fused selector already proved and owns this RMW's complete
            // three-node chain.  Its overlapping ordinary Lload relation is
            // neither an independent success nor an ordinary failure.
            if accepted_fused.contains(&node) {
                continue;
            }
            let synth = node | (1u64 << 62);
            let (total, loads, stores) = candidate_counts
                .get(&synth)
                .copied()
                .unwrap_or_default();
            let valid = !(expect_load && recursive_load_collisions.contains(&node))
                && !ambiguous_reaching_sites.contains(&node)
                && total == 1
                && if expect_load {
                    loads == 1 && stores == 0
                } else {
                    stores == 1 && loads == 0
                };
            if valid {
                complete.insert((node,));
            } else {
                incomplete.insert(node);
            }
        }
        db.rel_set(
            complete_relation,
            complete.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
    }
    if incomplete.is_empty() {
        return;
    }

    let mut diagnostics: BTreeSet<(Address, Address, Symbol)> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .copied()
        .collect();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        if incomplete.contains(node) {
            diagnostics.insert((*function, *node, "unsupported-stack-address"));
        }
    }
    db.rel_set(
        "unsupported_stack_address",
        diagnostics
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    let mut details: BTreeSet<(Address, Address, Symbol)> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
        .copied()
        .collect();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        if incomplete.contains(node) {
            details.insert((*function, *node, "indexed-stack-chain-incomplete"));
        }
    }
    db.rel_set(
        "unsupported_address_detail",
        details.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

fn unsupported_memory_site_for_node(
    node: Node,
    unsupported_sites: &BTreeSet<Address>,
) -> Option<Address> {
    if unsupported_sites.contains(&node) {
        return Some(node);
    }
    let real = node & !SYNTHETIC_NODE_MASK;
    (real != node && unsupported_sites.contains(&real)).then_some(real)
}

fn is_unsupported_synthetic_node(
    node: Node,
    unsupported_sites: &BTreeSet<Address>,
) -> bool {
    let real = node & !SYNTHETIC_NODE_MASK;
    node != real && unsupported_sites.contains(&real)
}

// `unsupported_stack_address` is the final, structured safety boundary shared
// by ordinary stack addressing and addr32.  Individual Asm rules deliberately
// remain useful for proved frame and raw-pointer cases, but several legacy
// producers bypass those common helpers (plain MOV BP fallbacks, immediate
// stores, LEA, and fused load/op/store lowering).  Reject the whole decoded
// instruction atomically: a synthetic arithmetic/store tail is no safer than
// the rejected load feeding it.  Collapse it to one real-node Inop and bridge
// the chain's external successors so rejected candidates, stack provenance,
// memberships, or CFG rewrites cannot escape into later passes.
fn suppress_unsupported_address_candidates(db: &mut DecompileDB) {
    let unsupported_site_owners: BTreeSet<(Address, Address)> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .map(|(function, access, _)| (*function, *access))
        .collect();
    let unsupported_sites: BTreeSet<Address> = unsupported_site_owners
        .iter()
        .map(|(_, access)| *access)
        .collect();
    if unsupported_sites.is_empty() {
        return;
    }

    // Diagnostics may outlive a function-membership row while passes are
    // being inspected independently.  Such rows still veto unsafe lowering,
    // but must not mint a phantom replacement instruction or CFG node.
    let old_members: Vec<(Node, Address)> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .copied()
        .collect();
    let mut node_owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for &(node, function) in &old_members {
        node_owners.entry(node).or_default().insert(function);
    }
    let shares_owner = |left: Node, right: Node| {
        node_owners
            .get(&left)
            .zip(node_owners.get(&right))
            .is_some_and(|(left_owners, right_owners)| !left_owners.is_disjoint(right_owners))
    };

    // Synthetic membership is not sufficient ownership for minting a real
    // replacement node.  In a partially inspected DB it may be the only stale
    // remnant of a rejected chain; require the real instruction's own row.
    let owned_sites: BTreeSet<Address> = old_members
        .iter()
        .filter_map(|(node, _)| unsupported_sites.contains(node).then_some(*node))
        .collect();

    let old_candidates: Vec<(Node, RTLInst)> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    let old_candidate_nodes: BTreeSet<Node> =
        old_candidates.iter().map(|(node, _)| *node).collect();

    // A rejected memory CMP owns a temporary consumed at the following JCC
    // or SETcc node.  Root-only filtering would leave that consumer reading a
    // now-undefined value.  Authenticate each dependency against the exact
    // structured diagnostic owner and both nodes' retained membership, then
    // remove only candidates which actually use that exact temporary.
    let mut rejected_cmp_dependencies: BTreeMap<Node, BTreeSet<(RTLReg, Address)>> =
        BTreeMap::new();
    let mut rejected_cmp_consumers: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    let mut rejected_cmp_consumers_by_root: BTreeMap<Address, BTreeSet<(Node, Address)>> =
        BTreeMap::new();
    for (root, consumer, temp, owner) in
        db.rel_iter::<(Address, Address, RTLReg, Address)>("cmp_memory_temp_consumer")
    {
        if unsupported_site_owners.contains(&(*owner, *root))
            && node_owners
                .get(root)
                .is_some_and(|owners| owners.contains(owner))
            && node_owners
                .get(consumer)
                .is_some_and(|owners| owners.contains(owner))
        {
            rejected_cmp_dependencies
                .entry(*consumer)
                .or_default()
                .insert((*temp, *owner));
            rejected_cmp_consumers
                .entry(*consumer)
                .or_default()
                .insert(*owner);
            rejected_cmp_consumers_by_root
                .entry(*root)
                .or_default()
                .insert((*consumer, *owner));
        }
    }

    let mut removed_cmp_iconds: Vec<(Node, Node, Node, BTreeSet<Address>)> = Vec::new();
    let mut candidates: Vec<(Node, RTLInst)> = old_candidates
        .iter()
        .filter_map(|(node, inst)| {
            if unsupported_memory_site_for_node(*node, &unsupported_sites).is_some() {
                return None;
            }
            let mut uses = BTreeSet::new();
            collect_rtl_uses(inst, &mut uses);
            let rejecting_owners: BTreeSet<Address> = rejected_cmp_dependencies
                .get(node)
                .into_iter()
                .flat_map(|dependencies| dependencies.iter())
                .filter_map(|(temp, owner)| uses.contains(temp).then_some(*owner))
                .collect();
            if !rejecting_owners.is_empty() {
                rejected_cmp_consumers
                    .entry(*node)
                    .or_default()
                    .extend(rejecting_owners.iter().copied());
                if let RTLInst::Icond(
                    _,
                    _,
                    Either::Right(true_target),
                    Either::Right(false_target),
                ) = inst
                {
                    removed_cmp_iconds.push((
                        *node,
                        *true_target,
                        *false_target,
                        rejecting_owners,
                    ));
                }
                None
            } else {
                Some((*node, inst.clone()))
            }
        })
        .collect();
    candidates.extend(
        owned_sites
            .iter()
            .copied()
            .map(|real| (real, RTLInst::Inop)),
    );
    // A JCC/SETcc represented solely by the rejected CMP-dependent candidate
    // still needs a real CFG anchor.  Insert it before computing surviving
    // nodes so the root's decoded fallback can reconnect to this consumer.
    let mut inserted_cmp_consumers: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (consumer, rejecting_owners) in rejected_cmp_consumers {
        if node_owners
            .get(&consumer)
            .is_some_and(|owners| !owners.is_disjoint(&rejecting_owners))
            && !candidates.iter().any(|(node, _)| *node == consumer)
        {
            candidates.push((consumer, RTLInst::Inop));
            inserted_cmp_consumers.insert(consumer, rejecting_owners);
        }
    }
    candidates.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    candidates.dedup();
    let surviving_candidate_nodes: BTreeSet<Node> =
        candidates.iter().map(|(node, _)| *node).collect();
    let surviving_explicit_branches: BTreeSet<(Node, Node)> = candidates
        .iter()
        .filter_map(|(node, inst)| match inst {
            RTLInst::Ibranch(Either::Right(target)) => Some((*node, *target)),
            _ => None,
        })
        .collect();
    let mut surviving_control_edges = BTreeSet::new();
    let mut surviving_control_nodes = BTreeSet::new();
    for (node, inst) in &candidates {
        match inst {
            RTLInst::Icond(
                _,
                _,
                Either::Right(true_target),
                Either::Right(false_target),
            ) => {
                surviving_control_nodes.insert(*node);
                surviving_control_edges.insert((*node, *true_target));
                surviving_control_edges.insert((*node, *false_target));
            }
            RTLInst::Ibranch(Either::Right(target)) => {
                surviving_control_nodes.insert(*node);
                surviving_control_edges.insert((*node, *target));
            }
            RTLInst::Ijumptable(_, targets) => {
                surviving_control_nodes.insert(*node);
                surviving_control_edges.extend(targets.iter().map(|target| (*node, *target)));
            }
            _ => {}
        }
    }
    db.rel_set(
        "rtl_inst_candidate",
        candidates.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let old_edges: Vec<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("rtl_succ_candidate")
        .copied()
        .collect();
    let raw_next: BTreeSet<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("rtl_next")
        .copied()
        .collect();
    let raw_control_next: BTreeSet<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("next")
        .copied()
        .collect();

    // Icond edges were materialized before this imperative filter.  Remove a
    // rejected condition's taken edge unless a surviving control candidate at
    // the same node explicitly claims it.  Its false edge is decoded
    // fallthrough and remains valid for an inserted Inop or ordinary survivor;
    // if another control candidate survives, that candidate owns the complete
    // successor set instead.
    let mut stale_cmp_control_edges = BTreeSet::new();
    let mut safe_cmp_fallthrough_edges = BTreeSet::new();
    for (consumer, true_target, false_target, rejecting_owners) in removed_cmp_iconds {
        if !surviving_control_edges.contains(&(consumer, true_target))
            && true_target != false_target
        {
            stale_cmp_control_edges.insert((consumer, true_target));
        }
        if surviving_control_nodes.contains(&consumer) {
            if !surviving_control_edges.contains(&(consumer, false_target)) {
                stale_cmp_control_edges.insert((consumer, false_target));
            }
        } else if node_owners
            .get(&false_target)
            .is_some_and(|owners| !owners.is_disjoint(&rejecting_owners))
            && surviving_candidate_nodes.contains(&false_target)
        {
            safe_cmp_fallthrough_edges.insert((consumer, false_target));
        } else {
            stale_cmp_control_edges.insert((consumer, false_target));
        }
    }
    // A SETcc may fail to materialize its dependent Ocmp candidate at all.
    // Provenance still inserts an Inop anchor above; give only that anchor its
    // exact decoded fallthrough, authenticated against the rejecting owner.
    for (consumer, rejecting_owners) in &inserted_cmp_consumers {
        for (_, destination) in raw_control_next
            .iter()
            .filter(|(source, _)| source == consumer)
        {
            if surviving_candidate_nodes.contains(destination)
                && node_owners
                    .get(destination)
                    .is_some_and(|owners| !owners.is_disjoint(rejecting_owners))
                && !is_unsupported_synthetic_node(*destination, &unsupported_sites)
                && !(unsupported_sites.contains(destination)
                    && !owned_sites.contains(destination))
            {
                safe_cmp_fallthrough_edges.insert((*consumer, *destination));
            }
        }
    }

    // Semantic LTL successors are authoritative, including deliberately
    // interprocedural call/return edges.  Only retain destinations which will
    // still be real CFG nodes after this rejection pass.
    let mut replacement_raw_exits: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for &(source, destination) in &raw_next {
        if !owned_sites.contains(&source)
            || !surviving_candidate_nodes.contains(&destination)
            || !node_owners.contains_key(&destination)
            || is_unsupported_synthetic_node(destination, &unsupported_sites)
            || (unsupported_sites.contains(&destination)
                && !owned_sites.contains(&destination))
        {
            continue;
        }
        // A memory CMP's semantic LTL successor can skip its non-LTL SETcc
        // node.  Once that SETcc candidate is rejected and replaced with an
        // Inop, prefer the exact decoded root->consumer path; otherwise the
        // replacement anchor and its fallthrough are unreachable.
        if rejected_cmp_consumers_by_root
            .get(&source)
            .is_some_and(|consumers| {
                !consumers.iter().any(|(consumer, owner)| {
                    *consumer == destination
                        && node_owners
                            .get(&destination)
                            .is_some_and(|owners| owners.contains(owner))
                })
            })
        {
            continue;
        }
        replacement_raw_exits
            .entry(source)
            .or_default()
            .insert(destination);
    }

    // A generic bridge may name both the one-synth and two-synth forms, and an
    // intermediate node may have a speculative bypass edge.  Only a real
    // candidate at the terminal synthetic node can supply a chain exit.
    let internal_synthetic_sources: BTreeSet<Node> = old_edges
        .iter()
        .filter_map(|(source, destination)| {
            let source_real = *source & !SYNTHETIC_NODE_MASK;
            let destination_real = *destination & !SYNTHETIC_NODE_MASK;
            (source_real == destination_real
                && is_unsupported_synthetic_node(*source, &unsupported_sites)
                && is_unsupported_synthetic_node(*destination, &unsupported_sites))
            .then_some(*source)
        })
        .collect();

    // Capture only live, same-function synthetic-tail exits.  A root's
    // semantic rtl_next is authoritative whenever present; using both used to
    // give the replacement Inop multiple successors after an unsafe chain was
    // removed.
    let mut synthetic_tail_exits: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for &(source, destination) in &old_edges {
        if !is_unsupported_synthetic_node(source, &unsupported_sites) {
            continue;
        }
        let real = source & !SYNTHETIC_NODE_MASK;
        if owned_sites.contains(&real)
            && old_candidate_nodes.contains(&source)
            && !internal_synthetic_sources.contains(&source)
            && surviving_candidate_nodes.contains(&destination)
            && shares_owner(real, source)
            && shares_owner(real, destination)
            && unsupported_memory_site_for_node(destination, &unsupported_sites) != Some(real)
            && !is_unsupported_synthetic_node(destination, &unsupported_sites)
        {
            synthetic_tail_exits
                .entry(real)
                .or_default()
                .insert(destination);
        }
    }

    // If an earlier rewrite redirected pred -> real into pred -> synthetic,
    // removing the synthetic endpoint must also un-negate the corresponding
    // raw incoming edge.  This is deliberately exact, not a blanket removal
    // of unrelated control-flow negations at the predecessor.
    let mut restore_incoming: BTreeSet<(Node, Node)> = old_edges
        .iter()
        .filter_map(|(source, destination)| {
            if !is_unsupported_synthetic_node(*destination, &unsupported_sites) {
                return None;
            }
            let real = *destination & !SYNTHETIC_NODE_MASK;
            if !owned_sites.contains(&real)
                || !surviving_candidate_nodes.contains(source)
                || !node_owners.contains_key(source)
                || is_unsupported_synthetic_node(*source, &unsupported_sites)
                || (unsupported_sites.contains(source) && !owned_sites.contains(source))
            {
                return None;
            }
            let semantic_edge = raw_next.contains(&(*source, real));
            let same_function_adjacency = raw_control_next.contains(&(*source, real))
                && shares_owner(*source, real);
            (semantic_edge || same_function_adjacency).then_some((*source, real))
        })
        .collect();
    for &(source, real) in &surviving_explicit_branches {
        if !owned_sites.contains(&real)
            || !surviving_candidate_nodes.contains(&source)
            || !raw_control_next.contains(&(source, real))
            || !shares_owner(source, real)
        {
            continue;
        }
        let has_other_surviving_exit = old_edges.iter().any(|(edge_source, destination)| {
            *edge_source == source
                && *destination != real
                && !is_unsupported_synthetic_node(*edge_source, &unsupported_sites)
                && !is_unsupported_synthetic_node(*destination, &unsupported_sites)
                && !(unsupported_sites.contains(destination)
                    && !owned_sites.contains(destination))
                && surviving_candidate_nodes.contains(destination)
        });
        if !has_other_surviving_exit {
            restore_incoming.insert((source, real));
        }
    }

    // Instructions lowered only through a synthetic chain have no rtl_next:
    // that relation intentionally contains LTL-to-LTL edges only.  If the
    // rejected chain also failed to expose a synthetic tail exit, walk the
    // same-function decoded byte adjacency to the nearest surviving RTL
    // candidate and bridge the replacement Inop to it.  This is not a control
    // flow relation, so it must never walk into an adjacent function; cycles
    // are nevertheless bounded by `seen` for hand-built test databases.
    let mut control_fallback_exits: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for &real in &owned_sites {
        if replacement_raw_exits
            .get(&real)
            .is_some_and(|exits| !exits.is_empty())
            || synthetic_tail_exits
                .get(&real)
                .is_some_and(|exits| !exits.is_empty())
        {
            continue;
        }
        let mut pending: Vec<Node> = raw_control_next
            .iter()
            .filter_map(|(source, destination)| (*source == real).then_some(*destination))
            .collect();
        let mut seen = BTreeSet::new();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) || is_unsupported_synthetic_node(node, &unsupported_sites) {
                continue;
            }
            if !shares_owner(real, node) {
                continue;
            }
            if surviving_candidate_nodes.contains(&node) {
                control_fallback_exits.entry(real).or_default().insert(node);
                continue;
            }
            pending.extend(
                raw_control_next
                    .iter()
                    .filter_map(|(source, destination)| (*source == node).then_some(*destination)),
            );
        }
    }

    // Finalize negations first.  The Ascent fixed point has already run, so
    // mutating this relation alone would not re-fire rtl_next -> successor.
    let mut negated: BTreeSet<(Node, Node)> = db
        .rel_iter::<(Node, Node)>("rtl_edge_negated")
        .filter(|(source, destination)| {
            !is_unsupported_synthetic_node(*source, &unsupported_sites)
                && !is_unsupported_synthetic_node(*destination, &unsupported_sites)
        })
        .copied()
        .collect();
    for (&source, destinations) in &replacement_raw_exits {
        for &destination in destinations {
            negated.remove(&(source, destination));
        }
    }
    for edge in &restore_incoming {
        negated.remove(edge);
    }
    for edge in &safe_cmp_fallthrough_edges {
        negated.remove(edge);
    }
    for (&real, exits) in &synthetic_tail_exits {
        if replacement_raw_exits
            .get(&real)
            .is_some_and(|raw_exits| !raw_exits.is_empty())
        {
            continue;
        }
        for &destination in exits {
            negated.remove(&(real, destination));
        }
    }
    for (&real, exits) in &control_fallback_exits {
        for &destination in exits {
            negated.remove(&(real, destination));
        }
    }
    db.rel_set(
        "rtl_edge_negated",
        negated
            .iter()
            .copied()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    // Preserve non-rejected derived edges, then explicitly replay every safe
    // raw edge against the final negation set.  Outgoing edges from a rejected
    // real root are rebuilt below rather than inherited from its old chain.
    let mut edges: BTreeSet<(Node, Node)> = old_edges
        .into_iter()
        .filter(|(source, destination)| {
            if stale_cmp_control_edges.contains(&(*source, *destination))
                || unsupported_sites.contains(source)
                || is_unsupported_synthetic_node(*source, &unsupported_sites)
                || is_unsupported_synthetic_node(*destination, &unsupported_sites)
                || (unsupported_sites.contains(destination)
                    && !owned_sites.contains(destination))
            {
                return false;
            }
            true
        })
        .collect();
    for &(source, destination) in &raw_next {
        if stale_cmp_control_edges.contains(&(source, destination))
            || is_unsupported_synthetic_node(source, &unsupported_sites)
            || is_unsupported_synthetic_node(destination, &unsupported_sites)
            || (unsupported_sites.contains(&source) && !owned_sites.contains(&source))
            || (unsupported_sites.contains(&destination)
                && !owned_sites.contains(&destination))
            || (owned_sites.contains(&source)
                && !replacement_raw_exits
                    .get(&source)
                    .is_some_and(|exits| exits.contains(&destination)))
            || negated.contains(&(source, destination))
        {
            continue;
        }
        edges.insert((source, destination));
    }
    edges.extend(
        restore_incoming
            .iter()
            .copied()
            .filter(|edge| !stale_cmp_control_edges.contains(edge)),
    );
    edges.extend(safe_cmp_fallthrough_edges);
    for (&real, exits) in &synthetic_tail_exits {
        if replacement_raw_exits
            .get(&real)
            .is_some_and(|raw_exits| !raw_exits.is_empty())
        {
            continue;
        }
        for &destination in exits {
            edges.insert((real, destination));
        }
    }
    for (real, exits) in control_fallback_exits {
        for destination in exits {
            edges.insert((real, destination));
        }
    }
    db.rel_set(
        "rtl_succ_candidate",
        edges.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let members: ascent::boxcar::Vec<(Node, Address)> = old_members
        .into_iter()
        .filter(|(node, _)| {
            unsupported_memory_site_for_node(*node, &unsupported_sites)
                .map_or(true, |real| *node == real)
        })
        .collect();
    db.rel_set("instr_in_function", members);

    let fused_members: ascent::boxcar::Vec<(Node, Address)> = db
        .rel_iter::<(Node, Address)>("sp_indexed_fused_member")
        .filter(|(node, _)| unsupported_memory_site_for_node(*node, &unsupported_sites).is_none())
        .copied()
        .collect();
    db.rel_set("sp_indexed_fused_member", fused_members);

    let synth_only: ascent::boxcar::Vec<(Node,)> = db
        .rel_iter::<(Node,)>("synth_only_addr")
        .filter(|(node,)| unsupported_memory_site_for_node(*node, &unsupported_sites).is_none())
        .copied()
        .collect();
    db.rel_set("synth_only_addr", synth_only);

    let mut indirect_loads: Vec<(Node, RTLReg, MemoryChunk, Addressing, Args)> = db
        .rel_iter::<(Node, RTLReg, MemoryChunk, Addressing, Args)>("call_through_memory_load")
        .filter(|(node, ..)| unsupported_memory_site_for_node(*node, &unsupported_sites).is_none())
        .map(|(node, target, chunk, addressing, args)| {
            (*node, *target, *chunk, addressing.clone(), args.clone())
        })
        .collect();
    indirect_loads.sort_by_cached_key(|row| format!("{row:?}"));
    indirect_loads.dedup();
    db.rel_set(
        "call_through_memory_load",
        indirect_loads
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    let ptr_rows: ascent::boxcar::Vec<(Node, RTLReg)> = db
        .rel_iter::<(Node, RTLReg)>("op_produces_ptr")
        .filter(|(node, _)| unsupported_memory_site_for_node(*node, &unsupported_sites).is_none())
        .copied()
        .collect();
    db.rel_set("op_produces_ptr", ptr_rows);

    let escaped_origins: ascent::boxcar::Vec<(Address, Node, i64, RTLReg)> = db
        .rel_iter::<(Address, Node, i64, RTLReg)>("slot_escaped_origin")
        .filter(|(_, origin, _, _)| {
            unsupported_memory_site_for_node(*origin, &unsupported_sites).is_none()
        })
        .copied()
        .collect();
    db.rel_set("slot_escaped_origin", escaped_origins);

    for relation in ["stack_xtl", "stack_var"] {
        let rows: ascent::boxcar::Vec<(Address, Node, i64, RTLReg)> = db
            .rel_iter::<(Address, Node, i64, RTLReg)>(relation)
            .filter(|(_, node, _, _)| {
                unsupported_memory_site_for_node(*node, &unsupported_sites).is_none()
            })
            .copied()
            .collect();
        db.rel_set(relation, rows);
    }

    let normalized_bases: ascent::boxcar::Vec<(Address, Node, RTLReg, i64)> = db
        .rel_iter::<(Address, Node, RTLReg, i64)>("normalized_stack_lea_base")
        .filter(|(_, node, _, _)| {
            unsupported_memory_site_for_node(*node, &unsupported_sites).is_none()
        })
        .copied()
        .collect();
    db.rel_set("normalized_stack_lea_base", normalized_bases);
}

// Materialize mutable Win64 home cells after RTLPassProgram reaches its fixed
// point.  Keeping this selector imperative is intentional: a negated guard on
// the generic RTL rules would make raw register escape facts recursive with
// rtl_inst_candidate and the called-address aggregates.  All public relations
// consumed by later passes are repaired together here, so the selector does
// not leave the pre-rewrite stack/type/deadness view behind.
fn select_canonical_unsafe_home_rewrites(db: &mut DecompileDB) {
    type Cell = (Address, usize);

    let seeded_storage: BTreeMap<Cell, RTLReg> = db
        .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
        .map(|(func, pos, slot)| ((*func, *pos), *slot))
        .collect();
    let vetoed: BTreeSet<Cell> = db
        .rel_iter::<(Address, usize)>("win64_home_canonical_veto")
        .copied()
        .collect();
    let unsafe_accesses: BTreeSet<(Node, Address, Mreg, i64, usize, i64)> = db
        .rel_iter::<(Node, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
        .copied()
        .collect();
    let spill_nodes: BTreeSet<(Address, usize, Node)> = db
        .rel_iter::<(Node, Address, Mreg, usize)>("win64_home_spill_candidate")
        .map(|(node, func, _, pos)| (*func, *pos, *node))
        .collect();

    let mut overlap_nodes: BTreeMap<Cell, BTreeSet<Node>> = BTreeMap::new();
    for &(node, func, pos) in db.rel_iter::<(Node, Address, usize)>("win64_home_overlap") {
        if seeded_storage.contains_key(&(func, pos)) {
            overlap_nodes.entry((func, pos)).or_default().insert(node);
        }
    }

    let mut shapes: BTreeMap<Cell, BTreeMap<Node, BTreeSet<UnsafeHomeShape>>> = BTreeMap::new();
    for &(node, func, base, raw_ofs, pos, entry_ofs, peer, move_class, width) in
        db.rel_iter::<(Node, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_store",
        )
    {
        shapes
            .entry((func, pos))
            .or_default()
            .entry(node)
            .or_default()
            .insert(UnsafeHomeShape::Store {
                base,
                raw_ofs,
                entry_ofs,
                peer,
                move_class,
                width,
            });
    }
    for &(node, func, base, raw_ofs, pos, entry_ofs, peer, move_class, width) in
        db.rel_iter::<(Node, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_load",
        )
    {
        shapes
            .entry((func, pos))
            .or_default()
            .entry(node)
            .or_default()
            .insert(UnsafeHomeShape::Load {
                base,
                raw_ofs,
                entry_ofs,
                peer,
                move_class,
                width,
            });
    }
    for (node, func, base, raw_ofs, pos, entry_ofs, peer, width, op) in db.rel_iter::<(
        Node,
        Address,
        Mreg,
        i64,
        usize,
        i64,
        Mreg,
        usize,
        Operation,
    )>("win64_home_scalar_extend_load")
    {
        shapes
            .entry((*func, *pos))
            .or_default()
            .entry(*node)
            .or_default()
            .insert(UnsafeHomeShape::ExtendLoad {
                base: *base,
                raw_ofs: *raw_ofs,
                entry_ofs: *entry_ofs,
                peer: *peer,
                width: *width,
                op: op.clone(),
            });
    }
    for &(node, func, base, raw_ofs, pos, entry_ofs, peer) in
        db.rel_iter::<(Node, Address, Mreg, i64, usize, i64, Mreg)>("win64_home_scalar_lea")
    {
        shapes
            .entry((func, pos))
            .or_default()
            .entry(node)
            .or_default()
            .insert(UnsafeHomeShape::Lea {
                base,
                raw_ofs,
                entry_ofs,
                peer,
            });
    }
    for (node, func, base, raw_ofs, pos, entry_ofs, peer, width, op) in db.rel_iter::<(
        Node,
        Address,
        Mreg,
        i64,
        usize,
        i64,
        Mreg,
        usize,
        Operation,
    )>("win64_home_scalar_arith_read")
    {
        shapes
            .entry((*func, *pos))
            .or_default()
            .entry(*node)
            .or_default()
            .insert(UnsafeHomeShape::ArithRead {
                base: *base,
                raw_ofs: *raw_ofs,
                entry_ofs: *entry_ofs,
                peer: *peer,
                width: *width,
                op: op.clone(),
            });
    }
    for (node, func, _mem, base, raw_ofs, pos, entry_ofs, width) in db.rel_iter::<(
        Node,
        Address,
        Symbol,
        Mreg,
        i64,
        usize,
        i64,
        usize,
    )>("win64_home_scalar_cmp_read")
    {
        shapes
            .entry((*func, *pos))
            .or_default()
            .entry(*node)
            .or_default()
            .insert(UnsafeHomeShape::Compare {
                base: *base,
                raw_ofs: *raw_ofs,
                entry_ofs: *entry_ofs,
                width: *width,
            });
    }

    // RTL candidates and rewrite nodes are node-global, while the home facts
    // above are function-scoped.  Shared/.cold ownership or two home-cell
    // interpretations at one node therefore cannot be resolved by blindly
    // inserting into a BTreeMap<Node, _>: one cell would overwrite another.
    let mut node_functions: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, func) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        node_functions.entry(*node).or_default().insert(*func);
    }
    let mut shape_cells_by_node: BTreeMap<Node, BTreeSet<Cell>> = BTreeMap::new();
    for (&cell, by_node) in &shapes {
        for &node in by_node.keys() {
            shape_cells_by_node.entry(node).or_default().insert(cell);
        }
    }
    let colliding_shape_nodes: BTreeSet<Node> = shape_cells_by_node
        .into_iter()
        .filter_map(|(node, cells)| (cells.len() != 1).then_some(node))
        .collect();

    // `xtl_canonical` is a decreasing lattice, so relations retain candidates
    // built from historical representatives.  The reaching-use relation was
    // already collapsed to its final representative above, but home-cell
    // candidates are deliberately outside the indexed-stack site set and can
    // still differ only by a stale loop value.  Normalize this selector's
    // private candidate snapshot before proving uniqueness; conditions and
    // control-flow targets are copied unchanged.
    let final_canonical = final_xtl_canonical_map(db);
    let rewrite = |value: RTLReg| final_canonical.get(&value).copied().unwrap_or(value);
    let rewrite_args = |args: &Arc<Vec<RTLReg>>| {
        Arc::new(args.iter().map(|value| rewrite(*value)).collect::<Vec<_>>())
    };
    let mut candidates: Vec<(Node, RTLInst)> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .map(|(node, inst)| {
            let normalized = match inst {
                RTLInst::Iop(op, args, destination) => {
                    RTLInst::Iop(op.clone(), rewrite_args(args), rewrite(*destination))
                }
                RTLInst::Iload(chunk, addressing, args, destination) => RTLInst::Iload(
                    *chunk,
                    addressing.clone(),
                    rewrite_args(args),
                    rewrite(*destination),
                ),
                RTLInst::Istore(chunk, addressing, args, source) => RTLInst::Istore(
                    *chunk,
                    addressing.clone(),
                    rewrite_args(args),
                    rewrite(*source),
                ),
                RTLInst::Icond(condition, args, if_true, if_false) => RTLInst::Icond(
                    condition.clone(),
                    rewrite_args(args),
                    if_true.clone(),
                    if_false.clone(),
                ),
                _ => inst.clone(),
            };
            (*node, normalized)
        })
        .collect();
    candidates.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    candidates.dedup();
    let mut candidates_at: BTreeMap<Node, Vec<&RTLInst>> = BTreeMap::new();
    for (node, inst) in &candidates {
        candidates_at.entry(*node).or_default().push(inst);
    }
    let mut reaching_reads: BTreeMap<(Node, Mreg), BTreeSet<RTLReg>> = BTreeMap::new();
    for (node, mreg, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reaching_use_rtl") {
        reaching_reads
            .entry((*node, *mreg))
            .or_default()
            .insert(*value);
    }
    let setcc_nodes: BTreeSet<Node> = db
        .rel_iter::<(Address, TestCond)>("setcc_testcond")
        .map(|(node, _)| *node)
        .collect();
    let mut adjacent_setcc: BTreeMap<Node, BTreeSet<Node>> = BTreeMap::new();
    for (source, destination) in db.rel_iter::<(Node, Node)>("next") {
        if setcc_nodes.contains(destination) {
            adjacent_setcc
                .entry(*source)
                .or_default()
                .insert(*destination);
        }
    }

    let mut storage = BTreeMap::new();
    let mut rewrites = BTreeMap::new();
    let mut rewritten_setcc_nodes = BTreeSet::new();
    let mut home_addresses = BTreeSet::new();
    let mut home_escaped = BTreeSet::new();
    let mut cell_types = BTreeMap::new();

    for (key @ (func, pos), slot) in seeded_storage {
        if vetoed.contains(&key) {
            continue;
        }
        let Some(cell_shapes) = shapes.get(&key) else {
            continue;
        };
        let access_nodes: BTreeSet<Node> = unsafe_accesses
            .iter()
            .filter_map(|(node, access_func, _, _, access_pos, _)| {
                (*access_func == func && *access_pos == pos).then_some(*node)
            })
            .collect();
        let shape_nodes: BTreeSet<Node> = cell_shapes.keys().copied().collect();
        if access_nodes != shape_nodes
            || overlap_nodes
                .get(&key)
                .is_some_and(|nodes| !nodes.is_subset(&shape_nodes))
            || cell_shapes.values().any(|rows| rows.len() != 1)
        {
            continue;
        }

        let has_storage_reason = cell_shapes.iter().any(|(node, rows)| {
            match rows.iter().next().expect("one checked home shape") {
                UnsafeHomeShape::Store { .. } => !spill_nodes.contains(&(func, pos, *node)),
                UnsafeHomeShape::Lea { .. } => true,
                UnsafeHomeShape::ArithRead { .. } => true,
                UnsafeHomeShape::Compare { .. } => true,
                UnsafeHomeShape::ExtendLoad { .. } => true,
                UnsafeHomeShape::Load {
                    raw_ofs, entry_ofs, ..
                } => raw_ofs != entry_ofs,
            }
        });
        if !has_storage_reason {
            continue;
        }

        let mut cell_rewrites = BTreeMap::new();
        let mut cell_setcc_nodes = BTreeSet::new();
        let mut cell_addresses = BTreeSet::new();
        let mut signature: Option<(usize, usize)> = None;
        let mut complete = true;

        for (&node, rows) in cell_shapes {
            let shape = rows
                .iter()
                .next()
                .expect("one checked home shape")
                .clone();
            let node_candidates = candidates_at.get(&node).map(Vec::as_slice).unwrap_or(&[]);
            match shape {
                UnsafeHomeShape::Store {
                    raw_ofs,
                    peer,
                    move_class,
                    width,
                    ..
                } => {
                    if signature.is_some_and(|seen| seen != (move_class, width)) {
                        complete = false;
                        break;
                    }
                    signature = Some((move_class, width));
                    let candidate_peers: BTreeSet<RTLReg> = node_candidates
                        .iter()
                        .filter_map(|inst| match inst {
                            RTLInst::Iop(Operation::Omove, args, _) if args.len() == 1 => {
                                Some(args[0])
                            }
                            RTLInst::Istore(_, Addressing::Aindexed(ofs), _, source)
                            | RTLInst::Istore(_, Addressing::Ainstack(ofs), _, source)
                                if *ofs == raw_ofs =>
                            {
                                Some(*source)
                            }
                            _ => None,
                        })
                        .collect();
                    // Source registers are reads.  reg_rtl also retains a
                    // node-local placeholder at some copied-base stores, so
                    // prefer the unique value proved to reach this use.  If
                    // reaching evidence is absent or ambiguous, retain the
                    // candidate-derived set and let the existing uniqueness
                    // check conservatively reject the rewrite.
                    let peers = reaching_reads
                        .get(&(node, peer))
                        .filter(|values| values.len() == 1)
                        .unwrap_or(&candidate_peers);
                    if peers.len() != 1 {
                        complete = false;
                        break;
                    }
                    let source = *peers.iter().next().unwrap();
                    cell_rewrites.insert(node, UnsafeHomeRewrite::Store { source, slot });
                }
                UnsafeHomeShape::Load {
                    raw_ofs,
                    move_class,
                    width,
                    ..
                } => {
                    if signature.is_some_and(|seen| seen != (move_class, width)) {
                        complete = false;
                        break;
                    }
                    signature = Some((move_class, width));
                    let peers: BTreeSet<RTLReg> = node_candidates
                        .iter()
                        .filter_map(|inst| match inst {
                            RTLInst::Iop(Operation::Omove, args, destination)
                                if args.len() == 1 =>
                            {
                                Some(*destination)
                            }
                            RTLInst::Iload(_, Addressing::Aindexed(ofs), _, destination)
                            | RTLInst::Iload(_, Addressing::Ainstack(ofs), _, destination)
                                if *ofs == raw_ofs =>
                            {
                                Some(*destination)
                            }
                            _ => None,
                        })
                        .collect();
                    if peers.len() != 1 {
                        complete = false;
                        break;
                    }
                    let destination = *peers.iter().next().unwrap();
                    cell_rewrites.insert(node, UnsafeHomeRewrite::Load { slot, destination });
                }
                UnsafeHomeShape::ExtendLoad { width, op, .. } => {
                    if signature.is_some_and(|seen| seen != (0, width)) {
                        complete = false;
                        break;
                    }
                    signature = Some((0, width));
                    let peers: BTreeSet<RTLReg> = node_candidates
                        .iter()
                        .filter_map(|inst| match inst {
                            RTLInst::Iop(candidate_op, args, destination)
                                if candidate_op == &op && args.as_ref() == &[slot] =>
                            {
                                Some(*destination)
                            }
                            _ => None,
                        })
                        .collect();
                    if peers.len() != 1 {
                        complete = false;
                        break;
                    }
                    let destination = *peers.iter().next().unwrap();
                    cell_rewrites.insert(
                        node,
                        UnsafeHomeRewrite::ExtendLoad {
                            slot,
                            destination,
                            op,
                        },
                    );
                }
                UnsafeHomeShape::Lea {
                    raw_ofs, entry_ofs, ..
                } => {
                    let peers: BTreeSet<RTLReg> = node_candidates
                        .iter()
                        .filter_map(|inst| match inst {
                            RTLInst::Iop(
                                Operation::Olea(Addressing::Aindexed(ofs))
                                | Operation::Olea(Addressing::Ainstack(ofs))
                                | Operation::Oleal(Addressing::Aindexed(ofs))
                                | Operation::Oleal(Addressing::Ainstack(ofs)),
                                _,
                                destination,
                            ) if *ofs == raw_ofs => Some(*destination),
                            _ => None,
                        })
                        .collect();
                    if peers.len() != 1 {
                        complete = false;
                        break;
                    }
                    let destination = *peers.iter().next().unwrap();
                    cell_rewrites.insert(
                        node,
                        UnsafeHomeRewrite::Lea {
                            entry_ofs,
                            destination,
                        },
                    );
                    cell_addresses.insert((node, slot));
                }
                UnsafeHomeShape::ArithRead {
                    width, op, peer, ..
                } => {
                    if signature.is_some_and(|seen| seen != (0, width)) {
                        complete = false;
                        break;
                    }
                    signature = Some((0, width));
                    let synthetic = node | (1u64 << 62);
                    let candidate_peers: BTreeSet<RTLReg> = candidates_at
                        .get(&synthetic)
                        .into_iter()
                        .flat_map(|rows| rows.iter())
                        .filter_map(|inst| match inst {
                            RTLInst::Iop(candidate_op, args, destination)
                                if candidate_op == &op
                                    && args.len() == 2
                                    && args[0] == *destination =>
                            {
                                Some(*destination)
                            }
                            _ => None,
                        })
                        .collect();
                    let peers = reaching_reads
                        .get(&(node, peer))
                        .filter(|values| values.len() == 1)
                        .unwrap_or(&candidate_peers);
                    if peers.len() != 1 {
                        complete = false;
                        break;
                    }
                    let destination = *peers.iter().next().unwrap();
                    cell_rewrites.insert(
                        node,
                        UnsafeHomeRewrite::ArithRead {
                            slot,
                            destination,
                            op,
                        },
                    );
                }
                UnsafeHomeShape::Compare { width, .. } => {
                    if signature.is_some_and(|seen| seen != (0, width)) {
                        complete = false;
                        break;
                    }
                    signature = Some((0, width));
                    let mut candidates: Vec<RTLInst> = node_candidates
                        .iter()
                        .filter_map(|inst| match inst {
                            RTLInst::Icond(_, args, _, _)
                                if args.iter().filter(|arg| **arg == slot).count() == 1 =>
                            {
                                Some((*inst).clone())
                            }
                            RTLInst::Iop(Operation::Ocmp(_), args, _)
                                if args.iter().filter(|arg| **arg == slot).count() == 1 =>
                            {
                                Some((*inst).clone())
                            }
                            _ => None,
                        })
                        .collect();
                    candidates.sort_by_cached_key(|inst| format!("{inst:?}"));
                    candidates.dedup();
                    if candidates.len() != 1 {
                        complete = false;
                        break;
                    }
                    if matches!(
                        candidates.first(),
                        Some(RTLInst::Iop(Operation::Ocmp(_), _, _))
                    ) {
                        let consumers = adjacent_setcc.get(&node);
                        if !consumers.is_some_and(|nodes| nodes.len() == 1) {
                            complete = false;
                            break;
                        }
                        cell_setcc_nodes.extend(consumers.unwrap().iter().copied());
                    }
                    cell_rewrites.insert(
                        node,
                        UnsafeHomeRewrite::Compare {
                            inst: candidates.pop().unwrap(),
                        },
                    );
                }
            }
        }

        let Some((move_class, width)) = signature else {
            continue;
        };
        let Some(primary_type) = home_move_xtype(move_class, width) else {
            continue;
        };
        if !complete || cell_rewrites.len() != cell_shapes.len() {
            continue;
        }
        if cell_rewrites.keys().any(|node| {
            colliding_shape_nodes.contains(node)
                || match node_functions.get(node) {
                    Some(owners) => owners.len() != 1 || !owners.contains(&func),
                    None => true,
                }
        }) {
            continue;
        }
        if cell_setcc_nodes.iter().any(|node| {
            colliding_shape_nodes.contains(node)
                || match node_functions.get(node) {
                    Some(owners) => owners.len() != 1 || !owners.contains(&func),
                    None => true,
                }
        }) {
            continue;
        }

        storage.insert(key, slot);
        cell_types.insert(slot, primary_type);
        if !cell_addresses.is_empty() {
            home_addresses.extend(cell_addresses);
            home_escaped.insert((func, slot));
        }
        rewrites.extend(cell_rewrites);
        rewritten_setcc_nodes.extend(cell_setcc_nodes);
    }

    // The positive relation above keeps the Datalog fixed point stratified;
    // candidate uniqueness is proved only here. If that final proof fails,
    // restore a structured per-site rejection before the public safety filter
    // runs so an ambiguous arithmetic chain can never leak into Clight.
    let mut failed_scalar_sites = BTreeSet::new();
    for (&cell, by_node) in &shapes {
        if storage.contains_key(&cell) {
            continue;
        }
        for (&node, rows) in by_node {
            for shape in rows {
                let detail = match shape {
                    UnsafeHomeShape::ArithRead { .. } => "home-arith-rewrite-incomplete",
                    UnsafeHomeShape::Compare { .. } => "home-compare-rewrite-incomplete",
                    UnsafeHomeShape::ExtendLoad { .. } => "home-extend-rewrite-incomplete",
                    _ => continue,
                };
                failed_scalar_sites.insert((cell.0, node, detail));
            }
        }
    }
    if !failed_scalar_sites.is_empty() {
        let mut reasons: BTreeSet<(Address, Address, Symbol)> = db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .copied()
            .collect();
        let mut details: BTreeSet<(Address, Address, Symbol)> = db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
            .copied()
            .collect();
        for (func, node, detail) in failed_scalar_sites {
            reasons.insert((func, node, "unsupported-stack-address"));
            details.insert((func, node, detail));
        }
        db.rel_set(
            "unsupported_stack_address",
            reasons.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "unsupported_address_detail",
            details.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
    }

    db.rel_set(
        "win64_home_storage",
        storage
            .iter()
            .map(|((func, pos), slot)| (*func, *pos, *slot))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "win64_home_address",
        home_addresses
            .iter()
            .copied()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "win64_home_escaped",
        home_escaped
            .iter()
            .copied()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "win64_home_slot_type",
        cell_types
            .iter()
            .map(|(slot, xtype)| (*slot, *xtype))
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    if rewrites.is_empty() {
        return;
    }

    let rewritten_nodes: BTreeSet<Node> = rewrites.keys().copied().collect();
    let mut rewritten_candidate_nodes = rewritten_nodes.clone();
    rewritten_candidate_nodes.extend(rewritten_setcc_nodes);
    for (node, rewrite) in &rewrites {
        if matches!(rewrite, UnsafeHomeRewrite::ArithRead { .. }) {
            rewritten_candidate_nodes.insert(*node | (1u64 << 62));
        }
    }

    // Remove stale raw evidence owned by rewritten nodes, but never insert a
    // normalized entry coordinate or the reserved home ID into a raw-offset
    // relation. This keeps a post-prologue [rsp+raw] local independent.
    for relation in ["stack_xtl", "stack_var"] {
        let rows: ascent::boxcar::Vec<(Address, Address, i64, RTLReg)> = db
            .rel_iter::<(Address, Address, i64, RTLReg)>(relation)
            .filter(|(_, node, _, _)| !rewritten_nodes.contains(node))
            .copied()
            .collect();
        db.rel_set(relation, rows);
    }

    let mut stack_chunks = BTreeSet::new();
    for (node, inst) in db.rel_iter::<(Node, LTLInst)>("ltl_inst") {
        if rewritten_nodes.contains(node) {
            continue;
        }
        let Some(functions) = node_functions.get(node) else {
            continue;
        };
        let row = match inst {
            LTLInst::Lload(chunk, Addressing::Ainstack(ofs), _, _)
            | LTLInst::Lstore(chunk, Addressing::Ainstack(ofs), _, _) => Some((*ofs, *chunk)),
            LTLInst::Lgetstack(_, ofs, typ, _) | LTLInst::Lsetstack(_, _, ofs, typ) => {
                Some((*ofs, typ_to_chunk(*typ)))
            }
            _ => None,
        };
        if let Some((ofs, chunk)) = row {
            for func in functions {
                stack_chunks.insert((*func, ofs, chunk));
            }
        }
    }
    for (node, ofs, _, typ) in db.rel_iter::<(Node, i64, i64, Typ)>("mach_imm_stack_init") {
        if !rewritten_nodes.contains(node) {
            if let Some(functions) = node_functions.get(node) {
                for func in functions {
                    stack_chunks.insert((*func, *ofs, typ_to_chunk(*typ)));
                }
            }
        }
    }
    db.rel_set(
        "stack_var_chunk",
        stack_chunks.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    // Rebuild the ordinary escaped-slot aggregate from origin-keyed evidence.
    // Suppress only a LEA node that was itself rewritten as a canonical home
    // address; a distinct post-prologue local at the same raw offset remains.
    let rewritten_address_nodes: BTreeSet<Node> =
        home_addresses.iter().map(|(node, _)| *node).collect();
    let mut escaped_by_offset: BTreeMap<(Address, i64), RTLReg> = BTreeMap::new();
    for (func, origin, ofs, reg) in
        db.rel_iter::<(Address, Node, i64, RTLReg)>("slot_escaped_origin")
    {
        if rewritten_address_nodes.contains(origin) {
            continue;
        }
        escaped_by_offset
            .entry((*func, *ofs))
            .and_modify(|current| *current = (*current).min(*reg))
            .or_insert(*reg);
    }
    db.rel_set(
        "slot_escaped_canonical",
        escaped_by_offset
            .into_iter()
            .map(|((func, ofs), reg)| (func, ofs, reg))
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    // A canonical cell's type is the full supported access signature.  Peer
    // registers may carry narrower or pointer refinements for unrelated uses;
    // copying those candidates lets refinement priority silently change the
    // backing cell's width/class.
    enforce_win64_home_slot_types(db);

    // candidates_at borrows candidates. The unique peers have already been
    // proved; selection never chooses an arbitrary minimum RTL ID.
    drop(candidates_at);
    let mut selected: Vec<(Node, RTLInst)> = candidates
        .into_iter()
        .filter(|(node, _)| !rewritten_candidate_nodes.contains(node))
        .collect();
    for (node, rewrite) in rewrites {
        let inst = match rewrite {
            UnsafeHomeRewrite::Store { source, slot } => {
                Some(RTLInst::Iop(Operation::Omove, Arc::new(vec![source]), slot))
            }
            UnsafeHomeRewrite::Load { slot, destination } => {
                Some(RTLInst::Iop(Operation::Omove, Arc::new(vec![slot]), destination))
            }
            UnsafeHomeRewrite::ExtendLoad {
                slot,
                destination,
                op,
            } => Some(RTLInst::Iop(op, Arc::new(vec![slot]), destination)),
            UnsafeHomeRewrite::Lea {
                entry_ofs,
                destination,
            } => Some(RTLInst::Iop(
                Operation::Oleal(Addressing::Ainstack(entry_ofs)),
                Arc::new(vec![]),
                destination,
            )),
            UnsafeHomeRewrite::ArithRead {
                slot,
                destination,
                op,
            } => {
                selected.push((node, RTLInst::Inop));
                selected.push((
                    node | (1u64 << 62),
                    RTLInst::Iop(
                        op,
                        Arc::new(vec![destination, slot]),
                        destination,
                    ),
                ));
                None
            }
            UnsafeHomeRewrite::Compare { inst } => Some(inst),
        };
        if let Some(inst) = inst {
            selected.push((node, inst));
        }
    }
    selected.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
    selected.dedup();

    let mut used = BTreeSet::new();
    let mut call_results = BTreeSet::new();
    let mut constants = Vec::new();
    for (_, inst) in &selected {
        collect_rtl_uses(inst, &mut used);
        if let RTLInst::Icall(_, _, _, Some(destination), _) = inst {
            call_results.insert(*destination);
        }
        if let RTLInst::Iop(op, args, destination) = inst {
            if args.is_empty() {
                if let Some(constant) =
                    crate::decompile::passes::cminor_pass::constant_from_operation(op)
                {
                    constants.push((*destination, constant));
                }
            }
        }
    }
    constants.sort_by_cached_key(|(reg, constant)| (*reg, format!("{constant:?}")));
    constants.dedup();
    db.rel_set(
        "dead_def",
        call_results
            .into_iter()
            .filter(|reg| !used.contains(reg))
            .map(|reg| (reg,))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "single_def_const",
        constants.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "rtl_inst_candidate",
        selected.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

pub struct RTLPass;

impl IRPass for RTLPass {
    fn name(&self) -> &'static str {
        "rtl"
    }

    fn run(&self, db: &mut DecompileDB) {
        // Public diagnostics are rebuilt from Asm's immutable seed plus any
        // imperative RTL normalization failures on every run.
        db.rel_set(
            "unsupported_stack_address",
            ascent::boxcar::Vec::<(Address, Address, Symbol)>::new(),
        );
        db.rel_set(
            "unsupported_address_detail",
            ascent::boxcar::Vec::<(Address, Address, Symbol)>::new(),
        );
        run_pass!(db, RTLPassProgram);
        canonicalize_indexed_stack_rtl_values(db);
        select_sp_indexed_fused_lowerings(db);
        materialize_sp_indexed_fused_members(db);
        normalize_addr32_rtl_outputs(db);
        classify_indexed_stack_lowerings(db);
        // Filter before home-cell selection so its peer/type/deadness repair is
        // computed from the safe candidate set, then reassert the invariant at
        // the public pass boundary in case a canonical rewrite was produced.
        suppress_unsupported_address_candidates(db);
        select_canonical_unsafe_home_rewrites(db);
        suppress_unsupported_address_candidates(db);
    }

    fn inputs(&self) -> &'static [&'static str] {
        RTLPassProgram::inputs_only()
    }

    fn outputs(&self) -> &'static [&'static str] {
        RTLPassProgram::rule_outputs()
    }
}

#[cfg(test)]
mod encoding_tests {
    use super::*;
    use crate::aarch64::mach::A64Mreg;

    fn on_rtl_program_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("rtl-program-test".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(test)
            .expect("spawn RTL program test")
            .join()
            .expect("RTL program test panicked");
    }

    #[test]
    fn dominance_workspaces_are_demand_driven() {
        on_rtl_program_stack(|| {
            let mut prog = RTLPassProgram::default();
            let function: Address = 0x8000;
            let block: Address = 0x8000;
            let definition: Node = 0x8010;
            let write: Node = 0x8020;
            let indexed_access: Node = 0x81f0;

            // A long straight-line block used to create every ordered pair in
            // both dominance relations, even though only these two queries
            // can be consumed by indexed-stack recovery.
            for address in (0x8010..=0x81f0).step_by(0x10) {
                prog.code_in_block.push((address, block));
                prog.real_addr_in_func.push((address, function));
            }
            prog.reg_def_dominance_query
                .push((function, definition, indexed_access));
            prog.sp_indexed_fused_load.push((
                indexed_access,
                Operation::Oadd,
                MemoryChunk::MInt32,
                4,
                8,
                Mreg::AX,
                Mreg::CX,
            ));
            prog.direct_stack_operand.push((write, Mreg::SP, -32, 4));
            prog.decoded_memory_write_operand
                .push((write, "rtl_dominance_test_operand"));

            prog.run();

            assert_eq!(
                prog.reg_def_dominates_use
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![(function, definition, indexed_access)]
            );
            assert_eq!(
                prog.sp_indexed_anchor_write_dominates
                    .iter()
                    .copied()
                    .collect::<Vec<_>>(),
                vec![(write, indexed_access, function)]
            );
        });
    }

    fn seed_rejected_site(db: &mut DecompileDB, function: Address, real: Node) {
        db.rel_push(
            "unsupported_stack_address",
            (function, real, "test-unsupported-address"),
        );
        db.rel_push("instr_in_function", (real, function));
        db.rel_push("rtl_inst_candidate", (real, RTLInst::Inop));
    }

    fn outgoing(db: &DecompileDB, source: Node) -> BTreeSet<Node> {
        db.rel_iter::<(Node, Node)>("rtl_succ_candidate")
            .filter_map(|(edge_source, destination)| {
                (*edge_source == source).then_some(*destination)
            })
            .collect()
    }

    fn candidate_uses(db: &DecompileDB, node: Node, value: RTLReg) -> bool {
        db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .filter(|(candidate_node, _)| *candidate_node == node)
            .any(|(_, inst)| {
                let mut uses = BTreeSet::new();
                collect_rtl_uses(inst, &mut uses);
                uses.contains(&value)
            })
    }

    #[derive(Clone, Copy)]
    enum CmpConsumerKind {
        Jcc,
        Setcc,
    }

    fn cmp_consumer_rejection_db(
        reason: Option<&'static str>,
        kind: CmpConsumerKind,
        independent_candidate: bool,
    ) -> (DecompileDB, Node, Node, Node, Node, RTLReg, RTLInst) {
        let mut db = DecompileDB::default();
        let function: Address = 0x7000;
        let root: Node = 0x7010;
        let consumer: Node = 0x7020;
        let fallthrough: Node = 0x7030;
        let taken: Node = 0x7040;
        let temp: RTLReg = 0x7050;
        let independent = RTLInst::Iop(Operation::Ointconst(7), Arc::new(vec![]), 0x7060);

        for node in [root, consumer, fallthrough, taken] {
            db.rel_push("instr_in_function", (node, function));
        }
        if let Some(reason) = reason {
            db.rel_push("unsupported_stack_address", (function, root, reason));
        }
        db.rel_push(
            "cmp_memory_temp_consumer",
            (root, consumer, temp, function),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                root,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(40),
                    Arc::new(vec![0x7001]),
                    temp,
                ),
            ),
        );
        match kind {
            CmpConsumerKind::Jcc => {
                db.rel_push(
                    "rtl_inst_candidate",
                    (
                        consumer,
                        RTLInst::Icond(
                            Condition::Ccomp(Comparison::Ceq),
                            Arc::new(vec![temp]),
                            Either::Right(taken),
                            Either::Right(fallthrough),
                        ),
                    ),
                );
                db.rel_push("rtl_succ_candidate", (consumer, taken));
                db.rel_push("rtl_succ_candidate", (consumer, fallthrough));
            }
            CmpConsumerKind::Setcc => {
                db.rel_push(
                    "rtl_inst_candidate",
                    (
                        consumer,
                        RTLInst::Iop(
                            Operation::Ocmp(Condition::Ccomp(Comparison::Ceq)),
                            Arc::new(vec![temp]),
                            0x7021,
                        ),
                    ),
                );
                db.rel_push("rtl_succ_candidate", (consumer, fallthrough));
                // SETcc has no LTL node, so the semantic stream bypasses it.
                // Suppression must prefer the exact decoded CMP->SETcc path.
                db.rel_push("rtl_next", (root, fallthrough));
                db.rel_push("rtl_succ_candidate", (root, fallthrough));
            }
        }
        if independent_candidate {
            db.rel_push("rtl_inst_candidate", (consumer, independent.clone()));
        }
        db.rel_push("rtl_inst_candidate", (fallthrough, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (taken, RTLInst::Inop));
        db.rel_push("rtl_succ_candidate", (root, consumer));
        db.rel_push("next", (root, consumer));
        db.rel_push("next", (consumer, fallthrough));
        db.rel_push("rtl_next", (consumer, fallthrough));

        (
            db,
            root,
            consumer,
            fallthrough,
            taken,
            temp,
            independent,
        )
    }

    #[test]
    fn rejected_rsp_cmp_jcc_removes_cross_node_temp_and_taken_edge() {
        let (mut db, root, consumer, fallthrough, taken, temp, _) =
            cmp_consumer_rejection_db(
                Some("unsupported-stack-address"),
                CmpConsumerKind::Jcc,
                false,
            );

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert_eq!(outgoing(&db, root), BTreeSet::from([consumer]));
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([fallthrough]));
        assert!(!outgoing(&db, consumer).contains(&taken));
        assert_eq!(
            db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, _)| *node == consumer)
                .map(|(_, inst)| inst.clone())
                .collect::<Vec<_>>(),
            vec![RTLInst::Inop]
        );
    }

    #[test]
    fn rejected_rsp_cmp_setcc_removes_cross_node_temp_and_keeps_fallthrough() {
        let (mut db, root, consumer, fallthrough, _, temp, _) = cmp_consumer_rejection_db(
            Some("unsupported-stack-address"),
            CmpConsumerKind::Setcc,
            false,
        );

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert_eq!(outgoing(&db, root), BTreeSet::from([consumer]));
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([fallthrough]));
    }

    #[test]
    fn rejected_cmp_provenance_anchors_missing_setcc_candidate() {
        let (mut db, root, consumer, fallthrough, _, temp, _) = cmp_consumer_rejection_db(
            Some("unsupported-stack-address"),
            CmpConsumerKind::Setcc,
            false,
        );
        let candidates: ascent::boxcar::Vec<(Node, RTLInst)> = db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .filter(|(node, _)| *node != consumer)
            .map(|(node, inst)| (*node, inst.clone()))
            .collect();
        db.rel_set("rtl_inst_candidate", candidates);
        let edges: ascent::boxcar::Vec<(Node, Node)> = db
            .rel_iter::<(Node, Node)>("rtl_succ_candidate")
            .filter(|(source, _)| *source != consumer)
            .copied()
            .collect();
        db.rel_set("rtl_succ_candidate", edges);
        let semantic: ascent::boxcar::Vec<(Node, Node)> = db
            .rel_iter::<(Node, Node)>("rtl_next")
            .filter(|(source, _)| *source != consumer)
            .copied()
            .collect();
        db.rel_set("rtl_next", semantic);

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert!(db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| *node == consumer && *inst == RTLInst::Inop));
        assert_eq!(outgoing(&db, root), BTreeSet::from([consumer]));
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([fallthrough]));
    }

    #[test]
    fn rejected_addr32_cmp_jcc_obeys_the_same_cross_node_boundary() {
        let (mut db, _, consumer, fallthrough, taken, temp, _) =
            cmp_consumer_rejection_db(
                Some("unsupported-addr32-address"),
                CmpConsumerKind::Jcc,
                false,
            );

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([fallthrough]));
        assert!(!outgoing(&db, consumer).contains(&taken));
    }

    #[test]
    fn safe_generic_pointer_cmp_keeps_its_cross_node_consumer() {
        let (mut db, root, consumer, _, taken, temp, _) =
            cmp_consumer_rejection_db(None, CmpConsumerKind::Jcc, false);
        // Force the suppression pass to run for an unrelated owner; the safe
        // CMP must not be swept merely because it has retained provenance.
        seed_rejected_site(&mut db, 0x8000, 0x8010);

        suppress_unsupported_address_candidates(&mut db);

        assert!(candidate_uses(&db, consumer, temp));
        assert!(db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| *node == root && matches!(inst, RTLInst::Iload(..))));
        assert!(outgoing(&db, consumer).contains(&taken));
    }

    #[test]
    fn mixed_cmp_consumer_retains_only_independent_same_node_candidate() {
        let (mut db, _, consumer, fallthrough, taken, temp, independent) =
            cmp_consumer_rejection_db(
                Some("unsupported-stack-address"),
                CmpConsumerKind::Jcc,
                true,
            );

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        let survivors: Vec<_> = db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .filter_map(|(node, inst)| (*node == consumer).then_some(inst.clone()))
            .collect();
        assert_eq!(survivors, vec![independent]);
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([fallthrough]));
        assert!(!outgoing(&db, consumer).contains(&taken));
    }

    #[test]
    fn mixed_cmp_consumer_defers_edges_to_independent_control_candidate() {
        let (mut db, _, consumer, fallthrough, taken, temp, _) =
            cmp_consumer_rejection_db(
                Some("unsupported-stack-address"),
                CmpConsumerKind::Jcc,
                false,
            );
        let independent = RTLInst::Ibranch(Either::Right(taken));
        db.rel_push("rtl_inst_candidate", (consumer, independent.clone()));

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert!(db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, inst)| *node == consumer && *inst == independent));
        assert_eq!(outgoing(&db, consumer), BTreeSet::from([taken]));
        assert!(!outgoing(&db, consumer).contains(&fallthrough));
    }

    #[test]
    fn rejected_cmp_does_not_bridge_fallthrough_across_owners() {
        let (mut db, _, consumer, fallthrough, _, temp, _) = cmp_consumer_rejection_db(
            Some("unsupported-stack-address"),
            CmpConsumerKind::Jcc,
            false,
        );
        let mut members: Vec<(Node, Address)> = db
            .rel_iter::<(Node, Address)>("instr_in_function")
            .filter(|(node, _)| *node != fallthrough)
            .copied()
            .collect();
        members.push((fallthrough, 0x9000));
        db.rel_set(
            "instr_in_function",
            members.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );

        suppress_unsupported_address_candidates(&mut db);

        assert!(!candidate_uses(&db, consumer, temp));
        assert!(outgoing(&db, consumer).is_empty());
    }

    #[test]
    fn stack_cell_tag_is_disjoint_and_preserves_node_order() {
        let node = 0x1000;
        let cell = fresh_stack_cell_reg(node);
        assert_eq!(cell & MREG_DISCRIMINANT_MASK, STACK_CELL_DISCRIMINANT);

        // The directly constructed AArch64 Unknown is the largest current
        // machine-register discriminator (normal constructors collapse it to
        // Mreg::Unknown).  Leave an explicit gap below the reserved cell tag.
        let max_mreg = mreg_discriminant(Mreg::A64(A64Mreg::Unknown));
        assert_eq!(max_mreg, 99);
        assert!(max_mreg < STACK_CELL_DISCRIMINANT);

        for reg in [Mreg::AX, Mreg::BP, Mreg::SP, Mreg::Unknown] {
            assert_ne!(cell, fresh_xtl_reg(node, reg));
        }
        assert!(cell < fresh_xtl_reg(node + 1, Mreg::AX));
    }

    #[test]
    fn synthetic_membership_does_not_own_rejected_real() {
        let mut db = DecompileDB::default();
        let function: Address = 0x1000;
        let predecessor: Node = 0x1010;
        let real: Node = 0x1020;
        let synthetic = real | (1u64 << 62);

        db.rel_push(
            "unsupported_stack_address",
            (function, real, "test-unsupported-address"),
        );
        db.rel_push("instr_in_function", (predecessor, function));
        db.rel_push("instr_in_function", (synthetic, function));
        db.rel_push("rtl_inst_candidate", (predecessor, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (synthetic, RTLInst::Inop));
        db.rel_push("rtl_succ_candidate", (predecessor, synthetic));
        db.rel_push("next", (predecessor, real));

        suppress_unsupported_address_candidates(&mut db);

        assert!(!db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .any(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == real));
        assert!(!db
            .rel_iter::<(Node, Node)>("rtl_succ_candidate")
            .any(|(source, destination)| *source == real || *destination == real));
    }

    #[test]
    fn dead_synthetic_exit_falls_back_to_live_same_function_candidate() {
        let mut db = DecompileDB::default();
        let function: Address = 0x2000;
        let real: Node = 0x2010;
        let synthetic = real | (1u64 << 62);
        let dead: Node = 0x2020;
        let live: Node = 0x2030;

        seed_rejected_site(&mut db, function, real);
        for node in [synthetic, dead, live] {
            db.rel_push("instr_in_function", (node, function));
        }
        db.rel_push("rtl_inst_candidate", (synthetic, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (live, RTLInst::Inop));
        db.rel_push("rtl_succ_candidate", (real, synthetic));
        db.rel_push("rtl_succ_candidate", (synthetic, dead));
        db.rel_push("next", (real, dead));
        db.rel_push("next", (dead, live));

        suppress_unsupported_address_candidates(&mut db);

        assert_eq!(outgoing(&db, real), BTreeSet::from([live]));
    }

    #[test]
    fn intermediate_synthetic_bypass_is_not_a_tail_exit() {
        let mut db = DecompileDB::default();
        let function: Address = 0x3000;
        let real: Node = 0x3010;
        let synth1 = real | (1u64 << 62);
        let synth2 = real | (1u64 << 63);
        let bypass: Node = 0x3020;
        let true_exit: Node = 0x3030;

        seed_rejected_site(&mut db, function, real);
        for node in [synth1, synth2, bypass, true_exit] {
            db.rel_push("instr_in_function", (node, function));
            db.rel_push("rtl_inst_candidate", (node, RTLInst::Inop));
        }
        for edge in [
            (real, synth1),
            (synth1, synth2),
            (synth1, bypass),
            (synth2, true_exit),
        ] {
            db.rel_push("rtl_succ_candidate", edge);
        }

        suppress_unsupported_address_candidates(&mut db);

        assert_eq!(outgoing(&db, real), BTreeSet::from([true_exit]));
    }

    fn cross_function_rejection_db(with_semantic_edge: bool) -> (DecompileDB, Node, Node) {
        let mut db = DecompileDB::default();
        let source_function: Address = 0x4000;
        let destination_function: Address = 0x5000;
        let real: Node = 0x4010;
        let synthetic = real | (1u64 << 62);
        let destination: Node = 0x5010;

        seed_rejected_site(&mut db, source_function, real);
        db.rel_push("instr_in_function", (synthetic, source_function));
        db.rel_push(
            "instr_in_function",
            (destination, destination_function),
        );
        db.rel_push("rtl_inst_candidate", (synthetic, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (destination, RTLInst::Inop));
        db.rel_push("rtl_succ_candidate", (real, synthetic));
        db.rel_push("rtl_succ_candidate", (synthetic, destination));
        db.rel_push("next", (real, destination));
        if with_semantic_edge {
            db.rel_push("rtl_next", (real, destination));
            db.rel_push("rtl_edge_negated", (real, destination));
        }
        (db, real, destination)
    }

    #[test]
    fn decoded_fallback_does_not_cross_function_boundary() {
        let (mut db, real, destination) = cross_function_rejection_db(false);

        suppress_unsupported_address_candidates(&mut db);

        assert!(!outgoing(&db, real).contains(&destination));
    }

    #[test]
    fn semantic_rtl_next_may_cross_function_boundary() {
        let (mut db, real, destination) = cross_function_rejection_db(true);

        suppress_unsupported_address_candidates(&mut db);

        assert_eq!(outgoing(&db, real), BTreeSet::from([destination]));
        assert!(!db
            .rel_iter::<(Node, Node)>("rtl_edge_negated")
            .any(|edge| *edge == (real, destination)));
    }

    fn fused_selector_db(extra_node: Node) -> (DecompileDB, Node) {
        let mut db = DecompileDB::default();
        let function: Address = 0x6000;
        let real: Node = 0x6010;
        let synth1 = real | (1u64 << 62);
        let synth2 = real | (1u64 << 63);

        db.rel_push(
            "sp_indexed_fused_load",
            (
                real,
                Operation::Oadd,
                MemoryChunk::MInt32,
                4i64,
                8i64,
                Mreg::AX,
                Mreg::CX,
            ),
        );
        db.rel_push("sp_indexed_fused_complete", (real,));
        db.rel_push("instr_in_function", (real, function));
        db.rel_push(
            "rtl_inst_candidate",
            (
                real,
                RTLInst::Iop(
                    Operation::Olea(Addressing::Ainstack(8)),
                    Arc::new(vec![]),
                    1,
                ),
            ),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                synth1,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 0),
                    Arc::new(vec![1, 2]),
                    3,
                ),
            ),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                synth2,
                RTLInst::Iop(Operation::Oadd, Arc::new(vec![4, 3]), 4),
            ),
        );
        db.rel_push("rtl_inst_candidate", (extra_node, RTLInst::Inop));
        (db, real)
    }

    #[test]
    fn fused_selector_rejects_every_retained_extra_candidate() {
        let real: Node = 0x6010;
        for extra_node in [real, real | (1u64 << 62), real | (1u64 << 63)] {
            let (mut db, site) = fused_selector_db(extra_node);

            select_sp_indexed_fused_lowerings(&mut db);

            assert!(!db
                .rel_iter::<(Node,)>("sp_indexed_fused_complete")
                .any(|(node,)| *node == site));
            assert!(db
                .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(_, access, reason)| {
                    *access == site && *reason == "unsupported-stack-address"
                }));
        }
    }
}
