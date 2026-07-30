

use crate::decompile::elevator::DecompileDB;
use crate::run_pass;

use std::sync::Arc;
use crate::decompile::passes::cminor_pass::*;
use crate::decompile::passes::csh_pass::*;
use crate::x86::asm::{Ireg, TestCond};
use crate::mreg::Mreg;
use crate::x86::mach::X86Mreg;
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use ascent::ascent_par;
use ascent::Dual;
use ascent::lattice::set::Set;
use either::Either;
use std::collections::HashMap;
use crate::util::DEFAULT_VAR;
use crate::decompile::passes::asm_pass::transl_addressing_rev;
use crate::decompile::passes::pass::IRPass;

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

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct RTLPassProgram;

    relation arch_bit(i64);
    // RSP adjustments from disassembly; drives sp_entry_ofs to anchor raw SP displacements to the function-entry frame (ABI-1/3.2e).
    relation adjusts_stack(Address, Symbol, i64);
    relation arg_constrained_as_ptr(Node, RTLReg);
    // RSP<->RBP register moves from disassembly; drives func_sets_frame_pointer for BP-relative stack-param rules.
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
    relation call_returns_value(Address, Mreg);
    relation ax_value_addr(Address, Address);
    relation func_ax_def(Address, Address, RTLReg);
    relation stack_mem_add_imm(Address, i64, i64, usize);
    relation stack_mem_sub_imm(Address, i64, i64, usize);

    // SP-indexed load/store detection: loads/stores with 2 mregs where mregs[0] == SP
    #[local] relation sp_indexed_load(Node);
    #[local] relation sp_indexed_store(Node);

    // BP-frame-pointer-indexed load/store detection (mregs[0] == BP in a framed function), expanded like the SP-indexed case, or the base collapses to the whole-frame Olea(Ainstack(0)) and walks off the frame.
    #[local] relation bp_indexed_load(Node);
    #[local] relation bp_indexed_store(Node);

    // The synthetic Olea(Ainstack(ofs)) base from an SP/BP-indexed expansion, driving a per-node stack_xtl and same-offset alias so it resolves to the array's named local instead of a bare frame integer.
    #[local] relation indexed_synth_stack_base(Address, Node, i64, RTLReg);

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
        if mregs[0] == Mreg::SP;

    // Detect SP-indexed stores (2 mregs where mregs[0] is SP)
    sp_indexed_store(addr) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::SP;

    // Detect BP-frame-pointer-indexed loads (2 mregs where mregs[0] is BP) in a framed function.
    bp_indexed_load(addr) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::BP,
        instr_in_function(addr, func_start),
        func_sets_frame_pointer(func_start);

    // Detect BP-frame-pointer-indexed stores (2 mregs where mregs[0] is BP) in a framed function.
    bp_indexed_store(addr) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, _, mregs, _)),
        if mregs.len() == 2,
        if mregs[0] == Mreg::BP,
        instr_in_function(addr, func_start),
        func_sets_frame_pointer(func_start);

    // Address-arg position whose register is NOT the load's destination: its value is the register's ordinary canonical at this node.
    load_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg != dst_reg,
        reg_rtl(addr, *arg, arg_rtl);

    // Address-arg position whose register IS reused as the load's destination: resolve via the load_overwrite_use_id shadow id, the value BEFORE the load writes, or two conflicting arg vectors drop the load.
    load_arg_mapping(addr, pos, arg_rtl) <--
        ltl_inst(addr, ?LTLInst::Lload(_, _, mregs, dst_reg)),
        !sp_indexed_load(addr),
        !bp_indexed_load(addr),
        for (pos, arg) in mregs.iter().enumerate(),
        if arg == dst_reg,
        load_overwrite_use_id(addr, *dst_reg, src_xtl),
        xtl_canonical(src_xtl, arg_rtl);

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
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, ofs, rtl_reg),
        let op = match typ {
            Typ::Tint => Operation::Ointconst(*imm_val),
            Typ::Tlong | Typ::Tany64 => Operation::Olongconst(*imm_val),
            _ => Operation::Ointconst(*imm_val),
        },
        let inst = RTLInst::Iop(op, Arc::new(vec![]), *rtl_reg);

    rtl_inst_candidate(addr, inst) <--
        mach_imm_stack_init(addr, ofs, imm_val, typ),
        instr_in_function(addr, func_start),
        !stack_xtl(func_start, addr, ofs, _),
        let rtl_reg = fresh_xtl_reg(*addr, Mreg::BP),
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

    // True for cmps with a non-BP/non-SP mem base; these fold into an Icond at the jcc address (not the cmp), so the cmp -> jcc edge must remain to reach the Icond.
    relation cmp_has_non_stack_mem(Address);
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, _, sym),
        op_indirect(sym, _, base_str, _, _, _, _),
        if !base_str.ends_with("BP") && !base_str.ends_with("SP");
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, sym, _),
        op_indirect(sym, _, base_str, _, _, _, _),
        if !base_str.ends_with("BP") && !base_str.ends_with("SP");
    // An INDEXED BP/SP operand is an array/heap element access, not a scalar frame slot, so flag it non-stack and let it take the generic Iload+Icond-at-jcc path or the jcc's taken edge is dropped.
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, _, sym),
        op_indirect(sym, _, _, idx_str, _, _, _),
        if *idx_str != "NONE" && !idx_str.is_empty();
    cmp_has_non_stack_mem(*addr) <--
        pcmp(addr, sym, _),
        op_indirect(sym, _, _, idx_str, _, _, _),
        if *idx_str != "NONE" && !idx_str.is_empty();

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
        // SP stays excluded, but BP is allowed for the INDEXED case: a BP base with an index is a genuine stack-array element write, not the scalar frame-slot init carried by mach_imm_stack_init.
        if base_mreg != Mreg::SP,
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
    #[local] relation func_sets_fp_stable(Address);
    func_sets_fp_stable(func) <--
        stack_base_move(addr, src, dst),
        if *src == "RSP" && *dst == "RBP",
        real_addr_in_func(addr, func);
    #[local] relation imm_indirect_store_bp_indexed(Address, i64, MemoryChunk, &'static str, i64, i64);
    imm_indirect_store_bp_indexed(*addr, *imm_val, mc.clone(), *idx_str, *scale, *disp) <--
        imm_indirect_store_indexed(addr, imm_val, mc, base_str, idx_str, scale, disp),
        if Mreg::x86(*base_str) == Mreg::BP,
        real_addr_in_func(addr, func_start),
        func_sets_fp_stable(func_start);

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
    indexed_synth_stack_base(func_start, synth1, *disp, bp_addr_rtl) <--
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

    // arith_load_op on a THREADED spilled slot reads the slot's canonical SSA reg directly: a real frame Iload would re-materialize the slot as *(&var_0 + k) garbage, so the load node carries an Inop.
    rtl_inst_candidate(addr, nop) <--
        arith_load_op(addr, _op, _chunk, base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        if *base_mreg == Mreg::BP,
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        if *base_mreg == Mreg::BP,
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, *stack_rtl]), *dst_rtl);

    // ABI-1: the per-function rank of an incoming-stack-arg offset (ascending offset = ascending SysV position), matching the index the signature side uses when it allocates fresh_stack_param_reg for positions 6+. Tiny per-function sets, not a pairwise blowup.
    #[local] relation stack_param_lower(Address, i64, i64);
    stack_param_lower(func, ofs, o2) <--
        stack_passed_param(func, ofs),
        stack_passed_param(func, o2),
        if *o2 < *ofs;
    #[local] relation stack_param_idx(Address, i64, usize);
    stack_param_idx(func, ofs, idx) <--
        stack_passed_param(func, ofs),
        agg idx = ascent::aggregators::count() in stack_param_lower(func, ofs, _);

    // ABI-1: memory-source arithmetic reading an INCOMING stack argument binds to the function's synthetic stack-param reg, so the body references pN instead of a fresh-RTEMP fallback.
    #[local] relation arith_load_uses_stack_param(Node);
    arith_load_uses_stack_param(addr) <--
        arith_load_op(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        if *base_mreg == Mreg::BP || *base_mreg == Mreg::SP,
        instr_in_function(addr, func_start),
        stack_param_idx(func_start, disp, _);

    // Keep the original node as a transparent Inop anchor while the stack-param arithmetic lives at the synthetic successor, or rtl_optimize's liveness cannot carry the RMW destination across it.
    rtl_inst_candidate(addr, nop) <--
        arith_load_uses_stack_param(addr),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        if *base_mreg == Mreg::BP || *base_mreg == Mreg::SP,
        reg_rtl(addr, *dst_mreg, dst_rtl),
        instr_in_function(addr, func_start),
        stack_param_idx(func_start, disp, idx),
        !arith_load_uses_stack_var(addr),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, param_reg]), *dst_rtl);

    // TR-5: the load's memory chunk supplies TYPE evidence for stack-passed positions, which previously had no evidence path at all and defaulted to Xany64 in prototypes (the int_pos register caps are correct - only 6 int arg regs exist - but positions 6+ never received types).
    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        arith_load_op(addr, _, chunk, base_mreg, disp, _),
        !has_ltl_op(addr),
        if *base_mreg == Mreg::BP || *base_mreg == Mreg::SP,
        instr_in_function(addr, func_start),
        stack_param_idx(func_start, disp, idx),
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
        stack_param_idx(func_start, disp, idx),
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
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        if *base_mreg == Mreg::BP;

    rtl_inst_candidate(addr, load_inst) <--
        arith_load_op(addr, _op, chunk, base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *base_mreg, base_rtl),
        !arith_load_uses_stack_var(addr),
        !arith_load_uses_stack_param(addr),
        let temp = fresh_xtl_reg(*addr, Mreg::x86("RTEMP")),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed(*disp), Arc::new(vec![*base_rtl]), temp);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        arith_load_op(addr, op, _chunk, _base_mreg, _disp, dst_mreg),
        !has_ltl_op(addr),
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

    // stack_unary_load_op: the slot value is already live in stack_rtl, so
    // the load node carries an Inop and the unary op at the synthetic address
    // consumes stack_rtl instead of a frame Iload.
    rtl_inst_candidate(addr, nop) <--
        stack_unary_load_op(addr, _op, _base_mreg, disp, _dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        let nop = RTLInst::Inop;

    // Unary op at synth: dst = op(slot). The destination is write-only, so
    // handle both an existing canonical SSA reg and the fresh case.
    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), *dst_rtl);

    rtl_inst_candidate(synthetic_addr, op_inst) <--
        stack_unary_load_op(addr, op, _base_mreg, disp, dst_mreg),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        !reg_xtl(addr, *dst_mreg, _),
        let synthetic_addr = *addr | (1u64 << 62),
        let dst_rtl = fresh_xtl_reg(*addr, *dst_mreg),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), dst_rtl);

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
        stack_param_idx(func_start, disp, _),
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
        stack_param_idx(func_start, disp, idx),
        !stack_var(func_start, addr, disp, _),
        reg_rtl(addr, *dst_mreg, dst_rtl),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let synthetic_addr = *addr | (1u64 << 62),
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*dst_rtl, param_reg]), *dst_rtl);

    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        float_arith_stack_op(addr, op, _base_mreg, disp, _dst_mreg),
        instr_in_function(addr, func_start),
        stack_param_idx(func_start, disp, idx),
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

    synth_only_addr(addr) <-- float_load_op(addr, _, _, _, _, _, _), !has_ltl_op(addr);

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

    // arith_store_reg: memory-dest arithmetic with a register source; a BP-relative RMW operates IN-PLACE on the slot's stack_rtl so the update threads to later reads of the same slot.
    #[local] relation arith_store_reg_uses_stack_var(Node);
    arith_store_reg_uses_stack_var(addr) <--
        arith_store_reg(addr, _, _, base_mreg, disp, _),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        if *base_mreg == Mreg::BP;

    // Stack-slot in-place RMW: the load/op/store triple collapses to a single in-place `slot = slot op src` writing the slot's stack_rtl (matches the Lsetstack model), no explicit memory load/store; the edge stays addr->next (no synthetic chain), see the guarded succ rules.
    rtl_inst_candidate(addr, op_inst) <--
        arith_store_reg(addr, op, _chunk, base_mreg, disp, src_mreg),
        !has_ltl_op(addr),
        reg_rtl(addr, *src_mreg, src_rtl),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        if *base_mreg == Mreg::BP,
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl, *src_rtl]), *stack_rtl);

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
        let synth1 = *addr | (1u64 << 62),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synth2, nop) <--
        arith_store_reg(addr, _, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_reg_uses_stack_var(addr),
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

    // arith_store_imm: memory-dest arithmetic with immediate; only fires without a real ltl op. BP-relative read-modify-write (e.g. addl $1, -4(%rbp)) operates IN-PLACE on the slot's stack_rtl so the updated value threads to later reads of the same slot (see arith_store_reg).
    #[local] relation arith_store_imm_uses_stack_var(Node);
    arith_store_imm_uses_stack_var(addr) <--
        arith_store_imm(addr, _, _, base_mreg, disp),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, _),
        if *base_mreg == Mreg::BP;

    // Stack-slot in-place RMW: the load/op/store triple collapses to one in-place op writing the slot's stack_rtl, so the edge stays addr->next with no synthetic chain.
    rtl_inst_candidate(addr, op_inst) <--
        arith_store_imm(addr, op, _chunk, base_mreg, disp),
        !has_ltl_op(addr),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, disp, stack_rtl),
        if *base_mreg == Mreg::BP,
        let op_inst = RTLInst::Iop(op.clone(), Arc::new(vec![*stack_rtl]), *stack_rtl);

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
        let synth1 = *addr | (1u64 << 62),
        let nop = RTLInst::Inop;

    rtl_inst_candidate(synth2, nop) <--
        arith_store_imm(addr, _, _, _, _),
        !has_ltl_op(addr),
        arith_store_imm_uses_stack_var(addr),
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


    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if mregs.is_empty(),
        reg_rtl(addr, *dst_reg, dst_rtl),
        let inst = RTLInst::Iop(op.clone(), Arc::new(vec![]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lop(op, mregs, dst_reg)),
        if mregs.is_empty(),
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
        op_indirect(divisor_sym, _, reg2, _, _, _, _),
        if reg2.ends_with("BP");

    // Stack-divisor div emits quotient and remainder as COMPETING candidates at the SAME node, so selection keeps only the used result and a dead quotient cannot overwrite the dividend AX.
    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odiv, args.clone(), *quot_rtl)) <--
        pidiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if reg2.ends_with("BP"),
        instr_in_function(addr, func_start),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Omod, args.clone(), *rem_rtl)) <--
        pidiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if reg2.ends_with("BP"),
        instr_in_function(addr, func_start),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::DX, rem_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Odivu, args.clone(), *quot_rtl)) <--
        pudiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if reg2.ends_with("BP"),
        instr_in_function(addr, func_start),
        reg_def_used(ax_def_addr, Mreg::AX, addr),
        reg_rtl(ax_def_addr, Mreg::AX, ax_rtl),
        stack_var(func_start, addr, disp, divisor_rtl),
        reg_rtl(addr, Mreg::AX, quot_rtl),
        let args = Arc::new(vec![*ax_rtl, *divisor_rtl]);

    rtl_inst_candidate(addr, RTLInst::Iop(Operation::Omodu, args.clone(), *rem_rtl)) <--
        pudiv(addr, divisor_sym, _),
        op_indirect(divisor_sym, reg1, reg2, reg3, mult, disp, sz),
        if reg2.ends_with("BP"),
        instr_in_function(addr, func_start),
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
        let arg_mreg = mregs[0],
        if arg_mreg != *dst_reg,
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, arg_mreg, arg_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![*arg_rtl]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() == 1,
        let arg_mreg = mregs[0],
        if arg_mreg == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        load_overwrite_use_id(addr, dst_reg, src_xtl),
        xtl_canonical(src_xtl, arg_rtl),
        let inst = RTLInst::Iload(*chunk, addressing.clone(), Arc::new(vec![*arg_rtl]), *dst_rtl);

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
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, sp_addr_rtl) <--
        ltl_inst(addr, ?LTLInst::Lload(_, addressing, mregs, _)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), sp_addr_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2(ofs) = addressing,
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
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
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, sp_addr_rtl) <--
        ltl_inst(addr, ?LTLInst::Lstore(_, addressing, mregs, _)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2scaled(_, ofs) | Addressing::Aindexed2(ofs) = addressing,
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let lea_inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*ofs)), Arc::new(vec![]), sp_addr_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        reg_rtl(addr, *src_reg, src_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
        let sp_addr_rtl = fresh_xtl_reg(*addr, Mreg::SP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![sp_addr_rtl, *idx_rtl]), *src_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if mregs.len() == 2 && mregs[0] == Mreg::SP,
        if let Addressing::Aindexed2(ofs) = addressing,
        reg_rtl(addr, *src_reg, src_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
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

    // BP-frame-pointer-indexed load/store, the structural mirror of the SP-indexed expansion: expand [BP, idx] to Olea(Ainstack(disp)) plus an indexed-at-0 access, gated on func_sets_frame_pointer.
    rtl_inst_candidate(addr, lea_inst), op_produces_ptr(addr, bp_addr_rtl),
    indexed_synth_stack_base(func_start, *addr, *ofs, bp_addr_rtl) <--
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
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        if mregs[1] != *dst_reg,
        reg_rtl(addr, *dst_reg, dst_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let load_inst = RTLInst::Iload(*chunk, Addressing::Aindexed2(0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *dst_rtl);

    // Collision where the load overwrites its own index register: the dst is the fresh def and the index must be the INCOMING value from load_overwrite_use_id.
    rtl_inst_candidate(synth, load_inst) <--
        bp_indexed_load(addr),
        ltl_inst(addr, ?LTLInst::Lload(chunk, addressing, mregs, dst_reg)),
        if let Addressing::Aindexed2scaled(scale, _) = addressing,
        if mregs[1] == *dst_reg,
        is_def(addr, def_xtl),
        reg_xtl(addr, *dst_reg, def_xtl),
        xtl_canonical(def_xtl, dst_rtl),
        load_overwrite_use_id(addr, dst_reg, idx_xtl),
        xtl_canonical(idx_xtl, idx_rtl),
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
        load_overwrite_use_id(addr, dst_reg, idx_xtl),
        xtl_canonical(idx_xtl, idx_rtl),
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
    indexed_synth_stack_base(func_start, *addr, *ofs, bp_addr_rtl) <--
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
        reg_rtl(addr, *src_reg, src_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
        let bp_addr_rtl = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_SP_BASE,
        let synth = *addr | (1u64 << 62),
        let store_inst = RTLInst::Istore(*chunk, Addressing::Aindexed2scaled(*scale, 0), Arc::new(vec![bp_addr_rtl, *idx_rtl]), *src_rtl);

    rtl_inst_candidate(synth, store_inst) <--
        bp_indexed_store(addr),
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if let Addressing::Aindexed2(_) = addressing,
        reg_rtl(addr, *src_reg, src_rtl),
        reg_rtl(addr, mregs[1], idx_rtl),
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

    // Give the BP-indexed synthetic Olea(Ainstack(ofs)) a per-node stack_xtl so cminor resolves it to Eaddrof(&local) instead of leaving it as the bare frame integer.
    stack_xtl(func_start, addr, ofs, base_reg) <--
        indexed_synth_stack_base(func_start, addr, ofs, base_reg);

    // Unify the synthetic base with every stack_xtl reg at the SAME frame offset so the indexed access and the address-of resolve to ONE named local; the FRESH_NS_SP_BASE bit makes the real local win the min.
    alias_edge(base_reg, other_reg) <--
        indexed_synth_stack_base(func_start, _, ofs, base_reg),
        stack_xtl(func_start, _, ofs, other_reg),
        if base_reg != other_reg;

    // Function-bounded skip walk: from a non-ltl-inst addr, walk `next` within the func until hitting one with ltl_inst. Only consumed by SP-indexed synth rules above.
    sp_synth_skip_to_ltl(start, func, dst) <--
        instr_in_function(start, func),
        next(start, dst),
        instr_in_function(dst, func),
        ltl_inst(*dst, _);

    sp_synth_skip_to_ltl(start, func, dst) <--
        instr_in_function(start, func),
        next(start, mid),
        !ltl_inst(*mid, _),
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
        if (!base_str.ends_with("BP") && !base_str.ends_with("SP")) || (*idx_str != "NONE" && !idx_str.is_empty()),
        let fresh_reg = fresh_xtl_reg(*addr, Mreg::x86("RTEMP"));

    temp_cmp_mreg_args(addr, pos, mreg) <--
        instruction(addr, _, _, mnem, dst, src, _, _, _, _),
        if mnem.starts_with("CMP"),
        op_immediate(dst, _, _),
        op_indirect(src, _, base_str, idx_str, scale, disp, _),
        if (!base_str.ends_with("BP") && !base_str.ends_with("SP")) || (*idx_str != "NONE" && !idx_str.is_empty()),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        let res = transl_addressing_rev(addrmode, None),
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
        if (!base_str.ends_with("BP") && !base_str.ends_with("SP")) || (*idx_str != "NONE" && !idx_str.is_empty()),
        temp_cmp_reg(addr, fresh_reg),
        let addrmode = build_cmp_addrmode(base_str, idx_str, *scale, *disp),
        let res = transl_addressing_rev(addrmode, None),
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
        if (!base_str.ends_with("BP") && !base_str.ends_with("SP")) || (*idx_str != "NONE" && !idx_str.is_empty()),
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
        let res = transl_addressing_rev(addrmode, None),
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
        let res = transl_addressing_rev(addrmode, None),
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
        let res = transl_addressing_rev(addrmode, None),
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
        let res = transl_addressing_rev(addrmode, None),
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
        if (!base_str.ends_with("BP") && !base_str.ends_with("SP")) || (*idx_str != "NONE" && !idx_str.is_empty()),
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
        !sp_indexed_store(addr),
        !bp_indexed_store(addr),
        reg_rtl(addr, *src_reg, src_rtl),
        store_args_collected(addr, args),
        let inst = RTLInst::Istore(*chunk, addressing.clone(), args.clone(), *src_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lstore(chunk, addressing, mregs, src_reg)),
        if mregs.is_empty(),
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
        if reg_2.ends_with("BP") || reg_2.ends_with("SP"),
        instr_in_function(addr, func_start),
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
        if reg_2.ends_with("BP") || reg_2.ends_with("SP"),
        op_register(sym2, reg_str),
        let reg1 = Mreg::x86(reg_str.to_string()),
        instr_in_function(addr, func_start),
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
        if reg_2a.ends_with("BP") || reg_2a.ends_with("SP"),
        op_indirect(sym2, reg_1, reg_2, reg_3, mult, disp, sz),
        if reg_2.ends_with("BP") || reg_2.ends_with("SP"),
        instr_in_function(addr, func_start),
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
        if reg_2.ends_with("BP") || reg_2.ends_with("SP"),
        op_immediate(sym2, imm_val, _),
        instr_in_function(addr, func_start),
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
        if reg_2.ends_with("BP") || reg_2.ends_with("SP"),
        instr_in_function(addr, func_start),
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

    // (3.2e) Extending load of an incoming stack arg binds its dst to the function's synthetic stack-param reg (position 6+idx), same as arith_load_uses_stack_param; without this, Lgetstack fallbacks fabricate an uninitialized local for the 7th+ int/short parameter.
    #[local] relation lgetstack_is_stack_param(Node);
    lgetstack_is_stack_param(addr) <--
        extending_stack_arg_load(addr, func_start, disp, _),
        stack_param_idx(func_start, disp, _);

    rtl_inst_candidate(addr, inst) <--
        extending_stack_arg_load(addr, func_start, disp, _),
        stack_param_idx(func_start, disp, idx),
        ltl_inst(addr, ?LTLInst::Lgetstack(_slot, _ofs, _typ, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        let param_reg = fresh_stack_param_reg(*func_start, *idx),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![param_reg]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        extending_stack_arg_load(addr, func_start, disp, _),
        stack_param_idx(func_start, disp, idx),
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
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*rtl_reg]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        reg_rtl(addr, *dst, dst_rtl),
        instr_in_function(addr, func_start),
        !stack_var(func_start, addr, ofs, _),
        !lgetstack_is_stack_param(addr),
        let fresh_src = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_STACK_SRC,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![fresh_src]), *dst_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        !reg_rtl(addr, dst, _),
        instr_in_function(addr, func_start),
        stack_var(func_start, addr, ofs, rtl_reg),
        !lgetstack_is_stack_param(addr),
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*rtl_reg]), fresh_dst);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lgetstack(slot, ofs, typ, dst)),
        !reg_rtl(addr, dst, _),
        instr_in_function(addr, func_start),
        !stack_var(func_start, addr, ofs, _),
        !lgetstack_is_stack_param(addr),
        let fresh_src = fresh_xtl_reg(*addr, Mreg::BP) | FRESH_NS_STACK_SRC,
        let fresh_dst = fresh_xtl_reg(*addr, *dst) | FRESH_NS_REG_DST,
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![fresh_src]), fresh_dst);

    // Read the spill SOURCE id at the DEF site, but bind the load's def id at a load_overwrites_base site, or the ADDRESS is spilled instead of the loaded value and the load's real def is DSE'd.
    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        instr_in_function(addr, func_start),
        reg_def_used(defaddr, *src, *addr),
        !load_overwrites_base(defaddr, src),
        reg_rtl(defaddr, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
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
        instr_in_function(addr, func_start),
        !reg_def_used(_, src, addr),
        is_arg_reg(src),
        func_param_validated(func_start, src),
        reg_rtl(func_start, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        instr_in_function(addr, func_start),
        reg_rtl(addr, *src, src_rtl),
        !stack_var(func_start, addr, ofs, _),
        let fresh_stack = fresh_xtl_reg(*addr, Mreg::BP),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), fresh_stack);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
        instr_in_function(addr, func_start),
        !reg_def_used(_, src, addr),
        is_arg_reg(src),
        !func_param_validated(func_start, src),
        reg_rtl(addr, *src, src_rtl),
        stack_var(func_start, addr, ofs, stack_rtl),
        let inst = RTLInst::Iop(Operation::Omove, Arc::new(vec![*src_rtl]), *stack_rtl);

    rtl_inst_candidate(addr, inst) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(src, slot, ofs, typ)),
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
        if (*base_str).ends_with("BP"),
        let dst_mreg = Mreg::x86(dst_str.to_string()),
        let dst_rtl = fresh_xtl_reg(*addr, dst_mreg),
        let inst = RTLInst::Iop(Operation::Olea(Addressing::Ainstack(*disp)), Arc::new(vec![]), dst_rtl);

    ltl_inst(addr, ltl) <--
        plea(addr, dst_sym, src_addr),
        op_register(dst_sym, dst_str),
        op_indirect(src_addr, _, base_str, _, _scale, disp, _),
        if (*base_str).ends_with("BP"),
        let dst_mreg = Mreg::x86(dst_str.to_string()),
        let ltl = LTLInst::Lop(Operation::Olea(Addressing::Ainstack(*disp)), Arc::new(vec![]), dst_mreg);


    stack_mem_add_imm(addr, disp, imm_val, sz) <--
        padd(addr, dst_sym, src_sym),
        op_indirect(dst_sym, _, base_str, _, _, disp, sz),
        if (*base_str).ends_with("BP"),
        op_immediate(src_sym, imm_val, _);

    stack_mem_sub_imm(addr, disp, imm_val, sz) <--
        psub(addr, dst_sym, src_sym),
        op_indirect(dst_sym, _, base_str, _, _, disp, sz),
        if (*base_str).ends_with("BP"),
        op_immediate(src_sym, imm_val, _);

    stack_xtl(func_start, addr, disp, rtl_reg) <--
        stack_mem_add_imm(addr, disp, _, _),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_xtl_reg(*addr, Mreg::BP);

    stack_xtl(func_start, addr, disp, rtl_reg) <--
        stack_mem_sub_imm(addr, disp, _, _),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_xtl_reg(*addr, Mreg::BP);

    stack_xtl(func_start, addr, ofs, rtl_reg) <--
        ltl_inst(addr, ?LTLInst::Lsetstack(_, _, ofs, _)),
        instr_in_function(addr, func_start),
        let rtl_reg = fresh_xtl_reg(*addr, Mreg::BP);

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
        let rtlreg = fresh_xtl_reg(*addr, Mreg::BP);

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
        let rtlreg = fresh_xtl_reg(*addr, Mreg::BP);

    stack_xtl(func_start, addr, ofs, rtlreg) <--
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Ainstack(ofs), _, _)),
        instr_in_function(addr, func_start),
        !stack_def_used(_, _, _, addr, _, ofs),
        let rtlreg = fresh_xtl_reg(*addr, Mreg::BP);

    stack_xtl(func_start, addr, ofs, rtlreg) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Olea(Addressing::Ainstack(ofs)), _, _)),
        instr_in_function(addr, func_start),
        let rtlreg = fresh_xtl_reg(*addr, Mreg::BP);


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
    #[local] relation slot_escaped_canonical_raw(Address, i64, RTLReg);
    slot_escaped_canonical_raw(func_start, ofs, canonical) <--
        slot_addr_escaped(func_start, lea_addr, ofs),
        stack_var(func_start, lea_addr, ofs, canonical);

    relation slot_escaped_canonical(Address, i64, RTLReg);
    slot_escaped_canonical(func, ofs, m) <--
        slot_escaped_canonical_raw(func, ofs, _),
        agg m = ascent::aggregators::min(c) in slot_escaped_canonical_raw(func, ofs, c);

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
        reg_rtl(defaddr, dst_reg, rtl_reg),
        abi_int_arg_position(dst_reg, pos),
        call_arg_position_allowed(call_addr, pos),
        instr_in_function(defaddr, _func_start);

    call_arg_mapping(*call_addr, pos, rtl_reg) <--
        tailcall_arg_setup_detected(defaddr, dst_reg, call_addr),
        reg_rtl(defaddr, dst_reg, rtl_reg),
        abi_int_arg_position(dst_reg, pos),
        call_arg_position_allowed(call_addr, pos),
        instr_in_function(defaddr, _func_start);

    // R3: outgoing STACK arguments anchored in the ENTRY frame so they reconcile against the call's RSP; an approximation that may under-recover an arg stored in an earlier block but never fabricates one for a local.
    #[local] relation og_arg_store(Address, i64, RTLReg, Address);
    og_arg_store(*st_addr, abs_slot, *src_rtl, *func) <--
        ltl_inst(st_addr, ?LTLInst::Lstore(_, Addressing::Aindexed(ofs), args, src)),
        if args.len() == 1 && args[0] == Mreg::SP,
        if *ofs >= 0,
        instr_in_function(st_addr, func),
        !stack_var(func, st_addr, *ofs, _),
        sp_entry_ofs(func, st_addr, sp_st),
        let abs_slot = *ofs + sp_st.0,
        reg_rtl(st_addr, *src, src_rtl);

    // Simple [rsp+disp] = reg stores lift through Lsetstack, so recover their entry-anchored slot; these shapes are WEAK and need callee-body corroboration, while the Lstore and PUSH shapes stay strong.
    #[local] relation og_arg_store_weak(Address, i64, RTLReg);

    og_arg_store(*st_addr, abs_slot, *src_rtl, *func),
    og_arg_store_weak(*st_addr, abs_slot, *src_rtl) <--
        ltl_inst(st_addr, ?LTLInst::Lsetstack(src, _, ofs, _)),
        pmov(st_addr, dst_sym, _),
        op_indirect(dst_sym, _, base_str, idx_str, _, disp, _),
        if Mreg::x86(*base_str) == Mreg::SP,
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
        if Mreg::x86(*base_str) == Mreg::SP,
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
        reg_rtl(defaddr, dst_reg, rtl_reg),
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

    // Frameless (-O2 FPO) functions forward a parameter into rbp as scratch, so gate the dst==BP exclusion on func_sets_frame_pointer or every pointer param held in rbp loses its pointer evidence.
    arg_reg_copy_site(func_start, *addr, *src, *dst) <--
        ltl_inst(addr, ?LTLInst::Lop(Operation::Omove, srcs, dst)),
        if srcs.len() == 1,
        for src in srcs.iter(),
        is_arg_reg(src),
        if *dst == Mreg::BP,
        arg_reg_param_live_at(func_start, addr, src),
        !func_sets_frame_pointer(func_start);

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


    relation stack_passed_param(Address, i64);

    // ABI-1: entry-anchored SP offset per instruction, so disp + sp_ofs >= 8 places a slot above the return address regardless of prologue depth; the chain stops at non-fallthrough instructions.
    #[local] lattice sp_entry_ofs(Address, Address, Dual<i64>);

    // Flow breakers: instructions whose linear successor is not reachable; breaking too eagerly only under-claims stack params, never fabricates.
    #[local] relation sp_chain_breaker(Address);
    sp_chain_breaker(addr) <--
        instruction(addr, _, _, mnem, _, _, _, _, _, _),
        if matches!(*mnem, "RET" | "RETF" | "RETFQ" | "JMP" | "LJMP" | "HLT"
            | "UD0" | "UD1" | "UD2" | "INT3" | "IRET" | "IRETD" | "IRETQ" | "SYSRET");

    sp_entry_ofs(func_start, func_start, Dual(0)) <--
        func_span(_, func_start, _);

    sp_entry_ofs(func_start, curaddr, Dual(prev_ofs.0 + *delta)) <--
        sp_entry_ofs(func_start, prevaddr, prev_ofs),
        next(prevaddr, curaddr),
        !sp_chain_breaker(prevaddr),
        instr_in_function(curaddr, func_start),
        adjusts_stack(prevaddr, _, delta);

    sp_entry_ofs(func_start, curaddr, prev_ofs) <--
        sp_entry_ofs(func_start, prevaddr, prev_ofs),
        next(prevaddr, curaddr),
        !sp_chain_breaker(prevaddr),
        instr_in_function(curaddr, func_start),
        !adjusts_stack(prevaddr, _, _);

    // BP-based arg detection requires an actual frame pointer: in frameless functions RBP is a callee-saved scratch (often a struct pointer), so a positive-offset BP load is a field deref, not an incoming arg. With a frame pointer, caller args start at BP+16 (BP+0=saved RBP, BP+8=return address).
    #[local] relation func_sets_frame_pointer(Address);
    func_sets_frame_pointer(func_start) <--
        stack_base_move(addr, src, dst),
        if *src == "RSP" && *dst == "RBP",
        instr_in_function(addr, func_start);

    stack_passed_param(func_start, *ofs) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Aindexed(ofs), args, _)),
        for arg in args.iter(),
        abi_incoming_bp_stack_base(stack_base),
        if *arg == Mreg::BP && *ofs >= *stack_base,
        func_sets_frame_pointer(func_start);

    // Detect functions that use BP as an indexed-addressing base (framed functions)
    #[local] relation func_uses_bp_base(Address);
    func_uses_bp_base(func_start) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Aindexed(_), args, _)),
        for arg in args.iter(),
        if *arg == Mreg::BP;
    func_uses_bp_base(func_start) <--
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lstore(_, Addressing::Aindexed(_), args, _)),
        for arg in args.iter(),
        if *arg == Mreg::BP;

    // SP rules use *ofs + sp_ofs.0 >= 8 (ENTRY-frame anchor); a missing sp_entry_ofs row conservatively yields no param claim.
    stack_passed_param(func_start, *ofs) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        ltl_inst(addr, ?LTLInst::Lload(_, Addressing::Aindexed(ofs), args, _)),
        sp_entry_ofs(func_start, addr, sp_ofs),
        for arg in args.iter(),
        abi_incoming_sp_stack_base(stack_base),
        if *arg == Mreg::SP && *ofs + sp_ofs.0 >= *stack_base,
        !func_uses_bp_base(func_start);

    // Memory-operand arithmetic like `add 0x8(%rsp), %eax` is lifted via arith_load_op
    stack_passed_param(func_start, *ofs) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        arith_load_op(addr, _, _, base, ofs, _),
        sp_entry_ofs(func_start, addr, sp_ofs),
        abi_incoming_sp_stack_base(stack_base),
        if *base == Mreg::SP && *ofs + sp_ofs.0 >= *stack_base,
        !func_uses_bp_base(func_start);

    stack_passed_param(func_start, *ofs) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        arith_load_op(addr, _, _, base, ofs, _),
        abi_incoming_bp_stack_base(stack_base),
        if *base == Mreg::BP && *ofs >= *stack_base,
        func_sets_frame_pointer(func_start);

    // Fused scalar float arithmetic with an SP/BP memory source bypasses LTL, so it needs its own incoming-parameter classification with the same entry-frame anchors.
    stack_passed_param(func_start, *disp) <--
        instr_in_function(addr, func_start),
        float_arith_stack_op(addr, _, base, disp, _),
        if *base == Mreg::SP,
        sp_entry_ofs(func_start, addr, sp_ofs),
        abi_incoming_sp_stack_base(stack_base),
        if *disp + sp_ofs.0 >= *stack_base,
        !func_uses_bp_base(func_start);

    stack_passed_param(func_start, *disp) <--
        instr_in_function(addr, func_start),
        float_arith_stack_op(addr, _, base, disp, _),
        if *base == Mreg::BP,
        abi_incoming_bp_stack_base(stack_base),
        if *disp >= *stack_base,
        func_sets_frame_pointer(func_start);

    // Sign/zero-extending stack-arg loads bypass Lload, so the capstone operand pins base, width and signedness; restricted to extending mnemonics since plain MOV includes callee-save reloads.
    #[local] relation extending_stack_arg_load(Node, Address, i64, MemoryChunk);

    // SP-relative: disp + sp_entry_ofs >= 8 anchors to the ENTRY frame; the framed-function exclusion must be capstone-level, and !stack_var would self-defeat against the asm pass's shadow Olea.
    extending_stack_arg_load(addr, *func_start, *disp, mc) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, _, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, msize),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == Mreg::SP,
        sp_entry_ofs(func_start, addr, sp_ofs),
        abi_incoming_sp_stack_base(stack_base),
        if *disp + sp_ofs.0 >= *stack_base,
        !func_bp_mem_base(func_start),
        !stack_def_used(_, _, _, addr, _, disp),
        if let Some(mc) = extending_load_chunk(mnem, *msize);

    // A normal Win64 stack argument is read with plain MOV, not an extending load; entry-anchored offsets distinguish it from a local, and the reaching stack-def veto excludes an overwritten slot.
    extending_stack_arg_load(addr, *func_start, *disp, mc) <--
        abi_shared_arg_slots(true),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, typ, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVSS" | "MOVSD" | "VMOVSS" | "VMOVSD"),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == Mreg::SP,
        sp_entry_ofs(func_start, addr, sp_ofs),
        abi_incoming_sp_stack_base(stack_base),
        if *disp + sp_ofs.0 >= *stack_base,
        !func_bp_mem_base(func_start),
        !stack_def_used(_, _, _, addr, _, disp),
        let mc = typ_to_chunk(*typ);

    extending_stack_arg_load(addr, *func_start, *disp, mc) <--
        abi_shared_arg_slots(true),
        instr_in_function(addr, func_start),
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, typ, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        if matches!(*mnem, "MOV" | "MOVSS" | "MOVSD" | "VMOVSS" | "VMOVSD"),
        op_indirect(src, _, base_str, idx_str, _, disp, _),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == Mreg::BP,
        abi_incoming_bp_stack_base(stack_base),
        if *disp >= *stack_base,
        func_sets_frame_pointer(func_start),
        !stack_def_used(_, _, _, addr, _, disp),
        let mc = typ_to_chunk(*typ);

    // BP-relative (framed): args at BP+16+; func_sets_frame_pointer is EXPLICIT so the rule cannot silently widen if lifting conditions change (frameless RBP is a callee-saved scratch, not a frame pointer).
    extending_stack_arg_load(addr, *func_start, *disp, mc) <--
        instr_in_function(addr, func_start),
        function_entry_count(func_start, addr, count),
        if *count < 512,
        ltl_inst(addr, ?LTLInst::Lgetstack(_, ofs, _, _)),
        instruction(addr, _, _, mnem, src, _, _, _, _, _),
        op_indirect(src, _, base_str, idx_str, _, disp, msize),
        if *disp == *ofs,
        if *idx_str == "NONE" || idx_str.is_empty(),
        if Mreg::x86(*base_str) == Mreg::BP,
        abi_incoming_bp_stack_base(stack_base),
        if *disp >= *stack_base,
        func_sets_frame_pointer(func_start),
        !stack_def_used(_, _, _, addr, _, disp),
        if let Some(mc) = extending_load_chunk(mnem, *msize);

    // Capstone-level framed-ness: subsumes func_uses_bp_base, which cannot see Mgetstack/Msetstack shapes.
    #[local] relation func_bp_mem_base(Address);
    func_bp_mem_base(func_start) <--
        instr_in_function(addr, func_start),
        instruction(addr, _, _, _, src, _, _, _, _, _),
        op_indirect(src, _, base_str, _, _, _, _),
        if Mreg::x86(*base_str) == Mreg::BP;
    func_bp_mem_base(func_start) <--
        instr_in_function(addr, func_start),
        instruction(addr, _, _, _, _, dst, _, _, _, _),
        op_indirect(dst, _, base_str, _, _, _, _),
        if Mreg::x86(*base_str) == Mreg::BP;

    stack_passed_param(func_start, *ofs) <--
        extending_stack_arg_load(_, func_start, ofs, _);

    // TR-5 companion: extension width/signedness pins the sub-int XType for the synthetic stack-param reg (more precise than the arith/Lload chunk rules).
    emit_function_param_type_candidate(func_start, param_reg, xt) <--
        extending_stack_arg_load(_, func_start, disp, chunk),
        stack_param_idx(func_start, disp, idx),
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
        stack_passed_param(func_start, _),
        agg count = ascent::aggregators::count() in stack_passed_param(func_start, _);

    emit_function_stack_param_count(func_start, 0) <--
        emit_function(func_start, _, _),
        !stack_passed_param(func_start, _);


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

    func_has_param_evidence(func_start, pos) <--
        func_gp_param_evidence(func_start, pos);
    // On Windows a used XMM register occupies that source-language ordinal in the same sequence as GP args.
    func_float_param_evidence_at(func_start, pos) <--
        abi_shared_arg_slots(true),
        func_float_param_used(func_start, mreg),
        abi_float_arg_position(mreg, pos);
    func_has_param_evidence(func_start, pos) <--
        func_float_param_evidence_at(func_start, pos);

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
    matches!(op,
        Operation::Oshl | Operation::Oshr | Operation::Oshru |
        Operation::Oshll | Operation::Oshrl | Operation::Oshrlu
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
    let sig_res = if has_return { XType::Xint } else { XType::Xvoid };
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
            X86Mreg::AX => 0, X86Mreg::BX => 1, X86Mreg::CX => 2, X86Mreg::DX => 3,
            X86Mreg::SI => 4, X86Mreg::DI => 5, X86Mreg::BP => 6,
            X86Mreg::R8 => 7, X86Mreg::R9 => 8, X86Mreg::R10 => 9, X86Mreg::R11 => 10,
            X86Mreg::R12 => 11, X86Mreg::R13 => 12, X86Mreg::R14 => 13, X86Mreg::R15 => 14,
            X86Mreg::X0 => 15, X86Mreg::X1 => 16, X86Mreg::X2 => 17, X86Mreg::X3 => 18,
            X86Mreg::X4 => 19, X86Mreg::X5 => 20, X86Mreg::X6 => 21, X86Mreg::X7 => 22,
            X86Mreg::X8 => 23, X86Mreg::X9 => 24, X86Mreg::X10 => 25, X86Mreg::X11 => 26,
            X86Mreg::X12 => 27, X86Mreg::X13 => 28, X86Mreg::X14 => 29, X86Mreg::X15 => 30,
            X86Mreg::FP0 => 31, X86Mreg::SP => 32, X86Mreg::Unknown => 33,
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
    let base = if has_base { Some(Ireg::from(base_str)) } else { None };
    let index = if has_idx { Some((Ireg::from(idx_str), scale)) } else { None };
    Addrmode {
        base,
        index,
        disp: Displacement::from(disp),
    }
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
        "SETE" | "SETZ"   => TestCond::CondE,
        "SETNE" | "SETNZ" => TestCond::CondNe,
        "SETL"            => TestCond::CondL,
        "SETLE"           => TestCond::CondLe,
        "SETG"            => TestCond::CondG,
        "SETGE"           => TestCond::CondGe,
        "SETB" | "SETC"   => TestCond::CondB,
        "SETBE"           => TestCond::CondBe,
        "SETA"            => TestCond::CondA,
        "SETAE" | "SETNC" => TestCond::CondAe,
        _ => return None,
    })
}

pub(crate) const FRESH_NS_STACK_SRC: u64 = 1 << 54;
pub(crate) const FRESH_NS_REG_DST: u64   = 1 << 55;
pub(crate) const FRESH_NS_SP_BASE: u64   = 1 << 56;
pub(crate) const FRESH_NS_STACK_PARAM: u64 = 1 << 57;

pub(crate) fn fresh_stack_param_reg(func_addr: Node, stack_idx: usize) -> RTLReg {
    (1u64 << 63) | FRESH_NS_STACK_PARAM | (func_addr << 6) | ((stack_idx as u64) & 0x3F)
}

pub fn convert_builtin_arg(
    node: Node,
    arg: &BuiltinArg<Mreg>,
    reg_rtl_map: &HashMap<(Node, Mreg), RTLReg>,
) -> BuiltinArg<RTLReg> {
    match arg {
        BuiltinArg::BA(mreg) => {
            BuiltinArg::BA(reg_rtl_map.get(&(node, *mreg)).copied().unwrap_or(DEFAULT_VAR as u64))
        }
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
            .and_modify(|cur| if *r < *cur { *cur = *r; })
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
    matches!(cond,
        Condition::Ccompimm(Comparison::Ceq, 0) | Condition::Ccompuimm(Comparison::Ceq, 0) |
        Condition::Ccompimm(Comparison::Cne, 0) | Condition::Ccompuimm(Comparison::Cne, 0) |
        Condition::Ccomplimm(Comparison::Ceq, 0) | Condition::Ccompluimm(Comparison::Ceq, 0) |
        Condition::Ccomplimm(Comparison::Cne, 0) | Condition::Ccompluimm(Comparison::Cne, 0)
    )
}

pub fn is_null_comparison_op(op: &Operation) -> bool {
    matches!(op,
        Operation::Ocmp(Condition::Ccompimm(Comparison::Ceq, 0)) |
        Operation::Ocmp(Condition::Ccompuimm(Comparison::Ceq, 0)) |
        Operation::Ocmp(Condition::Ccompimm(Comparison::Cne, 0)) |
        Operation::Ocmp(Condition::Ccompuimm(Comparison::Cne, 0)) |
        Operation::Ocmp(Condition::Ccomplimm(Comparison::Ceq, 0)) |
        Operation::Ocmp(Condition::Ccompluimm(Comparison::Ceq, 0)) |
        Operation::Ocmp(Condition::Ccomplimm(Comparison::Cne, 0)) |
        Operation::Ocmp(Condition::Ccompluimm(Comparison::Cne, 0))
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

pub struct RTLPass;

impl IRPass for RTLPass {
    fn name(&self) -> &'static str { "rtl" }

    fn run(&self, db: &mut DecompileDB) {
        run_pass!(db, RTLPassProgram);
    }

    fn inputs(&self) -> &'static [&'static str] {
        RTLPassProgram::inputs_only()
    }

    fn outputs(&self) -> &'static [&'static str] {
        RTLPassProgram::rule_outputs()
    }
}
