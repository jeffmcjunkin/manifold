use crate::declare_io_from;
use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;

use crate::mreg::Mreg;
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use ascent::ascent_par;
use either::Either;
use std::sync::Arc;

/// Map a memory chunk to the XType of the value loaded/stored.
fn chunk_xtype(chunk: &MemoryChunk) -> XType {
    match chunk {
        MemoryChunk::MBool => XType::Xbool,
        MemoryChunk::MInt8Signed => XType::Xint8signed,
        MemoryChunk::MInt8Unsigned => XType::Xint8unsigned,
        MemoryChunk::MInt16Signed => XType::Xint16signed,
        MemoryChunk::MInt16Unsigned => XType::Xint16unsigned,
        MemoryChunk::MInt32 | MemoryChunk::MAny32 => XType::Xint,
        MemoryChunk::MInt64 | MemoryChunk::MAny64 => XType::Xany64,
        MemoryChunk::MFloat32 => XType::Xsingle,
        MemoryChunk::MFloat64 => XType::Xfloat,
        // Xany64 is a width floor; Xlong would falsely claim signed-64.
        MemoryChunk::Unknown => XType::Xany64,
    }
}

/// Map an operation to its output register's XType, returning None for unclassified operations.
fn op_output_xtype(op: &Operation) -> Option<XType> {
    use Operation::*;
    match op {
        // Signed 32-bit
        Odiv | Omod | Oshr | Oshrimm(_) | Oshrximm(_) | Odivimm(_) | Omodimm(_) | Ocast32signed
        | Omulhs => Some(XType::Xint),

        // Unsigned 32-bit
        Odivu | Omodu | Oshru | Oshruimm(_) | Odivuimm(_) | Omoduimm(_) | Ocast32unsigned
        | Omulhu => Some(XType::Xintunsigned),

        // Ambiguous 32-bit (add/sub/mul/logic, same for signed and unsigned)
        Oadd | Osub | Omul | Oaddimm(_) | Omulimm(_) | Oand | Oor | Oxor | Onot | Oneg
        | Oandimm(_) | Oorimm(_) | Oxorimm(_) | Oshl | Oshlimm(_) | Olowlong | Ointconst(_)
        | Ointoffloat | Ointofsingle => Some(XType::Xint),

        // 64-bit LEA gets an integer-width floor (Xlong) only: gcc emits lea for plain integer arithmetic too, so forcing Xptr broke branchless selects; use-based is_ptr supplies Xptr in section 7.
        Oleal(_) => Some(XType::Xlong),

        // Signed 64-bit
        Odivl | Omodl | Oshrl | Oshrlimm(_) | Oshrxlimm(_) | Odivlimm(_) | Omodlimm(_)
        | Omullhs => Some(XType::Xlong),

        // Unsigned 64-bit
        Odivlu | Omodlu | Oshrlu | Oshrluimm(_) | Odivluimm(_) | Omodluimm(_) | Omullhu => {
            Some(XType::Xlongunsigned)
        }

        // Ambiguous 64-bit
        Oaddl | Osubl | Omull | Oaddlimm(_) | Omullimm(_) | Oandl | Oorl | Oxorl | Onotl
        | Onegl | Oandlimm(_) | Oorlimm(_) | Oxorlimm(_) | Oshll | Oshllimm(_) | Olongconst(_)
        | Olongoffloat | Olongofsingle => Some(XType::Xlong),

        // 32-bit LEA gets an integer-width floor (Xint) only, for the same reason as Oleal; pointerness is decided by use-based is_ptr evidence in section 7.
        Olea(_) => Some(XType::Xint),

        // Oindirectsymbol loads a symbol's address (e.g. GOT entry): unconditionally a pointer.
        Oindirectsymbol(_) => Some(XType::Xptr),

        // Sub-int casts
        Ocast8signed => Some(XType::Xint8signed),
        Ocast8unsigned => Some(XType::Xint8unsigned),
        Ocast16signed => Some(XType::Xint16signed),
        Ocast16unsigned => Some(XType::Xint16unsigned),

        // Float (double precision). Ofloatofsingle is CVTSS2SD (f32->f64) per asm_pass; the previous grouping had single<->double conversions swapped.
        Onegf | Oabsf | Oaddf | Osubf | Omulf | Odivf | Omaxf | Ominf | Ofloatofint
        | Ofloatoflong | Ofloatofsingle => Some(XType::Xfloat),

        // Float (single precision). Osingleoffloat is CVTSD2SS (f64 -> f32).
        Onegfs | Oabsfs | Oaddfs | Osubfs | Omulfs | Odivfs | Osingleofint | Osingleoflong
        | Osingleoffloat => Some(XType::Xsingle),

        // Comparison result (boolean)
        Ocmp(_) => Some(XType::Xbool),

        _ => None,
    }
}

/// Map a comparison condition to the XType of its operands.
fn cond_operand_xtype(cond: &Condition) -> Option<XType> {
    match cond {
        // Signed 32-bit comparison
        Condition::Ccomp(_) | Condition::Ccompimm(_, _) => Some(XType::Xint),
        // Unsigned 32-bit comparison
        Condition::Ccompu(_) | Condition::Ccompuimm(_, _) => Some(XType::Xintunsigned),
        // Signed 64-bit comparison
        Condition::Ccompl(_) => Some(XType::Xlong),
        Condition::Ccomplimm(_, 0) => Some(XType::Xany64), // possible NULL check
        Condition::Ccomplimm(_, _) => Some(XType::Xlong),
        // Unsigned 64-bit comparison
        Condition::Ccomplu(_) => Some(XType::Xlongunsigned),
        Condition::Ccompluimm(_, 0) => Some(XType::Xany64), // possible NULL check
        Condition::Ccompluimm(_, _) => Some(XType::Xlongunsigned),
        // Float comparison
        Condition::Ccompf(_) | Condition::Cnotcompf(_) => Some(XType::Xfloat),
        // Single comparison
        Condition::Ccompfs(_) | Condition::Cnotcompfs(_) => Some(XType::Xsingle),
        _ => None,
    }
}

/// Returns true if the operation CONSUMES f64 operands; direction-aware: Ointoffloat/Olongoffloat/Osingleoffloat have f64 args but non-f64 dsts, while int->float and single-precision ops must be excluded.
fn op_consumes_f64(op: &Operation) -> bool {
    use Operation::*;
    matches!(
        op,
        Onegf
            | Oabsf
            | Oaddf
            | Osubf
            | Omulf
            | Odivf
            | Omaxf
            | Ominf
            | Ointoffloat
            | Olongoffloat
            | Osingleoffloat
    )
}

/// Returns true if the condition compares f64 operands; single-precision compares excluded for the same width reason as op_consumes_f64.
fn is_f64_cond(cond: &Condition) -> bool {
    matches!(cond, Condition::Ccompf(_) | Condition::Cnotcompf(_))
}

/// True for a 32-bit integer comparison, positive counter-evidence that a value is an int rather than a pointer; the 64-bit forms are excluded since they apply to pointers too.
fn is_int_cmp_cond(cond: &Condition) -> bool {
    matches!(
        cond,
        Condition::Ccomp(_)
            | Condition::Ccompu(_)
            | Condition::Ccompimm(_, _)
            | Condition::Ccompuimm(_, _)
    )
}

// A compare-against-zero, excluded from hard-int evidence because a 32-bit cmp reg,0 could be a NULL check on a truncated value.
fn is_null_cmp_cond(cond: &Condition) -> bool {
    matches!(
        cond,
        Condition::Ccompimm(Comparison::Ceq, 0)
            | Condition::Ccompuimm(Comparison::Ceq, 0)
            | Condition::Ccompimm(Comparison::Cne, 0)
            | Condition::Ccompuimm(Comparison::Cne, 0)
    )
}

// Any pointer-flavored XType, used to single out candidates that must not forward across a register-reuse move into a hard-int destination.
fn is_pointer_xtype(xt: &XType) -> bool {
    matches!(
        xt,
        XType::Xptr
            | XType::Xcharptr
            | XType::Xcharptrptr
            | XType::Xintptr
            | XType::Xfloatptr
            | XType::Xsingleptr
            | XType::Xfuncptr
            | XType::XstructPtr(_)
    )
}

ascent_par! {
    #![measure_rule_times]

    #[swap_db]
    pub struct TypePassProgram;

    // Input relations (swapped from DecompileDB)

    relation emit_var_type_candidate(RTLReg, XType);
    relation win64_home_slot_type(RTLReg, XType);
    relation alias_edge(RTLReg, RTLReg);
    relation stack_var_chunk(Address, i64, MemoryChunk);
    relation stack_var(Address, Address, i64, RTLReg);
    relation ltl_inst(Node, LTLInst);
    relation rtl_inst(Node, RTLInst);
    relation reg_rtl(Node, Mreg, RTLReg);

    relation func_param_position_type(Address, usize, XType);
    relation emit_function(Address, Symbol, Node);
    relation emit_function_return_type_xtype_candidate(Address, XType);
    relation emit_function_return(Address, RTLReg);
    relation func_returns_float(Address);
    relation call_target_func(Node, Address);

    relation string_data(String, String, usize);
    relation ident_to_symbol(Ident, Symbol);

    relation known_func_param_is_ptr(Symbol, usize);
    relation known_func_returns_ptr(Symbol);
    relation known_func_returns_long(Symbol);
    relation known_extern_signature(Symbol, usize, XType, Arc<Vec<XType>>);

    relation call_site(Node, Symbol);
    relation call_arg(Node, usize, RTLReg);
    relation call_return_reg(Node, RTLReg);
    relation call_arg_mapping(Node, usize, RTLReg);
    relation abi_shared_arg_slots(bool);
    relation msvc_gs_cookie_guard_call(Node, Address, Node, Node, RTLReg);

    // Derive call_site from call_target_func + emit_function (internal calls)
    call_site(node, *name) <--
        call_target_func(node, target),
        emit_function(target, name, _);

    // Derive call_arg from call_arg_mapping
    call_arg(node, pos, reg) <--
        call_arg_mapping(node, pos, reg);

    // The authenticated MSVC cookie checker has a per-call ABI contract that
    // is independent of any global symbol or local target prototype.
    emit_var_type_candidate(argument, XType::Xlong) <--
        msvc_gs_cookie_guard_call(_, _, _, _, argument);


    // 1. Direct type emission from instructions; each instruction encodes width and signedness, emitting concrete types directly

    // From operations: op encodes the full output type
    emit_var_type_candidate(rtl_reg, xtype) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if let Some(xtype) = op_output_xtype(op),
        reg_rtl(node, *dst_mreg, rtl_reg);

    // From load chunks, suppressed when the value demonstrably holds a float: an xmm spill/reload uses a generic-width chunk, so tagging it Xint pollutes the float candidate with a truncating sibling.
    emit_var_type_candidate(rtl_reg, chunk_xtype(chunk)) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg),
        !value_float_type(rtl_reg, _);

    // From store chunks, width-preserving only (a sub-word store narrows the cell, not the source); integer chunks are suppressed when the stored value demonstrably holds a float.
    emit_var_type_candidate(rtl_reg, chunk_xtype(chunk)) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MFloat32 | MemoryChunk::MFloat64),
        reg_rtl(node, *src_mreg, rtl_reg);
    emit_var_type_candidate(rtl_reg, chunk_xtype(chunk)) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, _, _, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt32 | MemoryChunk::MAny32 | MemoryChunk::MInt64 | MemoryChunk::MAny64),
        reg_rtl(node, *src_mreg, rtl_reg),
        !value_float_type(rtl_reg, _);

    // From stack variable chunks, suppressing a narrow chunk's candidate when a strictly wider access exists at the same offset, since a reused slot would otherwise truncate the wide reg.
    relation stack_slot_has_wider_chunk(Address, i64, MemoryChunk);
    stack_slot_has_wider_chunk(func, ofs, *chunk) <--
        stack_var_chunk(func, ofs, chunk),
        stack_var_chunk(func, ofs, wider),
        if crate::decompile::passes::rtl_pass::chunk_size_bits(wider)
            > crate::decompile::passes::rtl_pass::chunk_size_bits(chunk);

    emit_var_type_candidate(rtl_reg, chunk_xtype(chunk)) <--
        stack_var(func, _, ofs, rtl_reg),
        stack_var_chunk(func, ofs, chunk),
        !stack_slot_has_wider_chunk(func, ofs, chunk),
        !value_float_type(rtl_reg, _);

    // From comparison operands: condition encodes operand type
    emit_var_type_candidate(rtl_reg, xtype) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocmp(cond), args, _)),
        if let Some(xtype) = cond_operand_xtype(cond),
        for mreg in args.iter(),
        reg_rtl(node, *mreg, rtl_reg);

    // 2. MOVSD upgrade (TR-2): MAny64 -> Xfloat only when the value demonstrably flows into an f64-consuming position or is produced by an f64-producing op; direction-blind and flow-blind old rules wrongly floated cvttsd2si dsts and missed values reaching float ops only through copies.

    #[local] relation f64_copy_edge(RTLReg, RTLReg);
    f64_copy_edge(args[0], *dst) <--
        rtl_inst(_, ?RTLInst::Iop(Operation::Omove, args, dst)),
        if args.len() == 1;

    // Propagate candidates across an Iop(Omove): a forwarded spill/reload is not a stack_var and would otherwise get no candidate at all, defaulting to int and truncating a 64-bit pointer.
    emit_var_type_candidate(*dst, xt.clone()) <--
        rtl_inst(_, ?RTLInst::Iop(Operation::Omove, args, dst)),
        if args.len() == 1,
        let src = args[0],
        emit_var_type_candidate(src, xt),
        if !is_pointer_xtype(xt),
        // Canonical Win64 home cells have an exact access-signature type.
        // In particular, do not recursively copy a narrower peer candidate
        // back through a rewritten spill Omove into the backing cell.
        !win64_home_slot_type(*dst, _);

    // Forward a POINTER candidate across a move only when the destination is not hard 32-bit-int, or the integer half of a reused register gets mistyped void* and its arithmetic corrupted.
    emit_var_type_candidate(*dst, xt.clone()) <--
        rtl_inst(_, ?RTLInst::Iop(Operation::Omove, args, dst)),
        if args.len() == 1,
        let src = args[0],
        emit_var_type_candidate(src, xt),
        if is_pointer_xtype(xt),
        !hard_int32(*dst),
        !win64_home_slot_type(*dst, _);

    // Operand positions that consume an f64 value.
    #[local] relation f64_use_reg(RTLReg);
    f64_use_reg(*arg) <--
        rtl_inst(_, ?RTLInst::Iop(op, args, _)),
        if op_consumes_f64(op),
        for arg in args.iter();
    f64_use_reg(*arg) <--
        rtl_inst(_, ?RTLInst::Iop(Operation::Ocmp(cond), args, _)),
        if is_f64_cond(cond),
        for arg in args.iter();
    f64_use_reg(*arg) <--
        rtl_inst(_, ?RTLInst::Icond(cond, args, _, _)),
        if is_f64_cond(cond),
        for arg in args.iter();

    #[local] relation flows_to_f64_use(RTLReg);
    flows_to_f64_use(reg) <-- f64_use_reg(reg);
    flows_to_f64_use(src) <-- f64_copy_edge(src, dst), flows_to_f64_use(dst);

    #[local] relation holds_f64_value(RTLReg);
    holds_f64_value(dst) <--
        rtl_inst(_, ?RTLInst::Iop(op, _, dst)),
        if is_float_operation(op);
    holds_f64_value(dst) <-- f64_copy_edge(src, dst), holds_f64_value(src);

    // RC-1 float/pointer brake: suppress a bare-float candidate on a register with pointer evidence, since opening the Z3 float axis renders the pointer as double; genuine floats are never is_ptr.
    emit_var_type_candidate(reg, XType::Xfloat) <--
        emit_var_type_candidate(reg, ?XType::Xany64),
        flows_to_f64_use(reg), !is_ptr(reg);
    emit_var_type_candidate(reg, XType::Xfloat) <--
        emit_var_type_candidate(reg, ?XType::Xany64),
        holds_f64_value(reg), !is_ptr(reg);

    // 2b. Float VALUE web: value_float_type carries the precise Xsingle/Xfloat a value reg holds, propagated over verbatim Omove copies only, never across arithmetic that could reinterpret bits.
    #[local] relation value_float_type(RTLReg, XType);
    value_float_type(dst, XType::Xfloat) <--
        rtl_inst(_, ?RTLInst::Iop(op, _, dst)),
        if is_float_operation(op);
    value_float_type(dst, XType::Xsingle) <--
        rtl_inst(_, ?RTLInst::Iop(op, _, dst)),
        if crate::x86::types::is_single_operation(op);
    value_float_type(rtl_reg, XType::Xfloat) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MFloat64),
        reg_rtl(node, *dst_mreg, rtl_reg);
    value_float_type(rtl_reg, XType::Xsingle) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, _, _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MFloat32),
        reg_rtl(node, *dst_mreg, rtl_reg);
    // Synthetic RTL loads carry no ltl_inst, so derive the same float evidence directly from rtl_inst, whose dst is the post-lowering value reg the backend consumes.
    value_float_type(*dst, XType::Xfloat) <--
        rtl_inst(_, ?RTLInst::Iload(chunk, _, _, dst)),
        if matches!(chunk, MemoryChunk::MFloat64);
    value_float_type(*dst, XType::Xsingle) <--
        rtl_inst(_, ?RTLInst::Iload(chunk, _, _, dst)),
        if matches!(chunk, MemoryChunk::MFloat32);
    // Operands of a floating conditional branch are floats; keyed on rtl_inst so synthetic branches count, since cond_operand_xtype covers only Ocmp and misses a param compared by a branch.
    value_float_type(*arg, XType::Xsingle) <--
        rtl_inst(_, ?RTLInst::Icond(cond, args, _, _)),
        if matches!(cond, Condition::Ccompfs(_) | Condition::Cnotcompfs(_)),
        for arg in args.iter();
    value_float_type(*arg, XType::Xfloat) <--
        rtl_inst(_, ?RTLInst::Icond(cond, args, _, _)),
        if matches!(cond, Condition::Ccompf(_) | Condition::Cnotcompf(_)),
        for arg in args.iter();
    // Verbatim-copy propagation in both directions: src and dst of an Omove are the same value, so floatness of either end carries to the other.
    value_float_type(dst, xt) <-- f64_copy_edge(src, dst), value_float_type(src, xt);
    value_float_type(src, xt) <-- f64_copy_edge(src, dst), value_float_type(dst, xt);

    // A stack slot accessed with a float chunk holds a floating value, so its canonical reg keeps float-ness across a non-forwarded spill/reload the Omove-only edge misses. LOOSE candidate.
    value_float_type(rtl_reg, XType::Xsingle) <--
        stack_var_chunk(func, ofs, chunk),
        if matches!(chunk, MemoryChunk::MFloat32),
        stack_var(func, _, ofs, rtl_reg);
    value_float_type(rtl_reg, XType::Xfloat) <--
        stack_var_chunk(func, ofs, chunk),
        if matches!(chunk, MemoryChunk::MFloat64),
        stack_var(func, _, ofs, rtl_reg);

    // A value reg with float evidence gets the float candidate so the solver keeps only float types, except on a pointer-evidenced register, where the bare-float candidate is suppressed.
    emit_var_type_candidate(reg, xt) <-- value_float_type(reg, xt), !is_ptr(reg);


    // 3. Pointer evidence (no is_not_ptr; pointers are a subtype of 8-byte int, no conflict)

    relation is_ptr(RTLReg);
    relation is_char_ptr(RTLReg);
    relation must_be_ptr(RTLReg);

    // HARD 32-bit-int evidence on a value-web node: a 32-bit compare operand is never a pointer, so it stops is_ptr propagating in; must_be_ptr is excluded so a hard pointer constraint wins.
    relation hard_int32(RTLReg);
    hard_int32(rtl_arg) <--
        ltl_inst(addr, ?LTLInst::Lcond(cond, mregs, _, _)),
        if is_int_cmp_cond(cond),
        if !is_null_cmp_cond(cond),
        for mreg in mregs.iter(),
        reg_rtl(addr, *mreg, rtl_arg),
        !must_be_ptr(rtl_arg);

    // Positively type a hard-int register as a 32-bit int, or a register that is only ever a compare operand carries no candidate at all and falls to the Xany64 void* default.
    emit_var_type_candidate(reg, XType::Xint) <-- hard_int32(reg);

    // HARD integer evidence: the destination of a float->int conversion is definitionally an integer, so suppress its Xptr candidate; must_be_ptr is excluded so a genuine constraint still wins.
    relation hard_int_conv_dest(RTLReg);
    hard_int_conv_dest(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(op, _, dst_mreg)),
        if matches!(op, Operation::Olongoffloat | Operation::Olongofsingle |
            Operation::Ointoffloat | Operation::Ointofsingle),
        reg_rtl(node, *dst_mreg, rtl_reg),
        !must_be_ptr(rtl_reg);

    // Pointer-producing operations
    relation op_produces_ptr(Node, RTLReg);
    relation base_addr_usage(Node, RTLReg, i64);

    is_ptr(reg) <-- op_produces_ptr(_, reg);
    is_ptr(reg) <-- base_addr_usage(_, reg, _);

    must_be_ptr(reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, reg),
        known_func_param_is_ptr(func_name, arg_idx),
        !msvc_gs_cookie_guard_call(node, _, _, _, _);

    must_be_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        known_func_returns_ptr(func_name);

    must_be_ptr(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg);

    must_be_ptr(callee_rtl) <--
        ltl_inst(node, ?LTLInst::Lcall(Either::Left(mreg))),
        reg_rtl(node, *mreg, callee_rtl);

    must_be_ptr(callee_rtl) <--
        ltl_inst(node, ?LTLInst::Ltailcall(Either::Left(mreg))),
        reg_rtl(node, *mreg, callee_rtl);

    is_ptr(reg) <-- must_be_ptr(reg);

    // From extern signatures
    is_ptr(arg_reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    is_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        known_extern_signature(func_name, _, ret_type, _),
        if matches!(ret_type, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    is_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        known_func_returns_ptr(func_name);

    // From internal function signatures
    relation internal_func_signature(Symbol, usize, XType);

    internal_func_signature(*name, *pos, *xtype) <--
        func_param_position_type(func_start, pos, xtype),
        emit_function(func_start, name, _);

    is_ptr(arg_reg) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        internal_func_signature(func_name, arg_idx, xtype),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if matches!(xtype, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    // From load/store base address
    is_ptr(base_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(_, addr, args, dst)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *dst,
        reg_rtl(node, base_mreg, base_rtl);

    is_ptr(base_rtl) <--
        ltl_inst(node, ?LTLInst::Lstore(_, addr, args, src)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *src,
        reg_rtl(node, base_mreg, base_rtl);

    is_ptr(base_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(_), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, base_rtl);

    // RTL load/store base address, for synthetic/fused derefs carrying no ltl_inst; without it a pointer misses is_ptr and a spurious float candidate survives into a (double)ptr variant.
    is_ptr(base) <--
        rtl_inst(_, ?RTLInst::Iload(_, addr, args, dst)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base = args[0],
        if base != *dst;

    is_ptr(base) <--
        rtl_inst(_, ?RTLInst::Istore(_, addr, args, src)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base = args[0],
        if base != *src;

    // Char pointer from byte load/store
    is_char_ptr(base_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned | MemoryChunk::MBool),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    is_char_ptr(base_rtl) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned | MemoryChunk::MBool),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    // String literal -> char pointer
    relation string_symbol(String);
    string_symbol(label.clone()) <-- string_data(label, _, _);

    is_char_ptr(rtl_reg) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oindirectsymbol(sym_id), _, dst_mreg)),
        reg_rtl(node, *dst_mreg, rtl_reg),
        ident_to_symbol(sym_id, label),
        string_symbol(label.to_string());

    // Pointer element type (for typed pointer derivation)
    relation ptr_element_type(RTLReg, XType);
    relation has_ptr_element_conflict(RTLReg);

    ptr_element_type(base_rtl, XType::Xint) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MInt32 | MemoryChunk::MAny32),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    ptr_element_type(base_rtl, XType::Xfloat) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MFloat64),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    ptr_element_type(base_rtl, XType::Xsingle) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MFloat32),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    // The int element type from a 32-bit store is suppressed when the stored value holds a float, since a mis-resolved movss chunk would type the pointer int* and truncate the store.
    ptr_element_type(base_rtl, XType::Xint) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, addr, args, src_mreg)),
        if matches!(chunk, MemoryChunk::MInt32 | MemoryChunk::MAny32),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, *src_mreg, src_rtl),
        !value_float_type(src_rtl, _),
        reg_rtl(node, base_mreg, base_rtl);

    ptr_element_type(base_rtl, XType::Xfloat) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MFloat64),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    ptr_element_type(base_rtl, XType::Xsingle) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, addr, args, _)),
        if matches!(chunk, MemoryChunk::MFloat32),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        reg_rtl(node, base_mreg, base_rtl);

    // Float element type from the STORED VALUE: when the chunk was mis-resolved to an integer width but the value is float, the pointer still points at a float, recovering float*/double* out-params.
    ptr_element_type(base_rtl, xt) <--
        ptr_store(base_rtl, src_rtl),
        value_float_type(src_rtl, xt);

    // From extern signatures
    ptr_element_type(arg_reg, XType::Xint) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if params[*arg_idx] == XType::Xintptr;

    ptr_element_type(arg_reg, XType::Xfloat) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if params[*arg_idx] == XType::Xfloatptr;

    ptr_element_type(arg_reg, XType::Xsingle) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if params[*arg_idx] == XType::Xsingleptr;

    // ptr_element_type alias propagation lives here (is_ptr/is_char_ptr live in rtl_pass).
    ptr_element_type(b, ty) <-- ptr_element_type(a, ty), alias_edge(a, b);
    ptr_element_type(a, ty) <-- ptr_element_type(b, ty), alias_edge(a, b);

    has_ptr_element_conflict(reg) <-- ptr_element_type(reg, a), ptr_element_type(reg, b), if a != b;

    relation has_int_ptr_type(RTLReg);
    relation has_float_ptr_type(RTLReg);
    relation has_single_ptr_type(RTLReg);

    has_int_ptr_type(reg) <-- is_ptr(reg), ptr_element_type(reg, ?XType::Xint), !has_ptr_element_conflict(reg);
    has_float_ptr_type(reg) <-- is_ptr(reg), ptr_element_type(reg, ?XType::Xfloat), !has_ptr_element_conflict(reg);
    has_single_ptr_type(reg) <-- is_ptr(reg), ptr_element_type(reg, ?XType::Xsingle), !has_ptr_element_conflict(reg);


    // 4. Pointer deref: tracks what a pointer dereferences to

    relation ptr_deref(RTLReg, RTLReg);
    // Directed relations: track store sources and load destinations separately
    relation ptr_store(RTLReg, RTLReg); // ptr_store(ptr, src): *ptr = src
    relation ptr_load(RTLReg, RTLReg);  // ptr_load(ptr, dst): dst = *ptr
    // Chunked variants: store->load propagation verifies same access width.
    relation ptr_store_chunk(RTLReg, RTLReg, MemoryChunk);
    relation ptr_load_chunk(RTLReg, RTLReg, MemoryChunk);

    // Load: dst = *p
    ptr_deref(base_rtl, dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(_, addr, args, dst_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *dst_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl);

    ptr_load(base_rtl, dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(_, addr, args, dst_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *dst_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl);

    ptr_load_chunk(base_rtl, dst_rtl, *chunk) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, addr, args, dst_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *dst_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl);

    // Store: *p = src
    ptr_deref(base_rtl, src_rtl) <--
        ltl_inst(node, ?LTLInst::Lstore(_, addr, args, src_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *src_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *src_mreg, src_rtl);

    ptr_store(base_rtl, src_rtl) <--
        ltl_inst(node, ?LTLInst::Lstore(_, addr, args, src_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *src_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *src_mreg, src_rtl);

    ptr_store_chunk(base_rtl, src_rtl, *chunk) <--
        ltl_inst(node, ?LTLInst::Lstore(chunk, addr, args, src_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        let base_mreg = args[0],
        if base_mreg != *src_mreg,
        reg_rtl(node, base_mreg, base_rtl),
        reg_rtl(node, *src_mreg, src_rtl);

    // Aliased pointers share deref targets (invariance for mutable pointers)
    ptr_deref(q, x) <-- ptr_deref(p, x), alias_edge(p, q);
    ptr_deref(p, x) <-- ptr_deref(q, x), alias_edge(p, q);
    ptr_store(q, x) <-- ptr_store(p, x), alias_edge(p, q);
    ptr_store(p, x) <-- ptr_store(q, x), alias_edge(p, q);
    ptr_load(q, x) <-- ptr_load(p, x), alias_edge(p, q);
    ptr_load(p, x) <-- ptr_load(q, x), alias_edge(p, q);
    ptr_store_chunk(q, x, c) <-- ptr_store_chunk(p, x, c), alias_edge(p, q);
    ptr_store_chunk(p, x, c) <-- ptr_store_chunk(q, x, c), alias_edge(p, q);
    ptr_load_chunk(q, x, c) <-- ptr_load_chunk(p, x, c), alias_edge(p, q);
    ptr_load_chunk(p, x, c) <-- ptr_load_chunk(q, x, c), alias_edge(p, q);

    // Store->load type propagation under a matching chunk guard; a bare-float candidate is not propagated onto a pointer-evidenced loaded value, but still flows to non-ptr loads.
    emit_var_type_candidate(dst, ty) <--
        emit_var_type_candidate(src, ty),
        ptr_store_chunk(p, src, chunk),
        ptr_load_chunk(p, dst, chunk),
        if !matches!(ty, XType::Xfloat | XType::Xsingle);
    emit_var_type_candidate(dst, ty) <--
        emit_var_type_candidate(src, ty),
        ptr_store_chunk(p, src, chunk),
        ptr_load_chunk(p, dst, chunk),
        if matches!(ty, XType::Xfloat | XType::Xsingle),
        !is_ptr(dst);


    // 6. Emit type candidates from known function signatures (extern and internal) into emit_var_type_candidate

    // Extern call arg: emit param type as candidate
    emit_var_type_candidate(arg_reg, params[*arg_idx].clone()) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if params[*arg_idx] != XType::Xany32 && params[*arg_idx] != XType::Xany64;

    // Extern call return: emit return type as candidate
    emit_var_type_candidate(ret_reg, *ret_type) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        known_extern_signature(func_name, _, ret_type, _),
        if *ret_type != XType::Xvoid && *ret_type != XType::Xany32 && *ret_type != XType::Xany64;

    // Internal function arg: never forward a bare-float param type across the integer-arg boundary, since call_arg carries only integer-register args and a spurious float opens the Z3 float axis.
    emit_var_type_candidate(arg_reg, *xtype) <--
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        internal_func_signature(func_name, arg_idx, xtype),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *xtype != XType::Xany32 && *xtype != XType::Xany64,
        if !matches!(xtype, XType::Xfloat | XType::Xsingle);

    emit_var_type_candidate(arg_reg, *xtype) <--
        abi_shared_arg_slots(true),
        call_site(node, func_name),
        call_arg(node, arg_idx, arg_reg),
        internal_func_signature(func_name, arg_idx, xtype),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if matches!(xtype, XType::Xfloat | XType::Xsingle);

    // Internal function return types
    relation internal_func_return_type(Symbol, XType);

    internal_func_return_type(*name, *xtype) <--
        emit_function_return_type_xtype_candidate(func_start, xtype),
        emit_function(func_start, name, _);

    // Internal call return: emit return type as candidate
    emit_var_type_candidate(ret_reg, *xtype) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        internal_func_return_type(func_name, xtype),
        if *xtype != XType::Xvoid && *xtype != XType::Xany32 && *xtype != XType::Xany64;

    // Internal pointer-returning functions: propagate is_ptr to call return registers
    is_ptr(ret_reg) <--
        call_site(node, func_name),
        call_return_reg(node, ret_reg),
        internal_func_return_type(func_name, xtype),
        if matches!(xtype, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    // Function's own return register: emit return type as candidate
    emit_var_type_candidate(ret_reg, *xtype) <--
        emit_function_return(func_start, ret_reg),
        emit_function_return_type_xtype_candidate(func_start, xtype),
        if *xtype != XType::Xvoid && *xtype != XType::Xany32 && *xtype != XType::Xany64;

    // BUG1a: re-derive the float RETURN-TYPE candidate here, where value_float_type is available, since at rtl time the reloaded return reg has none and the ladder falls to Xany64/long.
    emit_function_return_type_xtype_candidate(func_start, *xt) <--
        func_returns_float(func_start),
        emit_function_return(func_start, ret_reg),
        value_float_type(ret_reg, xt),
        if matches!(xt, XType::Xfloat | XType::Xsingle);


    // 7. Pointer subtype emission from pointer evidence (base types come from section 1 instructions)

    emit_var_type_candidate(reg, XType::Xcharptr) <-- is_char_ptr(reg);
    emit_var_type_candidate(reg, XType::Xintptr) <-- has_int_ptr_type(reg);
    emit_var_type_candidate(reg, XType::Xfloatptr) <-- has_float_ptr_type(reg);
    emit_var_type_candidate(reg, XType::Xsingleptr) <-- has_single_ptr_type(reg);
    // Suppress the bare void* candidate on a register with hard 32-bit-int or float->int-conversion evidence unless it is must_be_ptr, since Xptr would outrank the int candidate.
    emit_var_type_candidate(reg, XType::Xptr) <-- is_ptr(reg), !hard_int32(reg), !hard_int_conv_dest(reg), !must_be_ptr(reg);
    emit_var_type_candidate(reg, XType::Xptr) <-- is_ptr(reg), must_be_ptr(reg);

    // Disabled: bidirectional int/ptr pollution causes every int to become ptr candidate

    // 7b. Pointer arithmetic propagation: add/sub on a known pointer produces a pointer, each gated !hard_int32 so an integer expression lowered to an Olea is not made a pointer.
    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Olea(addr), args, dst_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oleal(addr), args, dst_mreg)),
        if matches!(addr, Addressing::Aindexed(_) | Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // &local (lea with Ainstack and no register base) is unconditionally a 64-bit pointer, recorded as must_be_ptr so its Xptr survives register reuse and the backend does not emit (int)&local.
    must_be_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(op, args, dst_mreg)),
        if matches!(op, Operation::Olea(Addressing::Ainstack(_)) | Operation::Oleal(Addressing::Ainstack(_))),
        if args.is_empty(),
        reg_rtl(node, *dst_mreg, dst_rtl);

    // Oaddl(ptr, offset) -> result is ptr
    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oaddl, args, dst_mreg)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // Osubl(ptr, offset) -> result is ptr (negative indexing)
    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osubl, args, dst_mreg)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // Oaddlimm(ptr, const) -> result is ptr
    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Oaddlimm(_), args, dst_mreg)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // Osel(cond, typ): conditional select; if either operand is ptr, result is ptr
    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osel(_, _), args, dst_mreg)),
        if args.len() >= 2,
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    is_ptr(dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Osel(_, _), args, dst_mreg)),
        if args.len() >= 2,
        reg_rtl(node, args[1], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // Omove of a pointer yields a pointer, except into a destination with hard 32-bit-int evidence, the bridge by which a reused register's pointer half contaminated its integer half.
    op_produces_ptr(node, dst_rtl) <--
        ltl_inst(node, ?LTLInst::Lop(Operation::Omove, args, dst_mreg)),
        if !args.is_empty(),
        reg_rtl(node, args[0], src_rtl),
        is_ptr(src_rtl),
        reg_rtl(node, *dst_mreg, dst_rtl),
        !hard_int32(dst_rtl);

    // 8. Function pointer detection: registers used as indirect call targets
    emit_var_type_candidate(*callee_reg, XType::Xfuncptr) <--
        rtl_inst(_, ?RTLInst::Icall(_, Either::Left(callee_reg), _, _, _));

    emit_var_type_candidate(*callee_reg, XType::Xfuncptr) <--
        rtl_inst(_, ?RTLInst::Itailcall(_, Either::Left(callee_reg), _));

    // 9. Global pointer type detection via dereference evidence, call constraints, and known sigs.
    relation global_addr_reg(Ident, RTLReg);
    relation emit_global_is_ptr(Ident);
    relation emit_global_is_char_ptr(Ident);
    // Ground-truth pointer evidence from ELF relocations: a data slot relocated to a data target is unconditionally a pointer, which the access-based signals miss for never-dereferenced tables.
    relation pointer_in_data(Address, Address);
    #[local] relation global_value_reg(Ident, RTLReg);
    // Integer counter-evidence: the loaded value participates in an integer comparison or non-pointer arithmetic, fed from the same global_value_reg so it weighs against the pointer signals.
    #[local] relation global_used_as_int(Ident);
    // Unambiguous pointer evidence: the loaded value is dereferenced as a memory base or carries a hard call/indirect-call constraint, outranking integer counter-evidence.
    #[local] relation global_ptr_evidence_strong(Ident);
    // Veto: with int counter-evidence and no strong pointer evidence, the weak pointer signals must not upgrade the global, so a contested counter stays its recovered scalar.
    #[local] relation global_ptr_vetoed(Ident);

    // Propagate global address register through aliases (may discover more via type_pass context)
    global_addr_reg(*ident, b) <-- global_addr_reg(ident, a), alias_edge(a, b);
    global_addr_reg(*ident, a) <-- global_addr_reg(ident, b), alias_edge(a, b);

    // Via Oindirectsymbol (PIC binaries): track the value register loaded from a global
    global_value_reg(*ident, loaded_rtl) <--
        global_addr_reg(ident, addr_rtl),
        ltl_inst(node, ?LTLInst::Lload(chunk, Addressing::Aindexed(0), args, dst_mreg)),
        if matches!(chunk, MemoryChunk::MAny64 | MemoryChunk::MInt64),
        if !args.is_empty(),
        reg_rtl(node, args[0], base_rtl),
        if *addr_rtl == *base_rtl,
        reg_rtl(node, *dst_mreg, loaded_rtl);

    // Via direct Aglobal addressing: track the value register
    global_value_reg(*ident, loaded_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, Addressing::Aglobal(ident, 0), _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MAny64 | MemoryChunk::MInt64),
        reg_rtl(node, *dst_mreg, loaded_rtl);

    global_value_reg(*ident, loaded_rtl) <--
        ltl_inst(node, ?LTLInst::Lload(chunk, Addressing::Abased(ident, 0), _, dst_mreg)),
        if matches!(chunk, MemoryChunk::MAny64 | MemoryChunk::MInt64),
        reg_rtl(node, *dst_mreg, loaded_rtl);

    // Propagate global value reg through aliases
    global_value_reg(*ident, b) <-- global_value_reg(ident, a), alias_edge(a, b);
    global_value_reg(*ident, a) <-- global_value_reg(ident, b), alias_edge(a, b);

    // Integer counter-evidence: loaded value is an operand of a 32-bit integer comparison (via Ocmp).
    global_used_as_int(*ident) <--
        global_value_reg(ident, loaded_rtl),
        ltl_inst(node, ?LTLInst::Lop(Operation::Ocmp(cond), args, _)),
        if is_int_cmp_cond(cond),
        for mreg in args.iter(),
        reg_rtl(node, *mreg, loaded_rtl);

    // Integer counter-evidence: loaded value is an operand of a 32-bit integer comparison (via Icond).
    global_used_as_int(*ident) <--
        global_value_reg(ident, loaded_rtl),
        rtl_inst(_, ?RTLInst::Icond(cond, args, _, _)),
        if is_int_cmp_cond(cond),
        if args.contains(loaded_rtl);

    // Integer counter-evidence: the loaded value is an operand of a 32-bit subtract whose counterpart is provably not a pointer, so this is integer subtraction, not pointer difference.
    global_used_as_int(*ident) <--
        global_value_reg(ident, loaded_rtl),
        rtl_inst(_, ?RTLInst::Iop(Operation::Osub, args, _)),
        if args.len() == 2,
        if args[0] == *loaded_rtl,
        let other = args[1],
        !is_ptr(other);

    global_used_as_int(*ident) <--
        global_value_reg(ident, loaded_rtl),
        rtl_inst(_, ?RTLInst::Iop(Operation::Osub, args, _)),
        if args.len() == 2,
        if args[1] == *loaded_rtl,
        let other = args[0],
        !is_ptr(other);

    // Strong pointer evidence: the loaded value is actually dereferenced as a memory base.
    global_ptr_evidence_strong(*ident) <--
        global_value_reg(ident, loaded_rtl),
        ptr_deref(loaded_rtl, _);

    // Strong pointer evidence: the loaded value carries a hard call/indirect-call pointer constraint.
    global_ptr_evidence_strong(*ident) <--
        global_value_reg(ident, loaded_rtl),
        must_be_ptr(loaded_rtl);

    // Veto the weak pointer signals when int counter-evidence holds and strong ptr evidence does not.
    global_ptr_vetoed(*ident) <--
        global_used_as_int(ident),
        !global_ptr_evidence_strong(ident);

    // A global is a pointer if its loaded value is dereferenced (used as memory base)
    emit_global_is_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        ptr_deref(loaded_rtl, _);

    // Also mark as pointer if the loaded value has must_be_ptr evidence (strong call constraint)
    emit_global_is_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        must_be_ptr(loaded_rtl);

    // Relocation ground truth: a data slot relocated to a STRING target is a string pointer; restricted to string targets so a relocation to internal-struct data keeps its baseline recovery.
    emit_global_is_ptr(*slot as usize) <--
        pointer_in_data(slot, target),
        string_data(label, _, _),
        if label.strip_prefix("L_").or_else(|| label.strip_prefix(".L_"))
            .and_then(|h| u64::from_str_radix(h, 16).ok()) == Some(*target);

    // A global is a pointer if passed as a call argument to a known pointer parameter; a weak signal, vetoed under dominant integer counter-evidence.
    emit_global_is_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        call_arg(node, arg_idx, loaded_rtl),
        call_site(node, func_name),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if matches!(params[*arg_idx], XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_)),
        !global_ptr_vetoed(ident);

    emit_global_is_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        call_arg(node, arg_idx, loaded_rtl),
        call_site(node, func_name),
        internal_func_signature(func_name, arg_idx, xtype),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if matches!(xtype, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr |
            XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_)),
        !global_ptr_vetoed(ident);

    emit_global_is_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        call_arg(node, arg_idx, loaded_rtl),
        call_site(node, func_name),
        known_func_param_is_ptr(func_name, arg_idx),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        !global_ptr_vetoed(ident);

    // A global is a char pointer if its loaded value is used to load/store bytes; a weak signal, vetoed under dominant integer counter-evidence.
    emit_global_is_char_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        is_char_ptr(loaded_rtl),
        !global_ptr_vetoed(ident);

    // A global is a char pointer if passed as a char* argument.
    emit_global_is_char_ptr(*ident) <--
        global_value_reg(ident, loaded_rtl),
        call_arg(node, arg_idx, loaded_rtl),
        call_site(node, func_name),
        known_extern_signature(func_name, _, _, params),
        !msvc_gs_cookie_guard_call(node, _, _, _, _),
        if *arg_idx < params.len(),
        if params[*arg_idx] == XType::Xcharptr,
        !global_ptr_vetoed(ident);
}

pub struct TypePass;

impl IRPass for TypePass {
    fn name(&self) -> &'static str {
        "type"
    }

    fn run(&self, db: &mut DecompileDB) {
        let mut prog = TypePassProgram::default();
        prog.swap_db_fields(db);

        prog.run();

        prog.swap_db_fields(db);
        crate::decompile::passes::rtl_pass::enforce_win64_home_types(db);
    }

    fn extra_reads(&self) -> &'static [&'static str] {
        &[
            "win64_home_backing_access",
            "win64_home_backing_selected_candidate",
        ]
    }

    declare_io_from!(TypePassProgram);
}
