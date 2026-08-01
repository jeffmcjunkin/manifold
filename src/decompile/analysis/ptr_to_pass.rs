use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::run_pass;

use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use ascent::ascent_par;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

// Pointer provenance analysis: tracks how each pointer base was derived
ascent_par! {
    #![measure_rule_times]
    #[swap_db]
    pub struct PtrToPassProgram;

    relation rtl_inst(Node, RTLInst);
    relation instr_in_function(Node, Address);
    relation emit_function_param_candidate(Address, RTLReg);
    relation allocation_site(Node, RTLReg, usize);

    // (func, reg, provenance_id)
    relation provenance_root(Address, RTLReg, u64);
    // (func, src_reg, dst_reg, edge_type, offset)
    relation provenance_edge(Address, RTLReg, RTLReg, EdgeType, i64);
    // (func, reg, root_reg, root_provenance_id, accumulated_offset, depth)
    relation provenance_chain(Address, RTLReg, RTLReg, u64, i64, usize);

    // Roots: function params and allocation return values
    provenance_root(func, param_reg, provenance_id) <--
        emit_function_param_candidate(func, param_reg),
        let provenance_id = compute_provenance_id(*func, *param_reg);

    provenance_root(func, result_reg, provenance_id) <--
        instr_in_function(node, func),
        allocation_site(node, result_reg, _),
        let provenance_id = compute_provenance_id(*func, *result_reg);

    // PT-2: 64KiB offset window guards against large immediates (absolute addresses, crypto constants) creating spurious Embed edges into integer registers and mis-typing them as pointers; measured on /bin/ls+/bin/echo, the window dropped zero real chain extensions.
    provenance_edge(func, src, dst, EdgeType::Embed, offset) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Olea(Addressing::Aindexed(offset)), args, dst)),
        if args.len() >= 1,
        if offset.abs() < 65536,
        let src = args[0];

    provenance_edge(func, src, dst, EdgeType::Embed, offset) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Oaddlimm(offset), args, dst)),
        if args.len() >= 1,
        if offset.abs() < 65536,
        let src = args[0];

    // Embed edges from LEA with indexed2 addressing (base + index + offset)
    provenance_edge(func, src, dst, EdgeType::Embed, offset) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Olea(Addressing::Aindexed2(offset)), args, dst)),
        if args.len() >= 1,
        if offset.abs() < 65536,
        let src = args[0];

    // Embed edges from LEA with scaled indexed addressing (base + index*scale + offset)
    provenance_edge(func, src, dst, EdgeType::Embed, offset) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Olea(Addressing::Aindexed2scaled(_, offset)), args, dst)),
        if args.len() >= 1,
        if offset.abs() < 65536,
        let src = args[0];

    // Oaddl/Osubl have non-constant RHS: use Assign (alias-only) so acc_ofs stays honest.
    provenance_edge(func, src, dst, EdgeType::Assign, 0) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Oaddl, args, dst)),
        if args.len() >= 1,
        let src = args[0];

    provenance_edge(func, src, dst, EdgeType::Assign, 0) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Osubl, args, dst)),
        if args.len() >= 1,
        let src = args[0];

    // Deref edges from 64-bit loads (pointer-to-pointer patterns)
    provenance_edge(func, src, dst, EdgeType::Deref, offset) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iload(chunk, Addressing::Aindexed(offset), args, dst)),
        if *chunk == MemoryChunk::MAny64 || *chunk == MemoryChunk::MInt64,
        if args.len() >= 1,
        if offset.abs() < 65536,
        let src = args[0];

    // Assign edges from Omove (register copy)
    provenance_edge(func, src, dst, EdgeType::Assign, 0) <--
        instr_in_function(node, func),
        rtl_inst(node, ?RTLInst::Iop(Operation::Omove, args, dst)),
        if args.len() == 1,
        let src = args[0];

    // PT-3: depth=6 is a termination guarantee, not a precision knob -- provenance_edge contains cycles, so an unbounded depth makes acc_ofs diverge; measured on corpus, depth 6 truncated nothing.
    provenance_chain(func, reg, reg, prov_id, 0, 0) <--
        provenance_root(func, reg, prov_id);

    provenance_chain(func, dst, root, prov_id, acc_ofs + offset, depth + 1) <--
        provenance_chain(func, src, root, prov_id, acc_ofs, depth),
        provenance_edge(func, src, dst, ?EdgeType::Embed, offset),
        if *depth < 6;

    provenance_chain(func, dst, root, prov_id, acc_ofs, depth + 1) <--
        provenance_chain(func, src, root, prov_id, acc_ofs, depth),
        provenance_edge(func, src, dst, ?EdgeType::Assign, _),
        if *depth < 6;

    // A chain has pointer evidence only when a member is actually DEREFERENCED; an Embed edge (LEA) is not standalone evidence, since gcc emits LEA for plain integer arithmetic too.
    #[local] relation chain_has_ptr_evidence(Address, RTLReg, u64);
    chain_has_ptr_evidence(func, root, prov_id) <--
        provenance_chain(func, src, root, prov_id, _, _),
        provenance_edge(func, src, _, ?EdgeType::Deref, _);

    relation emit_var_type_candidate(RTLReg, XType);
    emit_var_type_candidate(reg, XType::Xptr) <--
        provenance_chain(func, reg, root, prov_id, _, _),
        chain_has_ptr_evidence(func, root, prov_id);

    // HARD 32-bit-int evidence: an operand of a genuine 32-bit comparison (excluding the NULL-check form) is never a pointer, so it safely brakes forward root-evidence propagation.
    #[local] relation hard_int32(RTLReg);
    hard_int32(reg) <--
        rtl_inst(_, ?RTLInst::Icond(cond, args, _, _)),
        if is_hard_int_cmp_cond(cond),
        for reg in args.iter();

    // PT-1: an Assign-only copy chain types as pointer when its ROOT carries pointer evidence; forward-only, so the bidirectional int/ptr pollution the type pass disabled cannot arise.
    #[local] relation reg_has_ptr_candidate(RTLReg);
    reg_has_ptr_candidate(reg) <--
        emit_var_type_candidate(reg, xt),
        if matches!(xt, XType::Xptr | XType::Xcharptr | XType::Xcharptrptr | XType::Xintptr
            | XType::Xfloatptr | XType::Xsingleptr | XType::Xfuncptr | XType::XstructPtr(_));

    // Gated !hard_int32: do not carry a pointer root forward into the integer half of a register reused across roles; a genuine pointer member still types via its own deref evidence.
    emit_var_type_candidate(reg, XType::Xptr) <--
        provenance_chain(_, reg, root, _, _, _),
        reg_has_ptr_candidate(root),
        !hard_int32(reg);
}

// A genuine 32-bit comparison that is not the compare-against-0 NULL check; such an operand is an integer, never a pointer.
fn is_hard_int_cmp_cond(cond: &Condition) -> bool {
    let is32 = matches!(
        cond,
        Condition::Ccomp(_)
            | Condition::Ccompu(_)
            | Condition::Ccompimm(_, _)
            | Condition::Ccompuimm(_, _)
    );
    let is_null = matches!(
        cond,
        Condition::Ccompimm(Comparison::Ceq, 0)
            | Condition::Ccompuimm(Comparison::Ceq, 0)
            | Condition::Ccompimm(Comparison::Cne, 0)
            | Condition::Ccompuimm(Comparison::Cne, 0)
    );
    is32 && !is_null
}

// Unique provenance ID from function address and register
pub fn compute_provenance_id(func: Address, reg: RTLReg) -> u64 {
    let mut hasher = DefaultHasher::new();
    func.hash(&mut hasher);
    reg.hash(&mut hasher);
    hasher.finish()
}

const ALLOC_FUNCTIONS: &[&str] = &[
    "malloc",
    "calloc",
    "realloc",
    "reallocarray",
    "aligned_alloc",
    "xmalloc",
    "xcalloc",
    "xrealloc",
    "xreallocarray",
    "ximalloc",
    "xirealloc",
    "xicalloc",
    "xireallocarray",
    "xzalloc",
    "xizalloc",
    "xcharalloc",
    "xnmalloc",
    "xinmalloc",
    "xnrealloc",
    "x2realloc",
    "x2nrealloc",
    "xpalloc",
    "imalloc",
    "irealloc",
    "icalloc",
    "ireallocarray",
    "strdup",
    "strndup",
    "xstrdup",
    "xstrndup",
    "xmemdup",
    "ximemdup",
    "ximemdup0",
];

pub struct PtrToPass;

impl IRPass for PtrToPass {
    fn name(&self) -> &'static str {
        "ptr_to"
    }

    fn run(&self, db: &mut DecompileDB) {
        let alloc_set: HashSet<&str> = ALLOC_FUNCTIONS.iter().copied().collect();

        let call_sites: Vec<(Node, Symbol)> = db
            .rel_iter::<(Node, Symbol)>("call_site")
            .cloned()
            .collect();
        let call_return_regs: Vec<(Node, RTLReg)> = db
            .rel_iter::<(Node, RTLReg)>("call_return_reg")
            .cloned()
            .collect();

        let ret_reg_map: HashMap<Node, RTLReg> = call_return_regs.into_iter().collect();

        for &(node, func_name) in &call_sites {
            if alloc_set.contains(func_name) {
                if let Some(&ret_reg) = ret_reg_map.get(&node) {
                    db.rel_push("allocation_site", (node, ret_reg, 0usize));
                }
            }
        }

        run_pass!(db, PtrToPassProgram);
        emit_provenance_ptr_types(db);
        crate::decompile::passes::rtl_pass::enforce_win64_home_types(db);
    }

    fn inputs(&self) -> &'static [&'static str] {
        static INPUTS: &[&str] = &[
            "rtl_inst",
            "instr_in_function",
            "emit_function_param_candidate",
            "allocation_site",
            "call_site",
            "call_return_reg",
            "emit_var_type_candidate",
            "win64_home_slot_type",
            "win64_home_backing_access",
            "win64_home_backing_selected_candidate",
            "is_ptr",
        ];
        INPUTS
    }

    fn outputs(&self) -> &'static [&'static str] {
        static OUTPUTS: &[&str] = &[
            "allocation_site",
            "emit_var_type_candidate",
            "provenance_root",
            "provenance_edge",
            "provenance_chain",
            "is_ptr",
        ];
        OUTPUTS
    }
}

// Emit pointer types for provenance-tracked registers with consistent deref chunks
fn emit_provenance_ptr_types(db: &mut DecompileDB) {
    let chains: Vec<(Address, RTLReg)> = db
        .rel_iter::<(Address, RTLReg, RTLReg, u64, i64, usize)>("provenance_chain")
        .map(|&(func, reg, _, _, _, _)| (func, reg))
        .collect();

    if chains.is_empty() {
        return;
    }

    let tracked: HashSet<(Address, RTLReg)> = chains.into_iter().collect();

    // Map each tracked register to its provenance root(s) at acc-offset 0; only Assign-only chains let a deref pin the ROOT's pointee type, preserving PT-4's struct caveat.
    let mut reg_roots: HashMap<(Address, RTLReg), Vec<RTLReg>> = HashMap::new();
    for &(func, reg, root, _prov, acc_ofs, _depth) in
        db.rel_iter::<(Address, RTLReg, RTLReg, u64, i64, usize)>("provenance_chain")
    {
        if acc_ofs == 0 && root != reg {
            reg_roots.entry((func, reg)).or_default().push(root);
        }
    }

    let mut reg_deref_chunks: HashMap<RTLReg, Vec<MemoryChunk>> = HashMap::new();
    // Aggregate deref chunks per provenance ROOT, so a char * param recovers its element type through root + i; mixed widths fall back to Xptr.
    let mut root_deref_chunks: HashMap<(Address, RTLReg), Vec<MemoryChunk>> = HashMap::new();

    let instr_func: HashMap<Node, Address> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .map(|&(n, f)| (n, f))
        .collect();

    for &(node, ref inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        let (chunk, base_reg) = match inst {
            RTLInst::Iload(chunk, Addressing::Aindexed(_), args, _) if !args.is_empty() => {
                (*chunk, args[0])
            }
            RTLInst::Istore(chunk, Addressing::Aindexed(_), args, _) if !args.is_empty() => {
                (*chunk, args[0])
            }
            _ => continue,
        };

        if let Some(&func) = instr_func.get(&node) {
            if tracked.contains(&(func, base_reg)) {
                reg_deref_chunks.entry(base_reg).or_default().push(chunk);
                if let Some(roots) = reg_roots.get(&(func, base_reg)) {
                    for &root in roots {
                        root_deref_chunks
                            .entry((func, root))
                            .or_default()
                            .push(chunk);
                    }
                }
            }
        }
    }

    let chunks_to_xtype = |chunks: &[MemoryChunk]| -> XType {
        let first = chunks[0];
        let all_same = chunks.iter().all(|c| *c == first);
        // PT-4: mixed chunks mean a struct base (fields of different widths off one pointer), so falling back to Xptr is semantically correct -- guessing an element pointee type here would be wrong; struct identity is struct_recovery's job keyed on offsets, not this pass's.
        if all_same {
            pointee_xtype_for_chunk(first)
        } else {
            XType::Xptr
        }
    };

    for (reg, chunks) in &reg_deref_chunks {
        if chunks.is_empty() {
            continue;
        }
        db.rel_push("emit_var_type_candidate", (*reg, chunks_to_xtype(chunks)));
    }

    // Root element type from the union of zero-offset deref widths, restricted to CHAR: a wider element would rescale byte displacements into element units and walk off the object.
    for ((_func, root), chunks) in &root_deref_chunks {
        if chunks.is_empty() {
            continue;
        }
        let xt = chunks_to_xtype(chunks);
        if xt == XType::Xcharptr {
            db.rel_push("emit_var_type_candidate", (*root, xt));
        }
    }
}

// Pointee type implied by a single deref chunk width/class.
fn pointee_xtype_for_chunk(chunk: MemoryChunk) -> XType {
    match chunk {
        MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned | MemoryChunk::MBool => {
            XType::Xcharptr
        }
        MemoryChunk::MInt32 | MemoryChunk::MAny32 => XType::Xintptr,
        MemoryChunk::MFloat64 => XType::Xfloatptr,
        MemoryChunk::MFloat32 => XType::Xsingleptr,
        _ => XType::Xptr,
    }
}
