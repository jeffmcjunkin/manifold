use super::*;
use crate::decompile::disassembly::coff::{
    CoffAddressMap, CoffFunctionMap, CoffRelocationMap, CoffSectionMap,
};

const FUNCTION: Address = 0x1000;
const NODE: Node = 0x1010;
const NEXT: Node = 0x1020;
const BASE: RTLReg = 0x8000_0000_0000_0010;
const INDEX: RTLReg = 0x8000_0000_0000_0020;
const VALUE: RTLReg = 0x8000_0000_0000_0030;
const TEMP: RTLReg = 0x8000_0000_0000_0040;
const PLACEMENT_DEF: RTLReg = 0x8000_0000_0000_0050;
const ROOT_DEF: RTLReg = 0x8000_0000_0000_0060;
const PLACEMENT: Node = NODE - 4;
const TERMINAL: Node = NODE + 4;
const POST_TERMINAL: Node = NODE + 8;
const FINAL_RETURN: Node = NODE + 12;
const MEMORY: Symbol = "stage4_test_memory";
const REGISTER: Symbol = "stage4_test_register";
const EXTRA: Symbol = "stage4_test_extra";
const IMMEDIATE: Symbol = "stage4_test_immediate";

fn coff_map() -> CoffAddressMap {
    CoffAddressMap {
        schema: "manifold.coff-address-map.v1",
        loader_id: "amd64-coff-image-v1",
        architecture: "x86_64-pc-windows-msvc",
        image_base: FUNCTION,
        function_boundary_sidecar_sha256: None,
        sections: vec![CoffSectionMap {
            index: 1,
            name: ".text".into(),
            kind: "Text".into(),
            original_file_offset: Some(0x100),
            original_offset_start: 0,
            original_offset_end: 0x40,
            mapped_va_start: FUNCTION,
            mapped_va_end: FUNCTION + 0x40,
        }],
        functions: vec![CoffFunctionMap {
            original_name: "arbitrary_original".into(),
            provider_name: "arbitrary_provider".into(),
            section_index: 1,
            section_offset: 0,
            original_size: 0x40,
            manifold_size: 0x40,
            mapped_entry: FUNCTION,
            mapped_end: FUNCTION + 0x40,
        }],
        symbols: Vec::new(),
        externs: Vec::new(),
        relocations: Vec::new(),
    }
}

fn instruction(
    prefix: &'static str,
    mnemonic: &'static str,
    first: Symbol,
    second: Symbol,
) -> Cr8RawInstruction {
    (4, prefix, mnemonic, first, second, NO_OP, NO_OP, 0, 0)
}

fn push_instruction(
    db: &mut DecompileDB,
    relation: &'static str,
    node: Node,
    row: Cr8RawInstruction,
) {
    let (size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) = row;
    db.rel_push(
        relation,
        (
            node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1,
        ),
    );
}

type NodeRawInstruction = (
    Node,
    usize,
    &'static str,
    &'static str,
    Symbol,
    Symbol,
    Symbol,
    Symbol,
    usize,
    usize,
);

fn replace_instruction_pair(db: &mut DecompileDB, node: Node, row: Cr8RawInstruction) {
    for relation in ["instruction", "unrefinedinstruction"] {
        let mut rows: Vec<NodeRawInstruction> = db
            .rel_iter::<NodeRawInstruction>(relation)
            .filter_map(|existing| (existing.0 != node).then_some(*existing))
            .collect();
        let (size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) = row;
        rows.push((
            node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1,
        ));
        db.rel_set(
            relation,
            rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage4Fixture {
    Affine,
    Affine32,
    Zero,
}

const STAGE4_ENTRY: Node = FUNCTION + 8;

fn stage4_fixture(kind: Stage4Fixture) -> DecompileDB {
    let mut db = DecompileDB::default();
    db.coff_address_map = Some(coff_map());
    db.rel_push("instr_in_function", (STAGE4_ENTRY, FUNCTION));
    db.rel_push("instr_in_function", (NODE, FUNCTION));
    db.rel_push("instr_in_function", (NEXT, FUNCTION));
    db.rel_push("rtl_succ", (STAGE4_ENTRY, NODE));
    db.rel_push("rtl_succ", (NODE, NEXT));
    db.rel_push("emit_function", (FUNCTION, "stage4_fixture", STAGE4_ENTRY));
    db.rel_push("instruction_address_size", (NODE, 8_u8));

    let (instruction_row, ltl, final_row, defs, uses, bindings, params) = match kind {
        Stage4Fixture::Affine | Stage4Fixture::Affine32 => {
            let width: usize = if matches!(kind, Stage4Fixture::Affine32) {
                4
            } else {
                8
            };
            // Capstone reports LEA's memory-shaped address operand as a read
            // operand even though LEA has no architectural memory effect.
            db.rel_push("decoded_memory_read_operand", (NODE, MEMORY));
            let destination_name = if width == 4 { "EAX" } else { "RAX" };
            db.rel_push(
                "op_indirect",
                (MEMORY, "NONE", "RCX", "RDX", 4_i64, 12_i64, width),
            );
            db.rel_push("op_register", (REGISTER, destination_name));
            (
                instruction("", "LEA", MEMORY, REGISTER),
                LTLInst::Lop(
                    Operation::Olea(Addressing::Aindexed2scaled(4, 12)),
                    Arc::new(vec![Mreg::CX, Mreg::DX]),
                    Mreg::AX,
                ),
                RTLInst::Iop(
                    Operation::Olea(Addressing::Aindexed2scaled(4, 12)),
                    Arc::new(vec![BASE, INDEX]),
                    VALUE,
                ),
                vec![Mreg::AX],
                vec![Mreg::CX, Mreg::DX],
                vec![(Mreg::AX, VALUE), (Mreg::CX, BASE), (Mreg::DX, INDEX)],
                vec![BASE, INDEX],
            )
        }
        Stage4Fixture::Zero => {
            db.rel_push("op_register", (REGISTER, "EAX"));
            db.rel_push("op_register", (EXTRA, "EAX"));
            (
                instruction("", "XOR", REGISTER, EXTRA),
                LTLInst::Lop(
                    Operation::Oxor,
                    Arc::new(vec![Mreg::AX, Mreg::AX]),
                    Mreg::AX,
                ),
                RTLInst::Iop(Operation::Ointconst(0), Arc::new(Vec::new()), VALUE),
                vec![Mreg::AX],
                Vec::new(),
                vec![(Mreg::AX, VALUE)],
                Vec::new(),
            )
        }
    };
    push_instruction(&mut db, "instruction", NODE, instruction_row.clone());
    push_instruction(&mut db, "unrefinedinstruction", NODE, instruction_row);
    for register in defs {
        db.rel_push("decoded_reg_def", (NODE, register));
    }
    for register in uses {
        db.rel_push("decoded_reg_use", (NODE, register));
    }
    db.rel_push("ltl_inst", (NODE, ltl));
    match kind {
        Stage4Fixture::Zero => {
            db.rel_push(
                "rtl_inst_candidate",
                (
                    NODE,
                    RTLInst::Iop(Operation::Oxor, Arc::new(vec![VALUE, VALUE]), VALUE),
                ),
            );
        }
        Stage4Fixture::Affine | Stage4Fixture::Affine32 => {
            db.rel_push("rtl_inst_candidate", (NODE, final_row.clone()));
        }
    }
    db.rel_push("rtl_inst", (NODE, final_row));
    db.rel_push("rtl_inst", (NEXT, RTLInst::Ireturn(VALUE)));
    for (register, value) in bindings {
        db.rel_push("reg_rtl", (NODE, register, value));
    }
    for parameter in params {
        db.rel_push("emit_function_param_candidate", (FUNCTION, parameter));
    }
    db
}

fn stage4_compound_fixture(
    mnemonic: &'static str,
    width: usize,
    source_immediate: Option<(i64, usize)>,
) -> DecompileDB {
    stage4_compound_fixture_with_terminal(mnemonic, width, source_immediate, false)
}

fn scalar_stage4_access(
    node: Node,
    direction: ScalarMemoryDirection,
    width: usize,
    base_register: Mreg,
    base_value: RTLReg,
    value: RTLReg,
) -> ScalarMemoryAccessProof {
    let chunk = if width == 4 {
        MemoryChunk::MInt32
    } else {
        MemoryChunk::MAny64
    };
    ScalarMemoryAccessProof {
        function: FUNCTION,
        origin_node: node,
        selected_node: node,
        operand: MEMORY,
        direction,
        extension: ScalarMemoryExtension::Plain,
        encoded_destination_width: (direction == ScalarMemoryDirection::Read).then_some(width),
        result_chain: (direction == ScalarMemoryDirection::Read)
            .then_some(ScalarMemoryResultChain::Direct),
        address_size: 8,
        base_register,
        index_register: None,
        scale: 1,
        displacement: 0,
        width,
        value_width: width,
        downstream_value_width: (direction == ScalarMemoryDirection::Read).then_some(width),
        chunk,
        base_value: Some(base_value),
        index_value: None,
        value,
        address_param_leaves: Arc::new(vec![base_value]),
        synthetic_stack_origin: false,
        exact_scaled_index: false,
    }
}

fn stage4_compound_fixture_with_terminal(
    mnemonic: &'static str,
    width: usize,
    source_immediate: Option<(i64, usize)>,
    store_terminal: bool,
) -> DecompileDB {
    let mut db = DecompileDB::default();
    db.coff_address_map = Some(coff_map());
    let function_entry = PLACEMENT;
    let terminal_nodes: &[Node] = if store_terminal {
        &[TERMINAL, POST_TERMINAL, FINAL_RETURN]
    } else {
        &[TERMINAL]
    };
    for node in std::iter::once(PLACEMENT)
        .chain(std::iter::once(NODE))
        .chain(terminal_nodes.iter().copied())
    {
        db.rel_push("instr_in_function", (node, FUNCTION));
    }
    db.rel_push("rtl_succ", (PLACEMENT, TERMINAL));
    if store_terminal {
        db.rel_push("rtl_succ", (TERMINAL, POST_TERMINAL));
        db.rel_push("rtl_succ", (POST_TERMINAL, FINAL_RETURN));
    }
    db.rel_push(
        "emit_function",
        (FUNCTION, "stage4_compound_fixture", function_entry),
    );
    db.rel_push("instruction_address_size", (NODE, 8_u8));

    let (source_name, destination_name) = if width == 8 {
        ("RDX", "RAX")
    } else {
        ("EDX", "EAX")
    };
    let destination = Mreg::x86(destination_name);
    let source = Mreg::x86(source_name);
    let (_kind, operation) = match (mnemonic, width, source_immediate) {
        ("ADD", 4, None) => (Stage4SourceKind::Add, Operation::Oadd),
        ("ADD", 8, None) => (Stage4SourceKind::Add, Operation::Oaddl),
        ("SUB", 4, None) => (Stage4SourceKind::Sub, Operation::Osub),
        ("SUB", 8, None) => (Stage4SourceKind::Sub, Operation::Osubl),
        ("IMUL", 4, None) => (Stage4SourceKind::Mul, Operation::Omul),
        ("IMUL", 8, None) => (Stage4SourceKind::Mul, Operation::Omull),
        ("AND", 4, None) => (Stage4SourceKind::And, Operation::Oand),
        ("AND", 8, None) => (Stage4SourceKind::And, Operation::Oandl),
        ("OR", 4, None) => (Stage4SourceKind::Or, Operation::Oor),
        ("OR", 8, None) => (Stage4SourceKind::Or, Operation::Oorl),
        ("XOR", 4, None) => (Stage4SourceKind::Xor, Operation::Oxor),
        ("XOR", 8, None) => (Stage4SourceKind::Xor, Operation::Oxorl),
        ("ADD", 4, Some((value, _))) => (Stage4SourceKind::Add, Operation::Oaddimm(value)),
        ("ADD", 8, Some((value, _))) => (Stage4SourceKind::Add, Operation::Oaddlimm(value)),
        ("SUB", 4, Some((value, _))) => (
            Stage4SourceKind::Sub,
            Operation::Oaddimm(value.checked_neg().unwrap_or_default()),
        ),
        ("SUB", 8, Some((value, _))) => (
            Stage4SourceKind::Sub,
            Operation::Oaddlimm(value.checked_neg().unwrap_or_default()),
        ),
        ("IMUL", 4, Some((value, _))) => (Stage4SourceKind::Mul, Operation::Omulimm(value)),
        ("IMUL", 8, Some((value, _))) => (Stage4SourceKind::Mul, Operation::Omullimm(value)),
        ("AND", 4, Some((value, _))) => (Stage4SourceKind::And, Operation::Oandimm(value)),
        ("AND", 8, Some((value, _))) => (Stage4SourceKind::And, Operation::Oandlimm(value)),
        ("OR", 4, Some((value, _))) => (Stage4SourceKind::Or, Operation::Oorimm(value)),
        ("OR", 8, Some((value, _))) => (Stage4SourceKind::Or, Operation::Oorlimm(value)),
        ("XOR", 4, Some((value, _))) => (Stage4SourceKind::Xor, Operation::Oxorimm(value)),
        ("XOR", 8, Some((value, _))) => (Stage4SourceKind::Xor, Operation::Oxorlimm(value)),
        _ => panic!("unsupported fixture {mnemonic}/{width}"),
    };

    if let Some((immediate, encoded_width)) = source_immediate {
        let immediate_operand = if mnemonic == "IMUL" {
            IMMEDIATE
        } else {
            REGISTER
        };
        db.rel_push(
            "op_immediate",
            (immediate_operand, immediate, encoded_width),
        );
    } else {
        db.rel_push("op_register", (REGISTER, source_name));
    }
    db.rel_push("op_register", (EXTRA, destination_name));
    let instruction_row = if mnemonic == "IMUL" && source_immediate.is_some() {
        (4, "", mnemonic, EXTRA, EXTRA, IMMEDIATE, NO_OP, 0, 0)
    } else {
        instruction("", mnemonic, REGISTER, EXTRA)
    };
    push_instruction(&mut db, "instruction", NODE, instruction_row.clone());
    push_instruction(&mut db, "unrefinedinstruction", NODE, instruction_row);
    db.rel_push("decoded_reg_def", (NODE, destination));
    db.rel_push("decoded_reg_use", (NODE, destination));
    let (rtl_args, ltl_args) = if source_immediate.is_some() {
        (Arc::new(vec![VALUE]), Arc::new(vec![destination]))
    } else {
        db.rel_push("decoded_reg_use", (NODE, source));
        (
            Arc::new(vec![VALUE, INDEX]),
            Arc::new(vec![destination, source]),
        )
    };
    let selected_ltl = LTLInst::Lop(operation.clone(), ltl_args.clone(), destination);
    let selected_rtl = RTLInst::Iop(operation.clone(), rtl_args.clone(), ROOT_DEF);
    db.rel_push("ltl_inst", (NODE, selected_ltl));
    db.rel_push(
        "rtl_inst_candidate",
        (
            NODE,
            RTLInst::Iop(operation.clone(), rtl_args.clone(), VALUE),
        ),
    );
    db.rel_push("rtl_inst_candidate", (NODE, selected_rtl.clone()));
    let companion = if mnemonic == "ADD" && source_immediate.is_none() {
        stage4_addition_operations(width).map(|(_, companion)| companion)
    } else if matches!(mnemonic, "AND" | "OR") {
        stage4_and_or_width_companion(&operation)
    } else {
        None
    };
    if let Some(companion) = companion {
        db.rel_push(
            "ltl_inst",
            (NODE, LTLInst::Lop(companion.clone(), ltl_args, destination)),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iop(companion.clone(), rtl_args.clone(), VALUE),
            ),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (NODE, RTLInst::Iop(companion, rtl_args, ROOT_DEF)),
        );
    }
    let chunk = if width == 4 {
        MemoryChunk::MInt32
    } else {
        MemoryChunk::MAny64
    };
    let placement = RTLInst::Iload(chunk, Addressing::Aindexed(0), Arc::new(vec![BASE]), VALUE);
    let terminal = if store_terminal {
        RTLInst::Istore(chunk, Addressing::Aindexed(0), Arc::new(vec![TEMP]), VALUE)
    } else {
        RTLInst::Ireturn(VALUE)
    };
    db.rel_push("rtl_inst", (PLACEMENT, placement.clone()));
    db.rel_push("rtl_inst", (TERMINAL, terminal.clone()));
    if store_terminal {
        db.rel_push(
            "rtl_inst",
            (
                POST_TERMINAL,
                RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), TEMP),
            ),
        );
        db.rel_push("rtl_inst", (FINAL_RETURN, RTLInst::Ireturn(BASE)));
    }
    db.rel_push("stage4_preopt_selected_rtl", (PLACEMENT, placement));
    db.rel_push("stage4_preopt_selected_rtl", (NODE, selected_rtl));
    db.rel_push("stage4_preopt_selected_rtl", (TERMINAL, terminal));
    db.rel_push("stage4_preopt_selected_succ", (PLACEMENT, NODE));
    db.rel_push("stage4_preopt_selected_succ", (NODE, TERMINAL));
    if store_terminal {
        db.rel_push(
            "stage4_preopt_selected_rtl",
            (
                POST_TERMINAL,
                RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), TEMP),
            ),
        );
        db.rel_push(
            "stage4_preopt_selected_rtl",
            (FINAL_RETURN, RTLInst::Ireturn(BASE)),
        );
        db.rel_push("stage4_preopt_selected_succ", (TERMINAL, POST_TERMINAL));
        db.rel_push("stage4_preopt_selected_succ", (POST_TERMINAL, FINAL_RETURN));
    }
    db.rel_push("stage4_optimizer_eliminated_node", (NODE,));
    db.rel_push("reg_rtl", (NODE, destination, VALUE));
    db.rel_push("reg_rtl", (NODE, destination, ROOT_DEF));
    if source_immediate.is_none() {
        db.rel_push("reg_rtl", (NODE, source, INDEX));
        db.rel_push("reaching_use_rtl", (NODE, source, INDEX));
        db.rel_push("emit_function_param_candidate", (FUNCTION, INDEX));
    }
    db.rel_push("reg_def_used", (PLACEMENT, destination, NODE));
    db.rel_push("reg_def_used", (NODE, destination, TERMINAL));
    db.rel_push("reaching_use_rtl", (NODE, destination, VALUE));
    db.rel_push("is_def", (PLACEMENT, PLACEMENT_DEF));
    db.rel_push("is_def", (NODE, ROOT_DEF));
    db.rel_push("reg_xtl", (PLACEMENT, destination, PLACEMENT_DEF));
    // The real RTL fixed point retains the placement definition at the
    // destructive root's read position alongside its fresh root definition.
    db.rel_push("reg_xtl", (NODE, destination, PLACEMENT_DEF));
    db.rel_push("reg_xtl", (NODE, destination, ROOT_DEF));
    db.rel_push("xtl_canonical", (PLACEMENT_DEF, VALUE));
    db.rel_push("xtl_canonical", (ROOT_DEF, VALUE));
    let placement_access = scalar_stage4_access(
        PLACEMENT,
        ScalarMemoryDirection::Read,
        width,
        Mreg::CX,
        BASE,
        VALUE,
    );
    assert!(placement_access.is_closed_v1());
    db.rel_push(
        "authenticated_scalar_memory_access",
        (PLACEMENT, placement_access),
    );
    if store_terminal {
        let store_access = scalar_stage4_access(
            TERMINAL,
            ScalarMemoryDirection::Write,
            width,
            Mreg::R8,
            TEMP,
            VALUE,
        );
        assert!(store_access.is_closed_v1());
        db.rel_push(
            "authenticated_scalar_memory_access",
            (TERMINAL, store_access),
        );
    }

    let placement_row = instruction("", "MOV", MEMORY, EXTRA);
    push_instruction(&mut db, "instruction", PLACEMENT, placement_row.clone());
    push_instruction(&mut db, "unrefinedinstruction", PLACEMENT, placement_row);
    let terminal_row = if store_terminal {
        instruction("", "MOV", REGISTER, MEMORY)
    } else {
        (1, "", "RET", NO_OP, NO_OP, NO_OP, NO_OP, 0, 0)
    };
    push_instruction(&mut db, "instruction", TERMINAL, terminal_row.clone());
    push_instruction(&mut db, "unrefinedinstruction", TERMINAL, terminal_row);
    if store_terminal {
        let move_row = instruction("", "MOV", REGISTER, EXTRA);
        push_instruction(&mut db, "instruction", POST_TERMINAL, move_row.clone());
        push_instruction(&mut db, "unrefinedinstruction", POST_TERMINAL, move_row);
        let return_row = (1, "", "RET", NO_OP, NO_OP, NO_OP, NO_OP, 0, 0);
        push_instruction(&mut db, "instruction", FINAL_RETURN, return_row.clone());
        push_instruction(&mut db, "unrefinedinstruction", FINAL_RETURN, return_row);
    }
    db
}

fn stage4_zero_fixture(mnemonic: &'static str, width: usize) -> DecompileDB {
    let mut db = DecompileDB::default();
    db.coff_address_map = Some(coff_map());
    for node in [STAGE4_ENTRY, NODE, NEXT] {
        db.rel_push("instr_in_function", (node, FUNCTION));
    }
    db.rel_push("rtl_succ", (STAGE4_ENTRY, NODE));
    db.rel_push("rtl_succ", (NODE, NEXT));
    db.rel_push(
        "emit_function",
        (FUNCTION, "stage4_zero_fixture", STAGE4_ENTRY),
    );
    db.rel_push("instruction_address_size", (NODE, 8_u8));
    let register_name = match width {
        1 => "AL",
        2 => "AX",
        4 => "EAX",
        8 => "RAX",
        _ => panic!("test width"),
    };
    let register = Mreg::x86(register_name);
    db.rel_push("op_register", (REGISTER, register_name));
    db.rel_push("op_register", (EXTRA, register_name));
    let instruction_row = instruction("", mnemonic, REGISTER, EXTRA);
    push_instruction(&mut db, "instruction", NODE, instruction_row.clone());
    push_instruction(&mut db, "unrefinedinstruction", NODE, instruction_row);
    db.rel_push("decoded_reg_def", (NODE, register));
    if mnemonic == "SUB" {
        db.rel_push("decoded_reg_use", (NODE, register));
    }
    let operation = if width == 8 {
        Operation::Olongconst(0)
    } else {
        Operation::Ointconst(0)
    };
    let final_row = RTLInst::Iop(operation.clone(), Arc::new(Vec::new()), VALUE);
    if mnemonic == "XOR" {
        let xor_operation = if width == 8 {
            Operation::Oxorl
        } else {
            Operation::Oxor
        };
        db.rel_push(
            "ltl_inst",
            (
                NODE,
                LTLInst::Lop(
                    xor_operation.clone(),
                    Arc::new(vec![register, register]),
                    register,
                ),
            ),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iop(xor_operation, Arc::new(vec![VALUE, VALUE]), VALUE),
            ),
        );
    } else {
        let sub_operation = if width == 8 {
            Operation::Osubl
        } else {
            Operation::Osub
        };
        db.rel_push(
            "ltl_inst",
            (
                NODE,
                LTLInst::Lop(
                    sub_operation.clone(),
                    Arc::new(vec![register, register]),
                    register,
                ),
            ),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iop(sub_operation, Arc::new(vec![VALUE, VALUE]), VALUE),
            ),
        );
    }
    db.rel_push("rtl_inst", (NODE, final_row));
    db.rel_push("rtl_inst", (NEXT, RTLInst::Ireturn(VALUE)));
    db.rel_push("reg_rtl", (NODE, register, VALUE));
    db
}

fn stage4_proofs(mut db: DecompileDB) -> Vec<Stage4SourceProof> {
    materialize_authenticated_stage4_sources(&mut db);
    db.rel_iter::<(Node, Stage4SourceProof)>("authenticated_stage4_source")
        .map(|(_, proof)| proof.clone())
        .collect()
}

fn stage4_plans(mut db: DecompileDB) -> Vec<Stage4UsePlan> {
    materialize_authenticated_stage4_sources(&mut db);
    db.rel_iter::<(Node, Stage4UsePlan)>("authenticated_stage4_use_plan")
        .map(|(_, plan)| plan.clone())
        .collect()
}

fn set_stage4_preopt_root(db: &mut DecompileDB, instruction: RTLInst) {
    let mut rows: Vec<_> = db
        .rel_iter::<(Node, RTLInst)>("stage4_preopt_selected_rtl")
        .filter_map(|(node, instruction)| (*node != NODE).then_some((*node, instruction.clone())))
        .collect();
    rows.push((NODE, instruction));
    db.rel_set(
        "stage4_preopt_selected_rtl",
        rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

#[test]
fn stage4_machine_proofs_cover_affine_zero_and_live_range_closed_rmw() {
    for (fixture, expected_kind) in [
        (Stage4Fixture::Affine, Stage4SourceKind::AffineAddress),
        (Stage4Fixture::Affine32, Stage4SourceKind::AffineAddress),
        (Stage4Fixture::Zero, Stage4SourceKind::Zeroing),
    ] {
        let proofs = stage4_proofs(stage4_fixture(fixture));
        assert_eq!(proofs.len(), 1, "missing {fixture:?} proof: {proofs:?}");
        assert_eq!(proofs[0].kind, expected_kind);
        if matches!(fixture, Stage4Fixture::Affine32) {
            assert_eq!(proofs[0].width, 4);
        }
        assert!(proofs[0].is_closed_v1());

        let plans = stage4_plans(stage4_fixture(fixture));
        assert_eq!(plans.len(), 1, "missing {fixture:?} plan: {plans:?}");
        assert!(plans[0].is_closed_v1(&proofs[0]));
        assert_eq!(
            plans[0].sites.as_slice(),
            &[Stage4UseSite {
                node: NEXT,
                value: VALUE,
            }]
        );
    }

    for mnemonic in ["ADD", "SUB", "IMUL", "AND", "OR", "XOR"] {
        for width in [4, 8] {
            let proofs = stage4_proofs(stage4_compound_fixture(mnemonic, width, None));
            assert_eq!(proofs.len(), 1, "missing {mnemonic}/{width} register proof");
            assert!(proofs[0].kind.is_compound());
            assert_eq!(proofs[0].source_immediate, None);
        }
    }

    for mnemonic in ["ADD", "SUB", "IMUL", "AND", "OR", "XOR"] {
        for width in [4, 8] {
            let proofs = stage4_proofs(stage4_compound_fixture(mnemonic, width, Some((7, 0))));
            assert_eq!(
                proofs.len(),
                1,
                "missing {mnemonic}/{width} immediate proof"
            );
            assert!(proofs[0].kind.is_compound());
            assert_eq!(proofs[0].source_immediate, Some(7));
            assert_eq!(proofs[0].args.as_slice(), [VALUE]);
        }
    }

    let proofs = stage4_proofs(stage4_compound_fixture("ADD", 4, None));
    let plans = stage4_plans(stage4_compound_fixture("ADD", 4, None));
    assert_eq!(proofs.len(), 1);
    assert_eq!(plans.len(), 1);
    assert_eq!(
        proofs[0].root_boundary,
        Stage4RootBoundary::EliminatedMutation
    );
    assert_eq!(plans[0].placement_node, PLACEMENT);
    assert_eq!(
        plans[0].sites.as_slice(),
        &[Stage4UseSite {
            node: TERMINAL,
            value: VALUE,
        }]
    );
    assert!(plans[0].transports.is_empty());

    // Stage 4 intentionally covers only the common 32/64-bit scalar GPR
    // XOR/SUB zeroing idioms. Byte/word and SIMD zeroing remain explicit
    // exclusions until their partial-register semantics are proved.
    for mnemonic in ["XOR", "SUB"] {
        for width in [4, 8] {
            let proofs = stage4_proofs(stage4_zero_fixture(mnemonic, width));
            assert_eq!(proofs.len(), 1, "missing {mnemonic}/{width} zero proof");
            assert_eq!(proofs[0].kind, Stage4SourceKind::Zeroing);
        }
        for width in [1, 2] {
            assert!(stage4_proofs(stage4_zero_fixture(mnemonic, width)).is_empty());
        }
    }
}

#[test]
fn stage4_immediate_rmw_rejects_ambiguity_width_drift_and_live_old_values() {
    let mut duplicate = stage4_compound_fixture("ADD", 4, Some((7, 0)));
    duplicate.rel_push("op_immediate", (REGISTER, 9_i64, 0_usize));
    assert!(stage4_proofs(duplicate).is_empty());

    let mut cross_kind = stage4_compound_fixture("ADD", 4, Some((7, 0)));
    cross_kind.rel_push("op_register", (REGISTER, "EDX"));
    assert!(stage4_proofs(cross_kind).is_empty());

    let mut cross_indirect = stage4_compound_fixture("ADD", 4, Some((7, 0)));
    cross_indirect.rel_push(
        "op_indirect",
        (REGISTER, "NONE", "RCX", "NONE", 1_i64, 0_i64, 4_usize),
    );
    assert!(stage4_proofs(cross_indirect).is_empty());

    assert!(stage4_proofs(stage4_compound_fixture("ADD", 4, Some((7, 8)))).is_empty());
    assert!(stage4_proofs(stage4_compound_fixture(
        "ADD",
        4,
        Some((i64::from(i32::MAX) + 1, 0)),
    ))
    .is_empty());
    assert!(stage4_proofs(stage4_compound_fixture("SUB", 8, Some((i64::MIN, 0)))).is_empty());

    let mut old_live = stage4_compound_fixture("IMUL", 8, Some((3, 0)));
    old_live.rel_push("rtl_inst", (NEXT + 4, RTLInst::Ireturn(VALUE)));
    old_live.rel_push("instr_in_function", (NEXT + 4, FUNCTION));
    old_live.rel_push("rtl_succ", (NODE, NEXT + 4));
    assert!(stage4_proofs(old_live).is_empty());

    let mut duplicate_ltl = stage4_compound_fixture("XOR", 8, Some((3, 0)));
    duplicate_ltl.rel_push(
        "ltl_inst",
        (
            NODE,
            LTLInst::Lop(Operation::Oxorlimm(3), Arc::new(vec![Mreg::AX]), Mreg::AX),
        ),
    );
    assert!(stage4_proofs(duplicate_ltl).is_empty());

    let mut extra_candidate = stage4_compound_fixture("OR", 4, Some((3, 0)));
    extra_candidate.rel_push(
        "rtl_inst_candidate",
        (
            NODE,
            RTLInst::Iop(Operation::Oxorimm(3), Arc::new(vec![VALUE]), VALUE),
        ),
    );
    assert!(stage4_proofs(extra_candidate).is_empty());

    let xor_singleton = stage4_compound_fixture("XOR", 8, Some((31, 0)));
    assert_eq!(stage4_proofs(xor_singleton).len(), 1);

    let mut extra_xor_ltl_operation = stage4_compound_fixture("XOR", 8, Some((31, 0)));
    extra_xor_ltl_operation.rel_push(
        "ltl_inst",
        (
            NODE,
            LTLInst::Lop(Operation::Oxorimm(31), Arc::new(vec![Mreg::AX]), Mreg::AX),
        ),
    );
    assert!(stage4_proofs(extra_xor_ltl_operation).is_empty());

    let mut missing_xor_result = stage4_compound_fixture("XOR", 8, Some((31, 0)));
    let retained = missing_xor_result
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, instruction)| {
            *node != NODE
                || !matches!(instruction, RTLInst::Iop(_, _, result) if *result == ROOT_DEF)
        })
        .cloned()
        .collect::<ascent::boxcar::Vec<_>>();
    missing_xor_result.rel_set("rtl_inst_candidate", retained);
    assert!(stage4_proofs(missing_xor_result).is_empty());

    let mut extra_xor_operation = stage4_compound_fixture("XOR", 8, Some((31, 0)));
    extra_xor_operation.rel_push(
        "rtl_inst_candidate",
        (
            NODE,
            RTLInst::Iop(Operation::Oxorimm(31), Arc::new(vec![VALUE]), VALUE),
        ),
    );
    assert!(stage4_proofs(extra_xor_operation).is_empty());

    let mut duplicate_xor_result = stage4_compound_fixture("XOR", 8, Some((31, 0)));
    duplicate_xor_result.rel_push(
        "rtl_inst_candidate",
        (
            NODE,
            RTLInst::Iop(Operation::Oxorlimm(31), Arc::new(vec![VALUE]), ROOT_DEF),
        ),
    );
    assert!(stage4_proofs(duplicate_xor_result).is_empty());

    // AND/OR retain their exact LTL width twins, but a twin is evidence only:
    // the deterministic selected row must be the raw-width operation that the
    // proof can represent.
    let mut selected_width_companion = stage4_compound_fixture("AND", 8, Some((255, 0)));
    set_stage4_preopt_root(
        &mut selected_width_companion,
        RTLInst::Iop(Operation::Oandimm(255), Arc::new(vec![VALUE]), ROOT_DEF),
    );
    assert!(stage4_proofs(selected_width_companion).is_empty());
}

#[test]
fn stage4_eliminated_rmw_seals_fixed_point_identity_and_rhs_availability() {
    let mut missing_eliminated = stage4_compound_fixture("ADD", 8, None);
    missing_eliminated.rel_set(
        "stage4_optimizer_eliminated_node",
        ascent::boxcar::Vec::<(Node,)>::new(),
    );
    assert!(stage4_proofs(missing_eliminated).is_empty());

    let mut carrier_copy = stage4_compound_fixture("ADD", 8, None);
    carrier_copy.rel_push("stage4_optimizer_copy_substitution", (VALUE, TEMP));
    assert!(stage4_proofs(carrier_copy).is_empty());

    let mut source_copy = stage4_compound_fixture("ADD", 8, None);
    source_copy.rel_push("stage4_optimizer_copy_substitution", (INDEX, TEMP));
    assert!(stage4_proofs(source_copy).is_empty());

    let mut result_copy_destination = stage4_compound_fixture("ADD", 8, None);
    result_copy_destination.rel_push("stage4_optimizer_copy_substitution", (ROOT_DEF, TEMP));
    assert!(stage4_proofs(result_copy_destination).is_empty());

    let mut result_copy_source = stage4_compound_fixture("ADD", 8, None);
    result_copy_source.rel_push("stage4_optimizer_copy_substitution", (TEMP, ROOT_DEF));
    assert!(stage4_proofs(result_copy_source).is_empty());

    let mut source_not_final_param = stage4_compound_fixture("ADD", 8, None);
    source_not_final_param.rel_set(
        "emit_function_param_candidate",
        ascent::boxcar::Vec::<(Address, RTLReg)>::new(),
    );
    assert!(stage4_proofs(source_not_final_param).is_empty());

    let mut missing_carrier_destination = stage4_compound_fixture("ADD", 8, None);
    let retained = missing_carrier_destination
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, instruction)| {
            *node != NODE || !matches!(instruction, RTLInst::Iop(_, _, result) if *result == VALUE)
        })
        .cloned()
        .collect::<ascent::boxcar::Vec<_>>();
    missing_carrier_destination.rel_set("rtl_inst_candidate", retained);
    assert!(stage4_proofs(missing_carrier_destination).is_empty());
    let mut missing_root_destination = stage4_compound_fixture("ADD", 8, None);
    let retained = missing_root_destination
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, instruction)| {
            *node != NODE
                || !matches!(instruction, RTLInst::Iop(_, _, result) if *result == ROOT_DEF)
        })
        .cloned()
        .collect::<ascent::boxcar::Vec<_>>();
    missing_root_destination.rel_set("rtl_inst_candidate", retained);
    assert!(stage4_proofs(missing_root_destination).is_empty());

    let mut duplicate_root_destination = stage4_compound_fixture("ADD", 8, None);
    duplicate_root_destination.rel_push(
        "rtl_inst_candidate",
        (
            NODE,
            RTLInst::Iop(Operation::Oaddl, Arc::new(vec![VALUE, INDEX]), ROOT_DEF),
        ),
    );
    assert!(stage4_proofs(duplicate_root_destination).is_empty());

    let mut wrong_selected_destination = stage4_compound_fixture("ADD", 8, None);
    set_stage4_preopt_root(
        &mut wrong_selected_destination,
        RTLInst::Iop(Operation::Oaddl, Arc::new(vec![VALUE, INDEX]), VALUE),
    );
    assert!(stage4_proofs(wrong_selected_destination).is_empty());

    let mut hidden_carrier_preopt_definition = stage4_compound_fixture("ADD", 8, None);
    hidden_carrier_preopt_definition.rel_push(
        "stage4_preopt_selected_rtl",
        (
            POST_TERMINAL,
            RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), VALUE),
        ),
    );
    assert!(stage4_proofs(hidden_carrier_preopt_definition).is_empty());

    let mut hidden_result_preopt_definition = stage4_compound_fixture("ADD", 8, None);
    hidden_result_preopt_definition.rel_push(
        "stage4_preopt_selected_rtl",
        (
            POST_TERMINAL,
            RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), ROOT_DEF),
        ),
    );
    assert!(stage4_proofs(hidden_result_preopt_definition).is_empty());

    let mut hidden_result_preopt_use = stage4_compound_fixture("ADD", 8, None);
    hidden_result_preopt_use.rel_push(
        "stage4_preopt_selected_rtl",
        (POST_TERMINAL, RTLInst::Ireturn(ROOT_DEF)),
    );
    assert!(stage4_proofs(hidden_result_preopt_use).is_empty());

    let mut hidden_result_final_definition = stage4_compound_fixture("ADD", 8, None);
    hidden_result_final_definition.rel_push(
        "rtl_inst",
        (
            POST_TERMINAL,
            RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), ROOT_DEF),
        ),
    );
    assert!(stage4_proofs(hidden_result_final_definition).is_empty());

    let mut hidden_result_final_use = stage4_compound_fixture("ADD", 8, None);
    hidden_result_final_use.rel_push("rtl_inst", (POST_TERMINAL, RTLInst::Ireturn(ROOT_DEF)));
    assert!(stage4_proofs(hidden_result_final_use).is_empty());

    let mut wrong_source_reaching = stage4_compound_fixture("ADD", 8, None);
    wrong_source_reaching.rel_push("reaching_use_rtl", (NODE, Mreg::DX, TEMP));
    assert!(stage4_proofs(wrong_source_reaching).is_empty());

    let mut wrong_carrier_reaching = stage4_compound_fixture("ADD", 8, None);
    wrong_carrier_reaching.rel_push("reaching_use_rtl", (NODE, Mreg::AX, TEMP));
    assert!(stage4_proofs(wrong_carrier_reaching).is_empty());

    let mut competing_to_root = stage4_compound_fixture("ADD", 8, None);
    competing_to_root.rel_push("reg_def_used", (PLACEMENT + 1, Mreg::AX, NODE));
    assert!(stage4_proofs(competing_to_root).is_empty());

    let mut competing_from_placement = stage4_compound_fixture("ADD", 8, None);
    competing_from_placement.rel_push("reg_def_used", (PLACEMENT, Mreg::AX, TERMINAL));
    assert!(stage4_proofs(competing_from_placement).is_empty());

    let mut competing_from_root = stage4_compound_fixture("ADD", 8, None);
    competing_from_root.rel_push("reg_def_used", (NODE, Mreg::AX, POST_TERMINAL));
    assert!(stage4_proofs(competing_from_root).is_empty());

    let mut missing_reg_def = stage4_compound_fixture("ADD", 8, None);
    missing_reg_def.rel_set(
        "reg_def_used",
        ascent::boxcar::Vec::<(Node, Mreg, Node)>::new(),
    );
    assert!(stage4_proofs(missing_reg_def).is_empty());

    let mut competing_xtl = stage4_compound_fixture("ADD", 8, None);
    competing_xtl.rel_push("reg_xtl", (NODE, Mreg::AX, TEMP));
    assert!(stage4_proofs(competing_xtl).is_empty());
    let mut missing_root_read_xtl = stage4_compound_fixture("ADD", 8, None);
    let retained_xtl = missing_root_read_xtl
        .rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl")
        .filter(|row| **row != (NODE, Mreg::AX, PLACEMENT_DEF))
        .cloned()
        .collect::<ascent::boxcar::Vec<_>>();
    missing_root_read_xtl.rel_set("reg_xtl", retained_xtl);
    assert!(stage4_proofs(missing_root_read_xtl).is_empty());
    let mut wrong_root_read_xtl = stage4_compound_fixture("ADD", 8, None);
    let retained_xtl = wrong_root_read_xtl
        .rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl")
        .filter(|row| **row != (NODE, Mreg::AX, PLACEMENT_DEF))
        .cloned()
        .chain(std::iter::once((NODE, Mreg::AX, TEMP)))
        .collect::<ascent::boxcar::Vec<_>>();
    wrong_root_read_xtl.rel_set("reg_xtl", retained_xtl);
    wrong_root_read_xtl.rel_push("xtl_canonical", (TEMP, VALUE));
    assert!(stage4_proofs(wrong_root_read_xtl).is_empty());
    // Distinct instruction definitions must have distinct identities. Without
    // the explicit inequality, a forged shared id plus two duplicate root
    // rows has the same cardinality and canonical value as the sealed pair.
    let mut duplicate_shared_definition = stage4_compound_fixture("ADD", 8, None);
    duplicate_shared_definition.rel_set(
        "is_def",
        [(PLACEMENT, PLACEMENT_DEF), (NODE, PLACEMENT_DEF)]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    duplicate_shared_definition.rel_set(
        "reg_xtl",
        [
            (PLACEMENT, Mreg::AX, PLACEMENT_DEF),
            (NODE, Mreg::AX, PLACEMENT_DEF),
            (NODE, Mreg::AX, PLACEMENT_DEF),
        ]
        .into_iter()
        .collect::<ascent::boxcar::Vec<_>>(),
    );
    assert!(stage4_proofs(duplicate_shared_definition).is_empty());
    let mut missing_xtl = stage4_compound_fixture("ADD", 8, None);
    missing_xtl.rel_set(
        "reg_xtl",
        ascent::boxcar::Vec::<(Node, Mreg, RTLReg)>::new(),
    );
    assert!(stage4_proofs(missing_xtl).is_empty());

    let mut competing_definition = stage4_compound_fixture("ADD", 8, None);
    competing_definition.rel_push("is_def", (NODE, TEMP));
    assert!(stage4_proofs(competing_definition).is_empty());
    let mut missing_definition = stage4_compound_fixture("ADD", 8, None);
    missing_definition.rel_set("is_def", ascent::boxcar::Vec::<(Node, RTLReg)>::new());
    assert!(stage4_proofs(missing_definition).is_empty());

    let mut missing_reaching = stage4_compound_fixture("ADD", 8, None);
    missing_reaching.rel_set(
        "reaching_use_rtl",
        ascent::boxcar::Vec::<(Node, Mreg, RTLReg)>::new(),
    );
    assert!(stage4_proofs(missing_reaching).is_empty());

    // The Dual lattice may retain stale higher representatives. The minimum
    // final representative is authoritative; a lower competing value changes
    // that fixed point and must reject.
    let mut stale_higher = stage4_compound_fixture("ADD", 8, None);
    stale_higher.rel_push("xtl_canonical", (ROOT_DEF, VALUE + 1));
    assert_eq!(stage4_proofs(stale_higher).len(), 1);
    let mut competing_lower = stage4_compound_fixture("ADD", 8, None);
    competing_lower.rel_push("xtl_canonical", (ROOT_DEF, VALUE - 1));
    assert!(stage4_proofs(competing_lower).is_empty());

    let mut missing_load_proof = stage4_compound_fixture("ADD", 8, None);
    missing_load_proof.rel_set(
        "authenticated_scalar_memory_access",
        ascent::boxcar::Vec::<(Node, ScalarMemoryAccessProof)>::new(),
    );
    assert!(stage4_proofs(missing_load_proof).is_empty());

    let mut ambiguous_load_proof = stage4_compound_fixture("ADD", 8, None);
    ambiguous_load_proof.rel_push(
        "authenticated_scalar_memory_access",
        (
            PLACEMENT,
            scalar_stage4_access(
                PLACEMENT,
                ScalarMemoryDirection::Read,
                8,
                Mreg::CX,
                BASE,
                TEMP,
            ),
        ),
    );
    assert!(stage4_proofs(ambiguous_load_proof).is_empty());
}

#[test]
fn stage4_eliminated_store_keeps_scanning_flags_and_terminal_coff() {
    let store = stage4_compound_fixture_with_terminal("AND", 8, Some((255, 0)), true);
    let proofs = stage4_proofs(store);
    assert_eq!(proofs.len(), 1, "store -> MOV -> RET should close flags");

    let mut missing_imul_effect = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut missing_imul_effect,
        POST_TERMINAL,
        instruction("", "IMUL", REGISTER, EXTRA),
    );
    assert!(stage4_proofs(missing_imul_effect).is_empty());

    let mut imul_clobber = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut imul_clobber,
        POST_TERMINAL,
        instruction("", "IMUL", REGISTER, EXTRA),
    );
    imul_clobber.rel_push("decoded_reg_def", (POST_TERMINAL, Mreg::AX));
    imul_clobber.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::AX));
    imul_clobber.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::DX));
    assert_eq!(
        stage4_proofs(imul_clobber).len(),
        1,
        "a later exact IMUL destroys the earlier ADD flags",
    );

    let mut imul3_clobber = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut imul3_clobber,
        POST_TERMINAL,
        (4, "", "IMUL", EXTRA, EXTRA, IMMEDIATE, NO_OP, 0, 0),
    );
    imul3_clobber.rel_push("op_immediate", (IMMEDIATE, 3_i64, 0_usize));
    imul3_clobber.rel_push("decoded_reg_def", (POST_TERMINAL, Mreg::AX));
    imul3_clobber.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::AX));
    assert_eq!(
        stage4_proofs(imul3_clobber).len(),
        1,
        "exact [source,destination,immediate] IMUL destroys prior flags",
    );

    let mut forged_imul3_width = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut forged_imul3_width,
        POST_TERMINAL,
        (4, "", "IMUL", EXTRA, EXTRA, IMMEDIATE, NO_OP, 0, 0),
    );
    forged_imul3_width.rel_push("op_immediate", (IMMEDIATE, 3_i64, 4_usize));
    forged_imul3_width.rel_push("decoded_reg_def", (POST_TERMINAL, Mreg::AX));
    forged_imul3_width.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::AX));
    assert!(
        stage4_proofs(forged_imul3_width).is_empty(),
        "noncanonical nonzero immediate width passed IMUL flags proof",
    );

    let mut ambiguous_imul3_immediate = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut ambiguous_imul3_immediate,
        POST_TERMINAL,
        (4, "", "IMUL", EXTRA, EXTRA, IMMEDIATE, NO_OP, 0, 0),
    );
    ambiguous_imul3_immediate.rel_push("op_immediate", (IMMEDIATE, 3_i64, 0_usize));
    ambiguous_imul3_immediate.rel_push("op_immediate", (IMMEDIATE, 5_i64, 0_usize));
    ambiguous_imul3_immediate.rel_push("decoded_reg_def", (POST_TERMINAL, Mreg::AX));
    ambiguous_imul3_immediate.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::AX));
    assert!(
        stage4_proofs(ambiguous_imul3_immediate).is_empty(),
        "competing IMUL immediates passed flags proof",
    );

    let mut old_imul3_order = stage4_compound_fixture_with_terminal("ADD", 8, None, true);
    replace_instruction_pair(
        &mut old_imul3_order,
        POST_TERMINAL,
        (4, "", "IMUL", IMMEDIATE, EXTRA, EXTRA, NO_OP, 0, 0),
    );
    old_imul3_order.rel_push("op_immediate", (IMMEDIATE, 3_i64, 0_usize));
    old_imul3_order.rel_push("decoded_reg_def", (POST_TERMINAL, Mreg::AX));
    old_imul3_order.rel_push("decoded_reg_use", (POST_TERMINAL, Mreg::AX));
    assert!(
        stage4_proofs(old_imul3_order).is_empty(),
        "obsolete [immediate,source,destination] order passed flags proof",
    );

    for (mnemonic, raw) in [
        ("ADD", (4, "", "JNE", NO_OP, NO_OP, NO_OP, NO_OP, 0, 0)),
        ("OR", (4, "", "CMOVNE", REGISTER, EXTRA, NO_OP, NO_OP, 0, 0)),
        (
            "IMUL",
            (4, "", "SETNE", REGISTER, NO_OP, NO_OP, NO_OP, 0, 0),
        ),
    ] {
        let mut consumer = stage4_compound_fixture_with_terminal(mnemonic, 8, None, true);
        replace_instruction_pair(&mut consumer, POST_TERMINAL, raw);
        assert!(
            stage4_proofs(consumer).is_empty(),
            "{mnemonic} flags consumer survived"
        );
    }

    let mut direct_pair = stage4_compound_fixture("ADD", 8, None);
    direct_pair.rel_push("flags_and_jump_pair", (NODE, TERMINAL, "jne"));
    assert!(stage4_proofs(direct_pair).is_empty());

    let mut wrong_return_opcode = stage4_compound_fixture("ADD", 8, None);
    replace_instruction_pair(
        &mut wrong_return_opcode,
        TERMINAL,
        (4, "", "JNE", NO_OP, NO_OP, NO_OP, NO_OP, 0, 0),
    );
    assert!(stage4_proofs(wrong_return_opcode).is_empty());

    let mut terminal_owner_conflict = stage4_compound_fixture("ADD", 8, None);
    terminal_owner_conflict.rel_push("instr_in_function", (TERMINAL, FUNCTION + 0x100));
    assert!(stage4_proofs(terminal_owner_conflict).is_empty());

    let mut noncontiguous_root = stage4_compound_fixture("ADD", 8, None);
    let root = (8, "", "ADD", REGISTER, EXTRA, NO_OP, NO_OP, 0, 0);
    replace_instruction_pair(&mut noncontiguous_root, NODE, root);
    assert!(stage4_proofs(noncontiguous_root).is_empty());
}

#[test]
fn stage4_add_and_zero_accept_only_the_sealed_real_pipeline_companions() {
    let imul3 = stage4_compound_fixture("IMUL", 4, Some((3, 0)));
    assert_eq!(stage4_proofs(imul3).len(), 1);

    let mut old_imul3_order = stage4_compound_fixture("IMUL", 4, Some((3, 0)));
    replace_instruction_pair(
        &mut old_imul3_order,
        NODE,
        (4, "", "IMUL", IMMEDIATE, EXTRA, EXTRA, NO_OP, 0, 0),
    );
    assert!(
        stage4_proofs(old_imul3_order).is_empty(),
        "obsolete [immediate,source,destination] root order authenticated",
    );

    for width in [4, 8] {
        let mut forged_xor_use = stage4_zero_fixture("XOR", width);
        forged_xor_use.rel_push("decoded_reg_use", (NODE, Mreg::AX));
        assert!(
            stage4_proofs(forged_xor_use).is_empty(),
            "XOR-self/{width} accepted a decoder use suppressed by the loader",
        );

        let mut missing_sub_use = stage4_zero_fixture("SUB", width);
        missing_sub_use.rel_set(
            "decoded_reg_use",
            ascent::boxcar::Vec::<(Node, Mreg)>::new(),
        );
        assert!(
            stage4_proofs(missing_sub_use).is_empty(),
            "SUB-self/{width} accepted a missing decoder use",
        );

        let mut wrong_sub_use = stage4_zero_fixture("SUB", width);
        wrong_sub_use.rel_set(
            "decoded_reg_use",
            vec![(NODE, Mreg::CX)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(
            stage4_proofs(wrong_sub_use).is_empty(),
            "SUB-self/{width} accepted the wrong decoder use",
        );

        let mut add = stage4_compound_fixture("ADD", width, None);
        let (_, lea) = stage4_addition_operations(width).unwrap();
        set_stage4_preopt_root(
            &mut add,
            RTLInst::Iop(lea.clone(), Arc::new(vec![VALUE, INDEX]), ROOT_DEF),
        );
        let proofs = stage4_proofs(add);
        assert_eq!(proofs.len(), 1, "ADD-final-LEA/{width}");
        assert_eq!(proofs[0].operation, lea);

        let mut third_add = stage4_compound_fixture("ADD", width, None);
        third_add.rel_push(
            "ltl_inst",
            (
                NODE,
                LTLInst::Lop(
                    if width == 8 {
                        Operation::Osubl
                    } else {
                        Operation::Osub
                    },
                    Arc::new(vec![Mreg::AX, Mreg::DX]),
                    Mreg::AX,
                ),
            ),
        );
        assert!(stage4_proofs(third_add).is_empty());

        let mut forged_const_ltl = stage4_zero_fixture("XOR", width);
        let register = Mreg::AX;
        forged_const_ltl.rel_set(
            "ltl_inst",
            vec![(
                NODE,
                LTLInst::Lop(
                    if width == 8 {
                        Operation::Olongconst(0)
                    } else {
                        Operation::Ointconst(0)
                    },
                    Arc::new(Vec::new()),
                    register,
                ),
            )]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(stage4_proofs(forged_const_ltl).is_empty());

        let mut forged_const_input = stage4_zero_fixture("XOR", width);
        forged_const_input.rel_set(
            "rtl_inst_candidate",
            vec![(
                NODE,
                RTLInst::Iop(
                    if width == 8 {
                        Operation::Olongconst(0)
                    } else {
                        Operation::Ointconst(0)
                    },
                    Arc::new(Vec::new()),
                    VALUE,
                ),
            )]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(stage4_proofs(forged_const_input).is_empty());

        let mut missing_raw_ltl = stage4_zero_fixture("XOR", width);
        missing_raw_ltl.rel_set("ltl_inst", ascent::boxcar::Vec::<(Node, LTLInst)>::new());
        assert!(stage4_proofs(missing_raw_ltl).is_empty());

        let mut missing_raw_input = stage4_zero_fixture("XOR", width);
        missing_raw_input.rel_set(
            "rtl_inst_candidate",
            ascent::boxcar::Vec::<(Node, RTLInst)>::new(),
        );
        assert!(stage4_proofs(missing_raw_input).is_empty());

        let mut third_zero = stage4_zero_fixture("XOR", width);
        third_zero.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iop(
                    if width == 8 {
                        Operation::Osubl
                    } else {
                        Operation::Osub
                    },
                    Arc::new(vec![VALUE, VALUE]),
                    VALUE,
                ),
            ),
        );
        assert!(stage4_proofs(third_zero).is_empty());

        let sub = stage4_zero_fixture("SUB", width);
        let proofs = stage4_proofs(sub);
        assert_eq!(proofs.len(), 1, "SUB-self-final-const/{width}");
        assert_eq!(proofs[0].kind, Stage4SourceKind::Zeroing);
    }
}

#[test]
fn stage4_machine_proofs_reject_ambiguity_effect_relocation_and_cfg_gaps() {
    let mutations: Vec<(&str, Box<dyn Fn(&mut DecompileDB)>)> =
        vec![
            (
                "raw-mismatch",
                Box::new(|db| {
                    push_instruction(
                        db,
                        "unrefinedinstruction",
                        NODE,
                        instruction("", "SUB", REGISTER, EXTRA),
                    )
                }),
            ),
            (
                "competing-final",
                Box::new(|db| {
                    db.rel_push(
                        "rtl_inst",
                        (
                            NODE,
                            RTLInst::Iop(Operation::Osub, Arc::new(vec![VALUE, INDEX]), VALUE),
                        ),
                    )
                }),
            ),
            (
                "competing-ltl",
                Box::new(|db| {
                    db.rel_push(
                        "ltl_inst",
                        (
                            NODE,
                            LTLInst::Lop(
                                Operation::Osub,
                                Arc::new(vec![Mreg::AX, Mreg::DX]),
                                Mreg::AX,
                            ),
                        ),
                    )
                }),
            ),
            (
                "operand-ambiguity",
                Box::new(|db| db.rel_push("op_register", (REGISTER, "ECX"))),
            ),
            (
                "addr32-conflict",
                Box::new(|db| db.rel_push("instruction_address_size", (NODE, 4_u8))),
            ),
            (
                "memory-effect",
                Box::new(|db| db.rel_push("decoded_memory_read_operand", (NODE, MEMORY))),
            ),
            (
                "register-effect",
                Box::new(|db| db.rel_push("decoded_reg_use", (NODE, Mreg::CX))),
            ),
            (
                "rtl-binding-conflict",
                Box::new(|db| db.rel_push("reg_rtl", (NODE, Mreg::DX, VALUE))),
            ),
            (
                "ambiguous-owner",
                Box::new(|db| db.rel_push("instr_in_function", (NODE, FUNCTION + 0x100))),
            ),
            (
                "cfg-bypass",
                Box::new(|db| db.rel_push("stage4_preopt_selected_succ", (NODE, NEXT))),
            ),
            (
                "unowned-cfg",
                Box::new(|db| {
                    db.rel_push("stage4_preopt_selected_succ", (PLACEMENT + 1, NODE));
                }),
            ),
            (
                "hidden-memory-call-use",
                Box::new(|db| {
                    db.rel_push(
                        "call_through_memory_load",
                        (
                            NEXT + 4,
                            TEMP,
                            MemoryChunk::MAny64,
                            Addressing::Aindexed(0),
                            Arc::new(vec![VALUE]),
                        ),
                    );
                    db.rel_push("instr_in_function", (NEXT + 4, FUNCTION));
                }),
            ),
            (
                "relocation-overlap",
                Box::new(|db| {
                    db.coff_address_map.as_mut().expect("map").relocations.push(
                        CoffRelocationMap {
                            section_index: 1,
                            section_name: ".text".into(),
                            section_offset: 0x11,
                            mapped_field_va: NODE + 1,
                            relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                            width_bits: 32,
                            target_original_name: "external".into(),
                            target_mapped_address: 0x9000,
                            encoded_value: 0,
                        },
                    );
                }),
            ),
            (
                "old-rmw-value-still-live",
                Box::new(|db| {
                    db.rel_push("rtl_inst", (NEXT + 4, RTLInst::Ireturn(VALUE)));
                    db.rel_push("instr_in_function", (NEXT + 4, FUNCTION));
                    db.rel_push("rtl_succ", (PLACEMENT, NEXT + 4));
                }),
            ),
        ];
    for (name, mutate) in mutations {
        let mut db = stage4_compound_fixture("ADD", 4, None);
        mutate(&mut db);
        assert!(
            stage4_proofs(db).is_empty(),
            "{name} unexpectedly authenticated"
        );
    }
}

#[test]
fn stage4_affine_proof_rejects_stack_symbolic_and_unproved_leaves() {
    let mut missing_pseudo_read = stage4_fixture(Stage4Fixture::Affine);
    missing_pseudo_read.rel_set(
        "decoded_memory_read_operand",
        ascent::boxcar::Vec::<(Node, Symbol)>::new(),
    );
    assert!(stage4_proofs(missing_pseudo_read).is_empty());

    let mut extra_pseudo_read = stage4_fixture(Stage4Fixture::Affine);
    extra_pseudo_read.rel_push("decoded_memory_read_operand", (NODE, REGISTER));
    assert!(stage4_proofs(extra_pseudo_read).is_empty());

    let mut wrong_pseudo_read = stage4_fixture(Stage4Fixture::Affine);
    wrong_pseudo_read.rel_set(
        "decoded_memory_read_operand",
        vec![(NODE, REGISTER)]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    assert!(stage4_proofs(wrong_pseudo_read).is_empty());

    let mut competing_indirect = stage4_fixture(Stage4Fixture::Affine);
    competing_indirect.rel_push(
        "op_indirect",
        (MEMORY, "NONE", "RSP", "RDX", 4_i64, 12_i64, 8_usize),
    );
    assert!(stage4_proofs(competing_indirect).is_empty());

    let mut missing_leaf = stage4_fixture(Stage4Fixture::Affine);
    missing_leaf.rel_set(
        "emit_function_param_candidate",
        ascent::boxcar::Vec::<(Address, RTLReg)>::new(),
    );
    assert!(stage4_proofs(missing_leaf).is_empty());

    let mut width_mismatch = stage4_fixture(Stage4Fixture::Affine32);
    width_mismatch.rel_push(
        "op_indirect",
        (MEMORY, "NONE", "RCX", "RDX", 4_i64, 12_i64, 8_usize),
    );
    assert!(stage4_proofs(width_mismatch).is_empty());

    let mut destination_mismatch = stage4_fixture(Stage4Fixture::Affine32);
    destination_mismatch.rel_push("op_register", (REGISTER, "RAX"));
    assert!(stage4_proofs(destination_mismatch).is_empty());
}
