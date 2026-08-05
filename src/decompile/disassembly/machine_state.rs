//! Fail-closed recognition of tiny functions whose observable contract includes
//! non-ABI machine state.
//!
//! Ordinary C cannot express every x86-64 tail-transfer contract.  This module
//! deliberately recognizes only an exact decoder/CFG/COFF shape and exports a
//! typed semantic record.  It never copies instruction bytes and never selects
//! by symbol spelling.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::decompile::disassembly::coff::{CoffAddressMap, CoffFunctionMap};
use crate::decompile::elevator::DecompileDB;
use crate::mreg::Mreg;
use crate::x86::types::{Address, Node, Symbol};

pub const MACHINE_STATE_STUB_SCHEMA_ID: &str = "manifold.machine-state-stub.set-imm32-tail-jump.v1";

type Instruction = (
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
);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateFunctionIdentity {
    pub name: String,
    pub address: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRegisterWrite {
    pub instruction_address: String,
    pub register: &'static str,
    pub width_bits: u8,
    pub value: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRelocationIdentity {
    pub section_index: usize,
    pub section_name: String,
    pub section_offset: u64,
    pub relocation_type: String,
    pub width_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateTailTransfer {
    pub instruction_address: String,
    pub kind: &'static str,
    pub target_address: String,
    pub target_original_name: String,
    pub target_provider_name: String,
    pub relocation: MachineStateRelocationIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateStub {
    pub schema: &'static str,
    pub kind: &'static str,
    pub function: MachineStateFunctionIdentity,
    pub state_write: MachineStateRegisterWrite,
    pub transfer: MachineStateTailTransfer,
}

fn only<T: Copy>(values: impl IntoIterator<Item = T>) -> Option<T> {
    let mut values = values.into_iter();
    let value = values.next()?;
    values.next().is_none().then_some(value)
}

fn reinterpret_imm32(value: i64) -> Option<u32> {
    if (i32::MIN as i64..=u32::MAX as i64).contains(&value) {
        Some(value as u32)
    } else {
        None
    }
}

/// Recognize the exact sequence
///
/// ```text
/// mov r10d, imm32
/// jmp rel32-symbol
/// ```
///
/// as one machine-state-bearing direct tail transfer.  Every accepted field is
/// jointly authenticated by Capstone's structured operands, inferred ownership,
/// CFG edges, and one original COFF relocation.  Any ambiguity simply omits the
/// optional semantic record; ordinary Manifold C remains available.
pub fn recognize_machine_state_stubs(
    db: &DecompileDB,
    map: &CoffAddressMap,
) -> Vec<MachineStateStub> {
    let mut instructions: BTreeMap<Address, Vec<Instruction>> = BTreeMap::new();
    for row in db.rel_iter::<Instruction>("unrefinedinstruction") {
        instructions.entry(row.0).or_default().push(*row);
    }

    let mut registers: BTreeMap<Symbol, BTreeSet<&'static str>> = BTreeMap::new();
    for (operand, register) in db.rel_iter::<(Symbol, &'static str)>("op_register") {
        registers.entry(*operand).or_default().insert(*register);
    }
    let mut immediates: BTreeMap<Symbol, BTreeSet<i64>> = BTreeMap::new();
    for (operand, value, _) in db.rel_iter::<(Symbol, i64, usize)>("op_immediate") {
        immediates.entry(*operand).or_default().insert(*value);
    }

    let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        owners.entry(*node).or_default().insert(*function);
    }
    let mut next: BTreeMap<Node, BTreeSet<Node>> = BTreeMap::new();
    for (source, target) in db.rel_iter::<(Node, Node)>("next") {
        next.entry(*source).or_default().insert(*target);
    }
    let mut direct_jumps: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (source, target) in db.rel_iter::<(Node, Address)>("direct_jump") {
        direct_jumps.entry(*source).or_default().insert(*target);
    }
    let mut direct_calls: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (source, target) in db.rel_iter::<(Node, Address)>("direct_call") {
        direct_calls.entry(*source).or_default().insert(*target);
    }
    let mut cfg: BTreeMap<Node, BTreeSet<(Address, Symbol)>> = BTreeMap::new();
    for (source, target, kind) in db.rel_iter::<(Node, Address, Symbol)>("ddisasm_cfg_edge") {
        cfg.entry(*source).or_default().insert((*target, *kind));
    }
    let mut decoded_defs: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_def") {
        decoded_defs.entry(*node).or_default().insert(*register);
    }
    let mut decoded_uses: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_use") {
        decoded_uses.entry(*node).or_default().insert(*register);
    }

    let mut result = Vec::new();
    for function in &map.functions {
        if let Some(stub) = recognize_one(
            function,
            map,
            &instructions,
            &registers,
            &immediates,
            &owners,
            &next,
            &direct_jumps,
            &direct_calls,
            &cfg,
            &decoded_defs,
            &decoded_uses,
        ) {
            result.push(stub);
        }
    }
    result.sort_by(|left, right| {
        left.function
            .address
            .cmp(&right.function.address)
            .then_with(|| left.function.name.cmp(&right.function.name))
    });
    result
}

#[allow(clippy::too_many_arguments)]
fn recognize_one(
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
    instructions: &BTreeMap<Address, Vec<Instruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    next: &BTreeMap<Node, BTreeSet<Node>>,
    direct_jumps: &BTreeMap<Node, BTreeSet<Address>>,
    direct_calls: &BTreeMap<Node, BTreeSet<Address>>,
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
) -> Option<MachineStateStub> {
    let rows: Vec<Instruction> = instructions
        .range(function.mapped_entry..function.mapped_end)
        .flat_map(|(_, rows)| rows.iter().copied())
        .collect();
    if rows.len() != 2 {
        return None;
    }
    let mov = rows[0];
    let jump = rows[1];

    if mov.0 != function.mapped_entry
        || mov.1 != 6
        || mov.2 != ""
        || mov.3 != "MOV"
        || mov.6 != "0"
        || mov.7 != "0"
        || jump.0 != mov.0 + mov.1 as u64
        || jump.1 != 5
        || jump.2 != ""
        || jump.3 != "JMP"
        || jump.5 != "0"
        || jump.6 != "0"
        || jump.7 != "0"
        || jump.0 + jump.1 as u64 != function.mapped_end
    {
        return None;
    }

    if owners.get(&mov.0) != Some(&BTreeSet::from([function.mapped_entry]))
        || owners.get(&jump.0) != Some(&BTreeSet::from([function.mapped_entry]))
        || next.get(&mov.0) != Some(&BTreeSet::from([jump.0]))
        || next.get(&jump.0).is_some_and(|targets| !targets.is_empty())
        || direct_calls
            .get(&mov.0)
            .is_some_and(|targets| !targets.is_empty())
        || direct_calls
            .get(&jump.0)
            .is_some_and(|targets| !targets.is_empty())
        || cfg.get(&mov.0).is_some_and(|edges| !edges.is_empty())
    {
        return None;
    }

    if registers.get(mov.5) != Some(&BTreeSet::from(["R10D"]))
        || decoded_defs.get(&mov.0) != Some(&BTreeSet::from([Mreg::R10]))
        || decoded_uses
            .get(&mov.0)
            .is_some_and(|uses| !uses.is_empty())
        || decoded_defs
            .get(&jump.0)
            .is_some_and(|defs| !defs.is_empty())
        || decoded_uses
            .get(&jump.0)
            .is_some_and(|uses| !uses.is_empty())
    {
        return None;
    }
    let immediate = reinterpret_imm32(only(immediates.get(mov.4)?.iter().copied())?)?;

    let target = only(direct_jumps.get(&jump.0)?.iter().copied())?;
    if cfg.get(&jump.0) != Some(&BTreeSet::from([(target, "branch")])) {
        return None;
    }

    let relocations: Vec<_> = map
        .relocations
        .iter()
        .filter(|relocation| {
            relocation.section_index == function.section_index
                && relocation.section_name
                    == map
                        .sections
                        .iter()
                        .find(|section| section.index == function.section_index)
                        .map(|section| section.name.as_str())
                        .unwrap_or("")
                && relocation.mapped_field_va == jump.0 + 1
        })
        .collect();
    let relocation = only(relocations)?;
    if relocation.relocation_type != "IMAGE_REL_AMD64_REL32"
        || relocation.width_bits != 32
        || relocation.target_mapped_address != target
    {
        return None;
    }

    let target_symbols: Vec<_> = map
        .symbols
        .iter()
        .filter(|symbol| {
            symbol.original_name == relocation.target_original_name
                && symbol.mapped_address == target
                && symbol.kind == "function"
        })
        .collect();
    let target_symbol = only(target_symbols)?;

    Some(MachineStateStub {
        schema: MACHINE_STATE_STUB_SCHEMA_ID,
        kind: "set_imm32_direct_tail_jump",
        function: MachineStateFunctionIdentity {
            name: function.provider_name.clone(),
            address: format!("0x{:x}", function.mapped_entry),
        },
        state_write: MachineStateRegisterWrite {
            instruction_address: format!("0x{:x}", mov.0),
            register: "r10d",
            width_bits: 32,
            value: immediate,
        },
        transfer: MachineStateTailTransfer {
            instruction_address: format!("0x{:x}", jump.0),
            kind: "direct_tail_jump",
            target_address: format!("0x{target:x}"),
            target_original_name: target_symbol.original_name.clone(),
            target_provider_name: target_symbol.provider_name.clone(),
            relocation: MachineStateRelocationIdentity {
                section_index: relocation.section_index,
                section_name: relocation.section_name.clone(),
                section_offset: relocation.section_offset,
                relocation_type: relocation.relocation_type.clone(),
                width_bits: relocation.width_bits,
            },
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::disassembly::coff::{
        CoffFunctionMap, CoffRelocationMap, CoffSectionMap, CoffSymbolMap,
    };

    const NONE: Symbol = "0";
    const IMM: Symbol = "machine_state_test_imm";
    const DST: Symbol = "machine_state_test_dst";
    const TARGET: Symbol = "machine_state_test_target";
    const INDIRECT: Symbol = "machine_state_test_indirect";

    fn fixture() -> (DecompileDB, CoffAddressMap) {
        let mut db = DecompileDB::default();
        let entry = 0x1000_0000;
        let jump = entry + 6;
        let target = 0x1000_2000;
        db.rel_set(
            "unrefinedinstruction",
            vec![
                (entry, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
                (jump, 5, "", "JMP", TARGET, NONE, NONE, NONE, 0, 0),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_register",
            vec![(DST, "R10D")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_immediate",
            vec![(IMM, 0x89ab_cdef_i64, 4_usize), (TARGET, target as i64, 4)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "instr_in_function",
            vec![(entry, entry), (jump, entry)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "next",
            vec![(entry, jump)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "direct_jump",
            vec![(jump, target)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "direct_call",
            Vec::<(Node, Address)>::new()
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "ddisasm_cfg_edge",
            vec![(jump, target, "branch")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_def",
            vec![(entry, Mreg::R10)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_use",
            Vec::<(Node, Mreg)>::new()
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );

        let map = CoffAddressMap {
            schema: "manifold.coff-address-map.v1",
            loader_id: "amd64-coff-image-v1",
            architecture: "x86_64-pc-windows-msvc",
            image_base: entry,
            sections: vec![CoffSectionMap {
                index: 1,
                name: ".text$mn".into(),
                kind: "Text".into(),
                original_file_offset: Some(0x100),
                original_offset_start: 0,
                original_offset_end: 11,
                mapped_va_start: entry,
                mapped_va_end: entry + 11,
            }],
            functions: vec![CoffFunctionMap {
                original_name: "arbitrary_source_name".into(),
                provider_name: "coff_fn_arbitrary_source_name".into(),
                section_index: 1,
                section_offset: 0,
                original_size: 11,
                manifold_size: 11,
                mapped_entry: entry,
                mapped_end: entry + 11,
            }],
            symbols: vec![
                CoffSymbolMap {
                    original_name: "arbitrary_source_name".into(),
                    provider_name: "coff_fn_arbitrary_source_name".into(),
                    kind: "function".into(),
                    defined: true,
                    section_index: Some(1),
                    section_offset: Some(0),
                    mapped_address: entry,
                },
                CoffSymbolMap {
                    original_name: "arbitrary_destination".into(),
                    provider_name: "coff_ext_arbitrary_destination".into(),
                    kind: "function".into(),
                    defined: false,
                    section_index: None,
                    section_offset: None,
                    mapped_address: target,
                },
            ],
            externs: Vec::new(),
            relocations: vec![CoffRelocationMap {
                section_index: 1,
                section_name: ".text$mn".into(),
                section_offset: 7,
                mapped_field_va: jump + 1,
                relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                width_bits: 32,
                target_original_name: "arbitrary_destination".into(),
                target_mapped_address: target,
                encoded_value: 0,
            }],
        };
        (db, map)
    }

    #[test]
    fn recognizes_exact_arbitrary_named_stub() {
        let (db, map) = fixture();
        let result = recognize_machine_state_stubs(&db, &map);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].state_write.value, 0x89ab_cdef);
        assert_eq!(
            result[0].transfer.target_original_name,
            "arbitrary_destination"
        );
    }

    #[test]
    fn rejects_wrong_register_width_and_extra_effects() {
        let (mut db, map) = fixture();
        db.rel_set(
            "op_register",
            vec![(DST, "R10")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = fixture();
        db.rel_push("decoded_reg_def", (0x1000_0000, Mreg::R11));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, mut map) = fixture();
        map.functions[0].mapped_end += 1;
        map.functions[0].original_size += 1;
        map.functions[0].manifold_size += 1;
        db.rel_push(
            "unrefinedinstruction",
            (0x1000_000b, 1, "", "NOP", NONE, NONE, NONE, NONE, 0, 0),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
    }

    #[test]
    fn rejects_conditional_indirect_and_multiple_exit_shapes() {
        let (mut db, map) = fixture();
        let jump = 0x1000_0006;
        db.rel_set(
            "unrefinedinstruction",
            vec![
                (0x1000_0000, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
                (jump, 5, "", "JNE", TARGET, NONE, NONE, NONE, 0, 0),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = fixture();
        db.rel_set(
            "unrefinedinstruction",
            vec![
                (0x1000_0000, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
                (jump, 5, "", "JMP", INDIRECT, NONE, NONE, NONE, 0, 0),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_register",
            vec![(DST, "R10D"), (INDIRECT, "RAX")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "direct_jump",
            Vec::<(Node, Address)>::new()
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "ddisasm_cfg_edge",
            vec![(jump, 0, "indirect")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_use",
            vec![(jump, Mreg::RAX)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = fixture();
        db.rel_push("ddisasm_cfg_edge", (jump, 0x1000_3000, "branch"));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
    }

    #[test]
    fn rejects_ambiguous_or_unauthenticated_relocation() {
        let (db, mut map) = fixture();
        map.relocations.push(map.relocations[0].clone());
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = fixture();
        map.relocations[0].relocation_type = "IMAGE_REL_AMD64_REL32_1".into();
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
    }
}
