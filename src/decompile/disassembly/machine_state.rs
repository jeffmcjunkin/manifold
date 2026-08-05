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
pub const WIN64_SYSCALL_STUB_SCHEMA_ID: &str = "manifold.machine-state-stub.win64-syscall.v1";

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

type IndirectOperand = (&'static str, &'static str, &'static str, i64, i64, usize);

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateCoffFunctionIdentity {
    pub name: String,
    pub address: String,
    pub size: u64,
    pub section_index: usize,
    pub section_name: String,
    pub section_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64HomeStore {
    pub instruction_address: String,
    pub source_register: &'static str,
    pub stack_offset: u8,
    pub width_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64AbiState {
    pub name: &'static str,
    pub argument_registers: [&'static str; 4],
    pub home_kind: &'static str,
    pub home_stores: Vec<Win64HomeStore>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRegisterCopy {
    pub instruction_address: String,
    pub source_register: &'static str,
    pub destination_register: &'static str,
    pub width_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateImmediateWrite {
    pub instruction_address: String,
    pub register: &'static str,
    pub width_bits: u8,
    pub value: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateMemoryTest {
    pub instruction_address: String,
    pub address: String,
    pub width_bits: u8,
    pub mask: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateConditionalBranch {
    pub instruction_address: String,
    pub condition: &'static str,
    pub target_address: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateTransferPath {
    pub transfer_instruction_address: String,
    pub transfer_kind: &'static str,
    pub interrupt_vector: Option<u8>,
    pub return_instruction_address: String,
    pub return_kind: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64SyscallDispatch {
    pub test: MachineStateMemoryTest,
    pub branch: MachineStateConditionalBranch,
    pub native_path: MachineStateTransferPath,
    pub legacy_path: MachineStateTransferPath,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStatePadding {
    pub instruction_address: String,
    pub kind: &'static str,
    pub boundary: u8,
    pub size: u8,
}

/// A complete, relocation-free Win64 user-mode system-call wrapper contract.
/// Every literal here is decoded state; no original instruction bytes are
/// retained or exposed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64SyscallStub {
    pub schema: &'static str,
    pub kind: &'static str,
    pub function: MachineStateCoffFunctionIdentity,
    pub abi: Win64AbiState,
    pub register_copy: MachineStateRegisterCopy,
    pub service_number: MachineStateImmediateWrite,
    pub dispatch: Win64SyscallDispatch,
    pub padding: MachineStatePadding,
    pub relocations: Vec<MachineStateRelocationIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum MachineStateStubRecord {
    SetImm32DirectTailJump(MachineStateStub),
    Win64Syscall(Win64SyscallStub),
}

impl MachineStateStubRecord {
    pub(crate) fn function_name(&self) -> &str {
        match self {
            Self::SetImm32DirectTailJump(stub) => &stub.function.name,
            Self::Win64Syscall(stub) => &stub.function.name,
        }
    }

    pub(crate) fn function_address(&self) -> &str {
        match self {
            Self::SetImm32DirectTailJump(stub) => &stub.function.address,
            Self::Win64Syscall(stub) => &stub.function.address,
        }
    }

    fn sort_key(&self) -> (&str, &str) {
        (self.function_address(), self.function_name())
    }
}

fn only<T>(values: impl IntoIterator<Item = T>) -> Option<T> {
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

fn exact_register_operand(
    operand: Symbol,
    expected: &'static str,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
) -> bool {
    registers.get(operand) == Some(&BTreeSet::from([expected]))
        && !immediates.contains_key(operand)
        && !indirects.contains_key(operand)
}

fn exact_immediate_operand(
    operand: Symbol,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
) -> Option<i64> {
    if registers.contains_key(operand) || indirects.contains_key(operand) {
        return None;
    }
    only(immediates.get(operand)?.iter().copied())
}

fn exact_indirect_operand(
    operand: Symbol,
    expected: IndirectOperand,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
) -> bool {
    indirects.get(operand) == Some(&BTreeSet::from([expected]))
        && !registers.contains_key(operand)
        && !immediates.contains_key(operand)
}

fn exact_or_empty<T: Copy + Ord>(
    facts: &BTreeMap<Node, BTreeSet<T>>,
    node: Node,
    expected: impl IntoIterator<Item = T>,
) -> bool {
    let expected: BTreeSet<T> = expected.into_iter().collect();
    facts.get(&node).cloned().unwrap_or_default() == expected
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
) -> Vec<MachineStateStubRecord> {
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
    let mut indirects: BTreeMap<Symbol, BTreeSet<IndirectOperand>> = BTreeMap::new();
    for (operand, segment, base, index, scale, displacement, width) in db.rel_iter::<(
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )>("op_indirect")
    {
        indirects.entry(*operand).or_default().insert((
            *segment,
            *base,
            *index,
            *scale,
            *displacement,
            *width,
        ));
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
    let mut memory_reads: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_read_operand") {
        memory_reads.entry(*node).or_default().insert(*operand);
    }
    let mut memory_writes: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_write_operand") {
        memory_writes.entry(*node).or_default().insert(*operand);
    }
    let mut flags_and_jumps: BTreeMap<Node, BTreeSet<(Node, &'static str)>> = BTreeMap::new();
    for (source, jump, condition) in
        db.rel_iter::<(Node, Node, &'static str)>("flags_and_jump_pair")
    {
        flags_and_jumps
            .entry(*source)
            .or_default()
            .insert((*jump, *condition));
    }

    let mut result = Vec::new();
    for function in &map.functions {
        if let Some(stub) = recognize_one(
            function,
            map,
            &instructions,
            &registers,
            &immediates,
            &indirects,
            &owners,
            &next,
            &direct_jumps,
            &direct_calls,
            &cfg,
            &decoded_defs,
            &decoded_uses,
            &memory_reads,
            &memory_writes,
        ) {
            result.push(MachineStateStubRecord::SetImm32DirectTailJump(stub));
        }
        if let Some(stub) = recognize_win64_syscall(
            function,
            map,
            &instructions,
            &registers,
            &immediates,
            &indirects,
            &owners,
            &next,
            &direct_jumps,
            &direct_calls,
            &cfg,
            &decoded_defs,
            &decoded_uses,
            &memory_reads,
            &memory_writes,
            &flags_and_jumps,
        ) {
            result.push(MachineStateStubRecord::Win64Syscall(stub));
        }
    }
    result.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
    result
}

#[allow(clippy::too_many_arguments)]
fn recognize_one(
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
    instructions: &BTreeMap<Address, Vec<Instruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    next: &BTreeMap<Node, BTreeSet<Node>>,
    direct_jumps: &BTreeMap<Node, BTreeSet<Address>>,
    direct_calls: &BTreeMap<Node, BTreeSet<Address>>,
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    memory_reads: &BTreeMap<Node, BTreeSet<Symbol>>,
    memory_writes: &BTreeMap<Node, BTreeSet<Symbol>>,
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

    if !exact_register_operand(mov.5, "R10D", registers, immediates, indirects)
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
        || !exact_or_empty(memory_reads, mov.0, [])
        || !exact_or_empty(memory_writes, mov.0, [])
        || !exact_or_empty(memory_reads, jump.0, [])
        || !exact_or_empty(memory_writes, jump.0, [])
    {
        return None;
    }
    let immediate = reinterpret_imm32(exact_immediate_operand(
        mov.4, registers, immediates, indirects,
    )?)?;

    let target = only(direct_jumps.get(&jump.0)?.iter().copied())?;
    if exact_immediate_operand(jump.4, registers, immediates, indirects)? as u64 != target {
        return None;
    }
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

#[allow(clippy::too_many_arguments)]
fn recognize_win64_syscall(
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
    instructions: &BTreeMap<Address, Vec<Instruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    next: &BTreeMap<Node, BTreeSet<Node>>,
    direct_jumps: &BTreeMap<Node, BTreeSet<Address>>,
    direct_calls: &BTreeMap<Node, BTreeSet<Address>>,
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    memory_reads: &BTreeMap<Node, BTreeSet<Symbol>>,
    memory_writes: &BTreeMap<Node, BTreeSet<Symbol>>,
    flags_and_jumps: &BTreeMap<Node, BTreeSet<(Node, &'static str)>>,
) -> Option<Win64SyscallStub> {
    if map.schema != "manifold.coff-address-map.v1"
        || map.loader_id != "amd64-coff-image-v1"
        || map.architecture != "x86_64-pc-windows-msvc"
    {
        return None;
    }

    let variants = [0usize, 4usize].into_iter().filter_map(|home_count| {
        recognize_win64_syscall_variant(
            function,
            map,
            instructions,
            registers,
            immediates,
            indirects,
            owners,
            next,
            direct_jumps,
            direct_calls,
            cfg,
            decoded_defs,
            decoded_uses,
            memory_reads,
            memory_writes,
            flags_and_jumps,
            home_count,
        )
    });
    only(variants)
}

#[allow(clippy::too_many_arguments)]
fn recognize_win64_syscall_variant(
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
    instructions: &BTreeMap<Address, Vec<Instruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    next: &BTreeMap<Node, BTreeSet<Node>>,
    direct_jumps: &BTreeMap<Node, BTreeSet<Address>>,
    direct_calls: &BTreeMap<Node, BTreeSet<Address>>,
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    memory_reads: &BTreeMap<Node, BTreeSet<Symbol>>,
    memory_writes: &BTreeMap<Node, BTreeSet<Symbol>>,
    flags_and_jumps: &BTreeMap<Node, BTreeSet<(Node, &'static str)>>,
    home_count: usize,
) -> Option<Win64SyscallStub> {
    let (function_size, padding_size) = match home_count {
        0 => (32u64, 8usize),
        4 => (48u64, 4usize),
        _ => return None,
    };
    if function.original_size != function_size
        || function.manifold_size != function_size
        || function.mapped_end != function.mapped_entry.checked_add(function_size)?
        || function.mapped_entry & 0xf != 0
        || function.mapped_end & 0xf != 0
    {
        return None;
    }

    let section = only(
        map.sections
            .iter()
            .filter(|section| section.index == function.section_index),
    )?;
    if section.kind != "Text"
        || function.mapped_entry < section.mapped_va_start
        || function.mapped_end > section.mapped_va_end
        || function.section_offset < section.original_offset_start
        || function
            .section_offset
            .checked_add(function.original_size)?
            > section.original_offset_end
    {
        return None;
    }
    let original_end = function
        .section_offset
        .checked_add(function.original_size)?;
    if map.relocations.iter().any(|relocation| {
        if relocation.section_index != function.section_index {
            return false;
        }
        let width = u64::from(relocation.width_bits).div_ceil(8).max(1);
        let source_end = relocation.section_offset.saturating_add(width);
        let mapped_end = relocation.mapped_field_va.saturating_add(width);
        (relocation.section_offset < original_end && source_end > function.section_offset)
            || (relocation.mapped_field_va < function.mapped_end
                && mapped_end > function.mapped_entry)
    }) {
        return None;
    }

    let rows: Vec<Instruction> = instructions
        .range(function.mapped_entry..function.mapped_end)
        .flat_map(|(_, rows)| rows.iter().copied())
        .collect();
    if rows.len() != home_count + 9 {
        return None;
    }
    let body = home_count;
    let copy = rows[body];
    let service = rows[body + 1];
    let test = rows[body + 2];
    let branch = rows[body + 3];
    let native_transfer = rows[body + 4];
    let native_return = rows[body + 5];
    let legacy_transfer = rows[body + 6];
    let legacy_return = rows[body + 7];
    let padding = rows[body + 8];

    let expected_body = function.mapped_entry + (home_count as u64 * 5);
    let exact_no_extra_operands = |row: Instruction| row.6 == "0" && row.7 == "0";
    if copy.0 != expected_body
        || copy.1 != 3
        || copy.2 != ""
        || copy.3 != "MOV"
        || !exact_no_extra_operands(copy)
        || service.0 != copy.0 + 3
        || service.1 != 5
        || service.2 != ""
        || service.3 != "MOV"
        || !exact_no_extra_operands(service)
        || test.0 != service.0 + 5
        || test.1 != 8
        || test.2 != ""
        || test.3 != "TEST"
        || !exact_no_extra_operands(test)
        || branch.0 != test.0 + 8
        || branch.1 != 2
        || branch.2 != ""
        || branch.3 != "JNE"
        || branch.5 != "0"
        || !exact_no_extra_operands(branch)
        || native_transfer.0 != branch.0 + 2
        || native_transfer.1 != 2
        || native_transfer.2 != ""
        || native_transfer.3 != "SYSCALL"
        || [
            native_transfer.4,
            native_transfer.5,
            native_transfer.6,
            native_transfer.7,
        ] != ["0"; 4]
        || native_return.0 != native_transfer.0 + 2
        || native_return.1 != 1
        || native_return.2 != ""
        || native_return.3 != "RET"
        || [
            native_return.4,
            native_return.5,
            native_return.6,
            native_return.7,
        ] != ["0"; 4]
        || legacy_transfer.0 != native_return.0 + 1
        || legacy_transfer.1 != 2
        || legacy_transfer.2 != ""
        || legacy_transfer.3 != "INT"
        || legacy_transfer.5 != "0"
        || !exact_no_extra_operands(legacy_transfer)
        || legacy_return.0 != legacy_transfer.0 + 2
        || legacy_return.1 != 1
        || legacy_return.2 != ""
        || legacy_return.3 != "RET"
        || [
            legacy_return.4,
            legacy_return.5,
            legacy_return.6,
            legacy_return.7,
        ] != ["0"; 4]
        || padding.0 != legacy_return.0 + 1
        || padding.1 != padding_size
        || padding.2 != ""
        || padding.3 != "NOP"
        || padding.5 != "0"
        || !exact_no_extra_operands(padding)
        || padding.0 + padding.1 as u64 != function.mapped_end
    {
        return None;
    }

    let home_specs = [
        ("RCX", "rcx", Mreg::CX, 8u8),
        ("RDX", "rdx", Mreg::DX, 16u8),
        ("R8", "r8", Mreg::R8, 24u8),
        ("R9", "r9", Mreg::R9, 32u8),
    ];
    let mut home_stores = Vec::new();
    for (index, (decoded_name, wire_name, mreg, offset)) in
        home_specs.iter().copied().take(home_count).enumerate()
    {
        let row = rows[index];
        let expected_address = function.mapped_entry + index as u64 * 5;
        if row.0 != expected_address
            || row.1 != 5
            || row.2 != ""
            || row.3 != "MOV"
            || !exact_no_extra_operands(row)
            || !exact_register_operand(row.4, decoded_name, registers, immediates, indirects)
            || !exact_indirect_operand(
                row.5,
                ("NONE", "RSP", "NONE", 1, offset as i64, 8),
                registers,
                immediates,
                indirects,
            )
            || !exact_or_empty(decoded_defs, row.0, [])
            || !exact_or_empty(decoded_uses, row.0, [Mreg::SP, mreg])
            || !exact_or_empty(memory_reads, row.0, [])
            || !exact_or_empty(memory_writes, row.0, [row.5])
        {
            return None;
        }
        home_stores.push(Win64HomeStore {
            instruction_address: format!("0x{:x}", row.0),
            source_register: wire_name,
            stack_offset: offset,
            width_bits: 64,
        });
    }

    if !exact_register_operand(copy.4, "RCX", registers, immediates, indirects)
        || !exact_register_operand(copy.5, "R10", registers, immediates, indirects)
        || !exact_or_empty(decoded_defs, copy.0, [Mreg::R10])
        || !exact_or_empty(decoded_uses, copy.0, [Mreg::CX])
        || !exact_or_empty(memory_reads, copy.0, [])
        || !exact_or_empty(memory_writes, copy.0, [])
        || !exact_register_operand(service.5, "EAX", registers, immediates, indirects)
        || !exact_or_empty(decoded_defs, service.0, [Mreg::AX])
        || !exact_or_empty(decoded_uses, service.0, [])
        || !exact_or_empty(memory_reads, service.0, [])
        || !exact_or_empty(memory_writes, service.0, [])
    {
        return None;
    }
    let service_number = reinterpret_imm32(exact_immediate_operand(
        service.4, registers, immediates, indirects,
    )?)?;

    let mask = exact_immediate_operand(test.4, registers, immediates, indirects)?;
    let mask = u8::try_from(mask).ok()?;
    let test_operand = only(indirects.get(test.5)?.iter().copied())?;
    let dispatch_address = u64::try_from(test_operand.4).ok()?;
    if !exact_indirect_operand(
        test.5,
        ("NONE", "NONE", "NONE", 1, test_operand.4, 1),
        registers,
        immediates,
        indirects,
    ) || !exact_or_empty(decoded_defs, test.0, [])
        || !exact_or_empty(decoded_uses, test.0, [])
        || !exact_or_empty(memory_reads, test.0, [test.5])
        || !exact_or_empty(memory_writes, test.0, [])
    {
        return None;
    }

    let legacy_target = exact_immediate_operand(branch.4, registers, immediates, indirects)? as u64;
    if legacy_target != legacy_transfer.0
        || exact_immediate_operand(legacy_transfer.4, registers, immediates, indirects)? != 0x2e
        || !exact_or_empty(flags_and_jumps, test.0, [(branch.0, "ne")])
        || direct_jumps.get(&branch.0) != Some(&BTreeSet::from([legacy_transfer.0]))
        || cfg.get(&branch.0)
            != Some(&BTreeSet::from([
                (native_transfer.0, "fallthrough"),
                (legacy_transfer.0, "branch"),
            ]))
        || cfg.get(&native_transfer.0) != Some(&BTreeSet::from([(native_return.0, "fallthrough")]))
    {
        return None;
    }

    let nop_operand = match home_count {
        0 => ("NONE", "RAX", "RAX", 1, 0, 4),
        4 => ("NONE", "RAX", "NONE", 1, 0, 4),
        _ => return None,
    };
    if !exact_indirect_operand(padding.4, nop_operand, registers, immediates, indirects) {
        return None;
    }

    for (index, row) in rows.iter().copied().enumerate() {
        if owners.get(&row.0) != Some(&BTreeSet::from([function.mapped_entry]))
            || !exact_or_empty(
                next,
                row.0,
                rows.get(index + 1).map(|successor| successor.0),
            )
            || !exact_or_empty(direct_calls, row.0, [])
            || (row.0 != branch.0 && !exact_or_empty(direct_jumps, row.0, []))
            || (row.0 != branch.0 && row.0 != native_transfer.0 && !exact_or_empty(cfg, row.0, []))
            || (row.0 != test.0 && !exact_or_empty(flags_and_jumps, row.0, []))
        {
            return None;
        }
    }

    for row in [branch, native_transfer, legacy_transfer, padding] {
        if !exact_or_empty(decoded_defs, row.0, [])
            || !exact_or_empty(decoded_uses, row.0, [])
            || !exact_or_empty(memory_reads, row.0, [])
            || !exact_or_empty(memory_writes, row.0, [])
        {
            return None;
        }
    }
    for row in [native_return, legacy_return] {
        if !exact_or_empty(decoded_defs, row.0, [Mreg::SP])
            || !exact_or_empty(decoded_uses, row.0, [Mreg::SP])
            || !exact_or_empty(memory_reads, row.0, [])
            || !exact_or_empty(memory_writes, row.0, [])
        {
            return None;
        }
    }

    Some(Win64SyscallStub {
        schema: WIN64_SYSCALL_STUB_SCHEMA_ID,
        kind: "win64_syscall",
        function: MachineStateCoffFunctionIdentity {
            name: function.provider_name.clone(),
            address: format!("0x{:x}", function.mapped_entry),
            size: function_size,
            section_index: function.section_index,
            section_name: section.name.clone(),
            section_offset: function.section_offset,
        },
        abi: Win64AbiState {
            name: "win64",
            argument_registers: ["rcx", "rdx", "r8", "r9"],
            home_kind: if home_count == 0 {
                "none"
            } else {
                "four_register_arguments"
            },
            home_stores,
        },
        register_copy: MachineStateRegisterCopy {
            instruction_address: format!("0x{:x}", copy.0),
            source_register: "rcx",
            destination_register: "r10",
            width_bits: 64,
        },
        service_number: MachineStateImmediateWrite {
            instruction_address: format!("0x{:x}", service.0),
            register: "eax",
            width_bits: 32,
            value: service_number,
        },
        dispatch: Win64SyscallDispatch {
            test: MachineStateMemoryTest {
                instruction_address: format!("0x{:x}", test.0),
                address: format!("0x{dispatch_address:x}"),
                width_bits: 8,
                mask,
            },
            branch: MachineStateConditionalBranch {
                instruction_address: format!("0x{:x}", branch.0),
                condition: "not_equal",
                target_address: format!("0x{:x}", legacy_transfer.0),
            },
            native_path: MachineStateTransferPath {
                transfer_instruction_address: format!("0x{:x}", native_transfer.0),
                transfer_kind: "syscall",
                interrupt_vector: None,
                return_instruction_address: format!("0x{:x}", native_return.0),
                return_kind: "near_return",
            },
            legacy_path: MachineStateTransferPath {
                transfer_instruction_address: format!("0x{:x}", legacy_transfer.0),
                transfer_kind: "software_interrupt",
                interrupt_vector: Some(0x2e),
                return_instruction_address: format!("0x{:x}", legacy_return.0),
                return_kind: "near_return",
            },
        },
        padding: MachineStatePadding {
            instruction_address: format!("0x{:x}", padding.0),
            kind: "nop_alignment",
            boundary: 16,
            size: padding_size as u8,
        },
        relocations: Vec::new(),
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

    const HOME_RCX: Symbol = "syscall_home_rcx";
    const HOME_RDX: Symbol = "syscall_home_rdx";
    const HOME_R8: Symbol = "syscall_home_r8";
    const HOME_R9: Symbol = "syscall_home_r9";
    const HOME_MEM_8: Symbol = "syscall_home_mem_8";
    const HOME_MEM_16: Symbol = "syscall_home_mem_16";
    const HOME_MEM_24: Symbol = "syscall_home_mem_24";
    const HOME_MEM_32: Symbol = "syscall_home_mem_32";
    const COPY_RCX: Symbol = "syscall_copy_rcx";
    const COPY_R10: Symbol = "syscall_copy_r10";
    const SERVICE_IMM: Symbol = "syscall_service_imm";
    const SERVICE_EAX: Symbol = "syscall_service_eax";
    const TEST_MASK: Symbol = "syscall_test_mask";
    const TEST_MEMORY: Symbol = "syscall_test_memory";
    const LEGACY_TARGET: Symbol = "syscall_legacy_target";
    const LEGACY_VECTOR: Symbol = "syscall_legacy_vector";
    const PADDING_MEMORY: Symbol = "syscall_padding_memory";

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

    fn syscall_fixture(home_count: usize) -> (DecompileDB, CoffAddressMap) {
        assert!(matches!(home_count, 0 | 4));
        let mut db = DecompileDB::default();
        let entry = 0x1000_1000;
        let function_size = if home_count == 0 { 32 } else { 48 };
        let body = entry + home_count as u64 * 5;
        let copy = body;
        let service = copy + 3;
        let test = service + 5;
        let branch = test + 8;
        let native_transfer = branch + 2;
        let native_return = native_transfer + 2;
        let legacy_transfer = native_return + 1;
        let legacy_return = legacy_transfer + 2;
        let padding = legacy_return + 1;

        let mut rows: Vec<Instruction> = Vec::new();
        if home_count == 4 {
            rows.extend([
                (entry, 5, "", "MOV", HOME_RCX, HOME_MEM_8, NONE, NONE, 0, 0),
                (
                    entry + 5,
                    5,
                    "",
                    "MOV",
                    HOME_RDX,
                    HOME_MEM_16,
                    NONE,
                    NONE,
                    0,
                    0,
                ),
                (
                    entry + 10,
                    5,
                    "",
                    "MOV",
                    HOME_R8,
                    HOME_MEM_24,
                    NONE,
                    NONE,
                    0,
                    0,
                ),
                (
                    entry + 15,
                    5,
                    "",
                    "MOV",
                    HOME_R9,
                    HOME_MEM_32,
                    NONE,
                    NONE,
                    0,
                    0,
                ),
            ]);
        }
        rows.extend([
            (copy, 3, "", "MOV", COPY_RCX, COPY_R10, NONE, NONE, 0, 0),
            (
                service,
                5,
                "",
                "MOV",
                SERVICE_IMM,
                SERVICE_EAX,
                NONE,
                NONE,
                0,
                0,
            ),
            (
                test,
                8,
                "",
                "TEST",
                TEST_MASK,
                TEST_MEMORY,
                NONE,
                NONE,
                0,
                0,
            ),
            (branch, 2, "", "JNE", LEGACY_TARGET, NONE, NONE, NONE, 0, 0),
            (
                native_transfer,
                2,
                "",
                "SYSCALL",
                NONE,
                NONE,
                NONE,
                NONE,
                0,
                0,
            ),
            (native_return, 1, "", "RET", NONE, NONE, NONE, NONE, 0, 0),
            (
                legacy_transfer,
                2,
                "",
                "INT",
                LEGACY_VECTOR,
                NONE,
                NONE,
                NONE,
                0,
                0,
            ),
            (legacy_return, 1, "", "RET", NONE, NONE, NONE, NONE, 0, 0),
            (
                padding,
                if home_count == 0 { 8 } else { 4 },
                "",
                "NOP",
                PADDING_MEMORY,
                NONE,
                NONE,
                NONE,
                0,
                0,
            ),
        ]);
        db.rel_set(
            "unrefinedinstruction",
            rows.iter().copied().collect::<ascent::boxcar::Vec<_>>(),
        );

        let mut op_registers = vec![(COPY_RCX, "RCX"), (COPY_R10, "R10"), (SERVICE_EAX, "EAX")];
        if home_count == 4 {
            op_registers.extend([
                (HOME_RCX, "RCX"),
                (HOME_RDX, "RDX"),
                (HOME_R8, "R8"),
                (HOME_R9, "R9"),
            ]);
        }
        db.rel_set(
            "op_register",
            op_registers.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_immediate",
            vec![
                (SERVICE_IMM, 0x89ab_cdef_i64, 0usize),
                (TEST_MASK, 1, 0),
                (LEGACY_TARGET, legacy_transfer as i64, 0),
                (LEGACY_VECTOR, 0x2e, 0),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        let mut indirects = vec![(TEST_MEMORY, "NONE", "NONE", "NONE", 1, 0x7ffe_0308, 1usize)];
        if home_count == 0 {
            indirects.push((PADDING_MEMORY, "NONE", "RAX", "RAX", 1, 0, 4));
        } else {
            indirects.extend([
                (HOME_MEM_8, "NONE", "RSP", "NONE", 1, 8, 8),
                (HOME_MEM_16, "NONE", "RSP", "NONE", 1, 16, 8),
                (HOME_MEM_24, "NONE", "RSP", "NONE", 1, 24, 8),
                (HOME_MEM_32, "NONE", "RSP", "NONE", 1, 32, 8),
                (PADDING_MEMORY, "NONE", "RAX", "NONE", 1, 0, 4),
            ]);
        }
        db.rel_set(
            "op_indirect",
            indirects.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "instr_in_function",
            rows.iter()
                .map(|row| (row.0, entry))
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "next",
            rows.windows(2)
                .map(|pair| (pair[0].0, pair[1].0))
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "direct_jump",
            vec![(branch, legacy_transfer)]
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
            vec![
                (branch, legacy_transfer, "branch"),
                (branch, native_transfer, "fallthrough"),
                (native_transfer, native_return, "fallthrough"),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "flags_and_jump_pair",
            vec![(test, branch, "ne")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        let mut defs = vec![(copy, Mreg::R10), (service, Mreg::AX)];
        let mut uses = vec![
            (copy, Mreg::CX),
            (native_return, Mreg::SP),
            (legacy_return, Mreg::SP),
        ];
        defs.extend([(native_return, Mreg::SP), (legacy_return, Mreg::SP)]);
        if home_count == 4 {
            uses.extend([
                (entry, Mreg::SP),
                (entry, Mreg::CX),
                (entry + 5, Mreg::SP),
                (entry + 5, Mreg::DX),
                (entry + 10, Mreg::SP),
                (entry + 10, Mreg::R8),
                (entry + 15, Mreg::SP),
                (entry + 15, Mreg::R9),
            ]);
        }
        db.rel_set(
            "decoded_reg_def",
            defs.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_use",
            uses.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_memory_read_operand",
            vec![(test, TEST_MEMORY)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        let writes = if home_count == 4 {
            vec![
                (entry, HOME_MEM_8),
                (entry + 5, HOME_MEM_16),
                (entry + 10, HOME_MEM_24),
                (entry + 15, HOME_MEM_32),
            ]
        } else {
            Vec::new()
        };
        db.rel_set(
            "decoded_memory_write_operand",
            writes.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );

        let map = CoffAddressMap {
            schema: "manifold.coff-address-map.v1",
            loader_id: "amd64-coff-image-v1",
            architecture: "x86_64-pc-windows-msvc",
            image_base: 0x1000_0000,
            sections: vec![CoffSectionMap {
                index: 7,
                name: ".text$arbitrary".into(),
                kind: "Text".into(),
                original_file_offset: Some(0x200),
                original_offset_start: 0x80,
                original_offset_end: 0x80 + function_size,
                mapped_va_start: entry,
                mapped_va_end: entry + function_size,
            }],
            functions: vec![CoffFunctionMap {
                original_name: "unrelated_original_name".into(),
                provider_name: "coff_fn_unrelated_provider_name".into(),
                section_index: 7,
                section_offset: 0x80,
                original_size: function_size,
                manifold_size: function_size,
                mapped_entry: entry,
                mapped_end: entry + function_size,
            }],
            symbols: vec![CoffSymbolMap {
                original_name: "unrelated_original_name".into(),
                provider_name: "coff_fn_unrelated_provider_name".into(),
                kind: "function".into(),
                defined: true,
                section_index: Some(7),
                section_offset: Some(0x80),
                mapped_address: entry,
            }],
            externs: Vec::new(),
            relocations: Vec::new(),
        };
        (db, map)
    }

    #[test]
    fn recognizes_exact_arbitrary_named_stub() {
        let (db, map) = fixture();
        let result = recognize_machine_state_stubs(&db, &map);
        assert_eq!(result.len(), 1);
        let MachineStateStubRecord::SetImm32DirectTailJump(stub) = &result[0] else {
            panic!("expected direct-tail machine-state record")
        };
        assert_eq!(stub.state_write.value, 0x89ab_cdef);
        assert_eq!(stub.transfer.target_original_name, "arbitrary_destination");
    }

    #[test]
    fn recognizes_arbitrary_named_win64_syscalls_with_and_without_home_stores() {
        for home_count in [0, 4] {
            let (db, map) = syscall_fixture(home_count);
            let result = recognize_machine_state_stubs(&db, &map);
            assert_eq!(result.len(), 1, "home_count={home_count}");
            let MachineStateStubRecord::Win64Syscall(stub) = &result[0] else {
                panic!("expected Win64 syscall record")
            };
            assert_eq!(stub.function.name, "coff_fn_unrelated_provider_name");
            assert_eq!(stub.service_number.value, 0x89ab_cdef);
            assert_eq!(stub.dispatch.test.address, "0x7ffe0308");
            assert_eq!(stub.abi.home_stores.len(), home_count);
            assert_eq!(stub.relocations, Vec::new());
            assert_eq!(
                stub.abi.home_kind,
                if home_count == 0 {
                    "none"
                } else {
                    "four_register_arguments"
                }
            );
        }
    }

    #[test]
    fn rejects_syscall_home_control_relocation_and_padding_ambiguity() {
        let (mut db, map) = syscall_fixture(4);
        let malformed_indirects = db
            .rel_iter::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>("op_indirect")
            .map(|row| {
                if row.0 == HOME_MEM_16 {
                    (row.0, row.1, row.2, row.3, row.4, 24, row.6)
                } else {
                    *row
                }
            })
            .collect::<ascent::boxcar::Vec<_>>();
        db.rel_set("op_indirect", malformed_indirects);
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = syscall_fixture(0);
        db.rel_set(
            "flags_and_jump_pair",
            vec![(0x1000_1008, 0x1000_1010, "e")]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = syscall_fixture(0);
        map.relocations.push(CoffRelocationMap {
            section_index: 7,
            section_name: ".text$arbitrary".into(),
            section_offset: 0x88,
            mapped_field_va: 0x1000_1008,
            relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
            width_bits: 32,
            target_original_name: "unrelated_data".into(),
            target_mapped_address: 0x2000_0000,
            encoded_value: 0,
        });
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = syscall_fixture(0);
        map.relocations.push(CoffRelocationMap {
            section_index: 7,
            section_name: ".text$arbitrary".into(),
            section_offset: 0x7f,
            mapped_field_va: 0x1000_0fff,
            relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
            width_bits: 32,
            target_original_name: "overlapping_data".into(),
            target_mapped_address: 0x2000_0000,
            encoded_value: 0,
        });
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = syscall_fixture(0);
        db.rel_push("decoded_reg_def", (0x1000_1012, Mreg::R11));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
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
            vec![(jump, Mreg::AX)]
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
