//! Fail-closed recognition of tiny functions whose observable contract includes
//! non-ABI machine state.
//!
//! Ordinary C cannot express every x86-64 tail-transfer contract.  This module
//! deliberately recognizes only an exact decoder/CFG/COFF shape and exports a
//! typed semantic record.  It never copies instruction bytes and never selects
//! by symbol spelling.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::decompile::disassembly::coff::{
    CoffAddressMap, CoffExternalKind, CoffExternalMap, CoffFunctionMap, CoffRelocationMap,
};
use crate::decompile::elevator::DecompileDB;
use crate::mreg::Mreg;
use crate::x86::types::{Address, Node, Symbol};

pub const MACHINE_STATE_STUB_SCHEMA_ID: &str = "manifold.machine-state-stub.set-imm32-tail-jump.v1";
pub const WIN64_SYSCALL_STUB_SCHEMA_ID: &str = "manifold.machine-state-stub.win64-syscall.v1";
pub const WIN64_KERNEL_SERVICE_STUB_SCHEMA_ID: &str =
    "manifold.machine-state-stub.win64-kernel-service.v1";
pub const WIN64_DWORD_OUT_GETTER_STUB_SCHEMA_ID: &str =
    "manifold.machine-state-stub.win64-dword-out-getter.v1";
pub const GUARDED_INDIRECT_FORWARD_TAIL_SCHEMA_ID: &str =
    "manifold.machine-state-stub.guarded-indirect-forward-tail.v1";

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
pub struct MachineStateDispatchLoad {
    pub instruction_address: String,
    pub target_register: &'static str,
    pub table_register: &'static str,
    pub displacement: u32,
    pub width_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateForwardingContract {
    pub input_registers: [&'static str; 4],
    pub stack_allocation_bytes: u8,
    pub base_load_displacement: u8,
    pub dispatch_load: MachineStateDispatchLoad,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateGuardCall {
    pub instruction_address: String,
    pub kind: &'static str,
    pub argument_register: &'static str,
    pub result_register: &'static str,
    pub target_address: String,
    pub target_original_name: String,
    pub target_provider_name: String,
    pub target_kind: &'static str,
    pub relocation: MachineStateRelocationIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRegisterTailTransfer {
    pub instruction_address: String,
    pub kind: &'static str,
    pub target_register: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GuardedIndirectForwardTailStub {
    pub schema: &'static str,
    pub kind: &'static str,
    pub function: MachineStateFunctionIdentity,
    pub forwarding: MachineStateForwardingContract,
    pub guard: MachineStateGuardCall,
    pub transfer: MachineStateRegisterTailTransfer,
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
pub struct MachineStateInterruptControl {
    pub instruction_address: String,
    pub kind: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateStackAdjustment {
    pub instruction_address: String,
    pub register: &'static str,
    pub operation: &'static str,
    pub width_bits: u8,
    pub amount_bytes: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateStackPush {
    pub instruction_address: String,
    pub kind: &'static str,
    pub stack_width_bits: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_register: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub immediate_value: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub immediate_width_bits: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRelocatedAddressLoad {
    pub instruction_address: String,
    pub kind: &'static str,
    pub destination_register: &'static str,
    pub width_bits: u8,
    pub target_address: String,
    pub target_original_name: String,
    pub target_provider_name: String,
    pub target_kind: &'static str,
    pub relocation: MachineStateRelocationIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64KernelServiceStub {
    pub schema: &'static str,
    pub kind: &'static str,
    pub function: MachineStateCoffFunctionIdentity,
    pub abi: Win64AbiState,
    pub entry_stack_pointer: MachineStateRegisterCopy,
    pub interrupt_control: MachineStateInterruptControl,
    pub stack_adjustment: MachineStateStackAdjustment,
    pub pushes: Vec<MachineStateStackPush>,
    pub linkage: MachineStateRelocatedAddressLoad,
    pub service_number: MachineStateImmediateWrite,
    pub dispatcher: MachineStateTailTransfer,
    pub return_instruction_address: String,
    pub return_kind: &'static str,
    pub padding: MachineStatePadding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateRegisterZero {
    pub instruction_address: String,
    pub register: &'static str,
    pub width_bits: u8,
    pub kind: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MachineStateDwordFieldCopy {
    pub load_instruction_address: String,
    pub source_base_register: &'static str,
    pub source_displacement: i64,
    pub source_displacement_encoding_bits: u8,
    pub temporary_register: &'static str,
    pub store_instruction_address: String,
    pub destination_base_register: &'static str,
    pub destination_displacement: i64,
    pub width_bits: u8,
}

/// A complete relocation-free Win64 dword out-getter contract.  The explicit
/// displacement encoding retains the distinction between its 25-byte disp8
/// and 28-byte disp32 forms without retaining original instruction bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Win64DwordOutGetterStub {
    pub schema: &'static str,
    pub kind: &'static str,
    pub function: MachineStateCoffFunctionIdentity,
    pub abi: Win64AbiState,
    pub entry_stack_pointer: MachineStateRegisterCopy,
    pub register_copies: Vec<MachineStateRegisterCopy>,
    pub result_zero: MachineStateRegisterZero,
    pub home_base_register: &'static str,
    pub field_copy: MachineStateDwordFieldCopy,
    pub return_instruction_address: String,
    pub return_kind: &'static str,
    pub padding: MachineStatePadding,
    pub relocations: Vec<MachineStateRelocationIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum MachineStateStubRecord {
    SetImm32DirectTailJump(MachineStateStub),
    Win64Syscall(Win64SyscallStub),
    Win64KernelService(Win64KernelServiceStub),
    Win64DwordOutGetter(Win64DwordOutGetterStub),
    GuardedIndirectForwardTail(GuardedIndirectForwardTailStub),
}

impl MachineStateStubRecord {
    pub(crate) fn function_name(&self) -> &str {
        match self {
            Self::SetImm32DirectTailJump(stub) => &stub.function.name,
            Self::Win64Syscall(stub) => &stub.function.name,
            Self::Win64KernelService(stub) => &stub.function.name,
            Self::Win64DwordOutGetter(stub) => &stub.function.name,
            Self::GuardedIndirectForwardTail(stub) => &stub.function.name,
        }
    }

    pub(crate) fn function_address(&self) -> &str {
        match self {
            Self::SetImm32DirectTailJump(stub) => &stub.function.address,
            Self::Win64Syscall(stub) => &stub.function.address,
            Self::Win64KernelService(stub) => &stub.function.address,
            Self::Win64DwordOutGetter(stub) => &stub.function.address,
            Self::GuardedIndirectForwardTail(stub) => &stub.function.address,
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

fn exact_or_authenticated_padding_fallthrough(
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    padding: Node,
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
) -> bool {
    // CFG construction is section-wide, so an alignment NOP at an authenticated
    // COFF boundary may acquire a synthetic fallthrough into the next function
    // even though decoder `next` deliberately stops at that boundary.
    let actual = cfg.get(&padding).cloned().unwrap_or_default();
    if actual.is_empty() {
        return true;
    }
    if actual != BTreeSet::from([(function.mapped_end, "fallthrough")]) {
        return false;
    }

    let Some(original_end) = function.section_offset.checked_add(function.original_size) else {
        return false;
    };
    let Some(section) = only(
        map.sections
            .iter()
            .filter(|section| section.index == function.section_index),
    ) else {
        return false;
    };
    only(map.functions.iter().filter(|successor| {
        successor.section_index == function.section_index
            && successor.section_offset == original_end
            && successor.mapped_entry == function.mapped_end
            && successor.mapped_entry < successor.mapped_end
            && successor.section_offset < section.original_offset_end
            && successor.mapped_entry < section.mapped_va_end
    }))
    .is_some()
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
        if let Some(stub) = recognize_win64_kernel_service(
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
            result.push(MachineStateStubRecord::Win64KernelService(stub));
        }
        if let Some(stub) = recognize_win64_dword_out_getter(
            db,
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
            result.push(MachineStateStubRecord::Win64DwordOutGetter(stub));
        }
        if let Some(stub) = recognize_guarded_indirect_forward_tail(
            function,
            map,
            &instructions,
            &registers,
            &immediates,
            &indirects,
            &memory_reads,
            &memory_writes,
            &owners,
            &next,
            &direct_jumps,
            &direct_calls,
            &cfg,
            &decoded_defs,
            &decoded_uses,
        ) {
            result.push(MachineStateStubRecord::GuardedIndirectForwardTail(stub));
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
    // Capstone releases disagree on whether TEST's read-only r/m operand is
    // reported as ReadOnly or conservatively as ReadWrite.  Accept only those
    // two presentations of this already-authenticated operand; any different
    // or additional write still fails closed.
    let exact_test_memory_effect = exact_or_empty(memory_writes, test.0, [])
        || exact_or_empty(memory_writes, test.0, [test.5]);
    if !exact_indirect_operand(
        test.5,
        ("NONE", "NONE", "NONE", 1, test_operand.4, 1),
        registers,
        immediates,
        indirects,
    ) || !exact_or_empty(decoded_defs, test.0, [])
        || !exact_or_empty(decoded_uses, test.0, [])
        || !exact_or_empty(memory_reads, test.0, [test.5])
        || !exact_test_memory_effect
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
    let exact_padding_cfg =
        exact_or_authenticated_padding_fallthrough(cfg, padding.0, function, map);

    for (index, row) in rows.iter().copied().enumerate() {
        if owners.get(&row.0) != Some(&BTreeSet::from([function.mapped_entry]))
            || !exact_or_empty(
                next,
                row.0,
                rows.get(index + 1).map(|successor| successor.0),
            )
            || !exact_or_empty(direct_calls, row.0, [])
            || (row.0 != branch.0 && !exact_or_empty(direct_jumps, row.0, []))
            || (row.0 != branch.0
                && row.0 != native_transfer.0
                && row.0 != padding.0
                && !exact_or_empty(cfg, row.0, []))
            || (row.0 == padding.0 && !exact_padding_cfg)
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

fn exact_operand_register(
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

fn exact_operand_immediate(
    operand: Symbol,
    expected: i64,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
) -> bool {
    immediates.get(operand) == Some(&BTreeSet::from([expected]))
        && !registers.contains_key(operand)
        && !indirects.contains_key(operand)
}

fn exact_operand_memory(
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

fn exact_symbol_set(values: Option<&BTreeSet<Symbol>>, expected: &[Symbol]) -> bool {
    let expected: BTreeSet<Symbol> = expected.iter().copied().collect();
    values.map_or(expected.is_empty(), |values| values == &expected)
}

fn exact_register_effects(
    node: Node,
    expected_defs: &[Mreg],
    expected_uses: &[Mreg],
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
) -> bool {
    let defs: BTreeSet<Mreg> = expected_defs.iter().copied().collect();
    let uses: BTreeSet<Mreg> = expected_uses.iter().copied().collect();
    decoded_defs
        .get(&node)
        .map_or(defs.is_empty(), |actual| actual == &defs)
        && decoded_uses
            .get(&node)
            .map_or(uses.is_empty(), |actual| actual == &uses)
}

fn exact_instruction_header(
    instruction: Instruction,
    size: usize,
    mnemonic: &'static str,
    operand_count: usize,
) -> bool {
    let operands = [instruction.4, instruction.5, instruction.6, instruction.7];
    instruction.1 == size
        && instruction.2.is_empty()
        && instruction.3 == mnemonic
        && instruction.8 == 0
        && instruction.9 == 0
        && operands[..operand_count]
            .iter()
            .all(|operand| *operand != "0")
        && operands[operand_count..]
            .iter()
            .all(|operand| *operand == "0")
}

fn exact_undefined_function_external<'a>(
    map: &'a CoffAddressMap,
    relocation: &CoffRelocationMap,
) -> Option<&'a CoffExternalMap> {
    // OR matching makes duplicate names and aliased synthetic addresses
    // ambiguous instead of letting either half hide the other.
    let external = only(map.externs.iter().filter(|external| {
        external.original_name == relocation.target_original_name
            || external.synthetic_address == relocation.target_mapped_address
    }))?;
    let symbol = only(map.symbols.iter().filter(|symbol| {
        symbol.original_name == relocation.target_original_name
            || symbol.mapped_address == relocation.target_mapped_address
    }))?;
    (external.original_name == relocation.target_original_name
        && external.synthetic_address == relocation.target_mapped_address
        && external.kind == CoffExternalKind::Function
        && symbol.original_name == relocation.target_original_name
        && symbol.provider_name == external.provider_name
        && symbol.mapped_address == relocation.target_mapped_address
        && symbol.kind == "function"
        && !symbol.defined
        && symbol.section_index.is_none()
        && symbol.section_offset.is_none())
    .then_some(external)
}

/// Recognize one closed, 64-byte Win64 privileged service wrapper.  Selection
/// uses only structured operands/effects, exact ownership and CFG, and two
/// original COFF REL32 identities; names, addresses, bytes, and corpus keys do
/// not participate.
#[allow(clippy::too_many_arguments)]
fn recognize_win64_kernel_service(
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
) -> Option<Win64KernelServiceStub> {
    const SIZE: u64 = 64;
    if map.schema != "manifold.coff-address-map.v1"
        || map.loader_id != "amd64-coff-image-v1"
        || map.architecture != "x86_64-pc-windows-msvc"
        || function.original_size != SIZE
        || function.manifold_size != SIZE
        || function.mapped_end != function.mapped_entry.checked_add(SIZE)?
        || function.mapped_entry % SIZE != 0
    {
        return None;
    }
    let section = only(
        map.sections
            .iter()
            .filter(|section| section.index == function.section_index),
    )?;
    if section.kind != "Text"
        || function
            .section_offset
            .checked_sub(section.original_offset_start)?
            % SIZE
            != 0
        || function.mapped_entry < section.mapped_va_start
        || function.mapped_end > section.mapped_va_end
        || function.section_offset.checked_add(SIZE)? > section.original_offset_end
    {
        return None;
    }

    let rows: [Instruction; 16] = instructions
        .range(function.mapped_entry..function.mapped_end)
        .flat_map(|(_, rows)| rows.iter().copied())
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    let [h0, h1, h2, h3, copy, cli, sub, push_sp, pushfq, push_sel, lea, push_lea, service, jump, ret, pad] =
        rows;
    let headers = [
        (h0, 5, "MOV", 2),
        (h1, 5, "MOV", 2),
        (h2, 5, "MOV", 2),
        (h3, 5, "MOV", 2),
        (copy, 3, "MOV", 2),
        (cli, 1, "CLI", 0),
        (sub, 4, "SUB", 2),
        (push_sp, 1, "PUSH", 1),
        (pushfq, 1, "PUSHFQ", 0),
        (push_sel, 2, "PUSH", 1),
        (lea, 7, "LEA", 2),
        (push_lea, 1, "PUSH", 1),
        (service, 5, "MOV", 2),
        (jump, 5, "JMP", 1),
        (ret, 1, "RET", 0),
        (pad, 13, "NOP", 1),
    ];
    if h0.0 != function.mapped_entry
        || headers.into_iter().any(|(row, size, mnemonic, count)| {
            !exact_instruction_header(row, size, mnemonic, count)
        })
        || rows
            .windows(2)
            .any(|pair| pair[0].0.checked_add(pair[0].1 as u64) != Some(pair[1].0))
        || pad.0.checked_add(pad.1 as u64) != Some(function.mapped_end)
    {
        return None;
    }

    let home_specs = [
        (h0, "RCX", "rcx", Mreg::CX, 8u8),
        (h1, "RDX", "rdx", Mreg::DX, 16u8),
        (h2, "R8", "r8", Mreg::R8, 24u8),
        (h3, "R9", "r9", Mreg::R9, 32u8),
    ];
    let mut home_stores = Vec::with_capacity(4);
    for (row, decoded, wire, _, offset) in home_specs {
        if !exact_operand_register(row.4, decoded, registers, immediates, indirects)
            || !exact_operand_memory(
                row.5,
                ("NONE", "RSP", "NONE", 1, i64::from(offset), 8),
                registers,
                immediates,
                indirects,
            )
        {
            return None;
        }
        home_stores.push(Win64HomeStore {
            instruction_address: format!("0x{:x}", row.0),
            source_register: wire,
            stack_offset: offset,
            width_bits: 64,
        });
    }
    let lea_disp = only(indirects.get(lea.4)?.iter().copied())?.4;
    if !exact_operand_register(copy.4, "RSP", registers, immediates, indirects)
        || !exact_operand_register(copy.5, "RAX", registers, immediates, indirects)
        || !exact_operand_immediate(sub.4, 16, registers, immediates, indirects)
        || !exact_operand_register(sub.5, "RSP", registers, immediates, indirects)
        || !exact_operand_register(push_sp.4, "RAX", registers, immediates, indirects)
        || !exact_operand_immediate(push_sel.4, 16, registers, immediates, indirects)
        || !exact_operand_memory(
            lea.4,
            ("NONE", "RIP", "NONE", 1, lea_disp, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(lea.5, "RAX", registers, immediates, indirects)
        || !exact_operand_register(push_lea.4, "RAX", registers, immediates, indirects)
        || !exact_operand_register(service.5, "EAX", registers, immediates, indirects)
        || !exact_operand_memory(
            pad.4,
            ("NONE", "RAX", "RAX", 1, 0, 2),
            registers,
            immediates,
            indirects,
        )
    {
        return None;
    }
    let service_number = reinterpret_imm32(exact_immediate_operand(
        service.4, registers, immediates, indirects,
    )?)?;

    let memory: [(Option<Symbol>, Option<Symbol>); 16] = [
        (None, Some(h0.5)),
        (None, Some(h1.5)),
        (None, Some(h2.5)),
        (None, Some(h3.5)),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (Some(lea.4), None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
    ];
    let registers_expected: [(&[Mreg], &[Mreg]); 16] = [
        (&[], &[Mreg::SP, Mreg::CX]),
        (&[], &[Mreg::SP, Mreg::DX]),
        (&[], &[Mreg::SP, Mreg::R8]),
        (&[], &[Mreg::SP, Mreg::R9]),
        (&[Mreg::AX], &[Mreg::SP]),
        (&[], &[]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[Mreg::SP], &[Mreg::SP, Mreg::AX]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[Mreg::AX], &[]),
        (&[Mreg::SP], &[Mreg::SP, Mreg::AX]),
        (&[Mreg::AX], &[]),
        (&[], &[]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[], &[]),
    ];
    for ((row, (read, write)), (defs, uses)) in rows.iter().zip(memory).zip(registers_expected) {
        if !exact_symbol_set(
            memory_reads.get(&row.0),
            &read.into_iter().collect::<Vec<_>>(),
        ) || !exact_symbol_set(
            memory_writes.get(&row.0),
            &write.into_iter().collect::<Vec<_>>(),
        ) || !exact_register_effects(row.0, defs, uses, decoded_defs, decoded_uses)
        {
            return None;
        }
    }

    let jump_target = u64::try_from(exact_immediate_operand(
        jump.4, registers, immediates, indirects,
    )?)
    .ok()?;
    let lea_target = lea
        .0
        .checked_add(lea.1 as u64)?
        .checked_add_signed(lea_disp)?;
    let pad_cfg = exact_or_authenticated_padding_fallthrough(cfg, pad.0, function, map);
    for (index, row) in rows.iter().copied().enumerate() {
        let row_cfg = cfg.get(&row.0).cloned().unwrap_or_default();
        if owners.get(&row.0) != Some(&BTreeSet::from([function.mapped_entry]))
            || !exact_or_empty(next, row.0, rows.get(index + 1).map(|next| next.0))
            || !exact_or_empty(direct_calls, row.0, [])
            || (row.0 == jump.0
                && (direct_jumps.get(&row.0) != Some(&BTreeSet::from([jump_target]))
                    || row_cfg != BTreeSet::from([(jump_target, "branch")])))
            || (row.0 != jump.0 && !exact_or_empty(direct_jumps, row.0, []))
            || (row.0 != jump.0 && row.0 != pad.0 && !row_cfg.is_empty())
            || (row.0 == pad.0 && !pad_cfg)
        {
            return None;
        }
    }

    let original_end = function.section_offset.checked_add(SIZE)?;
    let relocations: Vec<_> = map
        .relocations
        .iter()
        .filter(|relocation| {
            if relocation.section_index != function.section_index {
                return false;
            }
            let width = u64::from(relocation.width_bits).div_ceil(8).max(1);
            (relocation.section_offset < original_end
                && relocation.section_offset.saturating_add(width) > function.section_offset)
                || (relocation.mapped_field_va < function.mapped_end
                    && relocation.mapped_field_va.saturating_add(width) > function.mapped_entry)
        })
        .collect();
    if relocations.len() != 2 {
        return None;
    }
    let lea_reloc = only(
        relocations
            .iter()
            .copied()
            .filter(|r| r.mapped_field_va == lea.0 + 3),
    )?;
    let jump_reloc = only(
        relocations
            .iter()
            .copied()
            .filter(|r| r.mapped_field_va == jump.0 + 1),
    )?;
    let validate_reloc = |relocation: &CoffRelocationMap,
                          instruction: Instruction,
                          field_offset: u64,
                          target: Address,
                          decoded_disp: Option<i64>|
     -> Option<()> {
        let next = instruction.0.checked_add(instruction.1 as u64)?;
        let displacement = i32::try_from(i128::from(target) - i128::from(next)).ok()?;
        let encoded = i64::from(u32::try_from(relocation.encoded_value).ok()? as i32);
        let section_offset = function
            .section_offset
            .checked_add(instruction.0.checked_sub(function.mapped_entry)?)?
            .checked_add(field_offset)?;
        (relocation.section_name == section.name
            && relocation.section_offset == section_offset
            && relocation.mapped_field_va == instruction.0.checked_add(field_offset)?
            && relocation.relocation_type == "IMAGE_REL_AMD64_REL32"
            && relocation.width_bits == 32
            && relocation.target_mapped_address == target
            && encoded == i64::from(displacement)
            && decoded_disp.map_or(true, |value| value == i64::from(displacement)))
        .then_some(())
    };
    validate_reloc(lea_reloc, lea, 3, lea_target, Some(lea_disp))?;
    validate_reloc(jump_reloc, jump, 1, jump_target, None)?;
    let linkage = exact_undefined_function_external(map, lea_reloc)?;
    let dispatcher = exact_undefined_function_external(map, jump_reloc)?;
    let relocation_identity = |relocation: &CoffRelocationMap| MachineStateRelocationIdentity {
        section_index: relocation.section_index,
        section_name: relocation.section_name.clone(),
        section_offset: relocation.section_offset,
        relocation_type: relocation.relocation_type.clone(),
        width_bits: relocation.width_bits,
    };
    let push = |row: Instruction, kind, source_register, immediate_value, immediate_width_bits| {
        MachineStateStackPush {
            instruction_address: format!("0x{:x}", row.0),
            kind,
            stack_width_bits: 64,
            source_register,
            immediate_value,
            immediate_width_bits,
        }
    };

    Some(Win64KernelServiceStub {
        schema: WIN64_KERNEL_SERVICE_STUB_SCHEMA_ID,
        kind: "win64_kernel_service",
        function: MachineStateCoffFunctionIdentity {
            name: function.provider_name.clone(),
            address: format!("0x{:x}", function.mapped_entry),
            size: SIZE,
            section_index: function.section_index,
            section_name: section.name.clone(),
            section_offset: function.section_offset,
        },
        abi: Win64AbiState {
            name: "win64",
            argument_registers: ["rcx", "rdx", "r8", "r9"],
            home_kind: "four_register_arguments",
            home_stores,
        },
        entry_stack_pointer: MachineStateRegisterCopy {
            instruction_address: format!("0x{:x}", copy.0),
            source_register: "rsp",
            destination_register: "rax",
            width_bits: 64,
        },
        interrupt_control: MachineStateInterruptControl {
            instruction_address: format!("0x{:x}", cli.0),
            kind: "disable_maskable_interrupts",
        },
        stack_adjustment: MachineStateStackAdjustment {
            instruction_address: format!("0x{:x}", sub.0),
            register: "rsp",
            operation: "subtract",
            width_bits: 64,
            amount_bytes: 16,
        },
        pushes: vec![
            push(push_sp, "register_value", Some("rax"), None, None),
            push(pushfq, "flags", None, None, None),
            push(push_sel, "immediate", None, Some(16), Some(8)),
            push(push_lea, "register_value", Some("rax"), None, None),
        ],
        linkage: MachineStateRelocatedAddressLoad {
            instruction_address: format!("0x{:x}", lea.0),
            kind: "rip_relative_address",
            destination_register: "rax",
            width_bits: 64,
            target_address: format!("0x{lea_target:x}"),
            target_original_name: linkage.original_name.clone(),
            target_provider_name: linkage.provider_name.clone(),
            target_kind: "function",
            relocation: relocation_identity(lea_reloc),
        },
        service_number: MachineStateImmediateWrite {
            instruction_address: format!("0x{:x}", service.0),
            register: "eax",
            width_bits: 32,
            value: service_number,
        },
        dispatcher: MachineStateTailTransfer {
            instruction_address: format!("0x{:x}", jump.0),
            kind: "direct_tail_jump",
            target_address: format!("0x{jump_target:x}"),
            target_original_name: dispatcher.original_name.clone(),
            target_provider_name: dispatcher.provider_name.clone(),
            relocation: relocation_identity(jump_reloc),
        },
        return_instruction_address: format!("0x{:x}", ret.0),
        return_kind: "near_return",
        padding: MachineStatePadding {
            instruction_address: format!("0x{:x}", pad.0),
            kind: "nop_alignment",
            boundary: SIZE as u8,
            size: pad.1 as u8,
        },
    })
}

fn exact_win64_dword_out_getter_encoding(
    db: &DecompileDB,
    section: &crate::decompile::disassembly::coff::CoffSectionMap,
    function: &CoffFunctionMap,
    source_displacement: i64,
    source_displacement_encoding_bits: u8,
) -> bool {
    let Some(section_file_offset) = section.original_file_offset else {
        return false;
    };
    let Some(function_section_delta) = function
        .section_offset
        .checked_sub(section.original_offset_start)
    else {
        return false;
    };
    let Some(file_start) = section_file_offset.checked_add(function_section_delta) else {
        return false;
    };
    let Some(file_end) = file_start.checked_add(function.original_size) else {
        return false;
    };
    let (Ok(file_start), Ok(file_end)) = (usize::try_from(file_start), usize::try_from(file_end))
    else {
        return false;
    };
    let Some(bytes) = db
        .loaded_binary_data
        .as_deref()
        .and_then(|data| data.get(file_start..file_end))
    else {
        return false;
    };

    // These are opcode-form witnesses, not reconstruction data: the semantic
    // record continues to carry only decoded state.  Checking the original
    // relocation-free COFF bytes closes same-semantics opcode-direction and
    // redundant-REX alternatives which instruction width alone cannot prove.
    const PREFIX: &[u8] = &[
        0x4c, 0x8b, 0xdc, // mov r11, rsp
        0x48, 0x8b, 0xc1, // mov rax, rcx
        0x48, 0x8b, 0xc2, // mov rax, rdx
        0x33, 0xc0, // xor eax, eax
        0x49, 0x89, 0x53, 0x10, // mov [r11+10h], rdx
        0x49, 0x89, 0x4b, 0x08, // mov [r11+8], rcx
    ];
    const SUFFIX: &[u8] = &[0x89, 0x0a, 0xc3]; // mov [rdx], ecx; ret
    if !bytes.starts_with(PREFIX) || !bytes.ends_with(SUFFIX) {
        return false;
    }
    let load = &bytes[PREFIX.len()..bytes.len() - SUFFIX.len()];
    match source_displacement_encoding_bits {
        8 => {
            let Ok(displacement) = i8::try_from(source_displacement) else {
                return false;
            };
            load == [0x8b, 0x49, displacement as u8]
        }
        32 => {
            let Ok(displacement) = i32::try_from(source_displacement) else {
                return false;
            };
            load.len() == 6 && load[..2] == [0x8b, 0x89] && load[2..] == displacement.to_le_bytes()
        }
        _ => false,
    }
}

/// Recognize one closed Win64 dword out-getter.  Selection uses decoded
/// operands/effects, exact linear ownership/CFG, the original COFF map, and an
/// exact non-serialized opcode-form witness from the relocation-free input.
/// Names, mapped addresses, and corpus membership do not participate.  The
/// accepted source displacement is signed, and its decoded instruction width
/// must prove the canonical disp8 or disp32 encoding.
#[allow(clippy::too_many_arguments)]
fn recognize_win64_dword_out_getter(
    db: &DecompileDB,
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
) -> Option<Win64DwordOutGetterStub> {
    if map.schema != "manifold.coff-address-map.v1"
        || map.loader_id != "amd64-coff-image-v1"
        || map.architecture != "x86_64-pc-windows-msvc"
        || !matches!(function.original_size, 25 | 28)
        || function.manifold_size != function.original_size
        || function.mapped_end != function.mapped_entry.checked_add(function.original_size)?
    {
        return None;
    }

    // Reject aliased or duplicated ownership identities instead of choosing a
    // convenient name/address half of an ambiguous COFF mapping.
    let mapped_function = only(map.functions.iter().filter(|candidate| {
        candidate.provider_name == function.provider_name
            || candidate.mapped_entry == function.mapped_entry
            || (candidate.section_index == function.section_index
                && candidate.section_offset == function.section_offset)
    }))?;
    if !std::ptr::eq(mapped_function, function) {
        return None;
    }
    let section = only(
        map.sections
            .iter()
            .filter(|section| section.index == function.section_index),
    )?;
    let original_section_size = section
        .original_offset_end
        .checked_sub(section.original_offset_start)?;
    let mapped_section_size = section.mapped_va_end.checked_sub(section.mapped_va_start)?;
    let original_function_delta = function
        .section_offset
        .checked_sub(section.original_offset_start)?;
    let mapped_function_delta = function.mapped_entry.checked_sub(section.mapped_va_start)?;
    if section.kind != "Text"
        || original_section_size != mapped_section_size
        || original_function_delta != mapped_function_delta
        || function
            .section_offset
            .checked_add(function.original_size)?
            > section.original_offset_end
        || function.mapped_end > section.mapped_va_end
    {
        return None;
    }
    let function_symbol = only(map.symbols.iter().filter(|symbol| {
        symbol.original_name == function.original_name
            || symbol.provider_name == function.provider_name
            || symbol.mapped_address == function.mapped_entry
    }))?;
    if function_symbol.original_name != function.original_name
        || function_symbol.provider_name != function.provider_name
        || function_symbol.kind != "function"
        || !function_symbol.defined
        || function_symbol.section_index != Some(function.section_index)
        || function_symbol.section_offset != Some(function.section_offset)
        || function_symbol.mapped_address != function.mapped_entry
    {
        return None;
    }

    let original_end = function
        .section_offset
        .checked_add(function.original_size)?;
    if map.relocations.iter().any(|relocation| {
        let width = u64::from(relocation.width_bits).div_ceil(8).max(1);
        let Some(mapped_relocation_end) = relocation.mapped_field_va.checked_add(width) else {
            return true;
        };
        if relocation.mapped_field_va < function.mapped_end
            && mapped_relocation_end > function.mapped_entry
        {
            return true;
        }
        if relocation.section_index != function.section_index {
            return false;
        }
        if relocation.section_name != section.name {
            return true;
        }
        let Some(original_relocation_end) = relocation.section_offset.checked_add(width) else {
            return true;
        };
        relocation.section_offset < original_end
            && original_relocation_end > function.section_offset
    }) {
        return None;
    }

    let rows: [Instruction; 9] = instructions
        .range(function.mapped_entry..function.mapped_end)
        .flat_map(|(_, rows)| rows.iter().copied())
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    let [entry_sp, copy_rcx, copy_rdx, zero, home_rdx, home_rcx, load, store, ret] = rows;
    let source_operand = only(indirects.get(load.4)?.iter().copied())?;
    let source_displacement = source_operand.4;
    let source_displacement_encoding_bits = match load.1 {
        3 if source_displacement != 0 && i8::try_from(source_displacement).is_ok() => 8,
        6 if i32::try_from(source_displacement).is_ok()
            && i8::try_from(source_displacement).is_err() =>
        {
            32
        }
        _ => return None,
    };
    if function.original_size != 22 + load.1 as u64 {
        return None;
    }
    if !exact_win64_dword_out_getter_encoding(
        db,
        section,
        function,
        source_displacement,
        source_displacement_encoding_bits,
    ) {
        return None;
    }
    let headers = [
        (entry_sp, 3, "MOV", 2),
        (copy_rcx, 3, "MOV", 2),
        (copy_rdx, 3, "MOV", 2),
        (zero, 2, "XOR", 2),
        (home_rdx, 4, "MOV", 2),
        (home_rcx, 4, "MOV", 2),
        (load, load.1, "MOV", 2),
        (store, 2, "MOV", 2),
        (ret, 1, "RET", 0),
    ];
    if entry_sp.0 != function.mapped_entry
        || headers.into_iter().any(|(row, size, mnemonic, count)| {
            !exact_instruction_header(row, size, mnemonic, count)
        })
        || rows
            .windows(2)
            .any(|pair| pair[0].0.checked_add(pair[0].1 as u64) != Some(pair[1].0))
        || ret.0.checked_add(ret.1 as u64) != Some(function.mapped_end)
    {
        return None;
    }

    if !exact_operand_register(entry_sp.4, "RSP", registers, immediates, indirects)
        || !exact_operand_register(entry_sp.5, "R11", registers, immediates, indirects)
        || !exact_operand_register(copy_rcx.4, "RCX", registers, immediates, indirects)
        || !exact_operand_register(copy_rcx.5, "RAX", registers, immediates, indirects)
        || !exact_operand_register(copy_rdx.4, "RDX", registers, immediates, indirects)
        || !exact_operand_register(copy_rdx.5, "RAX", registers, immediates, indirects)
        || !exact_operand_register(zero.4, "EAX", registers, immediates, indirects)
        || !exact_operand_register(zero.5, "EAX", registers, immediates, indirects)
        || !exact_operand_register(home_rdx.4, "RDX", registers, immediates, indirects)
        || !exact_operand_memory(
            home_rdx.5,
            ("NONE", "R11", "NONE", 1, 16, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(home_rcx.4, "RCX", registers, immediates, indirects)
        || !exact_operand_memory(
            home_rcx.5,
            ("NONE", "R11", "NONE", 1, 8, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_memory(
            load.4,
            ("NONE", "RCX", "NONE", 1, source_displacement, 4),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(load.5, "ECX", registers, immediates, indirects)
        || !exact_operand_register(store.4, "ECX", registers, immediates, indirects)
        || !exact_operand_memory(
            store.5,
            ("NONE", "RDX", "NONE", 1, 0, 4),
            registers,
            immediates,
            indirects,
        )
    {
        return None;
    }

    let memory: [(Option<Symbol>, Option<Symbol>); 9] = [
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (None, Some(home_rdx.5)),
        (None, Some(home_rcx.5)),
        (Some(load.4), None),
        (None, Some(store.5)),
        (None, None),
    ];
    let register_effects: [(&[Mreg], &[Mreg]); 9] = [
        (&[Mreg::R11], &[Mreg::SP]),
        (&[Mreg::AX], &[Mreg::CX]),
        (&[Mreg::AX], &[Mreg::DX]),
        (&[Mreg::AX], &[]),
        (&[], &[Mreg::R11, Mreg::DX]),
        (&[], &[Mreg::R11, Mreg::CX]),
        (&[Mreg::CX], &[Mreg::CX]),
        (&[], &[Mreg::DX, Mreg::CX]),
        (&[Mreg::SP], &[Mreg::SP]),
    ];
    for ((row, (read, write)), (defs, uses)) in rows.iter().zip(memory).zip(register_effects) {
        if !exact_symbol_set(
            memory_reads.get(&row.0),
            &read.into_iter().collect::<Vec<_>>(),
        ) || !exact_symbol_set(
            memory_writes.get(&row.0),
            &write.into_iter().collect::<Vec<_>>(),
        ) || !exact_register_effects(row.0, defs, uses, decoded_defs, decoded_uses)
        {
            return None;
        }
    }

    for (index, row) in rows.iter().copied().enumerate() {
        if owners.get(&row.0) != Some(&BTreeSet::from([function.mapped_entry]))
            || !exact_or_empty(next, row.0, rows.get(index + 1).map(|next| next.0))
            || !exact_or_empty(direct_jumps, row.0, [])
            || !exact_or_empty(direct_calls, row.0, [])
            || !exact_or_empty(cfg, row.0, [])
            || !exact_or_empty(flags_and_jumps, row.0, [])
        {
            return None;
        }
    }

    Some(Win64DwordOutGetterStub {
        schema: WIN64_DWORD_OUT_GETTER_STUB_SCHEMA_ID,
        kind: "win64_dword_out_getter",
        function: MachineStateCoffFunctionIdentity {
            name: function.provider_name.clone(),
            address: format!("0x{:x}", function.mapped_entry),
            size: function.original_size,
            section_index: function.section_index,
            section_name: section.name.clone(),
            section_offset: function.section_offset,
        },
        abi: Win64AbiState {
            name: "win64",
            argument_registers: ["rcx", "rdx", "r8", "r9"],
            home_kind: "first_two_register_arguments_via_entry_sp_alias",
            home_stores: vec![
                Win64HomeStore {
                    instruction_address: format!("0x{:x}", home_rdx.0),
                    source_register: "rdx",
                    stack_offset: 16,
                    width_bits: 64,
                },
                Win64HomeStore {
                    instruction_address: format!("0x{:x}", home_rcx.0),
                    source_register: "rcx",
                    stack_offset: 8,
                    width_bits: 64,
                },
            ],
        },
        entry_stack_pointer: MachineStateRegisterCopy {
            instruction_address: format!("0x{:x}", entry_sp.0),
            source_register: "rsp",
            destination_register: "r11",
            width_bits: 64,
        },
        register_copies: vec![
            MachineStateRegisterCopy {
                instruction_address: format!("0x{:x}", copy_rcx.0),
                source_register: "rcx",
                destination_register: "rax",
                width_bits: 64,
            },
            MachineStateRegisterCopy {
                instruction_address: format!("0x{:x}", copy_rdx.0),
                source_register: "rdx",
                destination_register: "rax",
                width_bits: 64,
            },
        ],
        result_zero: MachineStateRegisterZero {
            instruction_address: format!("0x{:x}", zero.0),
            register: "eax",
            width_bits: 32,
            kind: "xor_self",
        },
        home_base_register: "r11",
        field_copy: MachineStateDwordFieldCopy {
            load_instruction_address: format!("0x{:x}", load.0),
            source_base_register: "rcx",
            source_displacement,
            source_displacement_encoding_bits,
            temporary_register: "ecx",
            store_instruction_address: format!("0x{:x}", store.0),
            destination_base_register: "rdx",
            destination_displacement: 0,
            width_bits: 32,
        },
        return_instruction_address: format!("0x{:x}", ret.0),
        return_kind: "near_return",
        padding: MachineStatePadding {
            instruction_address: format!("0x{:x}", function.mapped_end),
            kind: "none",
            boundary: 1,
            size: 0,
        },
        relocations: Vec::new(),
    })
}

/// Recognize one complete Win64 forwarding contract whose indirect target is
/// checked through a relocation-backed pointer and then tail-transferred with
/// the four incoming volatile integer registers restored.
///
/// The accepted sequence is deliberately closed: one linear owned block,
/// exact stack slots, exact register/memory effects, one indirect guard call,
/// and one register-indirect tail exit.  The guard identity comes only from the
/// original COFF import-pointer relocation.  Symbol spelling, mapped address,
/// corpus membership, and raw instruction bytes never participate.
#[allow(clippy::too_many_arguments)]
fn recognize_guarded_indirect_forward_tail(
    function: &CoffFunctionMap,
    map: &CoffAddressMap,
    instructions: &BTreeMap<Address, Vec<Instruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediates: &BTreeMap<Symbol, BTreeSet<i64>>,
    indirects: &BTreeMap<Symbol, BTreeSet<IndirectOperand>>,
    memory_reads: &BTreeMap<Node, BTreeSet<Symbol>>,
    memory_writes: &BTreeMap<Node, BTreeSet<Symbol>>,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    next: &BTreeMap<Node, BTreeSet<Node>>,
    direct_jumps: &BTreeMap<Node, BTreeSet<Address>>,
    direct_calls: &BTreeMap<Node, BTreeSet<Address>>,
    cfg: &BTreeMap<Node, BTreeSet<(Address, Symbol)>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
) -> Option<GuardedIndirectForwardTailStub> {
    let rows: [Instruction; 16] = instructions
        .range(function.mapped_entry..function.mapped_end)
        .flat_map(|(_, rows)| rows.iter().copied())
        .collect::<Vec<_>>()
        .try_into()
        .ok()?;
    let [sub, save_rdx, save_r8, save_r9, load_base, save_base, load_table, load_target, guard, preserve_target, restore_rdx, restore_r8, restore_r9, restore_base, add, jump] =
        rows;

    if function.original_size != function.manifold_size
        || sub.0 != function.mapped_entry
        || !exact_instruction_header(sub, 4, "SUB", 2)
        || !exact_instruction_header(save_rdx, 5, "MOV", 2)
        || !exact_instruction_header(save_r8, 5, "MOV", 2)
        || !exact_instruction_header(save_r9, 5, "MOV", 2)
        || !exact_instruction_header(load_base, 4, "MOV", 2)
        || !exact_instruction_header(save_base, 5, "MOV", 2)
        || !exact_instruction_header(load_table, 3, "MOV", 2)
        || !matches!(load_target.1, 4 | 7)
        || !exact_instruction_header(load_target, load_target.1, "MOV", 2)
        || !exact_instruction_header(guard, 6, "CALL", 1)
        || !exact_instruction_header(preserve_target, 3, "MOV", 2)
        || !exact_instruction_header(restore_rdx, 5, "MOV", 2)
        || !exact_instruction_header(restore_r8, 5, "MOV", 2)
        || !exact_instruction_header(restore_r9, 5, "MOV", 2)
        || !exact_instruction_header(restore_base, 5, "MOV", 2)
        || !exact_instruction_header(add, 4, "ADD", 2)
        || !exact_instruction_header(jump, 3, "JMP", 1)
    {
        return None;
    }

    for pair in rows.windows(2) {
        if pair[0].0.checked_add(pair[0].1 as u64) != Some(pair[1].0)
            || next.get(&pair[0].0) != Some(&BTreeSet::from([pair[1].0]))
        {
            return None;
        }
    }
    if jump.0.checked_add(jump.1 as u64) != Some(function.mapped_end)
        || next.get(&jump.0).is_some_and(|targets| !targets.is_empty())
        || rows.iter().any(|instruction| {
            owners.get(&instruction.0) != Some(&BTreeSet::from([function.mapped_entry]))
                || direct_calls
                    .get(&instruction.0)
                    .is_some_and(|targets| !targets.is_empty())
                || direct_jumps
                    .get(&instruction.0)
                    .is_some_and(|targets| !targets.is_empty())
        })
    {
        return None;
    }

    // Capstone sees the loader-patched image, so retain the decoded signed
    // REL32 displacement and authenticate it against the COFF map below.
    let guard_displacement = only(indirects.get(guard.4)?.iter().copied())?.4;
    if !exact_operand_register(sub.5, "RSP", registers, immediates, indirects)
        || !exact_operand_immediate(sub.4, 72, registers, immediates, indirects)
        || !exact_operand_register(save_rdx.4, "RDX", registers, immediates, indirects)
        || !exact_operand_memory(
            save_rdx.5,
            ("NONE", "RSP", "NONE", 1, 40, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(save_r8.4, "R8", registers, immediates, indirects)
        || !exact_operand_memory(
            save_r8.5,
            ("NONE", "RSP", "NONE", 1, 48, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(save_r9.4, "R9", registers, immediates, indirects)
        || !exact_operand_memory(
            save_r9.5,
            ("NONE", "RSP", "NONE", 1, 56, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_memory(
            load_base.4,
            ("NONE", "RCX", "NONE", 1, 32, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(load_base.5, "RCX", registers, immediates, indirects)
        || !exact_operand_register(save_base.4, "RCX", registers, immediates, indirects)
        || !exact_operand_memory(
            save_base.5,
            ("NONE", "RSP", "NONE", 1, 32, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_memory(
            load_table.4,
            ("NONE", "RCX", "NONE", 1, 0, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(load_table.5, "R10", registers, immediates, indirects)
    {
        return None;
    }

    let (_, base, index, scale, displacement, width) =
        only(indirects.get(load_target.4)?.iter().copied())?;
    let displacement = u32::try_from(displacement).ok()?;
    let expected_target_size = if displacement <= i8::MAX as u32 { 4 } else { 7 };
    if base != "R10"
        || index != "NONE"
        || scale != 1
        || width != 8
        || displacement == 0
        || displacement > i32::MAX as u32
        || displacement % 8 != 0
        || load_target.1 != expected_target_size
        || !exact_operand_memory(
            load_target.4,
            ("NONE", "R10", "NONE", 1, i64::from(displacement), 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(load_target.5, "RCX", registers, immediates, indirects)
        || !exact_operand_memory(
            guard.4,
            ("NONE", "RIP", "NONE", 1, guard_displacement, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(preserve_target.4, "RCX", registers, immediates, indirects)
        || !exact_operand_register(preserve_target.5, "RAX", registers, immediates, indirects)
        || !exact_operand_memory(
            restore_rdx.4,
            ("NONE", "RSP", "NONE", 1, 40, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(restore_rdx.5, "RDX", registers, immediates, indirects)
        || !exact_operand_memory(
            restore_r8.4,
            ("NONE", "RSP", "NONE", 1, 48, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(restore_r8.5, "R8", registers, immediates, indirects)
        || !exact_operand_memory(
            restore_r9.4,
            ("NONE", "RSP", "NONE", 1, 56, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(restore_r9.5, "R9", registers, immediates, indirects)
        || !exact_operand_memory(
            restore_base.4,
            ("NONE", "RSP", "NONE", 1, 32, 8),
            registers,
            immediates,
            indirects,
        )
        || !exact_operand_register(restore_base.5, "RCX", registers, immediates, indirects)
        || !exact_operand_register(add.5, "RSP", registers, immediates, indirects)
        || !exact_operand_immediate(add.4, 72, registers, immediates, indirects)
        || !exact_operand_register(jump.4, "RAX", registers, immediates, indirects)
    {
        return None;
    }

    let memory_contract: [(Option<Symbol>, Option<Symbol>); 16] = [
        (None, None),
        (None, Some(save_rdx.5)),
        (None, Some(save_r8.5)),
        (None, Some(save_r9.5)),
        (Some(load_base.4), None),
        (None, Some(save_base.5)),
        (Some(load_table.4), None),
        (Some(load_target.4), None),
        (Some(guard.4), None),
        (None, None),
        (Some(restore_rdx.4), None),
        (Some(restore_r8.4), None),
        (Some(restore_r9.4), None),
        (Some(restore_base.4), None),
        (None, None),
        (None, None),
    ];
    for (instruction, (read, write)) in rows.iter().zip(memory_contract) {
        if !exact_symbol_set(
            memory_reads.get(&instruction.0),
            &read.into_iter().collect::<Vec<_>>(),
        ) || !exact_symbol_set(
            memory_writes.get(&instruction.0),
            &write.into_iter().collect::<Vec<_>>(),
        ) {
            return None;
        }
    }

    let register_contract: [(&[Mreg], &[Mreg]); 16] = [
        (&[Mreg::SP], &[Mreg::SP]),
        (&[], &[Mreg::SP, Mreg::DX]),
        (&[], &[Mreg::SP, Mreg::R8]),
        (&[], &[Mreg::SP, Mreg::R9]),
        (&[Mreg::CX], &[Mreg::CX]),
        (&[], &[Mreg::SP, Mreg::CX]),
        (&[Mreg::R10], &[Mreg::CX]),
        (&[Mreg::CX], &[Mreg::R10]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[Mreg::AX], &[Mreg::CX]),
        (&[Mreg::DX], &[Mreg::SP]),
        (&[Mreg::R8], &[Mreg::SP]),
        (&[Mreg::R9], &[Mreg::SP]),
        (&[Mreg::CX], &[Mreg::SP]),
        (&[Mreg::SP], &[Mreg::SP]),
        (&[], &[Mreg::AX]),
    ];
    if rows
        .iter()
        .zip(register_contract)
        .any(|(instruction, (defs, uses))| {
            !exact_register_effects(instruction.0, defs, uses, decoded_defs, decoded_uses)
        })
    {
        return None;
    }

    for instruction in rows {
        let expected = if instruction.0 == guard.0 {
            BTreeSet::from([(0, "indirect_call"), (preserve_target.0, "fallthrough")])
        } else if instruction.0 == jump.0 {
            BTreeSet::from([(0, "indirect")])
        } else {
            BTreeSet::new()
        };
        if cfg
            .get(&instruction.0)
            .map_or(!expected.is_empty(), |actual| actual != &expected)
        {
            return None;
        }
    }

    let relocations: Vec<_> = map
        .relocations
        .iter()
        .filter(|relocation| {
            relocation.section_index == function.section_index
                && relocation.mapped_field_va >= function.mapped_entry
                && relocation.mapped_field_va < function.mapped_end
        })
        .collect();
    let relocation = only(relocations)?;
    let sections: Vec<_> = map
        .sections
        .iter()
        .filter(|section| {
            section.index == function.section_index && section.name == relocation.section_name
        })
        .collect();
    let section = only(sections)?;
    let encoded_guard_displacement =
        i64::from(u32::try_from(relocation.encoded_value).ok()? as i32);
    let decoded_guard_target = guard
        .0
        .checked_add(guard.1 as u64)?
        .checked_add_signed(guard_displacement)?;
    if relocation.section_name != section.name
        || relocation.mapped_field_va != guard.0 + 2
        || relocation.section_offset
            != function
                .section_offset
                .checked_add(guard.0.checked_sub(function.mapped_entry)?)?
                .checked_add(2)?
        || relocation.relocation_type != "IMAGE_REL_AMD64_REL32"
        || relocation.width_bits != 32
        || guard_displacement != encoded_guard_displacement
        || decoded_guard_target != relocation.target_mapped_address
    {
        return None;
    }

    let guard_targets: Vec<_> = map
        .externs
        .iter()
        .filter(|external| {
            external.original_name == relocation.target_original_name
                && external.synthetic_address == relocation.target_mapped_address
                && external.kind == CoffExternalKind::ImportPointer
        })
        .collect();
    let guard_target = only(guard_targets)?;

    Some(GuardedIndirectForwardTailStub {
        schema: GUARDED_INDIRECT_FORWARD_TAIL_SCHEMA_ID,
        kind: "guarded_indirect_forward_tail",
        function: MachineStateFunctionIdentity {
            name: function.provider_name.clone(),
            address: format!("0x{:x}", function.mapped_entry),
        },
        forwarding: MachineStateForwardingContract {
            input_registers: ["rcx", "rdx", "r8", "r9"],
            stack_allocation_bytes: 72,
            base_load_displacement: 32,
            dispatch_load: MachineStateDispatchLoad {
                instruction_address: format!("0x{:x}", load_target.0),
                target_register: "rcx",
                table_register: "r10",
                displacement,
                width_bits: 64,
            },
        },
        guard: MachineStateGuardCall {
            instruction_address: format!("0x{:x}", guard.0),
            kind: "indirect_call_through_relocated_pointer",
            argument_register: "rcx",
            result_register: "rcx",
            target_address: format!("0x{:x}", guard_target.synthetic_address),
            target_original_name: guard_target.original_name.clone(),
            target_provider_name: guard_target.provider_name.clone(),
            target_kind: "import_pointer",
            relocation: MachineStateRelocationIdentity {
                section_index: relocation.section_index,
                section_name: relocation.section_name.clone(),
                section_offset: relocation.section_offset,
                relocation_type: relocation.relocation_type.clone(),
                width_bits: relocation.width_bits,
            },
        },
        transfer: MachineStateRegisterTailTransfer {
            instruction_address: format!("0x{:x}", jump.0),
            kind: "register_indirect_tail_jump",
            target_register: "rax",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompile::disassembly::coff::{
        CoffExternalMap, CoffFunctionMap, CoffRelocationMap, CoffSectionMap, CoffSymbolMap,
    };
    use crate::decompile::passes::asm_pass::AsmPass;
    use crate::decompile::passes::pass::IRPass;

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

    const KH0: Symbol = "kernel_home_0";
    const KH1: Symbol = "kernel_home_1";
    const KH2: Symbol = "kernel_home_2";
    const KH3: Symbol = "kernel_home_3";
    const KM0: Symbol = "kernel_mem_0";
    const KM1: Symbol = "kernel_mem_1";
    const KM2: Symbol = "kernel_mem_2";
    const KM3: Symbol = "kernel_mem_3";
    const KRSP: Symbol = "kernel_rsp";
    const KRAX: Symbol = "kernel_rax";
    const KEAX: Symbol = "kernel_eax";
    const KSUB: Symbol = "kernel_sub";
    const KSEL: Symbol = "kernel_selector";
    const KLEA: Symbol = "kernel_lea";
    const KSVC: Symbol = "kernel_service";
    const KJMP: Symbol = "kernel_jump";
    const KPAD: Symbol = "kernel_padding";

    const ORSP: Symbol = "out_getter_rsp";
    const OR11: Symbol = "out_getter_r11";
    const ORCX: Symbol = "out_getter_rcx";
    const ORDX: Symbol = "out_getter_rdx";
    const ORAX: Symbol = "out_getter_rax";
    const OEAX: Symbol = "out_getter_eax";
    const OECX: Symbol = "out_getter_ecx";
    const OHOME16: Symbol = "out_getter_home_16";
    const OHOME8: Symbol = "out_getter_home_8";
    const OFIELD: Symbol = "out_getter_field";
    const OOUT: Symbol = "out_getter_out";

    macro_rules! set_rows {
        ($db:expr, $relation:literal, $rows:expr) => {
            $db.rel_set(
                $relation,
                $rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            )
        };
    }

    fn fixture() -> (DecompileDB, CoffAddressMap) {
        let mut db = DecompileDB::default();
        let entry = 0x1000_0000;
        let jump = entry + 6;
        let target = 0x1000_2000;
        let rows: Vec<Instruction> = vec![
            (entry, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
            (jump, 5, "", "JMP", TARGET, NONE, NONE, NONE, 0, 0),
        ];
        db.rel_set(
            "unrefinedinstruction",
            rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
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
            function_boundary_sidecar_sha256: None,
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
        let mut indirects: Vec<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )> = vec![(TEST_MEMORY, "NONE", "NONE", "NONE", 1, 0x7ffe_0308, 1)];
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
            function_boundary_sidecar_sha256: None,
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

    fn kernel_service_fixture() -> (DecompileDB, CoffAddressMap) {
        let mut db = DecompileDB::default();
        let entry = 0x1000_0000_u64;
        let link_target = 0x1000_8010_u64;
        let dispatch_target = 0x1000_8000_u64;
        let ins = |offset, size, mnemonic, op1, op2| -> Instruction {
            (
                entry + offset,
                size,
                "",
                mnemonic,
                op1,
                op2,
                NONE,
                NONE,
                0,
                0,
            )
        };
        let rows = [
            ins(0x00, 5, "MOV", KH0, KM0),
            ins(0x05, 5, "MOV", KH1, KM1),
            ins(0x0a, 5, "MOV", KH2, KM2),
            ins(0x0f, 5, "MOV", KH3, KM3),
            ins(0x14, 3, "MOV", KRSP, KRAX),
            ins(0x17, 1, "CLI", NONE, NONE),
            ins(0x18, 4, "SUB", KSUB, KRSP),
            ins(0x1c, 1, "PUSH", KRAX, NONE),
            ins(0x1d, 1, "PUSHFQ", NONE, NONE),
            ins(0x1e, 2, "PUSH", KSEL, NONE),
            ins(0x20, 7, "LEA", KLEA, KRAX),
            ins(0x27, 1, "PUSH", KRAX, NONE),
            ins(0x28, 5, "MOV", KSVC, KEAX),
            ins(0x2d, 5, "JMP", KJMP, NONE),
            ins(0x32, 1, "RET", NONE, NONE),
            ins(0x33, 13, "NOP", KPAD, NONE),
        ];
        let addresses = rows.map(|row| row.0);
        let link_disp = i64::try_from(link_target - (entry + 0x27)).unwrap();
        let dispatch_disp = i64::try_from(dispatch_target - (entry + 0x32)).unwrap();

        set_rows!(db, "unrefinedinstruction", rows);
        set_rows!(
            db,
            "op_register",
            [
                (KH0, "RCX"),
                (KH1, "RDX"),
                (KH2, "R8"),
                (KH3, "R9"),
                (KRSP, "RSP"),
                (KRAX, "RAX"),
                (KEAX, "EAX"),
            ]
        );
        set_rows!(
            db,
            "op_immediate",
            [
                (KSUB, 16, 0_usize),
                (KSEL, 16, 0),
                (KSVC, 0x89ab_cdef_i64, 0),
                (KJMP, dispatch_target as i64, 0)
            ]
        );
        set_rows!(
            db,
            "op_indirect",
            [
                (KM0, "NONE", "RSP", "NONE", 1_i64, 8, 8_usize),
                (KM1, "NONE", "RSP", "NONE", 1, 16, 8),
                (KM2, "NONE", "RSP", "NONE", 1, 24, 8),
                (KM3, "NONE", "RSP", "NONE", 1, 32, 8),
                (KLEA, "NONE", "RIP", "NONE", 1, link_disp, 8),
                (KPAD, "NONE", "RAX", "RAX", 1, 0, 2),
            ]
        );
        set_rows!(
            db,
            "instr_in_function",
            addresses.map(|address| (address, entry))
        );
        set_rows!(
            db,
            "next",
            addresses.windows(2).map(|pair| (pair[0], pair[1]))
        );
        set_rows!(db, "direct_jump", [(entry + 0x2d, dispatch_target)]);
        set_rows!(db, "direct_call", Vec::<(Node, Address)>::new());
        set_rows!(
            db,
            "ddisasm_cfg_edge",
            [(entry + 0x2d, dispatch_target, "branch")]
        );
        set_rows!(db, "decoded_memory_read_operand", [(entry + 0x20, KLEA)]);
        set_rows!(
            db,
            "decoded_memory_write_operand",
            [
                (entry, KM0),
                (entry + 5, KM1),
                (entry + 10, KM2),
                (entry + 15, KM3)
            ]
        );
        set_rows!(
            db,
            "decoded_reg_def",
            [
                (entry + 0x14, Mreg::AX),
                (entry + 0x18, Mreg::SP),
                (entry + 0x1c, Mreg::SP),
                (entry + 0x1d, Mreg::SP),
                (entry + 0x1e, Mreg::SP),
                (entry + 0x20, Mreg::AX),
                (entry + 0x27, Mreg::SP),
                (entry + 0x28, Mreg::AX),
                (entry + 0x32, Mreg::SP),
            ]
        );
        set_rows!(
            db,
            "decoded_reg_use",
            [
                (entry, Mreg::SP),
                (entry, Mreg::CX),
                (entry + 5, Mreg::SP),
                (entry + 5, Mreg::DX),
                (entry + 10, Mreg::SP),
                (entry + 10, Mreg::R8),
                (entry + 15, Mreg::SP),
                (entry + 15, Mreg::R9),
                (entry + 0x14, Mreg::SP),
                (entry + 0x18, Mreg::SP),
                (entry + 0x1c, Mreg::SP),
                (entry + 0x1c, Mreg::AX),
                (entry + 0x1d, Mreg::SP),
                (entry + 0x1e, Mreg::SP),
                (entry + 0x27, Mreg::SP),
                (entry + 0x27, Mreg::AX),
                (entry + 0x32, Mreg::SP),
            ]
        );

        let external_symbol = |original: &str, provider: &str, address| CoffSymbolMap {
            original_name: original.into(),
            provider_name: provider.into(),
            kind: "function".into(),
            defined: false,
            section_index: None,
            section_offset: None,
            mapped_address: address,
        };
        let external = |original: &str, provider: &str, address| CoffExternalMap {
            original_name: original.into(),
            provider_name: provider.into(),
            kind: CoffExternalKind::Function,
            synthetic_address: address,
        };
        let relocation = |offset, original: &str, target, encoded| CoffRelocationMap {
            section_index: 1,
            section_name: ".text$generic".into(),
            section_offset: offset,
            mapped_field_va: entry + offset,
            relocation_type: "IMAGE_REL_AMD64_REL32".into(),
            width_bits: 32,
            target_original_name: original.into(),
            target_mapped_address: target,
            encoded_value: encoded as u64,
        };
        let map = CoffAddressMap {
            schema: "manifold.coff-address-map.v1",
            loader_id: "amd64-coff-image-v1",
            architecture: "x86_64-pc-windows-msvc",
            image_base: entry,
            function_boundary_sidecar_sha256: None,
            sections: vec![CoffSectionMap {
                index: 1,
                name: ".text$generic".into(),
                kind: "Text".into(),
                original_file_offset: Some(0x100),
                original_offset_start: 0,
                original_offset_end: 64,
                mapped_va_start: entry,
                mapped_va_end: entry + 64,
            }],
            functions: vec![CoffFunctionMap {
                original_name: "arbitrary_original".into(),
                provider_name: "coff_fn_arbitrary_provider".into(),
                section_index: 1,
                section_offset: 0,
                original_size: 64,
                manifold_size: 64,
                mapped_entry: entry,
                mapped_end: entry + 64,
            }],
            symbols: vec![
                CoffSymbolMap {
                    original_name: "arbitrary_original".into(),
                    provider_name: "coff_fn_arbitrary_provider".into(),
                    kind: "function".into(),
                    defined: true,
                    section_index: Some(1),
                    section_offset: Some(0),
                    mapped_address: entry,
                },
                external_symbol("arbitrary_link", "coff_ext_arbitrary_link", link_target),
                external_symbol(
                    "arbitrary_dispatch",
                    "coff_ext_arbitrary_dispatch",
                    dispatch_target,
                ),
            ],
            externs: vec![
                external("arbitrary_link", "coff_ext_arbitrary_link", link_target),
                external(
                    "arbitrary_dispatch",
                    "coff_ext_arbitrary_dispatch",
                    dispatch_target,
                ),
            ],
            relocations: vec![
                relocation(0x23, "arbitrary_link", link_target, link_disp),
                relocation(0x2e, "arbitrary_dispatch", dispatch_target, dispatch_disp),
            ],
        };
        (db, map)
    }

    fn dword_out_getter_fixture(
        source_displacement: i64,
        displacement_encoding_bits: u8,
    ) -> (DecompileDB, CoffAddressMap) {
        let mut db = DecompileDB::default();
        let entry = 0x1000_0000_u64;
        let load_size = match displacement_encoding_bits {
            8 => 3,
            32 => 6,
            _ => panic!("fixture displacement encoding must be disp8 or disp32"),
        };
        let function_size = 22 + load_size as u64;
        let mut loaded_binary_data = vec![0_u8; 0x200];
        loaded_binary_data.extend_from_slice(&[
            0x4c, 0x8b, 0xdc, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0xc2, 0x33, 0xc0, 0x49, 0x89, 0x53,
            0x10, 0x49, 0x89, 0x4b, 0x08,
        ]);
        match displacement_encoding_bits {
            8 => loaded_binary_data.extend_from_slice(&[0x8b, 0x49, source_displacement as u8]),
            32 => {
                loaded_binary_data.extend_from_slice(&[0x8b, 0x89]);
                loaded_binary_data.extend_from_slice(&(source_displacement as i32).to_le_bytes());
            }
            _ => unreachable!(),
        }
        loaded_binary_data.extend_from_slice(&[0x89, 0x0a, 0xc3]);
        assert_eq!(loaded_binary_data.len(), 0x200 + function_size as usize);
        db.loaded_binary_data = Some(std::sync::Arc::new(loaded_binary_data));
        let load_address = entry + 19;
        let store_address = load_address + load_size as u64;
        let return_address = store_address + 2;
        let ins = |offset, size, mnemonic, op1, op2| -> Instruction {
            (
                entry + offset,
                size,
                "",
                mnemonic,
                op1,
                op2,
                NONE,
                NONE,
                0,
                0,
            )
        };
        let rows = [
            ins(0, 3, "MOV", ORSP, OR11),
            ins(3, 3, "MOV", ORCX, ORAX),
            ins(6, 3, "MOV", ORDX, ORAX),
            ins(9, 2, "XOR", OEAX, OEAX),
            ins(11, 4, "MOV", ORDX, OHOME16),
            ins(15, 4, "MOV", ORCX, OHOME8),
            ins(19, load_size, "MOV", OFIELD, OECX),
            ins(19 + load_size as u64, 2, "MOV", OECX, OOUT),
            ins(21 + load_size as u64, 1, "RET", NONE, NONE),
        ];
        let addresses = rows.map(|row| row.0);
        set_rows!(db, "unrefinedinstruction", rows);
        set_rows!(
            db,
            "op_register",
            [
                (ORSP, "RSP"),
                (OR11, "R11"),
                (ORCX, "RCX"),
                (ORDX, "RDX"),
                (ORAX, "RAX"),
                (OEAX, "EAX"),
                (OECX, "ECX"),
            ]
        );
        set_rows!(
            db,
            "op_indirect",
            [
                (OHOME16, "NONE", "R11", "NONE", 1_i64, 16, 8_usize),
                (OHOME8, "NONE", "R11", "NONE", 1, 8, 8),
                (OFIELD, "NONE", "RCX", "NONE", 1, source_displacement, 4,),
                (OOUT, "NONE", "RDX", "NONE", 1, 0, 4),
            ]
        );
        set_rows!(
            db,
            "instr_in_function",
            addresses.map(|address| (address, entry))
        );
        set_rows!(
            db,
            "next",
            addresses.windows(2).map(|pair| (pair[0], pair[1]))
        );
        set_rows!(db, "direct_jump", Vec::<(Node, Address)>::new());
        set_rows!(db, "direct_call", Vec::<(Node, Address)>::new());
        set_rows!(
            db,
            "ddisasm_cfg_edge",
            Vec::<(Node, Address, Symbol)>::new()
        );
        set_rows!(db, "decoded_memory_read_operand", [(load_address, OFIELD)]);
        set_rows!(
            db,
            "decoded_memory_write_operand",
            [
                (entry + 11, OHOME16),
                (entry + 15, OHOME8),
                (store_address, OOUT),
            ]
        );
        set_rows!(
            db,
            "decoded_reg_def",
            [
                (entry, Mreg::R11),
                (entry + 3, Mreg::AX),
                (entry + 6, Mreg::AX),
                (entry + 9, Mreg::AX),
                (load_address, Mreg::CX),
                (return_address, Mreg::SP),
            ]
        );
        set_rows!(
            db,
            "decoded_reg_use",
            [
                (entry, Mreg::SP),
                (entry + 3, Mreg::CX),
                (entry + 6, Mreg::DX),
                (entry + 11, Mreg::R11),
                (entry + 11, Mreg::DX),
                (entry + 15, Mreg::R11),
                (entry + 15, Mreg::CX),
                (load_address, Mreg::CX),
                (store_address, Mreg::DX),
                (store_address, Mreg::CX),
                (return_address, Mreg::SP),
            ]
        );

        let map = CoffAddressMap {
            schema: "manifold.coff-address-map.v1",
            loader_id: "amd64-coff-image-v1",
            architecture: "x86_64-pc-windows-msvc",
            image_base: entry,
            function_boundary_sidecar_sha256: None,
            sections: vec![CoffSectionMap {
                index: 5,
                name: ".text$generic".into(),
                kind: "Text".into(),
                original_file_offset: Some(0x200),
                original_offset_start: 0x40,
                original_offset_end: 0x40 + function_size,
                mapped_va_start: entry,
                mapped_va_end: entry + function_size,
            }],
            functions: vec![CoffFunctionMap {
                original_name: "arbitrary_original_out_getter".into(),
                provider_name: "coff_fn_arbitrary_out_getter".into(),
                section_index: 5,
                section_offset: 0x40,
                original_size: function_size,
                manifold_size: function_size,
                mapped_entry: entry,
                mapped_end: entry + function_size,
            }],
            symbols: vec![CoffSymbolMap {
                original_name: "arbitrary_original_out_getter".into(),
                provider_name: "coff_fn_arbitrary_out_getter".into(),
                kind: "function".into(),
                defined: true,
                section_index: Some(5),
                section_offset: Some(0x40),
                mapped_address: entry,
            }],
            externs: Vec::new(),
            relocations: Vec::new(),
        };
        (db, map)
    }

    fn rewrite_dword_out_getter_input_byte(db: &mut DecompileDB, offset: usize, value: u8) {
        let data = std::sync::Arc::make_mut(
            db.loaded_binary_data
                .as_mut()
                .expect("dword out-getter fixture input bytes"),
        );
        data[0x200 + offset] = value;
    }

    fn rewrite_instruction(
        db: &mut DecompileDB,
        address: Address,
        rewrite: impl Fn(Instruction) -> Instruction,
    ) {
        let mut count = 0;
        let rows = db
            .rel_iter::<Instruction>("unrefinedinstruction")
            .map(|row| {
                if row.0 == address {
                    count += 1;
                    rewrite(*row)
                } else {
                    *row
                }
            })
            .collect::<ascent::boxcar::Vec<_>>();
        assert_eq!(count, 1);
        db.rel_set("unrefinedinstruction", rows);
    }

    fn replace_register(db: &mut DecompileDB, operand: Symbol, value: &'static str) {
        let mut count = 0;
        let rows = db
            .rel_iter::<(Symbol, &'static str)>("op_register")
            .map(|row| {
                if row.0 == operand {
                    count += 1;
                    (row.0, value)
                } else {
                    *row
                }
            })
            .collect::<ascent::boxcar::Vec<_>>();
        assert_eq!(count, 1);
        db.rel_set("op_register", rows);
    }

    fn replace_immediate(db: &mut DecompileDB, operand: Symbol, value: i64) {
        let mut count = 0;
        let rows = db
            .rel_iter::<(Symbol, i64, usize)>("op_immediate")
            .map(|row| {
                if row.0 == operand {
                    count += 1;
                    (row.0, value, row.2)
                } else {
                    *row
                }
            })
            .collect::<ascent::boxcar::Vec<_>>();
        assert_eq!(count, 1);
        db.rel_set("op_immediate", rows);
    }

    fn replace_indirect(db: &mut DecompileDB, operand: Symbol, value: IndirectOperand) {
        type Row = (
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        );
        let mut count = 0;
        let rows = db
            .rel_iter::<Row>("op_indirect")
            .map(|row| {
                if row.0 == operand {
                    count += 1;
                    (row.0, value.0, value.1, value.2, value.3, value.4, value.5)
                } else {
                    *row
                }
            })
            .collect::<ascent::boxcar::Vec<_>>();
        assert_eq!(count, 1);
        db.rel_set("op_indirect", rows);
    }

    fn remove_register_effect(
        db: &mut DecompileDB,
        relation: &'static str,
        node: Node,
        register: Mreg,
    ) {
        let rows = db
            .rel_iter::<(Node, Mreg)>(relation)
            .filter(|row| **row != (node, register))
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        db.rel_set(relation, rows);
    }

    fn remove_memory_effect(
        db: &mut DecompileDB,
        relation: &'static str,
        node: Node,
        operand: Symbol,
    ) {
        let rows = db
            .rel_iter::<(Node, Symbol)>(relation)
            .filter(|row| **row != (node, operand))
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        db.rel_set(relation, rows);
    }

    fn rejects_kernel_service(
        label: &str,
        mutation: impl Fn(&mut DecompileDB, &mut CoffAddressMap),
    ) {
        let (mut db, mut map) = kernel_service_fixture();
        mutation(&mut db, &mut map);
        assert!(
            recognize_machine_state_stubs(&db, &map).is_empty(),
            "{label}"
        );
    }

    fn rejects_dword_out_getter(
        label: &str,
        mutation: impl Fn(&mut DecompileDB, &mut CoffAddressMap),
    ) {
        let (mut db, mut map) = dword_out_getter_fixture(0x24, 8);
        mutation(&mut db, &mut map);
        assert!(
            recognize_machine_state_stubs(&db, &map).is_empty(),
            "{label}"
        );
    }

    fn push_coff_section_header(
        bytes: &mut Vec<u8>,
        name: &[u8],
        size: u32,
        raw_offset: u32,
        characteristics: u32,
    ) {
        let mut padded_name = [0u8; 8];
        padded_name[..name.len()].copy_from_slice(name);
        bytes.extend_from_slice(&padded_name);
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes.extend_from_slice(&raw_offset.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&characteristics.to_le_bytes());
    }

    fn push_coff_function_symbol(bytes: &mut Vec<u8>, name: &[u8], value: u32) {
        let mut padded_name = [0u8; 8];
        padded_name[..name.len()].copy_from_slice(name);
        bytes.extend_from_slice(&padded_name);
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes.extend_from_slice(&1i16.to_le_bytes());
        let function_type =
            object::pe::IMAGE_SYM_DTYPE_FUNCTION << object::pe::IMAGE_SYM_DTYPE_SHIFT;
        bytes.extend_from_slice(&function_type.to_le_bytes());
        bytes.push(object::pe::IMAGE_SYM_CLASS_EXTERNAL);
        bytes.push(0);
    }

    fn adjacent_syscall_coff_fixture() -> Vec<u8> {
        let mut text = Vec::new();
        for service_number in [0x4000u32, 0x4001] {
            text.extend_from_slice(&[0x4c, 0x8b, 0xd1]);
            text.push(0xb8);
            text.extend_from_slice(&service_number.to_le_bytes());
            text.extend_from_slice(&[
                0xf6, 0x04, 0x25, 0x08, 0x03, 0xfe, 0x7f, 0x01, 0x75, 0x03, 0x0f, 0x05, 0xc3, 0xcd,
                0x2e, 0xc3, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]);
        }
        assert_eq!(text.len(), 64);

        const SYMBOL_COUNT: u32 = 2;
        let raw_offset = 20 + 40;
        let symbol_offset = raw_offset + text.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&object::pe::IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&symbol_offset.to_le_bytes());
        bytes.extend_from_slice(&SYMBOL_COUNT.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        push_coff_section_header(
            &mut bytes,
            b".text",
            text.len() as u32,
            raw_offset,
            0x6000_0020,
        );
        bytes.extend_from_slice(&text);
        push_coff_function_symbol(&mut bytes, b"stub_a", 0);
        push_coff_function_symbol(&mut bytes, b"stub_b", 32);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes
    }

    fn adjacent_dword_out_getter_coff_fixture() -> Vec<u8> {
        const DISP8: &[u8] = &[
            0x4c, 0x8b, 0xdc, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0xc2, 0x33, 0xc0, 0x49, 0x89, 0x53,
            0x10, 0x49, 0x89, 0x4b, 0x08, 0x8b, 0x49, 0x24, 0x89, 0x0a, 0xc3,
        ];
        // Same decoded nine-instruction semantics and length as DISP8, but XOR
        // uses the alternate 31 /r opcode form.  It is an adjacent-owner
        // control, not reconstruction material.
        const ALTERNATE_XOR_CONTROL: &[u8] = &[
            0x4c, 0x8b, 0xdc, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0xc2, 0x31, 0xc0, 0x49, 0x89, 0x53,
            0x10, 0x49, 0x89, 0x4b, 0x08, 0x8b, 0x49, 0x24, 0x89, 0x0a, 0xc3,
        ];
        const REDUNDANT_REX_CONTROL: &[u8] = &[
            0x4c, 0x8b, 0xdc, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0xc2, 0x40, 0x33, 0xc0, 0x49, 0x89,
            0x53, 0x10, 0x49, 0x89, 0x4b, 0x08, 0x8b, 0x49, 0x24, 0x89, 0x0a, 0xc3,
        ];
        const DISP32: &[u8] = &[
            0x4c, 0x8b, 0xdc, 0x48, 0x8b, 0xc1, 0x48, 0x8b, 0xc2, 0x33, 0xc0, 0x49, 0x89, 0x53,
            0x10, 0x49, 0x89, 0x4b, 0x08, 0x8b, 0x89, 0xcc, 0xed, 0xff, 0xff, 0x89, 0x0a, 0xc3,
        ];
        let mut text = Vec::new();
        text.extend_from_slice(DISP8);
        text.extend_from_slice(ALTERNATE_XOR_CONTROL);
        text.extend_from_slice(REDUNDANT_REX_CONTROL);
        text.extend_from_slice(DISP32);
        assert_eq!(text.len(), 104);

        const SYMBOL_COUNT: u32 = 4;
        let raw_offset = 20 + 40;
        let symbol_offset = raw_offset + text.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&object::pe::IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&symbol_offset.to_le_bytes());
        bytes.extend_from_slice(&SYMBOL_COUNT.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        push_coff_section_header(
            &mut bytes,
            b".text$og",
            text.len() as u32,
            raw_offset,
            0x6000_0020,
        );
        bytes.extend_from_slice(&text);
        push_coff_function_symbol(&mut bytes, b"disp8", 0);
        push_coff_function_symbol(&mut bytes, b"alt_xor", DISP8.len() as u32);
        push_coff_function_symbol(
            &mut bytes,
            b"rex_ctrl",
            (DISP8.len() + ALTERNATE_XOR_CONTROL.len()) as u32,
        );
        push_coff_function_symbol(
            &mut bytes,
            b"disp32",
            (DISP8.len() + ALTERNATE_XOR_CONTROL.len() + REDUNDANT_REX_CONTROL.len()) as u32,
        );
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes
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
    fn recognizes_arbitrary_named_win64_kernel_service_contract() {
        let (db, map) = kernel_service_fixture();
        let result = recognize_machine_state_stubs(&db, &map);
        assert_eq!(result.len(), 1);
        let MachineStateStubRecord::Win64KernelService(stub) = &result[0] else {
            panic!("expected kernel-service machine-state record")
        };
        assert_eq!(stub.schema, WIN64_KERNEL_SERVICE_STUB_SCHEMA_ID);
        assert_eq!(stub.function.name, "coff_fn_arbitrary_provider");
        assert_eq!((stub.function.size, stub.function.section_offset), (64, 0));
        assert_eq!(
            stub.abi
                .home_stores
                .iter()
                .map(|home| { (home.source_register, home.stack_offset, home.width_bits) })
                .collect::<Vec<_>>(),
            [
                ("rcx", 8, 64),
                ("rdx", 16, 64),
                ("r8", 24, 64),
                ("r9", 32, 64)
            ]
        );
        assert_eq!(
            (
                stub.entry_stack_pointer.source_register,
                stub.entry_stack_pointer.destination_register,
                stub.interrupt_control.kind,
                stub.stack_adjustment.amount_bytes
            ),
            ("rsp", "rax", "disable_maskable_interrupts", 16)
        );
        assert_eq!(
            stub.pushes
                .iter()
                .map(|push| {
                    (
                        push.kind,
                        push.source_register,
                        push.immediate_value,
                        push.immediate_width_bits,
                        push.stack_width_bits,
                    )
                })
                .collect::<Vec<_>>(),
            [
                ("register_value", Some("rax"), None, None, 64),
                ("flags", None, None, None, 64),
                ("immediate", None, Some(16), Some(8), 64),
                ("register_value", Some("rax"), None, None, 64),
            ]
        );
        assert_eq!(stub.linkage.target_original_name, "arbitrary_link");
        assert_eq!(stub.linkage.target_kind, "function");
        assert_eq!(stub.linkage.relocation.section_offset, 0x23);
        assert_eq!(stub.service_number.value, 0x89ab_cdef);
        assert_eq!(stub.dispatcher.target_original_name, "arbitrary_dispatch");
        assert_eq!(stub.dispatcher.relocation.section_offset, 0x2e);
        assert_eq!(stub.return_kind, "near_return");
        assert_eq!((stub.padding.boundary, stub.padding.size), (64, 13));
    }

    #[test]
    fn win64_kernel_service_mutations_fail_closed() {
        const E: Address = 0x1000_0000;
        rejects_kernel_service("instruction width", |db, _| {
            rewrite_instruction(db, E, |mut row| {
                row.1 = 4;
                row
            });
        });
        rejects_kernel_service("operand width", |db, _| {
            replace_indirect(db, KM0, ("NONE", "RSP", "NONE", 1, 8, 4));
        });
        rejects_kernel_service("register width", |db, _| replace_register(db, KH0, "ECX"));
        rejects_kernel_service("wrong register", |db, _| replace_register(db, KRSP, "RBP"));
        rejects_kernel_service("instruction order", |db, _| {
            rewrite_instruction(db, E + 0x1c, |mut row| {
                row.3 = "PUSHFQ";
                row.4 = NONE;
                row
            });
            rewrite_instruction(db, E + 0x1d, |mut row| {
                row.3 = "PUSH";
                row.4 = KRAX;
                row
            });
        });
        rejects_kernel_service("home offset", |db, _| {
            replace_indirect(db, KM1, ("NONE", "RSP", "NONE", 1, 24, 8));
        });
        rejects_kernel_service("linkage base", |db, map| {
            let disp = i64::from(map.relocations[0].encoded_value as u32 as i32);
            replace_indirect(db, KLEA, ("NONE", "RAX", "NONE", 1, disp, 8));
        });
        rejects_kernel_service("stack immediate", |db, _| replace_immediate(db, KSUB, 8));
        rejects_kernel_service("selector immediate", |db, _| replace_immediate(db, KSEL, 8));
        rejects_kernel_service("non-imm32 service", |db, _| {
            replace_immediate(db, KSVC, 0x1_0000_0000)
        });
        rejects_kernel_service("missing relocation", |_, map| {
            map.relocations.remove(0);
        });
        rejects_kernel_service("ambiguous relocation", |_, map| {
            map.relocations.push(map.relocations[0].clone());
        });
        rejects_kernel_service("extra relocation", |_, map| {
            let mut relocation = map.relocations[0].clone();
            relocation.section_offset = 0x32;
            relocation.mapped_field_va = E + 0x32;
            map.relocations.push(relocation);
        });
        rejects_kernel_service("relocation type", |_, map| {
            map.relocations[0].relocation_type = "IMAGE_REL_AMD64_REL32_1".into();
        });
        rejects_kernel_service("relocation width", |_, map| {
            map.relocations[0].width_bits = 64
        });
        rejects_kernel_service("relocation field", |_, map| {
            map.relocations[0].mapped_field_va += 1
        });
        rejects_kernel_service("relocation offset", |_, map| {
            map.relocations[0].section_offset += 1
        });
        rejects_kernel_service("relocation addend", |_, map| {
            map.relocations[0].encoded_value += 1
        });
        rejects_kernel_service("missing external", |_, map| {
            map.externs.remove(0);
        });
        rejects_kernel_service("ambiguous external", |_, map| {
            map.externs.push(map.externs[0].clone());
        });
        rejects_kernel_service("external kind", |_, map| {
            map.externs[0].kind = CoffExternalKind::Data;
        });
        rejects_kernel_service("missing symbol", |_, map| {
            map.symbols.remove(1);
        });
        rejects_kernel_service("ambiguous symbol", |_, map| {
            map.symbols.push(map.symbols[1].clone());
        });
        rejects_kernel_service("symbol kind", |_, map| map.symbols[1].kind = "data".into());
        rejects_kernel_service("defined symbol", |_, map| map.symbols[1].defined = true);
        rejects_kernel_service("extra instruction", |db, _| {
            let extra: Instruction = (E + 0x34, 1, "", "NOP", NONE, NONE, NONE, NONE, 0, 0);
            db.rel_push("unrefinedinstruction", extra);
        });
        rejects_kernel_service("extra CFG edge", |db, _| {
            db.rel_push("ddisasm_cfg_edge", (E + 0x18, E + 0x28, "branch"));
        });
        rejects_kernel_service("extra direct edge", |db, _| {
            db.rel_push("direct_call", (E + 0x17, E + 0x1000));
        });
        rejects_kernel_service("extra owner", |db, _| {
            db.rel_push("instr_in_function", (E + 0x17, E + 0x1000));
        });
        rejects_kernel_service("extra effect", |db, _| {
            db.rel_push("decoded_reg_def", (E + 0x17, Mreg::R11));
        });
        rejects_kernel_service("padding size", |db, _| {
            rewrite_instruction(db, E + 0x33, |mut row| {
                row.1 = 12;
                row
            });
        });
        rejects_kernel_service("padding shape", |db, _| {
            replace_indirect(db, KPAD, ("NONE", "RAX", "RAX", 1, 0, 4));
        });
        rejects_kernel_service("original size", |_, map| {
            map.functions[0].original_size = 63
        });
        rejects_kernel_service("mapped size", |_, map| map.functions[0].manifold_size = 63);
        rejects_kernel_service("alignment", |_, map| map.functions[0].section_offset = 1);
    }

    #[test]
    fn recognizes_arbitrary_named_win64_dword_out_getters_with_signed_displacements() {
        for (displacement, encoding_bits, size) in [
            (0x24, 8, 25),
            (-0x24, 8, 25),
            (0x1234, 32, 28),
            (-0x1234, 32, 28),
        ] {
            let (db, map) = dword_out_getter_fixture(displacement, encoding_bits);
            let result = recognize_machine_state_stubs(&db, &map);
            assert_eq!(result.len(), 1, "displacement={displacement}");
            let MachineStateStubRecord::Win64DwordOutGetter(stub) = &result[0] else {
                panic!("expected dword out-getter machine-state record")
            };
            assert_eq!(stub.schema, WIN64_DWORD_OUT_GETTER_STUB_SCHEMA_ID);
            assert_eq!(stub.function.name, "coff_fn_arbitrary_out_getter");
            assert_eq!(
                (stub.function.size, stub.function.section_offset),
                (size, 0x40)
            );
            assert_eq!(stub.entry_stack_pointer.source_register, "rsp");
            assert_eq!(stub.entry_stack_pointer.destination_register, "r11");
            assert_eq!(
                stub.register_copies
                    .iter()
                    .map(|copy| (copy.source_register, copy.destination_register))
                    .collect::<Vec<_>>(),
                [("rcx", "rax"), ("rdx", "rax")]
            );
            assert_eq!(
                (stub.result_zero.kind, stub.result_zero.register),
                ("xor_self", "eax")
            );
            assert_eq!(stub.home_base_register, "r11");
            assert_eq!(
                stub.abi
                    .home_stores
                    .iter()
                    .map(|home| (home.source_register, home.stack_offset, home.width_bits))
                    .collect::<Vec<_>>(),
                [("rdx", 16, 64), ("rcx", 8, 64)]
            );
            assert_eq!(stub.field_copy.source_displacement, displacement);
            assert_eq!(
                stub.field_copy.source_displacement_encoding_bits,
                encoding_bits
            );
            assert_eq!(stub.field_copy.width_bits, 32);
            assert_eq!(stub.field_copy.destination_displacement, 0);
            assert_eq!(stub.return_kind, "near_return");
            assert_eq!((stub.padding.kind, stub.padding.size), ("none", 0));
            assert!(stub.relocations.is_empty());
            assert_eq!(
                serde_json::to_value(stub).unwrap()["field_copy"]["source_displacement"].as_i64(),
                Some(displacement)
            );
        }
    }

    #[test]
    fn win64_dword_out_getter_instruction_and_operand_mutations_fail_closed() {
        const E: Address = 0x1000_0000;
        let addresses = [
            E,
            E + 3,
            E + 6,
            E + 9,
            E + 11,
            E + 15,
            E + 19,
            E + 22,
            E + 24,
        ];
        for address in addresses {
            rejects_dword_out_getter("instruction width", move |db, _| {
                rewrite_instruction(db, address, |mut row| {
                    row.1 += 1;
                    row
                });
            });
            rejects_dword_out_getter("instruction mnemonic", move |db, _| {
                rewrite_instruction(db, address, |mut row| {
                    row.3 = "NOP";
                    row
                });
            });
        }
        rejects_dword_out_getter("instruction prefix", |db, _| {
            rewrite_instruction(db, E, |mut row| {
                row.2 = "LOCK";
                row
            });
        });
        rejects_dword_out_getter("extra operand", |db, _| {
            rewrite_instruction(db, E + 3, |mut row| {
                row.6 = ORDX;
                row
            });
        });
        rejects_dword_out_getter("nonzero instruction metadata", |db, _| {
            rewrite_instruction(db, E + 6, |mut row| {
                row.8 = 1;
                row
            });
        });
        rejects_dword_out_getter("instruction gap", |db, _| {
            rewrite_instruction(db, E + 9, |mut row| {
                row.0 += 1;
                row
            });
        });
        rejects_dword_out_getter("missing instruction", |db, _| {
            let rows = db
                .rel_iter::<Instruction>("unrefinedinstruction")
                .filter(|row| row.0 != E + 9)
                .copied()
                .collect::<ascent::boxcar::Vec<_>>();
            db.rel_set("unrefinedinstruction", rows);
        });
        rejects_dword_out_getter("extra instruction", |db, _| {
            let extra: Instruction = (E + 1, 1, "", "NOP", NONE, NONE, NONE, NONE, 0, 0);
            db.rel_push("unrefinedinstruction", extra);
        });
        rejects_dword_out_getter("home-store order", |db, _| {
            rewrite_instruction(db, E + 11, |mut row| {
                row.4 = ORCX;
                row.5 = OHOME8;
                row
            });
            rewrite_instruction(db, E + 15, |mut row| {
                row.4 = ORDX;
                row.5 = OHOME16;
                row
            });
        });

        for (operand, replacement) in [
            (ORSP, "RBP"),
            (OR11, "R10"),
            (ORCX, "ECX"),
            (ORDX, "EDX"),
            (ORAX, "R10"),
            (OEAX, "RAX"),
            (OECX, "CX"),
        ] {
            rejects_dword_out_getter("register/width", move |db, _| {
                replace_register(db, operand, replacement)
            });
        }
        for (operand, replacement) in [
            (OHOME16, ("NONE", "R11", "NONE", 1, 8, 8)),
            (OHOME8, ("NONE", "R11", "NONE", 1, 16, 8)),
            (OHOME16, ("NONE", "R10", "NONE", 1, 16, 8)),
            (OHOME16, ("FS", "R11", "NONE", 1, 16, 8)),
            (OHOME16, ("NONE", "R11", "RCX", 1, 16, 8)),
            (OHOME16, ("NONE", "R11", "NONE", 2, 16, 8)),
            (OHOME16, ("NONE", "R11", "NONE", 1, 16, 4)),
            (OFIELD, ("NONE", "RCX", "NONE", 1, 0x24, 8)),
            (OFIELD, ("NONE", "RCX", "NONE", 1, 0x24, 1)),
            (OFIELD, ("NONE", "RDX", "NONE", 1, 0x24, 4)),
            (OFIELD, ("NONE", "RCX", "RAX", 1, 0x24, 4)),
            (OOUT, ("NONE", "RDX", "NONE", 1, 4, 4)),
            (OOUT, ("NONE", "RDX", "NONE", 1, 0, 8)),
            (OOUT, ("NONE", "RDX", "NONE", 1, 0, 1)),
        ] {
            rejects_dword_out_getter("memory shape/width", move |db, _| {
                replace_indirect(db, operand, replacement)
            });
        }
        rejects_dword_out_getter("ambiguous register operand", |db, _| {
            db.rel_push("op_register", (ORSP, "RBP"));
        });
        rejects_dword_out_getter("register/immediate operand ambiguity", |db, _| {
            db.rel_push("op_immediate", (OEAX, 0_i64, 0_usize));
        });
        rejects_dword_out_getter("zero displacement cannot be disp8", |db, _| {
            replace_indirect(db, OFIELD, ("NONE", "RCX", "NONE", 1, 0, 4));
        });
        rejects_dword_out_getter("disp8 overflow", |db, _| {
            replace_indirect(db, OFIELD, ("NONE", "RCX", "NONE", 1, 128, 4));
        });
        rejects_dword_out_getter("missing original encoding evidence", |db, _| {
            db.loaded_binary_data = None;
        });
        rejects_dword_out_getter("alternate xor opcode form", |db, _| {
            rewrite_dword_out_getter_input_byte(db, 9, 0x31);
        });
        rejects_dword_out_getter("alternate mov opcode direction", |db, _| {
            rewrite_dword_out_getter_input_byte(db, 4, 0x89);
            rewrite_dword_out_getter_input_byte(db, 5, 0xc8);
        });

        for displacement in [0_i64, 0x7f, -0x80] {
            let (db, map) = dword_out_getter_fixture(displacement, 32);
            assert!(
                recognize_machine_state_stubs(&db, &map).is_empty(),
                "noncanonical disp32 displacement={displacement}"
            );
        }
        for displacement in [0x80_i64, -0x81, i64::from(i32::MAX) + 1] {
            let (db, map) = dword_out_getter_fixture(displacement, 8);
            assert!(
                recognize_machine_state_stubs(&db, &map).is_empty(),
                "invalid disp8 displacement={displacement}"
            );
        }
        for displacement in [i64::from(i32::MAX) + 1, i64::from(i32::MIN) - 1] {
            let (db, map) = dword_out_getter_fixture(displacement, 32);
            assert!(
                recognize_machine_state_stubs(&db, &map).is_empty(),
                "disp32 overflow displacement={displacement}"
            );
        }
    }

    #[test]
    fn win64_dword_out_getter_effect_cfg_and_coff_mutations_fail_closed() {
        const E: Address = 0x1000_0000;
        rejects_dword_out_getter("missing register definition", |db, _| {
            remove_register_effect(db, "decoded_reg_def", E + 9, Mreg::AX)
        });
        rejects_dword_out_getter("missing register use", |db, _| {
            remove_register_effect(db, "decoded_reg_use", E + 19, Mreg::CX)
        });
        rejects_dword_out_getter("extra register definition", |db, _| {
            db.rel_push("decoded_reg_def", (E + 11, Mreg::AX));
        });
        rejects_dword_out_getter("extra register use", |db, _| {
            db.rel_push("decoded_reg_use", (E + 3, Mreg::DX));
        });
        rejects_dword_out_getter("missing memory read", |db, _| {
            remove_memory_effect(db, "decoded_memory_read_operand", E + 19, OFIELD)
        });
        rejects_dword_out_getter("missing memory write", |db, _| {
            remove_memory_effect(db, "decoded_memory_write_operand", E + 22, OOUT)
        });
        rejects_dword_out_getter("extra memory effect", |db, _| {
            db.rel_push("decoded_memory_write_operand", (E + 19, OFIELD));
        });
        rejects_dword_out_getter("missing owner", |db, _| {
            let rows = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter(|row| row.0 != E + 9)
                .copied()
                .collect::<ascent::boxcar::Vec<_>>();
            db.rel_set("instr_in_function", rows);
        });
        rejects_dword_out_getter("ambiguous owner", |db, _| {
            db.rel_push("instr_in_function", (E + 9, E + 0x1000));
        });
        rejects_dword_out_getter("missing next edge", |db, _| {
            let rows = db
                .rel_iter::<(Node, Node)>("next")
                .filter(|row| **row != (E + 9, E + 11))
                .copied()
                .collect::<ascent::boxcar::Vec<_>>();
            db.rel_set("next", rows);
        });
        rejects_dword_out_getter("extra next edge", |db, _| {
            db.rel_push("next", (E + 9, E + 19));
        });
        rejects_dword_out_getter("ret fallthrough", |db, _| {
            db.rel_push("next", (E + 24, E + 25));
        });
        rejects_dword_out_getter("direct jump", |db, _| {
            db.rel_push("direct_jump", (E + 9, E + 19));
        });
        rejects_dword_out_getter("direct call", |db, _| {
            db.rel_push("direct_call", (E + 9, E + 0x1000));
        });
        rejects_dword_out_getter("CFG edge", |db, _| {
            db.rel_push("ddisasm_cfg_edge", (E + 9, E + 19, "branch"));
        });
        rejects_dword_out_getter("flags consumer", |db, _| {
            db.rel_push("flags_and_jump_pair", (E + 9, E + 19, "e"));
        });

        rejects_dword_out_getter("map schema", |_, map| map.schema = "wrong");
        rejects_dword_out_getter("loader", |_, map| map.loader_id = "wrong");
        rejects_dword_out_getter("architecture", |_, map| map.architecture = "wrong");
        rejects_dword_out_getter("original size", |_, map| {
            map.functions[0].original_size = 24
        });
        rejects_dword_out_getter("manifold size", |_, map| {
            map.functions[0].manifold_size = 24
        });
        rejects_dword_out_getter("mapped end", |_, map| map.functions[0].mapped_end -= 1);
        rejects_dword_out_getter("function section", |_, map| {
            map.functions[0].section_index = 6
        });
        rejects_dword_out_getter("function section offset", |_, map| {
            map.functions[0].section_offset += 1
        });
        rejects_dword_out_getter("ambiguous function", |_, map| {
            map.functions.push(map.functions[0].clone())
        });
        rejects_dword_out_getter("missing section", |_, map| map.sections.clear());
        rejects_dword_out_getter("ambiguous section", |_, map| {
            map.sections.push(map.sections[0].clone())
        });
        rejects_dword_out_getter("section kind", |_, map| {
            map.sections[0].kind = "Data".into()
        });
        rejects_dword_out_getter("mapped section lower bound", |_, map| {
            map.sections[0].mapped_va_start += 1
        });
        rejects_dword_out_getter("mapped section upper bound", |_, map| {
            map.sections[0].mapped_va_end -= 1
        });
        rejects_dword_out_getter("original section lower bound", |_, map| {
            map.sections[0].original_offset_start += 1
        });
        rejects_dword_out_getter("original section upper bound", |_, map| {
            map.sections[0].original_offset_end -= 1
        });
        rejects_dword_out_getter("unequal in-bounds section spans", |_, map| {
            map.sections[0].original_offset_start = 0x20;
            map.sections[0].original_offset_end = 0x80;
            map.sections[0].mapped_va_start = E - 0x20;
            map.sections[0].mapped_va_end = E + 0x41;
        });
        rejects_dword_out_getter("in-bounds section mapping misalignment", |_, map| {
            map.sections[0].original_offset_start = 0x20;
            map.sections[0].original_offset_end = 0x80;
            map.sections[0].mapped_va_start = E - 0x10;
            map.sections[0].mapped_va_end = E + 0x50;
        });
        rejects_dword_out_getter("in-bounds function mapping misalignment", |_, map| {
            map.sections[0].original_offset_start = 0x20;
            map.sections[0].original_offset_end = 0x80;
            map.sections[0].mapped_va_start = E - 0x20;
            map.sections[0].mapped_va_end = E + 0x40;
            map.functions[0].section_offset += 1;
            map.symbols[0].section_offset = Some(map.functions[0].section_offset);
        });
        rejects_dword_out_getter("missing function symbol", |_, map| map.symbols.clear());
        rejects_dword_out_getter("ambiguous function symbol", |_, map| {
            map.symbols.push(map.symbols[0].clone())
        });
        rejects_dword_out_getter("function symbol original name", |_, map| {
            map.symbols[0].original_name = "different".into()
        });
        rejects_dword_out_getter("function symbol provider name", |_, map| {
            map.symbols[0].provider_name = "different".into()
        });
        rejects_dword_out_getter("function symbol address", |_, map| {
            map.symbols[0].mapped_address += 1
        });
        rejects_dword_out_getter("function symbol kind", |_, map| {
            map.symbols[0].kind = "data".into()
        });
        rejects_dword_out_getter("undefined function symbol", |_, map| {
            map.symbols[0].defined = false
        });
        rejects_dword_out_getter("function symbol section", |_, map| {
            map.symbols[0].section_index = None
        });
        rejects_dword_out_getter("function symbol offset", |_, map| {
            map.symbols[0].section_offset = Some(0x41)
        });
        rejects_dword_out_getter("function relocation", |_, map| {
            map.relocations.push(CoffRelocationMap {
                section_index: 5,
                section_name: ".text$generic".into(),
                section_offset: 0x50,
                mapped_field_va: E + 0x10,
                relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
                width_bits: 32,
                target_original_name: "unrelated".into(),
                target_mapped_address: E + 0x1000,
                encoded_value: 0,
            });
        });
        rejects_dword_out_getter("wrong-section mapped relocation overlap", |_, map| {
            map.relocations.push(CoffRelocationMap {
                section_index: 6,
                section_name: ".other".into(),
                section_offset: 0,
                mapped_field_va: E + 0x10,
                relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
                width_bits: 32,
                target_original_name: "unrelated".into(),
                target_mapped_address: E + 0x1000,
                encoded_value: 0,
            });
        });
        rejects_dword_out_getter("same-index relocation section-name mismatch", |_, map| {
            map.relocations.push(CoffRelocationMap {
                section_index: 5,
                section_name: ".wrong".into(),
                section_offset: 0x100,
                mapped_field_va: E + 0x100,
                relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
                width_bits: 32,
                target_original_name: "unrelated".into(),
                target_mapped_address: E + 0x1000,
                encoded_value: 0,
            });
        });

        let (db, mut map) = dword_out_getter_fixture(0x24, 8);
        map.relocations.push(CoffRelocationMap {
            section_index: 6,
            section_name: ".other".into(),
            section_offset: 0,
            mapped_field_va: E + 0x100,
            relocation_type: "IMAGE_REL_AMD64_ADDR32".into(),
            width_bits: 32,
            target_original_name: "unrelated".into(),
            target_mapped_address: E + 0x1000,
            encoded_value: 0,
        });
        assert_eq!(
            recognize_machine_state_stubs(&db, &map).len(),
            1,
            "a nonoverlapping relocation in another section contaminated the owner"
        );
    }

    #[test]
    fn win64_dword_out_getter_recognizes_real_decoder_asm_and_adjacent_owners() {
        std::thread::Builder::new()
            .name("machine-state-dword-out-getter-coff-fixture".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                static FIXTURE_ID: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let path = std::env::temp_dir().join(format!(
                    "manifold-machine-state-dword-out-getter-{}-{}.obj",
                    std::process::id(),
                    FIXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                ));
                std::fs::write(&path, adjacent_dword_out_getter_coff_fixture()).unwrap();

                let mut db = DecompileDB::default();
                let map = crate::decompile::disassembly::load_from_binary(&mut db, &path).unwrap();
                let _ = std::fs::remove_file(&path);
                assert!(
                    db.loaded_binary_data.is_some(),
                    "real COFF loader did not retain its exact decoder input"
                );
                AsmPass.run(&mut db);

                assert_eq!(map.functions.len(), 4);
                assert!(map.relocations.is_empty());
                let section = &map.sections[0];
                assert_eq!(
                    section
                        .original_offset_end
                        .checked_sub(section.original_offset_start),
                    section.mapped_va_end.checked_sub(section.mapped_va_start),
                );
                for function in &map.functions {
                    assert_eq!(
                        function
                            .section_offset
                            .checked_sub(section.original_offset_start),
                        function.mapped_entry.checked_sub(section.mapped_va_start),
                    );
                    let instruction_addresses: Vec<_> = db
                        .rel_iter::<Instruction>("unrefinedinstruction")
                        .filter_map(|row| {
                            (function.mapped_entry <= row.0 && row.0 < function.mapped_end)
                                .then_some(row.0)
                        })
                        .collect();
                    assert_eq!(instruction_addresses.len(), 9, "{function:?}");
                    for address in instruction_addresses {
                        assert_eq!(
                            db.rel_iter::<(Node, Address)>("instr_in_function")
                                .filter_map(|(node, owner)| {
                                    (*node == address).then_some(*owner)
                                })
                                .collect::<BTreeSet<_>>(),
                            BTreeSet::from([function.mapped_entry]),
                            "adjacent function ownership leaked at 0x{address:x}",
                        );
                    }
                }

                let result = recognize_machine_state_stubs(&db, &map);
                assert_eq!(result.len(), 2, "unexpected records: {result:#?}");
                let recognized: BTreeMap<_, _> = result
                    .iter()
                    .map(|record| {
                        let MachineStateStubRecord::Win64DwordOutGetter(stub) = record else {
                            panic!("expected dword out-getter record, got {record:?}")
                        };
                        (
                            stub.function.address.clone(),
                            (
                                stub.field_copy.source_displacement,
                                stub.field_copy.source_displacement_encoding_bits,
                            ),
                        )
                    })
                    .collect();
                let function_by_original: BTreeMap<_, _> = map
                    .functions
                    .iter()
                    .map(|function| (function.original_name.as_str(), function))
                    .collect();
                assert_eq!(
                    recognized.get(&format!(
                        "0x{:x}",
                        function_by_original["disp8"].mapped_entry
                    )),
                    Some(&(0x24, 8)),
                );
                assert_eq!(
                    recognized.get(&format!(
                        "0x{:x}",
                        function_by_original["disp32"].mapped_entry
                    )),
                    Some(&(-0x1234, 32)),
                );
                for control in ["alt_xor", "rex_ctrl"] {
                    assert!(
                        !recognized.contains_key(&format!(
                            "0x{:x}",
                            function_by_original[control].mapped_entry
                        )),
                        "same-semantics encoding control {control} was accepted",
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn recognizes_real_decoder_facts_for_adjacent_coff_syscalls() {
        std::thread::Builder::new()
            .name("machine-state-syscall-coff-fixture".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                static FIXTURE_ID: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let path = std::env::temp_dir().join(format!(
                    "manifold-machine-state-syscall-{}-{}.obj",
                    std::process::id(),
                    FIXTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                ));
                std::fs::write(&path, adjacent_syscall_coff_fixture()).unwrap();

                let mut db = DecompileDB::default();
                let map = crate::decompile::disassembly::load_from_binary(&mut db, &path).unwrap();
                let _ = std::fs::remove_file(&path);
                AsmPass.run(&mut db);

                assert_eq!(map.functions.len(), 2);
                for function in &map.functions {
                    let test = function.mapped_entry + 8;
                    let reads: BTreeSet<_> = db
                        .rel_iter::<(Node, Symbol)>("decoded_memory_read_operand")
                        .filter_map(|(node, operand)| (*node == test).then_some(*operand))
                        .collect();
                    let writes: BTreeSet<_> = db
                        .rel_iter::<(Node, Symbol)>("decoded_memory_write_operand")
                        .filter_map(|(node, operand)| (*node == test).then_some(*operand))
                        .collect();
                    assert_eq!(reads.len(), 1);
                    assert_eq!(writes, reads);
                }

                let first_padding = map.functions[0].mapped_entry + 24;
                assert_eq!(
                    db.rel_iter::<(Node, Address, Symbol)>("ddisasm_cfg_edge")
                        .filter(|(source, _, _)| *source == first_padding)
                        .copied()
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from([
                        (first_padding, map.functions[1].mapped_entry, "fallthrough",)
                    ]),
                );

                let result = recognize_machine_state_stubs(&db, &map);
                assert_eq!(result.len(), 2);
                let service_numbers: BTreeSet<_> = result
                    .iter()
                    .map(|record| match record {
                        MachineStateStubRecord::Win64Syscall(stub) => stub.service_number.value,
                        other => panic!("expected Win64 syscall record, got {other:?}"),
                    })
                    .collect();
                assert_eq!(service_numbers, BTreeSet::from([0x4000, 0x4001]));
            })
            .unwrap()
            .join()
            .unwrap();
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
            vec![(0x1000_1008_u64, 0x1000_1010_u64, "e")]
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
        db.rel_push("decoded_reg_def", (0x1000_1012_u64, Mreg::R11));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = syscall_fixture(0);
        db.rel_push(
            "decoded_memory_write_operand",
            (0x1000_1008_u64, "unexpected_test_write"),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = syscall_fixture(0);
        db.rel_push(
            "ddisasm_cfg_edge",
            (0x1000_1018_u64, 0x1000_1020_u64, "fallthrough"),
        );
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
        db.rel_push("decoded_reg_def", (0x1000_0000_u64, Mreg::R11));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, mut map) = fixture();
        map.functions[0].mapped_end += 1;
        map.functions[0].original_size += 1;
        map.functions[0].manifold_size += 1;
        let extra: Instruction = (0x1000_000b, 1, "", "NOP", NONE, NONE, NONE, NONE, 0, 0);
        db.rel_push("unrefinedinstruction", extra);
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
    }

    #[test]
    fn rejects_conditional_indirect_and_multiple_exit_shapes() {
        let (mut db, map) = fixture();
        let jump: Address = 0x1000_0006;
        let conditional_rows: Vec<Instruction> = vec![
            (0x1000_0000, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
            (jump, 5, "", "JNE", TARGET, NONE, NONE, NONE, 0, 0),
        ];
        db.rel_set(
            "unrefinedinstruction",
            conditional_rows
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = fixture();
        let indirect_rows: Vec<Instruction> = vec![
            (0x1000_0000, 6, "", "MOV", IMM, DST, NONE, NONE, 0, 0),
            (jump, 5, "", "JMP", INDIRECT, NONE, NONE, NONE, 0, 0),
        ];
        db.rel_set(
            "unrefinedinstruction",
            indirect_rows
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
            vec![(jump, 0_u64, "indirect")]
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
        db.rel_push("ddisasm_cfg_edge", (jump, 0x1000_3000_u64, "branch"));
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

    fn guarded_fixture(displacement: i64) -> (DecompileDB, CoffAddressMap) {
        const IMM72: Symbol = "guarded_test_imm72";
        const RSP: Symbol = "guarded_test_rsp";
        const RAX: Symbol = "guarded_test_rax";
        const RCX: Symbol = "guarded_test_rcx";
        const RDX: Symbol = "guarded_test_rdx";
        const R8: Symbol = "guarded_test_r8";
        const R9: Symbol = "guarded_test_r9";
        const R10: Symbol = "guarded_test_r10";
        const RSP32: Symbol = "guarded_test_mem_rsp32";
        const RSP40: Symbol = "guarded_test_mem_rsp40";
        const RSP48: Symbol = "guarded_test_mem_rsp48";
        const RSP56: Symbol = "guarded_test_mem_rsp56";
        const RCX0: Symbol = "guarded_test_mem_rcx0";
        const RCX32: Symbol = "guarded_test_mem_rcx32";
        const R10DISP: Symbol = "guarded_test_mem_r10_disp";
        const RIP0: Symbol = "guarded_test_mem_rip0";

        let mut db = DecompileDB::default();
        let entry: Address = 0x1000_0000;
        let target_load_size: usize = if (1..=i8::MAX as i64).contains(&displacement) {
            4
        } else {
            7
        };
        let sizes: [usize; 16] = [
            4,
            5,
            5,
            5,
            4,
            5,
            3,
            target_load_size,
            6,
            3,
            5,
            5,
            5,
            5,
            4,
            3,
        ];
        let mut addresses: Vec<Address> = Vec::new();
        let mut cursor = entry;
        for size in sizes {
            addresses.push(cursor);
            cursor += size as u64;
        }
        let guard_target: Address = 0x1000_3000;
        let guard_displacement =
            i32::try_from(guard_target - (addresses[8] + 6)).expect("guard target fits rel32");
        let rows: Vec<Instruction> = vec![
            (addresses[0], 4, "", "SUB", IMM72, RSP, NONE, NONE, 0, 0),
            (addresses[1], 5, "", "MOV", RDX, RSP40, NONE, NONE, 0, 0),
            (addresses[2], 5, "", "MOV", R8, RSP48, NONE, NONE, 0, 0),
            (addresses[3], 5, "", "MOV", R9, RSP56, NONE, NONE, 0, 0),
            (addresses[4], 4, "", "MOV", RCX32, RCX, NONE, NONE, 0, 0),
            (addresses[5], 5, "", "MOV", RCX, RSP32, NONE, NONE, 0, 0),
            (addresses[6], 3, "", "MOV", RCX0, R10, NONE, NONE, 0, 0),
            (
                addresses[7],
                target_load_size,
                "",
                "MOV",
                R10DISP,
                RCX,
                NONE,
                NONE,
                0,
                0,
            ),
            (addresses[8], 6, "", "CALL", RIP0, NONE, NONE, NONE, 0, 0),
            (addresses[9], 3, "", "MOV", RCX, RAX, NONE, NONE, 0, 0),
            (addresses[10], 5, "", "MOV", RSP40, RDX, NONE, NONE, 0, 0),
            (addresses[11], 5, "", "MOV", RSP48, R8, NONE, NONE, 0, 0),
            (addresses[12], 5, "", "MOV", RSP56, R9, NONE, NONE, 0, 0),
            (addresses[13], 5, "", "MOV", RSP32, RCX, NONE, NONE, 0, 0),
            (addresses[14], 4, "", "ADD", IMM72, RSP, NONE, NONE, 0, 0),
            (addresses[15], 3, "", "JMP", RAX, NONE, NONE, NONE, 0, 0),
        ];
        db.rel_set(
            "unrefinedinstruction",
            rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_register",
            vec![
                (RSP, "RSP"),
                (RAX, "RAX"),
                (RCX, "RCX"),
                (RDX, "RDX"),
                (R8, "R8"),
                (R9, "R9"),
                (R10, "R10"),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_immediate",
            vec![(IMM72, 72_i64, 0_usize)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "op_indirect",
            Vec::<(
                Symbol,
                &'static str,
                &'static str,
                &'static str,
                i64,
                i64,
                usize,
            )>::from([
                (RSP32, "NONE", "RSP", "NONE", 1, 32, 8),
                (RSP40, "NONE", "RSP", "NONE", 1, 40, 8),
                (RSP48, "NONE", "RSP", "NONE", 1, 48, 8),
                (RSP56, "NONE", "RSP", "NONE", 1, 56, 8),
                (RCX0, "NONE", "RCX", "NONE", 1, 0, 8),
                (RCX32, "NONE", "RCX", "NONE", 1, 32, 8),
                (R10DISP, "NONE", "R10", "NONE", 1, displacement, 8),
                (
                    RIP0,
                    "NONE",
                    "RIP",
                    "NONE",
                    1,
                    i64::from(guard_displacement),
                    8,
                ),
            ])
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_memory_read_operand",
            vec![
                (addresses[4], RCX32),
                (addresses[6], RCX0),
                (addresses[7], R10DISP),
                (addresses[8], RIP0),
                (addresses[10], RSP40),
                (addresses[11], RSP48),
                (addresses[12], RSP56),
                (addresses[13], RSP32),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_memory_write_operand",
            vec![
                (addresses[1], RSP40),
                (addresses[2], RSP48),
                (addresses[3], RSP56),
                (addresses[5], RSP32),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "instr_in_function",
            addresses
                .iter()
                .map(|address| (*address, entry))
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "next",
            addresses
                .windows(2)
                .map(|pair| (pair[0], pair[1]))
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "direct_jump",
            Vec::<(Node, Address)>::new()
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
                (addresses[8], 0_u64, "indirect_call"),
                (addresses[8], addresses[9], "fallthrough"),
                (addresses[15], 0_u64, "indirect"),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_def",
            vec![
                (addresses[0], Mreg::SP),
                (addresses[4], Mreg::CX),
                (addresses[6], Mreg::R10),
                (addresses[7], Mreg::CX),
                (addresses[8], Mreg::SP),
                (addresses[9], Mreg::AX),
                (addresses[10], Mreg::DX),
                (addresses[11], Mreg::R8),
                (addresses[12], Mreg::R9),
                (addresses[13], Mreg::CX),
                (addresses[14], Mreg::SP),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "decoded_reg_use",
            vec![
                (addresses[0], Mreg::SP),
                (addresses[1], Mreg::SP),
                (addresses[1], Mreg::DX),
                (addresses[2], Mreg::SP),
                (addresses[2], Mreg::R8),
                (addresses[3], Mreg::SP),
                (addresses[3], Mreg::R9),
                (addresses[4], Mreg::CX),
                (addresses[5], Mreg::SP),
                (addresses[5], Mreg::CX),
                (addresses[6], Mreg::CX),
                (addresses[7], Mreg::R10),
                (addresses[8], Mreg::SP),
                (addresses[9], Mreg::CX),
                (addresses[10], Mreg::SP),
                (addresses[11], Mreg::SP),
                (addresses[12], Mreg::SP),
                (addresses[13], Mreg::SP),
                (addresses[14], Mreg::SP),
                (addresses[15], Mreg::AX),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );

        let size = cursor - entry;
        let guard_field = addresses[8] + 2;
        let map = CoffAddressMap {
            schema: "manifold.coff-address-map.v1",
            loader_id: "amd64-coff-image-v1",
            architecture: "x86_64-pc-windows-msvc",
            image_base: entry,
            function_boundary_sidecar_sha256: None,
            sections: vec![CoffSectionMap {
                index: 4,
                name: ".text$guarded".into(),
                kind: "Text".into(),
                original_file_offset: Some(0x200),
                original_offset_start: 0,
                original_offset_end: size,
                mapped_va_start: entry,
                mapped_va_end: cursor,
            }],
            functions: vec![CoffFunctionMap {
                original_name: "unrelated_original_spelling".into(),
                provider_name: "coff_fn_unrelated_provider_spelling".into(),
                section_index: 4,
                section_offset: 0,
                original_size: size,
                manifold_size: size,
                mapped_entry: entry,
                mapped_end: cursor,
            }],
            symbols: vec![CoffSymbolMap {
                original_name: "unrelated_original_spelling".into(),
                provider_name: "coff_fn_unrelated_provider_spelling".into(),
                kind: "function".into(),
                defined: true,
                section_index: Some(4),
                section_offset: Some(0),
                mapped_address: entry,
            }],
            externs: vec![CoffExternalMap {
                original_name: "unrelated_guard_pointer".into(),
                provider_name: "coff_ext_unrelated_guard_pointer".into(),
                kind: CoffExternalKind::ImportPointer,
                synthetic_address: guard_target,
            }],
            relocations: vec![CoffRelocationMap {
                section_index: 4,
                section_name: ".text$guarded".into(),
                section_offset: guard_field - entry,
                mapped_field_va: guard_field,
                relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                width_bits: 32,
                target_original_name: "unrelated_guard_pointer".into(),
                target_mapped_address: guard_target,
                encoded_value: u64::from(guard_displacement as u32),
            }],
        };
        (db, map)
    }

    fn guarded_stub(result: &[MachineStateStubRecord]) -> &GuardedIndirectForwardTailStub {
        match &result[0] {
            MachineStateStubRecord::GuardedIndirectForwardTail(stub) => stub,
            other => panic!("expected guarded forward tail, got {other:?}"),
        }
    }

    #[test]
    fn recognizes_arbitrary_named_guarded_forward_tail_encodings() {
        let (db, map) = guarded_fixture(0x218);
        let result = recognize_machine_state_stubs(&db, &map);
        assert_eq!(result.len(), 1);
        let stub = guarded_stub(&result);
        assert_eq!(stub.forwarding.dispatch_load.displacement, 0x218);
        assert_eq!(stub.guard.target_original_name, "unrelated_guard_pointer");
        assert_eq!(stub.transfer.target_register, "rax");

        let (db, map) = guarded_fixture(24);
        let result = recognize_machine_state_stubs(&db, &map);
        assert_eq!(
            guarded_stub(&result).guard.instruction_address,
            "0x10000023"
        );
        assert_eq!(map.functions[0].manifold_size, 71);
    }

    #[test]
    fn recognizes_loader_applied_guard_rel32() {
        let (db, map) = guarded_fixture(0x218);
        assert_ne!(map.relocations[0].encoded_value, 0);

        let result = recognize_machine_state_stubs(&db, &map);
        let stub = guarded_stub(&result);
        assert_eq!(
            stub.guard.target_address,
            format!("0x{:x}", map.relocations[0].target_mapped_address)
        );
    }

    #[test]
    fn guarded_forward_tail_fails_closed_on_effect_cfg_and_relocation_ambiguity() {
        let (mut db, map) = guarded_fixture(0x218);
        db.rel_push("decoded_reg_use", (0x1000_001f_u64, Mreg::R11));
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (mut db, map) = guarded_fixture(0x218);
        db.rel_push(
            "ddisasm_cfg_edge",
            (0x1000_0047_u64, 0x1000_4000_u64, "branch"),
        );
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = guarded_fixture(0x218);
        map.relocations.push(map.relocations[0].clone());
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = guarded_fixture(0x218);
        map.externs[0].kind = CoffExternalKind::Data;
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = guarded_fixture(0x218);
        map.relocations[0].encoded_value += 1;
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, mut map) = guarded_fixture(0x218);
        map.relocations[0].target_mapped_address += 1;
        map.externs[0].synthetic_address += 1;
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());

        let (db, map) = guarded_fixture(25);
        assert!(recognize_machine_state_stubs(&db, &map).is_empty());
    }
}
