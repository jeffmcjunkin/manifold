// RTLOptimizePass: imperative finalization of RTL candidates (disambiguation, copy-prop, DSE, nop collapse, inline-temp discovery), split out so the RTL pass proper stays pure Ascent.

use crate::decompile::disassembly::operand::NO_OP;
use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::mreg::Mreg;
use crate::x86::op::{Addressing, Comparison, Condition, Operation};
use crate::x86::types::*;
use either::Either;
use log::info;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

type Cr8RawInstruction = (
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

fn exact_raw_operand_count(row: &Cr8RawInstruction) -> Option<usize> {
    let operands = [row.3, row.4, row.5, row.6];
    let mut count = 0;
    let mut reached_padding = false;
    for operand in operands {
        if operand == NO_OP {
            reached_padding = true;
        } else if reached_padding {
            return None;
        } else {
            count += 1;
        }
    }
    Some(count)
}

#[cfg(test)]
mod scalar_lvalue_memory_access_tests {
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
    const MEMORY: Symbol = "scalar_test_memory";
    const REGISTER: Symbol = "scalar_test_register";
    const EXTRA: Symbol = "scalar_test_extra";

    #[derive(Clone, Copy, Debug)]
    enum Mutation {
        None,
        Prefix,
        WrongOrder,
        WrongWidth,
        WrongRegisterWidth,
        AmbiguousRegister,
        CrossKindOperands,
        WrongLtlValue,
        AmbiguousLtl,
        WrongDisplacement,
        WrongAddressSize,
        AmbiguousAddressSize,
        Segment,
        SymbolicBase,
        MissingEffect,
        ExtraEffect,
        MissingRegisterDef,
        ExtraRegisterDef,
        MissingRegisterUse,
        ExtraRegisterUse,
        RawMismatch,
        AmbiguousInstruction,
        ExtraOperand,
        AmbiguousOperand,
        MissingCandidate,
        CandidateDrift,
        AmbiguousFinal,
        AmbiguousOwner,
        OverlappingFunction,
        BadMapSpan,
        NonlinearSection,
        WrongSection,
        RelocationOverlap,
        MappedRelocationWrongSection,
        RelocationOverflow,
        WrongRelocationSectionName,
        RedefinedRoot,
        WrongDownstreamWidth,
        AmbiguousDownstreamWidth,
    }

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

    fn scalar_fixture(
        mnemonic: &'static str,
        width: usize,
        register_name: &'static str,
        chunk: MemoryChunk,
        direction: ScalarMemoryDirection,
        mutation: Mutation,
    ) -> DecompileDB {
        let mut db = DecompileDB::default();
        db.coff_address_map = Some(coff_map());
        let displacement = 12;
        let first = if direction == ScalarMemoryDirection::Read {
            MEMORY
        } else {
            REGISTER
        };
        let second = if direction == ScalarMemoryDirection::Read {
            REGISTER
        } else {
            MEMORY
        };
        let (first, second) = if matches!(mutation, Mutation::WrongOrder) {
            (second, first)
        } else {
            (first, second)
        };
        let mut decoded = instruction(
            if matches!(mutation, Mutation::Prefix) {
                "LOCK"
            } else {
                ""
            },
            mnemonic,
            first,
            second,
        );
        if matches!(mutation, Mutation::ExtraOperand) {
            decoded.5 = EXTRA;
            db.rel_push("op_immediate", (EXTRA, 1_i64, 1_usize));
        }
        push_instruction(&mut db, "instruction", NODE, decoded.clone());
        push_instruction(
            &mut db,
            "unrefinedinstruction",
            NODE,
            if matches!(mutation, Mutation::RawMismatch) {
                instruction("", "MOVZX", first, second)
            } else {
                decoded.clone()
            },
        );
        if matches!(mutation, Mutation::AmbiguousInstruction) {
            push_instruction(
                &mut db,
                "instruction",
                NODE,
                instruction("", "MOVZX", first, second),
            );
        }
        db.rel_push(
            "op_register",
            (
                REGISTER,
                if matches!(mutation, Mutation::WrongRegisterWidth) {
                    "RAX"
                } else {
                    register_name
                },
            ),
        );
        if matches!(mutation, Mutation::AmbiguousRegister) {
            db.rel_push("op_register", (REGISTER, "EDX"));
        }
        db.rel_push(
            "op_indirect",
            (
                MEMORY,
                if matches!(mutation, Mutation::Segment) {
                    "FS"
                } else {
                    "NONE"
                },
                if matches!(mutation, Mutation::SymbolicBase) {
                    "RIP"
                } else {
                    "RCX"
                },
                "NONE",
                1_i64,
                if matches!(mutation, Mutation::WrongDisplacement) {
                    displacement + 1
                } else {
                    displacement
                },
                if matches!(mutation, Mutation::WrongWidth) {
                    width.saturating_add(1)
                } else {
                    width
                },
            ),
        );
        if matches!(mutation, Mutation::AmbiguousOperand) {
            db.rel_push(
                "op_indirect",
                (MEMORY, "NONE", "RCX", "RDX", 2_i64, displacement, width),
            );
        }
        if matches!(mutation, Mutation::CrossKindOperands) {
            db.rel_push("op_register", (MEMORY, "RCX"));
            db.rel_push(
                "op_indirect",
                (REGISTER, "NONE", "RCX", "NONE", 1_i64, 0_i64, width),
            );
        }
        db.rel_push(
            "instruction_address_size",
            (
                NODE,
                if matches!(mutation, Mutation::WrongAddressSize) {
                    2_u8
                } else {
                    8_u8
                },
            ),
        );
        if matches!(mutation, Mutation::AmbiguousAddressSize) {
            db.rel_push("instruction_address_size", (NODE, 4_u8));
        }
        if !matches!(mutation, Mutation::MissingEffect) {
            let relation = if direction == ScalarMemoryDirection::Read {
                "decoded_memory_read_operand"
            } else {
                "decoded_memory_write_operand"
            };
            db.rel_push(relation, (NODE, MEMORY));
        }
        if matches!(mutation, Mutation::ExtraEffect) {
            db.rel_push("decoded_memory_read_operand", (NODE, EXTRA));
        }
        let value_mreg = Mreg::x86(register_name);
        let mut decoded_uses = BTreeSet::from([Mreg::CX]);
        let mut decoded_defs = BTreeSet::new();
        match direction {
            ScalarMemoryDirection::Read => {
                decoded_defs.insert(value_mreg);
            }
            ScalarMemoryDirection::Write => {
                decoded_uses.insert(value_mreg);
            }
        }
        if matches!(mutation, Mutation::MissingRegisterDef) {
            decoded_defs.clear();
        }
        if matches!(mutation, Mutation::ExtraRegisterDef) {
            decoded_defs.insert(Mreg::DX);
        }
        if matches!(mutation, Mutation::MissingRegisterUse) {
            decoded_uses.clear();
        }
        if matches!(mutation, Mutation::ExtraRegisterUse) {
            decoded_uses.insert(Mreg::DX);
        }
        for register in decoded_defs {
            db.rel_push("decoded_reg_def", (NODE, register));
        }
        for register in decoded_uses {
            db.rel_push("decoded_reg_use", (NODE, register));
        }

        let address = Addressing::Aindexed(displacement);
        let mregs = Arc::new(vec![Mreg::CX]);
        let args = Arc::new(vec![BASE]);
        let ltl = if direction == ScalarMemoryDirection::Read {
            LTLInst::Lload(
                chunk,
                address.clone(),
                mregs,
                if matches!(mutation, Mutation::WrongLtlValue) {
                    Mreg::DX
                } else {
                    Mreg::x86(register_name)
                },
            )
        } else {
            LTLInst::Lstore(
                chunk,
                address.clone(),
                mregs,
                if matches!(mutation, Mutation::WrongLtlValue) {
                    Mreg::DX
                } else {
                    Mreg::x86(register_name)
                },
            )
        };
        db.rel_push("ltl_inst", (NODE, ltl.clone()));
        if matches!(mutation, Mutation::AmbiguousLtl) {
            let competing = match ltl {
                LTLInst::Lload(chunk, _, mregs, destination) => LTLInst::Lload(
                    chunk,
                    Addressing::Aindexed(displacement + 1),
                    mregs,
                    destination,
                ),
                LTLInst::Lstore(chunk, _, mregs, source) => {
                    LTLInst::Lstore(chunk, Addressing::Aindexed(displacement + 1), mregs, source)
                }
                _ => unreachable!(),
            };
            db.rel_push("ltl_inst", (NODE, competing));
        }
        let selected = if direction == ScalarMemoryDirection::Read {
            RTLInst::Iload(chunk, address, args, VALUE)
        } else {
            RTLInst::Istore(chunk, address, args, VALUE)
        };
        db.rel_push("rtl_inst", (NODE, selected.clone()));
        if matches!(mutation, Mutation::AmbiguousFinal) {
            let competing = match &selected {
                RTLInst::Iload(chunk, addressing, _, destination) => RTLInst::Iload(
                    *chunk,
                    addressing.clone(),
                    Arc::new(vec![INDEX]),
                    *destination,
                ),
                RTLInst::Istore(chunk, addressing, _, source) => {
                    RTLInst::Istore(*chunk, addressing.clone(), Arc::new(vec![INDEX]), *source)
                }
                _ => unreachable!(),
            };
            db.rel_push("rtl_inst", (NODE, competing));
        }
        if !matches!(mutation, Mutation::MissingCandidate) {
            let candidate = if matches!(mutation, Mutation::CandidateDrift) {
                match &selected {
                    RTLInst::Iload(chunk, addressing, _, destination) => RTLInst::Iload(
                        *chunk,
                        addressing.clone(),
                        Arc::new(vec![INDEX]),
                        *destination,
                    ),
                    RTLInst::Istore(chunk, addressing, _, source) => {
                        RTLInst::Istore(*chunk, addressing.clone(), Arc::new(vec![INDEX]), *source)
                    }
                    _ => unreachable!(),
                }
            } else {
                selected
            };
            db.rel_push("rtl_inst_candidate", (NODE, candidate));
        }
        db.rel_push("rtl_inst", (NEXT, RTLInst::Ireturn(VALUE)));
        db.rel_push("rtl_inst_candidate", (NEXT, RTLInst::Ireturn(VALUE)));
        db.rel_push("rtl_succ", (NODE, NEXT));
        db.rel_push("instr_in_function", (NODE, FUNCTION));
        db.rel_push("instr_in_function", (NEXT, FUNCTION));
        if matches!(mutation, Mutation::AmbiguousOwner) {
            db.rel_push("instr_in_function", (NODE, FUNCTION + 0x200));
        }
        db.rel_push("emit_function", (FUNCTION, "arbitrary_function", NODE));
        db.rel_push("emit_function_param_candidate", (FUNCTION, BASE));
        if direction == ScalarMemoryDirection::Read {
            let xtype = if matches!(mutation, Mutation::WrongDownstreamWidth) {
                XType::Xint16unsigned
            } else {
                match x86_scalar_register_width(register_name) {
                    Some(4) => XType::Xint,
                    Some(8) => XType::Xlong,
                    Some(2) => XType::Xint16unsigned,
                    Some(1) => XType::Xint8unsigned,
                    _ => XType::Xvoid,
                }
            };
            db.rel_push("emit_var_type_candidate", (VALUE, xtype));
            if matches!(mutation, Mutation::AmbiguousDownstreamWidth) {
                db.rel_push("emit_var_type_candidate", (VALUE, XType::Xlong));
            }
        }

        if matches!(mutation, Mutation::OverlappingFunction) {
            db.coff_address_map
                .as_mut()
                .expect("map")
                .functions
                .push(CoffFunctionMap {
                    original_name: "overlap".into(),
                    provider_name: "overlap_provider".into(),
                    section_index: 1,
                    section_offset: 8,
                    original_size: 0x20,
                    manifold_size: 0x20,
                    mapped_entry: FUNCTION + 8,
                    mapped_end: FUNCTION + 0x28,
                });
        }
        if matches!(mutation, Mutation::BadMapSpan) {
            db.coff_address_map.as_mut().expect("map").functions[0].manifold_size -= 1;
        }
        if matches!(mutation, Mutation::NonlinearSection) {
            db.coff_address_map.as_mut().expect("map").sections[0].mapped_va_end += 1;
        }
        if matches!(mutation, Mutation::WrongSection) {
            db.coff_address_map.as_mut().expect("map").functions[0].section_index = 2;
        }
        if matches!(mutation, Mutation::RelocationOverlap) {
            db.coff_address_map
                .as_mut()
                .expect("map")
                .relocations
                .push(CoffRelocationMap {
                    section_index: 1,
                    section_name: ".text".into(),
                    section_offset: 0x11,
                    mapped_field_va: NODE + 1,
                    relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                    width_bits: 32,
                    target_original_name: "external".into(),
                    target_mapped_address: 0x9000,
                    encoded_value: 0,
                });
        }
        if matches!(mutation, Mutation::MappedRelocationWrongSection) {
            db.coff_address_map
                .as_mut()
                .expect("map")
                .relocations
                .push(CoffRelocationMap {
                    section_index: 99,
                    section_name: ".other".into(),
                    section_offset: 0,
                    mapped_field_va: NODE + 1,
                    relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                    width_bits: 32,
                    target_original_name: "external".into(),
                    target_mapped_address: 0x9000,
                    encoded_value: 0,
                });
        }
        if matches!(mutation, Mutation::RelocationOverflow) {
            db.coff_address_map
                .as_mut()
                .expect("map")
                .relocations
                .push(CoffRelocationMap {
                    section_index: 99,
                    section_name: ".other".into(),
                    section_offset: u64::MAX,
                    mapped_field_va: u64::MAX,
                    relocation_type: "IMAGE_REL_AMD64_ADDR64".into(),
                    width_bits: 64,
                    target_original_name: "external".into(),
                    target_mapped_address: 0x9000,
                    encoded_value: 0,
                });
        }
        if matches!(mutation, Mutation::WrongRelocationSectionName) {
            db.coff_address_map
                .as_mut()
                .expect("map")
                .relocations
                .push(CoffRelocationMap {
                    section_index: 1,
                    section_name: ".wrong".into(),
                    section_offset: 0x30,
                    mapped_field_va: FUNCTION + 0x30,
                    relocation_type: "IMAGE_REL_AMD64_REL32".into(),
                    width_bits: 32,
                    target_original_name: "external".into(),
                    target_mapped_address: 0x9000,
                    encoded_value: 0,
                });
        }
        if matches!(mutation, Mutation::RedefinedRoot) {
            db.rel_push(
                "rtl_inst",
                (
                    FUNCTION + 4,
                    RTLInst::Iop(Operation::Omove, Arc::new(vec![INDEX]), BASE),
                ),
            );
            db.rel_push("instr_in_function", (FUNCTION + 4, FUNCTION));
            db.rel_push("rtl_succ", (FUNCTION + 4, NODE));
        }
        db
    }

    fn proofs(mut db: DecompileDB) -> Vec<ScalarMemoryAccessProof> {
        let _ = materialize_authenticated_scalar_memory_accesses(&mut db);
        db.rel_iter::<(Node, ScalarMemoryAccessProof)>("authenticated_scalar_memory_access")
            .map(|(_, proof)| proof.clone())
            .collect()
    }

    fn add_ltl_signedness_companion(
        db: &mut DecompileDB,
        width: usize,
        register_name: &'static str,
        chunk: MemoryChunk,
        direction: ScalarMemoryDirection,
    ) {
        let companion = scalar_signedness_companion(chunk, width)
            .expect("byte/word fixture has a signedness companion");
        let value = Mreg::x86(register_name);
        let addressing = Addressing::Aindexed(12);
        let mregs = Arc::new(vec![Mreg::CX]);
        let row = match direction {
            ScalarMemoryDirection::Read => LTLInst::Lload(companion, addressing, mregs, value),
            ScalarMemoryDirection::Write => LTLInst::Lstore(companion, addressing, mregs, value),
        };
        db.rel_push("ltl_inst", (NODE, row));
    }

    fn add_rtl_signedness_companion(
        db: &mut DecompileDB,
        width: usize,
        chunk: MemoryChunk,
        direction: ScalarMemoryDirection,
    ) {
        let companion = scalar_signedness_companion(chunk, width)
            .expect("byte/word fixture has a signedness companion");
        let args = Arc::new(vec![BASE]);
        let row = match direction {
            ScalarMemoryDirection::Read => {
                RTLInst::Iload(companion, Addressing::Aindexed(12), args, VALUE)
            }
            ScalarMemoryDirection::Write => {
                RTLInst::Istore(companion, Addressing::Aindexed(12), args, VALUE)
            }
        };
        db.rel_push("rtl_inst_candidate", (NODE, row));
    }

    fn scalar_fixture_with_signedness_companions(
        mnemonic: &'static str,
        width: usize,
        register_name: &'static str,
        chunk: MemoryChunk,
        direction: ScalarMemoryDirection,
    ) -> DecompileDB {
        let mut db = scalar_fixture(
            mnemonic,
            width,
            register_name,
            chunk,
            direction,
            Mutation::None,
        );
        add_ltl_signedness_companion(&mut db, width, register_name, chunk, direction);
        add_rtl_signedness_companion(&mut db, width, chunk, direction);
        db
    }

    #[test]
    fn scalar_memory_access_authenticates_all_closed_width_and_extension_forms() {
        let cases = [
            (
                "MOV",
                4,
                "EAX",
                MemoryChunk::MInt32,
                ScalarMemoryExtension::Plain,
                4,
                4,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOV",
                8,
                "RAX",
                MemoryChunk::MInt64,
                ScalarMemoryExtension::Plain,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOVSX",
                1,
                "EAX",
                MemoryChunk::MInt8Signed,
                ScalarMemoryExtension::SignExtend,
                4,
                8,
                ScalarMemoryResultChain::ZeroUpper32,
            ),
            (
                "MOVSX",
                1,
                "RAX",
                MemoryChunk::MInt8Signed,
                ScalarMemoryExtension::SignExtend,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOVSX",
                2,
                "EAX",
                MemoryChunk::MInt16Signed,
                ScalarMemoryExtension::SignExtend,
                4,
                8,
                ScalarMemoryResultChain::ZeroUpper32,
            ),
            (
                "MOVSX",
                2,
                "RAX",
                MemoryChunk::MInt16Signed,
                ScalarMemoryExtension::SignExtend,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOVSXD",
                4,
                "RAX",
                MemoryChunk::MInt32,
                ScalarMemoryExtension::SignExtend,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOVZX",
                1,
                "EAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryExtension::ZeroExtend,
                4,
                8,
                ScalarMemoryResultChain::ZeroUpper32,
            ),
            (
                "MOVZX",
                1,
                "RAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryExtension::ZeroExtend,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
            (
                "MOVZX",
                2,
                "EAX",
                MemoryChunk::MInt16Unsigned,
                ScalarMemoryExtension::ZeroExtend,
                4,
                8,
                ScalarMemoryResultChain::ZeroUpper32,
            ),
            (
                "MOVZX",
                2,
                "RAX",
                MemoryChunk::MInt16Unsigned,
                ScalarMemoryExtension::ZeroExtend,
                8,
                8,
                ScalarMemoryResultChain::Direct,
            ),
        ];
        for (
            mnemonic,
            width,
            register,
            chunk,
            extension,
            encoded_width,
            result_width,
            result_chain,
        ) in cases
        {
            let rows = proofs(scalar_fixture(
                mnemonic,
                width,
                register,
                chunk,
                ScalarMemoryDirection::Read,
                Mutation::None,
            ));
            assert_eq!(rows.len(), 1, "{mnemonic}/{width}/{register}");
            assert_eq!(rows[0].width, width);
            assert_eq!(rows[0].extension, extension);
            assert_eq!(rows[0].encoded_destination_width, Some(encoded_width));
            assert_eq!(rows[0].value_width, result_width);
            assert_eq!(rows[0].downstream_value_width, Some(result_width));
            assert_eq!(rows[0].result_chain, Some(result_chain));
        }

        // Pre-Type describes the byte/word source chunk for real EAX
        // extensions.  That lowering artifact must not erase the independent
        // architectural fact that every r32 write clears the parent register.
        for (mnemonic, width, chunk, source_type) in [
            ("MOVSX", 1, MemoryChunk::MInt8Signed, XType::Xint8signed),
            (
                "MOVSX",
                2,
                MemoryChunk::MInt16Signed,
                XType::Xint16signed,
            ),
            (
                "MOVZX",
                1,
                MemoryChunk::MInt8Unsigned,
                XType::Xint8unsigned,
            ),
            (
                "MOVZX",
                2,
                MemoryChunk::MInt16Unsigned,
                XType::Xint16unsigned,
            ),
        ] {
            let mut db = scalar_fixture(
                mnemonic,
                width,
                "EAX",
                chunk,
                ScalarMemoryDirection::Read,
                Mutation::None,
            );
            db.rel_set(
                "emit_var_type_candidate",
                vec![(VALUE, source_type)]
                    .into_iter()
                    .collect::<ascent::boxcar::Vec<_>>(),
            );
            let rows = proofs(db);
            assert_eq!(rows.len(), 1, "source-typed {mnemonic}/{width}/EAX");
            assert_eq!(rows[0].encoded_destination_width, Some(4));
            assert_eq!(rows[0].value_width, 8);
            assert_eq!(rows[0].downstream_value_width, Some(8));
            assert_eq!(
                rows[0].result_chain,
                Some(ScalarMemoryResultChain::ZeroUpper32)
            );
        }

        for (mnemonic, width, chunk, extension) in [
            (
                "MOV",
                4,
                MemoryChunk::MInt32,
                ScalarMemoryExtension::ImplicitZeroExtend,
            ),
            (
                "MOVSX",
                1,
                MemoryChunk::MInt8Signed,
                ScalarMemoryExtension::SignExtend,
            ),
            (
                "MOVZX",
                1,
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryExtension::ZeroExtend,
            ),
        ] {
            let mut db = scalar_fixture(
                mnemonic,
                width,
                "EAX",
                chunk,
                ScalarMemoryDirection::Read,
                Mutation::None,
            );
            db.rel_set(
                "emit_var_type_candidate",
                vec![(VALUE, XType::Xlong)]
                    .into_iter()
                    .collect::<ascent::boxcar::Vec<_>>(),
            );
            let rows = proofs(db);
            assert_eq!(rows.len(), 1, "wide {mnemonic}/{width}/EAX");
            assert_eq!(rows[0].extension, extension);
            assert_eq!(rows[0].encoded_destination_width, Some(4));
            assert_eq!(rows[0].value_width, 8);
            assert_eq!(rows[0].downstream_value_width, Some(8));
            assert_eq!(
                rows[0].result_chain,
                Some(ScalarMemoryResultChain::ZeroUpper32)
            );
        }
        for (width, register, chunk) in [
            (1, "AL", MemoryChunk::MInt8Unsigned),
            (2, "AX", MemoryChunk::MInt16Unsigned),
            (4, "EAX", MemoryChunk::MInt32),
            (8, "RAX", MemoryChunk::MInt64),
        ] {
            let rows = proofs(scalar_fixture(
                "MOV",
                width,
                register,
                chunk,
                ScalarMemoryDirection::Write,
                Mutation::None,
            ));
            assert_eq!(rows.len(), 1, "store/{width}/{register}");
            assert_eq!(rows[0].downstream_value_width, None);
        }
        for (width, register, chunk) in [
            (1, "AL", MemoryChunk::MInt8Unsigned),
            (2, "AX", MemoryChunk::MInt16Unsigned),
        ] {
            assert!(proofs(scalar_fixture(
                "MOV",
                width,
                register,
                chunk,
                ScalarMemoryDirection::Read,
                Mutation::None,
            ))
            .is_empty());
        }
        assert!(proofs(scalar_fixture(
            "MOVSX",
            1,
            "AL",
            MemoryChunk::MInt8Signed,
            ScalarMemoryDirection::Read,
            Mutation::None,
        ))
        .is_empty());
        for mnemonic in ["MOVSX", "MOVZX"] {
            assert!(proofs(scalar_fixture(
                mnemonic,
                1,
                "AX",
                if mnemonic == "MOVSX" {
                    MemoryChunk::MInt8Signed
                } else {
                    MemoryChunk::MInt8Unsigned
                },
                ScalarMemoryDirection::Read,
                Mutation::None,
            ))
            .is_empty());
        }
        assert!(proofs(scalar_fixture(
            "MOV",
            1,
            "AH",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Write,
            Mutation::None,
        ))
        .is_empty());
        assert!(proofs(scalar_fixture(
            "MOVZX",
            1,
            "EAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Write,
            Mutation::None,
        ))
        .is_empty());
    }

    #[test]
    fn any64_transport_alias_is_closed_to_plain_gp_qword_movs() {
        for direction in [ScalarMemoryDirection::Read, ScalarMemoryDirection::Write] {
            let rows = proofs(scalar_fixture(
                "MOV",
                8,
                "RAX",
                MemoryChunk::MAny64,
                direction,
                Mutation::None,
            ));
            assert_eq!(rows.len(), 1, "qword transport {direction:?}");
            assert_eq!(rows[0].chunk, MemoryChunk::MAny64);
            assert!(rows[0].is_closed_v1());
        }

        for (mnemonic, width, register, chunk) in [
            ("MOVZX", 1, "RAX", MemoryChunk::MAny64),
            ("MOV", 4, "EAX", MemoryChunk::MAny64),
            ("MOV", 8, "RAX", MemoryChunk::MFloat64),
        ] {
            assert!(
                proofs(scalar_fixture(
                    mnemonic,
                    width,
                    register,
                    chunk,
                    ScalarMemoryDirection::Read,
                    Mutation::None,
                ))
                .is_empty(),
                "transport alias escaped: {mnemonic}/{width}/{register}/{chunk:?}"
            );
        }
    }

    #[test]
    fn plain_store_fixed_point_aliases_are_exact_and_stage4_private() {
        const ALIAS_ONE: RTLReg = 0x8000_0000_0000_0070;
        const ALIAS_TWO: RTLReg = 0x8000_0000_0000_0080;
        const EXTRA_ALIAS: RTLReg = 0x8000_0000_0000_0090;

        fn fixture() -> DecompileDB {
            let mut db = scalar_fixture(
                "MOV",
                8,
                "RAX",
                MemoryChunk::MAny64,
                ScalarMemoryDirection::Write,
                Mutation::None,
            );
            for source in [ALIAS_ONE, ALIAS_TWO] {
                db.rel_push(
                    "rtl_inst_candidate",
                    (
                        NODE,
                        RTLInst::Istore(
                            MemoryChunk::MAny64,
                            Addressing::Aindexed(12),
                            Arc::new(vec![BASE]),
                            source,
                        ),
                    ),
                );
            }
            for source in [VALUE, ALIAS_ONE, ALIAS_TWO] {
                db.rel_push("reg_xtl", (NODE, Mreg::AX, source));
                db.rel_push("xtl_canonical", (source, source));
                if source != VALUE {
                    db.rel_push("xtl_canonical", (source, VALUE));
                }
            }
            db.rel_push("reaching_use_rtl", (NODE, Mreg::AX, VALUE));
            db
        }

        fn remove_candidate(db: &mut DecompileDB, removed_source: RTLReg) {
            let rows = db
                .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, row)| {
                    *node != NODE
                        || !matches!(row, RTLInst::Istore(_, _, _, source) if *source == removed_source)
                })
                .cloned()
                .collect::<ascent::boxcar::Vec<_>>();
            db.rel_set("rtl_inst_candidate", rows);
        }

        fn private_proofs(mut db: DecompileDB) -> Vec<(Node, ScalarMemoryAccessProof)> {
            let private = materialize_authenticated_scalar_memory_accesses(&mut db);
            assert!(
                db.rel_iter::<(Node, ScalarMemoryAccessProof)>("authenticated_scalar_memory_access")
                    .next()
                    .is_none(),
                "fixed-point store aliases must stay Stage-4 private",
            );
            private
        }

        let rows = private_proofs(fixture());
        let [(node, proof)] = rows.as_slice() else {
            panic!("expected one private fixed-point store: {rows:#?}");
        };
        assert_eq!(*node, NODE);
        assert_eq!(proof.direction, ScalarMemoryDirection::Write);
        assert_eq!(proof.value, VALUE);

        let mut missing = fixture();
        remove_candidate(&mut missing, ALIAS_TWO);
        assert!(private_proofs(missing).is_empty());

        let mut extra = fixture();
        extra.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Istore(
                    MemoryChunk::MAny64,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    EXTRA_ALIAS,
                ),
            ),
        );
        extra.rel_push("xtl_canonical", (EXTRA_ALIAS, VALUE));
        assert!(private_proofs(extra).is_empty());

        let mut duplicate = fixture();
        duplicate.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Istore(
                    MemoryChunk::MAny64,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    ALIAS_ONE,
                ),
            ),
        );
        assert!(private_proofs(duplicate).is_empty());

        let mut wrong_canonical = fixture();
        let canonical_rows = wrong_canonical
            .rel_iter::<(RTLReg, RTLReg)>("xtl_canonical")
            .filter(|(source, canonical)| !(*source == ALIAS_TWO && *canonical == VALUE))
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        wrong_canonical.rel_set("xtl_canonical", canonical_rows);
        assert!(private_proofs(wrong_canonical).is_empty());

        let mut ambiguous_reaching = fixture();
        ambiguous_reaching.rel_push("reaching_use_rtl", (NODE, Mreg::AX, ALIAS_ONE));
        assert!(private_proofs(ambiguous_reaching).is_empty());

        let mut wrong_reaching = fixture();
        wrong_reaching.rel_set(
            "reaching_use_rtl",
            vec![(NODE, Mreg::AX, ALIAS_ONE)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(private_proofs(wrong_reaching).is_empty());

        let mut wrong_chunk = fixture();
        remove_candidate(&mut wrong_chunk, ALIAS_TWO);
        wrong_chunk.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Istore(
                    MemoryChunk::MInt64,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    ALIAS_TWO,
                ),
            ),
        );
        assert!(private_proofs(wrong_chunk).is_empty());

        let mut wrong_address = fixture();
        remove_candidate(&mut wrong_address, ALIAS_TWO);
        wrong_address.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Istore(
                    MemoryChunk::MAny64,
                    Addressing::Aindexed(16),
                    Arc::new(vec![BASE]),
                    ALIAS_TWO,
                ),
            ),
        );
        assert!(private_proofs(wrong_address).is_empty());

        let mut missing_identity = fixture();
        let xtl_rows = missing_identity
            .rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl")
            .filter(|(_, _, value)| *value != ALIAS_TWO)
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        missing_identity.rel_set("reg_xtl", xtl_rows);
        assert!(private_proofs(missing_identity).is_empty());
    }

    #[test]
    fn plain_read_fixed_point_base_alias_is_exact_and_stage4_private() {
        const HISTORICAL_BASE: RTLReg = 0x8000_0000_0000_0070;
        const EXTRA_BASE: RTLReg = 0x8000_0000_0000_0080;

        fn fixture() -> DecompileDB {
            let mut db = scalar_fixture(
                "MOV",
                4,
                "EAX",
                MemoryChunk::MInt32,
                ScalarMemoryDirection::Read,
                Mutation::None,
            );
            db.rel_push(
                "rtl_inst_candidate",
                (
                    NODE,
                    RTLInst::Iload(
                        MemoryChunk::MInt32,
                        Addressing::Aindexed(12),
                        Arc::new(vec![HISTORICAL_BASE]),
                        VALUE,
                    ),
                ),
            );
            for base in [BASE, HISTORICAL_BASE] {
                db.rel_push("reg_xtl", (NODE, Mreg::CX, base));
                db.rel_push("xtl_canonical", (base, base));
            }
            db.rel_push("xtl_canonical", (HISTORICAL_BASE, BASE));
            db.rel_push("reaching_use_rtl", (NODE, Mreg::CX, BASE));
            db
        }

        fn run(
            mut db: DecompileDB,
        ) -> (
            Vec<ScalarMemoryAccessProof>,
            Vec<(Node, ScalarMemoryAccessProof)>,
        ) {
            let private = materialize_authenticated_scalar_memory_accesses(&mut db);
            let public = db
                .rel_iter::<(Node, ScalarMemoryAccessProof)>(
                    "authenticated_scalar_memory_access",
                )
                .map(|(_, proof)| proof.clone())
                .collect();
            (public, private)
        }

        fn remove_historical_candidate(db: &mut DecompileDB) {
            let rows = db
                .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, row)| {
                    *node != NODE
                        || !matches!(
                            row,
                            RTLInst::Iload(_, _, args, _)
                                if args.as_slice() == [HISTORICAL_BASE]
                        )
                })
                .cloned()
                .collect::<ascent::boxcar::Vec<_>>();
            db.rel_set("rtl_inst_candidate", rows);
        }

        let (public, private) = run(fixture());
        assert!(public.is_empty(), "fixed-point alias must stay Stage-4 private");
        let [(node, proof)] = private.as_slice() else {
            panic!("expected one private fixed-point placement: {private:#?}");
        };
        assert_eq!(*node, NODE);
        assert_eq!(proof.base_value, Some(BASE));
        assert_eq!(proof.value, VALUE);
        assert_eq!(proof.direction, ScalarMemoryDirection::Read);

        let mut missing_candidate = fixture();
        remove_historical_candidate(&mut missing_candidate);
        let (public, private) = run(missing_candidate);
        assert!(private.is_empty());
        assert_eq!(public.len(), 1, "the remaining exact singleton is ordinary");

        let mut extra_candidate = fixture();
        extra_candidate.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(12),
                    Arc::new(vec![EXTRA_BASE]),
                    VALUE,
                ),
            ),
        );
        extra_candidate.rel_push("xtl_canonical", (EXTRA_BASE, BASE));
        assert!(run(extra_candidate).1.is_empty());

        let mut duplicate_candidate = fixture();
        duplicate_candidate.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(12),
                    Arc::new(vec![HISTORICAL_BASE]),
                    VALUE,
                ),
            ),
        );
        assert!(run(duplicate_candidate).1.is_empty());

        let mut wrong_canonical = fixture();
        let canonical_rows = wrong_canonical
            .rel_iter::<(RTLReg, RTLReg)>("xtl_canonical")
            .filter(|(source, canonical)| {
                !(*source == HISTORICAL_BASE && *canonical == BASE)
            })
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        wrong_canonical.rel_set("xtl_canonical", canonical_rows);
        assert!(run(wrong_canonical).1.is_empty());

        let mut ambiguous_reaching = fixture();
        ambiguous_reaching.rel_push(
            "reaching_use_rtl",
            (NODE, Mreg::CX, HISTORICAL_BASE),
        );
        assert!(run(ambiguous_reaching).1.is_empty());

        let mut wrong_reaching = fixture();
        wrong_reaching.rel_set(
            "reaching_use_rtl",
            vec![(NODE, Mreg::CX, HISTORICAL_BASE)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        assert!(run(wrong_reaching).1.is_empty());

        let mut wrong_chunk = fixture();
        remove_historical_candidate(&mut wrong_chunk);
        wrong_chunk.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MAny32,
                    Addressing::Aindexed(12),
                    Arc::new(vec![HISTORICAL_BASE]),
                    VALUE,
                ),
            ),
        );
        assert!(run(wrong_chunk).1.is_empty());

        let mut wrong_address = fixture();
        remove_historical_candidate(&mut wrong_address);
        wrong_address.rel_push(
            "rtl_inst_candidate",
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(16),
                    Arc::new(vec![HISTORICAL_BASE]),
                    VALUE,
                ),
            ),
        );
        assert!(run(wrong_address).1.is_empty());

        let mut missing_identity = fixture();
        let xtl_rows = missing_identity
            .rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl")
            .filter(|(_, _, value)| *value != HISTORICAL_BASE)
            .copied()
            .collect::<ascent::boxcar::Vec<_>>();
        missing_identity.rel_set("reg_xtl", xtl_rows);
        assert!(run(missing_identity).1.is_empty());
    }

    #[test]
    fn signedness_companion_is_closed_to_one_exact_byte_or_word_pipeline_pair() {
        for (mnemonic, width, register, chunk, direction) in [
            (
                "MOVZX",
                1,
                "EAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryDirection::Read,
            ),
            (
                "MOVZX",
                2,
                "EAX",
                MemoryChunk::MInt16Unsigned,
                ScalarMemoryDirection::Read,
            ),
            (
                "MOVSX",
                1,
                "EAX",
                MemoryChunk::MInt8Signed,
                ScalarMemoryDirection::Read,
            ),
            (
                "MOVSX",
                2,
                "EAX",
                MemoryChunk::MInt16Signed,
                ScalarMemoryDirection::Read,
            ),
            (
                "MOV",
                1,
                "AL",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryDirection::Write,
            ),
            (
                "MOV",
                2,
                "AX",
                MemoryChunk::MInt16Unsigned,
                ScalarMemoryDirection::Write,
            ),
        ] {
            let rows = proofs(scalar_fixture_with_signedness_companions(
                mnemonic, width, register, chunk, direction,
            ));
            assert_eq!(rows.len(), 1, "{mnemonic}/{width}/{register}/{direction:?}");
            assert_eq!(rows[0].chunk, chunk);
            assert_eq!(rows[0].direction, direction);
        }
    }

    #[test]
    fn extension_raw_semantics_accept_only_sealed_selected_transport_pairs() {
        for (mnemonic, width, register, selected, extension) in [
            (
                "MOVSX",
                1,
                "RAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryExtension::SignExtend,
            ),
            (
                "MOVSX",
                2,
                "RAX",
                MemoryChunk::MInt16Unsigned,
                ScalarMemoryExtension::SignExtend,
            ),
            (
                "MOVZX",
                1,
                "EAX",
                MemoryChunk::MInt8Signed,
                ScalarMemoryExtension::ZeroExtend,
            ),
            (
                "MOVZX",
                2,
                "EAX",
                MemoryChunk::MInt16Signed,
                ScalarMemoryExtension::ZeroExtend,
            ),
        ] {
            let rows = proofs(scalar_fixture_with_signedness_companions(
                mnemonic,
                width,
                register,
                selected,
                ScalarMemoryDirection::Read,
            ));
            assert_eq!(rows.len(), 1, "{mnemonic}/{width}/{register}/{selected:?}");
            assert_eq!(rows[0].extension, extension);
            assert_eq!(rows[0].chunk, selected);
            assert!(rows[0].is_closed_v1());
        }

        let singleton_opposite = scalar_fixture(
            "MOVSX",
            1,
            "RAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        assert!(
            proofs(singleton_opposite).is_empty(),
            "an opposite selected transport requires its exact twin in both retained layers"
        );

        let mut ltl_pair_only = scalar_fixture(
            "MOVSX",
            1,
            "RAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        add_ltl_signedness_companion(
            &mut ltl_pair_only,
            1,
            "RAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Read,
        );
        assert!(
            proofs(ltl_pair_only).is_empty(),
            "an LTL-only signedness pair authenticated"
        );

        let mut rtl_pair_only = scalar_fixture(
            "MOVZX",
            1,
            "EAX",
            MemoryChunk::MInt8Signed,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        add_rtl_signedness_companion(
            &mut rtl_pair_only,
            1,
            MemoryChunk::MInt8Signed,
            ScalarMemoryDirection::Read,
        );
        assert!(
            proofs(rtl_pair_only).is_empty(),
            "an RTL-only signedness pair authenticated"
        );

        for (mnemonic, width, register, selected) in [
            ("MOVSX", 1, "RAX", MemoryChunk::MInt16Unsigned),
            ("MOVZX", 2, "EAX", MemoryChunk::MInt8Signed),
        ] {
            assert!(
                proofs(scalar_fixture(
                    mnemonic,
                    width,
                    register,
                    selected,
                    ScalarMemoryDirection::Read,
                    Mutation::None,
                ))
                .is_empty(),
                "outside-pair transport authenticated: {mnemonic}/{width}/{selected:?}"
            );
        }
    }

    #[test]
    fn signedness_companion_rejects_every_other_ltl_interpretation() {
        let exact = || {
            LTLInst::Lload(
                MemoryChunk::MInt8Unsigned,
                Addressing::Aindexed(12),
                Arc::new(vec![Mreg::CX]),
                Mreg::AX,
            )
        };
        let rejected = [
            ("duplicate-exact", exact()),
            (
                "wrong-address",
                LTLInst::Lload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(13),
                    Arc::new(vec![Mreg::CX]),
                    Mreg::AX,
                ),
            ),
            (
                "wrong-args",
                LTLInst::Lload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![Mreg::DX]),
                    Mreg::AX,
                ),
            ),
            (
                "wrong-value",
                LTLInst::Lload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![Mreg::CX]),
                    Mreg::DX,
                ),
            ),
            (
                "cross-width",
                LTLInst::Lload(
                    MemoryChunk::MInt16Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![Mreg::CX]),
                    Mreg::AX,
                ),
            ),
            (
                "wrong-kind",
                LTLInst::Lstore(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![Mreg::CX]),
                    Mreg::AX,
                ),
            ),
        ];
        for (label, competing) in rejected {
            let mut db = scalar_fixture(
                "MOVZX",
                1,
                "EAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryDirection::Read,
                Mutation::None,
            );
            db.rel_push("ltl_inst", (NODE, competing));
            assert!(proofs(db).is_empty(), "{label} LTL row authenticated");
        }

        let mut third = scalar_fixture_with_signedness_companions(
            "MOVZX",
            1,
            "EAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Read,
        );
        third.rel_push("ltl_inst", (NODE, exact()));
        assert!(proofs(third).is_empty(), "third LTL row authenticated");
    }

    #[test]
    fn signedness_companion_rejects_every_other_rtl_candidate_interpretation() {
        let exact = || {
            RTLInst::Iload(
                MemoryChunk::MInt8Unsigned,
                Addressing::Aindexed(12),
                Arc::new(vec![BASE]),
                VALUE,
            )
        };
        let rejected = [
            ("duplicate-exact", exact()),
            (
                "wrong-address",
                RTLInst::Iload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(13),
                    Arc::new(vec![BASE]),
                    VALUE,
                ),
            ),
            (
                "wrong-args",
                RTLInst::Iload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![INDEX]),
                    VALUE,
                ),
            ),
            (
                "wrong-value",
                RTLInst::Iload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    TEMP,
                ),
            ),
            (
                "cross-width",
                RTLInst::Iload(
                    MemoryChunk::MInt16Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    VALUE,
                ),
            ),
            (
                "wrong-kind",
                RTLInst::Istore(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(12),
                    Arc::new(vec![BASE]),
                    VALUE,
                ),
            ),
        ];
        for (label, competing) in rejected {
            let mut db = scalar_fixture(
                "MOVZX",
                1,
                "EAX",
                MemoryChunk::MInt8Unsigned,
                ScalarMemoryDirection::Read,
                Mutation::None,
            );
            db.rel_push("rtl_inst_candidate", (NODE, competing));
            assert!(proofs(db).is_empty(), "{label} RTL candidate authenticated");
        }

        let mut third = scalar_fixture_with_signedness_companions(
            "MOVZX",
            1,
            "EAX",
            MemoryChunk::MInt8Unsigned,
            ScalarMemoryDirection::Read,
        );
        third.rel_push("rtl_inst_candidate", (NODE, exact()));
        assert!(
            proofs(third).is_empty(),
            "third RTL candidate authenticated"
        );
    }

    #[test]
    fn address_size_is_part_of_the_reversible_register_identity() {
        assert_eq!(
            decoded_scalar_addressing("ECX", "EDX", 4, -7, 4),
            Some((
                Addressing::Aaddr32(Box::new(Addressing::Aindexed2scaled(4, -7))),
                vec![Mreg::CX, Mreg::DX],
                false,
            ))
        );
        assert!(decoded_scalar_addressing("RCX", "RDX", 4, -7, 4).is_none());
        assert!(decoded_scalar_addressing("ECX", "EDX", 4, -7, 8).is_none());
        assert!(decoded_scalar_addressing("RCX", "NONE", 2, -7, 8).is_none());
        assert!(decoded_scalar_addressing("RCX", "RDX", 3, -7, 8).is_none());
        assert!(decoded_scalar_addressing("RIP", "NONE", 1, -7, 8).is_none());
    }

    #[test]
    fn scalar_coff_interval_index_is_unique_and_overlap_fail_closed() {
        let index = ScalarIntervalIndex::new(vec![
            ScalarInterval {
                start: 0x100,
                end: 0x120,
                payload: 1,
            },
            ScalarInterval {
                start: 0x110,
                end: 0x130,
                payload: 2,
            },
            ScalarInterval {
                start: 0x200,
                end: 0x210,
                payload: 3,
            },
        ]);
        assert_eq!(index.unique_overlap_payload(0x100, 0x101), Some(1));
        assert_eq!(index.unique_overlap_payload(0x10f, 0x110), Some(1));
        assert_eq!(index.unique_overlap_payload(0x110, 0x111), None);
        assert_eq!(index.unique_overlap_payload(0x200, 0x201), Some(3));
        assert!(index.has_overlap(0x12f, 0x131));
        assert!(!index.has_overlap(0x130, 0x140));
    }

    #[test]
    fn nonzero_original_section_origin_uses_equal_linear_coordinates() {
        let mut db = scalar_fixture(
            "MOV",
            4,
            "EAX",
            MemoryChunk::MInt32,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        let map = db.coff_address_map.as_mut().expect("map");
        map.sections[0].original_offset_start = 0x80;
        map.sections[0].original_offset_end = 0xc0;
        map.functions[0].section_offset = 0x80;
        assert_eq!(proofs(db).len(), 1);
    }

    fn defined_address_root(
        symbolic: bool,
        malformed_args: bool,
        nested_symbolic: bool,
    ) -> DecompileDB {
        let mut db = scalar_fixture(
            "MOV",
            4,
            "EAX",
            MemoryChunk::MInt32,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        let def = FUNCTION + 4;
        db.rel_set(
            "emit_function_param_candidate",
            vec![(FUNCTION, INDEX)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        let nested_def = FUNCTION + 2;
        db.rel_set(
            "emit_function",
            vec![(
                FUNCTION,
                "arbitrary_function",
                if nested_symbolic { nested_def } else { def },
            )]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        let operation = if nested_symbolic {
            Operation::Oaddl
        } else if symbolic {
            Operation::Olea(Addressing::Aglobal(7, 0))
        } else {
            Operation::Olea(Addressing::Aindexed(24))
        };
        let inst = RTLInst::Iop(
            operation,
            Arc::new(if malformed_args {
                Vec::new()
            } else if nested_symbolic {
                vec![TEMP, INDEX]
            } else {
                vec![INDEX]
            }),
            BASE,
        );
        db.rel_push("rtl_inst", (def, inst.clone()));
        db.rel_push("rtl_inst_candidate", (def, inst));
        db.rel_push("instr_in_function", (def, FUNCTION));
        db.rel_push("rtl_succ", (def, NODE));
        if nested_symbolic {
            let nested = RTLInst::Iop(
                Operation::Olea(Addressing::Aglobal(9, 0)),
                Arc::new(Vec::new()),
                TEMP,
            );
            db.rel_push("rtl_inst", (nested_def, nested.clone()));
            db.rel_push("rtl_inst_candidate", (nested_def, nested));
            db.rel_push("instr_in_function", (nested_def, FUNCTION));
            db.rel_push("rtl_succ", (nested_def, def));
        }
        db.rel_push("emit_inline_temp", BASE);
        if nested_symbolic {
            db.rel_push("emit_inline_temp", TEMP);
        }
        db
    }

    fn without_address_inline_markers(mut db: DecompileDB) -> DecompileDB {
        db.rel_set("emit_inline_temp", ascent::boxcar::Vec::<RTLReg>::new());
        db
    }

    fn noninline_address_root_mutation(mutation: &'static str) -> DecompileDB {
        let mut db = without_address_inline_markers(defined_address_root(false, false, false));
        let def = FUNCTION + 4;
        let extra = FUNCTION + 6;
        match mutation {
            "none" => {}
            "multiple-definitions" => {
                let competing = RTLInst::Iop(
                    Operation::Olea(Addressing::Aindexed(8)),
                    Arc::new(vec![INDEX]),
                    BASE,
                );
                db.rel_push("rtl_inst", (extra, competing.clone()));
                db.rel_push("rtl_inst_candidate", (extra, competing));
                db.rel_push("instr_in_function", (extra, FUNCTION));
            }
            "duplicate-definition-row" => {
                let duplicate = RTLInst::Iop(
                    Operation::Olea(Addressing::Aindexed(24)),
                    Arc::new(vec![INDEX]),
                    BASE,
                );
                db.rel_push("rtl_inst", (def, duplicate));
            }
            "multiple-uses" => {
                let competing = RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), TEMP);
                db.rel_push("rtl_inst", (extra, competing.clone()));
                db.rel_push("rtl_inst_candidate", (extra, competing));
                db.rel_push("instr_in_function", (extra, FUNCTION));
                db.rel_set(
                    "rtl_succ",
                    vec![(def, extra), (extra, NODE)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
            }
            "duplicate-use-row" => {
                db.rel_push(
                    "rtl_inst",
                    (
                        NODE,
                        RTLInst::Iload(
                            MemoryChunk::MInt32,
                            Addressing::Aindexed(0),
                            Arc::new(vec![BASE]),
                            VALUE,
                        ),
                    ),
                );
            }
            "hidden-memory-call-use" => {
                db.rel_push(
                    "call_through_memory_load",
                    (
                        extra,
                        TEMP,
                        MemoryChunk::MInt64,
                        Addressing::Aindexed(0),
                        Arc::new(vec![BASE]),
                    ),
                );
            }
            "definition-bypass" => {
                let entry = FUNCTION + 2;
                db.rel_push("rtl_inst", (entry, RTLInst::Inop));
                db.rel_push("rtl_inst_candidate", (entry, RTLInst::Inop));
                db.rel_push("instr_in_function", (entry, FUNCTION));
                db.rel_set(
                    "emit_function",
                    vec![(FUNCTION, "arbitrary_function", entry)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
                db.rel_set(
                    "rtl_succ",
                    vec![(entry, def), (entry, NODE), (def, NODE)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
            }
            "ambiguous-owner-bypass" | "unowned-bypass" => {
                let entry = FUNCTION + 2;
                db.rel_push("rtl_inst", (entry, RTLInst::Inop));
                db.rel_push("rtl_inst_candidate", (entry, RTLInst::Inop));
                db.rel_push("instr_in_function", (entry, FUNCTION));
                db.rel_push("rtl_inst", (extra, RTLInst::Inop));
                db.rel_push("rtl_inst_candidate", (extra, RTLInst::Inop));
                if mutation == "ambiguous-owner-bypass" {
                    db.rel_push("instr_in_function", (extra, FUNCTION));
                    db.rel_push("instr_in_function", (extra, FUNCTION + 0x200));
                }
                db.rel_set(
                    "emit_function",
                    vec![(FUNCTION, "arbitrary_function", entry)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
                db.rel_set(
                    "rtl_succ",
                    vec![(entry, def), (entry, extra), (def, NODE), (extra, NODE)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
            }
            "ambiguous-entry" => {
                db.rel_push(
                    "emit_function",
                    (FUNCTION, "competing_arbitrary_function", extra),
                );
            }
            "parameter-redefinition" => {
                let entry = FUNCTION + 2;
                let redefine = RTLInst::Iop(Operation::Ointconst(0), Arc::new(Vec::new()), INDEX);
                db.rel_push("rtl_inst", (entry, redefine.clone()));
                db.rel_push("rtl_inst_candidate", (entry, redefine));
                db.rel_push("instr_in_function", (entry, FUNCTION));
                db.rel_set(
                    "emit_function",
                    vec![(FUNCTION, "arbitrary_function", entry)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
                db.rel_set(
                    "rtl_succ",
                    vec![(entry, def), (def, NODE)]
                        .into_iter()
                        .collect::<ascent::boxcar::Vec<_>>(),
                );
            }
            _ => unreachable!(),
        }
        db
    }

    #[test]
    fn unique_use_address_dag_is_retained_but_symbolic_origin_is_rejected() {
        let rows = proofs(defined_address_root(false, false, false));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].base_value, Some(BASE));
        assert_eq!(rows[0].address_param_leaves.as_slice(), &[INDEX]);
        assert!(proofs(defined_address_root(true, false, false)).is_empty());
        assert!(proofs(defined_address_root(false, true, false)).is_empty());
        assert!(proofs(defined_address_root(false, false, true)).is_empty());
    }

    #[test]
    fn dead_at_use_address_dag_requires_exact_dominating_parameter_chain() {
        let rows = proofs(noninline_address_root_mutation("none"));
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].base_value, Some(BASE));
        assert_eq!(rows[0].address_param_leaves.as_slice(), &[INDEX]);

        for (label, db) in [
            (
                "symbolic-definition",
                without_address_inline_markers(defined_address_root(true, false, false)),
            ),
            (
                "malformed-definition",
                without_address_inline_markers(defined_address_root(false, true, false)),
            ),
            (
                "nested-symbolic-definition",
                without_address_inline_markers(defined_address_root(false, false, true)),
            ),
            (
                "multiple-definitions",
                noninline_address_root_mutation("multiple-definitions"),
            ),
            (
                "duplicate-definition-row",
                noninline_address_root_mutation("duplicate-definition-row"),
            ),
            (
                "multiple-uses",
                noninline_address_root_mutation("multiple-uses"),
            ),
            (
                "duplicate-use-row",
                noninline_address_root_mutation("duplicate-use-row"),
            ),
            (
                "hidden-memory-call-use",
                noninline_address_root_mutation("hidden-memory-call-use"),
            ),
            (
                "definition-bypass",
                noninline_address_root_mutation("definition-bypass"),
            ),
            (
                "ambiguous-owner-bypass",
                noninline_address_root_mutation("ambiguous-owner-bypass"),
            ),
            (
                "unowned-bypass",
                noninline_address_root_mutation("unowned-bypass"),
            ),
            (
                "ambiguous-entry",
                noninline_address_root_mutation("ambiguous-entry"),
            ),
            (
                "parameter-redefinition",
                noninline_address_root_mutation("parameter-redefinition"),
            ),
        ] {
            assert!(proofs(db).is_empty(), "{label} authenticated");
        }
    }

    #[test]
    fn cyclic_address_definition_dag_fails_closed() {
        let first = FUNCTION + 2;
        let second = FUNCTION + 4;
        let final_rtl = BTreeMap::from([
            (
                first,
                vec![RTLInst::Iop(Operation::Omove, Arc::new(vec![BASE]), TEMP)],
            ),
            (
                second,
                vec![RTLInst::Iop(Operation::Omove, Arc::new(vec![TEMP]), BASE)],
            ),
            (
                NODE,
                vec![RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(0),
                    Arc::new(vec![BASE]),
                    VALUE,
                )],
            ),
        ]);
        let definitions = BTreeMap::from([
            (TEMP, BTreeSet::from([first])),
            (BASE, BTreeSet::from([second])),
            (VALUE, BTreeSet::from([NODE])),
        ]);
        assert!(scalar_address_param_leaves(
            FUNCTION,
            NODE,
            &[BASE],
            first,
            &BTreeSet::from([first, second, NODE]),
            &definitions,
            &final_rtl,
            &BTreeMap::from([(BASE, vec![first, NODE]), (TEMP, vec![second])]),
            &HashMap::from([(first, vec![second]), (second, vec![NODE])]),
            &BTreeSet::new(),
            &BTreeSet::from([BASE, TEMP]),
            None,
            &mut ScalarAddressClosureCache::new(),
        )
        .is_none());
    }

    #[test]
    fn address_proof_work_limits_fail_closed() {
        let first = FUNCTION + 0x100;
        let mut allowed = BTreeSet::new();
        let mut succs = HashMap::new();
        for offset in 0..=SCALAR_ADDRESS_CFG_NODE_LIMIT {
            let node = first + offset as u64;
            allowed.insert(node);
            if offset != SCALAR_ADDRESS_CFG_NODE_LIMIT {
                succs.insert(node, vec![node + 1]);
            }
        }
        assert_eq!(
            graph_reaches_through_owned_region(
                &succs,
                &allowed,
                first,
                first + SCALAR_ADDRESS_CFG_NODE_LIMIT as u64,
                None,
            ),
            None
        );

        let mut visiting = BTreeSet::new();
        let mut cache = ScalarAddressClosureCache::new();
        let mut exhausted = 0;
        assert!(scalar_address_value_param_leaves(
            FUNCTION,
            BASE,
            NODE,
            NODE,
            &BTreeSet::from([NODE]),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &HashMap::new(),
            &BTreeSet::from([(FUNCTION, BASE)]),
            &BTreeSet::new(),
            &mut visiting,
            &mut cache,
            &mut exhausted,
        )
        .is_none());
    }

    #[test]
    fn scalar_memory_access_mutations_all_fail_closed() {
        let mutations = [
            Mutation::Prefix,
            Mutation::WrongOrder,
            Mutation::WrongWidth,
            Mutation::WrongRegisterWidth,
            Mutation::AmbiguousRegister,
            Mutation::CrossKindOperands,
            Mutation::WrongLtlValue,
            Mutation::AmbiguousLtl,
            Mutation::WrongDisplacement,
            Mutation::WrongAddressSize,
            Mutation::AmbiguousAddressSize,
            Mutation::Segment,
            Mutation::SymbolicBase,
            Mutation::MissingEffect,
            Mutation::ExtraEffect,
            Mutation::MissingRegisterDef,
            Mutation::ExtraRegisterDef,
            Mutation::MissingRegisterUse,
            Mutation::ExtraRegisterUse,
            Mutation::RawMismatch,
            Mutation::AmbiguousInstruction,
            Mutation::ExtraOperand,
            Mutation::AmbiguousOperand,
            Mutation::MissingCandidate,
            Mutation::CandidateDrift,
            Mutation::AmbiguousFinal,
            Mutation::AmbiguousOwner,
            Mutation::OverlappingFunction,
            Mutation::BadMapSpan,
            Mutation::NonlinearSection,
            Mutation::WrongSection,
            Mutation::RelocationOverlap,
            Mutation::MappedRelocationWrongSection,
            Mutation::RelocationOverflow,
            Mutation::WrongRelocationSectionName,
            Mutation::RedefinedRoot,
            Mutation::WrongDownstreamWidth,
            Mutation::AmbiguousDownstreamWidth,
        ];
        for mutation in mutations {
            assert!(
                proofs(scalar_fixture(
                    "MOV",
                    4,
                    "EAX",
                    MemoryChunk::MInt32,
                    ScalarMemoryDirection::Read,
                    mutation,
                ))
                .is_empty(),
                "mutation unexpectedly authenticated: {mutation:?}"
            );
        }
        assert!(proofs(scalar_fixture(
            "MOV",
            4,
            "EAX",
            MemoryChunk::MInt64,
            ScalarMemoryDirection::Read,
            Mutation::None,
        ))
        .is_empty());
        let mut missing_map = scalar_fixture(
            "MOV",
            4,
            "EAX",
            MemoryChunk::MInt32,
            ScalarMemoryDirection::Read,
            Mutation::None,
        );
        missing_map.coff_address_map = None;
        assert!(proofs(missing_map).is_empty());
    }

    fn synthetic_stack_fixture(
        extra_successor: bool,
        bad_origin: bool,
        bypass_origin: bool,
    ) -> DecompileDB {
        let selected = NODE | (1u64 << 62);
        let mut db = DecompileDB::default();
        db.coff_address_map = Some(coff_map());
        let decoded = instruction("", "MOV", MEMORY, REGISTER);
        push_instruction(&mut db, "instruction", NODE, decoded.clone());
        push_instruction(&mut db, "unrefinedinstruction", NODE, decoded);
        db.rel_push("op_register", (REGISTER, "EAX"));
        db.rel_push(
            "op_indirect",
            (MEMORY, "NONE", "RSP", "RDX", 4_i64, -32_i64, 4_usize),
        );
        db.rel_push("instruction_address_size", (NODE, 8_u8));
        db.rel_push("decoded_memory_read_operand", (NODE, MEMORY));
        db.rel_push("decoded_reg_def", (NODE, Mreg::AX));
        db.rel_push("decoded_reg_use", (NODE, Mreg::SP));
        db.rel_push("decoded_reg_use", (NODE, Mreg::DX));
        db.rel_push(
            "ltl_inst",
            (
                NODE,
                LTLInst::Lload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, -32),
                    Arc::new(vec![Mreg::SP, Mreg::DX]),
                    Mreg::AX,
                ),
            ),
        );
        let origin = RTLInst::Iop(
            Operation::Olea(Addressing::Ainstack(if bad_origin { -24 } else { -32 })),
            Arc::new(Vec::new()),
            TEMP,
        );
        let load = RTLInst::Iload(
            MemoryChunk::MInt32,
            Addressing::Aindexed2scaled(4, 0),
            Arc::new(vec![TEMP, INDEX]),
            VALUE,
        );
        db.rel_push("rtl_inst", (NODE, origin.clone()));
        db.rel_push("rtl_inst_candidate", (NODE, origin));
        db.rel_push("rtl_inst", (selected, load.clone()));
        db.rel_push("rtl_inst_candidate", (selected, load));
        db.rel_push("rtl_succ", (NODE, selected));
        db.rel_push("rtl_inst", (NEXT, RTLInst::Ireturn(VALUE)));
        db.rel_push("rtl_inst_candidate", (NEXT, RTLInst::Ireturn(VALUE)));
        db.rel_push("rtl_succ", (selected, NEXT));
        if extra_successor {
            db.rel_push("rtl_succ", (NODE, NEXT));
        }
        db.rel_push("instr_in_function", (NODE, FUNCTION));
        db.rel_push("instr_in_function", (selected, FUNCTION));
        db.rel_push("instr_in_function", (NEXT, FUNCTION));
        let entry = FUNCTION + 4;
        db.rel_push(
            "emit_function",
            (
                FUNCTION,
                "arbitrary_stack_function",
                if bypass_origin { entry } else { NODE },
            ),
        );
        if bypass_origin {
            db.rel_push("rtl_inst", (entry, RTLInst::Inop));
            db.rel_push("rtl_inst_candidate", (entry, RTLInst::Inop));
            db.rel_push("instr_in_function", (entry, FUNCTION));
            db.rel_push("rtl_succ", (entry, NODE));
            db.rel_push("rtl_succ", (entry, selected));
        }
        db.rel_push("emit_function_param_candidate", (FUNCTION, INDEX));
        db.rel_push("emit_var_type_candidate", (VALUE, XType::Xint));
        db
    }

    fn synthetic_stack_ownership_bypass(ambiguous: bool) -> DecompileDB {
        let mut db = synthetic_stack_fixture(false, false, false);
        let selected = NODE | (1u64 << 62);
        let entry = FUNCTION + 4;
        let intermediate = FUNCTION + 6;
        db.rel_push("rtl_inst", (entry, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (entry, RTLInst::Inop));
        db.rel_push("instr_in_function", (entry, FUNCTION));
        db.rel_push("rtl_inst", (intermediate, RTLInst::Inop));
        db.rel_push("rtl_inst_candidate", (intermediate, RTLInst::Inop));
        if ambiguous {
            db.rel_push("instr_in_function", (intermediate, FUNCTION));
            db.rel_push("instr_in_function", (intermediate, FUNCTION + 0x200));
        }
        db.rel_set(
            "emit_function",
            vec![(FUNCTION, "arbitrary_stack_function", entry)]
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        db.rel_set(
            "rtl_succ",
            vec![
                (entry, NODE),
                (entry, intermediate),
                (NODE, selected),
                (intermediate, selected),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        db
    }

    fn synthetic_stack_cfg_limit_fixture() -> DecompileDB {
        let mut db = synthetic_stack_fixture(false, false, false);
        let selected = NODE | (1u64 << 62);
        let first = FUNCTION + 0x100;
        let mut edges = Vec::new();
        let mut previous = NODE;
        for offset in 0..SCALAR_ADDRESS_CFG_NODE_LIMIT {
            let intermediate = first + offset as u64;
            db.rel_push("instr_in_function", (intermediate, FUNCTION));
            edges.push((previous, intermediate));
            previous = intermediate;
        }
        edges.push((previous, selected));
        db.rel_set(
            "rtl_succ",
            edges
                .into_iter()
                .collect::<ascent::boxcar::Vec<(Node, Node)>>(),
        );
        db
    }

    #[test]
    fn indexed_stack_origin_is_exact_and_ambiguity_fails_closed() {
        let rows = proofs(synthetic_stack_fixture(false, false, false));
        assert_eq!(rows.len(), 1);
        assert!(rows[0].synthetic_stack_origin);
        assert_eq!(rows[0].displacement, -32);
        assert!(rows[0].exact_scaled_index);
        assert_eq!(rows[0].address_param_leaves.as_slice(), &[INDEX]);
        assert!(proofs(synthetic_stack_fixture(true, false, false)).is_empty());
        assert!(proofs(synthetic_stack_fixture(false, true, false)).is_empty());
        assert!(proofs(synthetic_stack_fixture(false, false, true)).is_empty());
        assert!(proofs(synthetic_stack_ownership_bypass(true)).is_empty());
        assert!(proofs(synthetic_stack_ownership_bypass(false)).is_empty());
        assert!(proofs(synthetic_stack_cfg_limit_fixture()).is_empty());
    }

    fn final_plan_proof() -> ScalarMemoryAccessProof {
        ScalarMemoryAccessProof {
            function: FUNCTION,
            origin_node: NODE,
            selected_node: NODE,
            operand: MEMORY,
            direction: ScalarMemoryDirection::Read,
            extension: ScalarMemoryExtension::SignExtend,
            encoded_destination_width: Some(8),
            result_chain: Some(ScalarMemoryResultChain::Direct),
            address_size: 8,
            base_register: Mreg::AX,
            index_register: None,
            scale: 1,
            displacement: 0,
            width: 1,
            value_width: 8,
            downstream_value_width: Some(8),
            chunk: MemoryChunk::MInt8Signed,
            base_value: Some(BASE),
            index_value: None,
            value: VALUE,
            address_param_leaves: Arc::new(vec![BASE]),
            synthetic_stack_origin: false,
            exact_scaled_index: false,
        }
    }

    fn final_plan_context(
        rows: Vec<(Node, RTLInst)>,
        return_type: Option<XType>,
    ) -> ScalarMemoryFinalPlanContext {
        let mut db = DecompileDB::default();
        for (node, instruction) in rows {
            db.rel_push("rtl_inst", (node, instruction));
            db.rel_push("instr_in_function", (node, FUNCTION));
        }
        if let Some(return_type) = return_type {
            db.rel_push(
                "emit_function_return_type_xtype",
                (FUNCTION, return_type),
            );
        }
        ScalarMemoryFinalPlanContext::from_db(&db)
    }

    #[test]
    fn final_plan_allows_one_root_fanout_to_one_exact_shared_consumer() {
        let proof = final_plan_proof();
        let left = TEMP;
        let right = TEMP + 1;
        let consumer = NEXT + 8;
        let plan = ScalarMemoryUsePlan {
            function: FUNCTION,
            definition_node: NODE,
            value: VALUE,
            transports: Arc::new(vec![
                ScalarMemoryTransport {
                    node: NEXT,
                    input: VALUE,
                    output: left,
                    kind: ScalarMemoryTransportKind::Move,
                },
                ScalarMemoryTransport {
                    node: NEXT + 4,
                    input: VALUE,
                    output: right,
                    kind: ScalarMemoryTransportKind::Move,
                },
            ]),
            sites: Arc::new(vec![
                ScalarMemoryUseSite {
                    node: consumer,
                    value: left,
                    required_type: ScalarMemoryUseType::Signed64,
                },
                ScalarMemoryUseSite {
                    node: consumer,
                    value: right,
                    required_type: ScalarMemoryUseType::Signed64,
                },
            ]),
        };
        assert!(plan.is_closed_v1(&proof));
        let rows = vec![
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MInt8Signed,
                    Addressing::Aindexed(0),
                    Arc::new(vec![BASE]),
                    VALUE,
                ),
            ),
            (NEXT, RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE]), left)),
            (
                NEXT + 4,
                RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE]), right),
            ),
            (
                consumer,
                RTLInst::Iop(Operation::Oaddl, Arc::new(vec![left, right]), TEMP + 2),
            ),
        ];
        assert!(final_plan_context(rows.clone(), None).plan_is_closed(&proof, &plan));
        let mut duplicate = rows;
        duplicate.pop();
        duplicate.push((
            consumer,
            RTLInst::Iop(Operation::Oaddl, Arc::new(vec![left, left]), TEMP + 2),
        ));
        assert!(!final_plan_context(duplicate, None).plan_is_closed(&proof, &plan));
    }

    #[test]
    fn movsxd_transport_fanout_carries_exact_64_bit_results_to_qword_stores() {
        let mut proof = final_plan_proof();
        proof.extension = ScalarMemoryExtension::ImplicitZeroExtend;
        proof.encoded_destination_width = Some(4);
        proof.result_chain = Some(ScalarMemoryResultChain::ZeroUpper32);
        proof.width = 4;
        proof.chunk = MemoryChunk::MInt32;
        let signed = TEMP;
        let unsigned = TEMP + 1;
        let signed_store = NEXT + 8;
        let unsigned_store = NEXT + 12;
        let plan = ScalarMemoryUsePlan {
            function: FUNCTION,
            definition_node: NODE,
            value: VALUE,
            transports: Arc::new(vec![
                ScalarMemoryTransport {
                    node: NEXT,
                    input: VALUE,
                    output: signed,
                    kind: ScalarMemoryTransportKind::LaneCast(
                        ScalarMemoryUseType::Signed64,
                    ),
                },
                ScalarMemoryTransport {
                    node: NEXT + 4,
                    input: VALUE,
                    output: unsigned,
                    kind: ScalarMemoryTransportKind::LaneCast(
                        ScalarMemoryUseType::Unsigned64,
                    ),
                },
            ]),
            sites: Arc::new(vec![
                ScalarMemoryUseSite {
                    node: signed_store,
                    value: signed,
                    required_type: ScalarMemoryUseType::Signed64,
                },
                ScalarMemoryUseSite {
                    node: unsigned_store,
                    value: unsigned,
                    required_type: ScalarMemoryUseType::Unsigned64,
                },
            ]),
        };
        let rows = vec![
            (
                NODE,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed(0),
                    Arc::new(vec![BASE]),
                    VALUE,
                ),
            ),
            (
                NEXT,
                RTLInst::Iop(
                    Operation::Ocast32signed,
                    Arc::new(vec![VALUE]),
                    signed,
                ),
            ),
            (
                NEXT + 4,
                RTLInst::Iop(
                    Operation::Ocast32unsigned,
                    Arc::new(vec![VALUE]),
                    unsigned,
                ),
            ),
            (
                signed_store,
                RTLInst::Istore(
                    MemoryChunk::MAny64,
                    Addressing::Aindexed(0),
                    Arc::new(vec![BASE]),
                    signed,
                ),
            ),
            (
                unsigned_store,
                RTLInst::Istore(
                    MemoryChunk::MAny64,
                    Addressing::Aindexed(8),
                    Arc::new(vec![BASE]),
                    unsigned,
                ),
            ),
        ];
        assert!(plan.is_closed_v1(&proof));
        assert!(final_plan_context(rows, None).plan_is_closed(&proof, &plan));
    }

    #[test]
    fn final_plan_rebinds_calls_and_direct_or_transported_returns() {
        let proof = final_plan_proof();
        let load = (
            NODE,
            RTLInst::Iload(
                MemoryChunk::MInt8Signed,
                Addressing::Aindexed(0),
                Arc::new(vec![BASE]),
                VALUE,
            ),
        );
        let call_node = NEXT;
        let call_plan = ScalarMemoryUsePlan {
            function: FUNCTION,
            definition_node: NODE,
            value: VALUE,
            transports: Arc::new(Vec::new()),
            sites: Arc::new(vec![ScalarMemoryUseSite {
                node: call_node,
                value: VALUE,
                required_type: ScalarMemoryUseType::Signed64,
            }]),
        };
        let signature = |argument| Signature {
            sig_args: Arc::new(vec![argument]),
            sig_res: XType::Xvoid,
            sig_cc: CallConv::default(),
        };
        let call = |argument| {
            RTLInst::Icall(
                Some(signature(argument)),
                Either::Right(Either::Right("callee")),
                Arc::new(vec![VALUE]),
                None,
                call_node + 4,
            )
        };
        assert!(final_plan_context(vec![load.clone(), (call_node, call(XType::Xlong))], None)
            .plan_is_closed(&proof, &call_plan));
        assert!(!final_plan_context(vec![load.clone(), (call_node, call(XType::Xptr))], None)
            .plan_is_closed(&proof, &call_plan));
        let tail = |argument| {
            RTLInst::Itailcall(
                Some(signature(argument)),
                Either::Right(Either::Right("callee")),
                Arc::new(vec![VALUE]),
            )
        };
        assert!(final_plan_context(
            vec![load.clone(), (call_node, tail(XType::Xlong))],
            None,
        )
        .plan_is_closed(&proof, &call_plan));
        assert!(!final_plan_context(
            vec![load.clone(), (call_node, tail(XType::Xptr))],
            None,
        )
        .plan_is_closed(&proof, &call_plan));

        let direct_return = ScalarMemoryUsePlan {
            function: FUNCTION,
            definition_node: NODE,
            value: VALUE,
            transports: Arc::new(Vec::new()),
            sites: Arc::new(vec![ScalarMemoryUseSite {
                node: NEXT,
                value: VALUE,
                required_type: ScalarMemoryUseType::Signed64,
            }]),
        };
        assert!(final_plan_context(
            vec![load.clone(), (NEXT, RTLInst::Ireturn(VALUE))],
            Some(XType::Xlong),
        )
        .plan_is_closed(&proof, &direct_return));
        assert!(!final_plan_context(
            vec![load.clone(), (NEXT, RTLInst::Ireturn(VALUE))],
            Some(XType::Xptr),
        )
        .plan_is_closed(&proof, &direct_return));

        let narrow_context = final_plan_context(
            vec![load.clone(), (NEXT, RTLInst::Ireturn(VALUE))],
            Some(XType::Xint8signed),
        );
        assert!(!narrow_context.plan_is_closed(&proof, &direct_return));
        let finalized = narrow_context
            .finalized_plan(&proof, &direct_return)
            .expect("unique integral final return narrows the early obligation");
        assert_eq!(
            finalized.sites.as_slice(),
            &[ScalarMemoryUseSite {
                node: NEXT,
                value: VALUE,
                required_type: ScalarMemoryUseType::Signed8,
            }]
        );
        assert!(narrow_context.plan_is_closed(&proof, &finalized));
        assert!(final_plan_context(
            vec![load.clone(), (NEXT, RTLInst::Ireturn(VALUE))],
            Some(XType::Xptr),
        )
        .finalized_plan(&proof, &direct_return)
        .is_none());

        let mut narrow_proof = proof.clone();
        narrow_proof.encoded_destination_width = Some(4);
        narrow_proof.value_width = 4;
        narrow_proof.downstream_value_width = Some(4);
        narrow_proof.result_chain = Some(ScalarMemoryResultChain::Direct);
        assert!(final_plan_context(
            vec![load.clone(), (NEXT, RTLInst::Ireturn(VALUE))],
            Some(XType::Xlong),
        )
        .finalized_plan(&narrow_proof, &direct_return)
        .is_none());

        let transported_return = ScalarMemoryUsePlan {
            function: FUNCTION,
            definition_node: NODE,
            value: VALUE,
            transports: Arc::new(vec![ScalarMemoryTransport {
                node: NEXT,
                input: VALUE,
                output: TEMP,
                kind: ScalarMemoryTransportKind::Move,
            }]),
            sites: Arc::new(vec![ScalarMemoryUseSite {
                node: NEXT + 4,
                value: TEMP,
                required_type: ScalarMemoryUseType::Signed64,
            }]),
        };
        assert!(final_plan_context(
            vec![
                load,
                (NEXT, RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE]), TEMP)),
                (NEXT + 4, RTLInst::Ireturn(TEMP)),
            ],
            Some(XType::Xlong),
        )
        .plan_is_closed(&proof, &transported_return));
        assert!(!final_plan_context(
            vec![
                (
                    NODE,
                    RTLInst::Iload(
                        MemoryChunk::MInt8Signed,
                        Addressing::Aindexed(0),
                        Arc::new(vec![BASE]),
                        VALUE,
                    ),
                ),
                (NEXT, RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE]), TEMP)),
                (NEXT + 4, RTLInst::Ireturn(TEMP)),
            ],
            Some(XType::Xptr),
        )
        .plan_is_closed(&proof, &transported_return));
    }
}

#[cfg(test)]
#[path = "rtl_optimize_stage4_tests.rs"]
mod stage4_source_shape_tests;

fn low_byte_register(full: &str) -> Option<(&'static str, usize)> {
    match full {
        "RAX" => Some(("AL", 2)),
        "RBX" => Some(("BL", 3)),
        "RCX" => Some(("CL", 3)),
        "RDX" => Some(("DL", 3)),
        "RSI" => Some(("SIL", 4)),
        "RDI" => Some(("DIL", 4)),
        "R8" => Some(("R8B", 4)),
        "R9" => Some(("R9B", 4)),
        "R10" => Some(("R10B", 4)),
        "R11" => Some(("R11B", 4)),
        "R12" => Some(("R12B", 4)),
        "R13" => Some(("R13B", 4)),
        "R14" => Some(("R14B", 4)),
        "R15" => Some(("R15B", 4)),
        _ => None,
    }
}

fn rtl_definition(inst: &RTLInst) -> Option<RTLReg> {
    match inst {
        RTLInst::Iop(_, _, destination) | RTLInst::Iload(_, _, _, destination) => {
            Some(*destination)
        }
        RTLInst::Icall(_, _, _, destination, _) => *destination,
        RTLInst::Ibuiltin(_, _, BuiltinArg::BA(destination)) => Some(*destination),
        _ => None,
    }
}

const SCALAR_LVALUE_SYNTHETIC_MASK: Node = (1u64 << 62) | (1u64 << 63);

type ScalarIndirectOperand = (&'static str, &'static str, &'static str, i64, i64, usize);

fn x86_scalar_register_width(name: &str) -> Option<usize> {
    match name {
        "RAX" | "RBX" | "RCX" | "RDX" | "RSI" | "RDI" | "RBP" | "RSP" | "R8" | "R9" | "R10"
        | "R11" | "R12" | "R13" | "R14" | "R15" => Some(8),
        "EAX" | "EBX" | "ECX" | "EDX" | "ESI" | "EDI" | "EBP" | "ESP" | "R8D" | "R9D" | "R10D"
        | "R11D" | "R12D" | "R13D" | "R14D" | "R15D" => Some(4),
        "AX" | "BX" | "CX" | "DX" | "SI" | "DI" | "BP" | "SP" | "R8W" | "R9W" | "R10W" | "R11W"
        | "R12W" | "R13W" | "R14W" | "R15W" => Some(2),
        // AH/BH/CH/DH select bits 8..15 but collapse to the parent Mreg in
        // LTL.  V1 cannot reverse that alias without a sub-register offset, so
        // those four spellings deliberately have no scalar width here.
        "AL" | "BL" | "CL" | "DL" | "SIL" | "DIL" | "BPL" | "SPL" | "R8B" | "R9B" | "R10B"
        | "R11B" | "R12B" | "R13B" | "R14B" | "R15B" => Some(1),
        _ => None,
    }
}

fn x86_high8_register(name: &str) -> bool {
    matches!(name, "AH" | "BH" | "CH" | "DH")
}

fn scalar_integral_xtype_width(xtype: XType) -> Option<usize> {
    match xtype {
        XType::Xint8signed | XType::Xint8unsigned => Some(1),
        XType::Xint16signed | XType::Xint16unsigned => Some(2),
        XType::Xint | XType::Xintunsigned | XType::Xany32 => Some(4),
        XType::Xlong | XType::Xlongunsigned | XType::Xany64 => Some(8),
        XType::Xbool
        | XType::Xfloat
        | XType::Xsingle
        | XType::Xptr
        | XType::Xcharptr
        | XType::Xcharptrptr
        | XType::Xintptr
        | XType::Xfloatptr
        | XType::Xsingleptr
        | XType::Xfuncptr
        | XType::Xvoid
        | XType::XstructPtr(_) => None,
    }
}

fn scalar_unique_integral_xtype_width(types: Option<&BTreeSet<XType>>) -> Option<usize> {
    let types = types.filter(|types| !types.is_empty())?;
    let widths: Option<BTreeSet<usize>> = types
        .iter()
        .copied()
        .map(scalar_integral_xtype_width)
        .collect();
    let widths = widths?;
    (widths.len() == 1)
        .then(|| widths.iter().next().copied())
        .flatten()
}

fn scalar_mov_semantics(
    mnemonic: &str,
    direction: ScalarMemoryDirection,
    width: usize,
    value_width: usize,
) -> Option<(ScalarMemoryExtension, MemoryChunk)> {
    if !matches!(width, 1 | 2 | 4 | 8) {
        return None;
    }
    match (mnemonic, direction) {
        ("MOV", ScalarMemoryDirection::Read)
            if matches!(width, 1 | 2 | 4 | 8) && value_width == width =>
        {
            Some((
                ScalarMemoryExtension::Plain,
                match width {
                    1 => MemoryChunk::MInt8Unsigned,
                    2 => MemoryChunk::MInt16Unsigned,
                    4 => MemoryChunk::MInt32,
                    8 => MemoryChunk::MInt64,
                    _ => unreachable!(),
                },
            ))
        }
        ("MOV", ScalarMemoryDirection::Write) if value_width == width => Some((
            ScalarMemoryExtension::Plain,
            match width {
                1 => MemoryChunk::MInt8Unsigned,
                2 => MemoryChunk::MInt16Unsigned,
                4 => MemoryChunk::MInt32,
                8 => MemoryChunk::MInt64,
                _ => unreachable!(),
            },
        )),
        ("MOVSX", ScalarMemoryDirection::Read)
            if matches!(width, 1 | 2) && matches!(value_width, 4 | 8) && value_width > width =>
        {
            Some((
                ScalarMemoryExtension::SignExtend,
                if width == 1 {
                    MemoryChunk::MInt8Signed
                } else {
                    MemoryChunk::MInt16Signed
                },
            ))
        }
        ("MOVSXD", ScalarMemoryDirection::Read) if width == 4 && value_width == 8 => {
            Some((ScalarMemoryExtension::SignExtend, MemoryChunk::MInt32))
        }
        ("MOVZX", ScalarMemoryDirection::Read)
            if matches!(width, 1 | 2) && matches!(value_width, 4 | 8) && value_width > width =>
        {
            Some((
                ScalarMemoryExtension::ZeroExtend,
                if width == 1 {
                    MemoryChunk::MInt8Unsigned
                } else {
                    MemoryChunk::MInt16Unsigned
                },
            ))
        }
        _ => None,
    }
}

/// Bind raw opcode semantics to the selected final transport chunk.  An exact
/// chunk (or AsmPass's closed plain-qword `MAny64` alias) needs no companion.
/// MOVSX/MOVZX may select the opposite same-width byte/word signedness only
/// when both retained RTL and LTL layers contain the exact sealed pair.
/// Every other disagreement remains a hard veto.  The selected chunk is
/// retained in the proof; the raw opcode remains authoritative for extension.
fn scalar_transport_companion_required(
    canonical: MemoryChunk,
    selected: MemoryChunk,
    direction: ScalarMemoryDirection,
    extension: ScalarMemoryExtension,
    width: usize,
    value_width: usize,
) -> Option<bool> {
    if selected == canonical {
        return Some(false);
    }
    if matches!(
        (
            direction,
            extension,
            width,
            value_width,
            canonical,
            selected
        ),
        (
            ScalarMemoryDirection::Read | ScalarMemoryDirection::Write,
            ScalarMemoryExtension::Plain,
            8,
            8,
            MemoryChunk::MInt64,
            MemoryChunk::MAny64
        )
    ) {
        return Some(false);
    }
    if direction == ScalarMemoryDirection::Read
        && extension == ScalarMemoryExtension::Plain
        && matches!(width, 1 | 2)
        && value_width == width
        && scalar_signedness_companion(canonical, width) == Some(selected)
    {
        return Some(true);
    }
    (direction == ScalarMemoryDirection::Read
        && matches!(
            extension,
            ScalarMemoryExtension::SignExtend | ScalarMemoryExtension::ZeroExtend
        )
        && matches!(width, 1 | 2)
        && matches!(value_width, 4 | 8)
        && value_width > width
        && scalar_signedness_companion(canonical, width) == Some(selected))
    .then_some(true)
}

fn scalar_signedness_companion(chunk: MemoryChunk, width: usize) -> Option<MemoryChunk> {
    match (chunk, width) {
        (MemoryChunk::MInt8Signed, 1) => Some(MemoryChunk::MInt8Unsigned),
        (MemoryChunk::MInt8Unsigned, 1) => Some(MemoryChunk::MInt8Signed),
        (MemoryChunk::MInt16Signed, 2) => Some(MemoryChunk::MInt16Unsigned),
        (MemoryChunk::MInt16Unsigned, 2) => Some(MemoryChunk::MInt16Signed),
        _ => None,
    }
}

fn scalar_memory_use_type(width: usize, signed: bool) -> Option<ScalarMemoryUseType> {
    match (width, signed) {
        (1, true) => Some(ScalarMemoryUseType::Signed8),
        (1, false) => Some(ScalarMemoryUseType::Unsigned8),
        (2, true) => Some(ScalarMemoryUseType::Signed16),
        (2, false) => Some(ScalarMemoryUseType::Unsigned16),
        (4, true) => Some(ScalarMemoryUseType::Signed32),
        (4, false) => Some(ScalarMemoryUseType::Unsigned32),
        (8, true) => Some(ScalarMemoryUseType::Signed64),
        (8, false) => Some(ScalarMemoryUseType::Unsigned64),
        _ => None,
    }
}

pub(crate) fn scalar_memory_use_type_from_xtype(xtype: XType) -> Option<ScalarMemoryUseType> {
    match xtype {
        XType::Xint8signed => Some(ScalarMemoryUseType::Signed8),
        XType::Xint8unsigned => Some(ScalarMemoryUseType::Unsigned8),
        XType::Xint16signed => Some(ScalarMemoryUseType::Signed16),
        XType::Xint16unsigned => Some(ScalarMemoryUseType::Unsigned16),
        XType::Xint => Some(ScalarMemoryUseType::Signed32),
        XType::Xintunsigned => Some(ScalarMemoryUseType::Unsigned32),
        XType::Xlong => Some(ScalarMemoryUseType::Signed64),
        XType::Xlongunsigned => Some(ScalarMemoryUseType::Unsigned64),
        XType::Xany32
        | XType::Xany64
        | XType::Xbool
        | XType::Xfloat
        | XType::Xsingle
        | XType::Xptr
        | XType::Xcharptr
        | XType::Xcharptrptr
        | XType::Xintptr
        | XType::Xfloatptr
        | XType::Xsingleptr
        | XType::Xfuncptr
        | XType::Xvoid
        | XType::XstructPtr(_) => None,
    }
}

fn scalar_default_use_type(
    extension: ScalarMemoryExtension,
    chunk: MemoryChunk,
    value_width: usize,
) -> Option<ScalarMemoryUseType> {
    let signed = match extension {
        ScalarMemoryExtension::SignExtend => true,
        ScalarMemoryExtension::ZeroExtend | ScalarMemoryExtension::ImplicitZeroExtend => false,
        ScalarMemoryExtension::Plain => matches!(
            chunk,
            MemoryChunk::MInt8Signed | MemoryChunk::MInt16Signed
        ),
    };
    scalar_memory_use_type(value_width, signed)
}

fn scalar_use_type_at_width(
    fallback: ScalarMemoryUseType,
    wide_fallback: Option<ScalarMemoryUseType>,
    width: usize,
) -> Option<ScalarMemoryUseType> {
    (fallback.width() == width)
        .then_some(fallback)
        .or_else(|| wide_fallback.filter(|candidate| candidate.width() == width))
}

fn scalar_condition_use_type(
    condition: &Condition,
    fallback: ScalarMemoryUseType,
    wide_fallback: Option<ScalarMemoryUseType>,
) -> Option<ScalarMemoryUseType> {
    match condition {
        Condition::Ccomp(_) | Condition::Ccompimm(..) => {
            Some(ScalarMemoryUseType::Signed32)
        }
        Condition::Ccompu(_) | Condition::Ccompuimm(..) => {
            Some(ScalarMemoryUseType::Unsigned32)
        }
        Condition::Ccompl(_) | Condition::Ccomplimm(..) => {
            Some(ScalarMemoryUseType::Signed64)
        }
        Condition::Ccomplu(_) | Condition::Ccompluimm(..) => {
            Some(ScalarMemoryUseType::Unsigned64)
        }
        Condition::Cmaskzero(_) | Condition::Cmasknotzero(_) => {
            scalar_use_type_at_width(fallback, wide_fallback, 4)
        }
        Condition::Coverflow | Condition::Cnotoverflow => Some(fallback),
        Condition::Ccompf(_)
        | Condition::Cnotcompf(_)
        | Condition::Ccompfs(_)
        | Condition::Cnotcompfs(_) => None,
    }
}

fn scalar_operation_use_type(
    operation: &Operation,
    fallback: ScalarMemoryUseType,
    wide_fallback: Option<ScalarMemoryUseType>,
) -> Option<ScalarMemoryUseType> {
    use Operation::*;
    match operation {
        Ocast8signed => Some(ScalarMemoryUseType::Signed8),
        Ocast8unsigned => Some(ScalarMemoryUseType::Unsigned8),
        Ocast16signed => Some(ScalarMemoryUseType::Signed16),
        Ocast16unsigned => Some(ScalarMemoryUseType::Unsigned16),
        Ocast32signed => Some(ScalarMemoryUseType::Signed32),
        Ocast32unsigned => Some(ScalarMemoryUseType::Unsigned32),

        Omulhs | Odiv | Omod | Odivimm(_) | Omodimm(_) | Oshr | Oshrimm(_)
        | Oshrximm(_) => Some(ScalarMemoryUseType::Signed32),
        Omulhu | Odivu | Omodu | Odivuimm(_) | Omoduimm(_) | Oshru | Oshruimm(_) => {
            Some(ScalarMemoryUseType::Unsigned32)
        }
        Oadd | Oaddimm(_) | Osub | Omul | Omulimm(_) | Oand | Oandimm(_) | Oor
        | Oorimm(_) | Oxor | Oxorimm(_) | Onot | Oshl | Oshlimm(_) | Ororimm(_)
        | Oshldimm(_) | Oneg => scalar_use_type_at_width(fallback, wide_fallback, 4),

        Omullhs | Odivl | Omodl | Odivlimm(_) | Omodlimm(_) | Oshrl | Oshrlimm(_)
        | Oshrxlimm(_) => Some(ScalarMemoryUseType::Signed64),
        Omullhu | Odivlu | Omodlu | Odivluimm(_) | Omodluimm(_) | Oshrlu
        | Oshrluimm(_) => Some(ScalarMemoryUseType::Unsigned64),
        Oaddl | Oaddlimm(_) | Osubl | Omull | Omullimm(_) | Oandl | Oandlimm(_)
        | Oorl | Oorlimm(_) | Oxorl | Oxorlimm(_) | Onotl | Oshll | Oshllimm(_)
        | Ororlimm(_) | Onegl => scalar_use_type_at_width(fallback, wide_fallback, 8),

        Ocmp(condition) => scalar_condition_use_type(condition, fallback, wide_fallback),

        // A move/callee/address/float/select edge transfers more than one
        // local typing obligation. Stage 3 does not guess through it.
        Omove
        | Ointconst(_)
        | Olongconst(_)
        | Ofloatconst(_)
        | Osingleconst(_)
        | Oindirectsymbol(_)
        | Olea(_)
        | Oleal(_)
        | Omakelong
        | Olowlong
        | Ohighlong
        | Onegf
        | Oabsf
        | Oaddf
        | Osubf
        | Omulf
        | Odivf
        | Omaxf
        | Ominf
        | Onegfs
        | Oabsfs
        | Oaddfs
        | Osubfs
        | Omulfs
        | Odivfs
        | Osingleoffloat
        | Ofloatofsingle
        | Ointoffloat
        | Ofloatofint
        | Ointofsingle
        | Osingleofint
        | Olongoffloat
        | Ofloatoflong
        | Olongofsingle
        | Osingleoflong
        | Osel(..) => None,
    }
}

fn scalar_store_use_type(
    chunk: MemoryChunk,
    fallback: ScalarMemoryUseType,
    wide_fallback: Option<ScalarMemoryUseType>,
) -> Option<ScalarMemoryUseType> {
    match chunk {
        MemoryChunk::MInt8Signed => Some(ScalarMemoryUseType::Signed8),
        MemoryChunk::MInt8Unsigned => Some(ScalarMemoryUseType::Unsigned8),
        MemoryChunk::MInt16Signed => Some(ScalarMemoryUseType::Signed16),
        MemoryChunk::MInt16Unsigned => Some(ScalarMemoryUseType::Unsigned16),
        MemoryChunk::MInt32 | MemoryChunk::MAny32 => {
            scalar_use_type_at_width(fallback, wide_fallback, 4)
        }
        MemoryChunk::MInt64 | MemoryChunk::MAny64 => {
            scalar_use_type_at_width(fallback, wide_fallback, 8)
        }
        MemoryChunk::MBool
        | MemoryChunk::MFloat32
        | MemoryChunk::MFloat64
        | MemoryChunk::Unknown => None,
    }
}

fn scalar_value_use_type(
    instruction: &RTLInst,
    value: RTLReg,
    fallback: ScalarMemoryUseType,
    wide_fallback: Option<ScalarMemoryUseType>,
) -> Option<ScalarMemoryUseType> {
    let occurrence_count = inst_def_use(instruction)
        .1
        .into_iter()
        .filter(|candidate| *candidate == value)
        .count();
    if occurrence_count != 1 {
        return None;
    }
    match instruction {
        RTLInst::Iop(operation, args, _) if args.iter().filter(|arg| **arg == value).count() == 1 => {
            scalar_operation_use_type(operation, fallback, wide_fallback)
        }
        RTLInst::Istore(chunk, _, address, source)
            if *source == value && !address.iter().any(|argument| *argument == value) =>
        {
            scalar_store_use_type(*chunk, fallback, wide_fallback)
        }
        RTLInst::Icall(Some(signature), callee, args, _, _)
            if !matches!(callee, Either::Left(register) if *register == value) =>
        {
            let positions: Vec<usize> = args
                .iter()
                .enumerate()
                .filter_map(|(position, argument)| (*argument == value).then_some(position))
                .collect();
            let [position] = positions.as_slice() else {
                return None;
            };
            signature
                .sig_args
                .get(*position)
                .copied()
                .and_then(scalar_memory_use_type_from_xtype)
        }
        RTLInst::Itailcall(Some(signature), callee, args)
            if !matches!(callee, Either::Left(register) if *register == value) =>
        {
            let positions: Vec<usize> = args
                .iter()
                .enumerate()
                .filter_map(|(position, argument)| (*argument == value).then_some(position))
                .collect();
            let [position] = positions.as_slice() else {
                return None;
            };
            signature
                .sig_args
                .get(*position)
                .copied()
                .and_then(scalar_memory_use_type_from_xtype)
        }
        RTLInst::Icond(condition, args, _, _)
            if args.iter().filter(|argument| **argument == value).count() == 1 =>
        {
            scalar_condition_use_type(condition, fallback, wide_fallback)
        }
        RTLInst::Ireturn(returned) if *returned == value => Some(wide_fallback.unwrap_or(fallback)),
        RTLInst::Inop
        | RTLInst::Iop(..)
        | RTLInst::Iload(..)
        | RTLInst::Istore(..)
        | RTLInst::Icall(..)
        | RTLInst::Itailcall(..)
        | RTLInst::Ibuiltin(..)
        | RTLInst::Icond(..)
        | RTLInst::Ijumptable(..)
        | RTLInst::Ibranch(..)
        | RTLInst::Ireturn(..) => None,
    }
}

fn ltl_inst_mreg_effects(instruction: &LTLInst) -> Option<(BTreeSet<Mreg>, Vec<Mreg>)> {
    let mut definitions = BTreeSet::new();
    let mut uses = Vec::new();
    match instruction {
        LTLInst::Lop(_, args, destination) | LTLInst::Lload(_, _, args, destination) => {
            uses.extend(args.iter().copied());
            definitions.insert(*destination);
        }
        LTLInst::Lstore(_, _, args, source) => {
            uses.extend(args.iter().copied());
            uses.push(*source);
        }
        LTLInst::Lsetstack(source, _, _, _) => uses.push(*source),
        LTLInst::Lgetstack(_, _, _, destination) => {
            definitions.insert(*destination);
        }
        LTLInst::Lcond(_, args, _, _) => uses.extend(args.iter().copied()),
        LTLInst::Ljumptable(value, _) => uses.push(*value),
        // Calls carry their argument registers through the independently
        // authenticated call-argument map, not in the LTL instruction.  A
        // low-lane proof therefore cannot close across one of these rows.
        LTLInst::Lcall(_)
        | LTLInst::Ltailcall(_)
        | LTLInst::Lbuiltin(..)
        | LTLInst::Lbranch(_)
        | LTLInst::Lreturn => return None,
    }
    Some((definitions, uses))
}

/// Prove that one final RTL consumer observes exactly the low byte/word lane
/// selected by a plain partial-register MOV.  `reg_rtl` ties the surviving
/// value to one physical register at this instruction, while the immutable
/// decoder operands and LTL row independently require one use of that same
/// low-lane spelling.  Address uses, high-byte registers, whole-parent uses,
/// duplicate operands, and implicit ABI uses all reject.
fn scalar_low_lane_use_matches(
    node: Node,
    value: RTLReg,
    width: usize,
    instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    raw_instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    indirects: &BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>>,
    rtl_mregs: &BTreeMap<(Node, RTLReg), BTreeSet<Mreg>>,
    decoded_register_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_register_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    ltl: &BTreeMap<Node, Vec<LTLInst>>,
) -> bool {
    if !matches!(width, 1 | 2) {
        return false;
    }
    let Some(expected) = rtl_mregs.get(&(node, value)).and_then(|rows| {
        (rows.len() == 1)
            .then(|| rows.iter().next().copied())
            .flatten()
    }) else {
        return false;
    };
    let Some(instruction) = instructions.get(&node).and_then(|rows| unique_equal(rows)) else {
        return false;
    };
    if raw_instructions
        .get(&node)
        .and_then(|rows| unique_equal(rows))
        != Some(instruction)
    {
        return false;
    }
    let Some(operand_count) = exact_raw_operand_count(instruction) else {
        return false;
    };
    let operands = [instruction.3, instruction.4, instruction.5, instruction.6];
    let mut exact_lane_operands = 0usize;
    for (position, operand) in operands.into_iter().take(operand_count).enumerate() {
        if let Some(rows) = registers.get(&operand) {
            let Some(name) = (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
            else {
                return false;
            };
            if Mreg::x86(name) == expected {
                let write_only_destination = position == 1
                    && matches!(instruction.2, "MOV" | "MOVSX" | "MOVSXD" | "MOVZX" | "LEA")
                    && decoded_register_defs
                        .get(&node)
                        .is_some_and(|defs| defs.contains(&expected));
                if write_only_destination {
                    continue;
                }
                if x86_high8_register(name) || x86_scalar_register_width(name) != Some(width) {
                    return false;
                }
                exact_lane_operands += 1;
            }
        }
        if let Some(rows) = indirects.get(&operand) {
            let Some((_segment, base, index, _scale, _displacement, _operand_width)) =
                (rows.len() == 1)
                    .then(|| rows.iter().next().copied())
                    .flatten()
            else {
                return false;
            };
            if Mreg::x86(base) == expected
                || (!matches!(index, "" | "NONE") && Mreg::x86(index) == expected)
            {
                return false;
            }
        }
    }
    if exact_lane_operands != 1 {
        return false;
    }
    let Some((ltl_definitions, ltl_uses)) = ltl
        .get(&node)
        .and_then(|rows| unique_equal(rows))
        .and_then(ltl_inst_mreg_effects)
    else {
        return false;
    };
    let ltl_use_set: BTreeSet<_> = ltl_uses.iter().copied().collect();
    if decoded_register_defs.get(&node).cloned().unwrap_or_default() != ltl_definitions
        || decoded_register_uses.get(&node).cloned().unwrap_or_default() != ltl_use_set
    {
        return false;
    }
    ltl_uses
        .iter()
        .filter(|register| **register == expected)
        .count()
        == 1
}

/// Classify the only transport operations admitted to the private Stage-3
/// use graph.  A move/cast-looking instruction with any extra operand, an
/// in-place output, or a cast wider than the authenticated result is an error,
/// not a terminal use: accepting it as a leaf would hide a def/use edge.
fn scalar_memory_transport(
    proof: &ScalarMemoryAccessProof,
    instruction: &RTLInst,
    input: RTLReg,
) -> Result<Option<(RTLReg, ScalarMemoryTransportKind)>, ()> {
    let RTLInst::Iop(operation, args, output) = instruction else {
        return Ok(None);
    };
    let kind = match operation {
        Operation::Omove => Some(ScalarMemoryTransportKind::Move),
        Operation::Ocast8signed => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Signed8,
        )),
        Operation::Ocast8unsigned => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Unsigned8,
        )),
        Operation::Ocast16signed => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Signed16,
        )),
        Operation::Ocast16unsigned => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Unsigned16,
        )),
        // Cminor lowers these two operations to long-of-int/long-of-uint: the
        // input obligation is 32-bit, but the transported result is an exact
        // 64-bit carrier and may therefore feed an authenticated qword use.
        Operation::Ocast32signed => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Signed64,
        )),
        Operation::Ocast32unsigned => Some(ScalarMemoryTransportKind::LaneCast(
            ScalarMemoryUseType::Unsigned64,
        )),
        _ => None,
    };
    let Some(kind) = kind else {
        return Ok(None);
    };
    if args.as_slice() != [input] || *output == input {
        return Err(());
    }
    if let ScalarMemoryTransportKind::LaneCast(required_type) = kind {
        if required_type.width() > proof.value_width {
            return Err(());
        }
    }
    Ok(Some((*output, kind)))
}

#[allow(clippy::too_many_arguments)]
fn scalar_memory_use_plan(
    proof: &ScalarMemoryAccessProof,
    entry: Node,
    function_nodes: &BTreeSet<Node>,
    definitions: &BTreeMap<RTLReg, BTreeSet<Node>>,
    uses: &BTreeMap<RTLReg, Vec<Node>>,
    final_rtl: &BTreeMap<Node, Vec<RTLInst>>,
    succs: &HashMap<Node, Vec<Node>>,
    instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    raw_instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    indirects: &BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>>,
    rtl_mregs: &BTreeMap<(Node, RTLReg), BTreeSet<Mreg>>,
    decoded_register_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_register_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    ltl: &BTreeMap<Node, Vec<LTLInst>>,
    low_lane_fallback: Option<ScalarMemoryUseType>,
) -> Option<ScalarMemoryUsePlan> {
    if proof.direction != ScalarMemoryDirection::Read
        || definitions.get(&proof.value) != Some(&BTreeSet::from([proof.selected_node]))
    {
        return None;
    }
    let mut root_fallback = if proof.extension == ScalarMemoryExtension::Plain
        && matches!(proof.width, 1 | 2)
    {
        low_lane_fallback?
    } else {
        scalar_default_use_type(proof.extension, proof.chunk, proof.value_width)?
    };
    let root_wide_fallback = match proof.result_chain {
        Some(ScalarMemoryResultChain::ZeroUpper32) => {
            root_fallback = scalar_memory_use_type(
                proof.encoded_destination_width?,
                proof.extension == ScalarMemoryExtension::SignExtend,
            )?;
            Some(ScalarMemoryUseType::Unsigned64)
        }
        Some(ScalarMemoryResultChain::Direct) => None,
        None => return None,
    };
    let root_needs_low_lane = proof.extension == ScalarMemoryExtension::Plain
        && matches!(proof.width, 1 | 2);
    let mut pending = VecDeque::from([(
        proof.value,
        proof.selected_node,
        root_fallback,
        root_wide_fallback,
        root_needs_low_lane,
    )]);
    let mut queued_values = BTreeSet::from([proof.value]);
    let mut visited_values = BTreeSet::new();
    let mut visited_transport_nodes = BTreeSet::new();
    let mut visited_site_occurrences = BTreeSet::new();
    let mut use_count = 0usize;
    let mut transports = Vec::new();
    let mut sites = Vec::new();
    while let Some((
        current_value,
        current_definition,
        fallback,
        wide_fallback,
        needs_low_lane,
    )) = pending.pop_front()
    {
        if !visited_values.insert(current_value) || visited_values.len() > 128 {
            return None;
        }
        if definitions.get(&current_value) != Some(&BTreeSet::from([current_definition])) {
            return None;
        }
        let mut use_nodes = uses.get(&current_value)?.clone();
        use_nodes.sort_unstable();
        if use_nodes.is_empty()
            || use_nodes.len() > 128
            || use_nodes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return None;
        }
        use_count = use_count.checked_add(use_nodes.len())?;
        if use_count > 128 {
            return None;
        }

        for node in use_nodes {
            if node == current_definition || !function_nodes.contains(&node)
            {
                return None;
            }
            let definition_reaches_use = graph_reaches_through_owned_region(
                succs,
                function_nodes,
                current_definition,
                node,
                None,
            ) == Some(true);
            let no_definition_bypass = current_definition == entry
                || graph_reaches_through_owned_region(
                    succs,
                    function_nodes,
                    entry,
                    node,
                    Some(current_definition),
                ) == Some(false);
            if !definition_reaches_use || !no_definition_bypass {
                return None;
            }
            if needs_low_lane
                && !scalar_low_lane_use_matches(
                    node,
                    current_value,
                    proof.width,
                    instructions,
                    raw_instructions,
                    registers,
                    indirects,
                    rtl_mregs,
                    decoded_register_defs,
                    decoded_register_uses,
                    ltl,
                )
            {
                return None;
            }
            let instruction = final_rtl.get(&node).and_then(|rows| unique_equal(rows))?;
            if let Some((output, kind)) =
                scalar_memory_transport(proof, instruction, current_value).ok()?
            {
                if !visited_transport_nodes.insert(node)
                    || visited_site_occurrences
                        .iter()
                        .any(|(site_node, _)| *site_node == node)
                    || transports.len() >= 128
                    || definitions.get(&output) != Some(&BTreeSet::from([node]))
                    || !queued_values.insert(output)
                {
                    return None;
                }
                let (output_fallback, output_wide_fallback, output_needs_low_lane) = match kind {
                    ScalarMemoryTransportKind::Move => {
                        (fallback, wide_fallback, needs_low_lane)
                    }
                    ScalarMemoryTransportKind::LaneCast(required_type) => {
                        (required_type, None, false)
                    }
                };
                transports.push(ScalarMemoryTransport {
                    node,
                    input: current_value,
                    output,
                    kind,
                });
                pending.push_back((
                    output,
                    node,
                    output_fallback,
                    output_wide_fallback,
                    output_needs_low_lane,
                ));
                continue;
            }
            let required_type =
                scalar_value_use_type(instruction, current_value, fallback, wide_fallback)?;
            if visited_transport_nodes.contains(&node)
                || !visited_site_occurrences.insert((node, current_value))
            {
                return None;
            }
            if needs_low_lane
                && required_type.width() != proof.width
            {
                return None;
            }
            sites.push(ScalarMemoryUseSite {
                node,
                value: current_value,
                required_type,
            });
        }
    }
    transports.sort();
    sites.sort();
    // Repeated uses within one instruction are already rejected by
    // scalar_value_use_type.  A repeated node from a hidden/non-RTL use must
    // not disappear through deduplication either.
    if sites.windows(2).any(|pair| pair[0] >= pair[1]) {
        return None;
    }
    let plan = ScalarMemoryUsePlan {
        function: proof.function,
        definition_node: proof.selected_node,
        value: proof.value,
        transports: Arc::new(transports),
        sites: Arc::new(sites),
    };
    plan.is_closed_v1(proof).then_some(plan)
}

/// One post-signature index over the final RTL relation.  Cshminor builds it
/// once, after signature reconciliation has patched call rows and final
/// prototypes, then revalidates every early authenticated use plan against
/// these exact rows.  Conflicting RTL rows retain only their touched values as
/// an ambiguity veto; they can never disappear through relation iteration.
pub(crate) struct ScalarMemoryFinalPlanContext {
    instructions: BTreeMap<Node, RTLInst>,
    ambiguous_values: BTreeSet<RTLReg>,
    definitions: BTreeMap<RTLReg, Vec<Node>>,
    uses: BTreeMap<RTLReg, Vec<Node>>,
    owners: BTreeMap<Node, BTreeSet<Address>>,
    return_types: BTreeMap<Address, BTreeSet<XType>>,
}

impl ScalarMemoryFinalPlanContext {
    pub(crate) fn from_db(db: &DecompileDB) -> Self {
        let mut rows_by_node: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
        for (node, instruction) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
            rows_by_node
                .entry(*node)
                .or_default()
                .push(instruction.clone());
        }
        let mut instructions = BTreeMap::new();
        let mut ambiguous_values = BTreeSet::new();
        for (node, rows) in rows_by_node {
            if let Some(instruction) = unique_equal(&rows) {
                instructions.insert(node, instruction.clone());
            } else {
                for instruction in rows {
                    let (definition, uses) = inst_def_use(&instruction);
                    ambiguous_values.extend(definition);
                    ambiguous_values.extend(uses);
                }
            }
        }
        let mut definitions: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
        let mut uses: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
        for (node, instruction) in &instructions {
            let (definition, instruction_uses) = inst_def_use(instruction);
            if let Some(definition) = definition {
                definitions.entry(definition).or_default().push(*node);
            }
            for value in instruction_uses {
                uses.entry(value).or_default().push(*node);
            }
        }
        let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
        for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
            owners.entry(*node).or_default().insert(*function);
        }
        let mut return_types: BTreeMap<Address, BTreeSet<XType>> = BTreeMap::new();
        for (function, xtype) in
            db.rel_iter::<(Address, XType)>("emit_function_return_type_xtype")
        {
            return_types.entry(*function).or_default().insert(*xtype);
        }
        Self {
            instructions,
            ambiguous_values,
            definitions,
            uses,
            owners,
            return_types,
        }
    }

    fn node_is_exactly_owned(&self, node: Node, function: Address) -> bool {
        self.owners
            .get(&node)
            .is_some_and(|owners| owners.len() == 1 && owners.contains(&function))
    }

    /// Rebind an early return-site obligation to the exact reconciled return
    /// type before the final plan check.  RTLOptimize necessarily records an
    /// Ireturn using the architectural fallback because function signatures
    /// are reconciled later.  Cshminor may narrow that one obligation only
    /// when the final function return type is unique, integral, and no wider
    /// than the authenticated result.  Calls and every other use retain their
    /// original machine-derived requirement.
    pub(crate) fn finalized_plan(
        &self,
        proof: &ScalarMemoryAccessProof,
        plan: &ScalarMemoryUsePlan,
    ) -> Option<ScalarMemoryUsePlan> {
        if !proof.is_closed_v1() || !plan.is_closed_v1(proof) {
            return None;
        }
        let mut sites = plan.sites.as_ref().clone();
        for site in &mut sites {
            let Some(RTLInst::Ireturn(returned)) = self.instructions.get(&site.node) else {
                continue;
            };
            if *returned != site.value {
                return None;
            }
            let types = self.return_types.get(&proof.function)?;
            let xtype = (types.len() == 1)
                .then(|| types.iter().next().copied())
                .flatten()?;
            let required_type = scalar_memory_use_type_from_xtype(xtype)?;
            if required_type.width() > proof.value_width {
                return None;
            }
            site.required_type = required_type;
        }
        let finalized = ScalarMemoryUsePlan {
            function: plan.function,
            definition_node: plan.definition_node,
            value: plan.value,
            transports: plan.transports.clone(),
            sites: Arc::new(sites),
        };
        self.plan_is_closed(proof, &finalized)
            .then_some(finalized)
    }

    pub(crate) fn plan_is_closed(
        &self,
        proof: &ScalarMemoryAccessProof,
        plan: &ScalarMemoryUsePlan,
    ) -> bool {
        if !proof.is_closed_v1() || !plan.is_closed_v1(proof) {
            return false;
        }
        let mut states = BTreeMap::new();
        let mut root_fallback =
            scalar_default_use_type(proof.extension, proof.chunk, proof.value_width);
        let mut root_fallback = match root_fallback.take() {
            Some(value) => value,
            None => return false,
        };
        let root_wide_fallback = match proof.result_chain {
            Some(ScalarMemoryResultChain::ZeroUpper32) => {
                let Some(encoded) = scalar_memory_use_type(
                    proof.encoded_destination_width.unwrap_or_default(),
                    proof.extension == ScalarMemoryExtension::SignExtend,
                ) else {
                    return false;
                };
                root_fallback = encoded;
                Some(ScalarMemoryUseType::Unsigned64)
            }
            Some(ScalarMemoryResultChain::Direct) => None,
            None => return false,
        };
        states.insert(plan.value, (root_fallback, root_wide_fallback));
        while states.len() <= plan.transports.len() {
            let before = states.len();
            for transport in plan.transports.iter() {
                if states.contains_key(&transport.output) {
                    continue;
                }
                let Some((fallback, wide)) = states.get(&transport.input).copied() else {
                    continue;
                };
                states.insert(
                    transport.output,
                    match transport.kind {
                        ScalarMemoryTransportKind::Move => (fallback, wide),
                        ScalarMemoryTransportKind::LaneCast(required) => (required, None),
                    },
                );
            }
            if states.len() == plan.transports.len().saturating_add(1) {
                break;
            }
            if states.len() == before {
                return false;
            }
        }
        let values: BTreeSet<_> = states.keys().copied().collect();
        if values.iter().any(|value| self.ambiguous_values.contains(value)) {
            return false;
        }

        let mut expected_definitions = BTreeMap::from([(plan.value, proof.selected_node)]);
        let mut expected_uses: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
        for transport in plan.transports.iter() {
            if expected_definitions
                .insert(transport.output, transport.node)
                .is_some()
                || !self.node_is_exactly_owned(transport.node, proof.function)
            {
                return false;
            }
            expected_uses
                .entry(transport.input)
                .or_default()
                .push(transport.node);
            let Some(instruction) = self.instructions.get(&transport.node) else {
                return false;
            };
            if scalar_memory_transport(proof, instruction, transport.input)
                != Ok(Some((transport.output, transport.kind)))
            {
                return false;
            }
        }
        for site in plan.sites.iter() {
            if !self.node_is_exactly_owned(site.node, proof.function) {
                return false;
            }
            expected_uses.entry(site.value).or_default().push(site.node);
            let Some(instruction) = self.instructions.get(&site.node) else {
                return false;
            };
            let Some((fallback, wide_fallback)) = states.get(&site.value).copied() else {
                return false;
            };
            let final_type = match instruction {
                RTLInst::Ireturn(returned) if *returned == site.value => {
                    let Some(types) = self.return_types.get(&proof.function) else {
                        return false;
                    };
                    let Some(xtype) = (types.len() == 1)
                        .then(|| types.iter().next().copied())
                        .flatten()
                    else {
                        return false;
                    };
                    scalar_memory_use_type_from_xtype(xtype)
                }
                _ => scalar_value_use_type(
                    instruction,
                    site.value,
                    fallback,
                    wide_fallback,
                ),
            };
            if final_type != Some(site.required_type) {
                return false;
            }
        }
        for value in values {
            if self.definitions.get(&value).map(Vec::as_slice)
                != expected_definitions.get(&value).map(std::slice::from_ref)
            {
                return false;
            }
            let mut actual = self.uses.get(&value).cloned().unwrap_or_default();
            let mut expected = expected_uses.remove(&value).unwrap_or_default();
            actual.sort_unstable();
            expected.sort_unstable();
            if actual != expected {
                return false;
            }
        }
        let ownership_node = if proof.synthetic_stack_origin {
            proof.origin_node
        } else {
            proof.selected_node
        };
        self.node_is_exactly_owned(ownership_node, proof.function)
    }
}

fn scalar_ltl_row_matches(
    row: &LTLInst,
    direction: ScalarMemoryDirection,
    chunk: MemoryChunk,
    addressing: &Addressing,
    mregs: &[Mreg],
    value: Mreg,
) -> bool {
    match (direction, row) {
        (
            ScalarMemoryDirection::Read,
            LTLInst::Lload(row_chunk, row_addressing, row_mregs, destination),
        ) => {
            *row_chunk == chunk
                && row_addressing == addressing
                && row_mregs.as_slice() == mregs
                && *destination == value
        }
        (
            ScalarMemoryDirection::Write,
            LTLInst::Lstore(row_chunk, row_addressing, row_mregs, source),
        ) => {
            *row_chunk == chunk
                && row_addressing == addressing
                && row_mregs.as_slice() == mregs
                && *source == value
        }
        _ => false,
    }
}

fn scalar_rtl_row_matches(
    row: &RTLInst,
    direction: ScalarMemoryDirection,
    chunk: MemoryChunk,
    addressing: &Addressing,
    args: &[RTLReg],
    value: RTLReg,
) -> bool {
    match (direction, row) {
        (
            ScalarMemoryDirection::Read,
            RTLInst::Iload(row_chunk, row_addressing, row_args, destination),
        ) => {
            *row_chunk == chunk
                && row_addressing == addressing
                && row_args.as_slice() == args
                && *destination == value
        }
        (
            ScalarMemoryDirection::Write,
            RTLInst::Istore(row_chunk, row_addressing, row_args, source),
        ) => {
            *row_chunk == chunk
                && row_addressing == addressing
                && row_args.as_slice() == args
                && *source == value
        }
        _ => false,
    }
}

/// AsmPass deliberately emits the opposite signedness for each byte/word
/// load/store so later type selection can choose either spelling. Admit that
/// one known transport companion only after immutable decoder evidence has
/// fixed opcode semantics and final RTL has fixed the selected transport.
/// Any duplicate, third, cross-width, or structurally different interpretation
/// remains an ambiguity and vetoes the proof.
fn scalar_exact_or_signedness_companion_rows_match<T>(
    rows: &[T],
    expected_chunk: MemoryChunk,
    width: usize,
    companion_required: bool,
    row_matches: impl Fn(&T, MemoryChunk) -> bool,
) -> bool {
    match rows.len() {
        1 if !companion_required => row_matches(&rows[0], expected_chunk),
        2 => {
            let Some(companion) = scalar_signedness_companion(expected_chunk, width) else {
                return false;
            };
            rows.iter()
                .filter(|row| row_matches(row, expected_chunk))
                .count()
                == 1
                && rows
                    .iter()
                    .filter(|row| row_matches(row, companion))
                    .count()
                    == 1
        }
        _ => false,
    }
}

fn scalar_ltl_rows_match(
    rows: &[LTLInst],
    direction: ScalarMemoryDirection,
    expected_chunk: MemoryChunk,
    width: usize,
    companion_required: bool,
    addressing: &Addressing,
    mregs: &[Mreg],
    value: Mreg,
) -> bool {
    scalar_exact_or_signedness_companion_rows_match(
        rows,
        expected_chunk,
        width,
        companion_required,
        |row, chunk| scalar_ltl_row_matches(row, direction, chunk, addressing, mregs, value),
    )
}

fn scalar_rtl_rows_match(
    rows: &[RTLInst],
    direction: ScalarMemoryDirection,
    expected_chunk: MemoryChunk,
    width: usize,
    companion_required: bool,
    addressing: &Addressing,
    args: &[RTLReg],
    value: RTLReg,
) -> bool {
    scalar_exact_or_signedness_companion_rows_match(
        rows,
        expected_chunk,
        width,
        companion_required,
        |row, chunk| scalar_rtl_row_matches(row, direction, chunk, addressing, args, value),
    )
}

/// Final RTL may select the surviving source of a plain two-address store
/// after the Dual fixed point has retained the store-local identity and the
/// eliminated producer's historical identities.  Admit that transport-only
/// multiplicity only when the complete fixed-point web proves every distinct
/// candidate is the same final carrier.  This is deliberately separate from
/// byte/word signedness companions: chunks, addressing, arguments, and the
/// final store source must remain exact. A successful match is carried only
/// in the private Stage-4 access vector and is never published to the legacy
/// scalar-lvalue pipeline.
fn scalar_plain_write_fixed_point_rows_match(
    rows: &[RTLInst],
    chunk: MemoryChunk,
    width: usize,
    addressing: &Addressing,
    args: &[RTLReg],
    value: RTLReg,
    store_xtl_rows: &[RTLReg],
    reaching_rows: &[RTLReg],
    canonical_values: &BTreeMap<RTLReg, RTLReg>,
) -> bool {
    const MAX_FIXED_POINT_STORE_IDENTITIES: usize = 4;

    if !matches!(width, 4 | 8)
        || rows.len() < 2
        || rows.len() > MAX_FIXED_POINT_STORE_IDENTITIES
        || rows.len() != store_xtl_rows.len()
        || reaching_rows != [value]
        || store_xtl_rows.windows(2).any(|pair| pair[0] == pair[1])
    {
        return false;
    }
    let mut candidate_sources = Vec::with_capacity(rows.len());
    for row in rows {
        let RTLInst::Istore(row_chunk, row_addressing, row_args, source) = row else {
            return false;
        };
        if *row_chunk != chunk || row_addressing != addressing || row_args.as_slice() != args {
            return false;
        }
        candidate_sources.push(*source);
    }
    candidate_sources.sort_unstable();
    candidate_sources.as_slice() == store_xtl_rows
        && candidate_sources.windows(2).all(|pair| pair[0] != pair[1])
        && candidate_sources
            .iter()
            .filter(|source| **source == value)
            .count()
            == 1
        && candidate_sources
            .iter()
            .all(|source| canonical_values.get(source) == Some(&value))
}

/// A later plain load through the same architectural base register may retain
/// both the final parameter identity and one historical fixed-point identity
/// in `rtl_inst_candidate`, even though final RTL and reaching-use select the
/// parameter exactly.  This narrow matcher is Stage-4 placement provenance
/// only: it is never published as a general scalar-lvalue candidate.  Admit a
/// base-only load when the complete bounded candidate/base-identity sets are
/// equal and every historical base canonicalizes to the exact final base.
fn scalar_plain_read_fixed_point_base_rows_match(
    rows: &[RTLInst],
    chunk: MemoryChunk,
    width: usize,
    addressing: &Addressing,
    args: &[RTLReg],
    value: RTLReg,
    base_xtl_rows: &[RTLReg],
    reaching_rows: &[RTLReg],
    canonical_values: &BTreeMap<RTLReg, RTLReg>,
) -> bool {
    const MAX_FIXED_POINT_BASE_IDENTITIES: usize = 4;

    let [final_base] = args else { return false };
    if !matches!(width, 4 | 8)
        || rows.len() < 2
        || rows.len() > MAX_FIXED_POINT_BASE_IDENTITIES
        || rows.len() != base_xtl_rows.len()
        || reaching_rows != [*final_base]
        || base_xtl_rows.windows(2).any(|pair| pair[0] == pair[1])
    {
        return false;
    }
    let mut candidate_bases = Vec::with_capacity(rows.len());
    for row in rows {
        let RTLInst::Iload(row_chunk, row_addressing, row_args, destination) = row else {
            return false;
        };
        let [base] = row_args.as_slice() else {
            return false;
        };
        if *row_chunk != chunk || row_addressing != addressing || *destination != value {
            return false;
        }
        candidate_bases.push(*base);
    }
    candidate_bases.sort_unstable();
    candidate_bases.as_slice() == base_xtl_rows
        && candidate_bases.windows(2).all(|pair| pair[0] != pair[1])
        && candidate_bases
            .iter()
            .filter(|base| **base == *final_base)
            .count()
            == 1
        && candidate_bases
            .iter()
            .all(|base| canonical_values.get(base) == Some(final_base))
}

fn decoded_scalar_addressing(
    base_name: &str,
    index_name: &str,
    scale: i64,
    displacement: i64,
    address_size: u8,
) -> Option<(Addressing, Vec<Mreg>, bool)> {
    if !matches!(address_size, 4 | 8) || matches!(base_name, "" | "NONE" | "RIP" | "EIP") {
        return None;
    }
    let base = Mreg::x86(base_name);
    if base.is_unknown() || x86_scalar_register_width(base_name) != Some(address_size as usize) {
        return None;
    }
    let index = if matches!(index_name, "" | "NONE") {
        None
    } else {
        let index = Mreg::x86(index_name);
        if index.is_unknown()
            || x86_scalar_register_width(index_name) != Some(address_size as usize)
        {
            return None;
        }
        Some(index)
    };
    if index.is_none() && scale != 1 {
        return None;
    }

    let stack_base = address_size == 8 && matches!(base, Mreg::SP | Mreg::BP);
    let (inner, args, indexed_stack) = match index {
        // A bare stack displacement has no surviving RTL base value from
        // which a raw C lvalue can be reconstructed.  Indexed stack accesses
        // are represented by RTL's exact LEA-origin split below.
        None if stack_base => return None,
        None => (Addressing::Aindexed(displacement), vec![base], false),
        Some(index) if scale == 1 => (
            Addressing::Aindexed2(displacement),
            vec![base, index],
            stack_base,
        ),
        Some(index) if matches!(scale, 2 | 4 | 8) => (
            Addressing::Aindexed2scaled(scale, displacement),
            vec![base, index],
            stack_base,
        ),
        Some(_) => return None,
    };
    if address_size == 4 {
        Some((Addressing::Aaddr32(Box::new(inner)), args, false))
    } else {
        Some((inner, args, indexed_stack))
    }
}

fn unique_equal<T: PartialEq>(rows: &[T]) -> Option<&T> {
    let first = rows.first()?;
    rows.iter().all(|row| row == first).then_some(first)
}

const SCALAR_ADDRESS_DAG_NODE_LIMIT: usize = 128;
const SCALAR_ADDRESS_CFG_NODE_LIMIT: usize = 4096;

/// Prove reachability without hiding an ownership-ambiguous or unowned CFG
/// intermediate.  Expansion stops at `target`, so ownership beyond the
/// consumer is irrelevant; every reachable node before it must be one of the
/// function's uniquely owned nodes.  `None` is an ambiguity or work-limit
/// failure, while `Some(false)` is a closed proof that no path exists.
fn graph_reaches_through_owned_region(
    succs: &HashMap<Node, Vec<Node>>,
    allowed: &BTreeSet<Node>,
    start: Node,
    target: Node,
    blocked: Option<Node>,
) -> Option<bool> {
    if blocked == Some(start) || !allowed.contains(&start) || !allowed.contains(&target) {
        return None;
    }
    let mut reached = false;
    let mut seen = HashSet::new();
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        if !seen.insert(node) {
            continue;
        }
        if seen.len() > SCALAR_ADDRESS_CFG_NODE_LIMIT || !allowed.contains(&node) {
            return None;
        }
        if node == target {
            reached = true;
            continue;
        }
        if let Some(nexts) = succs.get(&node) {
            for next in nexts {
                if blocked == Some(*next) {
                    continue;
                }
                if !allowed.contains(next) {
                    return None;
                }
                if seen.len().saturating_add(queue.len()) >= SCALAR_ADDRESS_CFG_NODE_LIMIT {
                    return None;
                }
                queue.push_back(*next);
            }
        }
    }
    Some(reached)
}

fn scalar_addressing_is_nonsymbolic(addressing: &Addressing) -> bool {
    match addressing {
        Addressing::Aindexed(_)
        | Addressing::Aindexed2(_)
        | Addressing::Ascaled(_, _)
        | Addressing::Aindexed2scaled(_, _) => true,
        Addressing::Aaddr32(inner) => scalar_addressing_is_nonsymbolic(inner),
        Addressing::Aglobal(..)
        | Addressing::Abased(..)
        | Addressing::Abasedscaled(..)
        | Addressing::Ainstack(_)
        | Addressing::Unknown => false,
    }
}

fn scalar_addressing_arg_count(addressing: &Addressing) -> Option<usize> {
    match addressing {
        Addressing::Aindexed(_) | Addressing::Ascaled(_, _) => Some(1),
        Addressing::Aindexed2(_) | Addressing::Aindexed2scaled(_, _) => Some(2),
        Addressing::Ainstack(_) => Some(0),
        Addressing::Aaddr32(inner) => scalar_addressing_arg_count(inner),
        Addressing::Aglobal(..)
        | Addressing::Abased(..)
        | Addressing::Abasedscaled(..)
        | Addressing::Unknown => None,
    }
}

fn scalar_address_definition_is_closed(inst: &RTLInst) -> bool {
    match inst {
        RTLInst::Iop(Operation::Omove, args, _) => args.len() == 1,
        RTLInst::Iop(Operation::Oadd | Operation::Oaddl, args, _) => args.len() == 2,
        RTLInst::Iop(
            Operation::Oaddimm(_)
            | Operation::Omulimm(_)
            | Operation::Oshlimm(_)
            | Operation::Oaddlimm(_)
            | Operation::Omullimm(_)
            | Operation::Oshllimm(_),
            args,
            _,
        ) => args.len() == 1,
        RTLInst::Iop(Operation::Olea(addressing) | Operation::Oleal(addressing), args, _) => {
            scalar_addressing_is_nonsymbolic(addressing)
                && scalar_addressing_arg_count(addressing) == Some(args.len())
        }
        _ => false,
    }
}

type ScalarAddressClosureCache = BTreeMap<(Address, RTLReg, Node), Option<BTreeSet<RTLReg>>>;

/// Return the final parameter leaves of one ordinary address-DAG value.  The
/// non-inline extension binds an interior value to one exact consumer, so
/// memoize per (function, value, consumer) rather than recursively rewalking
/// the same authenticated chain for every access.
fn scalar_address_value_param_leaves(
    function: Address,
    value: RTLReg,
    consumer: Node,
    entry: Node,
    function_nodes: &BTreeSet<Node>,
    definitions: &BTreeMap<RTLReg, BTreeSet<Node>>,
    final_rtl: &BTreeMap<Node, Vec<RTLInst>>,
    uses: &BTreeMap<RTLReg, Vec<Node>>,
    succs: &HashMap<Node, Vec<Node>>,
    params: &BTreeSet<(Address, RTLReg)>,
    inline_temps: &BTreeSet<RTLReg>,
    visiting: &mut BTreeSet<RTLReg>,
    cache: &mut ScalarAddressClosureCache,
    dag_budget: &mut usize,
) -> Option<BTreeSet<RTLReg>> {
    let cache_key = (function, value, consumer);
    if let Some(cached) = cache.get(&cache_key) {
        return cached.clone();
    }
    if *dag_budget == 0 {
        cache.insert(cache_key, None);
        return None;
    }
    *dag_budget -= 1;
    if !visiting.insert(value) {
        cache.insert(cache_key, None);
        return None;
    }

    let defs = definitions.get(&value).cloned().unwrap_or_default();
    if defs.iter().any(|node| {
        !function_nodes.contains(node)
            || final_rtl
                .get(node)
                .and_then(|rows| unique_equal(rows))
                .is_none()
    }) {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    }
    if params.contains(&(function, value)) {
        if !defs.is_empty() {
            visiting.remove(&value);
            cache.insert(cache_key, None);
            return None;
        }
        visiting.remove(&value);
        let leaves = BTreeSet::from([value]);
        cache.insert(cache_key, Some(leaves.clone()));
        return Some(leaves);
    }
    if defs.len() != 1 {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    }
    let def = *defs.iter().next().expect("one definition");
    if def & SCALAR_LVALUE_SYNTHETIC_MASK != 0 {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    }
    let Some(inst) = final_rtl.get(&def).and_then(|rows| unique_equal(rows)) else {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    };
    if !scalar_address_definition_is_closed(inst) || rtl_definition(inst) != Some(value) {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    }
    // `emit_inline_temp` remains the established canonical-structuring proof.
    // Scalar authentication additionally admits an address-only value whose
    // exact definition/sole DAG use prove the same substitution without the
    // liveness-at-use requirement. This local extension never changes global
    // inlining or canonical C source selection.
    if !inline_temps.contains(&value) {
        let exact_single_definition_row = final_rtl.get(&def).is_some_and(|rows| rows.len() == 1);
        let exact_sole_use = uses
            .get(&value)
            .is_some_and(|nodes| nodes.as_slice() == [consumer]);
        let entry_reaches_def = def == entry
            || graph_reaches_through_owned_region(succs, function_nodes, entry, def, None)
                == Some(true);
        let def_reaches_consumer = def == consumer
            || graph_reaches_through_owned_region(succs, function_nodes, def, consumer, None)
                == Some(true);
        let no_bypass = def == entry
            || graph_reaches_through_owned_region(
                succs,
                function_nodes,
                entry,
                consumer,
                Some(def),
            ) == Some(false);
        let dominates_consumer = entry_reaches_def && def_reaches_consumer && no_bypass;
        if !exact_single_definition_row || !exact_sole_use || !dominates_consumer {
            visiting.remove(&value);
            cache.insert(cache_key, None);
            return None;
        }
    }
    let uses_in_definition = inst_def_use(inst).1;
    if uses_in_definition.is_empty() {
        visiting.remove(&value);
        cache.insert(cache_key, None);
        return None;
    }
    let mut leaves = BTreeSet::new();
    for input in uses_in_definition {
        let Some(input_leaves) = scalar_address_value_param_leaves(
            function,
            input,
            def,
            entry,
            function_nodes,
            definitions,
            final_rtl,
            uses,
            succs,
            params,
            inline_temps,
            visiting,
            cache,
            dag_budget,
        ) else {
            visiting.remove(&value);
            cache.insert(cache_key, None);
            return None;
        };
        leaves.extend(input_leaves);
    }
    visiting.remove(&value);
    if leaves.is_empty() {
        cache.insert(cache_key, None);
        return None;
    }
    cache.insert(cache_key, Some(leaves.clone()));
    Some(leaves)
}

fn scalar_address_param_leaves(
    function: Address,
    selected_node: Node,
    roots: &[RTLReg],
    entry: Node,
    function_nodes: &BTreeSet<Node>,
    definitions: &BTreeMap<RTLReg, BTreeSet<Node>>,
    final_rtl: &BTreeMap<Node, Vec<RTLInst>>,
    uses: &BTreeMap<RTLReg, Vec<Node>>,
    succs: &HashMap<Node, Vec<Node>>,
    params: &BTreeSet<(Address, RTLReg)>,
    inline_temps: &BTreeSet<RTLReg>,
    stack_origin: Option<(Node, RTLReg)>,
    closure_cache: &mut ScalarAddressClosureCache,
) -> Option<Arc<Vec<RTLReg>>> {
    if !function_nodes.contains(&entry) || !function_nodes.contains(&selected_node) {
        return None;
    }
    let mut visiting = BTreeSet::new();
    let mut dag_budget = SCALAR_ADDRESS_DAG_NODE_LIMIT;
    let mut leaves = BTreeSet::new();
    let mut saw_stack_origin = false;
    for root in roots {
        if let Some((def, stack_value)) = stack_origin.filter(|(_, value)| value == root) {
            if saw_stack_origin {
                return None;
            }
            saw_stack_origin = true;
            let defs = definitions.get(root)?;
            let exact_stack_def = defs.len() == 1 && defs.contains(&def);
            let exact_stack_inst = final_rtl
                .get(&def)
                .and_then(|rows| unique_equal(rows))
                .is_some_and(|inst| {
                    matches!(
                        inst,
                        RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), args, destination)
                            if args.is_empty() && destination == root
                    )
                });
            let entry_reaches_def = def == entry
                || graph_reaches_through_owned_region(succs, function_nodes, entry, def, None)
                    == Some(true);
            let def_reaches_selected = def == selected_node
                || graph_reaches_through_owned_region(
                    succs,
                    function_nodes,
                    def,
                    selected_node,
                    None,
                ) == Some(true);
            let no_bypass = def == entry
                || graph_reaches_through_owned_region(
                    succs,
                    function_nodes,
                    entry,
                    selected_node,
                    Some(def),
                ) == Some(false);
            let dominates = entry_reaches_def && def_reaches_selected && no_bypass;
            if !exact_stack_def || !exact_stack_inst || !dominates {
                return None;
            }
            continue;
        }
        leaves.extend(scalar_address_value_param_leaves(
            function,
            *root,
            selected_node,
            entry,
            function_nodes,
            definitions,
            final_rtl,
            uses,
            succs,
            params,
            inline_temps,
            &mut visiting,
            closure_cache,
            &mut dag_budget,
        )?);
    }
    if stack_origin.is_some() != saw_stack_origin || leaves.is_empty() {
        return None;
    }
    Some(Arc::new(leaves.into_iter().collect()))
}

#[derive(Clone, Copy, Debug)]
struct ScalarInterval {
    start: u64,
    end: u64,
    payload: usize,
}

#[derive(Debug, Default)]
struct ScalarIntervalIndex {
    intervals: Vec<ScalarInterval>,
    prefix_max_end: Vec<u64>,
}

impl ScalarIntervalIndex {
    fn new(mut intervals: Vec<ScalarInterval>) -> Self {
        intervals.retain(|interval| interval.start < interval.end);
        intervals.sort_by_key(|interval| (interval.start, interval.end, interval.payload));
        let mut running_end = 0;
        let prefix_max_end = intervals
            .iter()
            .map(|interval| {
                running_end = running_end.max(interval.end);
                running_end
            })
            .collect();
        Self {
            intervals,
            prefix_max_end,
        }
    }

    fn overlap_bounds(&self, start: u64, end: u64) -> Option<(usize, usize)> {
        if start >= end {
            return None;
        }
        let upper = self
            .intervals
            .partition_point(|interval| interval.start < end);
        let lower = self.prefix_max_end[..upper].partition_point(|seen_end| *seen_end <= start);
        (lower < upper).then_some((lower, upper))
    }

    fn has_overlap(&self, start: u64, end: u64) -> bool {
        self.overlap_bounds(start, end)
            .is_some_and(|(lower, upper)| {
                self.intervals[lower..upper]
                    .iter()
                    .any(|interval| interval.end > start)
            })
    }

    fn unique_overlap_payload(&self, start: u64, end: u64) -> Option<usize> {
        let (lower, upper) = self.overlap_bounds(start, end)?;
        let mut matches = self.intervals[lower..upper]
            .iter()
            .filter(|interval| interval.end > start)
            .map(|interval| interval.payload);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    }
}

/// Immutable interval/identity index over the COFF map.  Building it once
/// keeps authentication linearithmic in the map plus the actually overlapping
/// relocation count, rather than rescanning every function/section/relocation
/// for every selected memory effect.
struct ScalarCoffIndex {
    functions: ScalarIntervalIndex,
    valid_function_section: Vec<Option<usize>>,
    mapped_relocations: ScalarIntervalIndex,
    original_relocations: BTreeMap<usize, ScalarIntervalIndex>,
    relocation_section_names: BTreeMap<usize, BTreeSet<String>>,
    relocations_well_formed: bool,
}

impl ScalarCoffIndex {
    fn new(map: &crate::decompile::disassembly::coff::CoffAddressMap) -> Self {
        let mut section_rows: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, section) in map.sections.iter().enumerate() {
            section_rows.entry(section.index).or_default().push(index);
        }
        let valid_function_section = map
            .functions
            .iter()
            .map(|function| {
                let section_index = section_rows
                    .get(&function.section_index)
                    .filter(|rows| rows.len() == 1)?[0];
                let section = &map.sections[section_index];
                let original_section_size = section
                    .original_offset_end
                    .checked_sub(section.original_offset_start)?;
                let mapped_section_size =
                    section.mapped_va_end.checked_sub(section.mapped_va_start)?;
                let original_function_delta = function
                    .section_offset
                    .checked_sub(section.original_offset_start)?;
                let mapped_function_delta =
                    function.mapped_entry.checked_sub(section.mapped_va_start)?;
                let mapped_span = function.mapped_end.checked_sub(function.mapped_entry)?;
                let original_function_end = function
                    .section_offset
                    .checked_add(function.original_size)?;
                (section.kind == "Text"
                    && original_section_size == mapped_section_size
                    && original_function_delta == mapped_function_delta
                    && mapped_span == function.manifold_size
                    && function.original_size == function.manifold_size
                    && function.mapped_end <= section.mapped_va_end
                    && original_function_end <= section.original_offset_end)
                    .then_some(section_index)
            })
            .collect();
        let functions = ScalarIntervalIndex::new(
            map.functions
                .iter()
                .enumerate()
                .map(|(payload, function)| ScalarInterval {
                    start: function.mapped_entry,
                    end: function.mapped_end,
                    payload,
                })
                .collect(),
        );

        let mut mapped_relocations = Vec::new();
        let mut original_relocations: BTreeMap<usize, Vec<ScalarInterval>> = BTreeMap::new();
        let mut relocation_section_names: BTreeMap<usize, BTreeSet<String>> = BTreeMap::new();
        let mut relocations_well_formed = true;
        for (payload, relocation) in map.relocations.iter().enumerate() {
            let width = u64::from(relocation.width_bits).div_ceil(8).max(1);
            let Some(mapped_end) = relocation.mapped_field_va.checked_add(width) else {
                relocations_well_formed = false;
                continue;
            };
            let Some(original_end) = relocation.section_offset.checked_add(width) else {
                relocations_well_formed = false;
                continue;
            };
            mapped_relocations.push(ScalarInterval {
                start: relocation.mapped_field_va,
                end: mapped_end,
                payload,
            });
            original_relocations
                .entry(relocation.section_index)
                .or_default()
                .push(ScalarInterval {
                    start: relocation.section_offset,
                    end: original_end,
                    payload,
                });
            relocation_section_names
                .entry(relocation.section_index)
                .or_default()
                .insert(relocation.section_name.clone());
        }

        Self {
            functions,
            valid_function_section,
            mapped_relocations: ScalarIntervalIndex::new(mapped_relocations),
            original_relocations: original_relocations
                .into_iter()
                .map(|(section, intervals)| (section, ScalarIntervalIndex::new(intervals)))
                .collect(),
            relocation_section_names,
            relocations_well_formed,
        }
    }
}

/// Authenticate scalar MOV-family accesses only after RTL optimization has
/// selected and rewritten the surviving memory effects.  Every rejected or
/// ambiguous input produces no row; the ordinary decompilation path is never
/// removed or rewritten here.
fn materialize_authenticated_scalar_memory_accesses(
    db: &mut DecompileDB,
) -> Vec<(Node, ScalarMemoryAccessProof)> {
    let Some(coff_map) = db.coff_address_map.as_ref() else {
        db.rel_set(
            "authenticated_scalar_memory_access",
            ascent::boxcar::Vec::<(Node, ScalarMemoryAccessProof)>::new(),
        );
        db.rel_set(
            "authenticated_scalar_memory_use_plan",
            ascent::boxcar::Vec::<(Node, ScalarMemoryUsePlan)>::new(),
        );
        return Vec::new();
    };
    if coff_map.schema != "manifold.coff-address-map.v1"
        || coff_map.loader_id != "amd64-coff-image-v1"
        || coff_map.architecture != "x86_64-pc-windows-msvc"
    {
        db.rel_set(
            "authenticated_scalar_memory_access",
            ascent::boxcar::Vec::<(Node, ScalarMemoryAccessProof)>::new(),
        );
        db.rel_set(
            "authenticated_scalar_memory_use_plan",
            ascent::boxcar::Vec::<(Node, ScalarMemoryUsePlan)>::new(),
        );
        return Vec::new();
    }
    let coff_index = ScalarCoffIndex::new(coff_map);
    if !coff_index.relocations_well_formed {
        db.rel_set(
            "authenticated_scalar_memory_access",
            ascent::boxcar::Vec::<(Node, ScalarMemoryAccessProof)>::new(),
        );
        db.rel_set(
            "authenticated_scalar_memory_use_plan",
            ascent::boxcar::Vec::<(Node, ScalarMemoryUsePlan)>::new(),
        );
        return Vec::new();
    }

    let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        owners.entry(*node).or_default().insert(*function);
    }
    let mut instructions: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for (node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) in
        db.rel_iter::<(
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
        )>("instruction")
    {
        instructions.entry(*node).or_default().push((
            *size, *prefix, *mnemonic, *op1, *op2, *op3, *op4, *metadata0, *metadata1,
        ));
    }
    let mut raw_instructions: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for (node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) in
        db.rel_iter::<(
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
        )>("unrefinedinstruction")
    {
        raw_instructions.entry(*node).or_default().push((
            *size, *prefix, *mnemonic, *op1, *op2, *op3, *op4, *metadata0, *metadata1,
        ));
    }
    let mut indirects: BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>> = BTreeMap::new();
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
    let mut registers: BTreeMap<Symbol, BTreeSet<&'static str>> = BTreeMap::new();
    for (operand, register) in db.rel_iter::<(Symbol, &'static str)>("op_register") {
        registers.entry(*operand).or_default().insert(*register);
    }
    let immediate_operands: BTreeSet<Symbol> = db
        .rel_iter::<(Symbol, i64, usize)>("op_immediate")
        .map(|row| row.0)
        .collect();
    let mut reads: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_read_operand") {
        reads.entry(*node).or_default().insert(*operand);
    }
    let mut writes: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_write_operand") {
        writes.entry(*node).or_default().insert(*operand);
    }
    let mut decoded_register_defs: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_def") {
        decoded_register_defs
            .entry(*node)
            .or_default()
            .insert(*register);
    }
    let mut decoded_register_uses: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_use") {
        decoded_register_uses
            .entry(*node)
            .or_default()
            .insert(*register);
    }
    let mut address_sizes: BTreeMap<Node, BTreeSet<u8>> = BTreeMap::new();
    for (node, size) in db.rel_iter::<(Node, u8)>("instruction_address_size") {
        address_sizes.entry(*node).or_default().insert(*size);
    }
    let mut rtl_mregs: BTreeMap<(Node, RTLReg), BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reg_rtl") {
        rtl_mregs
            .entry((*node, *value))
            .or_default()
            .insert(*register);
    }
    let mut reg_xtl_rows: BTreeMap<(Node, Mreg), Vec<RTLReg>> = BTreeMap::new();
    for (node, register, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl") {
        reg_xtl_rows
            .entry((*node, *register))
            .or_default()
            .push(*value);
    }
    for rows in reg_xtl_rows.values_mut() {
        rows.sort_unstable();
    }
    // `xtl_canonical` is a Dual-lattice projection and deliberately retains
    // reflexive/historical rows.  Its final representative is the minimum per
    // identity, matching the optimizer's fixed-point semantics.
    let mut canonical_values: BTreeMap<RTLReg, RTLReg> = BTreeMap::new();
    for (value, canonical) in db.rel_iter::<(RTLReg, RTLReg)>("xtl_canonical") {
        canonical_values
            .entry(*value)
            .and_modify(|current| *current = (*current).min(*canonical))
            .or_insert(*canonical);
    }
    let mut reaching_values: BTreeMap<(Node, Mreg), Vec<RTLReg>> = BTreeMap::new();
    for (node, register, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reaching_use_rtl") {
        reaching_values
            .entry((*node, *register))
            .or_default()
            .push(*value);
    }
    for rows in reaching_values.values_mut() {
        rows.sort_unstable();
    }
    let mut ltl: BTreeMap<Node, Vec<LTLInst>> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, LTLInst)>("ltl_inst") {
        ltl.entry(*node).or_default().push(inst.clone());
    }
    let mut final_rtl: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        final_rtl.entry(*node).or_default().push(inst.clone());
    }
    let mut definitions: BTreeMap<RTLReg, BTreeSet<Node>> = BTreeMap::new();
    let mut uses: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
    for (node, rows) in &final_rtl {
        for inst in rows {
            let (definition, inst_uses) = inst_def_use(inst);
            if let Some(value) = definition {
                definitions.entry(value).or_default().insert(*node);
            }
            for value in inst_uses {
                uses.entry(value).or_default().push(*node);
            }
        }
    }
    // A memory-indirect call's address inputs are carried outside final RTL
    // after the target load is folded.  Count them exactly as the optimizer's
    // liveness model does, so such a hidden call use cannot masquerade as the
    // address DAG's sole scalar-memory consumer.
    for (node, _temp, _chunk, _addressing, args) in
        db.rel_iter::<(Node, RTLReg, MemoryChunk, Addressing, Arc<Vec<RTLReg>>)>(
            "call_through_memory_load",
        )
    {
        for value in args.iter() {
            uses.entry(*value).or_default().push(*node);
        }
    }
    for nodes in uses.values_mut() {
        nodes.sort_unstable();
    }
    let mut function_nodes: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for (node, functions) in &owners {
        if functions.len() == 1 {
            function_nodes
                .entry(*functions.iter().next().expect("one owner"))
                .or_default()
                .insert(*node);
        }
    }
    let mut input_rtl: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate") {
        input_rtl.entry(*node).or_default().push(inst.clone());
    }
    let mut succs: HashMap<Node, Vec<Node>> = HashMap::new();
    for (source, target) in db.rel_iter::<(Node, Node)>("rtl_succ") {
        succs.entry(*source).or_default().push(*target);
    }
    for targets in succs.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }
    let params: BTreeSet<(Address, RTLReg)> = db
        .rel_iter::<(Address, RTLReg)>("emit_function_param_candidate")
        .copied()
        .collect();
    let inline_temps: BTreeSet<RTLReg> =
        db.rel_iter::<RTLReg>("emit_inline_temp").copied().collect();
    let mut value_type_candidates: BTreeMap<RTLReg, BTreeSet<XType>> = BTreeMap::new();
    for (value, xtype) in db.rel_iter::<(RTLReg, XType)>("emit_var_type_candidate") {
        value_type_candidates
            .entry(*value)
            .or_default()
            .insert(*xtype);
    }
    let entries: BTreeMap<Address, BTreeSet<Node>> = {
        let mut grouped = BTreeMap::new();
        for (function, _, entry) in db.rel_iter::<(Address, Symbol, Node)>("emit_function") {
            grouped
                .entry(*function)
                .or_insert_with(BTreeSet::new)
                .insert(*entry);
        }
        grouped
    };

    let mut proofs = Vec::new();
    let mut stage4_private_accesses = Vec::new();
    let mut use_plans = Vec::new();
    let mut address_closure_cache = ScalarAddressClosureCache::new();
    for (selected_node, final_rows) in &final_rtl {
        let Some(final_inst) = unique_equal(final_rows) else {
            continue;
        };
        let (final_chunk, final_addressing, final_args, value, direction) = match final_inst {
            RTLInst::Iload(chunk, addressing, args, destination) => (
                *chunk,
                addressing,
                args,
                *destination,
                ScalarMemoryDirection::Read,
            ),
            RTLInst::Istore(chunk, addressing, args, source) => (
                *chunk,
                addressing,
                args,
                *source,
                ScalarMemoryDirection::Write,
            ),
            _ => continue,
        };
        let origin = *selected_node & !SCALAR_LVALUE_SYNTHETIC_MASK;
        let synthetic = origin != *selected_node;
        if synthetic && *selected_node != (origin | (1u64 << 62)) {
            continue;
        }
        let Some(function) = owners.get(&origin).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };
        if owners
            .get(selected_node)
            .map_or(true, |rows| rows.len() != 1 || !rows.contains(&function))
        {
            continue;
        }
        let Some(origin_end) = origin.checked_add(1) else {
            continue;
        };
        let Some(mapped_function_index) = coff_index
            .functions
            .unique_overlap_payload(origin, origin_end)
        else {
            continue;
        };
        let mapped_function = &coff_map.functions[mapped_function_index];
        if mapped_function.mapped_entry != function {
            continue;
        }
        let Some(section_index) = coff_index.valid_function_section[mapped_function_index] else {
            continue;
        };
        let section = &coff_map.sections[section_index];
        let Some(instruction) = instructions
            .get(&origin)
            .and_then(|rows| unique_equal(rows))
        else {
            continue;
        };
        if raw_instructions
            .get(&origin)
            .and_then(|rows| unique_equal(rows))
            != Some(instruction)
            || instruction.1 != ""
            || exact_raw_operand_count(instruction) != Some(2)
        {
            continue;
        }
        let instruction_end = match origin.checked_add(instruction.0 as u64) {
            Some(end) => end,
            None => continue,
        };
        if instruction_end > mapped_function.mapped_end {
            continue;
        }
        let Some(instruction_delta) = origin.checked_sub(mapped_function.mapped_entry) else {
            continue;
        };
        let Some(original_instruction_start) = mapped_function
            .section_offset
            .checked_add(instruction_delta)
        else {
            continue;
        };
        let Some(original_instruction_end) =
            original_instruction_start.checked_add(instruction.0 as u64)
        else {
            continue;
        };
        let mapped_relocation_overlap = coff_index
            .mapped_relocations
            .has_overlap(origin, instruction_end);
        let original_relocation_overlap = coff_index
            .original_relocations
            .get(&mapped_function.section_index)
            .is_some_and(|intervals| {
                intervals.has_overlap(original_instruction_start, original_instruction_end)
            });
        let wrong_relocation_section_name = coff_index
            .relocation_section_names
            .get(&mapped_function.section_index)
            .is_some_and(|names| names.iter().any(|name| name != &section.name));
        if mapped_relocation_overlap || original_relocation_overlap || wrong_relocation_section_name
        {
            continue;
        }

        let (register_operand, memory_operand) = match direction {
            // The instruction relation is source-first/destination-second,
            // matching AsmPass's pmov normalization.
            ScalarMemoryDirection::Read => (instruction.4, instruction.3),
            ScalarMemoryDirection::Write => (instruction.3, instruction.4),
        };
        if indirects.contains_key(&register_operand)
            || immediate_operands.contains(&register_operand)
            || registers.contains_key(&memory_operand)
            || immediate_operands.contains(&memory_operand)
        {
            continue;
        }
        let Some(register_name) = registers.get(&register_operand).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };
        if x86_high8_register(register_name) {
            continue;
        }
        let Some(value_width) = x86_scalar_register_width(register_name) else {
            continue;
        };
        let expected_value_mreg = Mreg::x86(register_name);
        if expected_value_mreg.is_unknown() {
            continue;
        }
        let Some((segment, base_name, index_name, scale, displacement, width)) =
            indirects.get(&memory_operand).and_then(|rows| {
                (rows.len() == 1)
                    .then(|| rows.iter().next().copied())
                    .flatten()
            })
        else {
            continue;
        };
        if segment != "NONE" {
            continue;
        }
        let Some((decoded_extension, canonical_chunk)) =
            scalar_mov_semantics(instruction.2, direction, width, value_width)
        else {
            continue;
        };
        let Some(companion_required) = scalar_transport_companion_required(
            canonical_chunk,
            final_chunk,
            direction,
            decoded_extension,
            width,
            value_width,
        ) else {
            continue;
        };
        // v1 does not replay copy propagation inside an address. The final
        // row must occur exactly once among the input candidates; only
        // AsmPass's sole same-width signedness companion may coexist with it.
        // A plain 32/64-bit write additionally admits the complete, bounded
        // fixed-point source-identity set when every identity canonically
        // equals the exact selected store carrier.
        let Some(input_rows) = input_rtl.get(selected_node) else {
            continue;
        };
        let ordinary_input_match = scalar_rtl_rows_match(
            input_rows,
            direction,
            final_chunk,
            width,
            companion_required,
            final_addressing,
            final_args,
            value,
        );
        let fixed_point_write_match = direction == ScalarMemoryDirection::Write
            && decoded_extension == ScalarMemoryExtension::Plain
            && !companion_required
            && !synthetic
            && scalar_plain_write_fixed_point_rows_match(
                input_rows,
                final_chunk,
                width,
                final_addressing,
                final_args,
                value,
                reg_xtl_rows
                    .get(&(*selected_node, expected_value_mreg))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                reaching_values
                    .get(&(*selected_node, expected_value_mreg))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                &canonical_values,
            );
        // LTL and the proof must retain the exact selected transport chunk;
        // the contract above authorizes raw-semantic equivalence, not a global
        // chunk normalization.
        let expected_chunk = final_chunk;
        if reads.get(&origin).cloned().unwrap_or_default()
            != if direction == ScalarMemoryDirection::Read {
                BTreeSet::from([memory_operand])
            } else {
                BTreeSet::new()
            }
            || writes.get(&origin).cloned().unwrap_or_default()
                != if direction == ScalarMemoryDirection::Write {
                    BTreeSet::from([memory_operand])
                } else {
                    BTreeSet::new()
                }
        {
            continue;
        }
        let Some(address_size) = address_sizes.get(&origin).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };
        let Some((expected_addressing, expected_mregs, indexed_stack)) =
            decoded_scalar_addressing(base_name, index_name, scale, displacement, address_size)
        else {
            continue;
        };
        let fixed_point_read_match = direction == ScalarMemoryDirection::Read
            && decoded_extension == ScalarMemoryExtension::Plain
            && !companion_required
            && !synthetic
            && expected_mregs.len() == 1
            && scalar_plain_read_fixed_point_base_rows_match(
                input_rows,
                final_chunk,
                width,
                final_addressing,
                final_args,
                value,
                reg_xtl_rows
                    .get(&(*selected_node, expected_mregs[0]))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                reaching_values
                    .get(&(*selected_node, expected_mregs[0]))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                &canonical_values,
            );
        if !ordinary_input_match && !fixed_point_write_match && !fixed_point_read_match {
            continue;
        }
        let mut expected_register_uses: BTreeSet<Mreg> = expected_mregs.iter().copied().collect();
        let expected_register_defs = match direction {
            ScalarMemoryDirection::Read => BTreeSet::from([expected_value_mreg]),
            ScalarMemoryDirection::Write => {
                expected_register_uses.insert(expected_value_mreg);
                BTreeSet::new()
            }
        };
        if decoded_register_defs
            .get(&origin)
            .cloned()
            .unwrap_or_default()
            != expected_register_defs
            || decoded_register_uses
                .get(&origin)
                .cloned()
                .unwrap_or_default()
                != expected_register_uses
        {
            continue;
        }
        let Some(ltl_rows) = ltl.get(&origin) else {
            continue;
        };
        if !scalar_ltl_rows_match(
            ltl_rows,
            direction,
            expected_chunk,
            width,
            companion_required,
            &expected_addressing,
            &expected_mregs,
            expected_value_mreg,
        ) || synthetic != indexed_stack
        {
            continue;
        }

        let (base_value, index_value) = if synthetic {
            let Some(origin_rtl) = final_rtl.get(&origin).and_then(|rows| unique_equal(rows))
            else {
                continue;
            };
            let Some(base_value) = (match origin_rtl {
                RTLInst::Iop(Operation::Olea(Addressing::Ainstack(ofs)), args, destination)
                    if *ofs == displacement && args.is_empty() =>
                {
                    Some(*destination)
                }
                _ => None,
            }) else {
                continue;
            };
            let expected_synthetic_addressing = match expected_addressing {
                Addressing::Aindexed2(_) => Addressing::Aindexed2(0),
                Addressing::Aindexed2scaled(decoded_scale, _) => {
                    Addressing::Aindexed2scaled(decoded_scale, 0)
                }
                _ => continue,
            };
            if *final_addressing != expected_synthetic_addressing
                || final_args.len() != 2
                || final_args[0] != base_value
                || !succs
                    .get(&origin)
                    .is_some_and(|targets| targets.as_slice() == [*selected_node])
            {
                continue;
            }
            (Some(base_value), Some(final_args[1]))
        } else {
            if *final_addressing != expected_addressing || final_args.len() != expected_mregs.len()
            {
                continue;
            }
            (final_args.first().copied(), final_args.get(1).copied())
        };
        let (
            extension,
            proof_value_width,
            downstream_value_width,
            result_chain,
            low_lane_fallback,
        ) =
            match (direction, decoded_extension) {
            // MOVSX/MOVZX define the architectural result width and
            // signedness independently of the source chunk.  The early type
            // pass often still carries that narrow chunk type for `value`;
            // the final pre-Csh gate removes only that exact lowering
            // artifact, publishes the opcode-defined result type, and rejects
            // every other late incompatible type.
            (ScalarMemoryDirection::Read, extension)
                if extension != ScalarMemoryExtension::Plain =>
            {
                if value_width == 4 {
                    // A MOVSX/MOVZX write to EAX always produces a 32-bit
                    // encoded value and architecturally clears RAX's upper
                    // half.  The pre-Type relation still describes the
                    // byte/word source chunk, so it cannot decide whether a
                    // later use observes 32 or 64 bits.  Seal the architectural
                    // result here; the closed use plan records every machine
                    // consumer, and Cshminor may narrow only an exact final
                    // return site after signature reconciliation.
                    (
                        extension,
                        8,
                        Some(8),
                        Some(ScalarMemoryResultChain::ZeroUpper32),
                        None,
                    )
                } else {
                    (
                        extension,
                        value_width,
                        Some(value_width),
                        Some(ScalarMemoryResultChain::Direct),
                        None,
                    )
                }
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain)
                if matches!(width, 1 | 2) =>
            {
                let Some(types) = value_type_candidates
                    .get(&value)
                    .filter(|types| !types.is_empty())
                else {
                    continue;
                };
                let signedness: Option<BTreeSet<bool>> = types
                    .iter()
                    .copied()
                    .map(|xtype| {
                        scalar_memory_use_type_from_xtype(xtype)
                            .map(ScalarMemoryUseType::signed)
                    })
                    .collect();
                let Some(signedness) = signedness else {
                    continue;
                };
                let Some(signed) = (signedness.len() == 1)
                    .then(|| signedness.iter().next().copied())
                    .flatten()
                else {
                    continue;
                };
                let selected_signed = matches!(
                    final_chunk,
                    MemoryChunk::MInt8Signed | MemoryChunk::MInt16Signed
                );
                if selected_signed != signed {
                    continue;
                }
                let Some(fallback) = scalar_memory_use_type(width, signed) else {
                    continue;
                };
                (
                    ScalarMemoryExtension::Plain,
                    width,
                    Some(width),
                    Some(ScalarMemoryResultChain::Direct),
                    Some(fallback),
                )
            }
            (ScalarMemoryDirection::Read, ScalarMemoryExtension::Plain) => {
                let Some(downstream_width) = scalar_unique_integral_xtype_width(
                    value_type_candidates.get(&value),
                ) else {
                    continue;
                };
                match (width, value_width, downstream_width) {
                    (4, 4, 4) => (
                        ScalarMemoryExtension::Plain,
                        4,
                        Some(4),
                        Some(ScalarMemoryResultChain::Direct),
                        None,
                    ),
                    (4, 4, 8) => (
                        ScalarMemoryExtension::ImplicitZeroExtend,
                        8,
                        Some(8),
                        Some(ScalarMemoryResultChain::ZeroUpper32),
                        None,
                    ),
                    (_, decoded_width, final_width) if decoded_width == final_width => (
                        ScalarMemoryExtension::Plain,
                        decoded_width,
                        Some(final_width),
                        Some(ScalarMemoryResultChain::Direct),
                        None,
                    ),
                    _ => continue,
                }
            }
            // Keep the guarded extension arm fail-closed if the enum grows or
            // its predicate changes; guards do not make a match exhaustive.
            (ScalarMemoryDirection::Read, _) => continue,
            (ScalarMemoryDirection::Write, ScalarMemoryExtension::Plain) => {
                (
                    ScalarMemoryExtension::Plain,
                    value_width,
                    None,
                    None,
                    None,
                )
            }
            (ScalarMemoryDirection::Write, _) => continue,
        };
        let roots: Vec<_> = base_value.into_iter().chain(index_value).collect();
        let Some(entry) = entries.get(&function).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };
        let Some(owned_nodes) = function_nodes.get(&function) else {
            continue;
        };
        let Some(address_param_leaves) = scalar_address_param_leaves(
            function,
            *selected_node,
            &roots,
            entry,
            owned_nodes,
            &definitions,
            &final_rtl,
            &uses,
            &succs,
            &params,
            &inline_temps,
            synthetic.then_some((origin, base_value.expect("synthetic stack base"))),
            &mut address_closure_cache,
        ) else {
            continue;
        };

        let proof = ScalarMemoryAccessProof {
            function,
            origin_node: origin,
            selected_node: *selected_node,
            operand: memory_operand,
            direction,
            extension,
            encoded_destination_width: (direction == ScalarMemoryDirection::Read)
                .then_some(value_width),
            result_chain,
            address_size,
            base_register: Mreg::x86(base_name),
            index_register: (!matches!(index_name, "" | "NONE")).then(|| Mreg::x86(index_name)),
            scale,
            displacement,
            width,
            value_width: proof_value_width,
            downstream_value_width,
            chunk: final_chunk,
            base_value,
            index_value,
            value,
            address_param_leaves,
            synthetic_stack_origin: synthetic,
            exact_scaled_index: address_size == 8
                && index_value.is_some()
                && (displacement == 0 || synthetic)
                && scale == width as i64,
        };
        if proof.is_closed_v1() {
            // Fixed-point alias exceptions exist solely to authenticate a
            // Stage-4 eliminated-mutation placement or terminal. Publishing
            // either exception through the legacy scalar relation would grow
            // the frozen v3 typed-lvalue/field portfolio merely by enabling
            // Stage-4 support.
            if fixed_point_read_match || fixed_point_write_match {
                stage4_private_accesses.push((proof.selected_node, proof));
                continue;
            }
            let use_plan = (proof.direction == ScalarMemoryDirection::Read).then(|| {
                scalar_memory_use_plan(
                    &proof,
                    entry,
                    owned_nodes,
                    &definitions,
                    &uses,
                    &final_rtl,
                    &succs,
                    &instructions,
                    &raw_instructions,
                    &registers,
                    &indirects,
                    &rtl_mregs,
                    &decoded_register_defs,
                    &decoded_register_uses,
                    &ltl,
                    low_lane_fallback,
                )
            });
            let use_plan = use_plan.flatten();
            let stage3_only = proof.direction == ScalarMemoryDirection::Read
                && (proof.extension != ScalarMemoryExtension::Plain
                    || matches!(proof.width, 1 | 2));
            if stage3_only && use_plan.is_none() {
                continue;
            }
            if let Some(plan) = use_plan {
                use_plans.push((proof.selected_node, plan));
            }
            proofs.push(proof);
        }
    }

    proofs.sort();
    proofs.dedup();
    db.rel_set(
        "authenticated_scalar_memory_access",
        proofs
            .into_iter()
            .map(|proof| (proof.selected_node, proof))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    use_plans.sort();
    use_plans.dedup();
    db.rel_set(
        "authenticated_scalar_memory_use_plan",
        use_plans
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    stage4_private_accesses.sort_unstable();
    stage4_private_accesses.dedup();
    stage4_private_accesses
}

fn clear_authenticated_stage4_sources(db: &mut DecompileDB) {
    db.rel_set(
        "authenticated_stage4_source",
        ascent::boxcar::Vec::<(Node, Stage4SourceProof)>::new(),
    );
    db.rel_set(
        "authenticated_stage4_use_plan",
        ascent::boxcar::Vec::<(Node, Stage4UsePlan)>::new(),
    );
}

fn stage4_exact_register(
    operand: Symbol,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    indirects: &BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>>,
    immediates: &BTreeSet<Symbol>,
) -> Option<&'static str> {
    if indirects.contains_key(&operand) || immediates.contains(&operand) {
        return None;
    }
    let rows = registers.get(&operand)?;
    (rows.len() == 1)
        .then(|| rows.iter().next().copied())
        .flatten()
}

fn stage4_exact_mreg_binding(
    rtl_mregs: &BTreeMap<(Node, RTLReg), BTreeSet<Mreg>>,
    node: Node,
    value: RTLReg,
    expected: Mreg,
) -> bool {
    rtl_mregs.get(&(node, value)) == Some(&BTreeSet::from([expected]))
}

fn stage4_selected_with_optional_companion<T: PartialEq>(
    rows: &[T],
    selected: &T,
    companion: Option<&T>,
) -> bool {
    match rows {
        [only] => only == selected,
        [first, second] => companion.is_some_and(|companion| {
            selected != companion
                && ((first == selected && second == companion)
                    || (second == selected && first == companion))
        }),
        _ => false,
    }
}

fn stage4_rmw_input_candidates_match(
    rows: &[RTLInst],
    authenticated_operations: &[Operation],
    args: &Args,
    carrier: RTLReg,
    selected_result: RTLReg,
) -> bool {
    if carrier == selected_result
        || authenticated_operations.is_empty()
        || authenticated_operations.len() > 2
        || authenticated_operations
            .iter()
            .enumerate()
            .any(|(index, operation)| authenticated_operations[..index].contains(operation))
    {
        return false;
    }
    let results = [carrier, selected_result];
    let expected_len = authenticated_operations.len() * results.len();
    rows.len() == expected_len
        && authenticated_operations.iter().all(|expected_operation| {
            results.into_iter().all(|expected_result| {
                rows.iter()
                    .filter(|row| {
                        matches!(
                            row,
                            RTLInst::Iop(row_operation, row_args, row_result)
                                if row_operation == expected_operation
                                    && row_args == args
                                    && *row_result == expected_result
                        )
                    })
                    .count()
                    == 1
            })
        })
}

fn stage4_addition_operations(width: usize) -> Option<(Operation, Operation)> {
    match width {
        4 => Some((
            Operation::Oadd,
            Operation::Olea(Addressing::Aindexed2(0)),
        )),
        8 => Some((
            Operation::Oaddl,
            Operation::Oleal(Addressing::Aindexed2(0)),
        )),
        _ => None,
    }
}

fn stage4_and_or_width_companion(operation: &Operation) -> Option<Operation> {
    match operation {
        Operation::Oand => Some(Operation::Oandl),
        Operation::Oandl => Some(Operation::Oand),
        Operation::Oor => Some(Operation::Oorl),
        Operation::Oorl => Some(Operation::Oor),
        Operation::Oandimm(value) => Some(Operation::Oandlimm(*value)),
        Operation::Oandlimm(value) => Some(Operation::Oandimm(*value)),
        Operation::Oorimm(value) => Some(Operation::Oorlimm(*value)),
        Operation::Oorlimm(value) => Some(Operation::Oorimm(*value)),
        _ => None,
    }
}

fn stage4_rmw_kind_and_operation(
    mnemonic: &str,
    width: usize,
    immediate: Option<i64>,
) -> Option<(Stage4SourceKind, Operation)> {
    Some(match (mnemonic, width, immediate) {
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
        ("ADD", 4, Some(value)) => (Stage4SourceKind::Add, Operation::Oaddimm(value)),
        ("ADD", 8, Some(value)) => (Stage4SourceKind::Add, Operation::Oaddlimm(value)),
        ("SUB", 4, Some(value)) => {
            (Stage4SourceKind::Sub, Operation::Oaddimm(value.checked_neg()?))
        }
        ("SUB", 8, Some(value)) => {
            (Stage4SourceKind::Sub, Operation::Oaddlimm(value.checked_neg()?))
        }
        ("IMUL", 4, Some(value)) => (Stage4SourceKind::Mul, Operation::Omulimm(value)),
        ("IMUL", 8, Some(value)) => (Stage4SourceKind::Mul, Operation::Omullimm(value)),
        ("AND", 4, Some(value)) => (Stage4SourceKind::And, Operation::Oandimm(value)),
        ("AND", 8, Some(value)) => (Stage4SourceKind::And, Operation::Oandlimm(value)),
        ("OR", 4, Some(value)) => (Stage4SourceKind::Or, Operation::Oorimm(value)),
        ("OR", 8, Some(value)) => (Stage4SourceKind::Or, Operation::Oorlimm(value)),
        ("XOR", 4, Some(value)) => (Stage4SourceKind::Xor, Operation::Oxorimm(value)),
        ("XOR", 8, Some(value)) => (Stage4SourceKind::Xor, Operation::Oxorlimm(value)),
        _ => return None,
    })
}

/// Prove that an eliminated arithmetic root's flags cannot be observed before
/// one exact terminal value use.  The ordinary flags relation covers only a
/// subset of Jcc producers and does not retain every CMOV/SETcc-style use, so
/// this private proof walks the exact selected pre-opt CFG itself.  V1 accepts
/// only a unique linear corridor containing flag-preserving moves/address
/// formation, or ending at one instruction that definitely overwrites the
/// arithmetic flags.  Any unknown instruction, prefix, branch, merge, cycle,
/// consumer, or work-limit event is fail closed.
fn stage4_exact_imul_flags_clobber(
    node: Node,
    instruction: &Cr8RawInstruction,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediate_rows: &BTreeMap<Symbol, BTreeSet<(i64, usize)>>,
    indirects: &BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
) -> bool {
    if instruction.1 != "" || instruction.2 != "IMUL" {
        return false;
    }
    let Some(count @ (2 | 3)) = exact_raw_operand_count(instruction) else {
        return false;
    };
    let operands = [instruction.3, instruction.4, instruction.5, instruction.6];
    let mut raw_registers = BTreeSet::new();
    let mut register_width = None;
    for operand in &operands[..count] {
        if indirects.contains_key(operand) {
            return false;
        }
        let register = registers.get(operand).and_then(|rows| {
            (rows.len() == 1).then(|| rows.iter().next().copied()).flatten()
        });
        let immediate = immediate_rows
            .get(operand)
            .is_some_and(|rows| rows.len() == 1);
        if register.is_some() == immediate {
            return false;
        }
        if let Some(name) = register {
            let Some(width @ (4 | 8)) = x86_scalar_register_width(name) else {
                return false;
            };
            if register_width.replace(width).is_some_and(|old| old != width) {
                return false;
            }
            let mreg = Mreg::x86(name);
            if mreg.is_unknown() || !raw_registers.insert(mreg) {
                // Repeated source==destination is the closed compound form and
                // is represented once in the architectural effect set.
                if !raw_registers.contains(&mreg) {
                    return false;
                }
            }
        }
    }
    // The loader keeps its AT&T-like normalized order: two-operand IMUL is
    // [source, destination], and the immediate form is
    // [source, destination, immediate].  Do not reuse Intel display order at
    // this raw/refined authentication boundary.
    if registers
        .get(&operands[0])
        .is_none_or(|rows| rows.len() != 1)
        || registers
            .get(&operands[1])
            .is_none_or(|rows| rows.len() != 1)
        || (count == 3
            && (immediate_rows
                .get(&operands[2])
                .is_none_or(|rows| {
                    rows.len() != 1 || rows.iter().any(|(_, encoded_width)| *encoded_width != 0)
                })
                || registers.contains_key(&operands[2])))
    {
        return false;
    }
    let Some(destination_name) = registers.get(&operands[1]).and_then(|rows| {
        (rows.len() == 1).then(|| rows.iter().next().copied()).flatten()
    }) else {
        return false;
    };
    let destination = Mreg::x86(destination_name);
    !destination.is_unknown()
        && raw_registers.contains(&destination)
        && decoded_defs.get(&node) == Some(&BTreeSet::from([destination]))
        && decoded_uses.get(&node) == Some(&raw_registers)
}

fn stage4_flags_are_dead_before_terminal(
    root: Node,
    terminal: Node,
    terminal_use: &Stage4TerminalUse,
    function: Address,
    coff_map: &crate::decompile::disassembly::coff::CoffAddressMap,
    coff_index: &ScalarCoffIndex,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    raw_instructions: &BTreeMap<Node, Vec<Cr8RawInstruction>>,
    registers: &BTreeMap<Symbol, BTreeSet<&'static str>>,
    immediate_rows: &BTreeMap<Symbol, BTreeSet<(i64, usize)>>,
    indirects: &BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>>,
    decoded_defs: &BTreeMap<Node, BTreeSet<Mreg>>,
    decoded_uses: &BTreeMap<Node, BTreeSet<Mreg>>,
    succs: &BTreeMap<Node, Vec<Node>>,
    preds: &BTreeMap<Node, Vec<Node>>,
) -> bool {
    let mut current = root;
    let mut seen = BTreeSet::from([root]);
    let mut passed_store_terminal = false;
    for _ in 0..SCALAR_ADDRESS_CFG_NODE_LIMIT {
        let [next] = succs.get(&current).map(Vec::as_slice).unwrap_or(&[]) else {
            return false;
        };
        if preds.get(next).map(Vec::as_slice) != Some(&[current][..])
            || !seen.insert(*next)
        {
            return false;
        }
        let [instruction] = instructions.get(next).map(Vec::as_slice).unwrap_or(&[]) else {
            return false;
        };
        let [raw] = raw_instructions.get(next).map(Vec::as_slice).unwrap_or(&[]) else {
            return false;
        };
        if instruction != raw
            || instruction.1 != ""
            || stage4_instruction_function(coff_map, coff_index, owners, *next, instruction)
                != Some(function)
        {
            return false;
        }
        if *next == terminal {
            match terminal_use {
                Stage4TerminalUse::Return => {
                    return instruction.2 == "RET"
                        && exact_raw_operand_count(instruction) == Some(0);
                }
                Stage4TerminalUse::Store { .. } => {
                    if instruction.2 != "MOV" || exact_raw_operand_count(instruction) != Some(2) {
                        return false;
                    }
                    passed_store_terminal = true;
                    current = *next;
                    continue;
                }
            }
        }
        if !passed_store_terminal {
            // V1 requires the authenticated return/store to be the root's
            // immediate selected successor.  No unmodelled instruction may be
            // skipped on the way to that terminal value use.
            return false;
        }
        let mnemonic = instruction.2;
        if mnemonic == "RET" {
            return exact_raw_operand_count(instruction) == Some(0);
        }
        let definitely_clobbers = matches!(
            mnemonic,
            "ADD" | "SUB" | "CMP" | "TEST" | "AND" | "OR" | "XOR" | "NEG"
        ) || stage4_exact_imul_flags_clobber(
            *next,
            instruction,
            registers,
            immediate_rows,
            indirects,
            decoded_defs,
            decoded_uses,
        );
        if definitely_clobbers {
            return true;
        }
        let preserves_without_observing = matches!(
            mnemonic,
            "MOV" | "MOVSX" | "MOVZX" | "MOVSXD" | "LEA" | "NOP"
        );
        if !preserves_without_observing {
            return false;
        }
        current = *next;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn stage4_instruction_function(
    coff_map: &crate::decompile::disassembly::coff::CoffAddressMap,
    coff_index: &ScalarCoffIndex,
    owners: &BTreeMap<Node, BTreeSet<Address>>,
    node: Node,
    instruction: &Cr8RawInstruction,
) -> Option<Address> {
    let function = owners
        .get(&node)
        .filter(|rows| rows.len() == 1)?
        .iter()
        .next()
        .copied()?;
    let instruction_end = node.checked_add(instruction.0 as u64)?;
    let mapped_function_index = coff_index
        .functions
        .unique_overlap_payload(node, instruction_end)?;
    let mapped_function = &coff_map.functions[mapped_function_index];
    if mapped_function.mapped_entry != function || instruction_end > mapped_function.mapped_end {
        return None;
    }
    let section_index = coff_index.valid_function_section[mapped_function_index]?;
    let section = &coff_map.sections[section_index];
    let instruction_delta = node.checked_sub(mapped_function.mapped_entry)?;
    let original_instruction_start = mapped_function
        .section_offset
        .checked_add(instruction_delta)?;
    let original_instruction_end = original_instruction_start.checked_add(instruction.0 as u64)?;
    if coff_index
        .mapped_relocations
        .has_overlap(node, instruction_end)
        || coff_index
            .original_relocations
            .get(&mapped_function.section_index)
            .is_some_and(|rows| {
                rows.has_overlap(original_instruction_start, original_instruction_end)
            })
        || coff_index
            .relocation_section_names
            .get(&mapped_function.section_index)
            .is_some_and(|names| names.iter().any(|name| name != &section.name))
    {
        return None;
    }
    Some(function)
}

#[allow(clippy::too_many_arguments)]
fn stage4_use_plan(
    proof: &Stage4SourceProof,
    placement_node: Node,
    entry: Node,
    function_nodes: &BTreeSet<Node>,
    definitions: &BTreeMap<RTLReg, BTreeSet<Node>>,
    uses: &BTreeMap<RTLReg, Vec<Node>>,
    final_rtl: &BTreeMap<Node, Vec<RTLInst>>,
    succs: &HashMap<Node, Vec<Node>>,
) -> Option<Stage4UsePlan> {
    let expected_definition = match proof.root_boundary {
        Stage4RootBoundary::FinalRtlDefinition => proof.selected_node,
        Stage4RootBoundary::EliminatedMutation => placement_node,
    };
    if definitions.get(&proof.value) != Some(&BTreeSet::from([expected_definition])) {
        return None;
    }
    let mut pending = VecDeque::from([(proof.value, expected_definition)]);
    let mut queued = BTreeSet::from([proof.value]);
    let mut visited = BTreeSet::new();
    let mut use_count = 0usize;
    let mut transports = Vec::new();
    let mut sites = Vec::new();

    while let Some((value, definition)) = pending.pop_front() {
        if !visited.insert(value)
            || visited.len() > 65
            || definitions.get(&value) != Some(&BTreeSet::from([definition]))
        {
            return None;
        }
        let use_nodes = uses.get(&value)?;
        if use_nodes.is_empty()
            || use_nodes.len() > 64
            || use_nodes.windows(2).any(|pair| pair[0] == pair[1])
        {
            return None;
        }
        use_count = use_count.checked_add(use_nodes.len())?;
        if use_count > 64 {
            return None;
        }
        for &node in use_nodes {
            if node == definition || !function_nodes.contains(&node) {
                return None;
            }
            let definition_reaches_use = graph_reaches_through_owned_region(
                succs,
                function_nodes,
                definition,
                node,
                None,
            ) == Some(true);
            let no_definition_bypass = definition == entry
                || graph_reaches_through_owned_region(
                    succs,
                    function_nodes,
                    entry,
                    node,
                    Some(definition),
                ) == Some(false);
            if !definition_reaches_use || !no_definition_bypass {
                return None;
            }
            let [instruction] = final_rtl.get(&node)?.as_slice() else {
                return None;
            };
            if let RTLInst::Iop(Operation::Omove, args, output) = instruction {
                if args.as_slice() != [value]
                    || *output == value
                    || !queued.insert(*output)
                    || definitions.get(output) != Some(&BTreeSet::from([node]))
                {
                    return None;
                }
                transports.push(Stage4Transport {
                    node,
                    input: value,
                    output: *output,
                });
                pending.push_back((*output, node));
            } else {
                if inst_def_use(instruction)
                    .1
                    .into_iter()
                    .filter(|candidate| *candidate == value)
                    .count()
                    != 1
                {
                    return None;
                }
                sites.push(Stage4UseSite { node, value });
            }
        }
    }
    transports.sort();
    sites.sort();
    let (placement_load, terminal_use) = match proof.root_boundary {
        Stage4RootBoundary::FinalRtlDefinition => (None, None),
        Stage4RootBoundary::EliminatedMutation => {
            if !transports.is_empty() || sites.len() != 1 {
                return None;
            }
            let [RTLInst::Iload(chunk, addressing, args, value)] =
                final_rtl.get(&placement_node)?.as_slice()
            else {
                return None;
            };
            if *value != proof.value {
                return None;
            }
            let terminal = match final_rtl.get(&sites[0].node)?.as_slice() {
                [RTLInst::Ireturn(value)] if *value == sites[0].value => {
                    Stage4TerminalUse::Return
                }
                [RTLInst::Istore(chunk, addressing, args, value)]
                    if *value == sites[0].value
                        && match chunk {
                            MemoryChunk::MInt32 | MemoryChunk::MAny32 => proof.width == 4,
                            MemoryChunk::MInt64 | MemoryChunk::MAny64 => proof.width == 8,
                            _ => false,
                        } =>
                {
                    Stage4TerminalUse::Store {
                        chunk: *chunk,
                        addressing: addressing.clone(),
                        args: args.clone(),
                    }
                }
                _ => return None,
            };
            (
                Some(Stage4PlacementLoad {
                    chunk: *chunk,
                    addressing: addressing.clone(),
                    args: args.clone(),
                }),
                Some(terminal),
            )
        }
    };
    let plan = Stage4UsePlan {
        function: proof.function,
        definition_node: proof.selected_node,
        placement_node,
        value: proof.value,
        placement_load,
        terminal_use,
        transports: Arc::new(transports),
        sites: Arc::new(sites),
    };
    plan.is_closed_v1(proof).then_some(plan)
}

/// Authenticate source-shape alternatives from immutable x86-64 COFF bytes
/// through decoder effects, LTL, and the uniquely selected final RTL row.
/// This relation is provenance only: it never changes the canonical RTL or
/// candidate pool.  Later private Clight views must revalidate the same proof
/// and complete use forest before assembling a bounded alternative.
#[cfg(test)]
fn materialize_authenticated_stage4_sources(db: &mut DecompileDB) {
    materialize_authenticated_stage4_sources_with_private_accesses(db, &[]);
}

fn materialize_authenticated_stage4_sources_with_private_accesses(
    db: &mut DecompileDB,
    private_accesses: &[(Node, ScalarMemoryAccessProof)],
) {
    let Some(coff_map) = db.coff_address_map.as_ref() else {
        clear_authenticated_stage4_sources(db);
        return;
    };
    if coff_map.schema != "manifold.coff-address-map.v1"
        || coff_map.loader_id != "amd64-coff-image-v1"
        || coff_map.architecture != "x86_64-pc-windows-msvc"
    {
        clear_authenticated_stage4_sources(db);
        return;
    }
    let coff_index = ScalarCoffIndex::new(coff_map);
    if !coff_index.relocations_well_formed {
        clear_authenticated_stage4_sources(db);
        return;
    }

    let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        owners.entry(*node).or_default().insert(*function);
    }
    let mut instructions: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for row in db.rel_iter::<(
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
    )>("instruction") {
        instructions.entry(row.0).or_default().push((
            row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8, row.9,
        ));
    }
    let mut raw_instructions: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for row in db.rel_iter::<(
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
    )>("unrefinedinstruction") {
        raw_instructions.entry(row.0).or_default().push((
            row.1, row.2, row.3, row.4, row.5, row.6, row.7, row.8, row.9,
        ));
    }
    let mut indirects: BTreeMap<Symbol, BTreeSet<ScalarIndirectOperand>> = BTreeMap::new();
    for row in db.rel_iter::<(
        Symbol,
        &'static str,
        &'static str,
        &'static str,
        i64,
        i64,
        usize,
    )>("op_indirect") {
        indirects
            .entry(row.0)
            .or_default()
            .insert((row.1, row.2, row.3, row.4, row.5, row.6));
    }
    let mut registers: BTreeMap<Symbol, BTreeSet<&'static str>> = BTreeMap::new();
    for (operand, register) in db.rel_iter::<(Symbol, &'static str)>("op_register") {
        registers.entry(*operand).or_default().insert(*register);
    }
    let mut immediate_rows: BTreeMap<Symbol, BTreeSet<(i64, usize)>> = BTreeMap::new();
    for (operand, value, width) in db.rel_iter::<(Symbol, i64, usize)>("op_immediate") {
        immediate_rows
            .entry(*operand)
            .or_default()
            .insert((*value, *width));
    }
    let immediates: BTreeSet<Symbol> = immediate_rows.keys().copied().collect();
    let mut reads: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_read_operand") {
        reads.entry(*node).or_default().insert(*operand);
    }
    let mut writes: BTreeMap<Node, BTreeSet<Symbol>> = BTreeMap::new();
    for (node, operand) in db.rel_iter::<(Node, Symbol)>("decoded_memory_write_operand") {
        writes.entry(*node).or_default().insert(*operand);
    }
    let mut decoded_defs: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_def") {
        decoded_defs.entry(*node).or_default().insert(*register);
    }
    let mut decoded_uses: BTreeMap<Node, BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register) in db.rel_iter::<(Node, Mreg)>("decoded_reg_use") {
        decoded_uses.entry(*node).or_default().insert(*register);
    }
    let mut address_sizes: BTreeMap<Node, BTreeSet<u8>> = BTreeMap::new();
    for (node, size) in db.rel_iter::<(Node, u8)>("instruction_address_size") {
        address_sizes.entry(*node).or_default().insert(*size);
    }
    let mut rtl_mregs: BTreeMap<(Node, RTLReg), BTreeSet<Mreg>> = BTreeMap::new();
    for (node, register, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reg_rtl") {
        rtl_mregs
            .entry((*node, *value))
            .or_default()
            .insert(*register);
    }
    // Retain row multiplicity for every cross-pass architectural-identity
    // relation.  An exact row plus a duplicate or competing interpretation is
    // ambiguity and must not be normalized away before the eliminated-RMW
    // proof consumes it.
    let mut reg_defs_reaching_use: BTreeMap<(Node, Mreg), Vec<Node>> = BTreeMap::new();
    let mut reg_uses_reached_from_def: BTreeMap<(Node, Mreg), Vec<Node>> = BTreeMap::new();
    for (definition, register, use_node) in
        db.rel_iter::<(Node, Mreg, Node)>("reg_def_used")
    {
        reg_defs_reaching_use
            .entry((*use_node, *register))
            .or_default()
            .push(*definition);
        reg_uses_reached_from_def
            .entry((*definition, *register))
            .or_default()
            .push(*use_node);
    }
    for rows in reg_defs_reaching_use.values_mut() {
        rows.sort_unstable();
    }
    for rows in reg_uses_reached_from_def.values_mut() {
        rows.sort_unstable();
    }
    let mut reg_xtl_rows: BTreeMap<(Node, Mreg), Vec<RTLReg>> = BTreeMap::new();
    for (node, register, value) in db.rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl") {
        reg_xtl_rows
            .entry((*node, *register))
            .or_default()
            .push(*value);
    }
    for rows in reg_xtl_rows.values_mut() {
        rows.sort_unstable();
    }
    let mut definitions_at_node: BTreeMap<Node, Vec<RTLReg>> = BTreeMap::new();
    for (node, value) in db.rel_iter::<(Node, RTLReg)>("is_def") {
        definitions_at_node.entry(*node).or_default().push(*value);
    }
    for rows in definitions_at_node.values_mut() {
        rows.sort_unstable();
    }
    // `xtl_canonical` is the projection of a Dual lattice and may retain
    // historical representatives.  Its semantic value is the minimum final
    // representative per id, not singleton row cardinality.
    let mut canonical_values: BTreeMap<RTLReg, RTLReg> = BTreeMap::new();
    for (value, canonical) in db.rel_iter::<(RTLReg, RTLReg)>("xtl_canonical") {
        canonical_values
            .entry(*value)
            .and_modify(|current| *current = (*current).min(*canonical))
            .or_insert(*canonical);
    }
    let mut reaching_values: BTreeMap<(Node, Mreg), Vec<RTLReg>> = BTreeMap::new();
    for (node, register, value) in
        db.rel_iter::<(Node, Mreg, RTLReg)>("reaching_use_rtl")
    {
        reaching_values
            .entry((*node, *register))
            .or_default()
            .push(*value);
    }
    for rows in reaching_values.values_mut() {
        rows.sort_unstable();
    }
    let mut scalar_accesses: BTreeMap<Node, Vec<ScalarMemoryAccessProof>> = BTreeMap::new();
    for (node, proof) in db
        .rel_iter::<(Node, ScalarMemoryAccessProof)>("authenticated_scalar_memory_access")
    {
        scalar_accesses.entry(*node).or_default().push(proof.clone());
    }
    for (node, proof) in private_accesses {
        scalar_accesses.entry(*node).or_default().push(proof.clone());
    }
    for rows in scalar_accesses.values_mut() {
        rows.sort_unstable();
    }
    let mut ltl: BTreeMap<Node, Vec<LTLInst>> = BTreeMap::new();
    for (node, instruction) in db.rel_iter::<(Node, LTLInst)>("ltl_inst") {
        ltl.entry(*node).or_default().push(instruction.clone());
    }
    let mut final_rtl: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, instruction) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        final_rtl.entry(*node).or_default().push(instruction.clone());
    }
    let mut input_rtl: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, instruction) in db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate") {
        input_rtl.entry(*node).or_default().push(instruction.clone());
    }
    let mut preopt_selected: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, instruction) in
        db.rel_iter::<(Node, RTLInst)>("stage4_preopt_selected_rtl")
    {
        preopt_selected
            .entry(*node)
            .or_default()
            .push(instruction.clone());
    }
    let optimizer_copy_substitutions: BTreeSet<(RTLReg, RTLReg)> = db
        .rel_iter::<(RTLReg, RTLReg)>("stage4_optimizer_copy_substitution")
        .copied()
        .collect();
    let optimizer_eliminated_nodes: BTreeSet<Node> = db
        .rel_iter::<(Node,)>("stage4_optimizer_eliminated_node")
        .map(|(node,)| *node)
        .collect();
    let mut preopt_definitions: BTreeMap<RTLReg, BTreeSet<Node>> = BTreeMap::new();
    let mut preopt_uses: BTreeMap<RTLReg, BTreeSet<Node>> = BTreeMap::new();
    for (node, rows) in &preopt_selected {
        for instruction in rows {
            let (definition, instruction_uses) = inst_def_use(instruction);
            if let Some(value) = definition {
                preopt_definitions.entry(value).or_default().insert(*node);
            }
            for value in instruction_uses {
                preopt_uses.entry(value).or_default().insert(*node);
            }
        }
    }
    let mut definitions: BTreeMap<RTLReg, BTreeSet<Node>> = BTreeMap::new();
    let mut uses: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
    for (node, rows) in &final_rtl {
        for instruction in rows {
            let (definition, instruction_uses) = inst_def_use(instruction);
            if let Some(value) = definition {
                definitions.entry(value).or_default().insert(*node);
            }
            for value in instruction_uses {
                uses.entry(value).or_default().push(*node);
            }
        }
    }
    for (node, _temp, _chunk, _addressing, args) in
        db.rel_iter::<(Node, RTLReg, MemoryChunk, Addressing, Arc<Vec<RTLReg>>)>(
            "call_through_memory_load",
        )
    {
        for value in args.iter() {
            uses.entry(*value).or_default().push(*node);
        }
    }
    for rows in uses.values_mut() {
        rows.sort_unstable();
    }
    let mut function_nodes: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for (node, functions) in &owners {
        if functions.len() == 1 {
            function_nodes
                .entry(*functions.iter().next().expect("one owner"))
                .or_default()
                .insert(*node);
        }
    }
    let mut succs: HashMap<Node, Vec<Node>> = HashMap::new();
    for (source, target) in db.rel_iter::<(Node, Node)>("rtl_succ") {
        succs.entry(*source).or_default().push(*target);
    }
    for targets in succs.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }
    let mut final_succs_exact: BTreeMap<Node, Vec<Node>> = BTreeMap::new();
    let mut final_preds_exact: BTreeMap<Node, Vec<Node>> = BTreeMap::new();
    for (source, target) in db.rel_iter::<(Node, Node)>("rtl_succ") {
        final_succs_exact.entry(*source).or_default().push(*target);
        final_preds_exact.entry(*target).or_default().push(*source);
    }
    for targets in final_succs_exact.values_mut() {
        targets.sort_unstable();
    }
    for sources in final_preds_exact.values_mut() {
        sources.sort_unstable();
    }
    let mut preopt_succs: HashMap<Node, Vec<Node>> = HashMap::new();
    let mut preopt_succs_exact: BTreeMap<Node, Vec<Node>> = BTreeMap::new();
    let mut preopt_preds_exact: BTreeMap<Node, Vec<Node>> = BTreeMap::new();
    for (source, target) in db.rel_iter::<(Node, Node)>("stage4_preopt_selected_succ") {
        preopt_succs.entry(*source).or_default().push(*target);
        preopt_succs_exact.entry(*source).or_default().push(*target);
        preopt_preds_exact.entry(*target).or_default().push(*source);
    }
    for targets in preopt_succs.values_mut() {
        targets.sort_unstable();
        targets.dedup();
    }
    for targets in preopt_succs_exact.values_mut() {
        targets.sort_unstable();
    }
    for sources in preopt_preds_exact.values_mut() {
        sources.sort_unstable();
    }
    let params: BTreeSet<(Address, RTLReg)> = db
        .rel_iter::<(Address, RTLReg)>("emit_function_param_candidate")
        .copied()
        .collect();
    let inline_temps: BTreeSet<RTLReg> =
        db.rel_iter::<RTLReg>("emit_inline_temp").copied().collect();
    let mut entries: BTreeMap<Address, BTreeSet<Node>> = BTreeMap::new();
    for (function, _, entry) in db.rel_iter::<(Address, Symbol, Node)>("emit_function") {
        entries.entry(*function).or_default().insert(*entry);
    }
    let stage4_flag_producers: BTreeSet<Node> = db
        .rel_iter::<(Node, Node, &'static str)>("flags_and_jump_pair")
        .map(|(producer, _, _)| *producer)
        .collect();

    let mut proofs = Vec::new();
    let mut plans = Vec::new();
    let mut address_closure_cache = ScalarAddressClosureCache::new();
    for (&node, rows) in &final_rtl {
        let [final_instruction] = rows.as_slice() else {
            continue;
        };
        let [instruction] = instructions.get(&node).map(Vec::as_slice).unwrap_or(&[]) else {
            continue;
        };
        let [raw_instruction] = raw_instructions
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if raw_instruction != instruction
            || instruction.1 != ""
            || exact_raw_operand_count(instruction) != Some(2)
            || address_sizes.get(&node) != Some(&BTreeSet::from([8]))
            || !writes.get(&node).is_none_or(BTreeSet::is_empty)
        {
            continue;
        }
        let Some(function) =
            stage4_instruction_function(coff_map, &coff_index, &owners, node, instruction)
        else {
            continue;
        };
        let Some(owned_nodes) = function_nodes.get(&function) else {
            continue;
        };
        let Some(entry) = entries.get(&function).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };

        let proof = match (instruction.2, final_instruction) {
            ("LEA", RTLInst::Iop(Operation::Olea(addressing), args, value)) => {
                let source_operand = instruction.3;
                let destination_operand = instruction.4;
                // Capstone marks LEA's address operand readable even though
                // the instruction has no architectural memory read.  Bind
                // that exact decoder-owned pseudo-read instead of treating
                // LEA like a load or accepting an unaccounted effect row.
                if reads.get(&node) != Some(&BTreeSet::from([source_operand]))
                    || registers.contains_key(&source_operand)
                    || immediates.contains(&source_operand)
                    || indirects.contains_key(&destination_operand)
                    || immediates.contains(&destination_operand)
                {
                    continue;
                }
                let Some(destination_name) = stage4_exact_register(
                    destination_operand,
                    &registers,
                    &indirects,
                    &immediates,
                ) else {
                    continue;
                };
                let Some((segment, base_name, index_name, scale, displacement, operand_width)) =
                    indirects.get(&source_operand).and_then(|rows| {
                        (rows.len() == 1)
                            .then(|| rows.iter().next().copied())
                            .flatten()
                    })
                else {
                    continue;
                };
                let Some(destination_width @ (4 | 8)) =
                    x86_scalar_register_width(destination_name)
                else {
                    continue;
                };
                if segment != "NONE" || operand_width != destination_width {
                    continue;
                }
                let destination = Mreg::x86(destination_name);
                if destination.is_unknown() {
                    continue;
                }
                let base = if matches!(base_name, "" | "NONE") {
                    None
                } else {
                    let register = Mreg::x86(base_name);
                    if register.is_unknown()
                        || matches!(register, Mreg::SP | Mreg::BP)
                        || x86_scalar_register_width(base_name) != Some(8)
                    {
                        continue;
                    }
                    Some(register)
                };
                let index = if matches!(index_name, "" | "NONE") {
                    None
                } else {
                    let register = Mreg::x86(index_name);
                    if register.is_unknown()
                        || matches!(register, Mreg::SP | Mreg::BP)
                        || x86_scalar_register_width(index_name) != Some(8)
                    {
                        continue;
                    }
                    Some(register)
                };
                let (expected_addressing, expected_mregs) = match (base, index, scale) {
                    (Some(base), None, 1) => {
                        (Addressing::Aindexed(displacement), vec![base])
                    }
                    (Some(base), Some(index), 1) => {
                        (Addressing::Aindexed2(displacement), vec![base, index])
                    }
                    (Some(base), Some(index), 2 | 4 | 8) => (
                        Addressing::Aindexed2scaled(scale, displacement),
                        vec![base, index],
                    ),
                    (None, Some(index), 1 | 2 | 4 | 8) => {
                        (Addressing::Ascaled(scale, displacement), vec![index])
                    }
                    _ => continue,
                };
                if *addressing != expected_addressing || args.len() != expected_mregs.len() {
                    continue;
                }
                let expected_defs = BTreeSet::from([destination]);
                let expected_uses: BTreeSet<_> = expected_mregs.iter().copied().collect();
                if decoded_defs.get(&node) != Some(&expected_defs)
                    || decoded_uses.get(&node) != Some(&expected_uses)
                {
                    continue;
                }
                let expected_ltl = LTLInst::Lop(
                    Operation::Olea(expected_addressing.clone()),
                    Arc::new(expected_mregs.clone()),
                    destination,
                );
                if !stage4_selected_with_optional_companion(
                    ltl.get(&node).map(Vec::as_slice).unwrap_or(&[]),
                    &expected_ltl,
                    None,
                )
                    || !stage4_selected_with_optional_companion(
                        input_rtl.get(&node).map(Vec::as_slice).unwrap_or(&[]),
                        final_instruction,
                        None,
                    )
                    || !stage4_exact_mreg_binding(&rtl_mregs, node, *value, destination)
                    || args.iter().zip(&expected_mregs).any(|(argument, register)| {
                        !stage4_exact_mreg_binding(&rtl_mregs, node, *argument, *register)
                    })
                {
                    continue;
                }
                let Some(address_param_leaves) = scalar_address_param_leaves(
                    function,
                    node,
                    args,
                    entry,
                    owned_nodes,
                    &definitions,
                    &final_rtl,
                    &uses,
                    &succs,
                    &params,
                    &inline_temps,
                    None,
                    &mut address_closure_cache,
                ) else {
                    continue;
                };
                Stage4SourceProof {
                    function,
                    origin_node: node,
                    selected_node: node,
                    kind: Stage4SourceKind::AffineAddress,
                    root_boundary: Stage4RootBoundary::FinalRtlDefinition,
                    width: destination_width,
                    operation: Operation::Olea(expected_addressing),
                    args: args.clone(),
                    source_immediate: None,
                    value: *value,
                    selected_result: *value,
                    address_param_leaves,
                }
            }
            ("XOR" | "SUB", RTLInst::Iop(operation, args, value))
                if args.is_empty()
                    && matches!(operation, Operation::Ointconst(0) | Operation::Olongconst(0)) =>
            {
                if !reads.get(&node).is_none_or(BTreeSet::is_empty) {
                    continue;
                }
                let Some(source_name) = stage4_exact_register(
                    instruction.3,
                    &registers,
                    &indirects,
                    &immediates,
                ) else {
                    continue;
                };
                let Some(destination_name) = stage4_exact_register(
                    instruction.4,
                    &registers,
                    &indirects,
                    &immediates,
                ) else {
                    continue;
                };
                let width = match (
                    x86_scalar_register_width(source_name),
                    x86_scalar_register_width(destination_name),
                ) {
                    (Some(width @ (4 | 8)), Some(other)) if width == other => width,
                    _ => continue,
                };
                let register = Mreg::x86(source_name);
                let expected_uses = if instruction.2 == "XOR" {
                    BTreeSet::new()
                } else {
                    BTreeSet::from([register])
                };
                if register.is_unknown()
                    || Mreg::x86(destination_name) != register
                    || *operation
                        != if width == 8 {
                            Operation::Olongconst(0)
                        } else {
                            Operation::Ointconst(0)
                        }
                    || decoded_defs.get(&node) != Some(&BTreeSet::from([register]))
                    || decoded_uses.get(&node).cloned().unwrap_or_default() != expected_uses
                {
                    continue;
                }
                // The loader and LTL retain the exact destructive self-op.
                // `self_zero_rewrite` changes only the selected final RTL row
                // to a constant, so no constant LTL/input companion exists.
                let raw_operation = match (instruction.2, width) {
                    ("XOR", 4) => Operation::Oxor,
                    ("XOR", 8) => Operation::Oxorl,
                    ("SUB", 4) => Operation::Osub,
                    ("SUB", 8) => Operation::Osubl,
                    _ => continue,
                };
                let selected_ltl = LTLInst::Lop(
                    raw_operation.clone(),
                    Arc::new(vec![register, register]),
                    register,
                );
                let selected_input = RTLInst::Iop(
                    raw_operation,
                    Arc::new(vec![*value, *value]),
                    *value,
                );
                if !stage4_selected_with_optional_companion(
                    ltl.get(&node).map(Vec::as_slice).unwrap_or(&[]),
                    &selected_ltl,
                    None,
                ) || !stage4_selected_with_optional_companion(
                    input_rtl.get(&node).map(Vec::as_slice).unwrap_or(&[]),
                    &selected_input,
                    None,
                ) || !stage4_exact_mreg_binding(&rtl_mregs, node, *value, register)
                {
                    continue;
                }
                Stage4SourceProof {
                    function,
                    origin_node: node,
                    selected_node: node,
                    kind: Stage4SourceKind::Zeroing,
                    root_boundary: Stage4RootBoundary::FinalRtlDefinition,
                    width,
                    operation: operation.clone(),
                    args: Arc::new(Vec::new()),
                    source_immediate: None,
                    value: *value,
                    selected_result: *value,
                    address_param_leaves: Arc::new(Vec::new()),
                }
            }
            _ => continue,
        };
        if !proof.is_closed_v1() {
            continue;
        }
        let Some(plan) = stage4_use_plan(
            &proof,
            proof.selected_node,
            entry,
            owned_nodes,
            &definitions,
            &uses,
            &final_rtl,
            &succs,
        ) else {
            continue;
        };
        proofs.push((node, proof));
        plans.push((node, plan));
    }

    // A common two-address x86 RMW reaches RTL as `v = v op rhs`.  The
    // current SSA optimization eliminates that self-definition and thereby
    // loses the architectural register mutation from canonical source.
    // Recover only the narrow case where the exact
    // deterministic pre-opt winner was eliminated, `v` itself was never copy
    // substituted, and one surviving final definition of `v` ends exactly at
    // the raw RMW instruction.  The private source view may then append the
    // mutation to that definition; canonical RTL and source stay untouched.
    for (&node, selected_rows) in &preopt_selected {
        if final_rtl.contains_key(&node) || !optimizer_eliminated_nodes.contains(&node) {
            continue;
        }
        let [RTLInst::Iop(selected_operation, selected_args, selected_result)] =
            selected_rows.as_slice()
        else {
            continue;
        };
        let [root_definition] = definitions_at_node
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if selected_result != root_definition {
            continue;
        }
        let [instruction] = instructions.get(&node).map(Vec::as_slice).unwrap_or(&[]) else {
            continue;
        };
        let [raw_instruction] = raw_instructions
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if raw_instruction != instruction
            || instruction.1 != ""
            || address_sizes.get(&node) != Some(&BTreeSet::from([8]))
            || !reads.get(&node).is_none_or(BTreeSet::is_empty)
            || !writes.get(&node).is_none_or(BTreeSet::is_empty)
            || !matches!(instruction.2, "ADD" | "SUB" | "IMUL" | "AND" | "OR" | "XOR")
            || stage4_flag_producers.contains(&node)
        {
            continue;
        }
        let Some(function) =
            stage4_instruction_function(coff_map, &coff_index, &owners, node, instruction)
        else {
            continue;
        };
        let Some(owned_nodes) = function_nodes.get(&function) else {
            continue;
        };
        let Some(entry) = entries.get(&function).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };

        let operand_count = exact_raw_operand_count(instruction);
        let first_immediate = immediate_rows.get(&instruction.3).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        });
        let (source_operand, destination_operand, source_immediate) =
            match (instruction.2, operand_count, first_immediate) {
                ("IMUL", Some(3), None) => {
                    let Some((value, encoded_width)) = immediate_rows
                        .get(&instruction.5)
                        .and_then(|rows| {
                            (rows.len() == 1)
                                .then(|| rows.iter().next().copied())
                                .flatten()
                        })
                    else {
                        continue;
                    };
                    let source = stage4_exact_register(
                        instruction.3,
                        &registers,
                        &indirects,
                        &immediates,
                    );
                    let destination = stage4_exact_register(
                        instruction.4,
                        &registers,
                        &indirects,
                        &immediates,
                    );
                    if source.is_none() || source != destination {
                        continue;
                    }
                    if encoded_width != 0 {
                        continue;
                    }
                    if registers.contains_key(&instruction.5)
                        || indirects.contains_key(&instruction.5)
                    {
                        continue;
                    }
                    (None, instruction.4, Some((value, encoded_width)))
                }
                (_, Some(2), Some((value, encoded_width))) if instruction.2 != "IMUL" => {
                    if encoded_width != 0
                        || registers.contains_key(&instruction.3)
                        || indirects.contains_key(&instruction.3)
                    {
                        continue;
                    }
                    (None, instruction.4, Some((value, encoded_width)))
                }
                (_, Some(2), None) => (Some(instruction.3), instruction.4, None),
                _ => continue,
            };
        let Some(destination_name) = stage4_exact_register(
            destination_operand,
            &registers,
            &indirects,
            &immediates,
        ) else {
            continue;
        };
        let Some(width @ (4 | 8)) = x86_scalar_register_width(destination_name) else {
            continue;
        };
        let destination = Mreg::x86(destination_name);
        if destination.is_unknown() {
            continue;
        }
        let source = if let Some(source_operand) = source_operand {
            let Some(source_name) = stage4_exact_register(
                source_operand,
                &registers,
                &indirects,
                &immediates,
            ) else {
                continue;
            };
            if x86_scalar_register_width(source_name) != Some(width) {
                continue;
            }
            let source = Mreg::x86(source_name);
            if source.is_unknown() || source == destination {
                continue;
            }
            Some(source)
        } else {
            None
        };
        let source_immediate = source_immediate.map(|(value, _)| value);
        if width == 4
            && source_immediate.is_some_and(|value| i64::from(value as i32) != value)
        {
            continue;
        }
        let Some((kind, expected_operation)) =
            stage4_rmw_kind_and_operation(instruction.2, width, source_immediate)
        else {
            continue;
        };
        let companion_operation = if instruction.2 == "ADD" && source_immediate.is_none() {
            stage4_addition_operations(width).map(|(_, companion)| companion)
        } else if matches!(kind, Stage4SourceKind::And | Stage4SourceKind::Or) {
            stage4_and_or_width_companion(&expected_operation)
        } else {
            None
        };
        let selected_operation_is_exact = if kind == Stage4SourceKind::Add {
            selected_operation == &expected_operation
                || companion_operation.as_ref() == Some(selected_operation)
        } else {
            // Unlike ADD's sealed Oadd/Olea divergence, the other source
            // families do not represent a companion winner.  A width-twin
            // may coexist in LTL/input evidence, but the deterministic winner
            // must remain the exact raw-width operation serialized in proof.
            selected_operation == &expected_operation
        };
        if !selected_operation_is_exact {
            continue;
        }
        let expected_mregs: Vec<Mreg> = std::iter::once(destination)
            .chain(source)
            .collect();
        let expected_defs = BTreeSet::from([destination]);
        let expected_uses: BTreeSet<Mreg> = expected_mregs.iter().copied().collect();
        if decoded_defs.get(&node) != Some(&expected_defs)
            || decoded_uses.get(&node) != Some(&expected_uses)
        {
            continue;
        }
        let expected_ltl = LTLInst::Lop(
            expected_operation.clone(),
            Arc::new(expected_mregs.clone()),
            destination,
        );
        let companion_ltl = companion_operation.as_ref().map(|operation| {
            LTLInst::Lop(
                operation.clone(),
                Arc::new(expected_mregs.clone()),
                destination,
            )
        });
        let ltl_rows = ltl.get(&node).map(Vec::as_slice).unwrap_or(&[]);
        if !stage4_selected_with_optional_companion(
            ltl_rows,
            &expected_ltl,
            companion_ltl.as_ref(),
        ) {
            continue;
        }
        let Some(authenticated_operations) = ltl_rows
            .iter()
            .map(|row| match row {
                LTLInst::Lop(operation, _, _) => Some(operation.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let expected_args: Arc<Vec<RTLReg>> = selected_args.clone();
        let Some(&carrier) = expected_args.first() else {
            continue;
        };
        let source_reaching_value_is_exact = match (source, expected_args.get(1).copied()) {
            (Some(register), Some(value)) => reaching_values
                .get(&(node, register))
                .map(Vec::as_slice)
                == Some(&[value][..]),
            (None, None) => true,
            _ => false,
        };
        let source_value = expected_args.get(1).copied();
        if *selected_result == carrier
            || expected_args.len() != if source_immediate.is_some() { 1 } else { 2 }
            || !stage4_exact_mreg_binding(&rtl_mregs, node, carrier, destination)
            || source.zip(expected_args.get(1).copied()).is_some_and(|(register, value)| {
                !stage4_exact_mreg_binding(&rtl_mregs, node, value, register)
            })
            || (source.is_some() != (expected_args.len() == 2))
            || !source_reaching_value_is_exact
            || source_value
                .is_some_and(|value| !params.contains(&(function, value)))
            || optimizer_copy_substitutions
                .iter()
                .any(|(destination, source)| {
                    *destination == carrier
                        || *source == carrier
                        || *destination == *selected_result
                        || *source == *selected_result
                        || source_value
                            .is_some_and(|value| *destination == value || *source == value)
                })
        {
            continue;
        }
        // The Dual fixed point deliberately retains both the surviving
        // carrier and the fresh root definition as candidate destinations.
        // The deterministic pre-opt winner uses the latter; it is sealed to
        // the carrier by the exact is_def/reg_xtl/canonical web below.  Admit
        // only the complete operation x destination cross-product so a
        // missing, duplicate, or third transport alias remains ambiguous.
        if !stage4_rmw_input_candidates_match(
            input_rtl.get(&node).map(Vec::as_slice).unwrap_or(&[]),
            &authenticated_operations,
            &expected_args,
            carrier,
            *selected_result,
        ) {
            continue;
        }

        let Some(placement_node) = definitions.get(&carrier).and_then(|rows| {
            (rows.len() == 1)
                .then(|| rows.iter().next().copied())
                .flatten()
        }) else {
            continue;
        };
        let [RTLInst::Iload(placement_chunk, _, _, placement_value)] = final_rtl
            .get(&placement_node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if *placement_value != carrier
            || match placement_chunk {
                MemoryChunk::MInt32 | MemoryChunk::MAny32 => width != 4,
                MemoryChunk::MInt64 | MemoryChunk::MAny64 => width != 8,
                _ => true,
            }
        {
            continue;
        }
        let [placement_access] = scalar_accesses
            .get(&placement_node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if !placement_access.is_closed_v1()
            || placement_access.function != function
            || placement_access.origin_node != placement_node
            || placement_access.selected_node != placement_node
            || placement_access.direction != ScalarMemoryDirection::Read
            || placement_access.extension != ScalarMemoryExtension::Plain
            || placement_access.synthetic_stack_origin
            || placement_access.width != width
            || placement_access.value_width != width
            || placement_access.chunk != *placement_chunk
            || placement_access.value != carrier
        {
            continue;
        }
        let [placement_instruction] = instructions
            .get(&placement_node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        let [placement_raw] = raw_instructions
            .get(&placement_node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if placement_instruction != placement_raw
            || placement_node.checked_add(placement_instruction.0 as u64) != Some(node)
            || stage4_instruction_function(
                coff_map,
                &coff_index,
                &owners,
                placement_node,
                placement_instruction,
            ) != Some(function)
        {
            continue;
        }
        let [placement_definition] = definitions_at_node
            .get(&placement_node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
        else {
            continue;
        };
        if placement_definition == root_definition {
            continue;
        }
        let root_xtl_rows = reg_xtl_rows
            .get(&(node, destination))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        // A real two-address Lop carries two exact architectural identities at
        // its destination: the fresh definition minted for this root and the
        // prior placement-load definition propagated into the destination's
        // read position.  Both must collapse to the same final carrier below;
        // any missing, substituted, or additional historical row is
        // ambiguity.  Synthetic fixtures used to model only the fresh root
        // definition, which made this singleton check reject every real RMW.
        let mut expected_root_xtl_rows = vec![*placement_definition, *root_definition];
        expected_root_xtl_rows.sort_unstable();
        if reg_defs_reaching_use
            .get(&(node, destination))
            .map(Vec::as_slice)
            != Some(&[placement_node][..])
            || reg_uses_reached_from_def
                .get(&(placement_node, destination))
                .map(Vec::as_slice)
                != Some(&[node][..])
            || reg_xtl_rows
                .get(&(placement_node, destination))
                .map(Vec::as_slice)
                != Some(&[*placement_definition][..])
            || canonical_values.get(placement_definition) != Some(&carrier)
            || root_xtl_rows != expected_root_xtl_rows.as_slice()
            || canonical_values.get(root_definition) != Some(&carrier)
            || reaching_values
                .get(&(node, destination))
                .map(Vec::as_slice)
                != Some(&[carrier][..])
        {
            continue;
        }
        let Some(&anchor) = preopt_succs
            .get(&node)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        if preopt_succs_exact
            .get(&placement_node)
            .map(Vec::as_slice)
            != Some(&[node][..])
            || preopt_preds_exact.get(&node).map(Vec::as_slice)
                != Some(&[placement_node][..])
            || preopt_succs_exact.get(&node).map(Vec::as_slice) != Some(&[anchor][..])
            || preopt_preds_exact.get(&anchor).map(Vec::as_slice) != Some(&[node][..])
            || node.checked_add(instruction.0 as u64) != Some(anchor)
            || final_succs_exact
                .get(&placement_node)
                .map(Vec::as_slice)
                != Some(&[anchor][..])
            || final_preds_exact.get(&anchor).map(Vec::as_slice)
                != Some(&[placement_node][..])
            || !final_rtl.contains_key(&anchor)
            || graph_reaches_through_owned_region(
                &preopt_succs,
                owned_nodes,
                entry,
                node,
                None,
            ) != Some(true)
            || graph_reaches_through_owned_region(
                &preopt_succs,
                owned_nodes,
                entry,
                anchor,
                Some(node),
            ) != Some(false)
        {
            continue;
        }
        let expected_carrier_preopt_defs = BTreeSet::from([placement_node]);
        let expected_result_preopt_defs = BTreeSet::from([node]);
        let expected_carrier_preopt_uses = uses
            .get(&carrier)
            .into_iter()
            .flatten()
            .copied()
            .chain(std::iter::once(node))
            .collect::<BTreeSet<_>>();
        if preopt_definitions.get(&carrier) != Some(&expected_carrier_preopt_defs)
            || preopt_definitions.get(selected_result) != Some(&expected_result_preopt_defs)
            || preopt_uses.get(&carrier) != Some(&expected_carrier_preopt_uses)
            || preopt_uses.contains_key(selected_result)
            || definitions.contains_key(selected_result)
            || uses.contains_key(selected_result)
        {
            continue;
        }

        let operation = if kind == Stage4SourceKind::Add {
            selected_operation.clone()
        } else {
            expected_operation
        };
        let proof = Stage4SourceProof {
            function,
            origin_node: node,
            selected_node: node,
            kind,
            root_boundary: Stage4RootBoundary::EliminatedMutation,
            width,
            operation,
            args: expected_args,
            source_immediate,
            value: carrier,
            selected_result: *selected_result,
            address_param_leaves: Arc::new(Vec::new()),
        };
        if !proof.is_closed_v1() {
            continue;
        }
        let Some(plan) = stage4_use_plan(
            &proof,
            placement_node,
            entry,
            owned_nodes,
            &definitions,
            &uses,
            &final_rtl,
            &succs,
        ) else {
            continue;
        };
        let [terminal_site] = plan.sites.as_slice() else {
            continue;
        };
        let Some(terminal_use) = plan.terminal_use.as_ref() else {
            continue;
        };
        let terminal_is_sealed = match terminal_use {
            Stage4TerminalUse::Return => {
                let [terminal_instruction] = instructions
                    .get(&terminal_site.node)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                else {
                    continue;
                };
                let [terminal_raw] = raw_instructions
                    .get(&terminal_site.node)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                else {
                    continue;
                };
                terminal_instruction == terminal_raw
                    && terminal_instruction.1 == ""
                    && terminal_instruction.2 == "RET"
                    && exact_raw_operand_count(terminal_instruction) == Some(0)
                    && stage4_instruction_function(
                        coff_map,
                        &coff_index,
                        &owners,
                        terminal_site.node,
                        terminal_instruction,
                    ) == Some(function)
            }
            Stage4TerminalUse::Store { chunk, .. } => {
                let [access] = scalar_accesses
                    .get(&terminal_site.node)
                    .map(Vec::as_slice)
                    .unwrap_or(&[])
                else {
                    continue;
                };
                access.is_closed_v1()
                    && access.function == function
                    && access.origin_node == terminal_site.node
                    && access.selected_node == terminal_site.node
                    && access.direction == ScalarMemoryDirection::Write
                    && access.extension == ScalarMemoryExtension::Plain
                    && !access.synthetic_stack_origin
                    && access.width == width
                    && access.value_width == width
                    && access.chunk == *chunk
                    && access.value == carrier
            }
        };
        if !plan.transports.is_empty()
            || anchor != terminal_site.node
            || !terminal_is_sealed
            || reg_uses_reached_from_def
                .get(&(node, destination))
                .map(Vec::as_slice)
                != Some(&[terminal_site.node][..])
            || reg_defs_reaching_use
                .get(&(terminal_site.node, destination))
                .map(Vec::as_slice)
                != Some(&[node][..])
            || !stage4_flags_are_dead_before_terminal(
                node,
                terminal_site.node,
                terminal_use,
                function,
                coff_map,
                &coff_index,
                &owners,
                &instructions,
                &raw_instructions,
                &registers,
                &immediate_rows,
                &indirects,
                &decoded_defs,
                &decoded_uses,
                &preopt_succs_exact,
                &preopt_preds_exact,
            )
        {
            continue;
        }
        let all_plan_nodes = plan
            .transports
            .iter()
            .map(|transport| transport.node)
            .chain(plan.sites.iter().map(|site| site.node));
        if all_plan_nodes.into_iter().any(|use_node| {
            graph_reaches_through_owned_region(
                &preopt_succs,
                owned_nodes,
                node,
                use_node,
                None,
            ) != Some(true)
                || graph_reaches_through_owned_region(
                    &preopt_succs,
                    owned_nodes,
                    entry,
                    use_node,
                    Some(node),
                ) != Some(false)
        }) {
            continue;
        }
        proofs.push((node, proof));
        plans.push((node, plan));
    }

    proofs.sort();
    proofs.dedup();
    plans.sort();
    plans.dedup();
    db.rel_set(
        "authenticated_stage4_source",
        proofs.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "authenticated_stage4_use_plan",
        plans.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

/// Recover the one width fact which the Mach condition algebra cannot carry:
/// an x64 CR8 read compared through its matching low byte.  This is deliberately
/// expression-local.  The CR8 call keeps its 64-bit intrinsic declaration,
/// while only the authenticated comparison operand receives an unsigned-byte
/// cast.  The accepted RTL value must have that condition as its sole use, so
/// later C variable coalescing may retain its ordinary integer type without
/// losing any live high bits.
///
/// Every accepted site is tied independently to immutable decoder evidence and
/// to the post-optimization RTL value.  Ambiguous raw operands, ownership,
/// blocks, LTL interpretations, or final RTL instructions all fail closed.  A
/// function with multiple otherwise-valid sites is also left unchanged: this
/// initial rule never chooses among repeated CR8 assignments.
fn materialize_cr8_byte_compares(db: &mut DecompileDB) {
    let empty = || ascent::boxcar::Vec::<(Node, RTLReg)>::new();
    if !db.abi().uses_shared_arg_slots() {
        db.rel_set("cr8_byte_compare", empty());
        return;
    }

    let mut registers: BTreeMap<Symbol, BTreeSet<&'static str>> = BTreeMap::new();
    for (operand, register) in db.rel_iter::<(Symbol, &'static str)>("op_register") {
        registers.entry(*operand).or_default().insert(*register);
    }
    if !registers.values().any(|rows| rows.contains("CR8")) {
        db.rel_set("cr8_byte_compare", empty());
        return;
    }

    let mut next: BTreeMap<Node, BTreeSet<Node>> = BTreeMap::new();
    for (src, dst) in db.rel_iter::<(Node, Node)>("next") {
        next.entry(*src).or_default().insert(*dst);
    }
    let mut blocks: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, block) in db.rel_iter::<(Node, Address)>("code_in_block") {
        blocks.entry(*node).or_default().insert(*block);
    }
    let mut owners: BTreeMap<Node, BTreeSet<Address>> = BTreeMap::new();
    for (node, function) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        owners.entry(*node).or_default().insert(*function);
    }
    let mut immediates: BTreeMap<Symbol, BTreeSet<(i64, usize)>> = BTreeMap::new();
    for (operand, value, width) in db.rel_iter::<(Symbol, i64, usize)>("op_immediate") {
        immediates
            .entry(*operand)
            .or_default()
            .insert((*value, *width));
    }
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
        .map(|row| row.0)
        .collect();
    let mut raw: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for (node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) in
        db.rel_iter::<(
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
        )>("unrefinedinstruction")
    {
        raw.entry(*node).or_default().push((
            *size, *prefix, *mnemonic, *op1, *op2, *op3, *op4, *metadata0, *metadata1,
        ));
    }
    let mut decoded: BTreeMap<Node, Vec<Cr8RawInstruction>> = BTreeMap::new();
    for (node, size, prefix, mnemonic, op1, op2, op3, op4, metadata0, metadata1) in
        db.rel_iter::<(
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
        )>("instruction")
    {
        decoded.entry(*node).or_default().push((
            *size, *prefix, *mnemonic, *op1, *op2, *op3, *op4, *metadata0, *metadata1,
        ));
    }
    let mut ltl: BTreeMap<Node, Vec<LTLInst>> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, LTLInst)>("ltl_inst") {
        ltl.entry(*node).or_default().push(inst.clone());
    }
    let mut final_rtl: BTreeMap<Node, Vec<RTLInst>> = BTreeMap::new();
    for (node, inst) in db.rel_iter::<(Node, RTLInst)>("rtl_inst") {
        final_rtl.entry(*node).or_default().push(inst.clone());
    }
    let mut definitions: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
    let mut uses: BTreeMap<RTLReg, Vec<Node>> = BTreeMap::new();
    for (node, rows) in &final_rtl {
        for inst in rows {
            if let Some(value) = rtl_definition(inst) {
                definitions.entry(value).or_default().push(*node);
            }
            for value in inst_def_use(inst).1 {
                uses.entry(value).or_default().push(*node);
            }
        }
    }

    let exact_register = |operand: Symbol, wanted: &str| {
        registers.get(&operand).is_some_and(|rows| {
            rows.len() == 1 && rows.iter().next().is_some_and(|name| *name == wanted)
        }) && !immediates.contains_key(&operand)
            && !indirect_operands.contains(&operand)
    };
    let exact_immediate_one = |operand: Symbol| {
        immediates
            .get(&operand)
            .is_some_and(|rows| rows == &BTreeSet::from([(1_i64, 0_usize)]))
            && !registers.contains_key(&operand)
            && !indirect_operands.contains(&operand)
    };

    let mut authenticated_reads = BTreeSet::new();
    let mut candidates = BTreeSet::new();
    for (&read, read_rows) in &raw {
        if read_rows.len() != 1 {
            continue;
        }
        let read_row = &read_rows[0];
        let Some(decoded_read) = decoded
            .get(&read)
            .filter(|rows| rows.len() == 1 && rows.first() == Some(read_row))
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        if read_row.0 != 4
            || read_row.1 != ""
            || read_row.2 != "MOV"
            || read_row.7 != 0
            || read_row.8 != 0
            || exact_raw_operand_count(read_row) != Some(2)
            || !exact_register(read_row.3, "CR8")
        {
            continue;
        }
        let Some(full_register) = registers
            .get(&read_row.4)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.iter().next().copied())
        else {
            continue;
        };
        let Some((low_register, compare_size)) = low_byte_register(full_register) else {
            continue;
        };
        if !exact_register(read_row.4, full_register) {
            continue;
        }
        let Some(read_blocks) = blocks.get(&read).filter(|rows| rows.len() == 1) else {
            continue;
        };
        let Some(read_owners) = owners.get(&read).filter(|rows| rows.len() == 1) else {
            continue;
        };
        let function = *read_owners
            .iter()
            .next()
            .expect("len()==1 guard: one function owner");
        authenticated_reads.insert((function, read));

        let Some(compare) = next
            .get(&read)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.iter().next().copied())
        else {
            continue;
        };
        if Address::try_from(read_row.0)
            .ok()
            .and_then(|size| read.checked_add(size))
            != Some(compare)
            || Address::try_from(decoded_read.0)
                .ok()
                .and_then(|size| read.checked_add(size))
                != Some(compare)
        {
            continue;
        }
        let Some(compare_blocks) = blocks.get(&compare).filter(|rows| rows.len() == 1) else {
            continue;
        };
        let Some(compare_owners) = owners.get(&compare).filter(|rows| rows.len() == 1) else {
            continue;
        };
        if read_blocks != compare_blocks || read_owners != compare_owners {
            continue;
        }

        let Some(compare_rows) = raw.get(&compare).filter(|rows| rows.len() == 1) else {
            continue;
        };
        let compare_row = &compare_rows[0];
        if !decoded
            .get(&compare)
            .is_some_and(|rows| rows.len() == 1 && rows.first() == Some(compare_row))
        {
            continue;
        }
        if compare_row.0 != compare_size
            || compare_row.1 != ""
            || compare_row.2 != "CMP"
            || compare_row.7 != 0
            || compare_row.8 != 0
            || exact_raw_operand_count(compare_row) != Some(2)
            || !exact_immediate_one(compare_row.3)
            || !exact_register(compare_row.4, low_register)
        {
            continue;
        }

        let Some(read_ltl) = ltl
            .get(&read)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        let Some(compare_ltl) = ltl
            .get(&compare)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        let LTLInst::Lbuiltin(read_name, read_args, BuiltinArg::BA(read_mreg)) = read_ltl else {
            continue;
        };
        let LTLInst::Lcond(Condition::Ccompuimm(_, 1), compare_args, _, _) = compare_ltl else {
            continue;
        };
        let raw_result_mreg = crate::mreg::Mreg::x86(full_register);
        if raw_result_mreg.is_unknown()
            || *read_mreg != raw_result_mreg
            || read_name != "__readcr8"
            || !read_args.is_empty()
            || compare_args.as_slice() != [*read_mreg]
        {
            continue;
        }

        let Some(read_rtl) = final_rtl
            .get(&read)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        let Some(compare_rtl) = final_rtl
            .get(&compare)
            .filter(|rows| rows.len() == 1)
            .and_then(|rows| rows.first())
        else {
            continue;
        };
        let RTLInst::Ibuiltin(read_name, read_args, BuiltinArg::BA(value)) = read_rtl else {
            continue;
        };
        let RTLInst::Icond(Condition::Ccompuimm(_, 1), compare_args, _, _) = compare_rtl else {
            continue;
        };
        if read_name != "__readcr8" || !read_args.is_empty() || compare_args.as_slice() != [*value]
        {
            continue;
        }
        if !definitions
            .get(value)
            .is_some_and(|nodes| nodes.as_slice() == [read])
            || !uses
                .get(value)
                .is_some_and(|nodes| nodes.as_slice() == [compare])
        {
            continue;
        }
        candidates.insert((function, compare, *value));
    }

    let mut read_counts: BTreeMap<Address, usize> = BTreeMap::new();
    for (function, _) in &authenticated_reads {
        *read_counts.entry(*function).or_default() += 1;
    }
    let mut candidate_counts: BTreeMap<Address, usize> = BTreeMap::new();
    for (function, _, _) in &candidates {
        *candidate_counts.entry(*function).or_default() += 1;
    }
    let output: BTreeSet<(Node, RTLReg)> = candidates
        .into_iter()
        .filter_map(|(function, compare, value)| {
            (read_counts.get(&function) == Some(&1) && candidate_counts.get(&function) == Some(&1))
                .then_some((compare, value))
        })
        .collect();

    db.rel_set(
        "cr8_byte_compare",
        output.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

#[cfg(debug_assertions)]
use ascent::ascent_par;

// Declarative RTL optimizer; _v2 outputs are diffed against the imperative results (debug-only: excluded entirely in release builds where nothing reads them).
#[cfg(debug_assertions)]
ascent_par! {
    #![measure_rule_times]

    pub struct RTLOptimizerProgram;

    // Seeded inputs.
    relation rtl_opt_inst(Node, RTLInst);
    relation rtl_opt_succ(Node, Node);
    relation rtl_opt_func(Node, Address);
    relation rtl_opt_param(Address, RTLReg);
    relation rtl_opt_entry(Address, Node);
    relation rtl_opt_escaped(Address, RTLReg);
    // Mem-indirect call target-load address regs (call node -> address reg), seeded from FunctionCFG::mem_call_addr_uses so the v2 mirror matches the imperative def/use model.
    relation rtl_opt_mem_call_use(Node, RTLReg);

    // def/use extracted from instructions.
    relation rtl_def(Address, Node, RTLReg);
    relation rtl_use(Address, Node, RTLReg);

    // Definitions
    rtl_def(func, node, *dst) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Iop(_, _, dst));

    rtl_def(func, node, *dst) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Iload(_, _, _, dst));

    rtl_def(func, node, *dst) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Icall(_, _, _, Some(dst), _));

    rtl_def(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Ibuiltin(_, _, BuiltinArg::BA(r)));

    // Uses
    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Iop(_, args, _)),
        for u in args.iter();

    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Iload(_, _, args, _)),
        for u in args.iter();

    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Istore(_, _, args, _)),
        for u in args.iter();

    rtl_use(func, node, *src) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Istore(_, _, _, src));

    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Icall(_, _, args, _, _)),
        for u in args.iter();

    rtl_use(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Icall(_, Either::Left(r), _, _, _));

    // A mem-indirect call also uses the regs forming its target-load address.
    rtl_use(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_mem_call_use(node, r);

    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Itailcall(_, _, args)),
        for u in args.iter();

    rtl_use(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Itailcall(_, Either::Left(r), _));

    rtl_use(func, node, *u) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Icond(_, args, _, _)),
        for u in args.iter();

    rtl_use(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Ijumptable(r, _));

    rtl_use(func, node, *r) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Ireturn(r));

    // Ibuiltin uses: flatten BuiltinArg recursively.
    rtl_use(func, node, reg) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Ibuiltin(_, args, _)),
        for arg in args.iter(),
        let regs = flatten_ba_uses(arg),
        for reg in regs.iter().copied();

    // Live-out via backward dataflow.
    relation live_out(Node, RTLReg);

    // r live at exit of n if r is used at some successor before being defined there
    live_out(n, r) <--
        rtl_opt_succ(n, succ),
        rtl_use(_, succ, r);

    live_out(n, r) <--
        rtl_opt_succ(n, succ),
        live_out(succ, r),
        !rtl_def(_, succ, r);

    // rtl_reach(a, b): b is reachable from a (trrel).
    #[local]
    #[ds(ascent_byods_rels::trrel)]
    relation rtl_reach(Node, Node);

    rtl_reach(src, dst) <-- rtl_opt_succ(src, dst);

    // Single-definition counting.
    #[local] relation reg_def_count(Address, RTLReg, usize);

    reg_def_count(func, r, count) <--
        rtl_def(func, _, r),
        agg count = ascent::aggregators::count() in rtl_def(func, _, r);

    // Self-zero-rewrite: xor r,r or sub r,r where args == dst.
    relation self_zero_v2(Node);
    self_zero_v2(*node) <--
        rtl_opt_inst(node, ?RTLInst::Iop(op, args, dst)),
        if matches!(op, Operation::Oxor | Operation::Osub | Operation::Oxorl | Operation::Osubl),
        if args.len() == 2,
        if args[0] == args[1],
        if args[0] == *dst;

    // Copy candidates: Omove with 1 arg and distinct src/dst.
    #[local] relation copy_candidate(Address, Node, RTLReg, RTLReg); // (func, copy_node, src, dst)
    copy_candidate(func, node, args[0], *dst) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Iop(Operation::Omove, args, dst)),
        if args.len() == 1,
        if args[0] != *dst;

    // dst must be defined exactly once, in this copy_node
    #[local] relation dst_single_def_at(Address, Node, RTLReg);
    dst_single_def_at(func, node, *dst) <--
        copy_candidate(func, node, _, dst),
        reg_def_count(func, dst, 1),
        rtl_def(func, node, dst);

    // There is a use of dst NOT reachable from the copy_node (unsafe)
    #[local] relation dst_use_unreachable(Address, Node, RTLReg); // (func, copy_node, dst)
    dst_use_unreachable(func, copy_node, *dst) <--
        copy_candidate(func, copy_node, _, dst),
        rtl_use(func, use_node, dst),
        if use_node != copy_node,
        !rtl_reach(copy_node, use_node);

    // Unsafe: a use equal to the copy_node itself (dst uses itself at its own def).
    #[local] relation dst_self_use(Address, Node, RTLReg);
    dst_self_use(func, node, *dst) <--
        copy_candidate(func, node, _, dst),
        rtl_use(func, node, dst);

    // src redefined between copy_node and a use of dst: a mid between (copy_node, use_node] defines src.
    #[local] relation src_redef_on_path(Address, Node, RTLReg); // (func, copy_node, src)
    src_redef_on_path(func, copy_node, *src) <--
        copy_candidate(func, copy_node, src, dst),
        rtl_use(func, use_node, dst),
        rtl_reach(copy_node, mid),
        rtl_reach(mid, use_node),
        if mid != copy_node,
        rtl_def(func, mid, src);

    // Case where mid == use_node (use_node itself is a redef of src).
    src_redef_on_path(func, copy_node, *src) <--
        copy_candidate(func, copy_node, src, dst),
        rtl_use(func, use_node, dst),
        rtl_reach(copy_node, use_node),
        if use_node != copy_node,
        rtl_def(func, use_node, src);

    // At least one use of dst exists
    #[local] relation dst_has_use(Address, Node, RTLReg);
    dst_has_use(func, node, *dst) <--
        copy_candidate(func, node, _, dst),
        rtl_use(func, _, dst);

    // Final substitution: dst -> src
    relation copy_subst_v2(RTLReg, RTLReg);
    // Copy-eliminated node (replace with Inop)
    relation copy_eliminated_v2(Node);

    copy_subst_v2(*dst, *src), copy_eliminated_v2(*copy_node) <--
        copy_candidate(func, copy_node, src, dst),
        !rtl_opt_param(func, dst),
        !rtl_opt_escaped(func, src),
        !rtl_opt_escaped(func, dst),
        dst_single_def_at(func, copy_node, dst),
        dst_has_use(func, copy_node, dst),
        !dst_self_use(func, copy_node, dst),
        !dst_use_unreachable(func, copy_node, dst),
        !src_redef_on_path(func, copy_node, src);

    // Resolved substitutions: follow dst -> src chains until fixpoint.
    relation copy_subst_resolved_v2(RTLReg, RTLReg);
    copy_subst_resolved_v2(dst, src) <-- copy_subst_v2(dst, src), !copy_subst_v2(src, _);
    copy_subst_resolved_v2(dst, resolved) <--
        copy_subst_v2(dst, src),
        copy_subst_resolved_v2(src, resolved);

    // Dead store: def not in live_out, is Iop or Iload, not param.
    relation dead_store_v2(Node);
    dead_store_v2(*node) <--
        rtl_def(func, node, reg),
        !rtl_opt_param(func, reg),
        !rtl_opt_escaped(func, reg),
        !live_out(node, reg),
        rtl_opt_inst(node, inst),
        if matches!(inst, RTLInst::Iop(_,_,_) | RTLInst::Iload(_,_,_,_));

    // Dead call dst: Icall with dst not live-out; drop the dst, keep the call.
    relation dead_call_dst_v2(Node);
    dead_call_dst_v2(*node) <--
        rtl_opt_func(node, func),
        rtl_opt_inst(node, ?RTLInst::Icall(_, _, _, Some(dst), _)),
        !rtl_opt_param(func, dst),
        !live_out(node, dst);

    // Inline-temp candidates: single-def single-use Iop whose args are live at the use site.
    #[local] relation reg_distinct_uses(Address, RTLReg, usize);
    // Count distinct use-nodes per reg; rtl_use is (func, node, reg) so set semantics apply.
    reg_distinct_uses(func, r, count) <--
        rtl_use(func, _, r),
        agg count = ascent::aggregators::count() in rtl_use(func, _, r);

    #[local] relation inline_temp_def(Address, RTLReg, Node);
    inline_temp_def(func, *reg, *def_node) <--
        reg_def_count(func, reg, 1),
        rtl_def(func, def_node, reg),
        !rtl_opt_param(func, reg),
        rtl_opt_inst(def_node, ?RTLInst::Iop(_, _, _));

    #[local] relation inline_temp_use(Address, RTLReg, Node);
    inline_temp_use(func, *reg, *use_node) <--
        reg_distinct_uses(func, reg, 1),
        rtl_use(func, use_node, reg);

    // def_args must be in live_in(use_node); live_in(u) = use(u) | (live_out(u) \ def(u)).
    #[local] relation live_in(Node, RTLReg);
    live_in(n, r) <-- rtl_use(_, n, r);
    live_in(n, r) <--
        live_out(n, r),
        !rtl_def(_, n, r);

    #[local] relation def_arg_not_live_at_use(Address, RTLReg);
    def_arg_not_live_at_use(func, *reg) <--
        inline_temp_def(func, reg, def_node),
        inline_temp_use(func, reg, use_node),
        rtl_opt_inst(def_node, ?RTLInst::Iop(_, args, _)),
        for a in args.iter(),
        !live_in(use_node, a);

    #[local] relation inline_temp_reach_ok(Address, RTLReg);
    inline_temp_reach_ok(func, *reg) <--
        inline_temp_def(func, reg, def_node),
        inline_temp_use(func, reg, use_node),
        rtl_reach(def_node, use_node);

    relation inline_temp_v2(RTLReg);
    inline_temp_v2(*reg) <--
        inline_temp_def(func, reg, _),
        inline_temp_use(func, reg, _),
        !rtl_opt_escaped(func, reg),
        !def_arg_not_live_at_use(func, reg),
        inline_temp_reach_ok(func, reg);
}

// Build and run RTLOptimizerProgram over a PassContext; returns the program for inspection.
#[cfg(debug_assertions)]
fn run_rtl_optimizer_program(ctx: &PassContext) -> RTLOptimizerProgram {
    let mut prog = RTLOptimizerProgram::default();

    for (&func_addr, func) in &ctx.functions {
        for (&node, inst) in &func.inst {
            prog.rtl_opt_inst.push((node, inst.clone()));
            prog.rtl_opt_func.push((node, func_addr));
        }
        for (&src, dsts) in &func.succs {
            if func.inst.contains_key(&src) {
                for &dst in dsts {
                    prog.rtl_opt_succ.push((src, dst));
                }
            }
        }
        for &reg in &func.params {
            prog.rtl_opt_param.push((func_addr, reg));
        }
        for &reg in &func.escaped_slot_regs {
            prog.rtl_opt_escaped.push((func_addr, reg));
        }
        for (&node, regs) in &func.mem_call_addr_uses {
            for &r in regs {
                prog.rtl_opt_mem_call_use.push((node, r));
            }
        }
        prog.rtl_opt_entry.push((func_addr, func.entry));
    }

    prog.run();
    prog
}

#[derive(Debug, Default, Clone, Copy)]
struct RtlOptStats {
    copies: usize,
    dead: usize,
    nops: usize,
    inline_temps: usize,
    zero_rewrites: usize,
}

impl RtlOptStats {
    fn accumulate(&mut self, other: Self) {
        self.copies += other.copies;
        self.dead += other.dead;
        self.nops += other.nops;
        self.inline_temps += other.inline_temps;
        self.zero_rewrites += other.zero_rewrites;
    }

    fn has_changes(self) -> bool {
        self.copies > 0
            || self.dead > 0
            || self.nops > 0
            || self.inline_temps > 0
            || self.zero_rewrites > 0
    }
}

#[cfg(debug_assertions)]
struct AscentV2Snapshot {
    self_zero: HashSet<Node>,
    dead_store: HashSet<Node>,
    dead_call: HashSet<Node>,
    copy_subst: HashMap<RTLReg, RTLReg>,
    inline_temp: HashSet<RTLReg>,
    dead_or_copy_eliminated: HashSet<Node>,
}

fn optimize_rtl_candidates(db: &mut DecompileDB) -> Option<RtlOptStats> {
    let mut ctx = PassContext::load(db);
    if ctx.functions.is_empty() {
        db.rel_set(
            "stage4_preopt_selected_rtl",
            ascent::boxcar::Vec::<(Node, RTLInst)>::new(),
        );
        db.rel_set(
            "stage4_optimizer_copy_substitution",
            ascent::boxcar::Vec::<(RTLReg, RTLReg)>::new(),
        );
        db.rel_set(
            "stage4_optimizer_eliminated_node",
            ascent::boxcar::Vec::<(Node,)>::new(),
        );
        db.rel_set(
            "stage4_preopt_selected_succ",
            ascent::boxcar::Vec::<(Node, Node)>::new(),
        );
        return None;
    }

    // Preserve the exact deterministic winner selected by PassContext before
    // any copy propagation or DSE.  This is provider-internal provenance for
    // Stage-4 alternatives only; canonical optimization and write-back remain
    // unchanged.  Candidate rows which lost the PassContext competition can
    // never become source alternatives merely because they match raw bytes.
    let mut stage4_preopt_selected = Vec::new();
    let mut stage4_preopt_succ = Vec::new();
    for function in ctx.functions.values() {
        stage4_preopt_selected.extend(
            function
                .inst
                .iter()
                .map(|(&node, instruction)| (node, instruction.clone())),
        );
        for (&source, targets) in &function.succs {
            stage4_preopt_succ.extend(targets.iter().map(|&target| (source, target)));
        }
    }
    stage4_preopt_selected
        .sort_by_cached_key(|(node, instruction)| (*node, format!("{instruction:?}")));
    stage4_preopt_succ.sort_unstable();

    let var_types = std::mem::take(&mut ctx.var_types);

    let func_regs: HashMap<Address, HashSet<RTLReg>> = ctx
        .functions
        .iter()
        .map(|(&addr, func)| {
            let mut regs = HashSet::new();
            for inst in func.inst.values() {
                collect_inst_regs(inst, &mut regs);
            }
            regs.extend(func.params.iter());
            (addr, regs)
        })
        .collect();

    let per_func_var_types: HashMap<Address, HashMap<RTLReg, XType>> = func_regs
        .iter()
        .map(|(&addr, regs)| {
            let ft: HashMap<RTLReg, XType> = regs
                .iter()
                .filter_map(|&reg| var_types.get(&reg).map(|&ty| (reg, ty)))
                .collect();
            (addr, ft)
        })
        .collect();
    let per_func_var_types = std::sync::Mutex::new(per_func_var_types);

    // Return-value register per function (RTLPass binding); after static-eq branch folding prunes the dead tail, a function whose recorded return reg no longer has a surviving def has lost its only return value and is genuinely void (e.g. deregister_tm_clones).
    let func_return_reg: HashMap<Address, RTLReg> = db
        .rel_iter::<(Address, RTLReg)>("emit_function_return")
        .map(|&(addr, reg)| (addr, reg))
        .collect();

    // RTLOptimizerProgram is the Ascent v2 of the imperative optimizer. Its outputs are only consumed by the debug-only RTL_V2_DIFF block below, so in release builds we skip it entirely (dominates the RTL pass at ~57s on a 700KB binary).
    #[cfg(debug_assertions)]
    let ascent_v2: Option<AscentV2Snapshot> = if std::env::var("RTL_V2_DIFF").is_ok() {
        let ascent_opt = run_rtl_optimizer_program(&ctx);
        let self_zero: HashSet<Node> = ascent_opt.self_zero_v2.iter().map(|&(n,)| n).collect();
        let dead_store: HashSet<Node> = ascent_opt.dead_store_v2.iter().map(|&(n,)| n).collect();
        let copy_eliminated: HashSet<Node> = ascent_opt
            .copy_eliminated_v2
            .iter()
            .map(|&(n,)| n)
            .collect();
        let dead_call: HashSet<Node> = ascent_opt.dead_call_dst_v2.iter().map(|&(n,)| n).collect();
        let copy_subst: HashMap<RTLReg, RTLReg> = ascent_opt
            .copy_subst_resolved_v2
            .iter()
            .map(|&(d, s)| (d, s))
            .collect();
        let inline_temp: HashSet<RTLReg> =
            ascent_opt.inline_temp_v2.iter().map(|&(r,)| r).collect();
        let dead_or_copy_eliminated: HashSet<Node> =
            dead_store.union(&copy_eliminated).copied().collect();
        Some(AscentV2Snapshot {
            self_zero,
            dead_store,
            dead_call,
            copy_subst,
            inline_temp,
            dead_or_copy_eliminated,
        })
    } else {
        None
    };

    let results: Vec<_> = ctx
        .functions
        .par_iter_mut()
        .map(|(&func_addr, func)| {
            let preopt_inst = func.inst.clone();
            let mut func_var_types = per_func_var_types
                .lock()
                .unwrap()
                .remove(&func_addr)
                .unwrap_or_default();

            // Track iteration-1 imperative outputs for diffing against Ascent.
            let imp_self_zero: HashSet<Node> = func
                .inst
                .iter()
                .filter_map(|(&n, inst)| match inst {
                    RTLInst::Iop(op, args, dst)
                        if args.len() == 2
                            && args[0] == args[1]
                            && args[0] == *dst
                            && matches!(
                                op,
                                Operation::Oxor
                                    | Operation::Osub
                                    | Operation::Oxorl
                                    | Operation::Osubl
                            ) =>
                    {
                        Some(n)
                    }
                    _ => None,
                })
                .collect();

            let folded_static_branch = fold_static_eq_branches(func);

            let zero_rewrites = self_zero_rewrite(func);

            let mut copies = 0usize;
            let mut dead = 0usize;
            let mut iter1_copy_subst: HashMap<RTLReg, RTLReg> = HashMap::new();
            let mut iter1_dead_store_nodes: HashSet<Node> = HashSet::new();
            let mut iter1_dead_call_nodes: HashSet<Node> = HashSet::new();
            let mut all_copy_substitutions = BTreeSet::new();
            let mut first_iter = true;
            loop {
                let du = DefUseInfo::build(func);
                let liveness = LivenessInfo::build(func, &du);

                // Capture snapshot for iteration-1 comparison
                let dead_before: HashMap<Node, RTLInst> = if first_iter {
                    func.inst.iter().map(|(&n, i)| (n, i.clone())).collect()
                } else {
                    HashMap::new()
                };

                let (c, copy_subst) = copy_propagation(func, &du, &liveness);
                all_copy_substitutions.extend(copy_subst.iter().map(|(&dst, &src)| (dst, src)));
                if first_iter {
                    iter1_copy_subst = copy_subst.clone();
                }
                for (&dst, &src) in &copy_subst {
                    if let Some(&ty) = func_var_types.get(&dst) {
                        func_var_types.entry(src).or_insert(ty);
                    }
                }
                let d = dead_store_elimination(func, &du, &liveness);

                if first_iter {
                    for (&n, before) in &dead_before {
                        let after = func.inst.get(&n);
                        match (before, after) {
                            (
                                RTLInst::Iop(_, _, _) | RTLInst::Iload(_, _, _, _),
                                Some(RTLInst::Inop),
                            ) => {
                                // Non-nop def now nop: could be copy-prop or dead-store; both map to Inop.
                                iter1_dead_store_nodes.insert(n);
                            }
                            (
                                RTLInst::Icall(_, _, _, Some(_), _),
                                Some(RTLInst::Icall(_, _, _, None, _)),
                            ) => {
                                iter1_dead_call_nodes.insert(n);
                            }
                            _ => {}
                        }
                    }
                }

                if c == 0 && d == 0 {
                    break;
                }
                first_iter = false;
                copies += c;
                dead += d;
            }

            let nops = nop_collapse(func);

            let du = DefUseInfo::build(func);
            let liveness = LivenessInfo::build(func, &du);
            let inlines = find_inline_temps(func, &du, &liveness);
            let eliminated_nodes: BTreeSet<Node> = preopt_inst
                .iter()
                .filter_map(|(&node, instruction)| {
                    (!matches!(instruction, RTLInst::Inop)
                        && func
                            .inst
                            .get(&node)
                            .is_none_or(|final_instruction| {
                                matches!(final_instruction, RTLInst::Inop)
                            }))
                    .then_some(node)
                })
                .collect();

            // The function became void if static-eq folding pruned the defining node of its recorded return value, leaving the return reg with no surviving def (and not a parameter); scoped to folded functions so normal returns are untouched.
            let became_void = folded_static_branch
                && func_return_reg.get(&func_addr).is_some_and(|ret_reg| {
                    !func.params.contains(ret_reg)
                        && du.defs.get(ret_reg).map_or(true, |d| d.is_empty())
                });

            (
                RtlOptStats {
                    copies,
                    dead,
                    nops,
                    inline_temps: inlines.len(),
                    zero_rewrites,
                },
                inlines,
                func_var_types,
                // iteration-1 imperative snapshots for v2 diffing
                func_addr,
                imp_self_zero,
                iter1_copy_subst,
                iter1_dead_store_nodes,
                iter1_dead_call_nodes,
                all_copy_substitutions,
                eliminated_nodes,
                became_void,
            )
        })
        .collect();

    let mut stats = RtlOptStats::default();
    let mut merged_var_types = var_types;

    let mut imp_self_zero_all: HashSet<Node> = HashSet::new();
    let mut imp_copy_subst_all: HashMap<RTLReg, RTLReg> = HashMap::new();
    let mut imp_dead_store_all: HashSet<Node> = HashSet::new();
    let mut imp_dead_call_all: HashSet<Node> = HashSet::new();
    let mut newly_void: HashSet<Address> = HashSet::new();
    let mut stage4_copy_substitutions = BTreeSet::new();
    let mut stage4_eliminated_nodes = BTreeSet::new();

    for (
        func_stats,
        inlines,
        func_vt,
        fa,
        isz,
        icp,
        idst,
        idcl,
        copy_substitutions,
        eliminated_nodes,
        became_void,
    ) in results
    {
        stats.accumulate(func_stats);
        ctx.inline_temps.extend(inlines);
        merged_var_types.extend(func_vt);
        imp_self_zero_all.extend(isz);
        for (d, s) in icp {
            imp_copy_subst_all.insert(d, s);
        }
        imp_dead_store_all.extend(idst);
        imp_dead_call_all.extend(idcl);
        stage4_copy_substitutions.extend(copy_substitutions);
        stage4_eliminated_nodes.extend(eliminated_nodes);
        if became_void {
            newly_void.insert(fa);
        }
    }

    // Diff logging: compare Ascent v2 vs imperative iteration-1 outputs (uses eprintln so diffs surface during test runs, where no env_logger is configured).
    #[cfg(debug_assertions)]
    if let Some(v2) = &ascent_v2 {
        let self_zero_missing: Vec<_> = imp_self_zero_all
            .difference(&v2.self_zero)
            .copied()
            .collect();
        let self_zero_extra: Vec<_> = v2
            .self_zero
            .difference(&imp_self_zero_all)
            .copied()
            .collect();
        if !self_zero_missing.is_empty() || !self_zero_extra.is_empty() {
            eprintln!(
                "rtl_v2 diff self_zero: imperative={} ascent={} missing_from_ascent={} extra_in_ascent={}",
                imp_self_zero_all.len(), v2.self_zero.len(),
                self_zero_missing.len(), self_zero_extra.len()
            );
        }
        // imp_dead_store_all = union of copy-eliminated and dead-store nodes (both Inop in iter 1); compare against the same Ascent union.
        let ds_missing: Vec<_> = imp_dead_store_all
            .difference(&v2.dead_or_copy_eliminated)
            .copied()
            .collect();
        let ds_extra: Vec<_> = v2
            .dead_or_copy_eliminated
            .difference(&imp_dead_store_all)
            .copied()
            .collect();
        if !ds_missing.is_empty() || !ds_extra.is_empty() {
            eprintln!(
                "rtl_v2 diff dead_or_copy_elim (iter1): imperative={} ascent_union={} missing_from_ascent={} extra_in_ascent={}",
                imp_dead_store_all.len(), v2.dead_or_copy_eliminated.len(),
                ds_missing.len(), ds_extra.len()
            );
        }
        let dc_missing: Vec<_> = imp_dead_call_all
            .difference(&v2.dead_call)
            .copied()
            .collect();
        let dc_extra: Vec<_> = v2
            .dead_call
            .difference(&imp_dead_call_all)
            .copied()
            .collect();
        if !dc_missing.is_empty() || !dc_extra.is_empty() {
            eprintln!(
                "rtl_v2 diff dead_call_dst (iter1): imperative={} ascent={} missing={} extra={}",
                imp_dead_call_all.len(),
                v2.dead_call.len(),
                dc_missing.len(),
                dc_extra.len()
            );
        }
        // ascent copy_subst is path-conservative; expected subset of imperative.
        let cp_missing: Vec<_> = imp_copy_subst_all
            .iter()
            .filter(|&(d, _)| !v2.copy_subst.contains_key(d))
            .map(|(&d, &s)| (d, s))
            .collect();
        let cp_mismatch: Vec<_> = v2
            .copy_subst
            .iter()
            .filter_map(|(&d, &s_asc)| {
                imp_copy_subst_all.get(&d).and_then(|&s_imp| {
                    if s_asc != s_imp {
                        Some((d, s_imp, s_asc))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let cp_extra: Vec<_> = v2
            .copy_subst
            .iter()
            .filter(|&(d, _)| !imp_copy_subst_all.contains_key(d))
            .map(|(&d, &s)| (d, s))
            .collect();
        if !cp_missing.is_empty() || !cp_extra.is_empty() || !cp_mismatch.is_empty() {
            eprintln!(
                "rtl_v2 diff copy_subst (iter1): imperative={} ascent={} ascent_missing={} ascent_extra={} value_mismatch={}",
                imp_copy_subst_all.len(), v2.copy_subst.len(),
                cp_missing.len(), cp_extra.len(), cp_mismatch.len()
            );
        }
        let it_missing: Vec<_> = ctx
            .inline_temps
            .difference(&v2.inline_temp)
            .copied()
            .collect();
        let it_extra: Vec<_> = v2
            .inline_temp
            .difference(&ctx.inline_temps)
            .copied()
            .collect();
        if !it_missing.is_empty() || !it_extra.is_empty() {
            eprintln!(
                "rtl_v2 diff inline_temp: imperative={} ascent={} missing={} extra={}",
                ctx.inline_temps.len(),
                v2.inline_temp.len(),
                it_missing.len(),
                it_extra.len()
            );
        }
    }

    ctx.var_types = merged_var_types;
    ctx.write_back(db);
    db.rel_set(
        "stage4_preopt_selected_rtl",
        stage4_preopt_selected
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "stage4_optimizer_copy_substitution",
        stage4_copy_substitutions
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "stage4_optimizer_eliminated_node",
        stage4_eliminated_nodes
            .into_iter()
            .map(|node| (node,))
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "stage4_preopt_selected_succ",
        stage4_preopt_succ
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    if !newly_void.is_empty() {
        mark_functions_void(db, &newly_void);
    }
    Some(stats)
}

// Retype functions that lost their only return value to void: emit_function_void_candidate wins the ladder, and stale has-return signals are dropped so they cannot reintroduce a bogus return type.
fn mark_functions_void(db: &mut DecompileDB, void_funcs: &HashSet<Address>) {
    let mut void_cand: Vec<(Address,)> = db
        .rel_iter::<(Address,)>("emit_function_void_candidate")
        .cloned()
        .collect();
    for &f in void_funcs {
        if !void_cand.iter().any(|&(a,)| a == f) {
            void_cand.push((f,));
        }
    }
    db.rel_set(
        "emit_function_void_candidate",
        void_cand.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let has_ret: Vec<(Address,)> = db
        .rel_iter::<(Address,)>("emit_function_has_return_candidate")
        .filter(|&&(a,)| !void_funcs.contains(&a))
        .cloned()
        .collect();
    db.rel_set(
        "emit_function_has_return_candidate",
        has_ret.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let ret: Vec<(Address, RTLReg)> = db
        .rel_iter::<(Address, RTLReg)>("emit_function_return")
        .filter(|&&(a, _)| !void_funcs.contains(&a))
        .cloned()
        .collect();
    db.rel_set(
        "emit_function_return",
        ret.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let ret_ct: Vec<(Address, ClightType)> = db
        .rel_iter::<(Address, ClightType)>("emit_function_return_type_candidate")
        .filter(|&&(a, _)| !void_funcs.contains(&a))
        .cloned()
        .collect();
    db.rel_set(
        "emit_function_return_type_candidate",
        ret_ct.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );

    let ret_xt: Vec<(Address, XType)> = db
        .rel_iter::<(Address, XType)>("emit_function_return_type_xtype_candidate")
        .filter(|&&(a, _)| !void_funcs.contains(&a))
        .cloned()
        .collect();
    db.rel_set(
        "emit_function_return_type_xtype_candidate",
        ret_xt.into_iter().collect::<ascent::boxcar::Vec<_>>(),
    );
}

pub(crate) fn trim_direct_call_args_to_callee_arity(db: &mut DecompileDB) {
    let per_call_contracts: HashSet<Node> = db
        .rel_iter::<(Node, Address, Node, Node, RTLReg)>("msvc_gs_cookie_guard_call")
        .map(|(call, ..)| *call)
        .collect();
    let mut callee_arity: HashMap<Address, usize> = db
        .rel_iter::<(Address, Signature)>("emit_function_signature_candidate")
        .fold(HashMap::new(), |mut acc, (addr, sig)| {
            acc.entry(*addr)
                .and_modify(|curr| *curr = (*curr).max(sig.sig_args.len()))
                .or_insert(sig.sig_args.len());
            acc
        });

    // RTL's first signature candidate holds register params only; include the upstream stack-param count now, or this pass deletes recovered stack arguments before reconciliation widens the callee.
    let shared_arg_slots = db.abi().uses_shared_arg_slots();
    let first_stack_arg_position = db.abi().first_stack_arg_position();
    for &(addr, stack_count) in db.rel_iter::<(Address, usize)>("emit_function_stack_param_count") {
        let arity = callee_arity.entry(addr).or_insert(0);
        if shared_arg_slots && stack_count > 0 {
            // Win64's home space fixes the first stack parameter at ordinal 4, so preserve unobserved register-slot gaps instead of treating stack params as a dense suffix.
            *arity = (*arity).max(first_stack_arg_position) + stack_count;
        } else {
            *arity += stack_count;
        }
    }

    let call_targets: HashMap<Node, Address> = db
        .rel_iter::<(Node, Address)>("call_target_func")
        .map(|(call_node, target)| (*call_node, *target))
        .collect();

    // Variadic callees receive more args than their fixed signature, so detect them structurally via the SysV variadic XMM register-save-area prologue rather than trimming their tail args.
    let varargs_callees: HashSet<Address> = db
        .rel_iter::<(Address,)>("func_has_variadic_xmm_prologue")
        .map(|&(addr,)| addr)
        .collect();

    let filtered_mapping: Vec<(Node, usize, RTLReg)> = db
        .rel_iter::<(Node, usize, RTLReg)>("call_arg_mapping")
        .filter_map(|&(node, pos, reg)| {
            if !per_call_contracts.contains(&node) {
                let Some(&target) = call_targets.get(&node) else {
                    return Some((node, pos, reg));
                };
                // Root E: keep the whole argument tail for a detected-variadic callee; arg_setup_candidate already restricts each position to a def that structurally reaches the call.
                let keep_varargs_tail = varargs_callees.contains(&target);
                if !keep_varargs_tail {
                    if let Some(&arity) = callee_arity.get(&target) {
                        if pos >= arity {
                            return None;
                        }
                    }
                }
            }
            Some((node, pos, reg))
        })
        .collect();

    db.rel_set(
        "call_arg_mapping",
        filtered_mapping
            .iter()
            .copied()
            .collect::<ascent::boxcar::Vec<_>>(),
    );
    db.rel_set(
        "call_arg",
        filtered_mapping
            .iter()
            .copied()
            .collect::<ascent::boxcar::Vec<_>>(),
    );

    let mut normalized_mapping: HashMap<(Node, usize), RTLReg> = HashMap::new();
    for (node, pos, reg) in filtered_mapping {
        if !per_call_contracts.contains(&node) {
            if let Some(&target) = call_targets.get(&node) {
                let keep_varargs_tail = varargs_callees.contains(&target);
                if !keep_varargs_tail {
                    if let Some(&arity) = callee_arity.get(&target) {
                        debug_assert!(pos < arity);
                    }
                }
            }
        }
        normalized_mapping
            .entry((node, pos))
            .and_modify(|curr| *curr = (*curr).max(reg))
            .or_insert(reg);
    }

    let mut args_by_call: HashMap<Node, Vec<(usize, RTLReg)>> = HashMap::new();
    for (&(node, pos), &reg) in &normalized_mapping {
        args_by_call.entry(node).or_default().push((pos, reg));
    }
    let rebuild_args = |node: Node| -> Option<Arc<Vec<RTLReg>>> {
        let mut pairs = args_by_call.get(&node)?.clone();
        pairs.sort_by_key(|(pos, _)| *pos);
        // Scatter by position (sentinel DEFAULT_VAR for holes so dropped leading args don't left-shift later ones); len = max_pos+1 is safe because pos <= 5 (6-entry ARG_REGS, rtl_pass.rs).
        let len = pairs.last().map(|(pos, _)| *pos + 1).unwrap_or(0);
        let mut args: Vec<RTLReg> = vec![crate::util::DEFAULT_VAR as RTLReg; len];
        for (pos, reg) in pairs {
            args[pos] = reg;
        }
        Some(Arc::new(args))
    };

    let mut call_nodes: HashSet<Node> = db
        .rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_args_collected_candidate")
        .map(|(node, _)| *node)
        .collect();
    call_nodes.extend(args_by_call.keys().copied());

    let mut existing_call_args: HashMap<Node, Arc<Vec<RTLReg>>> = HashMap::new();
    for &(node, ref args) in
        db.rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_args_collected_candidate")
    {
        let should_replace = existing_call_args
            .get(&node)
            .map(|curr| curr.len() < args.len())
            .unwrap_or(true);
        if should_replace {
            existing_call_args.insert(node, args.clone());
        }
    }

    let new_call_args: ascent::boxcar::Vec<(Node, Arc<Vec<RTLReg>>)> = call_nodes
        .into_iter()
        .map(|node| {
            let args = rebuild_args(node)
                .or_else(|| existing_call_args.get(&node).cloned())
                .unwrap_or_else(|| Arc::new(vec![]));
            (node, args)
        })
        .collect();
    db.rel_set("call_args_collected_candidate", new_call_args);

    let mut rebuilt_args: HashMap<Node, Arc<Vec<RTLReg>>> = db
        .rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_args_collected_candidate")
        .map(|(node, args)| (*node, args.clone()))
        .collect();

    // Merge float args after integer args for each call node
    for &(call_node, ref float_args) in
        db.rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_float_args_collected")
    {
        if per_call_contracts.contains(&call_node) {
            continue;
        }
        if !float_args.is_empty() {
            let int_args = rebuilt_args
                .entry(call_node)
                .or_insert_with(|| Arc::new(vec![]));
            let mut combined = (**int_args).clone();
            combined.extend_from_slice(float_args);
            *int_args = Arc::new(combined);
        }
    }

    let new_rtl_inst: ascent::boxcar::Vec<(Node, RTLInst)> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .map(|(node, inst)| {
            let replacement_args = rebuilt_args.get(node);
            let patched = match inst {
                RTLInst::Icall(sig, callee, args, dst, next) => RTLInst::Icall(
                    sig.clone(),
                    callee.clone(),
                    replacement_args.cloned().unwrap_or_else(|| args.clone()),
                    *dst,
                    *next,
                ),
                RTLInst::Itailcall(sig, callee, args) => RTLInst::Itailcall(
                    sig.clone(),
                    callee.clone(),
                    replacement_args.cloned().unwrap_or_else(|| args.clone()),
                ),
                _ => inst.clone(),
            };
            (*node, patched)
        })
        .collect();
    db.rel_set("rtl_inst_candidate", new_rtl_inst);
}

pub struct RTLOptimizePass;

impl IRPass for RTLOptimizePass {
    fn name(&self) -> &'static str {
        "rtl_optimize"
    }

    fn run(&self, db: &mut DecompileDB) {
        trim_direct_call_args_to_callee_arity(db);
        if let Some(stats) = optimize_rtl_candidates(db) {
            if stats.has_changes() {
                info!(
                    "rtl: {} copies propagated, {} dead stores, {} nops collapsed, {} inline temps, {} zero rewrites",
                    stats.copies,
                    stats.dead,
                    stats.nops,
                    stats.inline_temps,
                    stats.zero_rewrites,
                );
            }
        }
        materialize_cr8_byte_compares(db);
        let stage4_private_accesses = materialize_authenticated_scalar_memory_accesses(db);
        materialize_authenticated_stage4_sources_with_private_accesses(
            db,
            &stage4_private_accesses,
        );
    }

    fn inputs(&self) -> &'static [&'static str] {
        &[
            "rtl_inst_candidate",
            "rtl_succ_candidate",
            "instr_in_function",
            "emit_function",
            "emit_function_param_candidate",
            "emit_var_type_candidate",
            "emit_function_signature_candidate",
            "emit_function_stack_param_count",
            "emit_function_return",
            "call_target_func",
            "call_arg_mapping",
            "call_args_collected_candidate",
            "call_float_args_collected",
            "func_has_variadic_xmm_prologue",
            "msvc_gs_cookie_guard_call",
            "reg_def_used",
            "reg_xtl",
            "is_def",
            "xtl_canonical",
            "reaching_use_rtl",
        ]
    }

    fn outputs(&self) -> &'static [&'static str] {
        &[
            "rtl_inst",
            "rtl_succ",
            "rtl_next",
            "single_def_const",
            "emit_inline_temp",
            "call_arg_mapping",
            "call_arg",
            "call_args_collected_candidate",
            "cr8_byte_compare",
            "authenticated_scalar_memory_access",
            "authenticated_scalar_memory_use_plan",
            "authenticated_stage4_source",
            "authenticated_stage4_use_plan",
            "stage4_preopt_selected_rtl",
            "stage4_optimizer_copy_substitution",
            "stage4_optimizer_eliminated_node",
            "stage4_preopt_selected_succ",
            // Static-eq branch folding can retype a function void; these signal that to the signature reconciliation pass.
            "emit_function_void_candidate",
            "emit_function_has_return_candidate",
            "emit_function_return",
            "emit_function_return_type_candidate",
            "emit_function_return_type_xtype_candidate",
        ]
    }

    fn extra_reads(&self) -> &'static [&'static str] {
        // Read imperatively in PassContext::load: call_through_memory_load
        // keeps a vtable Iload alive; jump_table_* protects in-loop dispatch;
        // both ordinary and canonical-home escaped slots are mutable memory.
        &[
            "call_through_memory_load",
            "jump_table_impl",
            "jump_table_cmp",
            "jump_table_target",
            "slot_escaped_canonical",
            "win64_home_escaped",
            // Immutable decoder/LTL evidence used to recover the comparison-
            // local byte width after final RTL register rewriting.
            "next",
            "code_in_block",
            "flags_and_jump_pair",
            "ltl_inst",
            "op_register",
            "op_immediate",
            "op_indirect",
            "instruction",
            "unrefinedinstruction",
            "decoded_memory_read_operand",
            "decoded_memory_write_operand",
            "decoded_reg_def",
            "decoded_reg_use",
            "instruction_address_size",
            "reg_rtl",
            "reg_def_used",
            "reg_xtl",
            "is_def",
            "xtl_canonical",
            "reaching_use_rtl",
        ]
    }
}

pub(crate) struct FunctionCFG {
    #[allow(dead_code)]
    pub(crate) func_addr: Address,
    pub(crate) entry: Node,
    pub(crate) nodes: BTreeSet<Node>,
    pub(crate) inst: BTreeMap<Node, RTLInst>,
    pub(crate) succs: HashMap<Node, Vec<Node>>,
    pub(crate) preds: HashMap<Node, Vec<Node>>,
    pub(crate) params: HashSet<RTLReg>,
    // The address regs feeding a mem-indirect call's function-pointer load; rtl_pass threads them only into call_through_memory_load, so record them here or DSE kills the producing Iload.
    pub(crate) mem_call_addr_uses: HashMap<Node, Vec<RTLReg>>,
    // Jump-table dispatch entries of IN-LOOP tables: collapsing one threads the guard past the switch, so only these are protected from nop_collapse.
    pub(crate) dispatch_entry_nodes: HashSet<Node>,
    // Canonical SSA regs of address-escaped stack slots: the callee may write through the escaped pointer, a use reg liveness cannot see, so their stores are excluded from DSE.
    pub(crate) escaped_slot_regs: HashSet<RTLReg>,
}

struct PassContext {
    functions: BTreeMap<Address, FunctionCFG>,
    var_types: HashMap<RTLReg, XType>,
    inline_temps: HashSet<RTLReg>,
}

impl PassContext {
    fn load(db: &DecompileDB) -> Self {
        let rtl_insts: Vec<(Node, RTLInst)> = db
            .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
            .cloned()
            .collect();
        let rtl_succs: Vec<(Node, Node)> = db
            .rel_iter::<(Node, Node)>("rtl_succ_candidate")
            .cloned()
            .collect();
        let instr_in_func: Vec<(Node, Address)> = db
            .rel_iter::<(Node, Address)>("instr_in_function")
            .cloned()
            .collect();
        let emit_funcs: Vec<(Address, Symbol, Node)> = db
            .rel_iter::<(Address, Symbol, Node)>("emit_function")
            .cloned()
            .collect();
        let emit_params: Vec<(Address, RTLReg)> = db
            .rel_iter::<(Address, RTLReg)>("emit_function_param_candidate")
            .cloned()
            .collect();
        // Per jump table: dispatch entry, bounds-check guard, and case targets; loop membership is tested as "a case body reaches the guard", since the dispatch JMP's edges are not yet in the CFG.
        let mut dispatch_entry_by_table: HashMap<Node, Node> = HashMap::new();
        for &(impl_addr, jmp_addr) in db.rel_iter::<(Node, Node)>("jump_table_impl") {
            dispatch_entry_by_table
                .entry(jmp_addr)
                .and_modify(|e| {
                    if impl_addr < *e {
                        *e = impl_addr;
                    }
                })
                .or_insert(impl_addr);
        }
        let mut table_guard: HashMap<Node, Node> = HashMap::new();
        for &(jmp_addr, cmp_addr) in db.rel_iter::<(Node, Node)>("jump_table_cmp") {
            table_guard.entry(jmp_addr).or_insert(cmp_addr);
        }
        let mut table_cases: HashMap<Node, Vec<Node>> = HashMap::new();
        for &(jmp_addr, _idx, target) in db.rel_iter::<(Node, usize, Node)>("jump_table_target") {
            table_cases.entry(jmp_addr).or_default().push(target);
        }
        // (dispatch_entry, guard, case_targets) per table that has both an entry and a guard.
        let dispatch_loop_probes: Vec<(Node, Node, Vec<Node>)> = dispatch_entry_by_table
            .iter()
            .filter_map(|(&jmp, &entry)| {
                let guard = *table_guard.get(&jmp)?;
                let cases = table_cases.get(&jmp).cloned().unwrap_or_default();
                Some((entry, guard, cases))
            })
            .collect();

        // Address regs feeding a mem-indirect call's target load, per call node (node -> [base, idx?]); see FunctionCFG::mem_call_addr_uses. Declared in extra_reads().
        let mut mem_call_addr_uses: HashMap<Node, Vec<RTLReg>> = HashMap::new();
        for (node, _temp, _chunk, _addr, args) in db.rel_iter::<(
            Node,
            RTLReg,
            MemoryChunk,
            crate::x86::op::Addressing,
            Arc<Vec<RTLReg>>,
        )>("call_through_memory_load")
        {
            let entry = mem_call_addr_uses.entry(*node).or_default();
            for &r in args.iter() {
                if !entry.contains(&r) {
                    entry.push(r);
                }
            }
        }

        // Per function, the canonical SSA regs of address-escaped stack slots, whose stores must not be dead-store-eliminated. Declared in extra_reads().
        let mut escaped_slot_regs: HashMap<Address, HashSet<RTLReg>> = HashMap::new();
        for &(func, _ofs, reg) in db.rel_iter::<(Address, i64, RTLReg)>("slot_escaped_canonical") {
            escaped_slot_regs.entry(func).or_default().insert(reg);
        }
        for &(func, reg) in db.rel_iter::<(Address, RTLReg)>("win64_home_escaped") {
            escaped_slot_regs.entry(func).or_default().insert(reg);
        }
        // Per reg, keep the max-(refine_priority, type) candidate via fold-during-insert (avoids intermediate Vec; key embeds type so equal-key ties are identical values, making the winner iteration-order-independent).
        let var_types: HashMap<RTLReg, XType> = {
            let key = |ty: &XType| {
                (
                    crate::decompile::passes::clight_pass::xtype_refine_priority(ty),
                    *ty,
                )
            };
            let mut chosen: HashMap<RTLReg, XType> = HashMap::new();
            for (reg, xty) in db.rel_iter::<(RTLReg, XType)>("emit_var_type_candidate") {
                chosen
                    .entry(*reg)
                    .and_modify(|cur| {
                        if key(xty) > key(cur) {
                            *cur = *xty;
                        }
                    })
                    .or_insert(*xty);
            }
            chosen
        };

        let node_to_func: HashMap<Node, Address> = {
            let mut groups: HashMap<Node, Address> = HashMap::new();
            for &(n, f) in &instr_in_func {
                groups
                    .entry(n)
                    .and_modify(|curr| *curr = (*curr).min(f))
                    .or_insert(f);
            }
            groups
        };

        let func_entries: HashMap<Address, Node> = emit_funcs
            .iter()
            .map(|&(addr, _, entry)| (addr, entry))
            .collect();

        let mut func_params: HashMap<Address, HashSet<RTLReg>> = HashMap::new();
        for &(addr, reg) in &emit_params {
            func_params.entry(addr).or_default().insert(reg);
        }

        let mut func_node_candidates: HashMap<Address, HashMap<Node, Vec<RTLInst>>> =
            HashMap::new();
        for (node, inst) in &rtl_insts {
            if let Some(&func) = node_to_func.get(node) {
                func_node_candidates
                    .entry(func)
                    .or_default()
                    .entry(*node)
                    .or_default()
                    .push(inst.clone());
            }
        }

        let mut reg_usage_count: HashMap<RTLReg, usize> = HashMap::new();
        for (_, inst) in &rtl_insts {
            let mut regs = HashSet::new();
            collect_inst_regs(inst, &mut regs);
            for reg in regs {
                *reg_usage_count.entry(reg).or_insert(0) += 1;
            }
        }

        let mut func_insts: HashMap<Address, BTreeMap<Node, RTLInst>> = HashMap::new();
        for (func, node_candidates) in func_node_candidates {
            let insts = func_insts.entry(func).or_default();
            // INVARIANT: entries in func_node_candidates are created only via push, so each Vec has >= 1 element; malformed input changes which candidates appear, never empties an existing list.
            for (node, candidates) in node_candidates {
                if candidates.len() == 1 {
                    let only = candidates
                        .into_iter()
                        .next()
                        .expect("len()==1 guard: single candidate present");
                    insts.insert(node, only);
                } else {
                    let mut candidates = candidates;
                    candidates.sort_by_cached_key(|inst| format!("{:?}", inst));
                    // Icond wins categorically when Icond and Iop collide on one node, since Iop is side-effect-only while Icond carries the CFG edge; score only breaks ties among same-kind candidates.
                    let best = candidates.into_iter().max_by_key(|inst| {
                        let mut regs = HashSet::new();
                        collect_inst_regs(inst, &mut regs);
                        let score: usize = regs.iter()
                            .map(|r| reg_usage_count.get(r).copied().unwrap_or(0))
                            .sum();
                        let arg_bonus = match inst {
                            RTLInst::Iop(_, args, _) if !args.is_empty() => 1,
                            RTLInst::Iload(_, _, args, _) if !args.is_empty() => 1,
                            RTLInst::Icond(_, args, _, _) if !args.is_empty() => 1,
                            _ => 0,
                        };
                        let is_cond = matches!(inst, RTLInst::Icond(..));
                        (is_cond, score + arg_bonus)
                    }).expect("non-empty by construction: func_node_candidates entries are created by push (see invariant above)");
                    insts.insert(node, best);
                }
            }
        }

        let mut func_succs: HashMap<Address, HashMap<Node, Vec<Node>>> = HashMap::new();
        for &(src, dst) in &rtl_succs {
            if let Some(&func) = node_to_func.get(&src) {
                func_succs
                    .entry(func)
                    .or_default()
                    .entry(src)
                    .or_default()
                    .push(dst);
            }
        }

        let mut functions = BTreeMap::new();
        for (&func_addr, insts) in &func_insts {
            let entry = match func_entries.get(&func_addr) {
                Some(&e) => e,
                None => continue,
            };
            let nodes: BTreeSet<Node> = insts.keys().copied().collect();

            let raw_succs = func_succs.remove(&func_addr).unwrap_or_default();
            let mut succs: HashMap<Node, Vec<Node>> = HashMap::new();
            for (src, dsts) in raw_succs {
                if nodes.contains(&src) {
                    succs.insert(src, dsts);
                }
            }

            let nodes_vec: Vec<Node> = nodes.iter().copied().collect();
            for (i, &node) in nodes_vec.iter().enumerate() {
                if succs.contains_key(&node) {
                    continue;
                }
                let inst = match insts.get(&node) {
                    Some(inst) => inst,
                    None => continue,
                };
                let needs_fallthrough = matches!(
                    inst,
                    RTLInst::Inop
                        | RTLInst::Iop(..)
                        | RTLInst::Iload(..)
                        | RTLInst::Istore(..)
                        | RTLInst::Icall(..)
                        | RTLInst::Ibuiltin(..)
                );
                if needs_fallthrough {
                    if let Some(&next_node) = nodes_vec.get(i + 1) {
                        succs.entry(node).or_default().push(next_node);
                    }
                }
                match inst {
                    RTLInst::Ibranch(Either::Right(target)) => {
                        if !succs.contains_key(&node) {
                            if let Some(&dst) = nodes.range(*target..).next() {
                                succs.entry(node).or_default().push(dst);
                            }
                        }
                    }
                    _ => {}
                }
            }

            let mut preds: HashMap<Node, Vec<Node>> = HashMap::new();
            for (&src, dsts) in &succs {
                for &dst in dsts {
                    if nodes.contains(&dst) {
                        preds.entry(dst).or_default().push(src);
                    }
                }
            }

            let func_mem_call_uses: HashMap<Node, Vec<RTLReg>> = nodes
                .iter()
                .filter_map(|n| mem_call_addr_uses.get(n).map(|regs| (*n, regs.clone())))
                .collect();

            let mut func = FunctionCFG {
                func_addr,
                entry,
                nodes,
                inst: insts.clone(),
                succs,
                preds,
                params: func_params.remove(&func_addr).unwrap_or_default(),
                mem_call_addr_uses: func_mem_call_uses,
                dispatch_entry_nodes: HashSet::new(),
                escaped_slot_regs: escaped_slot_regs.remove(&func_addr).unwrap_or_default(),
            };
            // A dispatch entry is protected only when some case body flows back to the bounds-check guard, i.e. the switch is inside a loop; structural CFG reachability, no address comparison.
            let mut dispatch_entry_nodes: HashSet<Node> = HashSet::new();
            for (entry, guard, cases) in &dispatch_loop_probes {
                if !func.nodes.contains(entry) || dispatch_entry_nodes.contains(entry) {
                    continue;
                }
                let in_loop = cases
                    .iter()
                    .any(|c| func.nodes.contains(c) && reachable_from(&func, *c).contains(guard));
                if in_loop {
                    dispatch_entry_nodes.insert(*entry);
                }
            }
            func.dispatch_entry_nodes = dispatch_entry_nodes;

            functions.insert(func_addr, func);
        }

        PassContext {
            functions,
            var_types,
            inline_temps: HashSet::new(),
        }
    }

    fn write_back(self, db: &mut DecompileDB) {
        let new_insts = ascent::boxcar::Vec::<(Node, RTLInst)>::new();
        let new_succs = ascent::boxcar::Vec::<(Node, Node)>::new();
        let mut live_regs: HashSet<RTLReg> = HashSet::new();

        // Sort succs: func.succs is HashMap, so unsorted pushes make rtl_next/rtl_succ nondeterministic.
        for (_func_addr, func) in &self.functions {
            for (&node, inst) in &func.inst {
                new_insts.push((node, inst.clone()));
                collect_inst_regs(inst, &mut live_regs);
            }
            let mut src_nodes: Vec<Node> = func.succs.keys().copied().collect();
            src_nodes.sort();
            for src in src_nodes {
                if !func.inst.contains_key(&src) {
                    continue;
                }
                if let Some(dsts) = func.succs.get(&src) {
                    let mut sorted_dsts: Vec<Node> = dsts.clone();
                    sorted_dsts.sort();
                    for dst in sorted_dsts {
                        new_succs.push((src, dst));
                    }
                }
            }
        }

        let new_sdc = ascent::boxcar::Vec::<(RTLReg, Constant)>::new();
        for (_func_addr, func) in &self.functions {
            for (_, inst) in &func.inst {
                if let RTLInst::Iop(op, args, dst) = inst {
                    if args.is_empty() {
                        // A const store to an address-escaped slot is a real memory init, so excluding it from single_def_const stops inline_constants dropping the store and leaving the out-param uninitialized.
                        if func.escaped_slot_regs.contains(dst) {
                            continue;
                        }
                        if let Some(cst) =
                            crate::decompile::passes::cminor_pass::constant_from_operation(op)
                        {
                            new_sdc.push((*dst, cst));
                        }
                    }
                }
            }
        }
        db.rel_set("single_def_const", new_sdc);

        db.rel_set("rtl_next", new_succs.clone());
        db.rel_set("rtl_inst", new_insts);
        db.rel_set("rtl_succ", new_succs);

        let inline_vec = ascent::boxcar::Vec::<RTLReg>::new();
        // Sort to keep emit_inline_temp deterministic across runs (inline_temps is a HashSet).
        let mut inline_sorted: Vec<RTLReg> = self.inline_temps.iter().copied().collect();
        inline_sorted.sort();
        for reg in inline_sorted {
            if live_regs.contains(&reg) {
                inline_vec.push(reg);
            }
        }
        db.rel_set("emit_inline_temp", inline_vec);
    }
}

pub(crate) fn collect_inst_regs(inst: &RTLInst, regs: &mut HashSet<RTLReg>) {
    match inst {
        RTLInst::Inop => {}
        RTLInst::Iop(_, args, dst) => {
            for &a in args.iter() {
                regs.insert(a);
            }
            regs.insert(*dst);
        }
        RTLInst::Iload(_, _, args, dst) => {
            for &a in args.iter() {
                regs.insert(a);
            }
            regs.insert(*dst);
        }
        RTLInst::Istore(_, _, args, src) => {
            for &a in args.iter() {
                regs.insert(a);
            }
            regs.insert(*src);
        }
        RTLInst::Icall(_, callee, args, dst, _) => {
            if let Either::Left(r) = callee {
                regs.insert(*r);
            }
            for &a in args.iter() {
                regs.insert(a);
            }
            if let Some(d) = dst {
                regs.insert(*d);
            }
        }
        RTLInst::Itailcall(_, callee, args) => {
            if let Either::Left(r) = callee {
                regs.insert(*r);
            }
            for &a in args.iter() {
                regs.insert(a);
            }
        }
        RTLInst::Ibuiltin(_, args, res) => {
            collect_builtin_arg_regs(args, regs);
            collect_single_builtin_arg_regs(res, regs);
        }
        RTLInst::Icond(_, args, _, _) => {
            for &a in args.iter() {
                regs.insert(a);
            }
        }
        RTLInst::Ijumptable(reg, _) => {
            regs.insert(*reg);
        }
        RTLInst::Ibranch(_) => {}
        RTLInst::Ireturn(reg) => {
            regs.insert(*reg);
        }
    }
}

fn collect_builtin_arg_regs(args: &[BuiltinArg<RTLReg>], regs: &mut HashSet<RTLReg>) {
    for arg in args {
        collect_single_builtin_arg_regs(arg, regs);
    }
}

fn collect_single_builtin_arg_regs(arg: &BuiltinArg<RTLReg>, regs: &mut HashSet<RTLReg>) {
    match arg {
        BuiltinArg::BA(r) => {
            regs.insert(*r);
        }
        BuiltinArg::BASplitLong(a, b) | BuiltinArg::BAAddPtr(a, b) => {
            collect_single_builtin_arg_regs(a, regs);
            collect_single_builtin_arg_regs(b, regs);
        }
        _ => {}
    }
}

pub(crate) struct DefUseInfo {
    node_def: HashMap<Node, RTLReg>,
    node_uses: HashMap<Node, Vec<RTLReg>>,
    defs: HashMap<RTLReg, Vec<Node>>,
    uses: HashMap<RTLReg, Vec<Node>>,
}

impl DefUseInfo {
    pub(crate) fn build(func: &FunctionCFG) -> Self {
        let mut node_def: HashMap<Node, RTLReg> = HashMap::new();
        let mut node_uses: HashMap<Node, Vec<RTLReg>> = HashMap::new();
        let mut defs: HashMap<RTLReg, Vec<Node>> = HashMap::new();
        let mut uses: HashMap<RTLReg, Vec<Node>> = HashMap::new();

        for (&node, inst) in &func.inst {
            let (def, mut used) = inst_def_use(inst);
            // A mem-indirect call uses the regs that form its target-load address (the Icall callee is only a synthetic temp); without these, DSE eliminates the producing vtable Iload and the call target collapses to an uninitialized var.
            if let Some(extra) = func.mem_call_addr_uses.get(&node) {
                for &r in extra {
                    if !used.contains(&r) {
                        used.push(r);
                    }
                }
            }
            if let Some(d) = def {
                node_def.insert(node, d);
                defs.entry(d).or_default().push(node);
            }
            node_uses.insert(node, used.clone());
            for u in used {
                uses.entry(u).or_default().push(node);
            }
        }

        DefUseInfo {
            node_def,
            node_uses,
            defs,
            uses,
        }
    }
}

pub(crate) fn inst_def_use(inst: &RTLInst) -> (Option<RTLReg>, Vec<RTLReg>) {
    match inst {
        RTLInst::Inop => (None, vec![]),
        RTLInst::Iop(_, args, dst) => (Some(*dst), args.iter().copied().collect()),
        RTLInst::Iload(_, _, args, dst) => (Some(*dst), args.iter().copied().collect()),
        RTLInst::Istore(_, _, args, src) => {
            let mut used: Vec<RTLReg> = args.iter().copied().collect();
            used.push(*src);
            (None, used)
        }
        RTLInst::Icall(_, callee, args, dst, _) => {
            let mut used: Vec<RTLReg> = args.iter().copied().collect();
            if let Either::Left(r) = callee {
                used.push(*r);
            }
            (dst.as_ref().copied(), used)
        }
        RTLInst::Itailcall(_, callee, args) => {
            let mut used: Vec<RTLReg> = args.iter().copied().collect();
            if let Either::Left(r) = callee {
                used.push(*r);
            }
            (None, used)
        }
        RTLInst::Ibuiltin(_, args, res) => {
            let mut used = Vec::new();
            for a in args {
                collect_ba_uses(a, &mut used);
            }
            let def = match res {
                BuiltinArg::BA(r) => Some(*r),
                _ => None,
            };
            (def, used)
        }
        RTLInst::Icond(_, args, _, _) => (None, args.iter().copied().collect()),
        RTLInst::Ijumptable(reg, _) => (None, vec![*reg]),
        RTLInst::Ibranch(_) => (None, vec![]),
        RTLInst::Ireturn(reg) => (None, vec![*reg]),
    }
}

fn collect_ba_uses(arg: &BuiltinArg<RTLReg>, out: &mut Vec<RTLReg>) {
    match arg {
        BuiltinArg::BA(r) => out.push(*r),
        BuiltinArg::BASplitLong(a, b) | BuiltinArg::BAAddPtr(a, b) => {
            collect_ba_uses(a, out);
            collect_ba_uses(b, out);
        }
        _ => {}
    }
}

pub(crate) struct LivenessInfo {
    pub(crate) live_in: HashMap<Node, HashSet<RTLReg>>,
    pub(crate) live_out: HashMap<Node, HashSet<RTLReg>>,
}

impl LivenessInfo {
    pub(crate) fn build(func: &FunctionCFG, du: &DefUseInfo) -> Self {
        let mut live_in: HashMap<Node, HashSet<RTLReg>> = HashMap::new();
        let mut live_out: HashMap<Node, HashSet<RTLReg>> = HashMap::new();

        for &node in &func.nodes {
            live_in.insert(node, HashSet::new());
            live_out.insert(node, HashSet::new());
        }

        let mut worklist: VecDeque<Node> = func.nodes.iter().copied().collect();
        let mut in_worklist: HashSet<Node> = func.nodes.iter().copied().collect();

        while let Some(node) = worklist.pop_front() {
            in_worklist.remove(&node);

            let mut new_out = HashSet::new();
            if let Some(succs) = func.succs.get(&node) {
                for &s in succs {
                    if let Some(s_in) = live_in.get(&s) {
                        new_out.extend(s_in);
                    }
                }
            }

            let mut new_in = new_out.clone();
            if let Some(&def) = du.node_def.get(&node) {
                new_in.remove(&def);
            }
            if let Some(used) = du.node_uses.get(&node) {
                for &u in used {
                    new_in.insert(u);
                }
            }

            let old_in = match live_in.get(&node) {
                Some(li) => li,
                None => continue,
            };
            if &new_in != old_in {
                live_in.insert(node, new_in);
                live_out.insert(node, new_out);
                if let Some(preds) = func.preds.get(&node) {
                    for &p in preds {
                        if func.nodes.contains(&p) && in_worklist.insert(p) {
                            worklist.push_back(p);
                        }
                    }
                }
            } else if live_out.get(&node).map_or(true, |o| o != &new_out) {
                live_out.insert(node, new_out);
            }
        }

        LivenessInfo { live_in, live_out }
    }
}

fn reachable_from(func: &FunctionCFG, node: Node) -> HashSet<Node> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    if let Some(succs) = func.succs.get(&node) {
        for &s in succs {
            if visited.insert(s) {
                queue.push_back(s);
            }
        }
    }
    while let Some(n) = queue.pop_front() {
        if let Some(succs) = func.succs.get(&n) {
            for &s in succs {
                if visited.insert(s) {
                    queue.push_back(s);
                }
            }
        }
    }
    visited
}

// Whether `target` remains reachable from `start` when `blocked` is removed
// from the CFG.  This is the negative form of a dominance query: if a use is
// still reachable from the function entry without visiting its definition,
// that definition does not dominate the use.
fn reachable_avoiding(func: &FunctionCFG, start: Node, target: Node, blocked: Node) -> bool {
    if start == blocked {
        return false;
    }

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if node == target {
            return true;
        }
        if let Some(succs) = func.succs.get(&node) {
            for &succ in succs {
                if succ != blocked && func.nodes.contains(&succ) && visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }
    }

    false
}

fn src_not_redefined_on_paths_to_uses(
    func: &FunctionCFG,
    du: &DefUseInfo,
    copy_node: Node,
    src: RTLReg,
    dst_uses: &[Node],
) -> bool {
    let src_def_nodes = match du.defs.get(&src) {
        Some(defs) => defs,
        None => return true,
    };

    let redefs: Vec<Node> = src_def_nodes
        .iter()
        .filter(|&&n| n != copy_node)
        .copied()
        .collect();
    if redefs.is_empty() {
        return true;
    }

    let forward = reachable_from(func, copy_node);

    let reachable_redefs: Vec<Node> = redefs
        .iter()
        .filter(|n| forward.contains(n))
        .copied()
        .collect();
    if reachable_redefs.is_empty() {
        return true;
    }

    let mut backward = HashSet::new();
    let mut queue = VecDeque::new();
    for &u in dst_uses {
        if backward.insert(u) {
            queue.push_back(u);
        }
    }
    while let Some(node) = queue.pop_front() {
        if let Some(preds) = func.preds.get(&node) {
            for &p in preds {
                if backward.insert(p) {
                    queue.push_back(p);
                }
            }
        }
    }

    !reachable_redefs.iter().any(|n| backward.contains(n))
}

// True for an integer equality/inequality comparison, where `cmp x, x` is statically known (equal -> Ceq true / Cne false) regardless of value; ordered comparisons (Clt/Cle/Cgt/Cge) and float comparisons are excluded so only provably-determined branches fold.
fn const_eq_branch_taken(cond: &Condition) -> Option<bool> {
    match cond {
        Condition::Ccomp(c)
        | Condition::Ccompu(c)
        | Condition::Ccompl(c)
        | Condition::Ccomplu(c) => match c {
            Comparison::Ceq => Some(true),
            Comparison::Cne => Some(false),
            _ => None,
        },
        _ => None,
    }
}

// Constant-fold a two-register conditional branch whose operands provably hold the SAME value, then drop the unreachable nodes; equal only for the same register or an identical 0-arg constant.
fn fold_static_eq_branches(func: &mut FunctionCFG) -> bool {
    // single_def_const, enforced: a register counts as constant ONLY when that constant op is its sole def, or a multi-def register could fold a runtime branch through a non-reaching def.
    let du = DefUseInfo::build(func);
    let mut const_of: HashMap<RTLReg, Constant> = HashMap::new();
    for inst in func.inst.values() {
        if let RTLInst::Iop(op, args, dst) = inst {
            if args.is_empty() {
                if let Some(cst) =
                    crate::decompile::passes::cminor_pass::constant_from_operation(op)
                {
                    if du.defs.get(dst).is_some_and(|d| d.len() == 1) {
                        const_of.insert(*dst, cst);
                    }
                }
            }
        }
    }

    let mut folds: Vec<(Node, Node, Node)> = Vec::new();
    for (&node, inst) in &func.inst {
        if let RTLInst::Icond(cond, args, Either::Right(ifso), Either::Right(ifnot)) = inst {
            if args.len() != 2 {
                continue;
            }
            let same_value = args[0] == args[1]
                || matches!((const_of.get(&args[0]), const_of.get(&args[1])),
                    (Some(a), Some(b)) if a == b);
            if !same_value {
                continue;
            }
            match const_eq_branch_taken(cond) {
                Some(true) => folds.push((node, *ifso, *ifnot)),
                Some(false) => folds.push((node, *ifnot, *ifso)),
                None => {}
            }
        }
    }
    if folds.is_empty() {
        return false;
    }

    for &(node, taken, _dead) in &folds {
        func.inst
            .insert(node, RTLInst::Ibranch(Either::Right(taken)));
        func.succs.insert(node, vec![taken]);
    }

    // Recompute reachability from entry and drop now-unreachable nodes so the dead tail and its bogus return-value load cannot surface downstream.
    let mut reachable: HashSet<Node> = HashSet::new();
    let mut stack = vec![func.entry];
    reachable.insert(func.entry);
    while let Some(n) = stack.pop() {
        if let Some(succs) = func.succs.get(&n) {
            for &s in succs {
                if func.inst.contains_key(&s) && reachable.insert(s) {
                    stack.push(s);
                }
            }
        }
    }

    func.nodes.retain(|n| reachable.contains(n));
    func.inst.retain(|n, _| reachable.contains(n));
    func.succs.retain(|n, _| reachable.contains(n));
    for dsts in func.succs.values_mut() {
        dsts.retain(|d| reachable.contains(d));
    }
    let mut new_preds: HashMap<Node, Vec<Node>> = HashMap::new();
    for (&src, dsts) in &func.succs {
        for &dst in dsts {
            new_preds.entry(dst).or_default().push(src);
        }
    }
    func.preds = new_preds;
    true
}

pub(crate) fn self_zero_rewrite(func: &mut FunctionCFG) -> usize {
    let mut count = 0;
    for (_node, inst) in func.inst.iter_mut() {
        let rewrite = match inst {
            RTLInst::Iop(op, args, dst) if args.len() == 2 && args[0] == args[1] => match op {
                Operation::Oxor | Operation::Osub => Some(RTLInst::Iop(
                    Operation::Ointconst(0),
                    Arc::new(vec![]),
                    *dst,
                )),
                Operation::Oxorl | Operation::Osubl => Some(RTLInst::Iop(
                    Operation::Olongconst(0),
                    Arc::new(vec![]),
                    *dst,
                )),
                _ => None,
            },
            _ => None,
        };
        if let Some(new_inst) = rewrite {
            *inst = new_inst;
            count += 1;
        }
    }
    count
}

pub(crate) fn copy_propagation(
    func: &mut FunctionCFG,
    du: &DefUseInfo,
    _liveness: &LivenessInfo,
) -> (usize, HashMap<RTLReg, RTLReg>) {
    let mut candidates: Vec<(Node, RTLReg, RTLReg)> = Vec::new();

    for (&node, inst) in &func.inst {
        if let RTLInst::Iop(Operation::Omove, args, dst) = inst {
            if args.len() == 1 && args[0] != *dst {
                let src = args[0];
                candidates.push((node, src, *dst));
            }
        }
    }

    let mut subst: HashMap<RTLReg, RTLReg> = HashMap::new();
    let mut dead_nodes: HashSet<Node> = HashSet::new();

    // Pass 1: classify each Omove candidate as propagatable or not, recording the dst of every unsafe copy (notably one whose source is clobbered before a use).
    let mut unsafe_dsts: HashSet<RTLReg> = HashSet::new();
    let mut safe_list: Vec<(Node, RTLReg, RTLReg)> = Vec::new();
    for (copy_node, src, dst) in &candidates {
        let src = *src;
        let dst = *dst;
        let mut safe = !func.params.contains(&dst)
            // An escaped stack slot represents mutable memory, not an SSA
            // temporary.  Propagating either the initialization into later
            // reads or a later read back to its pre-call source would erase
            // mutations performed through an escaped address.
            && !func.escaped_slot_regs.contains(&src)
            && !func.escaped_slot_regs.contains(&dst)
            && du.defs.get(&dst).map_or(0, |d| d.len()) == 1;
        if safe {
            match du.uses.get(&dst) {
                None => safe = false,
                Some(dst_uses) => {
                    if dst_uses.iter().any(|&u| u == *copy_node) {
                        safe = false;
                    }
                    if safe {
                        let reachable = reachable_from(func, *copy_node);
                        if dst_uses.iter().any(|&u| !reachable.contains(&u)) {
                            safe = false;
                        }
                    }
                    if safe {
                        safe =
                            src_not_redefined_on_paths_to_uses(func, du, *copy_node, src, dst_uses);
                    }
                }
            }
        }
        if safe {
            safe_list.push((*copy_node, src, dst));
        } else {
            unsafe_dsts.insert(dst);
        }
    }

    // Pass 2: a copy whose SOURCE is an unsafe copy's dst must not propagate either, or a later dead-store pass drops the producer and the use reads uninitialized memory.
    for (copy_node, src, dst) in &safe_list {
        if unsafe_dsts.contains(src) {
            continue;
        }
        subst.insert(*dst, *src);
        dead_nodes.insert(*copy_node);
    }

    if subst.is_empty() {
        return (0, HashMap::new());
    }

    let mut resolved: HashMap<RTLReg, RTLReg> = HashMap::new();
    for (&dst, &src) in &subst {
        let mut target = src;
        let mut visited = HashSet::new();
        visited.insert(dst);
        while let Some(&next) = subst.get(&target) {
            if !visited.insert(next) {
                break;
            }
            target = next;
        }
        resolved.insert(dst, target);
    }

    let count = resolved.len();

    for (&node, inst) in func.inst.iter_mut() {
        if dead_nodes.contains(&node) {
            continue;
        }
        *inst = subst_in_inst(inst, &resolved);
    }

    for &node in &dead_nodes {
        if let Some(inst) = func.inst.get_mut(&node) {
            *inst = RTLInst::Inop;
        }
    }

    (count, resolved)
}

pub(crate) fn subst_in_inst(inst: &RTLInst, map: &HashMap<RTLReg, RTLReg>) -> RTLInst {
    match inst {
        RTLInst::Inop => RTLInst::Inop,
        RTLInst::Iop(op, args, dst) => {
            let new_args = subst_args(args, map);
            RTLInst::Iop(op.clone(), new_args, *dst)
        }
        RTLInst::Iload(chunk, addr, args, dst) => {
            let new_args = subst_args(args, map);
            RTLInst::Iload(*chunk, addr.clone(), new_args, *dst)
        }
        RTLInst::Istore(chunk, addr, args, src) => {
            let new_args = subst_args(args, map);
            let new_src = map.get(src).copied().unwrap_or(*src);
            RTLInst::Istore(*chunk, addr.clone(), new_args, new_src)
        }
        RTLInst::Icall(sig, callee, args, dst, next) => {
            let new_callee = match callee {
                Either::Left(r) => Either::Left(map.get(r).copied().unwrap_or(*r)),
                other => other.clone(),
            };
            let new_args = subst_args(args, map);
            RTLInst::Icall(sig.clone(), new_callee, new_args, *dst, *next)
        }
        RTLInst::Itailcall(sig, callee, args) => {
            let new_callee = match callee {
                Either::Left(r) => Either::Left(map.get(r).copied().unwrap_or(*r)),
                other => other.clone(),
            };
            let new_args = subst_args(args, map);
            RTLInst::Itailcall(sig.clone(), new_callee, new_args)
        }
        RTLInst::Ibuiltin(name, args, res) => {
            let new_args: Vec<_> = args.iter().map(|a| subst_ba(a, map)).collect();
            RTLInst::Ibuiltin(name.clone(), new_args, res.clone())
        }
        RTLInst::Icond(cond, args, ifso, ifnot) => {
            let new_args = subst_args(args, map);
            RTLInst::Icond(cond.clone(), new_args, ifso.clone(), ifnot.clone())
        }
        RTLInst::Ijumptable(reg, targets) => {
            let new_reg = map.get(reg).copied().unwrap_or(*reg);
            RTLInst::Ijumptable(new_reg, targets.clone())
        }
        RTLInst::Ibranch(target) => RTLInst::Ibranch(target.clone()),
        RTLInst::Ireturn(reg) => {
            let new_reg = map.get(reg).copied().unwrap_or(*reg);
            RTLInst::Ireturn(new_reg)
        }
    }
}

pub(crate) fn subst_args(args: &Args, map: &HashMap<RTLReg, RTLReg>) -> Args {
    Arc::new(
        args.iter()
            .map(|r| map.get(r).copied().unwrap_or(*r))
            .collect(),
    )
}

pub(crate) fn subst_ba(
    ba: &BuiltinArg<RTLReg>,
    map: &HashMap<RTLReg, RTLReg>,
) -> BuiltinArg<RTLReg> {
    match ba {
        BuiltinArg::BA(r) => BuiltinArg::BA(map.get(r).copied().unwrap_or(*r)),
        BuiltinArg::BASplitLong(a, b) => {
            BuiltinArg::BASplitLong(Box::new(subst_ba(a, map)), Box::new(subst_ba(b, map)))
        }
        BuiltinArg::BAAddPtr(a, b) => {
            BuiltinArg::BAAddPtr(Box::new(subst_ba(a, map)), Box::new(subst_ba(b, map)))
        }
        other => other.clone(),
    }
}

pub(crate) fn dead_store_elimination(
    func: &mut FunctionCFG,
    du: &DefUseInfo,
    liveness: &LivenessInfo,
) -> usize {
    let mut count = 0;

    let dead_nodes: Vec<Node> = func
        .inst
        .iter()
        .filter_map(|(&node, inst)| {
            let def_reg = du.node_def.get(&node)?;

            if func.params.contains(def_reg) {
                return None;
            }

            // A store to an address-escaped stack slot is never dead: the callee reads it through the escaped pointer, protecting every by-reference out-param's initialization.
            if func.escaped_slot_regs.contains(def_reg) {
                return None;
            }

            let live_out = liveness.live_out.get(&node)?;
            if live_out.contains(def_reg) {
                return None;
            }

            match inst {
                RTLInst::Iop(_, _, _) => Some(node),
                RTLInst::Iload(_, _, _, _) => Some(node),
                _ => None,
            }
        })
        .collect();

    for node in &dead_nodes {
        if let Some(inst) = func.inst.get_mut(node) {
            *inst = RTLInst::Inop;
            count += 1;
        }
    }

    let dead_call_nodes: Vec<Node> = func
        .inst
        .iter()
        .filter_map(|(&node, inst)| {
            if let RTLInst::Icall(_, _, _, Some(dst), _) = inst {
                if func.params.contains(dst) {
                    return None;
                }
                let live_out = liveness.live_out.get(&node)?;
                if !live_out.contains(dst) {
                    return Some(node);
                }
            }
            None
        })
        .collect();

    for node in dead_call_nodes {
        if let Some(RTLInst::Icall(sig, callee, args, _, next)) = func.inst.get(&node).cloned() {
            func.inst
                .insert(node, RTLInst::Icall(sig, callee, args, None, next));
            count += 1;
        }
    }

    count
}

pub(crate) fn nop_collapse(func: &mut FunctionCFG) -> usize {
    // An in-loop dispatch entry is DSE-nopped but not removable: keep it self-canonical, or threading the guard past the switch leaves a dead switch.
    let dispatch_entry = &func.dispatch_entry_nodes;
    let nop_nodes: Vec<Node> = func
        .inst
        .iter()
        .filter(|(&node, inst)| {
            matches!(inst, RTLInst::Inop) && node != func.entry && !dispatch_entry.contains(&node)
        })
        .map(|(&node, _)| node)
        .collect();

    if nop_nodes.is_empty() {
        return 0;
    }

    let nop_set: HashSet<Node> = nop_nodes.iter().copied().collect();

    let mut nop_target: HashMap<Node, Node> = HashMap::new();
    for &nop in &nop_nodes {
        let mut target = nop;
        let mut visited = HashSet::new();
        visited.insert(nop);
        loop {
            let succs = func.succs.get(&target);
            match succs.and_then(|s| if s.len() == 1 { Some(s[0]) } else { None }) {
                Some(next) if nop_set.contains(&next) && visited.insert(next) => {
                    target = next;
                }
                Some(next) if nop_set.contains(&next) => {
                    break;
                }
                Some(next) => {
                    nop_target.insert(nop, next);
                    break;
                }
                None => {
                    break;
                }
            }
        }
    }

    if nop_target.is_empty() {
        return 0;
    }

    let mut count = 0;

    for (&_pred, succs) in func.succs.iter_mut() {
        for succ in succs.iter_mut() {
            if let Some(&target) = nop_target.get(succ) {
                *succ = target;
            }
        }
    }

    for inst in func.inst.values_mut() {
        retarget_inst(inst, &nop_target);
    }

    for (&nop, _) in &nop_target {
        func.inst.remove(&nop);
        func.succs.remove(&nop);
        func.nodes.remove(&nop);
        count += 1;
    }

    loop {
        if !matches!(func.inst.get(&func.entry), Some(RTLInst::Inop)) {
            break;
        }
        let succs = func.succs.get(&func.entry);
        match succs.and_then(|s| if s.len() == 1 { Some(s[0]) } else { None }) {
            Some(target) if target != func.entry => {
                let old_entry = func.entry;
                func.entry = target;
                func.inst.remove(&old_entry);
                func.succs.remove(&old_entry);
                func.nodes.remove(&old_entry);
                count += 1;
            }
            _ => break,
        }
    }

    let mut preds: HashMap<Node, Vec<Node>> = HashMap::new();
    for (&src, dsts) in &func.succs {
        if func.inst.contains_key(&src) {
            for &dst in dsts {
                preds.entry(dst).or_default().push(src);
            }
        }
    }
    func.preds = preds;

    count
}

fn retarget_inst(inst: &mut RTLInst, nop_target: &HashMap<Node, Node>) {
    match inst {
        RTLInst::Icond(_, _, ifso, ifnot) => {
            retarget_either(ifso, nop_target);
            retarget_either(ifnot, nop_target);
        }
        RTLInst::Ibranch(target) => {
            retarget_either(target, nop_target);
        }
        RTLInst::Icall(_, _, _, _, next) => {
            if let Some(&target) = nop_target.get(next) {
                *next = target;
            }
        }
        RTLInst::Ijumptable(_, targets) => {
            let new_targets: Vec<Node> = targets
                .iter()
                .map(|n| nop_target.get(n).copied().unwrap_or(*n))
                .collect();
            if new_targets != **targets {
                *targets = Arc::new(new_targets);
            }
        }
        _ => {}
    }
}

pub(crate) fn retarget_either(target: &mut Either<Symbol, Node>, nop_target: &HashMap<Node, Node>) {
    if let Either::Right(node) = target {
        if let Some(&new_node) = nop_target.get(node) {
            *node = new_node;
        }
    }
}

pub(crate) fn find_inline_temps(
    func: &FunctionCFG,
    du: &DefUseInfo,
    liveness: &LivenessInfo,
) -> HashSet<RTLReg> {
    let mut result = HashSet::new();

    for (&reg, def_nodes) in &du.defs {
        if def_nodes.len() != 1 {
            continue;
        }
        let def_node = def_nodes[0];

        let use_node = match du.uses.get(&reg) {
            Some(u) => {
                let mut distinct: Vec<Node> = u.clone();
                distinct.sort_unstable();
                distinct.dedup();
                if distinct.len() != 1 {
                    continue;
                }
                distinct[0]
            }
            _ => continue,
        };

        if func.params.contains(&reg) {
            continue;
        }
        // An address-escaped canonical slot is mutable memory.  Treating its
        // sole syntactic def/use as an inline expression would move a pre-call
        // value across an opaque mutation through &slot.
        if func.escaped_slot_regs.contains(&reg) {
            continue;
        }

        let inst = match func.inst.get(&def_node) {
            Some(i) => i,
            None => continue,
        };
        // Only inline Iop temps; Iload is unsafe without aliasing/intervening store checks.
        let def_args: Vec<RTLReg> = match inst {
            RTLInst::Iop(_, args, _) => args.iter().copied().collect(),
            _ => continue,
        };

        let use_live_in = match liveness.live_in.get(&use_node) {
            Some(li) => li,
            None => continue,
        };
        if !def_args.iter().all(|a| use_live_in.contains(a)) {
            continue;
        }

        let reachable = reachable_from(func, def_node);
        if !reachable.contains(&use_node) {
            continue;
        }

        // Reachability from def to use is insufficient: a conditional def can
        // reach a join that also has a bypass edge.  Inlining such a def
        // fabricates its value on the bypass path (notably for a conditional
        // `lea stack, bp`).  Require the sole definition to dominate its use.
        if def_node != func.entry && reachable_avoiding(func, func.entry, use_node, def_node) {
            continue;
        }

        // Refuse to inline if any source operand is redefined on a CFG path from the temp's def to its use; liveness at the use does not preclude an intervening redefinition.
        if !def_args
            .iter()
            .all(|&a| src_not_redefined_on_paths_to_uses(func, du, def_node, a, &[use_node]))
        {
            continue;
        }

        result.insert(reg);
    }

    result
}

#[cfg(test)]
mod cr8_byte_compare_tests {
    use super::*;
    use crate::mreg::Mreg;

    const FUNCTION: Address = 0x1000;
    const READ: Node = 0x1010;
    const COMPARE: Node = 0x1014;
    const TAKEN: Node = 0x1020;
    const FALLTHROUGH: Node = 0x1018;
    const VALUE: RTLReg = 0x8000_0000_0000_1010;
    const SECOND_READ: Node = 0x1030;
    const SECOND_COMPARE: Node = 0x1034;
    const SECOND_VALUE: RTLReg = 0x8000_0000_0000_1030;
    const PADDING: Symbol = NO_OP;
    const CR8: Symbol = "cr8_width_cr8";
    const RAX: Symbol = "cr8_width_rax";
    const AL: Symbol = "cr8_width_al";
    const ONE: Symbol = "cr8_width_one";

    fn raw(
        node: Node,
        size: usize,
        mnemonic: &'static str,
        op1: Symbol,
        op2: Symbol,
    ) -> (
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
    ) {
        (node, size, "", mnemonic, op1, op2, PADDING, PADDING, 0, 0)
    }

    fn valid_db_at(compare: Node) -> DecompileDB {
        let condition = Condition::Ccompuimm(Comparison::Cle, 1);
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        db.rel_push("next", (READ, compare));
        db.rel_push("code_in_block", (READ, FUNCTION));
        db.rel_push("code_in_block", (compare, FUNCTION));
        db.rel_push("instr_in_function", (READ, FUNCTION));
        db.rel_push("instr_in_function", (compare, FUNCTION));
        db.rel_push("op_register", (CR8, "CR8"));
        db.rel_push("op_register", (RAX, "RAX"));
        db.rel_push("op_register", (AL, "AL"));
        db.rel_push("op_immediate", (ONE, 1_i64, 0_usize));
        db.rel_push("unrefinedinstruction", raw(READ, 4, "MOV", CR8, RAX));
        db.rel_push("instruction", raw(READ, 4, "MOV", CR8, RAX));
        db.rel_push("unrefinedinstruction", raw(compare, 2, "CMP", ONE, AL));
        db.rel_push("instruction", raw(compare, 2, "CMP", ONE, AL));
        db.rel_push(
            "ltl_inst",
            (
                READ,
                LTLInst::Lbuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(Mreg::AX)),
            ),
        );
        db.rel_push(
            "ltl_inst",
            (
                compare,
                LTLInst::Lcond(
                    condition.clone(),
                    Arc::new(vec![Mreg::AX]),
                    Either::Right(TAKEN),
                    Either::Right(FALLTHROUGH),
                ),
            ),
        );
        db.rel_push(
            "rtl_inst",
            (
                READ,
                RTLInst::Ibuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(VALUE)),
            ),
        );
        db.rel_push(
            "rtl_inst",
            (
                compare,
                RTLInst::Icond(
                    condition,
                    Arc::new(vec![VALUE]),
                    Either::Right(TAKEN),
                    Either::Right(FALLTHROUGH),
                ),
            ),
        );
        db
    }

    fn valid_db() -> DecompileDB {
        valid_db_at(COMPARE)
    }

    fn markers(db: &DecompileDB) -> BTreeSet<(Node, RTLReg)> {
        db.rel_iter::<(Node, RTLReg)>("cr8_byte_compare")
            .copied()
            .collect()
    }

    fn add_second_valid_site(db: &mut DecompileDB) {
        let condition = Condition::Ccompuimm(Comparison::Cle, 1);
        db.rel_push("next", (SECOND_READ, SECOND_COMPARE));
        db.rel_push("code_in_block", (SECOND_READ, FUNCTION));
        db.rel_push("code_in_block", (SECOND_COMPARE, FUNCTION));
        db.rel_push("instr_in_function", (SECOND_READ, FUNCTION));
        db.rel_push("instr_in_function", (SECOND_COMPARE, FUNCTION));
        db.rel_push("unrefinedinstruction", raw(SECOND_READ, 4, "MOV", CR8, RAX));
        db.rel_push("instruction", raw(SECOND_READ, 4, "MOV", CR8, RAX));
        db.rel_push(
            "unrefinedinstruction",
            raw(SECOND_COMPARE, 2, "CMP", ONE, AL),
        );
        db.rel_push("instruction", raw(SECOND_COMPARE, 2, "CMP", ONE, AL));
        db.rel_push(
            "ltl_inst",
            (
                SECOND_READ,
                LTLInst::Lbuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(Mreg::AX)),
            ),
        );
        db.rel_push(
            "ltl_inst",
            (
                SECOND_COMPARE,
                LTLInst::Lcond(
                    condition.clone(),
                    Arc::new(vec![Mreg::AX]),
                    Either::Right(TAKEN),
                    Either::Right(FALLTHROUGH),
                ),
            ),
        );
        db.rel_push(
            "rtl_inst",
            (
                SECOND_READ,
                RTLInst::Ibuiltin(
                    "__readcr8".to_string(),
                    vec![],
                    BuiltinArg::BA(SECOND_VALUE),
                ),
            ),
        );
        db.rel_push(
            "rtl_inst",
            (
                SECOND_COMPARE,
                RTLInst::Icond(
                    condition,
                    Arc::new(vec![SECOND_VALUE]),
                    Either::Right(TAKEN),
                    Either::Right(FALLTHROUGH),
                ),
            ),
        );
    }

    #[test]
    fn exact_cr8_low_byte_compare_is_materialized() {
        let mut db = valid_db();
        materialize_cr8_byte_compares(&mut db);
        assert_eq!(markers(&db), BTreeSet::from([(COMPARE, VALUE)]));
    }

    #[test]
    fn ambiguous_or_competing_evidence_fails_closed() {
        let mut wrong_abi = valid_db();
        wrong_abi.target_abi = Some(crate::abi::AbiConfig::sysv_x86_64());
        materialize_cr8_byte_compares(&mut wrong_abi);
        assert!(markers(&wrong_abi).is_empty());

        let mut competing_next = valid_db();
        competing_next.rel_push("next", (READ, FALLTHROUGH));
        materialize_cr8_byte_compares(&mut competing_next);
        assert!(markers(&competing_next).is_empty());

        let mut forged_nonadjacent_next = valid_db_at(COMPARE + 8);
        materialize_cr8_byte_compares(&mut forged_nonadjacent_next);
        assert!(markers(&forged_nonadjacent_next).is_empty());

        let mut competing_owner = valid_db();
        competing_owner.rel_push("instr_in_function", (COMPARE, 0x2000 as Address));
        materialize_cr8_byte_compares(&mut competing_owner);
        assert!(markers(&competing_owner).is_empty());

        let mut competing_register = valid_db();
        competing_register.rel_push("op_register", (AL, "CL"));
        materialize_cr8_byte_compares(&mut competing_register);
        assert!(markers(&competing_register).is_empty());

        let mut competing_immediate = valid_db();
        competing_immediate.rel_push("op_immediate", (ONE, 2_i64, 0_usize));
        materialize_cr8_byte_compares(&mut competing_immediate);
        assert!(markers(&competing_immediate).is_empty());

        let mut non_compare = valid_db();
        let non_compare_rows = vec![
            raw(READ, 4, "MOV", CR8, RAX),
            raw(COMPARE, 2, "ADD", ONE, AL),
        ];
        non_compare.rel_set(
            "unrefinedinstruction",
            non_compare_rows
                .clone()
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        non_compare.rel_set(
            "instruction",
            non_compare_rows
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        materialize_cr8_byte_compares(&mut non_compare);
        assert!(markers(&non_compare).is_empty());

        let mut extra_operand = valid_db();
        let extra_compare = (
            COMPARE, 2, "", "CMP", ONE, AL, RAX, PADDING, 0_usize, 0_usize,
        );
        let extra_operand_rows = vec![raw(READ, 4, "MOV", CR8, RAX), extra_compare];
        extra_operand.rel_set(
            "unrefinedinstruction",
            extra_operand_rows
                .clone()
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        extra_operand.rel_set(
            "instruction",
            extra_operand_rows
                .into_iter()
                .collect::<ascent::boxcar::Vec<_>>(),
        );
        materialize_cr8_byte_compares(&mut extra_operand);
        assert!(markers(&extra_operand).is_empty());

        let mut competing_indirect = valid_db();
        competing_indirect.rel_push(
            "op_indirect",
            (AL, "NONE", "RAX", "NONE", 1_i64, 0_i64, 1_usize),
        );
        materialize_cr8_byte_compares(&mut competing_indirect);
        assert!(markers(&competing_indirect).is_empty());

        let mut competing_ltl = valid_db();
        competing_ltl.rel_push("ltl_inst", (COMPARE, LTLInst::Lreturn));
        materialize_cr8_byte_compares(&mut competing_ltl);
        assert!(markers(&competing_ltl).is_empty());

        let mut mismatched_ltl_register = valid_db();
        mismatched_ltl_register.rel_set(
            "ltl_inst",
            vec![
                (
                    READ,
                    LTLInst::Lbuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(Mreg::CX)),
                ),
                (
                    COMPARE,
                    LTLInst::Lcond(
                        Condition::Ccompuimm(Comparison::Cle, 1),
                        Arc::new(vec![Mreg::CX]),
                        Either::Right(TAKEN),
                        Either::Right(FALLTHROUGH),
                    ),
                ),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        materialize_cr8_byte_compares(&mut mismatched_ltl_register);
        assert!(markers(&mismatched_ltl_register).is_empty());

        let mut competing_rtl = valid_db();
        competing_rtl.rel_push("rtl_inst", (READ, RTLInst::Inop));
        materialize_cr8_byte_compares(&mut competing_rtl);
        assert!(markers(&competing_rtl).is_empty());

        let mut duplicate_raw = valid_db();
        duplicate_raw.rel_push("unrefinedinstruction", raw(READ, 4, "MOV", CR8, RAX));
        materialize_cr8_byte_compares(&mut duplicate_raw);
        assert!(markers(&duplicate_raw).is_empty());

        let mut duplicate_decoded = valid_db();
        duplicate_decoded.rel_push("instruction", raw(READ, 4, "MOV", CR8, RAX));
        materialize_cr8_byte_compares(&mut duplicate_decoded);
        assert!(markers(&duplicate_decoded).is_empty());

        let mut duplicate_final_definition = valid_db();
        duplicate_final_definition.rel_push(
            "rtl_inst",
            (
                READ,
                RTLInst::Ibuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(VALUE)),
            ),
        );
        materialize_cr8_byte_compares(&mut duplicate_final_definition);
        assert!(markers(&duplicate_final_definition).is_empty());

        let mut redefined_value = valid_db();
        redefined_value.rel_push(
            "rtl_inst",
            (
                SECOND_READ,
                RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE + 1]), VALUE),
            ),
        );
        materialize_cr8_byte_compares(&mut redefined_value);
        assert!(markers(&redefined_value).is_empty());

        let mut value_used_after_compare = valid_db();
        value_used_after_compare.rel_push(
            "rtl_inst",
            (
                FALLTHROUGH,
                RTLInst::Iop(Operation::Omove, Arc::new(vec![VALUE]), VALUE + 1),
            ),
        );
        materialize_cr8_byte_compares(&mut value_used_after_compare);
        assert!(markers(&value_used_after_compare).is_empty());

        let mut second_incomplete_read = valid_db();
        second_incomplete_read.rel_push("code_in_block", (SECOND_READ, FUNCTION));
        second_incomplete_read.rel_push("instr_in_function", (SECOND_READ, FUNCTION));
        second_incomplete_read
            .rel_push("unrefinedinstruction", raw(SECOND_READ, 4, "MOV", CR8, RAX));
        second_incomplete_read.rel_push("instruction", raw(SECOND_READ, 4, "MOV", CR8, RAX));
        materialize_cr8_byte_compares(&mut second_incomplete_read);
        assert!(markers(&second_incomplete_read).is_empty());

        let mut repeated_sites = valid_db();
        add_second_valid_site(&mut repeated_sites);
        materialize_cr8_byte_compares(&mut repeated_sites);
        assert!(markers(&repeated_sites).is_empty());

        let mut mismatched_value = valid_db();
        mismatched_value.rel_set(
            "rtl_inst",
            vec![
                (
                    READ,
                    RTLInst::Ibuiltin("__readcr8".to_string(), vec![], BuiltinArg::BA(VALUE)),
                ),
                (
                    COMPARE,
                    RTLInst::Icond(
                        Condition::Ccompuimm(Comparison::Cle, 1),
                        Arc::new(vec![VALUE + 1]),
                        Either::Right(TAKEN),
                        Either::Right(FALLTHROUGH),
                    ),
                ),
            ]
            .into_iter()
            .collect::<ascent::boxcar::Vec<_>>(),
        );
        materialize_cr8_byte_compares(&mut mismatched_value);
        assert!(markers(&mismatched_value).is_empty());
    }
}

#[cfg(test)]
mod inline_temp_tests {
    use super::*;
    use crate::x86::op::Addressing;

    const ENTRY: Node = 0x100;
    const DEF: Node = 0x110;
    const BYPASS: Node = 0x120;
    const USE: Node = 0x130;
    const TEMP: RTLReg = 0x8000_0000_0000_0110;

    fn stack_address_cfg(with_bypass: bool) -> FunctionCFG {
        let mut nodes = BTreeSet::from([ENTRY, DEF, USE]);
        let mut inst = BTreeMap::from([
            (ENTRY, RTLInst::Inop),
            (
                DEF,
                RTLInst::Iop(
                    Operation::Olea(Addressing::Ainstack(0)),
                    Arc::new(Vec::new()),
                    TEMP,
                ),
            ),
            (USE, RTLInst::Ireturn(TEMP)),
        ]);
        let mut succs = HashMap::from([(DEF, vec![USE])]);

        if with_bypass {
            nodes.insert(BYPASS);
            inst.insert(BYPASS, RTLInst::Inop);
            succs.insert(ENTRY, vec![DEF, BYPASS]);
            succs.insert(BYPASS, vec![USE]);
        } else {
            succs.insert(ENTRY, vec![DEF]);
        }

        let mut preds: HashMap<Node, Vec<Node>> = HashMap::new();
        for (&src, destinations) in &succs {
            for &dst in destinations {
                preds.entry(dst).or_default().push(src);
            }
        }

        FunctionCFG {
            func_addr: ENTRY,
            entry: ENTRY,
            nodes,
            inst,
            succs,
            preds,
            params: HashSet::new(),
            mem_call_addr_uses: HashMap::new(),
            dispatch_entry_nodes: HashSet::new(),
            escaped_slot_regs: HashSet::new(),
        }
    }

    fn inline_temps(func: &FunctionCFG) -> HashSet<RTLReg> {
        let du = DefUseInfo::build(func);
        let liveness = LivenessInfo::build(func, &du);
        find_inline_temps(func, &du, &liveness)
    }

    #[test]
    fn per_call_contract_survives_direct_callee_arity_trimming() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        let call: Node = 0x220;
        let owner: Address = 0x200;
        let target: Address = 0x9000;
        let check: Node = 0x218;
        let ret: Node = 0x230;
        let argument: RTLReg = 0x8000_0000_0000_0218;
        let stale_float: RTLReg = 0x8000_0000_0000_0220;
        let guard_signature = Signature {
            sig_args: Arc::new(vec![XType::Xlong]),
            sig_res: XType::Xvoid,
            sig_cc: CallConv::default(),
        };
        db.rel_push(
            "emit_function_signature_candidate",
            (
                target,
                Signature {
                    sig_args: Arc::new(vec![]),
                    sig_res: XType::Xvoid,
                    sig_cc: CallConv::default(),
                },
            ),
        );
        db.rel_push("call_target_func", (call, target));
        db.rel_push("call_arg_mapping", (call, 0usize, argument));
        db.rel_push(
            "call_args_collected_candidate",
            (call, Arc::new(vec![argument])),
        );
        db.rel_push(
            "call_float_args_collected",
            (call, Arc::new(vec![stale_float])),
        );
        db.rel_push(
            "rtl_inst_candidate",
            (
                call,
                RTLInst::Icall(
                    Some(guard_signature.clone()),
                    Either::Right(Either::Left(target)),
                    Arc::new(vec![argument]),
                    None,
                    ret,
                ),
            ),
        );
        db.rel_push(
            "msvc_gs_cookie_guard_call",
            (call, owner, check, ret, argument),
        );

        trim_direct_call_args_to_callee_arity(&mut db);

        assert_eq!(
            db.rel_iter::<(Node, usize, RTLReg)>("call_arg_mapping")
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([(call, 0usize, argument)])
        );
        assert_eq!(
            db.rel_iter::<(Node, usize, RTLReg)>("call_arg")
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([(call, 0usize, argument)])
        );
        assert_eq!(
            db.rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_args_collected_candidate")
                .map(|(node, args)| (*node, args.as_ref().clone()))
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([(call, vec![argument])])
        );
        assert_eq!(
            db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter_map(|(node, inst)| {
                    (*node == call).then_some(inst).and_then(|inst| match inst {
                        RTLInst::Icall(signature, _, args, destination, _) => Some((
                            signature.clone(),
                            args.as_ref().clone(),
                            *destination,
                        )),
                        _ => None,
                    })
                })
                .collect::<Vec<_>>(),
            vec![(Some(guard_signature), vec![argument], None)]
        );
        assert!(RTLOptimizePass
            .inputs()
            .contains(&"msvc_gs_cookie_guard_call"));
    }

    #[test]
    fn conditional_stack_address_def_does_not_inline_across_bypass() {
        let func = stack_address_cfg(true);
        assert!(!inline_temps(&func).contains(&TEMP));
    }

    #[test]
    fn dominating_stack_address_def_remains_inlineable() {
        let func = stack_address_cfg(false);
        assert!(inline_temps(&func).contains(&TEMP));
    }
}
