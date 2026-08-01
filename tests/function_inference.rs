use manifold::decompile::disassembly::function::infer_functions;
use manifold::decompile::disassembly::instruction::DecodedInsn;
use manifold::decompile::elevator::DecompileDB;
use manifold::x86::types::{Address, Symbol};

fn instruction(address: Address, size: usize) -> DecodedInsn {
    DecodedInsn {
        address,
        size,
        mnemonic: "NOP",
        op_str: "",
        interrupt_vector: None,
    }
}

#[test]
fn direct_calls_only_seed_decoded_instruction_starts() {
    let mut db = DecompileDB::default();
    db.rel_push("direct_call", (0x1000_u64, 0x2000_u64));
    db.rel_push("direct_call", (0x1001_u64, 0x2001_u64));

    infer_functions(
        &mut db,
        &[instruction(0x1000, 1), instruction(0x2000, 2)],
    );

    let entries: Vec<_> = db
        .rel_iter::<(Symbol, Address)>("func_entry")
        .map(|(_, address)| *address)
        .collect();
    assert!(
        entries.contains(&0x2000),
        "a direct call to a decoded instruction start was not admitted: {entries:#x?}"
    );
    assert!(
        !entries.contains(&0x2001),
        "a direct call into the middle of an instruction became a function entry: {entries:#x?}"
    );
}
