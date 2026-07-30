use manifold::abi::{AbiConfig, BinaryFormat};

#[test]
fn amd64_coff_retains_the_complete_win64_stack_contract() {
    let mut coff = AbiConfig::win64();
    coff.format = BinaryFormat::Coff;

    assert!(coff.uses_shared_arg_slots());
    assert_eq!(coff.first_stack_arg_position(), 4);
    assert_eq!(coff.outgoing_stack_arg_base(), 32);
    assert_eq!(coff.incoming_sp_stack_arg_base(), 40);
    assert_eq!(coff.incoming_bp_stack_arg_base(), 48);
}

#[test]
fn sysv_amd64_keeps_independent_argument_sequences() {
    let sysv = AbiConfig::sysv_x86_64();

    assert!(!sysv.uses_shared_arg_slots());
    assert_eq!(sysv.first_stack_arg_position(), 6);
    assert_eq!(sysv.outgoing_stack_arg_base(), 0);
    assert_eq!(sysv.incoming_sp_stack_arg_base(), 8);
    assert_eq!(sysv.incoming_bp_stack_arg_base(), 16);
}
