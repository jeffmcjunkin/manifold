use either::Either;
use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::disassembly::operand::NO_OP;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::mreg::Mreg;
use manifold::x86::op::Addressing;
use manifold::x86::types::{Address, LTLInst, Node, RTLInst, RTLReg, Symbol, XType};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> PathBuf {
    assert!(
        command_exists("clang"),
        "MSVC /GS integration prerequisite missing: clang is not available"
    );
    let directory =
        std::env::temp_dir().join(format!("manifold_msvc_gs_cookie_{}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create MSVC /GS fixture directory");
    let source = directory.join("fixture.s");
    let object = directory.join("fixture.obj");
    std::fs::write(
        &source,
        r#"
        .data
        .p2align 3
        .globl arbitrary_process_cookie
arbitrary_process_cookie:
        .quad 0x4a6f79216e6f7421

        .p2align 3
arbitrary_return_double:
        .double 3.5

        .p2align 2
        .globl arbitrary_result_sink
arbitrary_result_sink:
        .long 0

        .text
        .extern opaque_guard_alpha
        .extern opaque_guard_void
        .extern opaque_guard_multi
        .extern opaque_guard_vs2013
        .extern opaque_guard_thunk_target
        .extern opaque_guard_lea
        .extern opaque_guard_rbp
        .extern opaque_guard_stale_xmm
        .extern opaque_guard_float
        .extern opaque_guard_alias
        .extern opaque_guard_frame_low
        .extern opaque_guard_frame_high
        .extern competing_slot_target
        .extern invalid_restore_target
        .extern partial_terminal_target
        .extern partial_nonglobal_target
        .extern ordinary_value_target

        .globl gs_nonvoid
        .def gs_nonvoid; .scl 2; .type 32; .endef
gs_nonvoid:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        nop
        leal 7(%rcx), %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_alpha
        addq $56, %rsp
        retq

        .globl gs_void
        .def gs_void; .scl 2; .type 32; .endef
gs_void:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_void
        addq $56, %rsp
        retq

        .globl gs_multiarm
        .def gs_multiarm; .scl 2; .type 32; .endef
gs_multiarm:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        testl %edx, %edx
        je .Lgs_multi_zero
        leal 19(%rcx), %eax
        movl %eax, arbitrary_result_sink(%rip)
        jmp .Lgs_multi_join
.Lgs_multi_zero:
        leal -3(%rcx), %eax
        movl %eax, arbitrary_result_sink(%rip)
.Lgs_multi_join:
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_multi
        addq $56, %rsp
        retq

        # The exact VS2013 shape from the motivating COFF provider: incoming
        # home stores, an 0x88-byte frame, cookie at +0x70, merged AX result,
        # RCX reload/XOR, relocation-backed call, restore, RET.
        .globl gs_vs2013_shape
        .def gs_vs2013_shape; .scl 2; .type 32; .endef
gs_vs2013_shape:
        movl %edx, 16(%rsp)
        movq %rcx, 8(%rsp)
        subq $136, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 112(%rsp)
        testl %edx, %edx
        je .Lgs_vs_zero
        movl $1, %eax
        jmp .Lgs_vs_join
.Lgs_vs_zero:
        movl $0, %eax
.Lgs_vs_join:
        movq 112(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_vs2013
        addq $136, %rsp
        retq

        .globl gs_lea_restore
        .def gs_lea_restore; .scl 2; .type 32; .endef
gs_lea_restore:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leal 23(%rcx), %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_lea
        leaq 56(%rsp), %rsp
        retq

        # The spill and reload name the same entry-frame cell through different
        # architectural bases. Normalized reaching stack-slot identity is the
        # authority for this equivalence.
        .globl gs_rbp_equivalent
        .def gs_rbp_equivalent; .scl 2; .type 32; .endef
gs_rbp_equivalent:
        pushq %rbp
        movq %rsp, %rbp
        subq $48, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leal 29(%rcx), %eax
        movq -8(%rbp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_rbp
        movq %rbp, %rsp
        popq %rbp
        retq

        # Stale shared-slot XMM0 evidence must never compete with the witnessed
        # RCX XOR destination for guard argument position zero.
        .globl gs_stale_xmm
        .def gs_stale_xmm; .scl 2; .type 32; .endef
gs_stale_xmm:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leal 37(%rcx), %eax
        movq %rdx, %xmm0
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_stale_xmm
        addq $56, %rsp
        retq

        # The MSVC cookie helper is transparent to the architectural return
        # registers. A pre-guard floating result in XMM0 must therefore reach
        # RET just like the integer result in RAX.
        .globl gs_float_return
        .def gs_float_return; .scl 2; .type 32; .endef
gs_float_return:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movsd arbitrary_return_double(%rip), %xmm0
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_float
        addq $56, %rsp
        retq

        # A decoded write through a uniquely reaching RSP alias is provably
        # disjoint from the qword cookie slot and must not cause a false veto.
        .globl gs_alias_disjoint
        .def gs_alias_disjoint; .scl 2; .type 32; .endef
gs_alias_disjoint:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movq %rsp, %r11
        movq %r8, 24(%r11)
        leal 43(%rcx), %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_alias
        addq $56, %rsp
        retq

        # The qword may touch either edge of the authenticated allocation,
        # but it must remain wholly inside the half-open entry-SP interval.
        .globl gs_frame_lower_boundary
        .def gs_frame_lower_boundary; .scl 2; .type 32; .endef
gs_frame_lower_boundary:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 0(%rsp)
        leal 47(%rcx), %eax
        movq 0(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_frame_low
        addq $56, %rsp
        retq

        .globl gs_frame_upper_boundary
        .def gs_frame_upper_boundary; .scl 2; .type 32; .endef
gs_frame_upper_boundary:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 48(%rsp)
        leal 53(%rcx), %eax
        movq 48(%rsp), %rcx
        xorq %rsp, %rcx
        callq opaque_guard_frame_high
        addq $56, %rsp
        retq

        # Same-section local target: the assembler resolves this CALL directly
        # and emits no relocation for the call itself.
        .globl gs_direct_local
        .def gs_direct_local; .scl 2; .type 32; .endef
gs_direct_local:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leal 41(%rcx), %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq guard_direct_local
        addq $56, %rsp
        retq

        .globl guard_direct_local
        .def guard_direct_local; .scl 2; .type 32; .endef
guard_direct_local:
        retq

        .section .text$gs_thunk_caller,"xr"
        .globl gs_thunk_relocation
        .def gs_thunk_relocation; .scl 2; .type 32; .endef
gs_thunk_relocation:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leal 31(%rcx), %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq guard_relocation_alias
        addq $56, %rsp
        retq

        # A distinct COFF text section forces a relocation to the same-object
        # forwarding thunk instead of an address-adjacent direct-call shortcut.
        .section .text$gs_thunk_target,"xr"
        .globl guard_relocation_thunk
        .def guard_relocation_thunk; .scl 2; .type 32; .endef
guard_relocation_thunk:
        jmp guard_relocation_middle

        # A true same-address COFF alias plus a second local forwarding hop.
        .globl guard_relocation_alias
        .set guard_relocation_alias, guard_relocation_thunk

        .section .text$gs_thunk_middle,"xr"
        .globl guard_relocation_middle
        .def guard_relocation_middle; .scl 2; .type 32; .endef
guard_relocation_middle:
        jmp opaque_guard_thunk_target

        .section .text$gs_negative_controls,"xr"
        .globl competing_cookie_slot
        .def competing_cookie_slot; .scl 2; .type 32; .endef
competing_cookie_slot:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $5, %eax
        testl %edx, %edx
        je .Lcompeting_join
        movq %r8, 40(%rsp)
.Lcompeting_join:
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl partial_overlap_byte
        .def partial_overlap_byte; .scl 2; .type 32; .endef
partial_overlap_byte:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movb %r8b, 47(%rsp)
        movl $14, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl partial_overlap_word
        .def partial_overlap_word; .scl 2; .type 32; .endef
partial_overlap_word:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movw %r8w, 39(%rsp)
        movl $15, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl partial_overlap_dword
        .def partial_overlap_dword; .scl 2; .type 32; .endef
partial_overlap_dword:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl %r8d, 44(%rsp)
        movl $16, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl alternate_alias_overlap
        .def alternate_alias_overlap; .scl 2; .type 32; .endef
alternate_alias_overlap:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movq %rsp, %r11
        movb %r8b, 47(%r11)
        movl $17, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl unproven_rbp_overlap
        .def unproven_rbp_overlap; .scl 2; .type 32; .endef
unproven_rbp_overlap:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movb %r8b, 47(%rbp)
        movl $18, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl unknown_base_write
        .def unknown_base_write; .scl 2; .type 32; .endef
unknown_base_write:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movb %r8b, (%r10)
        movl $19, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # A balanced stack excursion can still overwrite the saved cookie:
        # PUSH writes [RSP-8,RSP), even though the later ADD restores RSP.
        .globl balanced_push_overlap
        .def balanced_push_overlap; .scl 2; .type 32; .endef
balanced_push_overlap:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        addq $48, %rsp
        pushq %r8
        addq $8, %rsp
        subq $48, %rsp
        movl $20, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # CALL's implicit return-address write overlaps the same cell while
        # preserving RSP across the call/return pair.
        .globl balanced_call_overlap
        .def balanced_call_overlap; .scl 2; .type 32; .endef
balanced_call_overlap:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        addq $48, %rsp
        callq balanced_overlap_helper
        subq $48, %rsp
        movl $21, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl balanced_overlap_helper
        .def balanced_overlap_helper; .scl 2; .type 32; .endef
balanced_overlap_helper:
        retq

        # Even when CALL's pushed return address is disjoint from the cookie,
        # the called body has unknown memory effects and can clobber it.
        .globl body_call_unknown_clobber
        .def body_call_unknown_clobber; .scl 2; .type 32; .endef
body_call_unknown_clobber:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        callq body_call_helper
        movl $28, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl body_call_helper
        .def body_call_helper; .scl 2; .type 32; .endef
body_call_helper:
        retq

        # A branch can enter at the physical prologue chain's XOR and bypass
        # the cookie load. Physical adjacency alone must not authenticate it.
        .globl branch_entry_bypass
        .def branch_entry_bypass; .scl 2; .type 32; .endef
branch_entry_bypass:
        subq $56, %rsp
        testl %edx, %edx
        jne .Lbranch_bypass_load
        movq arbitrary_process_cookie(%rip), %rax
.Lbranch_bypass_load:
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $29, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # Known-noreturn calls have no live fallthrough, even if dead restore
        # bytes and RET happen to follow in physical address order.
        .globl dead_noreturn_epilogue
        .def dead_noreturn_epilogue; .scl 2; .type 32; .endef
dead_noreturn_epilogue:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $30, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq dead_guard_sink
        addq $56, %rsp
        retq

        .globl dead_guard_sink
        .def dead_guard_sink; .scl 2; .type 32; .endef
dead_guard_sink:
        ud2

        # String stores have an implicit destination and are conservatively a
        # veto even when the decoder emits no ordinary memory-write operand.
        .globl implicit_string_overlap
        .def implicit_string_overlap; .scl 2; .type 32; .endef
implicit_string_overlap:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        leaq 40(%rsp), %rdi
        movq %r8, %rax
        stosq
        movl $22, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # SETSSBSY has no explicit memory operand, but its architectural shadow
        # stack update is an implicit memory effect. A synthetic fallthrough
        # Lbranch must never count as proof that the cookie remained intact.
        .globl implicit_setssbsy_effect
        .def implicit_setssbsy_effect; .scl 2; .type 32; .endef
implicit_setssbsy_effect:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        .byte 0xf3, 0x0f, 0x01, 0xe8
        movl $31, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # SYSCALL is an opaque returning transfer and implicitly clobbers RCX
        # and R11. The later store must not reuse the pre-SYSCALL R11=RSP alias,
        # and the opaque node itself must fail the effect audit.
        .globl implicit_syscall_alias_clobber
        .def implicit_syscall_alias_clobber; .scl 2; .type 32; .endef
implicit_syscall_alias_clobber:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movq %rsp, %r11
        syscall
        movq %r8, 24(%r11)
        movl $32, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # A base-less operand is not global when FS/GS supplies its runtime
        # base. Both segment overrides therefore fail closed.
        .globl fs_baseless_write
        .def fs_baseless_write; .scl 2; .type 32; .endef
fs_baseless_write:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movb %r8b, %fs:0
        movl $23, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl gs_baseless_write
        .def gs_baseless_write; .scl 2; .type 32; .endef
gs_baseless_write:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movb %r8b, %gs:0
        movl $24, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        # Qwords crossing below the allocation, crossing entry RSP, or wholly
        # occupying the incoming return-address cell are not frame cookies.
        .globl cookie_below_frame
        .def cookie_below_frame; .scl 2; .type 32; .endef
cookie_below_frame:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, -1(%rsp)
        movl $25, %eax
        movq -1(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl cookie_crosses_entry_rsp
        .def cookie_crosses_entry_rsp; .scl 2; .type 32; .endef
cookie_crosses_entry_rsp:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 52(%rsp)
        movl $26, %eax
        movq 52(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl cookie_in_return_address
        .def cookie_in_return_address; .scl 2; .type 32; .endef
cookie_in_return_address:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 56(%rsp)
        movl $27, %eax
        movq 56(%rsp), %rcx
        xorq %rsp, %rcx
        callq competing_slot_target
        addq $56, %rsp
        retq

        .globl invalid_restore_add
        .def invalid_restore_add; .scl 2; .type 32; .endef
invalid_restore_add:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $6, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq invalid_restore_target
        addq $48, %rsp
        retq

        .globl invalid_restore_lea
        .def invalid_restore_lea; .scl 2; .type 32; .endef
invalid_restore_lea:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $7, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq invalid_restore_target
        leaq 64(%rsp), %rsp
        retq

        .globl invalid_restore_mov
        .def invalid_restore_mov; .scl 2; .type 32; .endef
invalid_restore_mov:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $8, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq invalid_restore_target
        movq %rbp, %rsp
        retq

        .globl invalid_restore_leave
        .def invalid_restore_leave; .scl 2; .type 32; .endef
invalid_restore_leave:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $10, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq invalid_restore_target
        leaveq
        retq

        .globl partial_wrong_slot
        .def partial_wrong_slot; .scl 2; .type 32; .endef
partial_wrong_slot:
        subq $56, %rsp
        movq arbitrary_process_cookie(%rip), %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $9, %eax
        movq 32(%rsp), %rcx
        xorq %rsp, %rcx
        callq partial_terminal_target
        addq $56, %rsp
        retq

        .globl partial_no_global_load
        .def partial_no_global_load; .scl 2; .type 32; .endef
partial_no_global_load:
        subq $56, %rsp
        movq %rdx, %rax
        xorq %rsp, %rax
        movq %rax, 40(%rsp)
        movl $13, %eax
        movq 40(%rsp), %rcx
        xorq %rsp, %rcx
        callq partial_nonglobal_target
        addq $56, %rsp
        retq

        .globl ordinary_terminal_call
        .def ordinary_terminal_call; .scl 2; .type 32; .endef
ordinary_terminal_call:
        subq $40, %rsp
        callq ordinary_value_target
        addq $40, %rsp
        retq

        # Ordinary use of the same internal thunk targeted by a real guard.
        # Its node is not entitled to the guard-only callsite rewrite.
        .globl ordinary_same_guard_target
        .def ordinary_same_guard_target; .scl 2; .type 32; .endef
ordinary_same_guard_target:
        subq $40, %rsp
        callq guard_relocation_thunk
        movl $77, %eax
        addq $40, %rsp
        retq

        # Conflicting ordinary evidence at the same local address: two input
        # registers and a consumed return. It must retain an independent
        # declaration while the structural /GS call uses an exact cast.
        .globl ordinary_same_guard_target_conflict
        .def ordinary_same_guard_target_conflict; .scl 2; .type 32; .endef
ordinary_same_guard_target_conflict:
        subq $40, %rsp
        movl $11, %ecx
        movl $12, %edx
        callq guard_relocation_thunk
        addl $5, %eax
        addq $40, %rsp
        retq

        # Ordinary non-guard evidence for the exact same external target used
        # by gs_nonvoid. Only the structurally authenticated call may acquire
        # the void(long) cast.
        .globl ordinary_same_external_target
        .def ordinary_same_external_target; .scl 2; .type 32; .endef
ordinary_same_external_target:
        subq $40, %rsp
        movl $21, %ecx
        movl $22, %edx
        callq opaque_guard_alpha
        addl $3, %eax
        addq $40, %rsp
        retq
"#,
    )
    .expect("write MSVC /GS fixture assembly");
    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("run clang over MSVC /GS fixture");
    assert!(status.success(), "MSVC /GS fixture assembly failed");
    object
}

fn fixture_object() -> &'static Path {
    static OBJECT: OnceLock<PathBuf> = OnceLock::new();
    OBJECT.get_or_init(build_fixture).as_path()
}

fn run_to_linear(object: &Path) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    CanaryVlaPass.run(&mut db);
    AsmPass.run(&mut db);
    StackAnalysisPass.run(&mut db);
    MachPass.run(&mut db);
    LinearPass.run(&mut db);
    db
}

fn run_through_rtl(object: &Path) -> DecompileDB {
    let mut db = run_to_linear(object);
    RTLPass.run(&mut db);
    db
}

fn on_pipeline_stack(test: impl FnOnce() + Send + 'static) {
    let result = std::thread::Builder::new()
        .name("msvc-gs-cookie-test".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(test)
        .expect("spawn MSVC /GS cookie test")
        .join();
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn function_starts(db: &DecompileDB) -> BTreeMap<String, Address> {
    let mut starts: BTreeMap<String, Address> = db
        .rel_iter::<(Symbol, Address, Address)>("func_span")
        .map(|(name, start, _)| (name.trim_start_matches("coff_fn_").to_string(), *start))
        .collect();
    for (address, name, _) in db.rel_iter::<(Address, Symbol, Symbol)>("symbols") {
        starts
            .entry(name.trim_start_matches("coff_fn_").to_string())
            .or_insert(*address);
    }
    starts
}

fn calls_in_function(db: &DecompileDB, function: Address) -> Vec<Node> {
    let members: BTreeSet<Node> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .filter_map(|(node, owner)| (*owner == function).then_some(*node))
        .collect();
    let mut calls: Vec<Node> = db
        .rel_iter::<(Node, LTLInst)>("ltl_inst")
        .filter_map(|(node, inst)| {
            (members.contains(node) && matches!(inst, LTLInst::Lcall(_))).then_some(*node)
        })
        .collect();
    calls.sort_unstable();
    calls.dedup();
    calls
}

type GuardFact = (Node, Address, Node, Node, RTLReg);
type RawRow = (
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

fn mnemonic_nodes_in_function(
    db: &DecompileDB,
    function: Address,
    mnemonic: &str,
) -> BTreeSet<Node> {
    let members: BTreeSet<Node> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .filter_map(|(node, owner)| (*owner == function).then_some(*node))
        .collect();
    db.rel_iter::<RawRow>("unrefinedinstruction")
        .filter_map(|row| (members.contains(&row.0) && row.3 == mnemonic).then_some(row.0))
        .collect()
}

fn guard_facts(db: &DecompileDB) -> Vec<GuardFact> {
    db.rel_iter::<GuardFact>("msvc_gs_cookie_guard_call")
        .copied()
        .collect()
}

fn assert_exact_guard_call(db: &DecompileDB, fact: GuardFact) {
    let (call, _, check_xor, _, argument) = fact;
    assert!(
        !db.rel_iter::<(Node, RTLReg)>("call_return_reg")
            .any(|(node, _)| *node == call),
        "guard call retained a return register at {call:#x}"
    );
    let mapped: BTreeSet<(usize, RTLReg)> = db
        .rel_iter::<(Node, usize, RTLReg)>("call_arg_mapping")
        .filter_map(|(node, position, value)| (*node == call).then_some((*position, *value)))
        .collect();
    assert_eq!(mapped, BTreeSet::from([(0usize, argument)]));

    let exact_defs: BTreeSet<RTLReg> = db
        .rel_iter::<(Node, RTLReg)>("is_def")
        .filter_map(|(node, value)| (*node == check_xor).then_some(*value))
        .collect();
    let xor_defs: BTreeSet<RTLReg> = db
        .rel_iter::<(Node, Mreg, RTLReg)>("reg_xtl")
        .filter_map(|(node, reg, value)| {
            (*node == check_xor && *reg == Mreg::CX && exact_defs.contains(value)).then_some(*value)
        })
        .collect();
    assert_eq!(xor_defs, BTreeSet::from([argument]));

    let candidates: Vec<&RTLInst> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(node, inst)| {
            (*node == call && matches!(inst, RTLInst::Icall(..))).then_some(inst)
        })
        .collect();
    assert!(
        !candidates.is_empty(),
        "missing call candidate at {call:#x}"
    );
    for candidate in candidates {
        let RTLInst::Icall(Some(signature), _, arguments, destination, _) = candidate else {
            panic!("guard call lacks an explicit signature: {candidate:?}");
        };
        assert_eq!(signature.sig_args.as_slice(), &[XType::Xlong]);
        assert_eq!(signature.sig_res, XType::Xvoid);
        assert_eq!(arguments.as_slice(), &[argument]);
        assert_eq!(*destination, None);
    }
}

fn assert_pre_call_ax_is_returned(db: &DecompileDB, fact: GuardFact) {
    let (call, function, _, ret, _) = fact;
    let setup_defs: BTreeSet<Node> = db
        .rel_iter::<(Node,)>("msvc_gs_cookie_setup_ax_def")
        .map(|(node,)| *node)
        .collect();
    let body_defs: BTreeSet<Node> = db
        .rel_iter::<(Node, Node, Mreg)>("def_reaches_return")
        .filter_map(|(return_node, def, reg)| {
            (*return_node == ret && *reg == Mreg::AX && *def != call && !setup_defs.contains(def))
                .then_some(*def)
        })
        .collect();
    assert!(
        !body_defs.is_empty(),
        "function {function:#x} has no pre-call AX definition reaching its return"
    );
    let body_values: BTreeSet<RTLReg> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(node, inst)| {
            body_defs
                .contains(node)
                .then_some(inst)
                .and_then(|inst| match inst {
                    RTLInst::Iop(_, _, destination) | RTLInst::Iload(_, _, _, destination) => {
                        Some(*destination)
                    }
                    _ => None,
                })
        })
        .collect();
    let returned: BTreeSet<RTLReg> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(node, inst)| {
            (*node == ret).then_some(inst).and_then(|inst| match inst {
                RTLInst::Ireturn(value) => Some(*value),
                _ => None,
            })
        })
        .collect();
    assert_eq!(returned.len(), 1);
    assert_eq!(
        body_values, returned,
        "guard call did not preserve the pre-call AX identity in {function:#x}"
    );
}

fn assert_branch_local_ax_uses_follow_return(db: &DecompileDB, fact: GuardFact) {
    let (_, function, _, ret, _) = fact;
    let returned: BTreeSet<RTLReg> = db
        .rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(node, inst)| {
            (*node == ret).then_some(inst).and_then(|inst| match inst {
                RTLInst::Ireturn(value) => Some(*value),
                _ => None,
            })
        })
        .collect();
    assert_eq!(returned.len(), 1);

    let global_stores: BTreeSet<(Node, RTLReg)> = db
        .rel_iter::<(Node, Address)>("instr_in_function")
        .filter(|(_, owner)| *owner == function)
        .flat_map(|(node, _)| {
            db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter_map(move |(candidate_node, inst)| {
                    (*candidate_node == *node)
                        .then_some(inst)
                        .and_then(|inst| match inst {
                            RTLInst::Istore(_, Addressing::Aglobal(_, _), _, source) => {
                                Some((*node, *source))
                            }
                            _ => None,
                        })
                })
        })
        .collect();
    let physical_store_sites: BTreeSet<Node> = global_stores
        .iter()
        .map(|(node, _)| *node & !((1u64 << 62) | (1u64 << 63)))
        .collect();
    assert_eq!(
        physical_store_sites.len(),
        2,
        "fixture did not retain both physical branch-local AX consumers in {function:#x}: \
         {global_stores:#x?}"
    );
    assert_eq!(
        global_stores
            .iter()
            .map(|(_, source)| *source)
            .collect::<BTreeSet<_>>(),
        returned,
        "branch-local AX consumers retained stale identities in {function:#x}"
    );
}

fn assert_pre_call_x0_is_returned(db: &DecompileDB, fact: GuardFact) {
    let (call, function, _, ret, _) = fact;
    let body_def = mnemonic_nodes_in_function(db, function, "MOVSD")
        .into_iter()
        .next()
        .expect("fixture floating return definition");
    assert!(
        db.rel_iter::<(Node, Node, Mreg)>("def_reaches_return")
            .any(|(return_node, def, reg)| {
                *return_node == ret && *def == body_def && *reg == Mreg::X0
            }),
        "guard call did not preserve the pre-call X0 definition in {function:#x}"
    );
    assert!(
        !db.rel_iter::<(Address, Mreg)>("asm_effective_def")
            .any(|(node, reg)| *node == call && *reg == Mreg::X0),
        "guard call remained an effective X0 definition"
    );
    assert!(
        !db.rel_iter::<(Address, Mreg)>("asm_def_kill")
            .any(|(node, reg)| *node == call && *reg == Mreg::X0),
        "guard call remained an X0 kill"
    );
    assert!(
        db.rel_iter::<(Address,)>("func_x0_def_reaches_return")
            .any(|(owner,)| *owner == function),
        "floating return classification was lost in {function:#x}"
    );
}

fn linear_guard_sites(db: &DecompileDB, function: Address) -> (Node, Node) {
    let call = calls_in_function(db, function)
        .into_iter()
        .next()
        .expect("fixture guard call");
    let check = db
        .rel_iter::<(Node, Node)>("next")
        .filter_map(|(src, dst)| {
            (*dst == call
                && db
                    .rel_iter::<(Node, Symbol, Symbol)>("pxor")
                    .any(|(xor, _, _)| xor == src))
            .then_some(*src)
        })
        .next()
        .expect("fixture check XOR");
    (check, call)
}

fn assert_mutated_guard_rejected(
    object: &Path,
    mutate: impl FnOnce(&mut DecompileDB, Address, Node, Node),
) {
    let mut db = run_to_linear(object);
    let starts = function_starts(&db);
    let function = starts["gs_nonvoid"];
    let (check, call) = linear_guard_sites(&db, function);
    mutate(&mut db, function, check, call);
    RTLPass.run(&mut db);
    assert!(
        !guard_facts(&db)
            .iter()
            .any(|(_, owner, _, _, _)| *owner == function),
        "ambiguous or incomplete fixture was accepted"
    );
}

#[test]
fn structural_external_guards_are_exact_per_call_and_ax_transparent() {
    on_pipeline_stack(|| {
        let object = fixture_object();
        let db = run_through_rtl(object);
        let starts = function_starts(&db);
        let facts = guard_facts(&db);

        assert_eq!(
            mnemonic_nodes_in_function(&db, starts["gs_nonvoid"], "NOP").len(),
            1,
            "positive NOP control was not decoded"
        );
        assert_eq!(
            mnemonic_nodes_in_function(&db, starts["implicit_setssbsy_effect"], "SETSSBSY",).len(),
            1,
            "SETSSBSY adversarial control was not decoded"
        );
        let syscall_nodes =
            mnemonic_nodes_in_function(&db, starts["implicit_syscall_alias_clobber"], "SYSCALL");
        assert_eq!(
            syscall_nodes.len(),
            1,
            "SYSCALL adversarial control was not decoded"
        );

        let positive = [
            "gs_nonvoid",
            "gs_void",
            "gs_multiarm",
            "gs_vs2013_shape",
            "gs_lea_restore",
            "gs_rbp_equivalent",
            "gs_stale_xmm",
            "gs_float_return",
            "gs_alias_disjoint",
            "gs_frame_lower_boundary",
            "gs_frame_upper_boundary",
        ];
        assert_eq!(
            facts.len(),
            positive.len(),
            "unexpected guard facts: {facts:#x?}"
        );
        for name in positive {
            let function = starts[name];
            let rows: Vec<GuardFact> = facts
                .iter()
                .copied()
                .filter(|(_, owner, _, _, _)| *owner == function)
                .collect();
            assert_eq!(rows.len(), 1, "{name} did not emit one exact fact");
            assert_exact_guard_call(&db, rows[0]);
            if matches!(name, "gs_nonvoid" | "gs_multiarm" | "gs_vs2013_shape") {
                assert_pre_call_ax_is_returned(&db, rows[0]);
            }
            if name == "gs_multiarm" {
                assert_branch_local_ax_uses_follow_return(&db, rows[0]);
            }
            if name == "gs_float_return" {
                assert_pre_call_x0_is_returned(&db, rows[0]);
            }
        }

        let negative = [
            "gs_direct_local",
            "gs_thunk_relocation",
            "competing_cookie_slot",
            "partial_overlap_byte",
            "partial_overlap_word",
            "partial_overlap_dword",
            "alternate_alias_overlap",
            "unproven_rbp_overlap",
            "unknown_base_write",
            "balanced_push_overlap",
            "balanced_call_overlap",
            "body_call_unknown_clobber",
            "branch_entry_bypass",
            "dead_noreturn_epilogue",
            "implicit_string_overlap",
            "implicit_setssbsy_effect",
            "implicit_syscall_alias_clobber",
            "fs_baseless_write",
            "gs_baseless_write",
            "cookie_below_frame",
            "cookie_crosses_entry_rsp",
            "cookie_in_return_address",
            "invalid_restore_add",
            "invalid_restore_lea",
            "invalid_restore_mov",
            "invalid_restore_leave",
            "partial_wrong_slot",
            "partial_no_global_load",
            "ordinary_terminal_call",
            "ordinary_same_guard_target",
            "ordinary_same_guard_target_conflict",
            "ordinary_same_external_target",
        ];
        for name in negative {
            let function = starts[name];
            assert!(
                !facts.iter().any(|(_, owner, _, _, _)| *owner == function),
                "{name} was incorrectly authenticated"
            );
        }

        let ordinary = calls_in_function(&db, starts["ordinary_terminal_call"])[0];
        assert!(
            db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, _)| *node == ordinary)
                .any(|(_, inst)| matches!(inst, RTLInst::Icall(_, _, _, Some(_), _))),
            "ordinary external call was rewritten as a void guard"
        );
        let same_target = calls_in_function(&db, starts["ordinary_same_external_target"])[0];
        assert!(
            db.rel_iter::<(Node, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, _)| *node == same_target)
                .any(|(_, inst)| matches!(inst, RTLInst::Icall(_, _, args, Some(_), _) if args.len() >= 2)),
            "ordinary same-target call lost its two arguments or return value"
        );
    });
}

#[test]
fn decode_control_target_and_ownership_ambiguities_fail_closed() {
    on_pipeline_stack(|| {
        let object = fixture_object();

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            let retained: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .filter(|(node, ..)| *node != check)
                .copied()
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                retained.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            let mut row = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .find(|(node, ..)| *node == check)
                .copied()
                .expect("check raw row");
            row.3 = "NOP";
            db.rel_push("unrefinedinstruction", row);
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if row.0 == check {
                        assert_eq!(row.6, NO_OP, "fixture XOR already has operand three");
                        row.6 = row.4;
                    }
                    row
                })
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if row.0 == check {
                        std::mem::swap(&mut row.4, &mut row.5);
                    }
                    row
                })
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, function, _, _| {
            let members: BTreeSet<Node> = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter_map(|(node, owner)| (*owner == function).then_some(*node))
                .collect();
            let load = db
                .rel_iter::<(Node, LTLInst)>("ltl_inst")
                .find_map(|(node, inst)| {
                    (members.contains(node)
                        && matches!(
                            inst,
                            LTLInst::Lload(_, manifold::x86::op::Addressing::Aglobal(_, _), _, Mreg::AX)
                        ))
                    .then_some(*node)
                })
                .expect("fixture cookie load");
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if row.0 == load {
                        row.4 = row.5;
                    }
                    row
                })
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, function, _, _| {
            let members: BTreeSet<Node> = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter_map(|(node, owner)| (*owner == function).then_some(*node))
                .collect();
            let spill = db
                .rel_iter::<(Node, Symbol, Symbol)>("pmov")
                .find_map(|(node, destination, source)| {
                    (members.contains(node)
                        && db
                            .rel_iter::<(Symbol, &'static str)>("op_register")
                            .any(|(operand, name)| operand == source && *name == "RAX")
                        && db
                            .rel_iter::<(
                                Symbol,
                                &'static str,
                                &'static str,
                                &'static str,
                                i64,
                                i64,
                                usize,
                            )>("op_indirect")
                            .any(|(operand, ..)| operand == destination))
                    .then_some(*node)
                })
                .expect("fixture cookie spill");
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if row.0 == spill {
                        std::mem::swap(&mut row.4, &mut row.5);
                    }
                    row
                })
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, function, _, call| {
            let members: BTreeSet<Node> = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter_map(|(node, owner)| (*owner == function).then_some(*node))
                .collect();
            let restore = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .find_map(|row| {
                    (members.contains(&row.0) && row.0 > call && matches!(row.3, "ADD" | "ADDQ"))
                        .then_some(row.0)
                })
                .expect("fixture terminal stack restore");
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if row.0 == restore {
                        assert_eq!(row.6, NO_OP, "fixture restore already has operand three");
                        row.6 = row.5;
                    }
                    row
                })
                .collect();
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, function, _, _| {
            let members: BTreeSet<Node> = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter_map(|(node, owner)| (*owner == function).then_some(*node))
                .collect();
            let lea = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .find_map(|row| (members.contains(&row.0) && row.3 == "LEA").then_some(row.0))
                .expect("fixture body LEA");
            let retained: Vec<(Node, LTLInst)> = db
                .rel_iter::<(Node, LTLInst)>("ltl_inst")
                .filter(|(node, _)| *node != lea)
                .map(|(node, inst)| (*node, inst.clone()))
                .collect();
            db.rel_set(
                "ltl_inst",
                retained.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, function, _, _| {
            let members: BTreeSet<Node> = db
                .rel_iter::<(Node, Address)>("instr_in_function")
                .filter_map(|(node, owner)| (*owner == function).then_some(*node))
                .collect();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before epoch")
                .as_nanos()
                ^ u128::from(std::process::id());
            let unknown: &'static str =
                Box::leak(format!("UNKNOWN_ZERO_OPERAND_{nonce:032X}").into_boxed_str());
            let mut rewritten = 0usize;
            let rows: Vec<RawRow> = db
                .rel_iter::<RawRow>("unrefinedinstruction")
                .map(|row| {
                    let mut row = *row;
                    if members.contains(&row.0) && row.3 == "NOP" && row.8 == 0 {
                        row.3 = unknown;
                        rewritten += 1;
                    }
                    row
                })
                .collect();
            assert_eq!(rewritten, 1, "fixture must contain one zero-operand NOP");
            db.rel_set(
                "unrefinedinstruction",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            db.rel_push("ltl_inst", (check, LTLInst::Lreturn));
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            db.rel_push("ddisasm_cfg_edge", (check, check + 1, "indirect"));
        });

        assert_mutated_guard_rejected(object, |db, _, _, call| {
            let mut rows: Vec<(Node, LTLInst)> = db
                .rel_iter::<(Node, LTLInst)>("ltl_inst")
                .filter(|(node, _)| *node != call)
                .map(|(node, inst)| (*node, inst.clone()))
                .collect();
            rows.push((call, LTLInst::Lcall(Either::Left(Mreg::x86("R10")))));
            db.rel_set(
                "ltl_inst",
                rows.into_iter().collect::<ascent::boxcar::Vec<_>>(),
            );
        });

        assert_mutated_guard_rejected(object, |db, _, _, call| {
            db.rel_push("call_targets_noreturn", (call,));
        });

        assert_mutated_guard_rejected(object, |db, _, check, _| {
            let foreign = function_starts(db)["ordinary_terminal_call"];
            db.rel_push("instr_in_function", (check, foreign));
        });

        assert_mutated_guard_rejected(object, |db, _, _, call| {
            let symbol = db
                .rel_iter::<(Node, LTLInst)>("ltl_inst")
                .find_map(|(node, inst)| {
                    (*node == call).then_some(inst).and_then(|inst| match inst {
                        LTLInst::Lcall(Either::Right(Either::Right(symbol))) => Some(*symbol),
                        _ => None,
                    })
                })
                .expect("relocated external symbol");
            let mut row = db
                .rel_iter::<(
                    Address,
                    usize,
                    Symbol,
                    Symbol,
                    Symbol,
                    usize,
                    Symbol,
                    usize,
                    Symbol,
                )>("symbol_table")
                .find(|row| row.8 == symbol)
                .copied()
                .expect("external symbol row");
            row.1 = row.1.saturating_add(1);
            db.rel_push("symbol_table", row);
        });

        assert_mutated_guard_rejected(object, |db, _, _, call| {
            let symbol = db
                .rel_iter::<(Node, LTLInst)>("ltl_inst")
                .find_map(|(node, inst)| {
                    (*node == call).then_some(inst).and_then(|inst| match inst {
                        LTLInst::Lcall(Either::Right(Either::Right(symbol))) => Some(*symbol),
                        _ => None,
                    })
                })
                .expect("relocated external symbol");
            let address = db
                .rel_iter::<(Symbol, Address)>("symbol_resolved_addr")
                .find_map(|(name, address)| (*name == symbol).then_some(*address))
                .expect("external target address");
            db.rel_push("symbols", (address, "ambiguous_external_alias", "Beg"));
        });
    });
}

#[test]
fn absent_sp_xor_ltl_is_not_a_contradiction() {
    on_pipeline_stack(|| {
        let object = fixture_object();
        let mut db = run_to_linear(object);
        let function = function_starts(&db)["gs_nonvoid"];
        let members: BTreeSet<Node> = db
            .rel_iter::<(Node, Address)>("instr_in_function")
            .filter_map(|(node, owner)| (*owner == function).then_some(*node))
            .collect();
        let xor_nodes: BTreeSet<Node> = db
            .rel_iter::<(Node, Symbol, Symbol)>("pxor")
            .filter_map(|(node, _, _)| members.contains(node).then_some(*node))
            .collect();
        assert_eq!(xor_nodes.len(), 2, "fixture must contain both /GS XORs");
        let retained: Vec<(Node, LTLInst)> = db
            .rel_iter::<(Node, LTLInst)>("ltl_inst")
            .filter(|(node, _)| !xor_nodes.contains(node))
            .map(|(node, inst)| (*node, inst.clone()))
            .collect();
        db.rel_set(
            "ltl_inst",
            retained.into_iter().collect::<ascent::boxcar::Vec<_>>(),
        );

        RTLPass.run(&mut db);
        let facts: Vec<GuardFact> = guard_facts(&db)
            .into_iter()
            .filter(|(_, owner, _, _, _)| *owner == function)
            .collect();
        assert_eq!(facts.len(), 1, "raw-exact /GS guard was not recovered");
        assert_exact_guard_call(&db, facts[0]);
        assert_pre_call_ax_is_returned(&db, facts[0]);
    });
}

fn declaration_for<'a>(source: &'a str, suffix: &str) -> &'a str {
    source
        .lines()
        .find(|line| {
            let line = line.trim();
            line.contains(suffix)
                && line.contains('(')
                && line.ends_with(';')
                && (line.starts_with("void ")
                    || line.starts_with("int ")
                    || line.starts_with("__int64 ")
                    || line.starts_with("extern "))
        })
        .unwrap_or_else(|| panic!("missing declaration containing {suffix}:\n{source}"))
        .trim()
}

#[test]
fn emitted_guard_calls_use_local_casts_and_c_compiles() {
    on_pipeline_stack(|| {
        let object = fixture_object();
        let emitted = object.with_extension("c");
        let output = Command::new(env!("CARGO_BIN_EXE_manifold"))
            .arg(object)
            .arg(&emitted)
            .output()
            .expect("run manifold over /GS fixture");
        assert!(
            output.status.success(),
            "manifold failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let source = std::fs::read_to_string(&emitted).expect("read emitted C");

        for suffix in [
            "opaque_guard_alpha",
            "opaque_guard_void",
            "opaque_guard_multi",
            "opaque_guard_stale_xmm",
            "opaque_guard_float",
        ] {
            let calls: Vec<&str> = source
                .lines()
                .map(str::trim)
                .filter(|line| {
                    line.contains(suffix)
                        && line.ends_with(';')
                        && !line.starts_with("void ")
                        && !line.starts_with("int ")
                        && !line.starts_with("__int64 ")
                        && !line.starts_with("extern ")
                })
                .collect();
            assert!(
                calls.iter().any(|line| line.contains("void (*)")),
                "missing per-call void(long) cast for {suffix}: {calls:?}\n{source}"
            );
            assert!(
                calls
                    .iter()
                    .filter(|line| line.contains("void (*)"))
                    .all(|line| !line.starts_with("return ")),
                "guard call consumed the body result: {calls:?}"
            );
        }

        let alpha_calls: Vec<&str> = source
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.contains("opaque_guard_alpha")
                    && line.ends_with(';')
                    && !line.starts_with("void ")
                    && !line.starts_with("int ")
                    && !line.starts_with("__int64 ")
                    && !line.starts_with("extern ")
            })
            .collect();
        assert!(alpha_calls.iter().any(|line| line.contains("void (*)")));
        assert!(
            alpha_calls
                .iter()
                .any(|line| !line.contains("void (*)") && line.contains(',')),
            "ordinary same-target call inherited the guard cast: {alpha_calls:?}"
        );
        let alpha_declaration = declaration_for(&source, "opaque_guard_alpha");
        let alpha_parameters = alpha_declaration
            .split_once('(')
            .and_then(|(_, tail)| tail.rsplit_once(')'))
            .map(|(parameters, _)| parameters.trim())
            .expect("alpha declaration parentheses");
        assert!(
            !alpha_declaration
                .strip_prefix("extern ")
                .unwrap_or(alpha_declaration)
                .starts_with("void ")
                && (alpha_parameters.is_empty() || alpha_parameters.contains(',')),
            "per-call recovery changed the shared global declaration: {alpha_declaration}"
        );

        let local_shape_calls: Vec<&str> = source
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.contains("guard_direct_local")
                    && line.ends_with(';')
                    && !line.starts_with("void ")
                    && !line.starts_with("int ")
                    && !line.starts_with("__int64 ")
                    && !line.starts_with("extern ")
            })
            .collect();
        assert!(
            local_shape_calls
                .iter()
                .all(|line| !line.contains("void (*)")),
            "a local target inherited the guard-only cast: {local_shape_calls:?}"
        );

        let _ordinary_declaration = declaration_for(&source, "ordinary_value_target");
        let syntax = Command::new("clang")
            .args([
                "--target=x86_64-pc-windows-msvc",
                "-fsyntax-only",
                "-fms-extensions",
                "-Wno-everything",
                "-x",
                "c",
            ])
            .arg(&emitted)
            .output()
            .expect("compile emitted C");
        assert!(
            syntax.status.success(),
            "emitted C did not compile:\nstdout:\n{}\nstderr:\n{}\nsource:\n{}",
            String::from_utf8_lossy(&syntax.stdout),
            String::from_utf8_lossy(&syntax.stderr),
            source
        );
    });
}
