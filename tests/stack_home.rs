use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use manifold::abi::BinaryFormat;
use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::analysis::type_pass::TypePass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::c_pass::types::{CType, IntSize, Signedness, TopLevelDecl};
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_optimize_pass::RTLOptimizePass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::mreg::Mreg;
use manifold::x86::asm::TestCond;
use manifold::x86::op::{Addressing, Comparison, Condition, Operation};
use manifold::x86::types::{
    Address, CminorBinop, CminorUnop, Constant, CsharpminorExpr, CsharpminorStmt, Ident, LTLInst,
    MachInst, MemoryChunk, RTLInst, RTLReg, Symbol, Typ, XType,
};

const SYNTH1: Address = 1u64 << 62;
const SYNTHETIC_NODE_MASK: Address = (1u64 << 62) | (1u64 << 63);
type InstructionRow = (
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

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "manifold_stack_home_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("failed to create stack/home fixture directory");
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text
        .extern __imp_home_tailcall
        .extern typed_slot_tailcall
        .def typed_slot_tailcall; .scl 2; .type 32; .endef
        .globl sp_indexed_alias_collision
        .def sp_indexed_alias_collision; .scl 2; .type 32; .endef
sp_indexed_alias_collision:
        subq $40, %rsp
        movl %edx, 8(%rsp)
        movl %ecx, %eax
        movl %eax, 8(%rsp,%rax,4)
        movl 8(%rsp,%rax,4), %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_ambiguous_reaching
        .def sp_indexed_ambiguous_reaching; .scl 2; .type 32; .endef
sp_indexed_ambiguous_reaching:
        subq $40, %rsp
        testl %ecx, %ecx
        je sp_indexed_ambiguous_reaching_right
        movl %edx, %eax
        jmp sp_indexed_ambiguous_reaching_join
sp_indexed_ambiguous_reaching_right:
        movl %r8d, %eax
sp_indexed_ambiguous_reaching_join:
        movl 8(%rsp,%rax,4), %edx
        movl %edx, %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_ambiguous
        .def sp_indexed_fused_ambiguous; .scl 2; .type 32; .endef
sp_indexed_fused_ambiguous:
        subq $40, %rsp
        testl %ecx, %ecx
        je sp_indexed_fused_ambiguous_right
        movl %edx, %eax
        jmp sp_indexed_fused_ambiguous_join
sp_indexed_fused_ambiguous_right:
        movl %r8d, %eax
sp_indexed_fused_ambiguous_join:
        addl 8(%rsp,%rax,4), %r9d
        movl %r9d, %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_partial_def
        .def sp_indexed_fused_partial_def; .scl 2; .type 32; .endef
sp_indexed_fused_partial_def:
        subq $40, %rsp
        movl %r8d, 8(%rsp)
        testl %edx, %edx
        je sp_indexed_fused_partial_def_join
        movl %r8d, %ecx
sp_indexed_fused_partial_def_join:
        addl 8(%rsp,%rcx,4), %r9d
        movl %r9d, %eax
        addq $40, %rsp
        retq

        .globl home_alias_roundtrip
        .def home_alias_roundtrip; .scl 2; .type 32; .endef
home_alias_roundtrip:
        movq %rsp, %rax
        movl %edx, 16(%rax)
        movl 16(%rax), %eax
        retq

        .globl home_pointer_indexed
        .def home_pointer_indexed; .scl 2; .type 32; .endef
home_pointer_indexed:
        movq %rsp, %r10
        movq %rcx, 8(%r10)
        movq 8(%rsp), %rax
        movl (%rax,%rdx,4), %r8d
        movl %r8d, %eax
        retq

        .globl wrong_home_source
        .def wrong_home_source; .scl 2; .type 32; .endef
wrong_home_source:
        movq %rsp, %rax
        movl %ecx, 16(%rax)
        movl 16(%rsp), %eax
        retq

        .globl home_reassigned
        .def home_reassigned; .scl 2; .type 32; .endef
home_reassigned:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_reassigned_cmp
        .def home_reassigned_cmp; .scl 2; .type 32; .endef
home_reassigned_cmp:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq $0, 8(%rsp)
        jne home_reassigned_cmp_nonzero
        xorl %eax, %eax
        retq
home_reassigned_cmp_nonzero:
        movl $1, %eax
        retq

        .globl home_reassigned_add
        .def home_reassigned_add; .scl 2; .type 32; .endef
home_reassigned_add:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
        addq 8(%rsp), %rax
        retq

        .globl home_import_tailcall
        .def home_import_tailcall; .scl 2; .type 32; .endef
home_import_tailcall:
        movq %rcx, 8(%rsp)
        movq %rdx, 16(%rsp)
        movq 8(%rsp), %rcx
        movq 16(%rsp), %rdx
        jmpq *__imp_home_tailcall(%rip)

        .globl home_typed_slot_tailcall
        .def home_typed_slot_tailcall; .scl 2; .type 32; .endef
home_typed_slot_tailcall:
        movq %rcx, 8(%rsp)
        movq %rdx, 16(%rsp)
        movq 8(%rsp), %rcx
        movq 16(%rsp), %rdx
        jmpq *typed_slot_tailcall(%rip)

        .globl home_epilogue_register_tailcall
        .def home_epilogue_register_tailcall; .scl 2; .type 32; .endef
home_epilogue_register_tailcall:
        pushq %rbx
        subq $32, %rsp
        movq %rcx, 8(%rsp)
        movq %rdx, %rax
        movq 8(%rsp), %rcx
        addq $32, %rsp
        popq %rbx
        jmp .Lhome_epilogue_register_tailcall_shared
.Lhome_epilogue_register_tailcall_shared:
        jmpq *%rax

        .globl home_adjacent_epilogue_register_tailcall
        .def home_adjacent_epilogue_register_tailcall; .scl 2; .type 32; .endef
home_adjacent_epilogue_register_tailcall:
        subq $40, %rsp
        movq %rcx, 8(%rsp)
        movq %rdx, %rax
        movq 8(%rsp), %rcx
        addq $40, %rsp
        jmpq *%rax

        .globl home_live_frame_register_dispatch
        .def home_live_frame_register_dispatch; .scl 2; .type 32; .endef
home_live_frame_register_dispatch:
        pushq %rbx
        subq $32, %rsp
        movq %rcx, 8(%rsp)
        jmpq *%rdx

        .globl home_frameless_register_dispatch
        .def home_frameless_register_dispatch; .scl 2; .type 32; .endef
home_frameless_register_dispatch:
        movq %rcx, 8(%rsp)
        jmpq *%rdx

        .globl home_reassigned_cmp_reg
        .def home_reassigned_cmp_reg; .scl 2; .type 32; .endef
home_reassigned_cmp_reg:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq %r8, 8(%rsp)
        jne home_reassigned_cmp_reg_nonzero
        xorl %eax, %eax
        retq
home_reassigned_cmp_reg_nonzero:
        movl $1, %eax
        retq

        .globl home_reassigned_cmp_setcc
        .def home_reassigned_cmp_setcc; .scl 2; .type 32; .endef
home_reassigned_cmp_setcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq $0, 8(%rsp)
        setne %al
        movzbl %al, %eax
        retq

        .globl home_reassigned_cmp_reg_setcc
        .def home_reassigned_cmp_reg_setcc; .scl 2; .type 32; .endef
home_reassigned_cmp_reg_setcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq %r8, 8(%rsp)
        sete %al
        movzbl %al, %eax
        retq

        .globl home_reassigned_cmp_loop
        .def home_reassigned_cmp_loop; .scl 2; .type 32; .endef
home_reassigned_cmp_loop:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
home_reassigned_cmp_loop_again:
        addq $1, %rax
        cmpq 8(%rsp), %rax
        jne home_reassigned_cmp_loop_again
        retq

        .globl home_cmp_after_alias_clobber
        .def home_cmp_after_alias_clobber; .scl 2; .type 32; .endef
home_cmp_after_alias_clobber:
        movq %rsp, %rax
        movq %r9, 32(%rax)
        movq %r8, 24(%rax)
        movq %rdx, 16(%rax)
        movq %rcx, 8(%rax)
        pushq %rbx
        movq 8(%rax), %rax
        xorl %ebx, %ebx
home_cmp_after_alias_clobber_again:
        incl %ebx
        cmpq 40(%rsp), %rbx
        jne home_cmp_after_alias_clobber_again
        popq %rbx
        retq

        .globl home_reassigned_movsxd
        .def home_reassigned_movsxd; .scl 2; .type 32; .endef
home_reassigned_movsxd:
        movl %ecx, 8(%rsp)
        movl %edx, 8(%rsp)
        movslq 8(%rsp), %rax
        retq

        .globl home_reassigned_movsx
        .def home_reassigned_movsx; .scl 2; .type 32; .endef
home_reassigned_movsx:
        movw %cx, 8(%rsp)
        movw %dx, 8(%rsp)
        movswl 8(%rsp), %eax
        retq

        .globl home_reassigned_movzx
        .def home_reassigned_movzx; .scl 2; .type 32; .endef
home_reassigned_movzx:
        movw %cx, 8(%rsp)
        movw %dx, 8(%rsp)
        movzwl 8(%rsp), %eax
        retq

        .globl home_partial_movzx_low
        .def home_partial_movzx_low; .scl 2; .type 32; .endef
home_partial_movzx_low:
        movl %ecx, 8(%rsp)
        movl %edx, 8(%rsp)
        movzwl 8(%rsp), %eax
        retq

        .globl home_partial_movzx_high_backing
        .def home_partial_movzx_high_backing; .scl 2; .type 32; .endef
home_partial_movzx_high_backing:
        movl %ecx, 8(%rsp)
        movl %edx, 8(%rsp)
        movzwl 10(%rsp), %eax
        retq

        .globl home_backing_movsx_high
        .def home_backing_movsx_high; .scl 2; .type 32; .endef
home_backing_movsx_high:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movswl 10(%rsp), %eax
        retq

        .globl home_backing_signed_cmp_jcc
        .def home_backing_signed_cmp_jcc; .scl 2; .type 32; .endef
home_backing_signed_cmp_jcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpw $-1, 10(%rsp)
        jl home_backing_signed_cmp_jcc_less
        xorl %eax, %eax
        retq
home_backing_signed_cmp_jcc_less:
        movl $1, %eax
        retq

        .globl home_backing_unsigned_cmp_setcc
        .def home_backing_unsigned_cmp_setcc; .scl 2; .type 32; .endef
home_backing_unsigned_cmp_setcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpw $7, 10(%rsp)
        setb %al
        movzbl %al, %eax
        retq

        .globl home_backing_partial_load
        .def home_backing_partial_load; .scl 2; .type 32; .endef
home_backing_partial_load:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movw 10(%rsp), %ax
        movzwl %ax, %eax
        retq

        .globl home_backing_partial_store
        .def home_backing_partial_store; .scl 2; .type 32; .endef
home_backing_partial_store:
        movq %rcx, 8(%rsp)
        movw %dx, 10(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_backing_add_high
        .def home_backing_add_high; .scl 2; .type 32; .endef
home_backing_add_high:
        movq %rcx, 8(%rsp)
        addw %dx, 10(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_backing_test_jcc
        .def home_backing_test_jcc; .scl 2; .type 32; .endef
home_backing_test_jcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        testb $4, 9(%rsp)
        jne home_backing_test_jcc_nonzero
        xorl %eax, %eax
        retq
home_backing_test_jcc_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_test_setcc
        .def home_backing_test_setcc; .scl 2; .type 32; .endef
home_backing_test_setcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        testw $8, 10(%rsp)
        setne %al
        movzbl %al, %eax
        retq

        .globl home_backing_cmov_cmp
        .def home_backing_cmov_cmp; .scl 2; .type 32; .endef
home_backing_cmov_cmp:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
        cmpq $0, %r9
        cmovneq 8(%rsp), %rax
        retq

        .globl home_backing_cmov_test
        .def home_backing_cmov_test; .scl 2; .type 32; .endef
home_backing_cmov_test:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
        testq $1, %r9
        cmoveq 8(%rsp), %rax
        retq

        .globl home_backing_div
        .def home_backing_div; .scl 2; .type 32; .endef
home_backing_div:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        pushq %rbp
        movq %rsp, %rbp
        movl %r8d, %eax
        xorl %edx, %edx
        divl 16(%rbp)
        popq %rbp
        retq

        .globl home_backing_descriptor
        .def home_backing_descriptor; .scl 2; .type 32; .endef
home_backing_descriptor:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        subq $40, %rsp
        leaq 48(%rsp), %rax
        movq %rax, 32(%rsp)
        leaq 32(%rsp), %rcx
        callq *__imp_home_tailcall(%rip)
        addq $40, %rsp
        retq

        .globl home_backing_cmp_setcc
        .def home_backing_cmp_setcc; .scl 2; .type 32; .endef
home_backing_cmp_setcc:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movzwl 10(%rsp), %r8d
        cmpq $0, 8(%rsp)
        setne %al
        movzbl %al, %eax
        addl %r8d, %eax
        retq

        .globl home_cmp_flag_clobber_rejected
        .def home_cmp_flag_clobber_rejected; .scl 2; .type 32; .endef
home_cmp_flag_clobber_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq $0, 8(%rsp)
        addq $1, %r8
        jne home_cmp_flag_clobber_rejected_nonzero
        xorl %eax, %eax
        retq
home_cmp_flag_clobber_rejected_nonzero:
        movl $1, %eax
        retq

        .globl home_cmp_scheduled_rejected
        .def home_cmp_scheduled_rejected; .scl 2; .type 32; .endef
home_cmp_scheduled_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        cmpq $0, 8(%rsp)
        movq %r8, %r9
        movq %rdx, %r8
        jne home_cmp_scheduled_rejected_nonzero
        xorl %eax, %eax
        retq
home_cmp_scheduled_rejected_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_cmp_flag_clobber_rejected
        .def home_backing_cmp_flag_clobber_rejected; .scl 2; .type 32; .endef
home_backing_cmp_flag_clobber_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movzwl 10(%rsp), %eax
        cmpq $0, 8(%rsp)
        addq $1, %r8
        jne home_backing_cmp_flag_clobber_rejected_nonzero
        xorl %eax, %eax
        retq
home_backing_cmp_flag_clobber_rejected_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_test_scheduled_rejected
        .def home_backing_test_scheduled_rejected; .scl 2; .type 32; .endef
home_backing_test_scheduled_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        testb $4, 9(%rsp)
        movq %r8, %r9
        jne home_backing_test_scheduled_rejected_nonzero
        xorl %eax, %eax
        retq
home_backing_test_scheduled_rejected_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_cmov_scheduled_rejected
        .def home_backing_cmov_scheduled_rejected; .scl 2; .type 32; .endef
home_backing_cmov_scheduled_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
        cmpq $0, %r9
        movq %r10, %r11
        cmovneq 8(%rsp), %rax
        retq

        .globl home_backing_add_jcc_rejected
        .def home_backing_add_jcc_rejected; .scl 2; .type 32; .endef
home_backing_add_jcc_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        jne home_backing_add_jcc_rejected_nonzero
        xorl %eax, %eax
        retq
home_backing_add_jcc_rejected_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_add_scheduled_jcc_rejected
        .def home_backing_add_scheduled_jcc_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_jcc_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        jne home_backing_add_scheduled_jcc_nonzero
        xorl %eax, %eax
        retq
home_backing_add_scheduled_jcc_nonzero:
        movl $1, %eax
        retq

        .globl home_backing_add_scheduled_setcc_rejected
        .def home_backing_add_scheduled_setcc_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_setcc_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        setne %al
        movzbl %al, %eax
        retq

        .globl home_backing_add_scheduled_cmov_rejected
        .def home_backing_add_scheduled_cmov_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_cmov_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r10, %r11
        cmovneq %r8, %r9
        movq %r9, %rax
        retq

        .globl home_backing_add_scheduled_adc_rejected
        .def home_backing_add_scheduled_adc_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_adc_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r10, %r11
        adcl $0, %eax
        retq

        .globl home_backing_add_scheduled_rcr_rejected
        .def home_backing_add_scheduled_rcr_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_rcr_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r10, %r11
        rcrl $1, %eax
        retq

        .globl home_backing_add_scheduled_setp_rejected
        .def home_backing_add_scheduled_setp_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_setp_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        setp %al
        movzbl %al, %eax
        retq

        .globl home_backing_add_scheduled_setnp_rejected
        .def home_backing_add_scheduled_setnp_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_setnp_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        setnp %al
        movzbl %al, %eax
        retq

        .globl home_backing_add_scheduled_jo_rejected
        .def home_backing_add_scheduled_jo_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_jo_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        jo home_backing_add_scheduled_jo_taken
        xorl %eax, %eax
        retq
home_backing_add_scheduled_jo_taken:
        movl $1, %eax
        retq

        .globl home_backing_add_scheduled_jno_rejected
        .def home_backing_add_scheduled_jno_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_jno_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        jno home_backing_add_scheduled_jno_taken
        xorl %eax, %eax
        retq
home_backing_add_scheduled_jno_taken:
        movl $1, %eax
        retq

        .globl home_backing_add_scheduled_seto_rejected
        .def home_backing_add_scheduled_seto_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_seto_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        seto %al
        movzbl %al, %eax
        retq

        .globl home_backing_add_scheduled_setno_rejected
        .def home_backing_add_scheduled_setno_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_setno_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r8, %r9
        setno %al
        movzbl %al, %eax
        retq

        .globl home_backing_add_scheduled_cmovo_rejected
        .def home_backing_add_scheduled_cmovo_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_cmovo_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r10, %r11
        cmovoq %r8, %r9
        movq %r9, %rax
        retq

        .globl home_backing_add_scheduled_cmovno_rejected
        .def home_backing_add_scheduled_cmovno_rejected; .scl 2; .type 32; .endef
home_backing_add_scheduled_cmovno_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        addw %dx, 10(%rsp)
        movq %r10, %r11
        cmovnoq %r8, %r9
        movq %r9, %rax
        retq

        .globl home_backing_lock_add_rejected
        .def home_backing_lock_add_rejected; .scl 2; .type 32; .endef
home_backing_lock_add_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        lock addw %dx, 10(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_backing_cmovs_rejected
        .def home_backing_cmovs_rejected; .scl 2; .type 32; .endef
home_backing_cmovs_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movq %r8, %rax
        cmpq $0, %r9
        cmovsq 8(%rsp), %rax
        retq

        .globl home_backing_cross_cell_rejected
        .def home_backing_cross_cell_rejected; .scl 2; .type 32; .endef
home_backing_cross_cell_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movl 14(%rsp), %eax
        retq

        .globl home_backing_indexed_rejected
        .def home_backing_indexed_rejected; .scl 2; .type 32; .endef
home_backing_indexed_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movl 8(%rsp,%r8,1), %eax
        retq

        .globl home_backing_rbp_xchg_rejected
        .def home_backing_rbp_xchg_rejected; .scl 2; .type 32; .endef
home_backing_rbp_xchg_rejected:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        pushq %rbp
        movq %rsp, %rbp
        xchgq %rdx, 16(%rbp)
        movq 16(%rbp), %rax
        popq %rbp
        retq

        .globl home_backing_partial_pointer_store_rejected
        .def home_backing_partial_pointer_store_rejected; .scl 2; .type 32; .endef
home_backing_partial_pointer_store_rejected:
        movq %rcx, 8(%rsp)
        leaq 8(%rsp), %rax
        subq $40, %rsp
        movl %eax, 32(%rsp)
        addq $40, %rsp
        xorl %eax, %eax
        retq

        .globl home_backing_segmented_store_rejected
        .def home_backing_segmented_store_rejected; .scl 2; .type 32; .endef
home_backing_segmented_store_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movw %r8w, %fs:10(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_backing_segmented_load_rejected
        .def home_backing_segmented_load_rejected; .scl 2; .type 32; .endef
home_backing_segmented_load_rejected:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        movw %gs:10(%rsp), %ax
        movzwl %ax, %eax
        retq

        .globl sp_immediate_rmw
        .def sp_immediate_rmw; .scl 2; .type 32; .endef
sp_immediate_rmw:
        subq $32, %rsp
        movl $7, 12(%rsp)
        movl %edx, %eax
        subl 12(%rsp), %eax
        addl $1, 12(%rsp)
        addq $32, %rsp
        retq

        .globl sp_indexed_fused_arith
        .def sp_indexed_fused_arith; .scl 2; .type 32; .endef
sp_indexed_fused_arith:
        subq $40, %rsp
        movl %r8d, 8(%rsp)
        movl %ecx, %eax
        addl 8(%rsp,%rdx,4), %eax
        subl 8(%rsp,%rdx,4), %eax
        andl 8(%rsp,%rdx,4), %eax
        orl 8(%rsp,%rdx,4), %eax
        xorl 8(%rsp,%rdx,4), %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_add_collision
        .def sp_indexed_fused_add_collision; .scl 2; .type 32; .endef
sp_indexed_fused_add_collision:
        subq $40, %rsp
        movl %edx, 8(%rsp)
        movl %ecx, %eax
        addl 8(%rsp,%rax,4), %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_field
        .def sp_indexed_fused_field; .scl 2; .type 32; .endef
sp_indexed_fused_field:
        subq $40, %rsp
        movq %r8, 8(%rsp)
        movl %ecx, %eax
        addl 12(%rsp,%rdx,8), %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_misaligned_field
        .def sp_indexed_fused_misaligned_field; .scl 2; .type 32; .endef
sp_indexed_fused_misaligned_field:
        subq $40, %rsp
        movq %r9, 0(%rsp)
        movq %r8, 4(%rsp)
        movl %ecx, %eax
        addl 8(%rsp,%rdx,8), %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_param_rmw
        .def sp_indexed_fused_param_rmw; .scl 2; .type 32; .endef
sp_indexed_fused_param_rmw:
        subq $40, %rsp
        movq %rdx, 8(%rsp)
        addl 8(%rsp,%rcx,4), %ecx
        movl %ecx, %eax
        addq $40, %rsp
        retq

        .globl sp_indexed_fused_entry_collision
        .def sp_indexed_fused_entry_collision; .scl 2; .type 32; .endef
sp_indexed_fused_entry_collision:
        addl 40(%rsp,%rcx,4), %ecx
        movl %ecx, %eax
        retq

        .globl sp_indexed_fused_loop_collision
        .def sp_indexed_fused_loop_collision; .scl 2; .type 32; .endef
sp_indexed_fused_loop_collision:
        nop
.Lsp_indexed_fused_loop_collision:
        addl 40(%rsp,%rcx,4), %ecx
        decl %edx
        jne .Lsp_indexed_fused_loop_collision
        movl %ecx, %eax
        retq

        .globl sp_indexed_load_entry_collision
        .def sp_indexed_load_entry_collision; .scl 2; .type 32; .endef
sp_indexed_load_entry_collision:
        movl 40(%rsp,%rcx,4), %ecx
        movl %ecx, %eax
        retq

        .globl sp_indexed_load_loop_collision
        .def sp_indexed_load_loop_collision; .scl 2; .type 32; .endef
sp_indexed_load_loop_collision:
        movl %ecx, %eax
.Lsp_indexed_load_loop_collision:
        movl 40(%rsp,%rax,4), %eax
        decl %edx
        jne .Lsp_indexed_load_loop_collision
        retq

        .globl sp_indexed_fused_gap
        .def sp_indexed_fused_gap; .scl 2; .type 32; .endef
sp_indexed_fused_gap:
        subq $40, %rsp
        movq %r8, 8(%rsp)
        movl %ecx, %eax
        addl 8(%rsp,%rdx,4), %eax
        nop
        subl 8(%rsp,%rdx,4), %eax
        addq $40, %rsp
        retq

        .globl fifth_imul
        .def fifth_imul; .scl 2; .type 32; .endef
fifth_imul:
        imull $7, 40(%rsp), %eax
        retq

        .globl sixth_to_double
        .def sixth_to_double; .scl 2; .type 32; .endef
sixth_to_double:
        cvtsi2sdl 48(%rsp), %xmm0
        retq

        .globl caller_home_vs_outgoing
        .def caller_home_vs_outgoing; .scl 2; .type 32; .endef
caller_home_vs_outgoing:
        movl %edx, 16(%rsp)
        subq $40, %rsp
        movl %ecx, 32(%rsp)
        callq callee_eighth
        addq $40, %rsp
        retq

        .globl callee_eighth
        .def callee_eighth; .scl 2; .type 32; .endef
callee_eighth:
        movl 64(%rsp), %eax
        retq

        .globl home_escape_mutated
        .def home_escape_mutated; .scl 2; .type 32; .endef
home_escape_mutated:
        movq %rcx, 8(%rsp)
        leaq 8(%rsp), %rcx
        subq $40, %rsp
        callq mutate_slot
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl mutate_slot
        .def mutate_slot; .scl 2; .type 32; .endef
mutate_slot:
        movq $99, (%rcx)
        retq

        .globl mutate_slot_plus8
        .def mutate_slot_plus8; .scl 2; .type 32; .endef
mutate_slot_plus8:
        movq $99, 8(%rcx)
        retq

        .globl home_mixed_base_clobber
        .def home_mixed_base_clobber; .scl 2; .type 32; .endef
home_mixed_base_clobber:
        movq %rsp, %r10
        movq %rcx, 8(%r10)
        movq %rdx, 8(%rsp)
        movq 8(%r10), %rax
        retq

        .globl home_disjoint_branch
        .def home_disjoint_branch; .scl 2; .type 32; .endef
home_disjoint_branch:
        testl %edx, %edx
        je 1f
        movq %rcx, 8(%rsp)
1:
        movq 8(%rsp), %rax
        retq

        .globl home_vs_postsub_indexed
        .def home_vs_postsub_indexed; .scl 2; .type 32; .endef
home_vs_postsub_indexed:
        movq %rcx, 8(%rsp)
        subq $40, %rsp
        movl %edx, 8(%rsp)
        movl %r8d, 8(%rsp,%rdx,4)
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl two_sp_depths
        .def two_sp_depths; .scl 2; .type 32; .endef
two_sp_depths:
        subq $32, %rsp
        movl %edx, 8(%rsp)
        addq $32, %rsp
        subq $64, %rsp
        movl %r8d, 8(%rsp,%rdx,4)
        addq $64, %rsp
        retq

        .globl dynamic_sp_unknown
        .def dynamic_sp_unknown; .scl 2; .type 32; .endef
dynamic_sp_unknown:
        subq %rax, %rsp
        movl 40(%rsp), %eax
        addq %rax, %rsp
        retq

        .globl branch_stack_unknown
        .def branch_stack_unknown; .scl 2; .type 32; .endef
branch_stack_unknown:
        testl %ecx, %ecx
        je 2f
        subq $16, %rsp
2:
        movl 40(%rsp), %eax
        testl %ecx, %ecx
        je 3f
        addq $16, %rsp
3:
        retq

        .globl fifth_inc
        .def fifth_inc; .scl 2; .type 32; .endef
fifth_inc:
        addl $1, 40(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl fifth_add_reg
        .def fifth_add_reg; .scl 2; .type 32; .endef
fifth_add_reg:
        addl %ecx, 40(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl alias_partial_before_fifth_inc
        .def alias_partial_before_fifth_inc; .scl 2; .type 32; .endef
alias_partial_before_fifth_inc:
        movq %rsp, %r10
        movb $7, 41(%r10)
        addl $1, 40(%rsp)
        movl 40(%rsp), %eax
        retq

        .globl outgoing_reuses_home_coordinate
        .def outgoing_reuses_home_coordinate; .scl 2; .type 32; .endef
outgoing_reuses_home_coordinate:
        subq $40, %rsp
        movq %rcx, 48(%rsp)
        callq callee_seventh
        addq $40, %rsp
        retq

        .globl callee_seventh
        .def callee_seventh; .scl 2; .type 32; .endef
callee_seventh:
        movq 56(%rsp), %rax
        retq

        .globl high_stack_offset
        .def high_stack_offset; .scl 2; .type 32; .endef
high_stack_offset:
        movl 552(%rsp), %eax
        retq

        .globl boundary_stack_offset
        .def boundary_stack_offset; .scl 2; .type 32; .endef
boundary_stack_offset:
        movl 544(%rsp), %eax
        retq

        .globl bp_fifth
        .def bp_fifth; .scl 2; .type 32; .endef
bp_fifth:
        pushq %rbp
        movq %rsp, %rbp
        movl 48(%rbp), %eax
        popq %rbp
        retq

        .globl postprologue_fifth
        .def postprologue_fifth; .scl 2; .type 32; .endef
postprologue_fifth:
        subq $40, %rsp
        movl 80(%rsp), %eax
        addq $40, %rsp
        retq

        .globl trap_rsp_fallthrough
        .def trap_rsp_fallthrough; .scl 2; .type 32; .endef
trap_rsp_fallthrough:
        subq $40, %rsp
        int3
        movl 80(%rsp), %eax
        addq $40, %rsp
        retq

        .globl trap_int2c_rsp_fallthrough
        .def trap_int2c_rsp_fallthrough; .scl 2; .type 32; .endef
trap_int2c_rsp_fallthrough:
        subq $40, %rsp
        int $0x2c
        movl 80(%rsp), %eax
        addq $40, %rsp
        retq

        .globl xmm_home_roundtrip
        .def xmm_home_roundtrip; .scl 2; .type 32; .endef
xmm_home_roundtrip:
        movsd %xmm0, 8(%rsp)
        movsd 8(%rsp), %xmm0
        retq

        .globl bp_postalloc_fifth
        .def bp_postalloc_fifth; .scl 2; .type 32; .endef
bp_postalloc_fifth:
        subq $40, %rsp
        movq %rsp, %rbp
        movl 80(%rbp), %eax
        addq $40, %rsp
        retq

        .globl bp_conditional_pointer
        .def bp_conditional_pointer; .scl 2; .type 32; .endef
bp_conditional_pointer:
        testl %ecx, %ecx
        je 3f
        movq %rsp, %rbp
3:
        movl %edx, 40(%rbp)
        movl 40(%rbp), %eax
        retq

        .globl bp_conditional_indexed_pointer
        .def bp_conditional_indexed_pointer; .scl 2; .type 32; .endef
bp_conditional_indexed_pointer:
        testl %ecx, %ecx
        je 4f
        movq %rsp, %rbp
4:
        movl 40(%rbp,%rdx,4), %eax
        retq

        .globl bp_conditional_clobber
        .def bp_conditional_clobber; .scl 2; .type 32; .endef
bp_conditional_clobber:
        movq %rsp, %rbp
        testl %ecx, %ecx
        je 5f
        movq %r8, %rbp
5:
        movl 40(%rbp), %eax
        retq

        .globl bp_conditional_lea
        .def bp_conditional_lea; .scl 2; .type 32; .endef
bp_conditional_lea:
        testl %ecx, %ecx
        je .Lbp_conditional_lea_join
        movq %rsp, %rbp
.Lbp_conditional_lea_join:
        leaq 40(%rbp), %rax
        retq

        .globl bp_clobbered_lea
        .def bp_clobbered_lea; .scl 2; .type 32; .endef
bp_clobbered_lea:
        movq %rsp, %rbp
        testl %ecx, %ecx
        je .Lbp_clobbered_lea_join
        movq %r8, %rbp
.Lbp_clobbered_lea_join:
        leaq 40(%rbp), %rax
        retq

        .globl bp_conditional_rmw
        .def bp_conditional_rmw; .scl 2; .type 32; .endef
bp_conditional_rmw:
        testl %ecx, %ecx
        je .Lbp_conditional_rmw_join
        movq %rsp, %rbp
.Lbp_conditional_rmw_join:
        addl $1, 40(%rbp)
        movl 40(%rbp), %eax
        retq

        .globl bp_clobbered_rmw
        .def bp_clobbered_rmw; .scl 2; .type 32; .endef
bp_clobbered_rmw:
        movq %rsp, %rbp
        testl %ecx, %ecx
        je .Lbp_clobbered_rmw_join
        movq %r8, %rbp
.Lbp_clobbered_rmw_join:
        subl $1, 40(%rbp)
        movl 40(%rbp), %eax
        retq

        .globl narrow_ebp_pointer
        .def narrow_ebp_pointer; .scl 2; .type 32; .endef
narrow_ebp_pointer:
        movq %rsp, %rbp
        movl 40(%ebp), %eax
        retq

        .globl narrow_esp_pointer
        .def narrow_esp_pointer; .scl 2; .type 32; .endef
narrow_esp_pointer:
        movl 40(%esp), %eax
        retq

        .globl narrow_eax_stack_alias
        .def narrow_eax_stack_alias; .scl 2; .type 32; .endef
narrow_eax_stack_alias:
        movq %rsp, %rax
        movl %edx, 16(%eax)
        movl 16(%eax), %eax
        retq

        .globl pop_rsp_unknown
        .def pop_rsp_unknown; .scl 2; .type 32; .endef
pop_rsp_unknown:
        movq %rsp, %r11
        popq %rsp
        movl 40(%rsp), %eax
        movq %r11, %rsp
        retq

        .globl home_unknown_indexed_bp
        .def home_unknown_indexed_bp; .scl 2; .type 32; .endef
home_unknown_indexed_bp:
        movq %rcx, 8(%rsp)
        testl %edx, %edx
        je 6f
        movq %rsp, %rbp
6:
        movq 8(%rbp,%rdx,1), %rax
        movq 8(%rsp), %rax
        retq

        .globl home_alias_call_escape
        .def home_alias_call_escape; .scl 2; .type 32; .endef
home_alias_call_escape:
        pushq %r12
        leaq 8(%rsp), %r12
        movq %rcx, 8(%r12)
        leaq 8(%r12), %rcx
        subq $32, %rsp
        callq mutate_slot
        addq $32, %rsp
        movq 8(%r12), %rax
        popq %r12
        retq

        .globl home_escape_numeric_use
        .def home_escape_numeric_use; .scl 2; .type 32; .endef
home_escape_numeric_use:
        movq %rcx, 8(%rsp)
        leaq 8(%rsp), %r10
        movq %r10, %rcx
        addq $1, %r10
        subq $40, %rsp
        callq mutate_slot
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl home_ambiguous_alias_call_escape
        .def home_ambiguous_alias_call_escape; .scl 2; .type 32; .endef
home_ambiguous_alias_call_escape:
        movq %rcx, 8(%rsp)
        testl %edx, %edx
        je .Lhome_alias_unrelated
        movq %rsp, %r10
        jmp .Lhome_alias_join
.Lhome_alias_unrelated:
        movq %r8, %r10
.Lhome_alias_join:
        movq %r10, %rcx
        subq $40, %rsp
        callq mutate_slot_plus8
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl home_direct_alias_call_escape
        .def home_direct_alias_call_escape; .scl 2; .type 32; .endef
home_direct_alias_call_escape:
        movq %rcx, 8(%rsp)
        movq %rsp, %rcx
        subq $40, %rsp
        callq mutate_slot_plus8
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl home_direct_alias_return
        .def home_direct_alias_return; .scl 2; .type 32; .endef
home_direct_alias_return:
        movq %rcx, 8(%rsp)
        movq %rsp, %rax
        retq

        .globl home_cross_class_reload
        .def home_cross_class_reload; .scl 2; .type 32; .endef
home_cross_class_reload:
        movl %ecx, 8(%rsp)
        movss 8(%rsp), %xmm0
        retq

        .globl home_xmm_mutation
        .def home_xmm_mutation; .scl 2; .type 32; .endef
home_xmm_mutation:
        movsd %xmm0, 8(%rsp)
        movss %xmm1, 8(%rsp)
        movsd 8(%rsp), %xmm0
        retq

        .globl home_partial_reload
        .def home_partial_reload; .scl 2; .type 32; .endef
home_partial_reload:
        movq %rcx, 8(%rsp)
        movzbl 8(%rsp), %eax
        retq

        .globl home_xchg_mutation
        .def home_xchg_mutation; .scl 2; .type 32; .endef
home_xchg_mutation:
        movq %rcx, 8(%rsp)
        xchgq %rdx, 8(%rsp)
        movq 8(%rsp), %rax
        retq

        .globl home_alias_arith_clobber
        .def home_alias_arith_clobber; .scl 2; .type 32; .endef
home_alias_arith_clobber:
        movq %rsp, %r10
        movq %rcx, 8(%r10)
        addq $8, %r10
        movq (%r10), %rax
        retq

        .globl home_postsub_raw_collision
        .def home_postsub_raw_collision; .scl 2; .type 32; .endef
home_postsub_raw_collision:
        movq %rcx, 8(%rsp)
        movq %rdx, 8(%rsp)
        subq $40, %rsp
        movq %r8, 8(%rsp)
        movq 8(%rsp), %r9
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl home_postsub_escaped_collision
        .def home_postsub_escaped_collision; .scl 2; .type 32; .endef
home_postsub_escaped_collision:
        movq %rcx, 8(%rsp)
        movq %r8, 8(%rsp)
        leaq 8(%rsp), %rcx
        subq $40, %rsp
        callq mutate_slot
        movq %rsp, %rbp
        movq %r9, 8(%rbp)
        leaq 8(%rbp), %rdx
        callq mutate_slot_second
        addq $40, %rsp
        movq 8(%rsp), %rax
        retq

        .globl mutate_slot_second
        .def mutate_slot_second; .scl 2; .type 32; .endef
mutate_slot_second:
        movq $77, (%rdx)
        retq

        .globl home_volatile_alias_after_call
        .def home_volatile_alias_after_call; .scl 2; .type 32; .endef
home_volatile_alias_after_call:
        movq %rsp, %r10
        movq %rcx, 8(%r10)
        subq $40, %rsp
        callq preserve_probe
        addq $40, %rsp
        addq $8, %r10
        movq (%r10), %rax
        retq

        .globl home_nonvolatile_alias_after_call
        .def home_nonvolatile_alias_after_call; .scl 2; .type 32; .endef
home_nonvolatile_alias_after_call:
        movq %rsp, %r12
        movq %r12, %r13
        movq %rcx, 8(%r13)
        subq $40, %rsp
        callq preserve_probe
        addq $40, %rsp
        addq $8, %r13
        movq (%r13), %rax
        retq

        .globl preserve_probe
        .def preserve_probe; .scl 2; .type 32; .endef
preserve_probe:
        retq

        .globl alias_mov_vs_lea_coordinates
        .def alias_mov_vs_lea_coordinates; .scl 2; .type 32; .endef
alias_mov_vs_lea_coordinates:
        subq $40, %rsp
        movq %rsp, %r10
        leaq 8(%rsp), %r11
        movl 48(%r10), %eax
        movl 40(%r11), %eax
        addq $40, %rsp
        retq

        .globl alias_two_hop_lea
        .def alias_two_hop_lea; .scl 2; .type 32; .endef
alias_two_hop_lea:
        movq %rsp, %r10
        leaq 8(%r10), %r11
        movq (%r11), %rax
        retq

        .globl home_conditional_spill_reload
        .def home_conditional_spill_reload; .scl 2; .type 32; .endef
home_conditional_spill_reload:
        testl %edx, %edx
        je 7f
        movq %rcx, 8(%rsp)
7:
        movq 8(%rsp), %rax
        retq

        .globl unknown_sp_indexed
        .def unknown_sp_indexed; .scl 2; .type 32; .endef
unknown_sp_indexed:
        testl %ecx, %ecx
        je 8f
        subq $16, %rsp
8:
        movl %edx, 8(%rsp,%r9,4)
        movl 8(%rsp,%r9,4), %eax
        testl %ecx, %ecx
        je 9f
        addq $16, %rsp
9:
        retq

        .globl home_wide_overlap
        .def home_wide_overlap; .scl 2; .type 32; .endef
home_wide_overlap:
        movq %rcx, 8(%rsp)
        movdqu 4(%rsp), %xmm1
        movq 8(%rsp), %rax
        retq
"#,
    )
    .expect("failed to write stack/home fixture assembly");

    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to run clang over stack/home fixture");
    assert!(status.success(), "stack/home fixture assembly failed");
    object
}

fn fixture() -> &'static Path {
    static FIXTURE: OnceLock<PathBuf> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_path()
}

fn load_rtl_relations(object: &Path) -> DecompileDB {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);
    CanaryVlaPass.run(&mut db);
    AsmPass.run(&mut db);
    StackAnalysisPass.run(&mut db);
    MachPass.run(&mut db);
    LinearPass.run(&mut db);
    RTLPass.run(&mut db);
    db
}

fn function_span(db: &DecompileDB, name: &str) -> (Address, Address) {
    let coff_name = format!("coff_fn_{name}");
    db.rel_iter::<(Symbol, Address, Address)>("func_span")
        .find_map(|(symbol, start, end)| {
            (*symbol == name || *symbol == coff_name).then_some((*start, *end))
        })
        .unwrap_or_else(|| {
            let mut available: Vec<_> = db
                .rel_iter::<(Symbol, Address, Address)>("func_span")
                .map(|(symbol, start, end)| (*symbol, *start, *end))
                .collect();
            available.sort_by_key(|(_, start, _)| *start);
            panic!("missing fixture function {name}; available={available:#x?}")
        })
}

fn in_span(address: Address, span: (Address, Address)) -> bool {
    address >= span.0 && address < span.1
}

fn printed_function_definition<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let needle = format!("{name}(");
    for (start, _) in text.match_indices(&needle) {
        let tail = &text[start..];
        let Some(open_brace) = tail.find('{') else {
            continue;
        };
        if tail.find(';').is_some_and(|semicolon| semicolon < open_brace) {
            continue;
        }
        let end = tail.find("\n}\n").map_or(tail.len(), |end| end + 3);
        return Some(&tail[..end]);
    }
    None
}

fn rtl_candidates(db: &DecompileDB, address: Address) -> Vec<RTLInst> {
    db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(row_address, inst)| (*row_address == address).then_some(inst.clone()))
        .collect()
}

fn test_rtl_inst_uses(inst: &RTLInst, value: RTLReg) -> bool {
    match inst {
        RTLInst::Iop(_, args, _)
        | RTLInst::Iload(_, _, args, _)
        | RTLInst::Icond(_, args, _, _) => args.contains(&value),
        RTLInst::Istore(_, _, args, source) => {
            args.contains(&value) || *source == value
        }
        RTLInst::Icall(_, callee, args, _, _)
        | RTLInst::Itailcall(_, callee, args) => {
            matches!(callee, either::Either::Left(reg) if *reg == value)
                || args.contains(&value)
        }
        RTLInst::Ijumptable(reg, _) | RTLInst::Ireturn(reg) => *reg == value,
        _ => false,
    }
}

fn address_bearing_candidates(db: &DecompileDB, address: Address) -> Vec<RTLInst> {
    db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(node, inst)| {
            ((*node & !SYNTHETIC_NODE_MASK) == address
                && matches!(
                    inst,
                    RTLInst::Iload(..)
                        | RTLInst::Istore(..)
                        | RTLInst::Iop(Operation::Olea(_) | Operation::Oleal(_), _, _)
                ))
            .then_some(inst.clone())
        })
        .collect()
}

fn assert_sp_indexed_relations(db: &DecompileDB) {
    let span = function_span(db, "sp_indexed_alias_collision");
    let mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, span))
        .cloned()
        .collect();

    let indexed_stores: Vec<_> = mach
        .iter()
        .filter(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mstore(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::SP, Mreg::AX]
            )
        })
        .collect();
    let indexed_loads: Vec<_> = mach
        .iter()
        .filter(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::SP, Mreg::AX]
            )
        })
        .collect();
    assert_eq!(
        indexed_stores.len(),
        1,
        "missing/duplicate SP indexed store: {mach:#x?}"
    );
    assert_eq!(
        indexed_loads.len(),
        1,
        "missing/duplicate SP indexed load: {mach:#x?}"
    );

    let store_addr = indexed_stores[0].0;
    let load_addr = indexed_loads[0].0;
    for address in [store_addr, load_addr] {
        assert!(
            !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, access, reason)| {
                    (*func, *access, *reason) == (span.0, address, "unsupported-stack-address")
                }),
            "proved indexed RSP access was marked unsupported at {address:#x}"
        );
    }

    let store_candidates = rtl_candidates(db, store_addr | SYNTH1);
    let load_candidates = rtl_candidates(db, load_addr | SYNTH1);
    let store_reg_rtls: Vec<_> = db
        .rel_iter::<(Address, Mreg, RTLReg)>("reg_rtl")
        .filter(|(address, _, _)| *address == store_addr)
        .copied()
        .collect();
    let store_reg_xtls: Vec<_> = db
        .rel_iter::<(Address, Mreg, RTLReg)>("reg_xtl")
        .filter(|(address, _, _)| *address == store_addr)
        .copied()
        .collect();
    let store_reaching_defs: Vec<_> = db
        .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
        .filter(|(_, _, use_address)| *use_address == store_addr)
        .copied()
        .collect();
    assert_eq!(
        store_candidates
            .iter()
            .filter(|inst| matches!(inst, RTLInst::Istore(..)))
            .count(),
        1,
        "SP indexed store must have one synthetic Istore: {store_candidates:#x?}; \
         reg_rtl={store_reg_rtls:#x?}; reg_xtl={store_reg_xtls:#x?}; \
         reaching={store_reaching_defs:#x?}"
    );
    assert!(store_candidates.iter().any(|inst| {
        matches!(
            inst,
            RTLInst::Istore(_, Addressing::Aindexed2scaled(4, 0), args, src)
                if args.len() == 2 && args[1] == *src
        )
    }));
    assert_eq!(
        load_candidates
            .iter()
            .filter(|inst| matches!(inst, RTLInst::Iload(..)))
            .count(),
        1,
        "SP dst/index collision must have one synthetic Iload: {load_candidates:#x?}"
    );
    assert!(load_candidates.iter().any(|inst| {
        matches!(
            inst,
            RTLInst::Iload(_, Addressing::Aindexed2scaled(4, 0), args, dst)
                if args.len() == 2 && args[1] != *dst
        )
    }));

    let scalar_addr = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span) && matches!(inst, LTLInst::Lsetstack(_, _, 8, _)))
                .then_some(*address)
        })
        .expect("missing scalar SP slot used to anchor indexed base");
    let scalar_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == span.0 && *address == scalar_addr && *offset == 8).then_some(*reg)
        })
        .collect();
    let indexed_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == span.0 && *address == store_addr && *offset == 8).then_some(*reg)
        })
        .collect();
    assert!(
        !scalar_vars.is_disjoint(&indexed_vars),
        "synthetic SP base must alias the same coordinate as the scalar local: \
         scalar={scalar_vars:#x?}, indexed={indexed_vars:#x?}"
    );
}

fn assert_incomplete_indexed_stack_lowering_is_atomic(db: &DecompileDB) {
    let span = function_span(db, "sp_indexed_ambiguous_reaching");
    let access = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span)
                && matches!(inst,
                    MachInst::Mload(
                        MemoryChunk::MInt32,
                        Addressing::Aindexed2scaled(4, 8),
                        args,
                        Mreg::DX
                    ) if args.as_ref() == &[Mreg::SP, Mreg::AX]))
            .then_some(*address)
        })
        .expect("missing unresolved indexed-stack load fixture");
    assert!(
        db.rel_iter::<(Address,)>("sp_indexed_load")
            .any(|(address,)| *address == access),
        "structural indexed load missing at {access:#x}: ltl={:#x?}, indexed={:#x?}, rsp={:#x?}",
        db.rel_iter::<(Address, LTLInst)>("ltl_inst")
            .filter(|(address, _)| *address == access)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Mreg, i64, usize)>("indexed_stack_operand")
            .filter(|(address, _, _, _)| *address == access)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Address)>("rsp_frame_at")
            .filter(|(address, _)| *address == access)
            .collect::<Vec<_>>()
    );
    assert!(!db
        .rel_iter::<(Address,)>("sp_indexed_load_complete")
        .any(|(address,)| *address == access));
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(func, address, reason)| {
            (*func, *address, *reason)
                == (span.0, access, "unsupported-stack-address")
        }));

    let rooted: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access)
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert_eq!(
        rooted,
        vec![(access, RTLInst::Inop)],
        "incomplete indexed lowering was not collapsed atomically"
    );
    assert!(!db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .any(|(node, _)| *node != access && (*node & !SYNTHETIC_NODE_MASK) == access));
    assert!(!db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .any(|(source, destination)| {
            (*source != access && (*source & !SYNTHETIC_NODE_MASK) == access)
                || (*destination != access
                    && (*destination & !SYNTHETIC_NODE_MASK) == access)
        }));
    for relation in ["stack_xtl", "stack_var"] {
        assert!(!db
            .rel_iter::<(Address, Address, i64, RTLReg)>(relation)
            .any(|(_, node, _, _)| (*node & !SYNTHETIC_NODE_MASK) == access));
    }
    assert!(!db
        .rel_iter::<(Address, Address, RTLReg, i64)>("normalized_stack_lea_base")
        .any(|(_, node, _, _)| (*node & !SYNTHETIC_NODE_MASK) == access));
    let raw_exits: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_next")
        .filter(|(source, _)| *source == access)
        .copied()
        .collect();
    let restored_exits: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .filter(|(source, _)| *source == access)
        .copied()
        .collect();
    assert!(
        !restored_exits.is_empty(),
        "rejected indexed lowering left the replacement Inop as a dead end"
    );
    assert!(restored_exits.iter().all(|(_, destination)| {
        *destination == (*destination & !SYNTHETIC_NODE_MASK)
    }));
    for edge in raw_exits {
        assert!(db
            .rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .any(|candidate| *candidate == edge));
        assert!(!db
            .rel_iter::<(Address, Address)>("rtl_edge_negated")
            .any(|negated| *negated == edge));
    }
}

fn assert_ambiguous_fused_indexed_lowering_is_atomic(db: &DecompileDB) {
    let span = function_span(db, "sp_indexed_fused_ambiguous");
    let access = db
        .rel_iter::<(Address, Operation, MemoryChunk, i64, i64, Mreg, Mreg)>(
            "sp_indexed_fused_load",
        )
        .find_map(|(address, op, chunk, scale, displacement, index, destination)| {
            (in_span(*address, span)
                && *op == Operation::Oadd
                && *chunk == MemoryChunk::MInt32
                && (*scale, *displacement, *index, *destination)
                    == (4, 8, Mreg::AX, Mreg::R9))
            .then_some(*address)
        })
        .expect("missing ambiguous fused indexed-stack fixture");
    assert!(!db
        .rel_iter::<(Address,)>("sp_indexed_fused_complete")
        .any(|(address,)| *address == access));
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(func, address, reason)| {
            (*func, *address, *reason)
                == (span.0, access, "unsupported-stack-address")
        }));

    let rooted: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access)
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert_eq!(rooted, vec![(access, RTLInst::Inop)]);
    assert!(!db
        .rel_iter::<(Address, Address)>("sp_indexed_fused_member")
        .any(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access));
    assert!(!db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .any(|(source, destination)| {
            (*source != access && (*source & !SYNTHETIC_NODE_MASK) == access)
                || (*destination != access
                    && (*destination & !SYNTHETIC_NODE_MASK) == access)
        }));

    let partial_span = function_span(db, "sp_indexed_fused_partial_def");
    let partial = db
        .rel_iter::<(Address, Operation, MemoryChunk, i64, i64, Mreg, Mreg)>(
            "sp_indexed_fused_load",
        )
        .find_map(|(address, op, chunk, scale, displacement, index, destination)| {
            (in_span(*address, partial_span)
                && *op == Operation::Oadd
                && *chunk == MemoryChunk::MInt32
                && (*scale, *displacement, *index, *destination)
                    == (4, 8, Mreg::CX, Mreg::R9))
            .then_some(*address)
        })
        .expect("missing one-arm-defined fused indexed-stack fixture");
    assert!(!db
        .rel_iter::<(Address, Mreg, Address)>("abi_livein_reaches_use")
        .any(|row| *row == (partial_span.0, Mreg::CX, partial)));
    let real_origins: HashSet<_> = db
        .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
        .filter_map(|(definition, mreg, usage)| {
            (*usage == partial && *mreg == Mreg::CX && *definition != partial)
                .then_some(*definition)
        })
        .collect();
    assert_eq!(
        real_origins.len(),
        1,
        "partial-def fixture needs exactly one real reaching origin: {real_origins:#x?}"
    );
    let definition = *real_origins.iter().next().unwrap();
    assert!(!db
        .rel_iter::<(Address, Address, Address)>("reg_def_dominates_use")
        .any(|row| *row == (partial_span.0, definition, partial)));
    assert!(!db
        .rel_iter::<(Address,)>("sp_indexed_fused_complete")
        .any(|(address,)| *address == partial));
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(func, address, reason)| {
            (*func, *address, *reason)
                == (partial_span.0, partial, "unsupported-stack-address")
        }));
    let partial_rooted: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == partial)
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert_eq!(partial_rooted, vec![(partial, RTLInst::Inop)]);
}

fn assert_sp_indexed_fused_arithmetic(db: &DecompileDB) {
    type FusedRow = (
        Address,
        Operation,
        MemoryChunk,
        Addressing,
        Arc<Vec<Mreg>>,
        Mreg,
        bool,
    );

    let span = function_span(db, "sp_indexed_fused_arith");
    let rows: Vec<_> = db
        .rel_iter::<FusedRow>("float_load_op")
        .filter(|(address, _, _, addressing, args, dst, unary)| {
            in_span(*address, span)
                && *addressing == Addressing::Aindexed2scaled(4, 8)
                && args.as_ref() == &[Mreg::SP, Mreg::DX]
                && *dst == Mreg::AX
                && !*unary
        })
        .cloned()
        .collect();
    assert_eq!(rows.len(), 5, "missing indexed-RSP fused rows: {rows:#x?}");

    for expected_op in [
        Operation::Oadd,
        Operation::Osub,
        Operation::Oand,
        Operation::Oor,
        Operation::Oxor,
    ] {
        let (address, _, chunk, _, _, _, _) = rows
            .iter()
            .find(|(_, op, _, _, _, _, _)| *op == expected_op)
            .unwrap_or_else(|| panic!("missing {expected_op:?}: {rows:#x?}"));
        assert_eq!(*chunk, MemoryChunk::MInt32);
        assert!(
            !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, access, _)| (*func, *access) == (span.0, *address)),
            "proved indexed-RSP fused op was rejected at {address:#x}: index_reaching={:#x?}, dst_reaching={:#x?}, dst_defs={:#x?}, dst_xtl={:#x?}, local_defs={:#x?}",
            db.rel_iter::<(Address, Mreg, RTLReg)>("reaching_use_rtl")
                .filter(|(node, mreg, _)| *node == *address && *mreg == Mreg::DX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg, RTLReg)>("reaching_use_rtl")
                .filter(|(node, mreg, _)| *node == *address && *mreg == Mreg::AX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg, Address)>("reg_def_used")
                .filter(|(_, mreg, use_node)| *use_node == *address && *mreg == Mreg::AX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg, RTLReg)>("reg_xtl")
                .filter(|(node, mreg, _)| *node <= *address && *mreg == Mreg::AX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, RTLReg)>("is_def")
                .filter(|(node, _)| *node <= *address)
                .collect::<Vec<_>>()
        );
        let exact_dst_reaching: HashSet<_> = db
            .rel_iter::<(Address, Mreg, RTLReg)>("reaching_use_rtl")
            .filter_map(|(node, mreg, value)| {
                (*node == *address && *mreg == Mreg::AX).then_some(*value)
            })
            .collect();
        assert_eq!(
            exact_dst_reaching.len(),
            1,
            "exact prior AX definition retained historical canonicals at {address:#x}: \
             {exact_dst_reaching:#x?}"
        );

        let real = rtl_candidates(db, *address);
        let sp_base = real
            .iter()
            .find_map(|inst| match inst {
                RTLInst::Iop(Operation::Olea(Addressing::Ainstack(8)), args, dst)
                    if args.is_empty() =>
                {
                    Some(*dst)
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing synthetic stack base at {address:#x}: {real:#x?}"));

        let synth1 = *address | SYNTH1;
        let load_candidates = rtl_candidates(db, synth1);
        let (index, temp) = load_candidates
            .iter()
            .find_map(|inst| match inst {
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 0),
                    args,
                    dst,
                ) if args.len() == 2 && args[0] == sp_base => Some((args[1], *dst)),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("missing indexed stack load at {synth1:#x}: {load_candidates:#x?}")
            });
        assert_ne!(index, temp, "indexed load overwrote its address input");

        let synth2 = *address | (1u64 << 63);
        let op_candidates = rtl_candidates(db, synth2);
        assert!(op_candidates.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(op, args, dst)
                if *op == expected_op && args.len() == 2
                    && args[0] == *dst && args[1] == temp)
        }), "missing fused arithmetic at {synth2:#x}: {op_candidates:#x?}");

        let next = db
            .rel_iter::<(Address, Address)>("next")
            .find_map(|(source, target)| (*source == *address).then_some(*target))
            .expect("fused instruction has no linear successor");
        let edges: HashSet<_> = db
            .rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .copied()
            .collect();
        assert!(edges.contains(&(*address, synth1)));
        assert!(edges.contains(&(synth1, synth2)));
        assert!(edges.contains(&(synth2, next)));
        assert!(
            !edges.contains(&(*address, next)),
            "fused operation retained a direct skip edge"
        );
        assert!(db
            .rel_iter::<(Address, Address)>("rtl_edge_negated")
            .any(|row| *row == (*address, next)));
        assert!(db
            .rel_iter::<(Address, Address)>("instr_in_function")
            .any(|row| *row == (synth1, span.0)));
        assert!(db
            .rel_iter::<(Address, Address)>("instr_in_function")
            .any(|row| *row == (synth2, span.0)));
    }

    let collision = function_span(db, "sp_indexed_fused_add_collision");
    let collision_row = db
        .rel_iter::<FusedRow>("float_load_op")
        .find(|(address, op, _, addressing, args, dst, unary)| {
            in_span(*address, collision)
                && *op == Operation::Oadd
                && *addressing == Addressing::Aindexed2scaled(4, 8)
                && args.as_ref() == &[Mreg::SP, Mreg::AX]
                && *dst == Mreg::AX
                && !*unary
        })
        .expect("missing destination/index collision row");
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, access, _)| (*func, *access) == (collision.0, collision_row.0))
    );
    let collision_loads = rtl_candidates(db, collision_row.0 | SYNTH1);
    let collision_ops = rtl_candidates(db, collision_row.0 | (1u64 << 63));
    let (index, temp) = collision_loads
        .iter()
        .find_map(|inst| match inst {
            RTLInst::Iload(_, Addressing::Aindexed2scaled(4, 0), args, dst)
                if args.len() == 2 =>
            {
                Some((args[1], *dst))
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("collision load vanished: {collision_loads:#x?}"));
    assert!(collision_ops.iter().any(|inst| {
        matches!(inst, RTLInst::Iop(Operation::Oadd, args, dst)
            if args.as_ref() == &[index, temp] && *dst == index)
    }), "destination/index collision lost its incoming value: {collision_ops:#x?}");

    let field_span = function_span(db, "sp_indexed_fused_field");
    let field_addr = db
        .rel_iter::<FusedRow>("float_load_op")
        .find_map(|(address, op, _, addressing, args, dst, unary)| {
            (in_span(*address, field_span)
                && *op == Operation::Oadd
                && *addressing == Addressing::Aindexed2scaled(8, 12)
                && args.as_ref() == &[Mreg::SP, Mreg::DX]
                && *dst == Mreg::AX
                && !*unary)
                .then_some(*address)
        })
        .expect("missing field-displacement fused row");
    let field_real = rtl_candidates(db, field_addr);
    let field_load = rtl_candidates(db, field_addr | SYNTH1);
    let field_op = rtl_candidates(db, field_addr | (1u64 << 63));
    assert_eq!(field_real.len(), 1, "field access retained competing root candidates: {field_real:#x?}");
    assert_eq!(field_load.len(), 1, "field access retained competing loads: {field_load:#x?}");
    assert_eq!(field_op.len(), 1, "field access retained competing operations: {field_op:#x?}");
    assert!(field_real.iter().any(|inst| {
        matches!(inst,
            RTLInst::Iop(Operation::Olea(Addressing::Ainstack(8)), args, _)
                if args.is_empty())
    }), "field access did not canonicalize to its stride base");
    assert!(field_load.iter().any(|inst| {
        matches!(inst,
            RTLInst::Iload(MemoryChunk::MInt32, Addressing::Aindexed2scaled(8, 4), args, _)
                if args.len() == 2)
    }), "field access did not preserve its inner displacement");
    let mut field_members: Vec<_> = db
        .rel_iter::<(Address, Address)>("sp_indexed_fused_member")
        .filter(|(node, func)| {
            *func == field_span.0 && (*node & !SYNTHETIC_NODE_MASK) == field_addr
        })
        .copied()
        .collect();
    field_members.sort_unstable();
    field_members.dedup();
    assert_eq!(field_members.len(), 2,
               "field chain must own exactly two synthetic members: {field_members:#x?}");

    let misaligned_span = function_span(db, "sp_indexed_fused_misaligned_field");
    let misaligned_addr = db
        .rel_iter::<FusedRow>("float_load_op")
        .find_map(|(address, op, _, addressing, args, dst, unary)| {
            (in_span(*address, misaligned_span)
                && *op == Operation::Oadd
                && *addressing == Addressing::Aindexed2scaled(8, 8)
                && args.as_ref() == &[Mreg::SP, Mreg::DX]
                && *dst == Mreg::AX
                && !*unary)
                .then_some(*address)
        })
        .expect("missing misaligned field-displacement fused row");
    let misaligned_real = rtl_candidates(db, misaligned_addr);
    assert!(misaligned_real.iter().any(|inst| {
        matches!(inst,
            RTLInst::Iop(Operation::Olea(Addressing::Ainstack(4)), args, _)
                if args.is_empty())
    }), "misaligned aggregate did not retain its evidenced phase: {misaligned_real:#x?}");
    assert!(!misaligned_real.iter().any(|inst| {
        matches!(inst,
            RTLInst::Iop(Operation::Olea(Addressing::Ainstack(0)), _, _))
    }), "misaligned aggregate was falsely rebased onto the adjacent scalar");
    assert!(rtl_candidates(db, misaligned_addr | SYNTH1).iter().any(|inst| {
        matches!(inst,
            RTLInst::Iload(MemoryChunk::MInt32, Addressing::Aindexed2scaled(8, 4), args, _)
                if args.len() == 2)
    }), "misaligned aggregate field lost its exact effective address");

    let param_span = function_span(db, "sp_indexed_fused_param_rmw");
    let param_addr = db
        .rel_iter::<FusedRow>("float_load_op")
        .find_map(|(address, op, _, _, args, dst, unary)| {
            (in_span(*address, param_span)
                && *op == Operation::Oadd
                && args.as_ref() == &[Mreg::SP, Mreg::CX]
                && *dst == Mreg::CX
                && !*unary)
                .then_some(*address)
        })
        .expect("missing direct parameter RMW fused row");
    let param_loads = rtl_candidates(db, param_addr | SYNTH1);
    let param_ops = rtl_candidates(db, param_addr | (1u64 << 63));
    assert_eq!(param_loads.len(), 1,
               "direct integer parameter RMW retained competing loads: {param_loads:#x?}");
    assert_eq!(param_ops.len(), 1,
               "direct integer parameter RMW retained competing value webs: {param_ops:#x?}");
    let (param_index, param_temp) = match &param_loads[0] {
        RTLInst::Iload(_, Addressing::Aindexed2scaled(4, 0), args, temp)
            if args.len() == 2 => (args[1], *temp),
        other => panic!("unexpected direct parameter RMW load: {other:#x?}"),
    };
    assert!(matches!(&param_ops[0],
        RTLInst::Iop(Operation::Oadd, args, dst)
            if args.as_ref() == &[param_index, param_temp] && *dst == param_index),
        "direct integer parameter RMW did not update its incoming web: {param_ops:#x?}");

    let gap_span = function_span(db, "sp_indexed_fused_gap");
    let mut gap_ops: Vec<_> = db
        .rel_iter::<FusedRow>("float_load_op")
        .filter_map(|(address, op, _, _, args, dst, unary)| {
            (in_span(*address, gap_span)
                && matches!(op, Operation::Oadd | Operation::Osub)
                && args.as_ref() == &[Mreg::SP, Mreg::DX]
                && *dst == Mreg::AX
                && !*unary)
                .then_some(*address)
        })
        .collect();
    gap_ops.sort_unstable();
    assert_eq!(gap_ops.len(), 2, "missing adjacent fused operations: {gap_ops:#x?}");
    let gap_edges: HashSet<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .copied()
        .collect();
    assert!(gap_edges.contains(&(gap_ops[0] | (1u64 << 63), gap_ops[1])),
            "tail bridge skipped the next fused operation: {gap_edges:#x?}");
}

fn assert_indexed_entry_and_loop_collision_matrix(db: &DecompileDB) {
    type FusedShape = (
        Address,
        Operation,
        MemoryChunk,
        i64,
        i64,
        Mreg,
        Mreg,
    );

    for (name, expect_back_edge) in [
        ("sp_indexed_fused_entry_collision", false),
        ("sp_indexed_fused_loop_collision", true),
    ] {
        let span = function_span(db, name);
        let access = db
            .rel_iter::<FusedShape>("sp_indexed_fused_load")
            .find_map(|(node, op, chunk, scale, displacement, index, destination)| {
                (in_span(*node, span)
                    && *op == Operation::Oadd
                    && *chunk == MemoryChunk::MInt32
                    && (*scale, *displacement, *index, *destination)
                        == (4, 40, Mreg::CX, Mreg::CX))
                .then_some(*node)
            })
            .unwrap_or_else(|| panic!("missing {name} fused collision"));
        if expect_back_edge {
            assert_ne!(access, span.0, "{name} needs a distinct loop header");
        } else {
            assert_eq!(access, span.0, "{name} must exercise a literal entry node");
        }
        assert!(
            db.rel_iter::<(Address, Mreg, Address)>("abi_livein_reaches_use")
                .any(|row| *row == (span.0, Mreg::CX, access)),
            "{name} lost its ABI CX input: validated={:#x?}, live={:#x?}, uses={:#x?}, defs={:#x?}",
            db.rel_iter::<(Address, Mreg)>("func_param_validated")
                .filter(|(function, mreg)| *function == span.0 && *mreg == Mreg::CX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Address, Mreg)>("arg_reg_param_live_at")
                .filter(|(function, node, mreg)| {
                    *function == span.0 && *node == access && *mreg == Mreg::CX
                })
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg)>("reg_use")
                .filter(|(node, mreg)| *node == access && *mreg == Mreg::CX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg)>("reg_def")
                .filter(|(node, mreg)| *node == access && *mreg == Mreg::CX)
                .collect::<Vec<_>>()
        );
        assert!(db
            .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
            .any(|row| *row == (span.0, Mreg::CX, access)));
        if expect_back_edge {
            assert!(db
                .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
                .any(|row| *row == (access, Mreg::CX, access)));
        }
        assert!(
            db.rel_iter::<(Address,)>("sp_indexed_fused_complete")
                .any(|(node,)| *node == access),
            "{name} fused collision was not complete: reaching={:#x?}, origins={:#x?}, candidates={:#x?}, seeds={:#x?}, unsupported={:#x?}",
            db.rel_iter::<(Address, Mreg, RTLReg)>("reaching_use_rtl")
                .filter(|(node, mreg, _)| *node == access && *mreg == Mreg::CX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Mreg, Address)>("reg_def_used")
                .filter(|(_, mreg, usage)| *usage == access && *mreg == Mreg::CX)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
                .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address_seed")
                .filter(|(_, node, _)| *node == access)
                .collect::<Vec<_>>(),
            db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .filter(|(_, node, _)| *node == access)
                .collect::<Vec<_>>()
        );
        assert!(!db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, node, _)| (*func, *node) == (span.0, access)));
        assert!(!db
            .rel_iter::<(Address,)>("sp_indexed_load_complete")
            .any(|(node,)| *node == access));

        let reaching: HashSet<_> = db
            .rel_iter::<(Address, Mreg, RTLReg)>("reaching_use_rtl")
            .filter_map(|(node, mreg, value)| {
                (*node == access && *mreg == Mreg::CX).then_some(*value)
            })
            .collect();
        assert_eq!(
            reaching.len(),
            1,
            "{name} retained historical entry/self representatives: {reaching:#x?}"
        );
        assert_eq!(rtl_candidates(db, access).len(), 1);
        let loads = rtl_candidates(db, access | SYNTH1);
        assert_eq!(loads.len(), 1, "{name} retained competing fused loads");
        let temp = match &loads[0] {
            RTLInst::Iload(_, _, args, destination) if args.len() == 2 => {
                assert_ne!(args[1], *destination);
                *destination
            }
            other => panic!("{name} has the wrong fused load: {other:#x?}"),
        };
        let operations = rtl_candidates(db, access | (1u64 << 63));
        assert_eq!(
            operations.len(),
            1,
            "{name} retained competing fused operations"
        );
        assert!(matches!(
            &operations[0],
            RTLInst::Iop(Operation::Oadd, args, destination)
                if args.len() == 2
                    && args[0] == *destination
                    && args[1] == temp
        ));
        let has_back_edge = db
            .rel_iter::<(Address, Address)>("rtl_next")
            .any(|(source, destination)| *source != access && *destination == access);
        assert_eq!(has_back_edge, expect_back_edge, "{name} CFG shape changed");
    }

    for (name, mreg, expect_back_edge) in [
        ("sp_indexed_load_entry_collision", Mreg::CX, false),
        ("sp_indexed_load_loop_collision", Mreg::AX, true),
    ] {
        let span = function_span(db, name);
        let access = db
            .rel_iter::<(Address,)>("sp_indexed_load")
            .find_map(|(node,)| in_span(*node, span).then_some(*node))
            .unwrap_or_else(|| panic!("missing {name} ordinary indexed load"));
        if expect_back_edge {
            assert_ne!(access, span.0, "{name} needs a distinct loop header");
        } else {
            assert_eq!(access, span.0, "{name} must exercise a literal entry node");
        }
        assert!(db
            .rel_iter::<(Address, Mreg)>("load_overwrites_base")
            .any(|row| *row == (access, mreg)));
        assert!(db
            .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
            .any(|row| *row == (access, mreg, access)));
        if expect_back_edge {
            assert!(db
                .rel_iter::<(Address, Mreg, Address)>("reg_def_used")
                .any(|row| *row == (span.0, Mreg::AX, access)));
        }
        assert!(!db
            .rel_iter::<(Address,)>("sp_indexed_load_complete")
            .any(|(node,)| *node == access));
        assert!(db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, node, reason)| {
                (*func, *node, *reason)
                    == (span.0, access, "unsupported-stack-address")
            }));
        let rooted: Vec<_> = db
            .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
            .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access)
            .map(|(node, inst)| (*node, inst.clone()))
            .collect();
        assert_eq!(rooted, vec![(access, RTLInst::Inop)]);
        let has_back_edge = db
            .rel_iter::<(Address, Address)>("rtl_next")
            .any(|(source, destination)| *source != access && *destination == access);
        assert_eq!(has_back_edge, expect_back_edge, "{name} CFG shape changed");
    }
}

fn assert_home_relations(db: &DecompileDB) {
    let span = function_span(db, "home_alias_roundtrip");
    let spill_candidates: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    let reload_cells: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload_cell")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    let moves: Vec<_> = db
        .rel_iter::<(Address, Symbol, Symbol)>("pmov")
        .filter(|(address, _, _)| in_span(*address, span))
        .copied()
        .collect();
    let aliases: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .filter(|(func, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let live_args: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg)>("arg_reg_param_live_at")
        .filter(|(func, _, _)| *func == span.0)
        .copied()
        .collect();
    let ltl: Vec<_> = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .filter(|(address, _)| in_span(*address, span))
        .cloned()
        .collect();
    let spills: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(
        spills.len(),
        1,
        "expected one exact /homeparams spill: spills={spills:#x?}, candidates={spill_candidates:#x?}, reload_cells={reload_cells:#x?}, moves={moves:#x?}, aliases={aliases:#x?}, live_args={live_args:#x?}, ltl={ltl:#x?}"
    );
    let (spill_addr, _, source, position) = spills[0];
    assert_eq!((source, position), (Mreg::DX, 1));
    assert!(db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .any(|(func, use_addr, alias, base)| {
            (*func, *use_addr, *alias, *base) == (span.0, spill_addr, Mreg::AX, 0)
        }));

    let reloads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(reloads.len(), 1, "expected one home reload: {reloads:#x?}");
    let (reload_addr, _, reload_source, reload_pos) = reloads[0];
    assert_eq!((reload_source, reload_pos), (Mreg::DX, 1));
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, _, _)| *func == span.0),
        "a safe immutable /homeparams spill/reload was rejected as unsupported"
    );

    let spill_candidates = rtl_candidates(db, spill_addr);
    assert!(spill_candidates.contains(&RTLInst::Inop));
    assert!(
        !spill_candidates
            .iter()
            .any(|inst| matches!(inst, RTLInst::Istore(..))),
        "home spill must not survive as an address-valued Istore: {spill_candidates:#x?}"
    );
    let entry_dx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| (*address == span.0 && *reg == Mreg::DX).then_some(*id))
        .collect();
    let reload_candidates = rtl_candidates(db, reload_addr);
    assert!(reload_candidates.iter().any(|inst| {
        matches!(inst, RTLInst::Iop(Operation::Omove, args, _) if args.len() == 1 && entry_dx.contains(&args[0]))
    }));
    assert!(
        !reload_candidates
            .iter()
            .any(|inst| matches!(inst, RTLInst::Iload(..))),
        "home reload must be a parameter value move, not a frame load: {reload_candidates:#x?}"
    );

    let wrong = function_span(db, "wrong_home_source");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, _, _)| *func == wrong.0));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, _, _)| *func == wrong.0));

    let reassigned = function_span(db, "home_reassigned");
    assert!(db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
        .any(|(_, func, source, position)| {
            (*func, *source, *position) == (reassigned.0, Mreg::CX, 0)
        }));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, _, _)| *func == reassigned.0));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, _, _)| *func == reassigned.0));
    let reassigned_reload = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, reassigned)
                && matches!(inst, MachInst::Mgetstack(8, Typ::Tany64, Mreg::AX)))
            .then_some(*address)
        })
        .expect("missing reassigned home-slot reload");
    let reassigned_entry_cx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| {
            (*address == reassigned.0 && *reg == Mreg::CX).then_some(*id)
        })
        .collect();
    assert!(!reassigned_entry_cx.is_empty());
    let reassigned_candidates = rtl_candidates(db, reassigned_reload);
    assert!(
        !reassigned_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(Operation::Omove, args, _)
                    if args.len() == 1 && reassigned_entry_cx.contains(&args[0])
            )
        }),
        "a reassigned home slot must read its reaching value, not the entry parameter: {reassigned_candidates:#x?}"
    );
}

fn assert_home_pointer_reload_beats_shadow_address(db: &DecompileDB) {
    let span = function_span(db, "home_pointer_indexed");
    let reloads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .filter(|(_, func, _, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(
        reloads.len(),
        1,
        "expected one pointer home reload: {reloads:#x?}"
    );
    let (reload_addr, _, source, position) = reloads[0];
    assert_eq!((source, position), (Mreg::CX, 0));

    let reload_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| *address == reload_addr)
        .map(|(_, inst)| inst.clone())
        .collect();
    assert!(
        reload_mach
            .iter()
            .any(|inst| matches!(inst, MachInst::Mgetstack(8, Typ::Tany64, Mreg::AX))),
        "pointer home reload lost its value interpretation: {reload_mach:#x?}"
    );
    assert!(
        reload_mach.iter().any(|inst| {
            matches!(
                inst,
                MachInst::Mop(Operation::Olea(Addressing::Ainstack(8)), args, Mreg::AX)
                    if args.is_empty()
            )
        }),
        "fixture must retain the competing address interpretation: {reload_mach:#x?}"
    );

    let entry_cx: HashSet<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, reg, id)| (*address == span.0 && *reg == Mreg::CX).then_some(*id))
        .collect();
    let reload_candidates = rtl_candidates(db, reload_addr);
    let reload_value = reload_candidates
        .iter()
        .find_map(|inst| match inst {
            RTLInst::Iop(Operation::Omove, args, dst)
                if args.len() == 1 && entry_cx.contains(&args[0]) =>
            {
                Some(*dst)
            }
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!("home pointer reload did not select the incoming value: {reload_candidates:#x?}")
        });
    assert!(
        !reload_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(Operation::Olea(Addressing::Ainstack(8)), args, _)
                    if args.is_empty()
            )
        }),
        "home pointer reload kept a competing stack address: {reload_candidates:#x?}"
    );

    let indexed_addr = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span)
                && matches!(
                    inst,
                    MachInst::Mload(
                        MemoryChunk::MInt32,
                        Addressing::Aindexed2scaled(4, 0),
                        args,
                        Mreg::R8
                    ) if args.as_ref() == &[Mreg::AX, Mreg::DX]
                ))
            .then_some(*address)
        })
        .expect("missing pointer/index load after the home reload");
    let indexed_candidates = rtl_candidates(db, indexed_addr);
    assert!(
        indexed_candidates.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 0),
                    args,
                    _
                ) if args.len() == 2 && args[0] == reload_value
            )
        }),
        "indexed dereference did not consume the homed pointer value: reload={reload_candidates:#x?}, indexed={indexed_candidates:#x?}"
    );
}

fn assert_stack_param_and_rmw_relations(db: &DecompileDB) {
    let rmw = function_span(db, "sp_immediate_rmw");
    let inits: Vec<_> = db
        .rel_iter::<(Address, i64, i64, Typ)>("mach_imm_stack_init")
        .filter(|(address, _, _, _)| in_span(*address, rmw))
        .copied()
        .collect();
    assert!(
        inits
            .iter()
            .any(|(_, offset, value, _)| (*offset, *value) == (12, 7)),
        "SP immediate local must retain its raw displacement: {inits:#x?}"
    );
    assert!(!db
        .rel_iter::<(Address, usize)>("stack_param_ordinal")
        .any(|(func, _)| *func == rmw.0));

    let load_rows: Vec<_> = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64, Mreg)>("arith_load_op")
        .filter(|(address, _, _, _, _, _)| in_span(*address, rmw))
        .cloned()
        .collect();
    let (load_addr, load_op) = load_rows
        .iter()
        .find_map(|(address, op, _, base, disp, _)| {
            (in_span(*address, rmw) && *base == Mreg::SP && *disp == 12 && *op == Operation::Osub)
                .then_some((*address, op.clone()))
        })
        .unwrap_or_else(|| panic!("missing SP local memory-source RMW: {load_rows:#x?}"));
    let (store_addr, store_op) = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64)>("arith_store_imm")
        .find_map(|(address, op, _, base, disp)| {
            (in_span(*address, rmw) && *base == Mreg::SP && *disp == 12)
                .then_some((*address, op.clone()))
        })
        .expect("missing SP local memory-destination RMW");
    let load_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == rmw.0 && *address == load_addr && *offset == 12).then_some(*reg)
        })
        .collect();
    let store_vars: HashSet<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, address, offset, reg)| {
            (*func == rmw.0 && *address == store_addr && *offset == 12).then_some(*reg)
        })
        .collect();
    assert!(!load_vars.is_empty() && !store_vars.is_empty());
    assert!(rtl_candidates(db, load_addr | SYNTH1).iter().any(|inst| {
        matches!(inst, RTLInst::Iop(op, args, _) if *op == load_op && args.len() == 2 && load_vars.contains(&args[1]))
    }));
    assert!(rtl_candidates(db, store_addr).iter().any(|inst| {
        matches!(inst, RTLInst::Iop(op, args, dst) if *op == store_op && args.len() == 1 && store_vars.contains(&args[0]) && store_vars.contains(dst))
    }));

    let fifth = function_span(db, "fifth_imul");
    let sixth = function_span(db, "sixth_to_double");
    let fifth_access = db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .find(|(_, func, disp, ordinal)| *func == fifth.0 && *disp == 40 && *ordinal == 0)
        .copied()
        .expect("fifth argument must retain ABI ordinal zero");
    let sixth_access = db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .find(|(_, func, disp, ordinal)| *func == sixth.0 && *disp == 48 && *ordinal == 1)
        .copied()
        .expect("sparse sixth argument must retain ABI ordinal one");
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (fifth.0, 1)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (sixth.0, 2)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (fifth.0, 5)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (sixth.0, 6)));

    let boundary = function_span(db, "boundary_stack_offset");
    let boundary_access = db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .find(|(_, func, disp, ordinal)| (*func, *disp, *ordinal) == (boundary.0, 544, 63))
        .copied()
        .expect("ordinal 63 must remain inside the stack-parameter cap");
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (boundary.0, 64)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (boundary.0, 68)));
    let boundary_param = rtl_candidates(db, boundary_access.0)
        .iter()
        .find_map(|inst| match inst {
            RTLInst::Iop(Operation::Omove, args, _) if args.len() == 1 => Some(args[0]),
            _ => None,
        })
        .expect("ordinal-63 load did not consume its synthetic parameter");
    assert!(db
        .rel_iter::<(Address, u64, XType)>("emit_function_param_type_candidate")
        .any(|(func, reg, ty)| {
            (*func, *reg, *ty) == (boundary.0, boundary_param, XType::Xint)
        }));

    for (access, expected_op, expected_type) in [
        (fifth_access, Operation::Omulimm(7), XType::Xint),
        (sixth_access, Operation::Ofloatofint, XType::Xint),
    ] {
        let candidates = rtl_candidates(db, access.0 | SYNTH1);
        let param = candidates
            .iter()
            .find_map(|inst| match inst {
                RTLInst::Iop(op, args, _) if *op == expected_op && args.len() == 1 => Some(args[0]),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!("stack unary op did not consume a parameter: {candidates:#x?}")
            });
        assert!(db
            .rel_iter::<(Address, u64, XType)>("emit_function_param_type_candidate")
            .any(|(func, reg, ty)| (*func, *reg, *ty) == (access.1, param, expected_type)));
    }
}

fn assert_home_store_is_not_outgoing(db: &DecompileDB) {
    let caller = function_span(db, "caller_home_vs_outgoing");
    let callee = function_span(db, "callee_eighth");
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_stack_param_count")
        .any(|(func, count)| (*func, *count) == (callee.0, 4)));
    assert!(db
        .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
        .any(|(func, count)| (*func, *count) == (callee.0, 8)));
    let call = db
        .rel_iter::<(Address, Address)>("call_target_func")
        .find_map(|(call, target)| (in_span(*call, caller) && *target == callee.0).then_some(*call))
        .expect("missing local fixture call");
    let evidence: HashSet<_> = db
        .rel_iter::<(Address, usize)>("call_has_arg_evidence")
        .filter_map(|(site, position)| (*site == call).then_some(*position))
        .collect();
    assert!(
        evidence.contains(&4),
        "real outgoing fifth argument was lost: {evidence:?}"
    );
    assert!(
        !evidence.contains(&7),
        "entry home spill was misclassified as an outgoing eighth argument: {evidence:?}"
    );
}

fn entry_regs(db: &DecompileDB, function: Address, reg: Mreg) -> HashSet<u64> {
    db.rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(address, candidate, id)| {
            (*address == function && *candidate == reg).then_some(*id)
        })
        .collect()
}

fn translated_regs_at(db: &DecompileDB, address: Address, reg: Mreg) -> HashSet<u64> {
    db.rel_iter::<(Address, Mreg, u64)>("reg_xtl")
        .filter_map(|(candidate, candidate_reg, id)| {
            (*candidate == address && *candidate_reg == reg).then_some(*id)
        })
        .collect()
}

fn assert_unsafe_home_cells_remain_storage(db: &DecompileDB) {
    for (name, expected_param) in [
        ("home_reassigned", Mreg::CX),
        ("home_escape_mutated", Mreg::CX),
        ("home_mixed_base_clobber", Mreg::CX),
    ] {
        let span = function_span(db, name);
        let candidates: Vec<_> = db
            .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
            .filter(|(_, func, _, _)| *func == span.0)
            .copied()
            .collect();
        assert_eq!(
            candidates.len(),
            1,
            "{name} must have one positive home-spill candidate: {candidates:#x?}"
        );
        assert_eq!((candidates[0].2, candidates[0].3), (expected_param, 0));
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
                .any(|(_, func, _, _)| *func == span.0),
            "{name} must retain mutable local storage"
        );
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
                .any(|(_, func, _, _)| *func == span.0),
            "{name} reload must not fold to the entry parameter"
        );

        let spill_rtl = rtl_candidates(db, candidates[0].0);
        let entry = entry_regs(db, span.0, expected_param);
        assert!(
            spill_rtl.iter().any(|inst| {
                matches!(inst,
                    RTLInst::Iop(Operation::Omove, args, _)
                        if args.len() == 1 && entry.contains(&args[0])
                ) || matches!(inst,
                    RTLInst::Istore(_, _, _, src) if entry.contains(src)
                )
            }),
            "{name} must retain a real write of the entry parameter: {spill_rtl:#x?}"
        );
        assert!(
            !spill_rtl.contains(&RTLInst::Inop),
            "{name} home initialization must not disappear: {spill_rtl:#x?}"
        );

        for (reload, func, _, _) in db
            .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload_cell")
            .filter(|(_, func, _, _)| *func == span.0)
        {
            assert_eq!(*func, span.0);
            let reload_rtl = rtl_candidates(db, *reload);
            assert!(
                !reload_rtl.iter().any(|inst| {
                    matches!(inst, RTLInst::Iop(Operation::Omove, args, _) if args.len() == 1 && entry.contains(&args[0]))
                }),
                "{name} reload incorrectly reused the entry parameter: {reload_rtl:#x?}"
            );
            assert!(
                !reload_rtl.iter().any(|inst| matches!(
                    inst,
                    RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), args, _)
                        if args.is_empty()
                )),
                "{name} reload retained the competing address interpretation: {reload_rtl:#x?}"
            );
        }
    }

    let disjoint = function_span(db, "home_disjoint_branch");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, _, _)| *func == disjoint.0));
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, _, _)| *func == disjoint.0));
}

fn canonical_home_storage_at_position_with_type(
    db: &DecompileDB,
    name: &str,
    expected_position: usize,
    expected_type: XType,
) -> ((Address, Address), u64) {
    let span = function_span(db, name);
    let storage: Vec<_> = db
        .rel_iter::<(Address, usize, u64)>("win64_home_storage")
        .filter_map(|(func, pos, slot)| (*func == span.0).then_some((*pos, *slot)))
        .collect();
    let vetoes: Vec<_> = db
        .rel_iter::<(Address, usize)>("win64_home_canonical_veto")
        .filter(|(func, _)| *func == span.0)
        .copied()
        .collect();
    let accesses: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
        .filter(|(_, func, _, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let stores: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_store",
        )
        .filter(|(_, func, _, _, _, _, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let loads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_load",
        )
        .filter(|(_, func, _, _, _, _, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let arith: Vec<_> = db
        .rel_iter::<(
            Address,
            Address,
            Mreg,
            i64,
            usize,
            i64,
            Mreg,
            usize,
            Operation,
        )>("win64_home_scalar_arith_read")
        .filter(|(_, func, _, _, _, _, _, _, _)| *func == span.0)
        .map(|row| row.clone())
        .collect();
    let access_nodes: HashSet<_> = accesses.iter().map(|(node, ..)| *node).collect();
    let reaching: Vec<_> = db
        .rel_iter::<(Address, Mreg, u64)>("reaching_use_rtl")
        .filter(|(node, _, _)| access_nodes.contains(node))
        .copied()
        .collect();
    let candidate_rows: Vec<_> = accesses
        .iter()
        .map(|(node, ..)| (*node, rtl_candidates(db, *node)))
        .collect();
    let cmp_reads: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol, Mreg, i64, usize, i64, usize)>(
            "win64_home_scalar_cmp_read",
        )
        .filter(|(_, func, _, _, _, _, _, _)| *func == span.0)
        .cloned()
        .collect();
    let branches: Vec<_> = db
        .rel_iter::<(Address, TestCond, Symbol)>("pjcc")
        .filter(|(node, _, _)| *node >= span.0 && *node < span.1)
        .cloned()
        .collect();
    let next_edges: Vec<_> = db
        .rel_iter::<(Address, Address)>("next")
        .filter(|(source, _)| *source >= span.0 && *source < span.1)
        .copied()
        .collect();
    let resolved_symbols: Vec<_> = db
        .rel_iter::<(Symbol, Address)>("symbol_resolved_addr")
        .filter(|(_, address)| *address >= span.0 && *address < span.1)
        .cloned()
        .collect();
    let cmp_consumers: Vec<_> = db
        .rel_iter::<(Address, Address)>("win64_home_cmp_jcc_consumer")
        .filter(|(source, _)| *source >= span.0 && *source < span.1)
        .copied()
        .collect();
    let owners: Vec<_> = db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .filter(|(node, _)| *node >= span.0 && *node < span.1)
        .copied()
        .collect();
    let cmp_operands: Vec<_> = db
        .rel_iter::<(Address, Symbol, Symbol)>("pcmp")
        .filter(|(node, _, _)| *node >= span.0 && *node < span.1)
        .cloned()
        .collect();
    let operand_symbols: HashSet<_> = cmp_operands
        .iter()
        .flat_map(|(_, left, right)| [*left, *right])
        .collect();
    let register_operands: Vec<_> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .filter(|(symbol, _)| operand_symbols.contains(symbol))
        .cloned()
        .collect();
    let immediate_operands: Vec<_> = db
        .rel_iter::<(Symbol, i64, usize)>("op_immediate")
        .filter(|(symbol, _, _)| operand_symbols.contains(symbol))
        .cloned()
        .collect();
    let unsupported: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .filter(|(func, _, _)| *func == span.0)
        .cloned()
        .collect();
    let unsupported_details: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
        .filter(|(func, _, _)| *func == span.0)
        .cloned()
        .collect();
    let overlaps: Vec<_> = db
        .rel_iter::<(Address, Address, usize)>("win64_home_overlap")
        .filter(|(_, func, pos)| *func == span.0 && *pos == 0)
        .copied()
        .collect();
    let owned_blocks: HashSet<_> = db
        .rel_iter::<(Address, Address)>("block_in_function")
        .filter_map(|(block, func)| (*func == span.0).then_some(*block))
        .collect();
    let block_code: Vec<_> = db
        .rel_iter::<(Address, Address)>("code_in_block")
        .filter(|(_, block)| owned_blocks.contains(block))
        .copied()
        .collect();
    assert_eq!(
        storage.len(),
        1,
        "{name} must have one canonical mutable home cell: {storage:#x?}; \
         vetoes={vetoes:#x?}; accesses={accesses:#x?}; stores={stores:#x?}; \
         loads={loads:#x?}; arith={arith:#x?}; reaching={reaching:#x?}; \
         candidates={candidate_rows:#x?}; cmp_reads={cmp_reads:#x?}; \
         branches={branches:#x?}; next={next_edges:#x?}; \
         resolved_symbols={resolved_symbols:#x?}; \
         cmp_consumers={cmp_consumers:#x?}; owners={owners:#x?}; \
         cmp_operands={cmp_operands:#x?}; register_operands={register_operands:#x?}; \
         immediate_operands={immediate_operands:#x?}; unsupported={unsupported:#x?}; \
         unsupported_details={unsupported_details:#x?}; \
         overlaps={overlaps:#x?}; owned_blocks={owned_blocks:#x?}; \
         block_code={block_code:#x?}"
    );
    assert_eq!(
        storage[0].0, expected_position,
        "{name} used the wrong home ordinal"
    );
    assert!(
        !db.rel_iter::<(Address, Address, i64, u64)>("stack_var")
            .any(|(func, _, _, reg)| *func == span.0 && *reg == storage[0].1),
        "{name} reused a raw-offset stack identity as canonical storage"
    );
    let types: HashSet<_> = db
        .rel_iter::<(u64, XType)>("emit_var_type_candidate")
        .filter_map(|(reg, xtype)| (*reg == storage[0].1).then_some(*xtype))
        .collect();
    assert_eq!(
        types,
        HashSet::from([expected_type]),
        "{name} canonical cell inherited an incompatible peer type"
    );
    (span, storage[0].1)
}

fn canonical_home_storage_with_type(
    db: &DecompileDB,
    name: &str,
    expected_type: XType,
) -> ((Address, Address), u64) {
    canonical_home_storage_at_position_with_type(db, name, 0, expected_type)
}

fn canonical_home_storage(db: &DecompileDB, name: &str) -> ((Address, Address), u64) {
    canonical_home_storage_with_type(db, name, XType::Xany64)
}

fn assert_home_backing_profile(
    db: &DecompileDB,
    name: &str,
    expected_rows: usize,
    expected_profile: &[(i64, usize, MemoryChunk, bool, bool, bool)],
) -> ((Address, Address), RTLReg) {
    let span = function_span(db, name);
    let storage: Vec<_> = db
        .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
        .filter_map(|(func, pos, slot)| (*func == span.0).then_some((*pos, *slot)))
        .collect();
    let backing_required: Vec<_> = db
        .rel_iter::<(Address, usize)>("win64_home_backing_required")
        .filter(|(func, _)| *func == span.0)
        .copied()
        .collect();
    let overlaps: Vec<_> = db
        .rel_iter::<(Address, Address, usize)>("win64_home_overlap")
        .filter(|(_, func, _)| *func == span.0)
        .copied()
        .collect();
    let bounded: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, i64, usize)>(
            "win64_home_backing_candidate",
        )
        .filter(|(_, func, _, _, _, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let overlap_nodes: HashSet<_> = overlaps.iter().map(|(node, _, _)| *node).collect();
    let decoded_reads: Vec<_> = db
        .rel_iter::<(Address, Symbol)>("decoded_memory_read_operand")
        .filter(|(node, _)| overlap_nodes.contains(node))
        .copied()
        .collect();
    let decoded_writes: Vec<_> = db
        .rel_iter::<(Address, Symbol)>("decoded_memory_write_operand")
        .filter(|(node, _)| overlap_nodes.contains(node))
        .copied()
        .collect();
    let modes: Vec<_> = db
        .rel_iter::<(Address, bool, bool, bool)>("win64_home_backing_mode")
        .filter(|(node, _, _, _)| overlap_nodes.contains(node))
        .copied()
        .collect();
    let vetoes: Vec<_> = db
        .rel_iter::<(Address, usize)>("win64_home_canonical_veto")
        .filter(|(func, _)| *func == span.0)
        .copied()
        .collect();
    let stack_aliases: Vec<_> = db
        .rel_iter::<(Address, Address, i64, RTLReg)>("stack_var")
        .filter(|(func, _, _, _)| *func == span.0)
        .copied()
        .collect();
    let rtl_candidates: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| overlap_nodes.contains(&(*node & !SYNTHETIC_NODE_MASK)))
        .cloned()
        .collect();
    let unsupported_reasons: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .filter(|(func, _, _)| *func == span.0)
        .copied()
        .collect();
    let unsupported_details: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
        .filter(|(func, _, _)| *func == span.0)
        .copied()
        .collect();
    let backing_move_stores: Vec<_> = db
        .rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
            "win64_home_backing_move_store",
        )
        .filter(|(node, func, _, _, _, _)| {
            *func == span.0 && overlap_nodes.contains(node)
        })
        .copied()
        .collect();
    let backing_move_loads: Vec<_> = db
        .rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
            "win64_home_backing_move_load",
        )
        .filter(|(node, func, _, _, _, _)| {
            *func == span.0 && overlap_nodes.contains(node)
        })
        .copied()
        .collect();
    let semantic_accesses: Vec<_> = db
        .rel_iter::<(Address,)>("win64_home_backing_semantic_access")
        .filter(|(node,)| overlap_nodes.contains(node))
        .copied()
        .collect();
    let semantic_bridges: Vec<_> = db
        .rel_iter::<(Address, Address, Address)>(
            "win64_home_backing_semantic_bridge",
        )
        .filter(|(access, _, _)| overlap_nodes.contains(access))
        .copied()
        .collect();
    let semantic_consumed: Vec<_> = db
        .rel_iter::<(Address, Address)>(
            "win64_home_backing_semantic_consumed",
        )
        .filter(|(access, _)| overlap_nodes.contains(access))
        .copied()
        .collect();
    let semantic_successors: Vec<_> = db
        .rel_iter::<(Address, Address)>("win64_home_backing_semantic_succ")
        .filter(|(access, _)| overlap_nodes.contains(access))
        .copied()
        .collect();
    let backing_tests: Vec<_> = db
        .rel_iter::<(Address, Address, usize, i64, usize)>(
            "win64_home_backing_test_imm",
        )
        .filter(|(node, func, _, _, _)| {
            *func == span.0 && overlap_nodes.contains(node)
        })
        .copied()
        .collect();
    let decoded_edges: Vec<_> = db
        .rel_iter::<(Address, Address)>("next")
        .filter(|(source, destination)| {
            in_span(*source, span) || in_span(*destination, span)
        })
        .copied()
        .collect();
    let ltl_edges: Vec<_> = db
        .rel_iter::<(Address, Address)>("ltl_succ")
        .filter(|(source, destination)| {
            in_span(*source, span) || in_span(*destination, span)
        })
        .copied()
        .collect();
    let semantic_negated: Vec<_> = db
        .rel_iter::<(Address, Address)>(
            "win64_home_backing_semantic_edge_negated",
        )
        .filter(|(access, _)| overlap_nodes.contains(access))
        .copied()
        .collect();
    let exact_addr_defs: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>(
            "win64_home_exact_addr_def",
        )
        .filter(|(func, node, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let home_cells: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize)>("win64_home_cell")
        .filter(|(node, func, _, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let lea_rows: Vec<_> = db
        .rel_iter::<(Address, Symbol, Symbol)>("plea")
        .filter(|(node, _, _)| in_span(*node, span))
        .copied()
        .collect();
    let direct_stack_rows: Vec<_> = db
        .rel_iter::<(Address, Mreg, i64, usize)>("direct_stack_operand")
        .filter(|(node, _, _, _)| in_span(*node, span))
        .copied()
        .collect();
    let sp_based_rows: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_based_mem_at")
        .filter(|(node, func, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let exact_addr_at: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>(
            "win64_home_exact_addr_at",
        )
        .filter(|(func, node, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let raw_reg_edges: Vec<_> = db
        .rel_iter::<(Address, Mreg, Address)>("raw_reg_def_used")
        .filter(|(def, _, use_node)| in_span(*def, span) || in_span(*use_node, span))
        .copied()
        .collect();
    let dominating_reg_edges: Vec<_> = db
        .rel_iter::<(Address, Address, Address)>("reg_def_dominates_use")
        .filter(|(func, def, use_node)| {
            *func == span.0 && (in_span(*def, span) || in_span(*use_node, span))
        })
        .copied()
        .collect();
    let competing_reg_defs: Vec<_> = db
        .rel_iter::<(Address, Mreg, Address)>("competing_reaching_reg_def")
        .filter(|(def, _, use_node)| in_span(*def, span) || in_span(*use_node, span))
        .copied()
        .collect();
    let exact_addr_store_uses: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>(
            "win64_home_exact_addr_store_use",
        )
        .filter(|(func, node, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let exact_addr_calls: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, usize)>(
            "win64_home_exact_addr_call",
        )
        .filter(|(func, node, _, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let address_taken: Vec<_> = db
        .rel_iter::<(Address, Address, usize)>(
            "win64_home_lea_address_taken",
        )
        .filter(|(_, func, _)| *func == span.0)
        .copied()
        .collect();
    let alias_uses: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg)>("sp_may_alias_at")
        .filter(|(func, node, _)| *func == span.0 && in_span(*node, span))
        .copied()
        .collect();
    let asm_uses: Vec<_> = db
        .rel_iter::<(Address, Mreg)>("asm_reg_use")
        .filter(|(node, _)| in_span(*node, span))
        .copied()
        .collect();
    let supported_accesses: Vec<_> = db
        .rel_iter::<(Address, Address, usize)>(
            "win64_home_supported_access",
        )
        .filter(|(_, func, _)| *func == span.0)
        .copied()
        .collect();
    assert_eq!(
        storage.len(),
        1,
        "{name} must select exactly one backing home cell: {storage:#x?}; \
         required={backing_required:#x?}; overlaps={overlaps:#x?}; \
         bounded={bounded:#x?}; reads={decoded_reads:#x?}; \
         writes={decoded_writes:#x?}; modes={modes:#x?}; \
         vetoes={vetoes:#x?}; aliases={stack_aliases:#x?}; \
         backing_move_stores={backing_move_stores:#x?}; \
         backing_move_loads={backing_move_loads:#x?}; \
         backing_tests={backing_tests:#x?}; \
         semantic_accesses={semantic_accesses:#x?}; \
         semantic_bridges={semantic_bridges:#x?}; \
         semantic_consumed={semantic_consumed:#x?}; \
         semantic_successors={semantic_successors:#x?}; \
         semantic_negated={semantic_negated:#x?}; \
         exact_addr_defs={exact_addr_defs:#x?}; home_cells={home_cells:#x?}; \
         lea_rows={lea_rows:#x?}; direct_stack_rows={direct_stack_rows:#x?}; \
         sp_based_rows={sp_based_rows:#x?}; exact_addr_at={exact_addr_at:#x?}; \
         raw_reg_edges={raw_reg_edges:#x?}; \
         dominating_reg_edges={dominating_reg_edges:#x?}; \
         competing_reg_defs={competing_reg_defs:#x?}; \
         exact_addr_store_uses={exact_addr_store_uses:#x?}; \
         exact_addr_calls={exact_addr_calls:#x?}; \
         address_taken={address_taken:#x?}; alias_uses={alias_uses:#x?}; \
         asm_uses={asm_uses:#x?}; supported_accesses={supported_accesses:#x?}; \
         decoded_edges={decoded_edges:#x?}; ltl_edges={ltl_edges:#x?}; \
         unsupported_reasons={unsupported_reasons:#x?}; \
         unsupported_details={unsupported_details:#x?}; \
         rtl_candidates={rtl_candidates:#x?}"
    );
    let (position, slot) = storage[0];
    assert_eq!(position, 0, "{name} selected the wrong home position");
    assert!(
        db.rel_iter::<(Address, usize)>("win64_home_backing_required")
            .any(|row| *row == (span.0, position)),
        "{name} did not atomically promote its home cell"
    );
    assert!(
        db.rel_iter::<(RTLReg, XType)>("win64_home_slot_type")
            .any(|row| *row == (slot, XType::Xany64)),
        "{name} backing cell lost its exact eight-byte type"
    );
    assert!(
        db.rel_iter::<(Address, RTLReg)>("win64_home_escaped")
            .any(|row| *row == (span.0, slot)),
        "{name} backing cell is not protected from scalar propagation"
    );
    assert!(
        !db.rel_iter::<(Address, usize)>("win64_home_canonical_veto")
            .any(|(func, pos)| (*func, *pos) == (span.0, position)),
        "{name} retained a canonicalization veto"
    );

    let rows: Vec<_> = db
        .rel_iter::<(
            Address,
            RTLReg,
            i64,
            usize,
            MemoryChunk,
            bool,
            bool,
            bool,
        )>("win64_home_backing_access")
        .filter_map(|(
            node,
            candidate_slot,
            byte_ofs,
            width,
            chunk,
            read,
            write,
            address,
        )| {
            (in_span(*node, span) && *candidate_slot == slot).then_some((
                *node,
                *byte_ofs,
                *width,
                *chunk,
                *read,
                *write,
                *address,
            ))
        })
        .collect();
    assert_eq!(
        rows.len(),
        expected_rows,
        "{name} lost or duplicated a backing access: {rows:#x?}"
    );
    let observed: HashSet<_> = rows
        .iter()
        .map(|(_, byte_ofs, width, chunk, read, write, address)| {
            (*byte_ofs, *width, *chunk, *read, *write, *address)
        })
        .collect();
    let expected: HashSet<_> = expected_profile.iter().copied().collect();
    assert_eq!(
        observed, expected,
        "{name} changed its byte-range/mode profile"
    );
    for (node, _, _, _, read, write, address) in &rows {
        let modes: HashSet<_> = db
            .rel_iter::<(Address, bool, bool, bool)>("win64_home_backing_mode")
            .filter_map(|(candidate, candidate_read, candidate_write, candidate_address)| {
                (*candidate == *node).then_some((
                    *candidate_read,
                    *candidate_write,
                    *candidate_address,
                ))
            })
            .collect();
        assert_eq!(
            modes,
            HashSet::from([(*read, *write, *address)]),
            "{name} did not retain one exact decoder-derived mode at {node:#x}"
        );
    }
    for access in rows.iter().map(|(node, ..)| *node).collect::<HashSet<_>>() {
        let mut selected: Vec<_> = db
            .rel_iter::<(Address, Address, RTLInst)>(
                "win64_home_backing_selected_candidate",
            )
            .filter_map(|(real, candidate, inst)| {
                (*real == access).then_some((*candidate, inst.clone()))
            })
            .collect();
        selected.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
        selected.dedup();
        let mut actual: Vec<_> = db
            .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
            .filter_map(|(candidate, inst)| {
                ((*candidate & !SYNTHETIC_NODE_MASK) == access)
                    .then_some((*candidate, inst.clone()))
            })
            .collect();
        actual.sort_by_cached_key(|(node, inst)| (*node, format!("{inst:?}")));
        actual.dedup();
        assert_eq!(
            actual, selected,
            "{name} retained an RTL candidate outside its selected equivalence class at {access:#x}"
        );
    }

    let selected_nodes: HashSet<_> = rows.iter().map(|(node, ..)| *node).collect();
    let overlap_nodes: HashSet<_> = db
        .rel_iter::<(Address, Address, usize)>("win64_home_overlap")
        .filter_map(|(node, func, pos)| {
            ((*func, *pos) == (span.0, position)).then_some(*node)
        })
        .collect();
    assert_eq!(
        selected_nodes, overlap_nodes,
        "{name} did not select every authoritative overlap atomically"
    );
    assert!(
        !db.rel_iter::<(Address, Address, i64, RTLReg)>("stack_var")
            .any(|(func, node, _, _)| *func == span.0 && selected_nodes.contains(node)),
        "{name} retained a competing raw stack identity"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, node, _)| *func == span.0 && selected_nodes.contains(node)),
        "{name} retained a rejection at a selected backing access"
    );
    (span, slot)
}

fn assert_canonical_home_backing(db: &DecompileDB) {
    for (name, rows, profile) in [
        (
            "home_partial_movzx_high_backing",
            3,
            &[
                (0, 4, MemoryChunk::MInt32, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, false, false),
            ][..],
        ),
        (
            "home_backing_movsx_high",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Signed, true, false, false),
            ][..],
        ),
        (
            "home_backing_signed_cmp_jcc",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Signed, true, false, false),
            ][..],
        ),
        (
            "home_backing_unsigned_cmp_setcc",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, false, false),
            ][..],
        ),
        (
            "home_backing_partial_load",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, false, false),
            ][..],
        ),
        (
            "home_backing_partial_store",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, false, true, false),
                (0, 8, MemoryChunk::MInt64, true, false, false),
            ][..],
        ),
        (
            "home_backing_add_high",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, true, false),
                (0, 8, MemoryChunk::MInt64, true, false, false),
            ][..],
        ),
        (
            "home_backing_test_jcc",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (1, 1, MemoryChunk::MInt8Unsigned, true, false, false),
            ][..],
        ),
        (
            "home_backing_test_setcc",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, false, false),
            ][..],
        ),
        (
            "home_backing_cmov_cmp",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (0, 8, MemoryChunk::MInt64, true, false, false),
            ][..],
        ),
        (
            "home_backing_cmov_test",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (0, 8, MemoryChunk::MInt64, true, false, false),
            ][..],
        ),
        (
            "home_backing_div",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (0, 4, MemoryChunk::MInt32, true, false, false),
            ][..],
        ),
        (
            "home_backing_descriptor",
            3,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (0, 8, MemoryChunk::MInt64, false, false, true),
            ][..],
        ),
        (
            "home_backing_cmp_setcc",
            4,
            &[
                (0, 8, MemoryChunk::MInt64, false, true, false),
                (2, 2, MemoryChunk::MInt16Unsigned, true, false, false),
                (0, 8, MemoryChunk::MInt64, true, false, false),
            ][..],
        ),
    ] {
        let (span, slot) = assert_home_backing_profile(db, name, rows, profile);
        let semantic_nodes: HashSet<_> = db
            .rel_iter::<(Address,)>("win64_home_backing_semantic_access")
            .filter_map(|(node,)| in_span(*node, span).then_some(*node))
            .collect();
        if name.contains("_test_") || name.contains("_cmov_") {
            assert_eq!(
                semantic_nodes.len(),
                1,
                "{name} lost its unique normalized TEST/CMOV witness"
            );
            let node = *semantic_nodes.iter().next().unwrap();
            let candidates = rtl_candidates(db, node);
            assert_eq!(
                candidates.len(),
                1,
                "{name} retained a competing semantic RTL candidate: {candidates:#x?}"
            );
            assert!(
                candidates.iter().all(|inst| match inst {
                    RTLInst::Icond(_, args, _, _)
                    | RTLInst::Iop(Operation::Ocmp(_) | Operation::Osel(_, _), args, _) => {
                        args.iter().filter(|arg| **arg == slot).count() == 1
                    }
                    _ => false,
                }),
                "{name} lacks one semantic RTL candidate over its backing slot: {candidates:#x?}"
            );
            let consumed: Vec<_> = db
                .rel_iter::<(Address, Address)>("win64_home_backing_semantic_consumed")
                .filter_map(|(access, consumer)| (*access == node).then_some(*consumer))
                .collect();
            assert_eq!(
                consumed.len(),
                1,
                "{name} must consume one exact flag producer/consumer"
            );
            assert!(
                rtl_candidates(db, consumed[0]).is_empty(),
                "{name} retained its separately executable consumed flag node"
            );
            let successors: HashSet<_> = db
                .rel_iter::<(Address, Address)>("rtl_succ_candidate")
                .filter_map(|(source, destination)| {
                    (*source == node).then_some(*destination)
                })
                .collect();
            assert!(
                !successors.is_empty(),
                "{name} semantic node was not spliced into the RTL CFG"
            );
            assert!(
                db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
                    .any(|(_, destination)| *destination == node),
                "{name} has no incoming RTL edge after skip-over repair"
            );
        }
        if name == "home_backing_div" {
            assert!(
                db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
                    .any(|(node, inst)| {
                        in_span(*node & !SYNTHETIC_NODE_MASK, span)
                            && matches!(inst, RTLInst::Iop(Operation::Odivu, args, _)
                                if args.contains(&slot))
                    }),
                "home_backing_div did not reuse the ordinary DIV lowering"
            );
        }
        if name == "home_backing_descriptor" {
            assert!(
                db.rel_iter::<(Address, RTLReg)>("win64_home_address")
                    .any(|(node, candidate_slot)| {
                        in_span(*node, span) && *candidate_slot == slot
                    }),
                "descriptor LEA was not authenticated as &backing"
            );
        }
    }

    // Partial backing comparisons use their decoded subrange width and retain
    // the signed/unsigned flag consumer exactly.  They are deliberately not
    // reclassified as full-width scalar accesses.
    for (name, expected_condition, expect_setcc) in [
        (
            "home_backing_signed_cmp_jcc",
            Condition::Ccompimm(Comparison::Clt, -1),
            false,
        ),
        (
            "home_backing_unsigned_cmp_setcc",
            Condition::Ccompuimm(Comparison::Clt, 7),
            true,
        ),
    ] {
        let span = function_span(db, name);
        let slot = db
            .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
            .find_map(|(func, pos, slot)| {
                (*func == span.0 && *pos == 0).then_some(*slot)
            })
            .unwrap_or_else(|| panic!("{name} lost its backing slot"));
        let compares: Vec<_> = db
            .rel_iter::<(Address, Address, Symbol, Mreg, i64, usize, i64, usize)>(
                "win64_home_backing_cmp_read",
            )
            .filter(|(_, func, _, _, _, _, _, _)| *func == span.0)
            .copied()
            .collect();
        assert_eq!(
            compares.len(),
            1,
            "{name} must retain one exact bounded CMP projection: {compares:#x?}"
        );
        let (compare, _, _, _, raw_disp, position, _, width) = compares[0];
        assert_eq!(
            (raw_disp, position, width),
            (10, 0, 2),
            "{name} changed its partial CMP coordinates"
        );
        assert!(
            !db.rel_iter::<(Address, Address, Symbol, Mreg, i64, usize, i64, usize)>(
                "win64_home_scalar_cmp_read",
            )
            .any(|(node, _, _, _, _, _, _, _)| *node == compare),
            "{name} incorrectly widened its backing CMP into the scalar closed set"
        );
        let candidates = rtl_candidates(db, compare);
        assert_eq!(
            candidates.len(),
            1,
            "{name} retained a competing CMP candidate: {candidates:#x?}"
        );
        let condition_matches = match &candidates[0] {
            RTLInst::Icond(condition, args, _, _) if !expect_setcc => {
                *condition == expected_condition && args.as_ref() == &[slot]
            }
            RTLInst::Iop(Operation::Ocmp(condition), args, _) if expect_setcc => {
                *condition == expected_condition && args.as_ref() == &[slot]
            }
            _ => false,
        };
        assert!(
            condition_matches,
            "{name} lost CMP signedness, width, operand order, or canonical slot: {candidates:#x?}"
        );
        if expect_setcc {
            let setcc = db
                .rel_iter::<(Address, Address)>("next")
                .find_map(|(source, destination)| {
                    (*source == compare
                        && db
                            .rel_iter::<(Address, TestCond)>("setcc_testcond")
                            .any(|(node, _)| node == destination))
                    .then_some(*destination)
                })
                .expect("partial backing CMP lost its adjacent SETcc");
            assert!(
                rtl_candidates(db, setcc).is_empty(),
                "{name} retained a separately executable SETcc"
            );
        }
    }

    let partial_store_span = function_span(db, "home_backing_partial_store");
    let partial_store_slot = db
        .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
        .find_map(|(func, pos, slot)| {
            (*func == partial_store_span.0 && *pos == 0).then_some(*slot)
        })
        .expect("partial backing store lost its canonical slot");
    let partial_stores: Vec<_> = db
        .rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
            "win64_home_backing_move_store",
        )
        .filter(|(_, func, raw_disp, width, _, _)| {
            *func == partial_store_span.0 && *raw_disp == 10 && *width == 2
        })
        .copied()
        .collect();
    assert_eq!(
        partial_stores.len(),
        1,
        "partial MOV store lacks one exact reaching-source projection: {partial_stores:#x?}"
    );
    let partial_store_candidates = rtl_candidates(db, partial_stores[0].0);
    assert!(
        matches!(partial_store_candidates.as_slice(),
            [RTLInst::Iop(Operation::Omove, args, destination)]
                if *destination == partial_store_slot
                    && args.len() == 1
                    && args[0] != partial_store_slot),
        "partial MOV store did not select one canonical-slot write: {partial_store_candidates:#x?}"
    );

    let partial_load_span = function_span(db, "home_backing_partial_load");
    let partial_load_slot = db
        .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
        .find_map(|(func, pos, slot)| {
            (*func == partial_load_span.0 && *pos == 0).then_some(*slot)
        })
        .expect("partial backing load lost its canonical slot");
    let partial_loads: Vec<_> = db
        .rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
            "win64_home_backing_move_load",
        )
        .filter(|(_, func, raw_disp, width, _, _)| {
            *func == partial_load_span.0 && *raw_disp == 10 && *width == 2
        })
        .copied()
        .collect();
    assert_eq!(
        partial_loads.len(),
        1,
        "partial MOV load lacks one exact destination-def projection: {partial_loads:#x?}"
    );
    let partial_load_candidates = rtl_candidates(db, partial_loads[0].0);
    assert!(
        matches!(partial_load_candidates.as_slice(),
            [RTLInst::Iop(Operation::Omove, args, destination)]
                if args.as_ref() == &[partial_load_slot]
                    && *destination != partial_load_slot),
        "partial MOV load did not select one canonical-slot read: {partial_load_candidates:#x?}"
    );

    for (name, expected_chunk) in [
        (
            "home_partial_movzx_high_backing",
            MemoryChunk::MInt16Unsigned,
        ),
        ("home_backing_movsx_high", MemoryChunk::MInt16Signed),
    ] {
        let span = function_span(db, name);
        let read_nodes: HashSet<_> = db
            .rel_iter::<(
                Address,
                RTLReg,
                i64,
                usize,
                MemoryChunk,
                bool,
                bool,
                bool,
            )>("win64_home_backing_access")
            .filter_map(|(node, _, _, width, chunk, read, write, address)| {
                (in_span(*node, span)
                    && *width == 2
                    && *chunk == expected_chunk
                    && *read
                    && !*write
                    && !*address)
                    .then_some(*node)
            })
            .collect();
        assert_eq!(
            read_nodes.len(),
            1,
            "{name} lost its unique authenticated extending-load access"
        );
        let slot = db
            .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
            .find_map(|(func, pos, slot)| {
                (*func == span.0 && *pos == 0).then_some(*slot)
            })
            .expect("extending-load fixture lost canonical backing storage");
        let selected: Vec<_> = db
            .rel_iter::<(Address, Address, RTLInst)>(
                "win64_home_backing_selected_candidate",
            )
            .filter_map(|(real, _, inst)| read_nodes.contains(real).then_some(inst.clone()))
            .collect();
        assert_eq!(
            selected.len(),
            1,
            "{name} did not select one exact extending-load candidate: {selected:#x?}"
        );
        assert!(
            selected.iter().all(|inst| !matches!(
                inst,
                RTLInst::Iop(
                    Operation::Olea(Addressing::Ainstack(_))
                        | Operation::Oleal(Addressing::Ainstack(_)),
                    _,
                    _
                )
            )),
            "{name} retained its zero-argument stack-address shadow"
        );
        assert!(
            selected.iter().all(|inst| match inst {
                RTLInst::Iload(chunk, _, _, _) => *chunk == expected_chunk,
                other => test_rtl_inst_uses(other, slot),
            }),
            "{name} selected a candidate without its authenticated backing read: {selected:#x?}"
        );
    }
}

fn assert_home_accesses_use_slot_at_position(
    db: &DecompileDB,
    name: &str,
    span: (Address, Address),
    slot: u64,
    expected_position: usize,
) {
    let accesses: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
        .filter(|(_, func, _, _, _, _)| *func == span.0)
        .copied()
        .collect();
    assert!(
        !accesses.is_empty(),
        "{name} lost unsafe-home access evidence"
    );
    for (address, _, _, _, pos, entry_offset) in accesses {
        assert_eq!(
            pos, expected_position,
            "{name} access used the wrong home ordinal"
        );
        assert_eq!(
            entry_offset,
            8 + expected_position as i64 * 8,
            "{name} access was not normalized to entry-SP coordinates"
        );
        let rtl = rtl_candidates(db, address);
        assert_eq!(
            rtl.len(),
            1,
            "{name} access kept competing RTL candidates: {rtl:#x?}"
        );
        let uses_slot = match &rtl[0] {
            RTLInst::Iop(Operation::Omove, args, destination) => {
                *destination == slot || args.as_ref() == &[slot]
            }
            RTLInst::Iop(
                Operation::Ocast8signed
                | Operation::Ocast8unsigned
                | Operation::Ocast16signed
                | Operation::Ocast16unsigned
                | Operation::Ocast32signed,
                args,
                _,
            ) => args.as_ref() == &[slot],
            RTLInst::Iop(Operation::Ocmp(_), args, _) => args.contains(&slot),
            RTLInst::Iop(Operation::Oleal(Addressing::Ainstack(8)), args, _) => {
                args.is_empty()
                    && db
                        .rel_iter::<(Address, u64)>("win64_home_address")
                        .any(|(node, candidate)| (*node, *candidate) == (address, slot))
            }
            RTLInst::Inop => rtl_candidates(db, address | SYNTH1).iter().any(|inst| {
                matches!(inst, RTLInst::Iop(_, args, _) if args.contains(&slot))
            }),
            RTLInst::Icond(_, args, _, _) => args.contains(&slot),
            _ => false,
        };
        assert!(
            uses_slot,
            "{name} access did not use its canonical home cell: {rtl:#x?}"
        );
        assert!(
            !db.rel_iter::<(Address, Address, i64, u64)>("stack_var")
                .any(|(func, node, _, _)| (*func, *node) == (span.0, address)),
            "{name} retained stale raw stack_var evidence at rewritten node {address:#x}"
        );
    }
}

fn assert_home_accesses_use_slot(
    db: &DecompileDB,
    name: &str,
    span: (Address, Address),
    slot: u64,
) {
    assert_home_accesses_use_slot_at_position(db, name, span, slot, 0);
}

fn assert_canonical_unsafe_home_storage(db: &DecompileDB) {
    let (reassigned, reassigned_slot) = canonical_home_storage(db, "home_reassigned");
    assert_home_accesses_use_slot(db, "home_reassigned", reassigned, reassigned_slot);

    let (reassigned_cmp, reassigned_cmp_slot) =
        canonical_home_storage(db, "home_reassigned_cmp");
    assert_home_accesses_use_slot(
        db,
        "home_reassigned_cmp",
        reassigned_cmp,
        reassigned_cmp_slot,
    );
    let cmp = db
        .rel_iter::<(Address, Address, Symbol, Mreg, i64, usize, i64, usize)>(
            "win64_home_scalar_cmp_read",
        )
        .find(|(_, func, _, _, _, _, _, _)| *func == reassigned_cmp.0)
        .expect("mutable home CMP did not retain its structured comparison shape");
    assert!(
        rtl_candidates(db, cmp.0).iter().any(|inst| {
            matches!(inst, RTLInst::Icond(_, args, _, _) if args.contains(&reassigned_cmp_slot))
        }),
        "mutable home CMP did not consume the canonical cell"
    );

    let (reassigned_cmp_reg, reassigned_cmp_reg_slot) =
        canonical_home_storage(db, "home_reassigned_cmp_reg");
    assert_home_accesses_use_slot(
        db,
        "home_reassigned_cmp_reg",
        reassigned_cmp_reg,
        reassigned_cmp_reg_slot,
    );

    for name in [
        "home_reassigned_cmp_setcc",
        "home_reassigned_cmp_reg_setcc",
    ] {
        let (span, slot) = canonical_home_storage(db, name);
        assert_home_accesses_use_slot(db, name, span, slot);
        let compare = db
            .rel_iter::<(Address, Address, Symbol, Mreg, i64, usize, i64, usize)>(
                "win64_home_scalar_cmp_read",
            )
            .find(|(_, func, _, _, _, _, _, _)| *func == span.0)
            .unwrap_or_else(|| panic!("{name} lost its structured comparison shape"));
        assert!(
            rtl_candidates(db, compare.0).iter().any(|inst| match inst {
                RTLInst::Iop(Operation::Ocmp(_), args, _) => args.contains(&slot),
                _ => false,
            }),
            "{name} SETcc comparison did not consume canonical storage"
        );
        let setcc = db
            .rel_iter::<(Address, Address)>("next")
            .find_map(|(source, destination)| {
                (*source == compare.0
                    && db
                        .rel_iter::<(Address, TestCond)>("setcc_testcond")
                        .any(|(node, _)| node == destination))
                .then_some(*destination)
            })
            .unwrap_or_else(|| panic!("{name} lost its adjacent SETcc consumer"));
        assert!(
            rtl_candidates(db, setcc).is_empty(),
            "{name} retained a stale SETcc candidate after fusing its home comparison"
        );
        let fallthrough = db
            .rel_iter::<(Address, Address)>("next")
            .find_map(|(source, destination)| (*source == setcc).then_some(*destination))
            .unwrap_or_else(|| panic!("{name} SETcc lost its decoded fallthrough"));
        let successors: HashSet<_> = db
            .rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .filter_map(|(source, destination)| (*source == compare.0).then_some(*destination))
            .collect();
        assert!(
            successors.contains(&fallthrough) && !successors.contains(&setcc),
            "{name} did not bypass its consumed adjacent SETcc: {successors:#x?}"
        );
    }

    let (cmp_loop_span, cmp_loop_slot) =
        canonical_home_storage(db, "home_reassigned_cmp_loop");
    assert_home_accesses_use_slot(
        db,
        "home_reassigned_cmp_loop",
        cmp_loop_span,
        cmp_loop_slot,
    );
    let (alias_cmp_span, alias_cmp_slot) = canonical_home_storage_at_position_with_type(
        db,
        "home_cmp_after_alias_clobber",
        3,
        XType::Xany64,
    );
    assert_home_accesses_use_slot_at_position(
        db,
        "home_cmp_after_alias_clobber",
        alias_cmp_span,
        alias_cmp_slot,
        3,
    );

    for (name, expected_type, expected_op) in [
        (
            "home_reassigned_movsxd",
            XType::Xint,
            Operation::Ocast32signed,
        ),
        (
            "home_reassigned_movsx",
            XType::Xint16unsigned,
            Operation::Ocast16signed,
        ),
        (
            "home_reassigned_movzx",
            XType::Xint16unsigned,
            Operation::Ocast16unsigned,
        ),
        (
            "home_partial_movzx_low",
            XType::Xint,
            Operation::Ocast16unsigned,
        ),
        (
            "home_partial_reload",
            XType::Xany64,
            Operation::Ocast8unsigned,
        ),
    ] {
        let (span, slot) = canonical_home_storage_with_type(db, name, expected_type);
        assert_home_accesses_use_slot(db, name, span, slot);
        let extend = db
            .rel_iter::<(
                Address,
                Address,
                Mreg,
                i64,
                usize,
                i64,
                Mreg,
                usize,
                Operation,
            )>("win64_home_scalar_extend_load")
            .find(|(_, func, _, _, _, _, _, _, op)| {
                *func == span.0 && op == &expected_op
            })
            .unwrap_or_else(|| panic!("{name} lost its extending-load shape"));
        assert!(
            rtl_candidates(db, extend.0).iter().any(|inst| {
                matches!(inst, RTLInst::Iop(op, args, _)
                    if op == &expected_op && args.as_ref() == &[slot])
            }),
            "{name} did not cast from canonical storage"
        );
    }

    let (reassigned_add, reassigned_add_slot) =
        canonical_home_storage(db, "home_reassigned_add");
    assert_home_accesses_use_slot(
        db,
        "home_reassigned_add",
        reassigned_add,
        reassigned_add_slot,
    );
    let add = db
        .rel_iter::<(
            Address,
            Address,
            Mreg,
            i64,
            usize,
            i64,
            Mreg,
            usize,
            Operation,
        )>("win64_home_scalar_arith_read")
        .find(|(_, func, _, _, _, _, _, _, _)| *func == reassigned_add.0)
        .expect("mutable home ADD did not retain its structured arithmetic shape");
    assert!(
        rtl_candidates(db, add.0 | SYNTH1).iter().any(|inst| {
            matches!(inst, RTLInst::Iop(op, args, _) if op == &add.8 && args.contains(&reassigned_add_slot))
        }),
        "mutable home ADD did not consume the canonical cell"
    );

    let (literal_escape, literal_escape_slot) = canonical_home_storage(db, "home_escape_mutated");
    assert_home_accesses_use_slot(
        db,
        "home_escape_mutated",
        literal_escape,
        literal_escape_slot,
    );

    let (mixed, mixed_slot) = canonical_home_storage(db, "home_mixed_base_clobber");
    let mixed_stores: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_store",
        )
        .filter(|(_, func, base, _, _, _, _, _, _)| {
            *func == mixed.0 && *base != Mreg::SP && *base != Mreg::BP
        })
        .cloned()
        .collect();
    let mixed_loads: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_load",
        )
        .filter(|(_, func, base, _, _, _, _, _, _)| {
            *func == mixed.0 && *base != Mreg::SP && *base != Mreg::BP
        })
        .cloned()
        .collect();
    assert_eq!(mixed_stores.len(), 1, "mixed-base copied store shape lost");
    assert_eq!(mixed_loads.len(), 1, "mixed-base copied load shape lost");
    let (store, _, base, raw_offset, pos, entry_offset, _, move_class, width) =
        mixed_stores[0];
    assert_eq!(
        (base, raw_offset, pos, entry_offset, move_class, width),
        (Mreg::R10, 8, 0, 8, 0, 8)
    );
    let store_rtl = rtl_candidates(db, store);
    assert!(
        store_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, destination)
            if args.len() == 1 && *destination == mixed_slot)
        }),
        "copied home store did not write the canonical cell: {store_rtl:#x?}"
    );
    assert!(
        !store_rtl
            .iter()
            .any(|inst| matches!(inst, RTLInst::Istore(..))),
        "copied home store retained generic pointer memory: {store_rtl:#x?}"
    );

    let (load, _, base, raw_offset, pos, entry_offset, destination, move_class, width) =
        mixed_loads[0];
    assert_eq!(
        (base, raw_offset, pos, entry_offset, move_class, width),
        (Mreg::R10, 8, 0, 8, 0, 8)
    );
    let load_rtl = rtl_candidates(db, load);
    let destinations = translated_regs_at(db, load, destination);
    assert!(
        load_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, candidate_destination)
            if args.as_ref() == &[mixed_slot] && destinations.contains(candidate_destination))
        }),
        "copied home load did not read the canonical cell: {load_rtl:#x?}"
    );
    assert!(
        !load_rtl
            .iter()
            .any(|inst| matches!(inst, RTLInst::Iload(..))),
        "copied home load retained generic pointer memory: {load_rtl:#x?}"
    );

    let literal_clobber = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
        .find_map(|(address, func, base, raw, pos, entry)| {
            (*func == mixed.0 && *base == Mreg::SP).then_some((*address, *raw, *pos, *entry))
        })
        .expect("mixed-base fixture lost its literal-SP clobber");
    assert_eq!(
        (literal_clobber.1, literal_clobber.2, literal_clobber.3),
        (8, 0, 8)
    );
    let literal_rtl = rtl_candidates(db, literal_clobber.0);
    assert!(
        literal_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, destination)
            if args.len() == 1 && *destination == mixed_slot)
        }),
        "literal and copied stores did not share storage: {literal_rtl:#x?}"
    );
    assert_home_accesses_use_slot(db, "home_mixed_base_clobber", mixed, mixed_slot);

    let (escaped, escaped_slot) = canonical_home_storage(db, "home_alias_call_escape");
    let escaped_store = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_store",
        )
        .find(|(_, func, base, _, _, _, _, _, _)| {
            *func == escaped.0 && *base != Mreg::SP && *base != Mreg::BP
        })
        .cloned()
        .expect("alias-call fixture lost its copied store");
    let escaped_load = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_load",
        )
        .find(|(_, func, base, _, _, _, _, _, _)| {
            *func == escaped.0 && *base != Mreg::SP && *base != Mreg::BP
        })
        .cloned()
        .expect("alias-call fixture lost its copied load");
    let escaped_lea = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg)>("win64_home_scalar_lea")
        .find(|(_, func, base, _, _, _, _)| {
            *func == escaped.0 && *base != Mreg::SP && *base != Mreg::BP
        })
        .copied()
        .expect("alias-call fixture lost its copied LEA");
    let store_rtl = rtl_candidates(db, escaped_store.0);
    assert!(
        store_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, _, destination)
            if *destination == escaped_slot)
        }),
        "escaped copied store did not write canonical storage: {store_rtl:#x?}"
    );
    let load_rtl = rtl_candidates(db, escaped_load.0);
    let load_destinations = translated_regs_at(db, escaped_load.0, escaped_load.6);
    assert!(
        load_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, destination)
            if args.as_ref() == &[escaped_slot] && load_destinations.contains(destination))
        }),
        "escaped copied load did not read canonical storage: {load_rtl:#x?}"
    );
    let lea_rtl = rtl_candidates(db, escaped_lea.0);
    let lea_destinations = translated_regs_at(db, escaped_lea.0, escaped_lea.6);
    assert!(
        lea_rtl.iter().any(|inst| {
            matches!(inst,
            RTLInst::Iop(Operation::Oleal(Addressing::Ainstack(8)), args, destination)
                if args.is_empty() && lea_destinations.contains(destination))
        }),
        "escaped copied LEA did not use canonical entry-SP coordinates: {lea_rtl:#x?}"
    );
    assert!(
        !lea_rtl.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(Operation::Olea(Addressing::Aindexed(_)), _, _)
            )
        }),
        "escaped copied LEA retained the copied-base coordinate: {lea_rtl:#x?}"
    );
    assert!(
        db.rel_iter::<(Address, u64)>("win64_home_escaped")
            .any(|(func, slot)| (*func, *slot) == (escaped.0, escaped_slot)),
        "address-taken home cell was not protected from dead-store elimination"
    );
    assert_home_accesses_use_slot(db, "home_alias_call_escape", escaped, escaped_slot);

    let (affine, affine_slot) = canonical_home_storage(db, "home_alias_arith_clobber");
    let affine_load = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg, usize, usize)>(
            "win64_home_scalar_load",
        )
        .find(|(_, func, base, raw, _, entry, _, _, _)| {
            *func == affine.0 && *base == Mreg::R10 && *raw == 0 && *entry == 8
        })
        .cloned()
        .expect("affine alias load did not retain entry-SP coordinate 8");
    assert!(
        db.rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
            .any(|(func, address, reg, offset)| {
                (*func, *address, *reg, *offset) == (affine.0, affine_load.0, Mreg::R10, 8)
            }),
        "in-place ADD alias did not propagate to the final load"
    );
    let affine_rtl = rtl_candidates(db, affine_load.0);
    let affine_destinations = translated_regs_at(db, affine_load.0, affine_load.6);
    assert!(
        affine_rtl.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, destination)
            if args.as_ref() == &[affine_slot] && affine_destinations.contains(destination))
        }),
        "affine copied load did not read canonical storage: {affine_rtl:#x?}"
    );
    assert!(
        !affine_rtl
            .iter()
            .any(|inst| matches!(inst, RTLInst::Iload(..))),
        "affine copied load retained generic pointer memory: {affine_rtl:#x?}"
    );
    assert_home_accesses_use_slot(db, "home_alias_arith_clobber", affine, affine_slot);

    let (nonvolatile, nonvolatile_slot) =
        canonical_home_storage(db, "home_nonvolatile_alias_after_call");
    assert_home_accesses_use_slot(
        db,
        "home_nonvolatile_alias_after_call",
        nonvolatile,
        nonvolatile_slot,
    );
}

fn assert_postsub_raw_offset_does_not_alias_home(db: &DecompileDB) {
    let (span, slot) = canonical_home_storage(db, "home_postsub_raw_collision");
    let home_nodes: HashSet<_> = db
        .rel_iter::<(Address, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
        .filter_map(|(node, func, _, _, _, _)| (*func == span.0).then_some(*node))
        .collect();
    assert_home_accesses_use_slot(db, "home_postsub_raw_collision", span, slot);

    let postsub_nodes: Vec<_> = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .filter_map(|(node, inst)| {
            (in_span(*node, span)
                && !home_nodes.contains(node)
                && matches!(
                    inst,
                    LTLInst::Lsetstack(_, _, 8, _) | LTLInst::Lgetstack(_, 8, _, _)
                ))
            .then_some(*node)
        })
        .collect();
    assert!(
        postsub_nodes.len() >= 2,
        "missing post-sub raw-offset accesses"
    );
    for node in postsub_nodes {
        let rtl = rtl_candidates(db, node);
        assert!(
            !rtl.iter().any(|inst| match inst {
                RTLInst::Iop(Operation::Omove, args, destination) => {
                    *destination == slot || args.as_ref() == &[slot]
                }
                _ => false,
            }),
            "post-sub [rsp+8] collided with entry home storage: {rtl:#x?}"
        );
        assert!(
            db.rel_iter::<(Address, Address, i64, u64)>("stack_var")
                .any(|(func, candidate, ofs, reg)| {
                    (*func, *candidate, *ofs) == (span.0, node, 8) && *reg != slot
                }),
            "post-sub raw local lost its independent stack identity"
        );
    }
}

fn assert_postsub_escaped_offset_keeps_ordinary_origin(db: &DecompileDB) {
    let (span, home_slot) = canonical_home_storage(db, "home_postsub_escaped_collision");
    assert_home_accesses_use_slot(db, "home_postsub_escaped_collision", span, home_slot);

    let home_address_nodes: HashSet<_> = db
        .rel_iter::<(Address, u64)>("win64_home_address")
        .filter_map(|(node, slot)| (*slot == home_slot).then_some(*node))
        .collect();
    assert_eq!(
        home_address_nodes.len(),
        1,
        "missing canonical home LEA origin"
    );

    let ordinary_origins: Vec<_> = db
        .rel_iter::<(Address, Address, i64, u64)>("slot_escaped_origin")
        .filter_map(|(func, origin, ofs, reg)| {
            (*func == span.0 && *ofs == 8 && !home_address_nodes.contains(origin))
                .then_some((*origin, *reg))
        })
        .collect();
    assert_eq!(
        ordinary_origins.len(),
        1,
        "same-raw-offset post-prologue escaped LEA origin was lost: {ordinary_origins:#x?}"
    );
    let (ordinary_origin, ordinary_reg) = ordinary_origins[0];
    assert_ne!(
        ordinary_reg, home_slot,
        "ordinary local reused canonical home storage"
    );
    assert!(
        db.rel_iter::<(Address, Address, i64, u64)>("stack_var")
            .any(|(func, node, ofs, reg)| {
                (*func, *node, *ofs, *reg) == (span.0, ordinary_origin, 8, ordinary_reg)
            }),
        "ordinary escaped LEA lost its node-local stack identity"
    );
    assert!(
        db.rel_iter::<(Address, i64, u64)>("slot_escaped_canonical")
            .any(|(func, ofs, reg)| { (*func, *ofs, *reg) == (span.0, 8, ordinary_reg) }),
        "home-origin filtering deleted the unrelated escaped local protection"
    );
}

fn assert_optimized_canonical_homes(db: &DecompileDB) {
    for (name, position, expected_type) in [
        ("home_reassigned", 0, XType::Xany64),
        ("home_reassigned_cmp", 0, XType::Xany64),
        ("home_reassigned_cmp_reg", 0, XType::Xany64),
        ("home_reassigned_cmp_setcc", 0, XType::Xany64),
        ("home_reassigned_cmp_reg_setcc", 0, XType::Xany64),
        ("home_reassigned_cmp_loop", 0, XType::Xany64),
        ("home_cmp_after_alias_clobber", 3, XType::Xany64),
        ("home_reassigned_add", 0, XType::Xany64),
        ("home_mixed_base_clobber", 0, XType::Xany64),
        ("home_alias_call_escape", 0, XType::Xany64),
        ("home_alias_arith_clobber", 0, XType::Xany64),
        ("home_partial_movzx_low", 0, XType::Xint),
        ("home_partial_reload", 0, XType::Xany64),
    ] {
        let span = function_span(db, name);
        let slot = db
            .rel_iter::<(Address, usize, u64)>("win64_home_storage")
            .find_map(|(func, pos, slot)| {
                (*func == span.0 && *pos == position).then_some(*slot)
            })
            .unwrap_or_else(|| panic!("{name} lost canonical storage after RTLOptimize"));
        let types: HashSet<_> = db
            .rel_iter::<(u64, XType)>("emit_var_type_candidate")
            .filter_map(|(reg, xtype)| (*reg == slot).then_some(*xtype))
            .collect();
        assert_eq!(
            types,
            HashSet::from([expected_type]),
            "{name} optimizer observed a non-signature canonical-slot type"
        );
        let access_nodes: HashSet<_> = db
            .rel_iter::<(Address, Address, Mreg, i64, usize, i64)>("win64_unsafe_home_access")
            .filter_map(|(node, func, _, _, _, _)| (*func == span.0).then_some(*node))
            .collect();
        let raw_stack_regs: HashSet<_> = db
            .rel_iter::<(Address, Address, i64, u64)>("stack_var")
            .filter_map(|(func, _, _, reg)| (*func == span.0).then_some(*reg))
            .collect();
        for (node, inst) in db
            .rel_iter::<(Address, RTLInst)>("rtl_inst")
            .filter(|(node, _)| access_nodes.contains(node))
        {
            assert!(
                matches!(inst, RTLInst::Inop)
                    || matches!(inst,
                        RTLInst::Iop(Operation::Omove, args, destination)
                            if *destination == slot || args.as_ref() == &[slot])
                    || matches!(inst,
                        RTLInst::Iop(Operation::Oleal(Addressing::Ainstack(8)), args, _)
                            if args.is_empty())
                    // RTLOptimize may propagate the unique last stored SSA value
                    // through a comparison and eliminate the cell read entirely.
                    // That is safe; only a reintroduced raw stack identity is not.
                    || matches!(inst,
                        RTLInst::Icond(_, args, _, _)
                            if args.contains(&slot)
                                || args.iter().all(|arg| !raw_stack_regs.contains(arg)))
                    || matches!(inst,
                        RTLInst::Iop(Operation::Ocmp(_), args, _)
                            if args.contains(&slot)
                                || args.iter().all(|arg| !raw_stack_regs.contains(arg)))
                    || matches!(inst,
                        RTLInst::Iop(
                            Operation::Ocast8signed
                                | Operation::Ocast8unsigned
                                | Operation::Ocast16signed
                                | Operation::Ocast16unsigned
                                | Operation::Ocast32signed,
                            args,
                            _,
                        ) if args.as_ref() == &[slot]
                            || args.iter().all(|arg| !raw_stack_regs.contains(arg))),
                "{name} reintroduced split/raw storage at {node:#x}: {inst:#x?}"
            );
        }
    }

    let escaped = function_span(db, "home_alias_call_escape");
    let slot = db
        .rel_iter::<(Address, usize, u64)>("win64_home_storage")
        .find_map(|(func, pos, slot)| (*func == escaped.0 && *pos == 0).then_some(*slot))
        .unwrap();
    let optimized: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst")
        .filter(|(node, _)| in_span(*node, escaped))
        .cloned()
        .collect();
    assert!(
        optimized.iter().any(|(_, inst)| {
            matches!(inst, RTLInst::Iop(Operation::Omove, _, destination) if *destination == slot)
        }),
        "escaped home initialization was optimized away: {optimized:#x?}"
    );
    assert!(
        optimized.iter().any(|(_, inst)| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, _) if args.as_ref() == &[slot])
        }),
        "escaped home read was optimized across its call: {optimized:#x?}"
    );

    let collision = function_span(db, "home_postsub_escaped_collision");
    let ordinary_slot = db
        .rel_iter::<(Address, i64, u64)>("slot_escaped_canonical")
        .find_map(|(func, ofs, reg)| (*func == collision.0 && *ofs == 8).then_some(*reg))
        .expect("ordinary escaped collision slot lost after RTLOptimize");
    assert!(
        db.rel_iter::<(Address, RTLInst)>("rtl_inst")
            .any(|(node, inst)| {
                in_span(*node, collision)
                    && matches!(inst, RTLInst::Iop(Operation::Omove, _, destination)
                    if *destination == ordinary_slot)
            }),
        "ordinary escaped collision-slot store was dead-store-eliminated"
    );
}

fn assert_optimized_home_backing(db: &DecompileDB) {
    for name in [
        "home_partial_movzx_high_backing",
        "home_backing_movsx_high",
        "home_backing_signed_cmp_jcc",
        "home_backing_unsigned_cmp_setcc",
        "home_backing_partial_load",
        "home_backing_partial_store",
        "home_backing_add_high",
        "home_backing_test_jcc",
        "home_backing_test_setcc",
        "home_backing_cmov_cmp",
        "home_backing_cmov_test",
        "home_backing_div",
        "home_backing_descriptor",
        "home_backing_cmp_setcc",
    ] {
        let span = function_span(db, name);
        let slot = db
            .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
            .find_map(|(func, pos, slot)| {
                (*func == span.0 && *pos == 0).then_some(*slot)
            })
            .unwrap_or_else(|| panic!("{name} lost backing storage after RTLOptimize"));
        assert!(
            db.rel_iter::<(Address, RTLReg)>("win64_home_escaped")
                .any(|row| *row == (span.0, slot)),
            "{name} lost its optimization barrier"
        );
        let types: HashSet<_> = db
            .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
            .filter_map(|(reg, xtype)| (*reg == slot).then_some(*xtype))
            .collect();
        assert_eq!(
            types,
            HashSet::from([XType::Xany64]),
            "{name} optimizer changed its byte-object type"
        );
        let accesses: Vec<_> = db
            .rel_iter::<(
                Address,
                RTLReg,
                i64,
                usize,
                MemoryChunk,
                bool,
                bool,
                bool,
            )>("win64_home_backing_access")
            .filter_map(|(node, candidate_slot, _, _, _, read, write, address)| {
                (in_span(*node, span) && *candidate_slot == slot)
                    .then_some((*node, *read, *write, *address))
            })
            .collect();
        let access_nodes: HashSet<_> = accesses.iter().map(|(node, ..)| *node).collect();
        assert!(
            !access_nodes.is_empty(),
            "{name} lost all backing metadata during optimization"
        );
        assert!(
            !db.rel_iter::<(Address, Address, i64, RTLReg)>("stack_var")
                .any(|(func, node, _, _)| {
                    *func == span.0 && access_nodes.contains(node)
                }),
            "{name} optimizer reintroduced a raw stack identity"
        );
        for (access, read, write, address) in accesses {
            let optimized: Vec<_> = db
                .rel_iter::<(Address, RTLInst)>("rtl_inst")
                .filter_map(|(node, inst)| {
                    ((*node & !SYNTHETIC_NODE_MASK) == access)
                        .then_some((*node, inst.clone()))
                })
                .collect();
            assert!(
                !optimized.is_empty(),
                "{name} lost its selected optimized RTL class at {access:#x}"
            );
            let unique_nodes: HashSet<_> = optimized.iter().map(|(node, _)| *node).collect();
            assert_eq!(
                unique_nodes.len(),
                optimized.len(),
                "{name} optimizer retained competing candidates at {access:#x}: {optimized:#x?}"
            );
            let uses_slot = optimized.iter().any(|(_, inst)| {
                test_rtl_inst_uses(inst, slot) || matches!(inst, RTLInst::Iload(..))
            });
            let writes_slot = optimized.iter().any(|(_, inst)| {
                matches!(inst, RTLInst::Istore(..))
                    || matches!(inst, RTLInst::Iop(_, _, destination) if *destination == slot)
            });
            let takes_address = optimized.iter().any(|(_, inst)| {
                matches!(inst,
                    RTLInst::Iop(
                        Operation::Olea(Addressing::Ainstack(_))
                            | Operation::Oleal(Addressing::Ainstack(_)),
                        _,
                        _
                    ))
            });
            assert_eq!(uses_slot, read, "{name} changed the selected read mode");
            assert_eq!(writes_slot, write, "{name} changed the selected write mode");
            assert_eq!(takes_address, address, "{name} changed the selected address mode");
        }
    }
}

fn assert_post_type_canonical_homes(db: &DecompileDB) {
    let selected: Vec<_> = db
        .rel_iter::<(Address, usize, u64)>("win64_home_storage")
        .copied()
        .collect();
    assert!(
        !selected.is_empty(),
        "TypePass lost all canonical home storage"
    );
    for (_, _, slot) in selected {
        let locked: Vec<_> = db
            .rel_iter::<(u64, XType)>("win64_home_slot_type")
            .filter_map(|(reg, xtype)| (*reg == slot).then_some(*xtype))
            .collect();
        assert_eq!(locked.len(), 1, "canonical home must have one type lock");
        let candidates: HashSet<_> = db
            .rel_iter::<(u64, XType)>("emit_var_type_candidate")
            .filter_map(|(reg, xtype)| (*reg == slot).then_some(*xtype))
            .collect();
        assert_eq!(
            candidates,
            HashSet::from([locked[0]]),
            "TypePass Omove propagation contaminated canonical home slot {slot:#x}"
        );
    }
}

fn stack_vars_at(
    db: &DecompileDB,
    function: Address,
    address: Address,
    offset: i64,
) -> HashSet<u64> {
    db.rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .filter_map(|(func, candidate, ofs, reg)| {
            (*func == function && *candidate == address && *ofs == offset).then_some(*reg)
        })
        .collect()
}

fn indexed_store_at(db: &DecompileDB, span: (Address, Address)) -> Address {
    db.rel_iter::<(Address, MachInst)>("mach_inst")
        .find_map(|(address, inst)| {
            (in_span(*address, span)
                && matches!(
                    inst,
                    MachInst::Mstore(MemoryChunk::MInt32, Addressing::Aindexed2scaled(4, 8), _, _)
                ))
            .then_some(*address)
        })
        .unwrap_or_else(|| panic!("missing indexed store in {span:#x?}"))
}

fn assert_indexed_stack_cells_use_normalized_coordinates(db: &DecompileDB) {
    let home = function_span(db, "home_vs_postsub_indexed");
    let home_spill = db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
        .find_map(|(addr, func, _, pos)| (*func == home.0 && *pos == 0).then_some(*addr))
        .expect("missing pre-prologue home spill");
    let indexed = indexed_store_at(db, home);
    assert!(
        !db.rel_iter::<(Address, Address, usize)>("win64_home_hard_veto_site")
            .any(|(node, func, _)| (*node, *func) == (indexed, home.0)),
        "disjoint post-prologue indexed local was classified as a hard home access"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, node, _)| (*func, *node) == (home.0, indexed)),
        "disjoint post-prologue indexed local was marked unsupported"
    );
    let home_vars = stack_vars_at(db, home.0, home_spill, 8);
    let indexed_vars = stack_vars_at(db, home.0, indexed, 8);
    assert!(!home_vars.is_empty() && !indexed_vars.is_empty());
    assert!(
        home_vars.is_disjoint(&indexed_vars),
        "pre-sub home and post-sub indexed local share only a raw displacement: home={home_vars:#x?}, indexed={indexed_vars:#x?}"
    );
    let postsub_scalar = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .find_map(|(addr, inst)| {
            (in_span(*addr, home)
                && *addr != home_spill
                && matches!(inst, LTLInst::Lsetstack(_, _, 8, _)))
            .then_some(*addr)
        })
        .expect("missing post-sub scalar slot beside indexed local");
    let postsub_scalar_vars = stack_vars_at(db, home.0, postsub_scalar, 8);
    assert!(
        !postsub_scalar_vars.is_disjoint(&indexed_vars),
        "same-depth scalar and indexed stack cells did not alias: scalar={postsub_scalar_vars:#x?}, indexed={indexed_vars:#x?}"
    );

    let depths = function_span(db, "two_sp_depths");
    let first_scalar = db
        .rel_iter::<(Address, LTLInst)>("ltl_inst")
        .find_map(|(addr, inst)| {
            (in_span(*addr, depths) && matches!(inst, LTLInst::Lsetstack(_, _, 8, _)))
                .then_some(*addr)
        })
        .expect("missing first-depth scalar slot");
    let second_indexed = indexed_store_at(db, depths);
    let first_vars = stack_vars_at(db, depths.0, first_scalar, 8);
    let second_vars = stack_vars_at(db, depths.0, second_indexed, 8);
    assert!(!first_vars.is_empty() && !second_vars.is_empty());
    assert!(
        first_vars.is_disjoint(&second_vars),
        "equal raw offsets at distinct SP depths must not alias: first={first_vars:#x?}, second={second_vars:#x?}"
    );
}

fn assert_stack_address_destinations_are_not_storage(db: &DecompileDB) {
    let mut checked = 0usize;
    for (node, inst) in db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate") {
        let destination = match inst {
            RTLInst::Iop(
                Operation::Olea(Addressing::Ainstack(_))
                | Operation::Oleal(Addressing::Ainstack(_)),
                _,
                destination,
            ) => *destination,
            _ => continue,
        };
        let cells: Vec<_> = db
            .rel_iter::<(Address, Address, i64, u64)>("stack_var")
            .filter_map(|(_, candidate, _, cell)| (*candidate == *node).then_some(*cell))
            .collect();
        if cells.is_empty() {
            continue;
        }
        checked += 1;
        assert!(
            !cells.contains(&destination),
            "stack address destination aliases its own storage at {node:#x}: destination={destination:#x}, cells={cells:#x?}"
        );
    }
    assert!(
        checked > 0,
        "fixture did not exercise a named stack address"
    );
}

fn assert_unknown_sp_and_high_offsets_do_not_infer_params(db: &DecompileDB) {
    for name in [
        "dynamic_sp_unknown",
        "branch_stack_unknown",
        "high_stack_offset",
    ] {
        let span = function_span(db, name);
        assert!(
            !db.rel_iter::<(Address, usize)>("stack_param_ordinal")
                .any(|(func, _)| *func == span.0),
            "{name} fabricated a stack parameter"
        );
        assert!(
            !db.rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
                .any(|(_, func, _, _)| *func == span.0),
            "{name} fabricated a concrete stack-parameter access"
        );
    }
}

fn assert_incoming_stack_rmw_is_initialized_and_survives(db: &DecompileDB) {
    for name in ["fifth_inc", "fifth_add_reg"] {
        let span = function_span(db, name);
        let access = db
            .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
            .find(|(_, func, disp, ordinal)| *func == span.0 && *disp == 40 && *ordinal == 0)
            .copied()
            .unwrap_or_else(|| panic!("{name} lost its fifth-parameter RMW"));
        assert!(db
            .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
            .any(|(func, count)| (*func, *count) == (span.0, 5)));

        let local = stack_vars_at(db, span.0, access.0, 40);
        assert!(
            !local.is_empty(),
            "{name} did not materialize mutable storage"
        );
        let init = rtl_candidates(db, access.0);
        assert!(init.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(Operation::Omove, args, dst) if args.len() == 1 && local.contains(dst))
        }), "{name} did not initialize the mutable local: {init:#x?}");
        let update = rtl_candidates(db, access.0 | SYNTH1);
        assert!(update.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(op, args, dst) if *op != Operation::Omove && !args.is_empty() && local.contains(dst))
        }), "{name} RMW update disappeared: {update:#x?}");
    }

    let overlap = function_span(db, "alias_partial_before_fifth_inc");
    let rmw = db
        .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64)>("arith_store_imm")
        .find_map(|(address, _, chunk, base, disp)| {
            (in_span(*address, overlap)
                && *chunk == MemoryChunk::MInt32
                && *base == Mreg::SP
                && *disp == 40)
                .then_some(*address)
        })
        .expect("missing copied-alias overlap fixture RMW");
    let prior = db
        .rel_iter::<(Address, Address)>("stack_param_partial_write")
        .find_map(|(candidate, write)| (*candidate == rmw).then_some(*write))
        .expect("copied-SP byte write did not veto pristine fifth-parameter seeding");
    assert!(
        db.rel_iter::<(Address, Address, Mreg, i64, i64, i64)>("normalized_stack_write_range",)
            .any(|(node, func, base, disp, start, end)| {
                (*node, *func, *base, *disp, *start, *end)
                    == (prior, overlap.0, Mreg::R10, 41, 41, 42)
            }),
        "copied-SP prior write lost its exact normalized byte range"
    );
    assert!(
        !db.rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
            .any(|(node, func, disp, _)| { (*node, *func, *disp) == (rmw, overlap.0, 40) }),
        "overlapped RMW was initialized again from the pristine ABI parameter"
    );
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(func, access, reason)| {
            (*func, *access, *reason) == (overlap.0, rmw, "unsupported-stack-address")
        }));
    let candidates = address_bearing_candidates(db, rmw);
    assert!(
        candidates.is_empty(),
        "overlapped copied-SP RMW retained address-bearing RTL: {candidates:#x?}"
    );
}

fn assert_outgoing_home_coordinate_is_not_suppressed(db: &DecompileDB) {
    let caller = function_span(db, "outgoing_reuses_home_coordinate");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
        .any(|(_, func, _, _)| *func == caller.0));
    let call = db
        .rel_iter::<(Address, Address)>("call_target_func")
        .find_map(|(site, target)| {
            let callee = function_span(db, "callee_seventh");
            (in_span(*site, caller) && *target == callee.0).then_some(*site)
        })
        .expect("missing outgoing-coordinate fixture call");
    assert!(
        db.rel_iter::<(Address, usize)>("call_has_arg_evidence")
            .any(|(site, position)| (*site, *position) == (call, 6)),
        "outgoing seventh argument was suppressed as a home spill"
    );
}

fn assert_resolved_import_tailcall_keeps_frame_proofs(db: &DecompileDB) {
    let span = function_span(db, "home_import_tailcall");
    assert!(
        db.rel_iter::<(Address,)>("is_extern_tailcall_jmp")
            .any(|(address,)| in_span(*address, span)),
        "COFF import-pointer JMP was not recognized as a resolved tail call"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, _, _)| *func == span.0),
        "resolved outgoing import tail call poisoned the function's frame provenance"
    );
}

fn assert_indirect_tailcall_proofs_are_structural(db: &DecompileDB) {
    let typed_slot = function_span(db, "home_typed_slot_tailcall");
    assert!(
        db.rel_iter::<(Address,)>("is_extern_tailcall_jmp")
            .any(|(address,)| in_span(*address, typed_slot)),
        "function-typed RIP memory slot was not classified as an import tail call"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, _, _)| *func == typed_slot.0),
        "typed import-pointer tail call poisoned frame provenance"
    );

    for name in [
        "home_epilogue_register_tailcall",
        "home_adjacent_epilogue_register_tailcall",
    ] {
        let epilogue = function_span(db, name);
        let proven_jump = db
            .rel_iter::<(Address,)>("is_epilogue_indirect_tailcall_jmp")
            .find_map(|(address,)| in_span(*address, epilogue).then_some(*address))
            .unwrap_or_else(|| {
                panic!("{name}: restored register tail call lacked an epilogue proof")
            });
        let tailcalls: Vec<_> = db
            .rel_iter::<(Address, MachInst)>("mach_inst")
            .filter(|(address, inst)| {
                in_span(*address, epilogue) && matches!(inst, MachInst::Mtailcall(_))
            })
            .collect();
        assert_eq!(
            tailcalls.len(),
            1,
            "{name}: structural and adjacent-pair rules emitted duplicate tail calls"
        );
        assert_eq!(tailcalls[0].0, proven_jump);
        assert!(!db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, _, _)| *func == epilogue.0));
    }

    for name in [
        "home_live_frame_register_dispatch",
        "home_frameless_register_dispatch",
    ] {
        let span = function_span(db, name);
        assert!(
            !db.rel_iter::<(Address,)>("is_epilogue_indirect_tailcall_jmp")
                .any(|(address,)| in_span(*address, span)),
            "{name} received a tail-call proof without a dominated, restored frame"
        );
        assert!(
            db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, _, _)| *func == span.0),
            "{name} no longer guards unresolved internal control flow"
        );
    }
}

fn assert_bp_postprologue_and_xmm_forms(db: &DecompileDB) {
    for name in [
        "bp_fifth",
        "postprologue_fifth",
        "trap_rsp_fallthrough",
        "trap_int2c_rsp_fallthrough",
        "bp_postalloc_fifth",
    ] {
        let span = function_span(db, name);
        assert!(db
            .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
            .any(|(_, func, _, ordinal)| *func == span.0 && *ordinal == 0));
        assert!(db
            .rel_iter::<(Address, usize)>("emit_function_param_count_candidate")
            .any(|(func, count)| (*func, *count) == (span.0, 5)));
        assert!(
            !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, _, _)| *func == span.0),
            "{name} lost its proved stack coordinate"
        );
    }

    let xmm = function_span(db, "xmm_home_roundtrip");
    assert!(db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, reg, pos)| (*func, *reg, *pos) == (xmm.0, Mreg::X0, 0)));
    assert!(db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
        .any(|(_, func, reg, pos)| (*func, *reg, *pos) == (xmm.0, Mreg::X0, 0)));
}

fn assert_bp_provenance_preserves_pointer_memory(db: &DecompileDB) {
    let direct = function_span(db, "bp_conditional_pointer");
    let direct_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, direct))
        .cloned()
        .collect();
    assert!(
        direct_mach.iter().any(|(_, inst)| {
            matches!(inst,
                MachInst::Mstore(MemoryChunk::MInt32, Addressing::Aindexed(40), args, Mreg::DX)
                    if args.as_ref() == &[Mreg::BP]
            )
        }),
        "conditional BP store was fabricated as a frame slot: {direct_mach:#x?}"
    );
    assert!(
        direct_mach.iter().any(|(_, inst)| {
            matches!(inst,
                MachInst::Mload(MemoryChunk::MInt32, Addressing::Aindexed(40), args, Mreg::AX)
                    if args.as_ref() == &[Mreg::BP]
            )
        }),
        "conditional BP load was fabricated as a frame slot: {direct_mach:#x?}"
    );
    assert!(
        !direct_mach.iter().any(|(_, inst)| {
            matches!(
                inst,
                MachInst::Mgetstack(40, _, _) | MachInst::Msetstack(_, 40, _)
            )
        }),
        "conditional BP accesses must remain pointer memory: {direct_mach:#x?}"
    );

    let indexed = function_span(db, "bp_conditional_indexed_pointer");
    let indexed_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, indexed))
        .cloned()
        .collect();
    assert!(
        indexed_mach.iter().any(|(_, inst)| {
            matches!(inst,
                MachInst::Mload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 40),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::BP, Mreg::DX]
            )
        }),
        "conditional indexed BP load lost pointer semantics: {indexed_mach:#x?}"
    );
    let indexed_access = indexed_mach
        .iter()
        .find_map(|(address, inst)| {
            matches!(inst,
                MachInst::Mload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 40),
                    args,
                    Mreg::AX
                ) if args.as_ref() == &[Mreg::BP, Mreg::DX]
            )
            .then_some(*address)
        })
        .expect("missing conditional indexed BP load address");
    let indexed_rtl = address_bearing_candidates(db, indexed_access);
    assert!(
        indexed_rtl.iter().any(|inst| {
            matches!(inst,
                RTLInst::Iload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 40),
                    args,
                    _
                ) if args.len() == 2
            )
        }),
        "conditional indexed BP load did not use generic pointer RTL: {indexed_rtl:#x?}"
    );
    assert!(
        !indexed_rtl.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iop(
                    Operation::Olea(Addressing::Ainstack(_))
                        | Operation::Oleal(Addressing::Ainstack(_)),
                    _,
                    _
                )
            )
        }),
        "conditional indexed BP load fabricated a stack base: {indexed_rtl:#x?}"
    );

    let clobber = function_span(db, "bp_conditional_clobber");
    let clobber_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, clobber))
        .cloned()
        .collect();
    assert!(
        clobber_mach.iter().any(|(_, inst)| {
            matches!(inst,
                MachInst::Mload(MemoryChunk::MInt32, Addressing::Aindexed(40), args, Mreg::AX)
                    if args.as_ref() == &[Mreg::BP]
            )
        }),
        "conditionally clobbered BP load became a frame slot: {clobber_mach:#x?}"
    );

    for span in [direct, indexed, clobber] {
        assert!(!db
            .rel_iter::<(Address, Address)>("bp_frame_at")
            .any(|(access, func)| *func == span.0 && in_span(*access, span)));
        assert!(!db
            .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
            .any(|(_, func, _, _)| *func == span.0));
    }

    let postalloc = function_span(db, "bp_postalloc_fifth");
    assert!(db
        .rel_iter::<(Address, Address)>("bp_frame_at")
        .any(|(access, func)| *func == postalloc.0 && in_span(*access, postalloc)));
}

fn assert_bp_shortcuts_and_narrow_bases_stay_pointer_memory(db: &DecompileDB) {
    for name in ["bp_conditional_lea", "bp_clobbered_lea"] {
        let span = function_span(db, name);
        let mach: Vec<_> = db
            .rel_iter::<(Address, MachInst)>("mach_inst")
            .filter(|(address, _)| in_span(*address, span))
            .cloned()
            .collect();
        assert!(
            mach.iter().any(|(_, inst)| {
                matches!(
                    inst,
                    MachInst::Mop(Operation::Olea(Addressing::Aindexed(40)), args, Mreg::AX)
                        if args.as_ref() == &[Mreg::BP]
                )
            }),
            "{name} lost its generic RBP pointer LEA: {mach:#x?}"
        );
        assert!(
            !mach.iter().any(|(_, inst)| {
                matches!(
                    inst,
                    MachInst::Mop(
                        Operation::Olea(Addressing::Ainstack(40))
                            | Operation::Oleal(Addressing::Ainstack(40)),
                        _,
                        _
                    )
                )
            }),
            "{name} fabricated a stack-slot address: {mach:#x?}"
        );
        assert!(!db
            .rel_iter::<(Address, Address, i64)>("bp_base_at")
            .any(|(func, access, _)| *func == span.0 && in_span(*access, span)));
    }

    for (name, expected_op) in [
        ("bp_conditional_rmw", Operation::Oaddimm(1)),
        ("bp_clobbered_rmw", Operation::Oaddimm(-1)),
    ] {
        let span = function_span(db, name);
        let rmw_addr = db
            .rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64)>("arith_store_imm")
            .find_map(|(address, op, chunk, base, disp)| {
                (in_span(*address, span)
                    && *op == expected_op
                    && *chunk == MemoryChunk::MInt32
                    && *base == Mreg::BP
                    && *disp == 40)
                    .then_some(*address)
            })
            .unwrap_or_else(|| panic!("{name} lost its pointer RMW relation"));
        assert!(!db
            .rel_iter::<(Address, i64, i64, usize)>("stack_mem_add_imm")
            .any(|(address, _, _, _)| in_span(*address, span)));
        assert!(!db
            .rel_iter::<(Address, i64, i64, usize)>("stack_mem_sub_imm")
            .any(|(address, _, _, _)| in_span(*address, span)));
        assert!(!db
            .rel_iter::<(Address, Address, i64)>("bp_base_at")
            .any(|(func, access, _)| (*func, *access) == (span.0, rmw_addr)));
        let load = rtl_candidates(db, rmw_addr);
        assert!(
            load.iter().any(|inst| {
                matches!(
                    inst,
                    RTLInst::Iload(MemoryChunk::MInt32, Addressing::Aindexed(40), args, _)
                        if args.len() == 1
                )
            }),
            "{name} collapsed its pointer RMW into a stack variable: {load:#x?}"
        );
        let store = rtl_candidates(db, rmw_addr | (1u64 << 63));
        assert!(
            store.iter().any(|inst| {
                matches!(
                    inst,
                    RTLInst::Istore(MemoryChunk::MInt32, Addressing::Aindexed(40), args, _)
                        if args.len() == 1
                )
            }),
            "{name} lost the generic pointer store: {store:#x?}"
        );
    }

    for (name, base) in [("narrow_ebp_pointer", Mreg::BP)] {
        let span = function_span(db, name);
        let mach: Vec<_> = db
            .rel_iter::<(Address, MachInst)>("mach_inst")
            .filter(|(address, _)| in_span(*address, span))
            .cloned()
            .collect();
        assert!(
            mach.iter().any(|(_, inst)| {
                matches!(
                    inst,
                    MachInst::Mload(
                        MemoryChunk::MInt32,
                        Addressing::Aaddr32(inner),
                        args,
                        Mreg::AX
                    ) if matches!(inner.as_ref(), Addressing::Aindexed(40))
                        && args.as_ref() == &[base]
                )
            }),
            "{name} lost its address-size-overridden pointer load: {mach:#x?}"
        );
        assert!(!mach
            .iter()
            .any(|(_, inst)| matches!(inst, MachInst::Mgetstack(40, _, _))));
        assert!(!db
            .rel_iter::<(Address, Mreg, i64, usize)>("direct_stack_operand")
            .any(|(address, seen_base, _, _)| { in_span(*address, span) && *seen_base == base }));
        assert!(!db
            .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
            .any(|(_, func, _, _)| *func == span.0));
    }

    let narrow_esp = function_span(db, "narrow_esp_pointer");
    let narrow_esp_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter(|(address, _)| in_span(*address, narrow_esp))
        .cloned()
        .collect();
    assert!(
        narrow_esp_mach
            .iter()
            .all(|(_, inst)| !matches!(inst, MachInst::Mload(..) | MachInst::Mgetstack(..))),
        "addr32 ESP access was mis-lowered as an ordinary stack/pointer load: {narrow_esp_mach:#x?}"
    );
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address_seed")
            .any(|(func, address, reason)| {
                *func == narrow_esp.0
                    && in_span(*address, narrow_esp)
                    && *reason == "unsupported-addr32-address"
            }),
        "addr32 ESP access lacks its structured unsupported diagnostic"
    );
    assert!(!db
        .rel_iter::<(Address, Mreg, i64, usize)>("direct_stack_operand")
        .any(|(address, base, _, _)| { in_span(*address, narrow_esp) && *base == Mreg::SP }));
    assert!(!db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(_, func, _, _)| *func == narrow_esp.0));

    let narrow_alias = function_span(db, "narrow_eax_stack_alias");
    assert!(
        !db.rel_iter::<(Address, Mreg, i64, usize)>("direct_stack_operand")
            .any(|(address, base, _, _)| { in_span(*address, narrow_alias) && *base == Mreg::AX }),
        "[eax] was accepted as a 64-bit copied-stack operand"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Mreg, i64, usize)>("win64_home_cell")
            .any(|(_, func, _, _, _)| *func == narrow_alias.0),
        "[eax] fabricated a Win64 home-cell access"
    );
    assert!(
        !db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
            .any(|(_, func, _, _)| *func == narrow_alias.0),
        "[eax] fabricated a Win64 home spill"
    );
    assert!(
        !db.rel_iter::<(Address, usize, u64)>("win64_home_storage")
            .any(|(func, _, _)| *func == narrow_alias.0),
        "[eax] fabricated canonical home storage"
    );

    let pop = function_span(db, "pop_rsp_unknown");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, i64, usize)>("incoming_stack_slot")
        .any(|(_, func, _, _, _)| *func == pop.0));
    assert!(!db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(_, func, _, _)| *func == pop.0));
}

fn assert_home_safety_vetoes(db: &DecompileDB) {
    for (name, expected_reg) in [
        ("home_unknown_indexed_bp", Mreg::CX),
        ("home_ambiguous_alias_call_escape", Mreg::CX),
        ("home_direct_alias_call_escape", Mreg::CX),
        ("home_direct_alias_return", Mreg::CX),
        ("home_cross_class_reload", Mreg::CX),
        ("home_xmm_mutation", Mreg::X0),
        ("home_xchg_mutation", Mreg::CX),
        ("home_conditional_spill_reload", Mreg::CX),
        ("home_wide_overlap", Mreg::CX),
        ("home_escape_numeric_use", Mreg::CX),
        ("home_cmp_flag_clobber_rejected", Mreg::CX),
        ("home_cmp_scheduled_rejected", Mreg::CX),
        ("home_backing_cmp_flag_clobber_rejected", Mreg::CX),
        ("home_backing_test_scheduled_rejected", Mreg::CX),
        ("home_backing_cmov_scheduled_rejected", Mreg::CX),
        ("home_backing_add_jcc_rejected", Mreg::CX),
        ("home_backing_add_scheduled_jcc_rejected", Mreg::CX),
        ("home_backing_add_scheduled_setcc_rejected", Mreg::CX),
        ("home_backing_add_scheduled_cmov_rejected", Mreg::CX),
        ("home_backing_add_scheduled_adc_rejected", Mreg::CX),
        ("home_backing_add_scheduled_rcr_rejected", Mreg::CX),
        ("home_backing_add_scheduled_setp_rejected", Mreg::CX),
        ("home_backing_add_scheduled_setnp_rejected", Mreg::CX),
        ("home_backing_add_scheduled_jo_rejected", Mreg::CX),
        ("home_backing_add_scheduled_jno_rejected", Mreg::CX),
        ("home_backing_add_scheduled_seto_rejected", Mreg::CX),
        ("home_backing_add_scheduled_setno_rejected", Mreg::CX),
        ("home_backing_add_scheduled_cmovo_rejected", Mreg::CX),
        ("home_backing_add_scheduled_cmovno_rejected", Mreg::CX),
        ("home_backing_lock_add_rejected", Mreg::CX),
        ("home_backing_cmovs_rejected", Mreg::CX),
        ("home_backing_cross_cell_rejected", Mreg::CX),
        ("home_backing_indexed_rejected", Mreg::CX),
        ("home_backing_rbp_xchg_rejected", Mreg::CX),
        ("home_backing_partial_pointer_store_rejected", Mreg::CX),
        ("home_backing_segmented_store_rejected", Mreg::CX),
        ("home_backing_segmented_load_rejected", Mreg::CX),
    ] {
        let span = function_span(db, name);
        assert!(
            db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill_candidate")
                .any(|(_, func, reg, pos)| { (*func, *reg, *pos) == (span.0, expected_reg, 0) }),
            "{name} lost its positive initial-spill evidence"
        );
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
                .any(|(_, func, _, _)| *func == span.0),
            "{name} unsafely folded its home spill"
        );
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, usize)>("win64_home_reload")
                .any(|(_, func, _, _)| *func == span.0),
            "{name} unsafely folded a reload"
        );
        assert!(
            !db.rel_iter::<(Address, usize, u64)>("win64_home_storage")
                .any(|(func, _, _)| *func == span.0),
            "{name} received canonical storage despite an unsupported access"
        );
    }

    for name in [
        "home_cross_class_reload",
        "home_xmm_mutation",
        "home_xchg_mutation",
        "home_wide_overlap",
        "home_cmp_flag_clobber_rejected",
        "home_cmp_scheduled_rejected",
        "home_backing_cmp_flag_clobber_rejected",
        "home_backing_test_scheduled_rejected",
        "home_backing_cmov_scheduled_rejected",
        "home_backing_add_jcc_rejected",
        "home_backing_add_scheduled_jcc_rejected",
        "home_backing_add_scheduled_setcc_rejected",
        "home_backing_add_scheduled_cmov_rejected",
        "home_backing_add_scheduled_adc_rejected",
        "home_backing_add_scheduled_rcr_rejected",
        "home_backing_add_scheduled_setp_rejected",
        "home_backing_add_scheduled_setnp_rejected",
        "home_backing_add_scheduled_jo_rejected",
        "home_backing_add_scheduled_jno_rejected",
        "home_backing_add_scheduled_seto_rejected",
        "home_backing_add_scheduled_setno_rejected",
        "home_backing_add_scheduled_cmovo_rejected",
        "home_backing_add_scheduled_cmovno_rejected",
        "home_backing_lock_add_rejected",
        "home_backing_cmovs_rejected",
        "home_backing_cross_cell_rejected",
        "home_backing_indexed_rejected",
        "home_backing_rbp_xchg_rejected",
        "home_backing_partial_pointer_store_rejected",
        "home_backing_segmented_store_rejected",
        "home_backing_segmented_load_rejected",
    ] {
        let span = function_span(db, name);
        let mut rejected_sites: Vec<_> = db
            .rel_iter::<(Address, Address, usize)>("win64_home_overlap")
            .filter_map(|(access, func, _)| {
                ((*func == span.0)
                    && db
                        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                        .any(|(unsupported_func, unsupported_access, reason)| {
                            (*unsupported_func, *unsupported_access, *reason)
                                == (span.0, *access, "unsupported-stack-address")
                        }))
                .then_some(*access)
            })
            .collect();
        rejected_sites.extend(
            db.rel_iter::<(Address, Address, usize)>("win64_home_hard_veto_site")
                .filter_map(|(access, func, _)| {
                    ((*func == span.0)
                        && db.rel_iter::<(Address, Address, Symbol)>(
                            "unsupported_stack_address",
                        )
                        .any(|(unsupported_func, unsupported_access, reason)| {
                            (*unsupported_func, *unsupported_access, *reason)
                                == (span.0, *access, "unsupported-stack-address")
                        }))
                    .then_some(*access)
                }),
        );
        rejected_sites.sort_unstable();
        rejected_sites.dedup();
        assert!(
            !rejected_sites.is_empty(),
            "{name} lost structured rejection for its non-scalar home access"
        );
        for access in rejected_sites {
            let candidates = address_bearing_candidates(db, access);
            assert!(
                candidates.is_empty(),
                "{name} retained address-bearing RTL at rejected home access {access:#x}: {candidates:#x?}"
            );
        }
    }

    let indexed_home = function_span(db, "home_backing_indexed_rejected");
    let hard_sites: Vec<_> = db
        .rel_iter::<(Address, Address, usize)>("win64_home_hard_veto_site")
        .filter_map(|(node, func, pos)| {
            (*func == indexed_home.0).then_some((*node, *pos))
        })
        .collect();
    assert_eq!(
        hard_sites.len(),
        1,
        "indexed home fixture must expose one exact hard-veto site: {hard_sites:#x?}"
    );
    assert_eq!(hard_sites[0].1, 0);
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
            .any(|(func, node, detail)| {
                (*func, *node, *detail)
                    == (
                        indexed_home.0,
                        hard_sites[0].0,
                        "home-cell-shape-unrepresentable",
                    )
            }),
        "indexed home hard-veto site lost its exact structured detail"
    );

    for name in [
        "home_backing_segmented_store_rejected",
        "home_backing_segmented_load_rejected",
    ] {
        let span = function_span(db, name);
        let segmented: Vec<_> = db
            .rel_iter::<(Address,)>("win64_home_segmented_memory_access")
            .filter_map(|(access,)| in_span(*access, span).then_some(*access))
            .collect();
        assert_eq!(
            segmented.len(),
            1,
            "{name} lost its exact segmented-memory witness: {segmented:#x?}"
        );
        let access = segmented[0];
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, i64, usize, i64, i64, usize)>(
                "win64_home_bounded_access",
            )
            .any(|(node, _, _, _, _, _, _, _)| *node == access),
            "{name} admitted segmented memory into bounded home backing"
        );
        assert!(
            !db.rel_iter::<(Address, Address, Mreg, i64, usize, i64, i64, usize)>(
                "win64_home_backing_candidate",
            )
            .any(|(node, _, _, _, _, _, _, _)| *node == access),
            "{name} promoted segmented memory into a backing candidate"
        );
        assert!(
            !db.rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
                "win64_home_backing_move_store",
            )
            .any(|(node, _, _, _, _, _)| *node == access)
                && !db.rel_iter::<(Address, Address, i64, usize, RTLReg, RTLReg)>(
                    "win64_home_backing_move_load",
                )
                .any(|(node, _, _, _, _, _)| *node == access),
            "{name} resurrected a segmented access through backing MOV"
        );
    }

    for (name, consumer_mnemonic) in [
        ("home_backing_add_jcc_rejected", "JNE"),
        ("home_backing_add_scheduled_jcc_rejected", "JNE"),
        ("home_backing_add_scheduled_setcc_rejected", "SETNE"),
        ("home_backing_add_scheduled_cmov_rejected", "CMOVNE"),
        ("home_backing_add_scheduled_adc_rejected", "ADC"),
        ("home_backing_add_scheduled_rcr_rejected", "RCR"),
        ("home_backing_add_scheduled_setp_rejected", "SETP"),
        ("home_backing_add_scheduled_setnp_rejected", "SETNP"),
        ("home_backing_add_scheduled_jo_rejected", "JO"),
        ("home_backing_add_scheduled_jno_rejected", "JNO"),
        ("home_backing_add_scheduled_seto_rejected", "SETO"),
        ("home_backing_add_scheduled_setno_rejected", "SETNO"),
        ("home_backing_add_scheduled_cmovo_rejected", "CMOVO"),
        ("home_backing_add_scheduled_cmovno_rejected", "CMOVNO"),
    ] {
        let span = function_span(db, name);
        let flagged_writes: Vec<_> = db
            .rel_iter::<(Address,)>("win64_home_backing_flagged_write")
            .filter_map(|(node,)| in_span(*node, span).then_some(*node))
            .collect();
        assert_eq!(
            flagged_writes.len(),
            1,
            "{name} did not retain one exact live-flags veto: {flagged_writes:#x?}"
        );
        assert!(
            db.rel_iter::<InstructionRow>("instruction").any(
                |(node, _, _, mnemonic, _, _, _, _, _, _)| {
                    *node == flagged_writes[0] && *mnemonic == "ADD"
                }
            ),
            "{name} attached its live-flags veto to a node other than the home ADD"
        );
        let consumers: Vec<_> = db
            .rel_iter::<InstructionRow>("instruction")
            .filter_map(|(node, _, _, mnemonic, _, _, _, _, _, _)| {
                (in_span(*node, span) && *mnemonic == consumer_mnemonic).then_some(*node)
            })
            .collect();
        assert_eq!(
            consumers.len(),
            1,
            "{name} lost its exact raw {consumer_mnemonic} consumer: {consumers:#x?}"
        );
        assert!(
            consumers[0] > flagged_writes[0],
            "{name} raw {consumer_mnemonic} does not follow the flagged home ADD"
        );
    }
    let lock = function_span(db, "home_backing_lock_add_rejected");
    assert!(
        db.rel_iter::<(Address,)>("win64_home_backing_lock_access")
            .any(|(node,)| in_span(*node, lock)),
        "LOCK-prefixed backing access lost its decoder-owned veto"
    );
    let cmovs = function_span(db, "home_backing_cmovs_rejected");
    assert!(
        db.rel_iter::<(Address,)>("win64_home_backing_cmov_sf")
            .any(|(node,)| in_span(*node, cmovs)),
        "CMOVS backing access lost its SF-semantics veto"
    );

    for name in [
        "home_ambiguous_alias_call_escape",
        "home_direct_alias_call_escape",
    ] {
        let span = function_span(db, name);
        let overlaps = db
            .rel_iter::<(Address, Address, usize)>("win64_home_overlap")
            .filter(|(_, func, pos)| (*func, *pos) == (span.0, 0))
            .count();
        assert_eq!(
            overlaps, 2,
            "{name} must be vetoed by alias escape, not an extra overlapping access"
        );
    }

    let mixed_use = function_span(db, "home_escape_numeric_use");
    assert!(
        db.rel_iter::<(Address, usize)>("win64_home_canonical_veto")
            .any(|(func, pos)| (*func, *pos) == (mixed_use.0, 0)),
        "an exact home-address web used by both a call and ADD was canonicalized"
    );
    assert!(
        db.rel_iter::<(Address, Address, Mreg, i64, usize, i64, Mreg)>("win64_home_scalar_lea",)
            .any(|(_, func, _, _, pos, _, _)| (*func, *pos) == (mixed_use.0, 0)),
        "mixed-use regression never reached the exact scalar-LEA selector path"
    );

    // A call kills volatile R10. Its post-call arithmetic/load must therefore
    // be ordinary unknown-pointer work, not a function-wide remembered stack
    // alias that prevents the otherwise safe /homeparams fold.
    let killed_alias = function_span(db, "home_volatile_alias_after_call");
    assert!(db
        .rel_iter::<(Address, Address, Mreg, usize)>("win64_home_spill")
        .any(|(_, func, reg, pos)| { (*func, *reg, *pos) == (killed_alias.0, Mreg::CX, 0) }));
    assert!(!db
        .rel_iter::<(Address, usize, u64)>("win64_home_storage")
        .any(|(func, _, _)| *func == killed_alias.0));
    let killed_call = db
        .rel_iter::<(Address, Address)>("call_target_func")
        .find_map(|(address, _)| in_span(*address, killed_alias).then_some(*address))
        .expect("volatile-alias fixture lost its direct call");
    assert!(!db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .any(|(func, access, reg, _)| {
            *func == killed_alias.0 && *access > killed_call && *reg == Mreg::R10
        }));

    for name in [
        "wrong_home_source",
        "home_disjoint_branch",
        "home_vs_postsub_indexed",
    ] {
        let span = function_span(db, name);
        assert!(
            !db.rel_iter::<(Address, usize, u64)>("win64_home_storage")
                .any(|(func, _, _)| *func == span.0),
            "{name} unexpectedly received canonical home storage"
        );
    }
}

fn assert_alias_coordinate_provenance(db: &DecompileDB) {
    let mov_lea = function_span(db, "alias_mov_vs_lea_coordinates");
    let rows: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .filter(|(func, _, _, _)| *func == mov_lea.0)
        .copied()
        .collect();
    assert!(
        rows.iter()
            .any(|(_, _, reg, offset)| { (*reg, *offset) == (Mreg::R10, -40) }),
        "post-allocation MOV RSP alias lost its coordinate: {rows:#x?}"
    );
    assert!(
        rows.iter()
            .any(|(_, _, reg, offset)| { (*reg, *offset) == (Mreg::R11, -32) }),
        "post-allocation LEA RSP alias reused MOV identity: {rows:#x?}"
    );

    let two_hop = function_span(db, "alias_two_hop_lea");
    let two_hop_rows: Vec<_> = db
        .rel_iter::<(Address, Address, Mreg, i64)>("sp_base_alias_at")
        .filter(|(func, _, _, _)| *func == two_hop.0)
        .copied()
        .collect();
    assert!(
        two_hop_rows
            .iter()
            .any(|(_, _, reg, offset)| { (*reg, *offset) == (Mreg::R11, 8) }),
        "LEA-of-copied-SP provenance did not compose: {two_hop_rows:#x?}"
    );
}

fn assert_unknown_sp_indexed_accesses_are_rejected(db: &DecompileDB) {
    let span = function_span(db, "unknown_sp_indexed");
    let accesses: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| {
            if !in_span(*address, span) {
                return None;
            }
            let kind = match inst {
                MachInst::Mstore(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::DX,
                ) if args.as_ref() == &[Mreg::SP, Mreg::R9] => 0,
                MachInst::Mload(
                    MemoryChunk::MInt32,
                    Addressing::Aindexed2scaled(4, 8),
                    args,
                    Mreg::AX,
                ) if args.as_ref() == &[Mreg::SP, Mreg::R9] => 1,
                _ => return None,
            };
            Some((*address, kind))
        })
        .collect();
    assert_eq!(accesses.len(), 2, "missing unknown-SP indexed accesses");

    for (address, _) in accesses {
        assert!(
            db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, access, reason)| {
                    (*func, *access, *reason) == (span.0, address, "unsupported-stack-address")
                }),
            "unknown-SP indexed access lacks a structured rejection at {address:#x}"
        );
        let original = address_bearing_candidates(db, address);
        assert!(
            original.is_empty(),
            "unknown-SP indexed access retained unsound RTL: {original:#x?}"
        );
    }
    assert!(!db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(_, func, _, _)| *func == span.0));
}

fn csharp_home_address(expr: &CsharpminorExpr, slot: Ident, byte_ofs: i64) -> bool {
    let base = |expr: &CsharpminorExpr| {
        matches!(
            expr,
            CsharpminorExpr::Eunop(CminorUnop::Olongofintu, inner)
                if matches!(inner.as_ref(), CsharpminorExpr::Eaddrof(ident) if *ident == slot)
        )
    };
    if byte_ofs == 0 {
        return base(expr);
    }
    matches!(
        expr,
        CsharpminorExpr::Ebinop(CminorBinop::Oaddl, left, right)
            if base(left)
                && matches!(right.as_ref(), CsharpminorExpr::Econst(Constant::Olongconst(ofs)) if *ofs == byte_ofs)
    )
}

fn csharp_expr_has_home_load(
    expr: &CsharpminorExpr,
    chunk: MemoryChunk,
    slot: Ident,
    byte_ofs: i64,
) -> bool {
    match expr {
        CsharpminorExpr::Eload(candidate, address)
            if *candidate == chunk && csharp_home_address(address, slot, byte_ofs) =>
        {
            true
        }
        CsharpminorExpr::Eunop(_, inner) => {
            csharp_expr_has_home_load(inner, chunk, slot, byte_ofs)
        }
        CsharpminorExpr::Ebinop(_, left, right) => {
            csharp_expr_has_home_load(left, chunk, slot, byte_ofs)
                || csharp_expr_has_home_load(right, chunk, slot, byte_ofs)
        }
        CsharpminorExpr::Econdition(condition, if_true, if_false) => {
            csharp_expr_has_home_load(condition, chunk, slot, byte_ofs)
                || csharp_expr_has_home_load(if_true, chunk, slot, byte_ofs)
                || csharp_expr_has_home_load(if_false, chunk, slot, byte_ofs)
        }
        _ => false,
    }
}

fn csharp_stmt_has_home_load(
    stmt: &CsharpminorStmt,
    chunk: MemoryChunk,
    slot: Ident,
    byte_ofs: i64,
) -> bool {
    let has_load = |expr: &CsharpminorExpr| {
        csharp_expr_has_home_load(expr, chunk, slot, byte_ofs)
    };
    match stmt {
        CsharpminorStmt::Sset(_, value)
        | CsharpminorStmt::Sjumptable(value, _)
        | CsharpminorStmt::Sreturn(value) => has_load(value),
        CsharpminorStmt::Sstore(_, address, value) => {
            has_load(address) || has_load(value)
        }
        CsharpminorStmt::Scall(_, _, callee, args)
        | CsharpminorStmt::Stailcall(_, callee, args) => {
            let callee_has_load = match callee {
                either::Either::Left(value) => has_load(value),
                _ => false,
            };
            callee_has_load || args.iter().any(has_load)
        }
        CsharpminorStmt::Scond(_, args, _, _)
        | CsharpminorStmt::Sifthenelse(_, args, _, _) => args.iter().any(has_load),
        CsharpminorStmt::Sseq(statements) => statements
            .iter()
            .any(|statement| csharp_stmt_has_home_load(statement, chunk, slot, byte_ofs)),
        CsharpminorStmt::Sloop(body) => {
            csharp_stmt_has_home_load(body, chunk, slot, byte_ofs)
        }
        _ => false,
    }
}

fn assert_csharp_home_backing(db: &DecompileDB) {
    for name in [
        "home_partial_movzx_high_backing",
        "home_backing_movsx_high",
        "home_backing_signed_cmp_jcc",
        "home_backing_unsigned_cmp_setcc",
        "home_backing_partial_load",
        "home_backing_partial_store",
        "home_backing_add_high",
        "home_backing_test_jcc",
        "home_backing_test_setcc",
        "home_backing_cmov_cmp",
        "home_backing_cmov_test",
        "home_backing_div",
        "home_backing_descriptor",
        "home_backing_cmp_setcc",
    ] {
        let span = function_span(db, name);
        let slot = db
            .rel_iter::<(Address, usize, RTLReg)>("win64_home_storage")
            .find_map(|(func, pos, slot)| {
                (*func == span.0 && *pos == 0).then_some(*slot)
            })
            .unwrap_or_else(|| panic!("{name} lost its backing local before Cshminor"));
        let slot_ident = manifold::decompile::passes::csh_pass::ident_from_reg(slot);
        let accesses: Vec<_> = db
            .rel_iter::<(
                Address,
                RTLReg,
                i64,
                usize,
                MemoryChunk,
                bool,
                bool,
                bool,
            )>("win64_home_backing_access")
            .filter_map(|(
                node,
                candidate_slot,
                byte_ofs,
                width,
                chunk,
                read,
                write,
                address,
            )| {
                (in_span(*node, span) && *candidate_slot == slot).then_some((
                    *node,
                    *byte_ofs,
                    *width,
                    *chunk,
                    *read,
                    *write,
                    *address,
                ))
            })
            .collect();
        assert!(!accesses.is_empty(), "{name} lost backing metadata");
        for (node, byte_ofs, width, chunk, read, write, address) in accesses {
            let candidates: Vec<_> = db
                .rel_iter::<(Address, CsharpminorStmt)>("csharp_stmt_candidate")
                .filter_map(|(candidate_node, stmt)| {
                    ((*candidate_node & !SYNTHETIC_NODE_MASK) == node)
                        .then_some(stmt.clone())
                })
                .collect();
            assert!(
                !candidates.is_empty(),
                "{name} has no Cshminor candidate at backing access {node:#x}"
            );
            if address {
                assert_eq!(byte_ofs, 0, "descriptor fixture should take &cell");
                assert!(
                    candidates.iter().any(|stmt| matches!(
                        stmt,
                        CsharpminorStmt::Sset(_, CsharpminorExpr::Eaddrof(ident))
                            if *ident == slot_ident
                    )),
                    "{name} did not lower its authenticated LEA to &backing: {candidates:#x?}"
                );
            } else if write {
                assert!(
                    candidates.iter().any(|stmt| {
                        matches!(stmt, CsharpminorStmt::Sstore(candidate, address, _)
                            if *candidate == chunk
                                && csharp_home_address(address, slot_ident, byte_ofs))
                    }),
                    "{name} did not materialize its {width}-byte write at +{byte_ofs}: {candidates:#x?}"
                );
                if name == "home_backing_add_high" && byte_ofs == 2 {
                    assert!(
                        candidates.iter().any(|stmt| {
                            matches!(stmt, CsharpminorStmt::Sstore(_, _, value)
                                if csharp_expr_has_home_load(
                                    value,
                                    chunk,
                                    slot_ident,
                                    byte_ofs,
                                ))
                        }),
                        "{name} lost the read half of its partial RMW: {candidates:#x?}"
                    );
                }
            } else {
                assert!(read, "{name} has a non-address access with no mode");
                assert!(
                    candidates
                        .iter()
                        .any(|stmt| csharp_stmt_has_home_load(
                            stmt,
                            chunk,
                            slot_ident,
                            byte_ofs,
                        )),
                    "{name} did not materialize its {width}-byte read at +{byte_ofs}: {candidates:#x?}"
                );
            }
        }
    }
}

fn assert_final_output_compiles(object: &Path) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("failed to build stack/home pipeline thread pool");
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, false, false));

    assert_csharp_home_backing(&db);

    let tu = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("stack/home pipeline must emit an optimized translation unit");

    // This path uses the dependency scheduler (Asm runs in a parallel stage on
    // x86).  The decoder snapshot must be an Asm output, not an input-only
    // wrapper mutation that the stage merge silently drops before RTL.
    let mixed_use = function_span(&db, "home_escape_numeric_use");
    assert!(
        db.rel_iter::<(Address, Mreg)>("asm_reg_use")
            .any(|(node, reg)| in_span(*node, mixed_use) && *reg == Mreg::R10),
        "scheduled pipeline dropped Asm's immutable decoded-use snapshot"
    );
    assert!(
        db.rel_iter::<(Address, Mreg)>("asm_reg_def")
            .any(|(node, reg)| in_span(*node, mixed_use) && *reg == Mreg::R10),
        "scheduled pipeline dropped Asm's immutable decoded-def snapshot"
    );

    // The aliased home address escapes to an opaque call.  Its initialization
    // and post-call read therefore have memory semantics and must survive RTL
    // copy propagation; otherwise the generated function can incorrectly
    // return the original entry parameter after the callee mutates the slot.
    let escaped = function_span(&db, "home_alias_call_escape");
    let escaped_slot = db
        .rel_iter::<(Address, usize, u64)>("win64_home_storage")
        .find_map(|(func, pos, slot)| (*func == escaped.0 && *pos == 0).then_some(*slot))
        .expect("escaped home fixture lost its canonical slot");
    assert!(
        db.rel_iter::<(Address, u64)>("win64_home_escaped")
            .any(|row| *row == (escaped.0, escaped_slot)),
        "scheduled pipeline lost the escaped-home propagation barrier"
    );
    let optimized_escaped: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst")
        .filter(|(address, _)| in_span(*address, escaped))
        .cloned()
        .collect();
    assert!(
        optimized_escaped.iter().any(|(_, inst)| {
            matches!(inst,
                RTLInst::Iop(Operation::Omove, args, destination)
                    if args.len() == 1 && *destination == escaped_slot)
        }),
        "escaped home initialization was propagated away: {optimized_escaped:#x?}"
    );
    assert!(
        optimized_escaped.iter().any(|(_, inst)| {
            matches!(inst,
                RTLInst::Iop(Operation::Omove, args, _)
                    if args.as_ref() == &[escaped_slot])
        }),
        "escaped home post-call read was propagated away: {optimized_escaped:#x?}"
    );

    let escaped_ident = manifold::decompile::passes::csh_pass::ident_from_reg(escaped_slot);
    let escaped_definition = tu
        .decls
        .iter()
        .find_map(|decl| match decl {
            TopLevelDecl::FuncDef(function)
                if function.name == "home_alias_call_escape"
                    || function.name == "coff_fn_home_alias_call_escape" =>
            {
                Some(function)
            }
            _ => None,
        })
        .expect("final TU lost home_alias_call_escape");
    let escaped_locals: Vec<_> = escaped_definition
        .local_vars
        .iter()
        .filter(|local| local.ty == CType::Int(IntSize::Long, Signedness::Signed))
        .collect();
    assert!(
        !escaped_locals.is_empty(),
        "final emission lost the canonical qword local after variable coalescing: {:#?}",
        escaped_definition.local_vars
    );
    let structured_escaped: Vec<_> = db
        .rel_iter::<(Address, CsharpminorStmt)>("csharp_stmt")
        .filter(|(node, _)| in_span(*node & !((1u64 << 62) | (1u64 << 63)), escaped))
        .cloned()
        .collect();
    let escaped_candidates: Vec<_> = db
        .rel_iter::<(Address, CsharpminorStmt)>("csharp_stmt_candidate")
        .filter(|(node, _)| in_span(*node & !SYNTHETIC_NODE_MASK, escaped))
        .cloned()
        .collect();
    assert!(
        structured_escaped.iter().any(|(_, stmt)| {
            matches!(stmt,
            CsharpminorStmt::Sset(destination, CsharpminorExpr::Evar(_))
                if *destination == escaped_slot)
        }),
        "Structuring propagated away escaped home initialization: {structured_escaped:#x?}"
    );
    assert!(
        structured_escaped.iter().any(|(_, stmt)| {
            matches!(stmt,
            CsharpminorStmt::Sset(_, CsharpminorExpr::Evar(source))
                if *source == escaped_slot)
        }),
        "Structuring propagated escaped home read across call: structured={structured_escaped:#x?}, candidates={escaped_candidates:#x?}"
    );
    assert!(
        structured_escaped.iter().any(|(_, stmt)| {
            matches!(stmt,
            CsharpminorStmt::Scall(_, _, _, args)
                if args.iter().any(|arg| matches!(arg,
                    CsharpminorExpr::Eaddrof(ident) if *ident == escaped_ident)))
        }),
        "canonical home address did not lower to &slot: {structured_escaped:#x?}"
    );

    let text = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        tu,
        BinaryFormat::Coff,
    );
    for name in ["home_backing_movsx_high", "home_backing_signed_cmp_jcc"] {
        let body = printed_function_definition(&text, name)
            .unwrap_or_else(|| panic!("final optimized TU lost definition of {name}"));
        assert!(
            body.contains("short *") && !body.contains("unsigned short *"),
            "{name} final AST lost its signed 16-bit backing load:\n{body}"
        );
    }
    for name in [
        "home_backing_unsigned_cmp_setcc",
        "home_backing_partial_load",
    ] {
        let body = printed_function_definition(&text, name)
            .unwrap_or_else(|| panic!("final optimized TU lost definition of {name}"));
        assert!(
            body.contains("unsigned short *"),
            "{name} final AST lost its unsigned 16-bit backing load:\n{body}"
        );
    }
    for function in [
        "home_escape_mutated",
        "home_reassigned_cmp",
        "home_reassigned_cmp_reg",
        "home_reassigned_cmp_setcc",
        "home_reassigned_cmp_reg_setcc",
        "home_reassigned_cmp_loop",
        "home_cmp_after_alias_clobber",
        "home_reassigned_movsxd",
        "home_reassigned_movsx",
        "home_reassigned_movzx",
        "home_partial_movzx_low",
        "home_partial_movzx_high_backing",
        "home_backing_movsx_high",
        "home_backing_signed_cmp_jcc",
        "home_backing_unsigned_cmp_setcc",
        "home_backing_partial_load",
        "home_backing_partial_store",
        "home_backing_add_high",
        "home_backing_test_jcc",
        "home_backing_test_setcc",
        "home_backing_cmov_cmp",
        "home_backing_cmov_test",
        "home_backing_div",
        "home_backing_descriptor",
        "home_backing_cmp_setcc",
        "home_partial_reload",
        "home_reassigned_add",
        "home_import_tailcall",
        "home_typed_slot_tailcall",
        "home_epilogue_register_tailcall",
        "home_adjacent_epilogue_register_tailcall",
        "fifth_inc",
        "outgoing_reuses_home_coordinate",
        "sp_indexed_fused_arith",
        "sp_indexed_fused_add_collision",
        "sp_indexed_fused_field",
        "sp_indexed_fused_misaligned_field",
        "sp_indexed_fused_param_rmw",
        "sp_indexed_fused_gap",
    ] {
        let coff_name = format!("coff_fn_{function}");
        assert!(
            tu.decls.iter().any(|decl| {
                matches!(decl, TopLevelDecl::FuncDef(f)
                    if f.name == function || f.name == coff_name)
            }),
            "final translation unit lost the definition of {function}:\n{text}"
        );
    }

    // Clight deliberately emits C's unspecified-parameter declaration for an
    // untyped COFF import.  The downstream C++ adapter supplies the recovered
    // prototype; relax this fixture-only declaration for the C++ syntax smoke.
    let syntax_text = text
        .replace(
            "int coff_ext_home_tailcall();",
            "int coff_ext_home_tailcall(...);",
        )
        .replace(
            "int coff_ext_typed_slot_tailcall();",
            "int coff_ext_typed_slot_tailcall(...);",
        );
    let output = object.with_extension("generated.cpp");
    std::fs::write(&output, &syntax_text).expect("failed to write stack/home generated C++");
    let compiled = Command::new("clang++")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            "c++",
            "-fsyntax-only",
        ])
        .arg(&output)
        .output()
        .expect("failed to run clang++ over stack/home output");
    assert!(
        compiled.status.success(),
        "stack/home final TU does not compile as C++:\n{}\n{text}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn coff_stack_and_home_relations_preserve_values_and_abi_ordinals() {
    if !command_exists("clang") || !command_exists("clang++") {
        eprintln!("skipping stack/home relation test: clang or clang++ unavailable");
        return;
    }
    let object = fixture().to_path_buf();
    std::thread::Builder::new()
        .name("stack-home-relations".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let mut db = load_rtl_relations(&object);
            assert_sp_indexed_relations(&db);
            assert_incomplete_indexed_stack_lowering_is_atomic(&db);
            assert_ambiguous_fused_indexed_lowering_is_atomic(&db);
            assert_sp_indexed_fused_arithmetic(&db);
            assert_indexed_entry_and_loop_collision_matrix(&db);
            assert_home_relations(&db);
            assert_home_pointer_reload_beats_shadow_address(&db);
            assert_stack_param_and_rmw_relations(&db);
            assert_home_store_is_not_outgoing(&db);
            assert_unsafe_home_cells_remain_storage(&db);
            assert_canonical_unsafe_home_storage(&db);
            assert_canonical_home_backing(&db);
            assert_postsub_raw_offset_does_not_alias_home(&db);
            assert_postsub_escaped_offset_keeps_ordinary_origin(&db);
            assert_indexed_stack_cells_use_normalized_coordinates(&db);
            assert_stack_address_destinations_are_not_storage(&db);
            assert_unknown_sp_and_high_offsets_do_not_infer_params(&db);
            assert_incoming_stack_rmw_is_initialized_and_survives(&db);
            assert_outgoing_home_coordinate_is_not_suppressed(&db);
            assert_resolved_import_tailcall_keeps_frame_proofs(&db);
            assert_indirect_tailcall_proofs_are_structural(&db);
            assert_bp_postprologue_and_xmm_forms(&db);
            assert_bp_provenance_preserves_pointer_memory(&db);
            assert_bp_shortcuts_and_narrow_bases_stay_pointer_memory(&db);
            assert_home_safety_vetoes(&db);
            assert_alias_coordinate_provenance(&db);
            assert_unknown_sp_indexed_accesses_are_rejected(&db);
            RTLOptimizePass.run(&mut db);
            assert_optimized_canonical_homes(&db);
            assert_optimized_home_backing(&db);
            TypePass.run(&mut db);
            assert_post_type_canonical_homes(&db);
            drop(db);
            assert_final_output_compiles(&object);
        })
        .expect("failed to spawn stack/home relation test thread")
        .join()
        .expect("stack/home relation test thread panicked");
}
