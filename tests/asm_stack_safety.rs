use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use manifold::decompile::analysis::canary_vla_pass::CanaryVlaPass;
use manifold::decompile::analysis::stack_pass::StackAnalysisPass;
use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::asm_pass::AsmPass;
use manifold::decompile::passes::linear_pass::LinearPass;
use manifold::decompile::passes::mach_pass::MachPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::decompile::passes::rtl_pass::RTLPass;
use manifold::mreg::Mreg;
use manifold::x86::asm::TestCond;
use manifold::x86::op::{Addressing, Operation};
use manifold::x86::types::{Addrmode, Address, MachInst, MemoryChunk, RTLInst, Symbol};

const SYNTH1: Address = 1u64 << 62;
const SYNTHETIC_NODE_MASK: Address = (1u64 << 62) | (1u64 << 63);

type FloatLoadOp = (
    Address,
    Operation,
    MemoryChunk,
    Addressing,
    Arc<Vec<Mreg>>,
    Mreg,
    bool,
);

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") {
        eprintln!("skipping asm stack-safety test: clang unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_asm_stack_safety_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    std::fs::write(
        &source,
        r#"
        .text

        .globl rsp_after_pop_imul
        .def rsp_after_pop_imul; .scl 2; .type 32; .endef
rsp_after_pop_imul:
        popq %rsp
        imull $7, 40(%rsp), %eax
        retq

        .globl rsp_after_mov_cvtsi_sd
        .def rsp_after_mov_cvtsi_sd; .scl 2; .type 32; .endef
rsp_after_mov_cvtsi_sd:
        movq %rcx, %rsp
        cvtsi2sdl 40(%rsp), %xmm0
        retq

        .globl rsp_after_and_sse
        .def rsp_after_and_sse; .scl 2; .type 32; .endef
rsp_after_and_sse:
        andq $-16, %rsp
        addsd 40(%rsp), %xmm0
        retq

        .globl rsp_dynamic_cvtsi_ss
        .def rsp_dynamic_cvtsi_ss; .scl 2; .type 32; .endef
rsp_dynamic_cvtsi_ss:
        subq %rax, %rsp
        cvtsi2ssl 40(%rsp), %xmm0
        retq

        .globl rsp_branch_imul
        .def rsp_branch_imul; .scl 2; .type 32; .endef
rsp_branch_imul:
        testl %ecx, %ecx
        je .Lrsp_branch_imul_join
        subq $16, %rsp
.Lrsp_branch_imul_join:
        imull $9, 40(%rsp), %eax
        retq

        .globl rsp_after_mov_load
        .def rsp_after_mov_load; .scl 2; .type 32; .endef
rsp_after_mov_load:
        movq %rcx, %rsp
        movl 8(%rsp), %eax
        retq

        .globl rsp_after_mov_store
        .def rsp_after_mov_store; .scl 2; .type 32; .endef
rsp_after_mov_store:
        movq %rcx, %rsp
        movl %edx, 8(%rsp)
        retq

        .globl rsp_after_mov_two_accesses
        .def rsp_after_mov_two_accesses; .scl 2; .type 32; .endef
rsp_after_mov_two_accesses:
        movq %rcx, %rsp
        movl 8(%rsp), %eax
        addl 12(%rsp), %eax
        retq

        .globl rsp_after_mov_rmw_chain
        .def rsp_after_mov_rmw_chain; .scl 2; .type 32; .endef
rsp_after_mov_rmw_chain:
        movq %rcx, %rsp
        addl $1, 8(%rsp)
        xorl %eax, %eax
        retq

        .globl rsp_equal_join_sse
        .def rsp_equal_join_sse; .scl 2; .type 32; .endef
rsp_equal_join_sse:
        testl %ecx, %ecx
        je .Lrsp_equal_join_sse_join
        subq $16, %rsp
        addq $16, %rsp
.Lrsp_equal_join_sse_join:
        mulsd 40(%rsp), %xmm0
        retq

        .globl rbp_conditional_imul
        .def rbp_conditional_imul; .scl 2; .type 32; .endef
rbp_conditional_imul:
        testl %ecx, %ecx
        je .Lrbp_conditional_imul_join
        movq %rsp, %rbp
.Lrbp_conditional_imul_join:
        imull $11, 40(%rbp), %eax
        retq

        .globl rbp_clobbered_cvtsi_sd
        .def rbp_clobbered_cvtsi_sd; .scl 2; .type 32; .endef
rbp_clobbered_cvtsi_sd:
        movq %rsp, %rbp
        testl %ecx, %ecx
        je .Lrbp_clobbered_cvtsi_sd_join
        movq %r8, %rbp
.Lrbp_clobbered_cvtsi_sd_join:
        cvtsi2sdl 40(%rbp), %xmm0
        retq

        .globl rbp_conditional_sse
        .def rbp_conditional_sse; .scl 2; .type 32; .endef
rbp_conditional_sse:
        testl %ecx, %ecx
        je .Lrbp_conditional_sse_join
        movq %rsp, %rbp
.Lrbp_conditional_sse_join:
        mulsd 40(%rbp), %xmm0
        retq

        .globl rsp_safe_cvtsi
        .def rsp_safe_cvtsi; .scl 2; .type 32; .endef
rsp_safe_cvtsi:
        subq $16, %rsp
        movl %ecx, 4(%rsp)
        cvtsi2ssl 4(%rsp), %xmm0
        addq $16, %rsp
        retq

        .globl rsp_safe_sse
        .def rsp_safe_sse; .scl 2; .type 32; .endef
rsp_safe_sse:
        subq $16, %rsp
        movsd %xmm1, 8(%rsp)
        addsd 8(%rsp), %xmm0
        addq $16, %rsp
        retq

        .globl rbp_safe_imul
        .def rbp_safe_imul; .scl 2; .type 32; .endef
rbp_safe_imul:
        pushq %rbp
        movq %rsp, %rbp
        movl %ecx, -4(%rbp)
        imull $13, -4(%rbp), %eax
        popq %rbp
        retq

        .globl rbp_scratch_load
        .def rbp_scratch_load; .scl 2; .type 32; .endef
rbp_scratch_load:
        movq %rcx, %rbp
        movl 8(%rbp), %eax
        retq

        .globl rbp_scratch_store
        .def rbp_scratch_store; .scl 2; .type 32; .endef
rbp_scratch_store:
        movq %rcx, %rbp
        movl %edx, 8(%rbp)
        retq

        .globl rsp_unknown_cmp_jcc
        .def rsp_unknown_cmp_jcc; .scl 2; .type 32; .endef
rsp_unknown_cmp_jcc:
        movq %rcx, %rsp
        cmpl $1, 8(%rsp)
        je .Lrsp_unknown_cmp_jcc_true
        xorl %eax, %eax
        retq
.Lrsp_unknown_cmp_jcc_true:
        movl $1, %eax
        retq

        .globl rsp_unknown_cmp_setcc
        .def rsp_unknown_cmp_setcc; .scl 2; .type 32; .endef
rsp_unknown_cmp_setcc:
        movq %rcx, %rsp
        cmpl $1, 8(%rsp)
        sete %al
        movzbl %al, %eax
        retq

        .globl rbp_scratch_cmp_jcc
        .def rbp_scratch_cmp_jcc; .scl 2; .type 32; .endef
rbp_scratch_cmp_jcc:
        movq %rcx, %rbp
        cmpl $1, 8(%rbp)
        je .Lrbp_scratch_cmp_jcc_true
        xorl %eax, %eax
        retq
.Lrbp_scratch_cmp_jcc_true:
        movl $1, %eax
        retq

        .globl addr32_frame_indexed_cmp_jcc
        .def addr32_frame_indexed_cmp_jcc; .scl 2; .type 32; .endef
addr32_frame_indexed_cmp_jcc:
        movq %rsp, %rbp
        # cmp dword ptr [ebp + ecx*4 + 8], 1
        .byte 0x67, 0x83, 0x7c, 0x8d, 0x08, 0x01
        je .Laddr32_frame_indexed_cmp_jcc_true
        xorl %eax, %eax
        retq
.Laddr32_frame_indexed_cmp_jcc_true:
        movl $1, %eax
        retq

        .globl rbp_from_invalid_rsp
        .def rbp_from_invalid_rsp; .scl 2; .type 32; .endef
rbp_from_invalid_rsp:
        movq %rcx, %rsp
        movq %rsp, %rbp
        movl 8(%rbp), %eax
        retq

        .globl rsp_loop_drift
        .def rsp_loop_drift; .scl 2; .type 32; .endef
rsp_loop_drift:
.Lrsp_loop_drift:
        subq $8, %rsp
        decl %ecx
        jne .Lrsp_loop_drift
        movl 40(%rsp), %eax
        retq

        .globl rsp_pushw_balanced
        .def rsp_pushw_balanced; .scl 2; .type 32; .endef
rsp_pushw_balanced:
        pushw %ax
        popw %bx
        movl 8(%rsp), %eax
        retq

        .globl rsp_popw_sp
        .def rsp_popw_sp; .scl 2; .type 32; .endef
rsp_popw_sp:
        popw %sp
        movl 8(%rsp), %eax
        retq

        .globl rsp_indexed_imul
        .def rsp_indexed_imul; .scl 2; .type 32; .endef
rsp_indexed_imul:
        imull $7, 40(%rsp,%rcx,4), %eax
        retq

        .globl rsp_indexed_cvtsi
        .def rsp_indexed_cvtsi; .scl 2; .type 32; .endef
rsp_indexed_cvtsi:
        cvtsi2sdl 40(%rsp,%rcx,4), %xmm0
        retq

        .globl rsp_indexed_sse
        .def rsp_indexed_sse; .scl 2; .type 32; .endef
rsp_indexed_sse:
        addsd 40(%rsp,%rcx,4), %xmm0
        retq

        .globl rsp_indexed_lea
        .def rsp_indexed_lea; .scl 2; .type 32; .endef
rsp_indexed_lea:
        leaq 40(%rsp,%rcx,4), %rax
        retq

        .globl rbp_scratch_ucomisd
        .def rbp_scratch_ucomisd; .scl 2; .type 32; .endef
rbp_scratch_ucomisd:
        movq %rcx, %rbp
        ucomisd 8(%rbp), %xmm0
        ja .Lrbp_scratch_ucomisd_true
        xorl %eax, %eax
        retq
.Lrbp_scratch_ucomisd_true:
        movl $1, %eax
        retq

        .globl shared_stack_a
        .def shared_stack_a; .scl 2; .type 32; .endef
shared_stack_a:
        jmp .Lshared_stack_tail

        .globl shared_stack_b
        .def shared_stack_b; .scl 2; .type 32; .endef
shared_stack_b:
        jmp .Lshared_stack_tail
.Lshared_stack_tail:
        movl 8(%rsp), %eax
        retq

        .globl unresolved_indirect_stack
        .def unresolved_indirect_stack; .scl 2; .type 32; .endef
unresolved_indirect_stack:
        testl %ecx, %ecx
        je .Lunresolved_indirect_access
        movq %rdx, %rsp
        jmp *%rax
.Lunresolved_indirect_access:
        movl 8(%rsp), %eax
        retq

        .globl unsupported_internal_callee
        .def unsupported_internal_callee; .scl 2; .type 32; .endef
unsupported_internal_callee:
        movq %rcx, %rsp
        movl 8(%rsp), %eax
        retq

        .globl safe_calls_unsupported
        .def safe_calls_unsupported; .scl 2; .type 32; .endef
safe_calls_unsupported:
        subq $40, %rsp
        callq unsupported_internal_callee
        addq $40, %rsp
        retq

        .globl safe_sibling_return
        .def safe_sibling_return; .scl 2; .type 32; .endef
safe_sibling_return:
        leal 1(%rcx), %eax
        retq

        .globl addr32_eax_load
        .def addr32_eax_load; .scl 2; .type 32; .endef
addr32_eax_load:
        movq %rcx, %rax
        # mov edx, dword ptr [eax]
        .byte 0x67, 0x8b, 0x10
        movl %edx, %eax
        retq

        .globl addr32_eax_indexed
        .def addr32_eax_indexed; .scl 2; .type 32; .endef
addr32_eax_indexed:
        movq %rcx, %rax
        movq %rdx, %rcx
        # mov edx, dword ptr [eax + ecx*4 + 0x20]
        .byte 0x67, 0x8b, 0x54, 0x88, 0x20
        movl %edx, %eax
        retq

        .globl addr32_index_only
        .def addr32_index_only; .scl 2; .type 32; .endef
addr32_index_only:
        movq %rcx, %rax
        movq %rdx, %rcx
        # mov edx, dword ptr [ecx*4 + 0x20]
        .byte 0x67, 0x8b, 0x14, 0x8d, 0x20, 0x00, 0x00, 0x00
        movl %edx, %eax
        retq

        .globl addr32_scale1
        .def addr32_scale1; .scl 2; .type 32; .endef
addr32_scale1:
        movq %rcx, %rax
        movq %rdx, %rcx
        # mov edx, dword ptr [eax + ecx + 0x20]
        .byte 0x67, 0x8b, 0x54, 0x08, 0x20
        movl %edx, %eax
        retq

        .globl addr32_rmw_store
        .def addr32_rmw_store; .scl 2; .type 32; .endef
addr32_rmw_store:
        movq %rcx, %rax
        # add dword ptr [eax], 1
        .byte 0x67, 0x83, 0x00, 0x01
        retq

        .globl addr32_lea
        .def addr32_lea; .scl 2; .type 32; .endef
addr32_lea:
        movq %rcx, %rax
        movq %rdx, %rcx
        # lea rdx, [eax + ecx*4 + 0x20]
        .byte 0x67, 0x48, 0x8d, 0x54, 0x88, 0x20
        movq %rdx, %rax
        retq

        .globl addr32_esp_unsupported
        .def addr32_esp_unsupported; .scl 2; .type 32; .endef
addr32_esp_unsupported:
        # mov eax, dword ptr [esp]
        .byte 0x67, 0x8b, 0x04, 0x24
        retq

        .globl addr32_frame_ebp_unsupported
        .def addr32_frame_ebp_unsupported; .scl 2; .type 32; .endef
addr32_frame_ebp_unsupported:
        movq %rsp, %rbp
        # mov eax, dword ptr [ebp + 8]
        .byte 0x67, 0x8b, 0x45, 0x08
        retq

        .globl addr32_scratch_ebp
        .def addr32_scratch_ebp; .scl 2; .type 32; .endef
addr32_scratch_ebp:
        movq %rcx, %rbp
        # mov eax, dword ptr [ebp + 8]
        .byte 0x67, 0x8b, 0x45, 0x08
        retq

        .globl addr32_shared_ebp_frame
        .def addr32_shared_ebp_frame; .scl 2; .type 32; .endef
addr32_shared_ebp_frame:
        movq %rsp, %rbp
        jmp .Laddr32_shared_ebp_tail

        .globl addr32_shared_ebp_scratch
        .def addr32_shared_ebp_scratch; .scl 2; .type 32; .endef
addr32_shared_ebp_scratch:
        movq %rcx, %rbp
        jmp .Laddr32_shared_ebp_tail
        jmp .Laddr32_shared_ebp_tail
.Laddr32_shared_ebp_tail:
        # mov eax, dword ptr [ebp + 8]
        .byte 0x67, 0x8b, 0x45, 0x08
        retq

        .globl addr32_unresolved_ebp
        .def addr32_unresolved_ebp; .scl 2; .type 32; .endef
addr32_unresolved_ebp:
        movq %rcx, %rbp
        testl %edx, %edx
        je .Laddr32_unresolved_ebp_access
        jmp *%rax
.Laddr32_unresolved_ebp_access:
        # mov eax, dword ptr [ebp + 8]
        .byte 0x67, 0x8b, 0x45, 0x08
        retq

        .globl addr32_synthetic_store
        .def addr32_synthetic_store; .scl 2; .type 32; .endef
addr32_synthetic_store:
        movq %rcx, %rax
        # mov dword ptr [eax], 1
        .byte 0x67, 0xc7, 0x00, 0x01, 0x00, 0x00, 0x00
        retq

        .globl addr32_indirect_call
        .def addr32_indirect_call; .scl 2; .type 32; .endef
addr32_indirect_call:
        movq %rcx, %rax
        # call qword ptr [eax]
        .byte 0x67, 0xff, 0x10
        retq

        .globl addr32_unhandled_memory
        .def addr32_unhandled_memory; .scl 2; .type 32; .endef
addr32_unhandled_memory:
        movq %rcx, %rax
        # prefetcht0 byte ptr [eax]
        .byte 0x67, 0x0f, 0x18, 0x08
        retq

        .globl addr64_load_control
        .def addr64_load_control; .scl 2; .type 32; .endef
addr64_load_control:
        movq %rcx, %rax
        movl (%rax), %edx
        movl %edx, %eax
        retq

        .globl addr64_lea_control
        .def addr64_lea_control; .scl 2; .type 32; .endef
addr64_lea_control:
        leaq 32(%rcx,%rdx,4), %rax
        retq

        .globl addr32_safe_sibling
        .def addr32_safe_sibling; .scl 2; .type 32; .endef
addr32_safe_sibling:
        movl $42, %eax
        retq

        .globl rsp_diamond_stress
        .def rsp_diamond_stress; .scl 2; .type 32; .endef
rsp_diamond_stress:
        testl %ecx, %ecx
        je .Ldiamond_0
        subq $1, %rsp
.Ldiamond_0:
        testl %ecx, %ecx
        je .Ldiamond_1
        subq $2, %rsp
.Ldiamond_1:
        testl %ecx, %ecx
        je .Ldiamond_2
        subq $4, %rsp
.Ldiamond_2:
        testl %ecx, %ecx
        je .Ldiamond_3
        subq $8, %rsp
.Ldiamond_3:
        testl %ecx, %ecx
        je .Ldiamond_4
        subq $16, %rsp
.Ldiamond_4:
        testl %ecx, %ecx
        je .Ldiamond_5
        subq $32, %rsp
.Ldiamond_5:
        testl %ecx, %ecx
        je .Ldiamond_6
        subq $64, %rsp
.Ldiamond_6:
        testl %ecx, %ecx
        je .Ldiamond_7
        subq $128, %rsp
.Ldiamond_7:
        testl %ecx, %ecx
        je .Ldiamond_8
        subq $256, %rsp
.Ldiamond_8:
        testl %ecx, %ecx
        je .Ldiamond_9
        subq $512, %rsp
.Ldiamond_9:
        testl %ecx, %ecx
        je .Ldiamond_10
        subq $1024, %rsp
.Ldiamond_10:
        testl %ecx, %ecx
        je .Ldiamond_11
        subq $2048, %rsp
.Ldiamond_11:
        movl 8(%rsp), %eax
        retq
"#,
    )
    .ok()?;

    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .ok()?;
    assert!(status.success(), "asm stack-safety fixture assembly failed");
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
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
        .unwrap_or_else(|| panic!("missing fixture function {name}"))
}

fn in_span(address: Address, span: (Address, Address)) -> bool {
    address >= span.0 && address < span.1
}

fn is_fixture_name(actual: &str, expected: &str) -> bool {
    actual == expected || actual.strip_prefix("coff_fn_") == Some(expected)
}

fn is_unsigned_i32_type(value: &serde_json::Value) -> bool {
    value["tag"] == "Tint" && value["size"] == "I32" && value["sign"] == "Unsigned"
}

fn count_unsigned_i32_casts(value: &serde_json::Value) -> usize {
    let here = usize::from(value["tag"] == "Ecast" && is_unsigned_i32_type(&value["ty"]));
    here + match value {
        serde_json::Value::Array(items) => items.iter().map(count_unsigned_i32_casts).sum(),
        serde_json::Value::Object(fields) => fields.values().map(count_unsigned_i32_casts).sum(),
        _ => 0,
    }
}

fn contains_binop(value: &serde_json::Value, operation: &str) -> bool {
    if value["tag"] == "Ebinop" && value["op"] == operation {
        return true;
    }
    match value {
        serde_json::Value::Array(items) => items.iter().any(|item| contains_binop(item, operation)),
        serde_json::Value::Object(fields) => {
            fields.values().any(|item| contains_binop(item, operation))
        }
        _ => false,
    }
}

fn contains_addr32_modulo_zero_extend(value: &serde_json::Value) -> bool {
    if value["tag"] == "Ecast" && value["ty"]["tag"] == "Tlong" {
        let inner = &value["expr"];
        if count_unsigned_i32_casts(inner) >= 2
            && contains_binop(inner, "Oadd")
            && contains_binop(inner, "Omul")
        {
            return true;
        }
    }
    match value {
        serde_json::Value::Array(items) => items.iter().any(contains_addr32_modulo_zero_extend),
        serde_json::Value::Object(fields) => {
            fields.values().any(contains_addr32_modulo_zero_extend)
        }
        _ => false,
    }
}

fn contains_named_evar(value: &serde_json::Value, name: &str) -> bool {
    if value["tag"] == "Evar" && value["name"] == name {
        return true;
    }
    match value {
        serde_json::Value::Array(items) => items.iter().any(|item| contains_named_evar(item, name)),
        serde_json::Value::Object(fields) => {
            fields.values().any(|item| contains_named_evar(item, name))
        }
        _ => false,
    }
}

fn parse_hex_address(value: &serde_json::Value) -> u64 {
    let text = value.as_str().expect("hex address is a string");
    u64::from_str_radix(text.strip_prefix("0x").expect("hex address prefix"), 16)
        .expect("valid hex address")
}

fn rtl_candidates(db: &DecompileDB, address: Address) -> Vec<RTLInst> {
    db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter_map(|(row_address, inst)| (*row_address == address).then_some(inst.clone()))
        .collect()
}

fn cmp_candidate_uses_temp(inst: &RTLInst, temp: u64) -> bool {
    match inst {
        RTLInst::Iop(_, args, _) | RTLInst::Iload(_, _, args, _)
        | RTLInst::Icond(_, args, _, _) => args.contains(&temp),
        RTLInst::Istore(_, _, args, source) => args.contains(&temp) || *source == temp,
        RTLInst::Icall(_, callee, args, _, _) | RTLInst::Itailcall(_, callee, args) => {
            args.contains(&temp)
                || matches!(callee, either::Either::Left(value) if *value == temp)
        }
        RTLInst::Ijumptable(value, _) | RTLInst::Ireturn(value) => *value == temp,
        _ => false,
    }
}

fn assert_no_address_bearing_candidate(db: &DecompileDB, context: &str, access: Address) {
    let is_lea = db
        .rel_iter::<InstructionRow>("instruction")
        .any(|(address, _, _, mnemonic, ..)| *address == access && *mnemonic == "LEA");
    let leaked: Vec<(Address, RTLInst)> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, inst)| {
            (*node == access || (*node & !SYNTHETIC_NODE_MASK) == access)
                && matches!(
                    inst,
                    RTLInst::Iload(..)
                        | RTLInst::Istore(..)
                        | RTLInst::Iop(Operation::Olea(_) | Operation::Oleal(_), _, _)
                )
                || ((*node == access || (*node & !SYNTHETIC_NODE_MASK) == access)
                    && is_lea
                    && matches!(inst, RTLInst::Iop(..)))
        })
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert!(
        leaked.is_empty(),
        "{context} retained address-bearing RTL at 0x{access:x}: {leaked:#x?}"
    );

    let rooted_candidates: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access)
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert!(
        rooted_candidates.len() == 1
            && rooted_candidates[0].0 == access
            && rooted_candidates[0].1 == RTLInst::Inop,
        "{context} did not collapse rejected RTL atomically at 0x{access:x}: \
         {rooted_candidates:#x?}"
    );

    let stale_members: Vec<_> = db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .filter(|(node, _)| *node != access && (*node & !SYNTHETIC_NODE_MASK) == access)
        .copied()
        .collect();
    assert!(
        stale_members.is_empty(),
        "{context} retained synthetic function membership at 0x{access:x}: \
         {stale_members:#x?}"
    );

    let stale_edges: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .filter(|(source, destination)| {
            (*source != access && (*source & !SYNTHETIC_NODE_MASK) == access)
                || (*destination != access
                    && (*destination & !SYNTHETIC_NODE_MASK) == access)
        })
        .copied()
        .collect();
    assert!(
        stale_edges.is_empty(),
        "{context} retained synthetic CFG edges at 0x{access:x}: {stale_edges:#x?}"
    );
    assert!(
        !db.rel_iter::<(Address, Address)>("rtl_edge_negated")
            .any(|(source, destination)| {
                (*source != access && (*source & !SYNTHETIC_NODE_MASK) == access)
                    || (*destination != access
                        && (*destination & !SYNTHETIC_NODE_MASK) == access)
            }),
        "{context} retained a negated synthetic edge at 0x{access:x}"
    );
    assert!(
        !db.rel_iter::<(Address, Address)>("sp_indexed_fused_member")
            .any(|(node, _)| (*node & !SYNTHETIC_NODE_MASK) == access),
        "{context} retained fused synthetic membership at 0x{access:x}"
    );

    assert!(
        !db.rel_iter::<(Address, u64, MemoryChunk, Addressing, Arc<Vec<u64>>)>(
            "call_through_memory_load"
        )
        .any(|(node, ..)| { *node == access || (*node & !SYNTHETIC_NODE_MASK) == access }),
        "{context} retained a memory-indirect call load at 0x{access:x}"
    );
    assert!(
        !db.rel_iter::<(Address, u64)>("op_produces_ptr")
            .any(|(node, _)| { *node == access || (*node & !SYNTHETIC_NODE_MASK) == access }),
        "{context} retained pointer evidence at 0x{access:x}"
    );
    assert!(
        !db.rel_iter::<(Address,)>("synth_only_addr")
            .any(|(node,)| (*node & !SYNTHETIC_NODE_MASK) == access),
        "{context} retained synth-only classification at 0x{access:x}"
    );

    let diagnostic_owners: HashSet<Address> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .filter_map(|(owner, root, _)| (*root == access).then_some(*owner))
        .collect();
    for (root, consumer, temp, owner) in db
        .rel_iter::<(Address, Address, u64, Address)>("cmp_memory_temp_consumer")
        .filter(|(root, _, _, owner)| {
            *root == access && diagnostic_owners.contains(owner)
        })
    {
        let leaked: Vec<_> = db
            .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
            .filter(|(node, inst)| {
                *node == *consumer && cmp_candidate_uses_temp(inst, *temp)
            })
            .map(|(node, inst)| (*node, inst.clone()))
            .collect();
        assert!(
            leaked.is_empty(),
            "{context} retained CMP temp 0x{temp:x} from rejected root \
             0x{root:x} at consumer 0x{consumer:x} for owner 0x{owner:x}: \
             {leaked:#x?}"
        );
    }
}

fn assert_rejected_rmw_chain_is_atomic(db: &DecompileDB) {
    let name = "rsp_after_mov_rmw_chain";
    let access = instruction_address(db, name, "ADD");
    assert!(
        db.rel_iter::<(Address, Operation, MemoryChunk, Mreg, i64)>("arith_store_imm")
            .any(|(address, _, _, _, _)| *address == access),
        "{name} did not exercise the load/op/store chain generator"
    );
    assert_stack_unsupported(db, name, access);

    let raw_exits: Vec<_> = db
        .rel_iter::<(Address, Address)>("next")
        .filter(|(source, _)| *source == access)
        .copied()
        .collect();
    let restored_exits: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .filter(|(source, destination)| {
            *source == access && *destination == (*destination & !SYNTHETIC_NODE_MASK)
        })
        .copied()
        .collect();
    assert!(
        !restored_exits.is_empty(),
        "{name} did not bridge the rejected chain to an external successor"
    );
    for edge in raw_exits {
        assert!(
            db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
                .any(|candidate| *candidate == edge),
            "{name} did not restore raw outgoing edge {edge:#x?}"
        );
        assert!(
            !db.rel_iter::<(Address, Address)>("rtl_edge_negated")
                .any(|negated| *negated == edge),
            "{name} left restored outgoing edge negated: {edge:#x?}"
        );
    }

    let raw_incoming: Vec<_> = db
        .rel_iter::<(Address, Address)>("next")
        .filter(|(_, destination)| *destination == access)
        .copied()
        .collect();
    let restored_incoming: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .filter(|(source, destination)| {
            *destination == access && *source == (*source & !SYNTHETIC_NODE_MASK)
        })
        .copied()
        .collect();
    let all_incoming: Vec<_> = db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .filter(|(_, destination)| *destination == access)
        .copied()
        .collect();
    let raw_predecessor_candidates: Vec<_> = db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .filter(|(node, _)| raw_incoming.iter().any(|(source, _)| source == node))
        .map(|(node, inst)| (*node, inst.clone()))
        .collect();
    assert!(
        !restored_incoming.is_empty(),
        "{name} did not reconnect an external predecessor to the replacement Inop: \
         raw={raw_incoming:#x?}, all={all_incoming:#x?}, \
         predecessor_candidates={raw_predecessor_candidates:#x?}"
    );
    for edge in raw_incoming {
        assert!(
            db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
                .any(|candidate| *candidate == edge),
            "{name} did not preserve raw incoming edge {edge:#x?}"
        );
        assert!(
            !db.rel_iter::<(Address, Address)>("rtl_edge_negated")
                .any(|negated| *negated == edge),
            "{name} left restored incoming edge negated: {edge:#x?}"
        );
    }
}

fn assert_all_structured_unsupported_sites_are_filtered(db: &DecompileDB) {
    let mut sites: Vec<Address> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .map(|(_, access, _)| *access)
        .collect();
    sites.sort_unstable();
    sites.dedup();
    for access in sites {
        assert_no_address_bearing_candidate(db, "structured unsupported site", access);
    }
}

#[derive(Clone)]
struct UnsafeCase {
    name: &'static str,
    mnemonic: &'static str,
    op: Operation,
    chunk: MemoryChunk,
    base: Mreg,
    disp: i64,
    unary: bool,
}

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

fn instruction_address(db: &DecompileDB, name: &str, mnemonic: &str) -> Address {
    let span = function_span(db, name);
    db.rel_iter::<InstructionRow>("instruction")
        .find_map(|(address, _, _, row_mnemonic, ..)| {
            (in_span(*address, span) && *row_mnemonic == mnemonic).then_some(*address)
        })
        .unwrap_or_else(|| panic!("missing {mnemonic} instruction in {name}"))
}

fn memory_access_address(db: &DecompileDB, name: &str, base: &'static str, disp: i64) -> Address {
    let operands: HashSet<Symbol> = db
        .rel_iter::<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )>("op_indirect")
        .filter_map(|(operand, _, row_base, _, _, row_disp, _)| {
            (*row_base == base && *row_disp == disp).then_some(*operand)
        })
        .collect();
    let span = function_span(db, name);
    db.rel_iter::<InstructionRow>("instruction")
        .find_map(|(address, _, _, _, op1, op2, op3, op4, _, _)| {
            (in_span(*address, span)
                && [*op1, *op2, *op3, *op4]
                    .iter()
                    .any(|operand| operands.contains(operand)))
            .then_some(*address)
        })
        .unwrap_or_else(|| panic!("missing [{base}{disp:+}] access in {name}"))
}

fn assert_stack_unsupported(db: &DecompileDB, name: &str, access: Address) {
    let span = function_span(db, name);
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, row_access, reason)| {
                (*func, *row_access, *reason) == (span.0, access, "unsupported-stack-address")
            }),
        "{name} did not retain its structured unsupported site at 0x{access:x}"
    );
    assert_no_address_bearing_candidate(db, name, access);
}

fn assert_addr32_unsupported(db: &DecompileDB, name: &str, access: Address) {
    let span = function_span(db, name);
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, row_access, reason)| {
                (*func, *row_access, *reason) == (span.0, access, "unsupported-addr32-address")
            }),
        "{name} did not retain its addr32 unsupported site at 0x{access:x}"
    );
    assert_no_address_bearing_candidate(db, name, access);
}

fn cmp_temp_provenance(
    db: &DecompileDB,
    name: &str,
    consumer_mnemonic: &str,
) -> (Address, Address, Address, Vec<u64>) {
    let span = function_span(db, name);
    let root = instruction_address(db, name, "CMP");
    let consumer = instruction_address(db, name, consumer_mnemonic);
    let mut temps: Vec<u64> = db
        .rel_iter::<(Address, Address, u64, Address)>("cmp_memory_temp_consumer")
        .filter_map(|(row_root, row_consumer, temp, owner)| {
            (*row_root == root && *row_consumer == consumer && *owner == span.0)
                .then_some(*temp)
        })
        .collect();
    temps.sort_unstable();
    temps.dedup();
    assert!(
        !temps.is_empty(),
        "{name} lost exact CMP root/consumer/temp/owner provenance"
    );
    (span.0, root, consumer, temps)
}

fn jcc_taken_target(db: &DecompileDB, branch: Address) -> Address {
    let symbol = db
        .rel_iter::<(Address, TestCond, Symbol)>("pjcc")
        .find_map(|(address, _, symbol)| (*address == branch).then_some(*symbol))
        .unwrap_or_else(|| panic!("missing JCC relation at 0x{branch:x}"));
    db.rel_iter::<(Symbol, Address)>("symbol_resolved_addr")
        .find_map(|(row_symbol, target)| (*row_symbol == symbol).then_some(*target))
        .unwrap_or_else(|| panic!("unresolved JCC target at 0x{branch:x}"))
}

fn assert_cmp_temp_definitions_are_total(db: &DecompileDB) {
    for (root, consumer, temp, owner) in
        db.rel_iter::<(Address, Address, u64, Address)>("cmp_memory_temp_consumer")
    {
        let structured_unsupported = db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(row_owner, row_root, _)| (*row_owner, *row_root) == (*owner, *root));
        let defining_load = rtl_candidates(db, *root).iter().any(|inst| {
            matches!(inst, RTLInst::Iload(_, _, _, destination) if *destination == *temp)
        });
        let consuming_candidate = rtl_candidates(db, *consumer)
            .iter()
            .any(|inst| cmp_candidate_uses_temp(inst, *temp));

        if structured_unsupported {
            assert!(
                !consuming_candidate,
                "rejected CMP root 0x{root:x} retained temporary 0x{temp:x} at \
                 consumer 0x{consumer:x} for owner 0x{owner:x}"
            );
        } else {
            assert!(
                defining_load,
                "safe CMP consumer 0x{consumer:x} reads temporary 0x{temp:x} \
                 without a defining Iload at root 0x{root:x} for owner 0x{owner:x}: \
                 root={:#x?}, consumer={:#x?}",
                rtl_candidates(db, *root),
                rtl_candidates(db, *consumer),
            );
            assert!(
                consuming_candidate,
                "safe CMP provenance lost its consumer at 0x{consumer:x} for \
                 temporary 0x{temp:x}, root 0x{root:x}, owner 0x{owner:x}"
            );
        }
    }
}

fn assert_cmp_cross_node_safety(db: &DecompileDB) {
    assert_cmp_temp_definitions_are_total(db);

    for (name, reason) in [
        ("rsp_unknown_cmp_jcc", "unsupported-stack-address"),
        (
            "addr32_frame_indexed_cmp_jcc",
            "unsupported-addr32-address",
        ),
    ] {
        let (owner, root, consumer, temps) = cmp_temp_provenance(db, name, "JE");
        if reason == "unsupported-stack-address" {
            assert_stack_unsupported(db, name, root);
        } else {
            assert_addr32_unsupported(db, name, root);
        }
        assert!(db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|row| *row == (owner, root, reason)));
        for temp in temps {
            assert!(
                !rtl_candidates(db, consumer)
                    .iter()
                    .any(|inst| cmp_candidate_uses_temp(inst, temp)),
                "{name} retained rejected CMP temp 0x{temp:x} at 0x{consumer:x}"
            );
        }
        let taken = jcc_taken_target(db, consumer);
        assert!(
            !db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
                .any(|edge| *edge == (consumer, taken)),
            "{name} retained the rejected CMP condition's taken edge"
        );
        let fallthrough = db
            .rel_iter::<(Address, Address)>("next")
            .find_map(|(source, destination)| {
                (*source == consumer).then_some(*destination)
            })
            .expect("JCC fallthrough");
        assert!(db
            .rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .any(|edge| *edge == (root, consumer)));
        assert!(db
            .rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .any(|edge| *edge == (consumer, fallthrough)));
    }

    let (owner, root, consumer, temps) =
        cmp_temp_provenance(db, "rsp_unknown_cmp_setcc", "SETE");
    assert_stack_unsupported(db, "rsp_unknown_cmp_setcc", root);
    assert!(db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|row| *row == (owner, root, "unsupported-stack-address")));
    for temp in temps {
        assert!(!rtl_candidates(db, consumer)
            .iter()
            .any(|inst| cmp_candidate_uses_temp(inst, temp)));
    }
    let setcc_fallthrough = db
        .rel_iter::<(Address, Address)>("next")
        .find_map(|(source, destination)| (*source == consumer).then_some(*destination))
        .expect("SETcc fallthrough");
    assert!(
        db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .any(|edge| *edge == (root, consumer)),
        "SETcc root was not reconnected: outgoing={:#x?}, semantic={:#x?}, \
         decoded={:#x?}, consumer_candidates={:#x?}, owners={:#x?}, \
         negated={:#x?}",
        db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
            .filter(|(source, _)| *source == root)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Address)>("rtl_next")
            .filter(|(source, _)| *source == root)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Address)>("next")
            .filter(|(source, _)| *source == root)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
            .filter(|(node, _)| *node == consumer)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Address)>("instr_in_function")
            .filter(|(node, _)| *node == root || *node == consumer)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Address)>("rtl_edge_negated")
            .filter(|(source, _)| *source == root)
            .collect::<Vec<_>>()
    );
    assert!(db
        .rel_iter::<(Address, Address)>("rtl_succ_candidate")
        .any(|edge| *edge == (consumer, setcc_fallthrough)));

    let safe_name = "rbp_scratch_cmp_jcc";
    let (owner, root, consumer, temps) = cmp_temp_provenance(db, safe_name, "JE");
    assert!(!db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(row_owner, _, _)| *row_owner == owner));
    assert!(
        rtl_candidates(db, root)
            .iter()
            .any(|inst| matches!(inst, RTLInst::Iload(..))),
        "safe scratch-RBP CMP lost its root load: candidates={:#x?}, \
         mreg_args={:#x?}, rtl_args={:#x?}, bp_xtl={:#x?}, reaching={:#x?}",
        rtl_candidates(db, root),
        db.rel_iter::<(Address, usize, Mreg)>("temp_cmp_mreg_args")
            .filter(|(node, _, _)| *node == root)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, usize, u64)>("temp_cmp_args_rtl")
            .filter(|(node, _, _)| *node == root)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Mreg, u64)>("reg_xtl")
            .filter(|(node, mreg, _)| *node == root && *mreg == Mreg::BP)
            .collect::<Vec<_>>(),
        db.rel_iter::<(Address, Mreg, Address)>("reg_def_used")
            .filter(|(_, mreg, usage)| *usage == root && *mreg == Mreg::BP)
            .collect::<Vec<_>>()
    );
    for temp in temps {
        assert!(
            rtl_candidates(db, consumer)
                .iter()
                .any(|inst| cmp_candidate_uses_temp(inst, temp)),
            "safe scratch-RBP CMP lost temp 0x{temp:x}"
        );
    }
    let safe_taken = jcc_taken_target(db, consumer);
    let safe_fallthrough = db
        .rel_iter::<(Address, Address)>("next")
        .find_map(|(source, destination)| (*source == consumer).then_some(*destination))
        .expect("safe JCC fallthrough");
    for edge in [(consumer, safe_taken), (consumer, safe_fallthrough)] {
        assert!(
            db.rel_iter::<(Address, Address)>("rtl_succ_candidate")
                .any(|candidate| *candidate == edge),
            "safe scratch-RBP CMP lost CFG edge {edge:#x?}"
        );
    }
}

fn assert_addr32_semantics(db: &DecompileDB) {
    let eax_load = memory_access_address(db, "addr32_eax_load", "EAX", 0);
    assert!(db
        .rel_iter::<(Address, u8)>("instruction_address_size")
        .any(|(addr, size)| (*addr, *size) == (eax_load, 4)));
    assert!(rtl_candidates(db, eax_load).iter().any(|inst| matches!(
        inst,
        RTLInst::Iload(
            MemoryChunk::MInt32,
            Addressing::Aaddr32(inner),
            args,
            _
        ) if **inner == Addressing::Aindexed(0) && args.len() == 1
    )));

    let indexed = memory_access_address(db, "addr32_eax_indexed", "EAX", 32);
    assert!(rtl_candidates(db, indexed).iter().any(|inst| matches!(
        inst,
        RTLInst::Iload(
            MemoryChunk::MInt32,
            Addressing::Aaddr32(inner),
            args,
            _
        ) if **inner == Addressing::Aindexed2scaled(4, 32) && args.len() == 2
    )));

    let index_only = memory_access_address(db, "addr32_index_only", "NONE", 32);
    let index_only_candidates = rtl_candidates(db, index_only);
    let index_only_reasons: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .filter(|(_, access, _)| *access == index_only)
        .copied()
        .collect();
    let index_only_mach: Vec<_> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| (*address == index_only).then_some(inst.clone()))
        .collect();
    let index_only_instruction: Vec<_> = db
        .rel_iter::<InstructionRow>("instruction")
        .filter(|(address, ..)| *address == index_only)
        .copied()
        .collect();
    let index_only_operands: HashSet<Symbol> = index_only_instruction
        .iter()
        .flat_map(|(_, _, _, _, op1, op2, op3, op4, _, _)| [*op1, *op2, *op3, *op4])
        .collect();
    let index_only_indirect: Vec<_> = db
        .rel_iter::<(
            Symbol,
            &'static str,
            &'static str,
            &'static str,
            i64,
            i64,
            usize,
        )>("op_indirect")
        .filter(|(operand, ..)| index_only_operands.contains(operand))
        .copied()
        .collect();
    let index_only_registers: Vec<_> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .filter(|(operand, _)| index_only_operands.contains(operand))
        .copied()
        .collect();
    let index_only_pmov: Vec<_> = db
        .rel_iter::<(Address, Symbol, Symbol)>("pmov")
        .filter(|(address, _, _)| *address == index_only)
        .copied()
        .collect();
    let index_only_transl_load: Vec<_> = db
        .rel_iter::<(Address, MemoryChunk, Addrmode, Arc<Vec<Mreg>>, Mreg)>("transl_load")
        .filter_map(|(address, chunk, mode, args, destination)| {
            (*address == index_only).then_some((
                *chunk,
                *mode,
                args.clone(),
                *destination,
            ))
        })
        .collect();
    assert!(
        index_only_candidates.iter().any(|inst| matches!(
            inst,
            RTLInst::Iload(
                MemoryChunk::MInt32,
                Addressing::Aaddr32(inner),
                args,
                _
            ) if **inner == Addressing::Ascaled(4, 32) && args.len() == 1
        )),
        "addr32 index-only load candidates: {index_only_candidates:#?}; reasons: {index_only_reasons:#?}; Mach: {index_only_mach:#?}; instruction: {index_only_instruction:#?}; indirect: {index_only_indirect:#?}; registers: {index_only_registers:#?}; pmov: {index_only_pmov:#?}; transl_load: {index_only_transl_load:#?}"
    );

    let scale1 = memory_access_address(db, "addr32_scale1", "EAX", 32);
    assert!(rtl_candidates(db, scale1).iter().any(|inst| matches!(
        inst,
        RTLInst::Iload(
            MemoryChunk::MInt32,
            Addressing::Aaddr32(inner),
            args,
            _
        ) if **inner == Addressing::Aindexed2(32) && args.len() == 2
    )));

    let rmw = memory_access_address(db, "addr32_rmw_store", "EAX", 0);
    assert!(db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .any(|(node, inst)| {
            *node == (rmw | (1u64 << 63))
                && matches!(
                    inst,
                    RTLInst::Istore(
                        MemoryChunk::MInt32,
                        Addressing::Aaddr32(inner),
                        args,
                        _
                    ) if **inner == Addressing::Aindexed(0) && args.len() == 1
                )
        }));

    let lea = instruction_address(db, "addr32_lea", "LEA");
    assert!(rtl_candidates(db, lea).iter().any(|inst| matches!(
        inst,
        RTLInst::Iop(
            Operation::Olea(Addressing::Aaddr32(inner))
                | Operation::Oleal(Addressing::Aaddr32(inner)),
            args,
            _
        ) if **inner == Addressing::Aindexed2scaled(4, 32) && args.len() == 2
    )));

    let scratch = memory_access_address(db, "addr32_scratch_ebp", "EBP", 8);
    assert!(rtl_candidates(db, scratch).iter().any(|inst| matches!(
        inst,
        RTLInst::Iload(_, Addressing::Aaddr32(inner), args, _)
            if **inner == Addressing::Aindexed(8) && args.len() == 1
    )));
    assert!(!db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .any(|(func, _, _)| *func == function_span(db, "addr32_scratch_ebp").0));

    let synthetic_store = memory_access_address(db, "addr32_synthetic_store", "EAX", 0);
    assert!(db
        .rel_iter::<(Address, RTLInst)>("rtl_inst_candidate")
        .any(|(node, inst)| {
            (*node & !SYNTHETIC_NODE_MASK) == synthetic_store
                && matches!(
                    inst,
                    RTLInst::Istore(_, Addressing::Aaddr32(inner), args, _)
                        if **inner == Addressing::Aindexed(0) && args.len() == 1
                )
        }));

    let indirect_call = memory_access_address(db, "addr32_indirect_call", "EAX", 0);
    assert!(db
        .rel_iter::<(Address, u64, MemoryChunk, Addressing, Arc<Vec<u64>>)>(
            "call_through_memory_load"
        )
        .any(|(node, _, chunk, addressing, args)| {
            *node == indirect_call
                && *chunk == MemoryChunk::MInt64
                && matches!(
                    addressing,
                    Addressing::Aaddr32(inner) if **inner == Addressing::Aindexed(0)
                )
                && args.len() == 1
        }));

    for (name, base, disp) in [
        ("addr32_esp_unsupported", "ESP", 0),
        ("addr32_frame_ebp_unsupported", "EBP", 8),
        ("addr32_unhandled_memory", "EAX", 0),
    ] {
        let access = memory_access_address(db, name, base, disp);
        assert_addr32_unsupported(db, name, access);
    }

    let addr64_load = memory_access_address(db, "addr64_load_control", "RAX", 0);
    assert!(rtl_candidates(db, addr64_load).iter().any(|inst| matches!(
        inst,
        RTLInst::Iload(_, Addressing::Aindexed(0), args, _) if args.len() == 1
    )));
    assert!(!rtl_candidates(db, addr64_load)
        .iter()
        .any(|inst| matches!(inst, RTLInst::Iload(_, Addressing::Aaddr32(_), _, _))));

    let addr64_lea = instruction_address(db, "addr64_lea_control", "LEA");
    assert!(rtl_candidates(db, addr64_lea).iter().any(|inst| matches!(
        inst,
        RTLInst::Iop(
            Operation::Olea(Addressing::Aindexed2scaled(4, 32))
                | Operation::Oleal(Addressing::Aindexed2scaled(4, 32)),
            args,
            _
        ) if args.len() == 2
    )));
}

fn assert_plain_memory_fallbacks(db: &DecompileDB) {
    for name in ["rsp_after_mov_load", "rsp_after_mov_store"] {
        let span = function_span(db, name);
        let access = memory_access_address(db, name, "RSP", 8);
        assert!(db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, row_access, reason)| {
                (*func, *row_access, *reason) == (span.0, access, "unsupported-stack-address")
            }));
        let candidates = rtl_candidates(db, access);
        assert!(
            !candidates.iter().any(|inst| matches!(
                inst,
                RTLInst::Iload(_, Addressing::Ainstack(_), _, _)
                    | RTLInst::Istore(_, Addressing::Ainstack(_), _, _)
            )),
            "{name} fabricated a stack slot: {candidates:#x?}"
        );
        assert!(
            !candidates.iter().any(|inst| matches!(
                inst,
                RTLInst::Iload(_, Addressing::Aindexed(_), args, _)
                    | RTLInst::Istore(_, Addressing::Aindexed(_), args, _)
                        if args.is_empty()
            )),
            "{name} emitted an absolute-looking empty-base fallback: {candidates:#x?}"
        );
    }

    let multi_span = function_span(db, "rsp_after_mov_two_accesses");
    let mut multi_rows: Vec<_> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .filter(|(func, _, reason)| *func == multi_span.0 && *reason == "unsupported-stack-address")
        .map(|(_, access, _)| *access)
        .collect();
    multi_rows.sort_unstable();
    multi_rows.dedup();
    assert_eq!(
        multi_rows.len(),
        2,
        "multi-access diagnostics must retain both sites"
    );

    for (name, is_store) in [("rbp_scratch_load", false), ("rbp_scratch_store", true)] {
        let access = memory_access_address(db, name, "RBP", 8);
        let candidates = rtl_candidates(db, access);
        let found = candidates.iter().any(|inst| match (is_store, inst) {
            (false, RTLInst::Iload(MemoryChunk::MInt32, Addressing::Aindexed(8), args, _)) => {
                args.len() == 1
            }
            (true, RTLInst::Istore(MemoryChunk::MInt32, Addressing::Aindexed(8), args, _)) => {
                args.len() == 1
            }
            _ => false,
        });
        assert!(
            found,
            "{name} lost its one-base generic pointer access: {candidates:#x?}"
        );
        assert!(
            !candidates.iter().any(|inst| matches!(
                inst,
                RTLInst::Iload(_, Addressing::Ainstack(_), _, _)
                    | RTLInst::Istore(_, Addressing::Ainstack(_), _, _)
            )),
            "{name} fabricated a stack slot: {candidates:#x?}"
        );
    }
}

fn assert_unsafe_case_uses_generic_pointer_path(db: &DecompileDB, case: &UnsafeCase) {
    let span = function_span(db, case.name);
    let expected_address = instruction_address(db, case.name, case.mnemonic);

    if case.base == Mreg::SP {
        assert!(db
            .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
            .any(|(func, access, reason)| {
                (*func, *access, *reason) == (span.0, expected_address, "unsupported-stack-address")
            }));
        assert!(!db
            .rel_iter::<FloatLoadOp>("float_load_op")
            .any(|(address, ..)| *address == expected_address));
        assert!(!db
            .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("stack_unary_load_op")
            .any(|(address, ..)| *address == expected_address));
        assert!(!db
            .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("float_arith_stack_op")
            .any(|(address, ..)| *address == expected_address));
        assert!(!rtl_candidates(db, expected_address)
            .iter()
            .any(|inst| { matches!(inst, RTLInst::Iload(_, Addressing::Ainstack(_), _, _)) }));
        return;
    }

    let row = db
        .rel_iter::<FloatLoadOp>("float_load_op")
        .find(|(addr, op, chunk, addressing, args, _, unary)| {
            in_span(*addr, span)
                && *op == case.op
                && *chunk == case.chunk
                && *addressing == Addressing::Aindexed(case.disp)
                && args.as_ref() == &[case.base]
                && *unary == case.unary
        })
        .cloned()
        .unwrap_or_else(|| panic!("{} lost its generic load+op fallback", case.name));
    let address = row.0;
    assert_eq!(address, expected_address);

    assert!(!db
        .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("stack_unary_load_op")
        .any(|(addr, _, _, _, _)| *addr == address));
    assert!(!db
        .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("float_arith_stack_op")
        .any(|(addr, _, _, _, _)| *addr == address));

    assert!(!db
        .rel_iter::<(Address, Address)>("bp_frame_at")
        .any(|(access, func)| (*access, *func) == (address, span.0)));

    assert!(!db
        .rel_iter::<(Address, Address, i64, usize)>("stack_param_access")
        .any(|(access, func, _, _)| (*access, *func) == (address, span.0)));
    assert!(!db
        .rel_iter::<(Address, Address, i64, u64)>("stack_var")
        .any(|(func, access, _, _)| (*func, *access) == (span.0, address)));

    let load = rtl_candidates(db, address);
    assert!(
        load.iter().any(|inst| {
            matches!(inst,
            RTLInst::Iload(chunk, Addressing::Aindexed(disp), args, _)
                if *chunk == case.chunk && *disp == case.disp && args.len() == 1)
        }),
        "{} lost its generic RTL pointer load: {load:#x?}",
        case.name
    );
    assert!(
        !load.iter().any(|inst| {
            matches!(
                inst,
                RTLInst::Iload(_, Addressing::Ainstack(_), _, _)
                    | RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), _, _)
                    | RTLInst::Iop(Operation::Oleal(Addressing::Ainstack(_)), _, _)
            )
        }),
        "{} fabricated Ainstack at the unsafe access: {load:#x?}",
        case.name
    );

    let operation = rtl_candidates(db, address | SYNTH1);
    assert!(
        operation.iter().any(|inst| {
            matches!(inst, RTLInst::Iop(op, args, _) if *op == case.op && !args.is_empty())
        }),
        "{} lost the operation after its pointer load: {operation:#x?}",
        case.name
    );
}

fn assert_safe_stack_shortcuts_remain(db: &DecompileDB) {
    for (name, op, base, disp) in [
        ("rsp_safe_cvtsi", Operation::Osingleofint, Mreg::SP, 4),
        ("rbp_safe_imul", Operation::Omulimm(13), Mreg::BP, -4),
    ] {
        let span = function_span(db, name);
        let address = db
            .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("stack_unary_load_op")
            .find_map(|(address, row_op, row_base, row_disp, _)| {
                (in_span(*address, span) && *row_op == op && *row_base == base && *row_disp == disp)
                    .then_some(*address)
            })
            .unwrap_or_else(|| panic!("{name} lost its proved scalar stack shortcut"));
        assert!(!db
            .rel_iter::<FloatLoadOp>("float_load_op")
            .any(|(candidate, _, _, _, _, _, _)| *candidate == address));
    }

    for name in ["rsp_safe_sse", "rsp_equal_join_sse"] {
        let span = function_span(db, name);
        let address = db
            .rel_iter::<(Address, Operation, Mreg, i64, Mreg)>("float_arith_stack_op")
            .find_map(|(address, op, base, _, _)| {
                (in_span(*address, span)
                    && *op
                        == if name == "rsp_safe_sse" {
                            Operation::Oaddf
                        } else {
                            Operation::Omulf
                        }
                    && *base == Mreg::SP)
                    .then_some(*address)
            })
            .unwrap_or_else(|| panic!("{name} lost its proved float stack shortcut"));
        assert!(db
            .rel_iter::<(Address, Address)>("rsp_frame_at")
            .any(|(access, func)| (*access, *func) == (address, span.0)));
    }

    for name in [
        "rsp_safe_cvtsi",
        "rsp_safe_sse",
        "rsp_equal_join_sse",
        "rbp_safe_imul",
        "rbp_conditional_imul",
        "rbp_clobbered_cvtsi_sd",
        "rbp_conditional_sse",
        "rbp_scratch_load",
        "rbp_scratch_store",
        "rsp_pushw_balanced",
        "rbp_scratch_ucomisd",
    ] {
        let span = function_span(db, name);
        assert!(
            !db.rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
                .any(|(func, _, _)| *func == span.0),
            "{name} was spuriously suppressed"
        );
    }
}

fn assert_cfg_and_width_safety(db: &DecompileDB) {
    let invalid_bp = memory_access_address(db, "rbp_from_invalid_rsp", "RBP", 8);
    assert_stack_unsupported(db, "rbp_from_invalid_rsp", invalid_bp);
    let invalid_bp_span = function_span(db, "rbp_from_invalid_rsp");
    assert!(!db
        .rel_iter::<(Address, Address)>("bp_frame_at")
        .any(|(access, func)| (*access, *func) == (invalid_bp, invalid_bp_span.0)));
    assert!(
        !rtl_candidates(db, invalid_bp).iter().any(|inst| matches!(
            inst,
            RTLInst::Iload(_, Addressing::Ainstack(_), _, _)
                | RTLInst::Iload(_, Addressing::Aindexed(_), _, _)
        )),
        "invalid RSP-derived RBP emitted a load candidate"
    );

    for name in ["rsp_loop_drift", "rsp_diamond_stress"] {
        let access = memory_access_address(
            db,
            name,
            "RSP",
            if name == "rsp_loop_drift" { 40 } else { 8 },
        );
        assert_stack_unsupported(db, name, access);
    }

    // The lattice has at most one exported exact offset per owned node. The
    // twelve-diamond fixture has 4096 path offsets under path enumeration.
    let diamond = function_span(db, "rsp_diamond_stress");
    let owned_nodes = db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .filter(|(_, func)| *func == diamond.0)
        .count();
    let exact_rows = db
        .rel_iter::<(Address, Address, i64)>("rsp_frame_offset_at")
        .filter(|(func, _, _)| *func == diamond.0)
        .count();
    assert!(
        exact_rows <= owned_nodes,
        "bounded stack state emitted {exact_rows} rows for {owned_nodes} nodes"
    );

    let push = instruction_address(db, "rsp_pushw_balanced", "PUSH");
    let pop = instruction_address(db, "rsp_pushw_balanced", "POP");
    assert!(db
        .rel_iter::<(Address, Symbol, i64)>("adjusts_stack")
        .any(|row| *row == (push, "RSP", -2)));
    assert!(db
        .rel_iter::<(Address, Symbol, i64)>("adjusts_stack")
        .any(|row| *row == (pop, "RSP", 2)));
    let balanced_access = memory_access_address(db, "rsp_pushw_balanced", "RSP", 8);
    let balanced_span = function_span(db, "rsp_pushw_balanced");
    assert!(db
        .rel_iter::<(Address, Address)>("rsp_frame_at")
        .any(|(access, func)| (*access, *func) == (balanced_access, balanced_span.0)));

    let pop_sp = instruction_address(db, "rsp_popw_sp", "POP");
    assert!(!db
        .rel_iter::<(Address, Symbol, i64)>("adjusts_stack")
        .any(|(address, _, _)| *address == pop_sp));
    let pop_sp_access = memory_access_address(db, "rsp_popw_sp", "RSP", 8);
    assert_stack_unsupported(db, "rsp_popw_sp", pop_sp_access);
}

fn assert_indexed_rsp_is_never_scalarized(db: &DecompileDB) {
    for (name, mnemonic) in [
        ("rsp_indexed_imul", "IMUL"),
        ("rsp_indexed_cvtsi", "CVTSI2SD"),
        ("rsp_indexed_sse", "ADDSD"),
        ("rsp_indexed_lea", "LEA"),
    ] {
        let address = instruction_address(db, name, mnemonic);
        assert_stack_unsupported(db, name, address);
        assert!(
            !rtl_candidates(db, address).iter().any(|inst| matches!(
                inst,
                RTLInst::Iload(_, Addressing::Ainstack(_), _, _)
                    | RTLInst::Iop(Operation::Olea(Addressing::Ainstack(_)), _, _)
                    | RTLInst::Iop(Operation::Oleal(Addressing::Ainstack(_)), _, _)
            )),
            "{name} dropped its index into a scalar stack address"
        );
    }
}

fn assert_unproved_rbp_float_compare_loads_fp0(db: &DecompileDB) {
    let compare = instruction_address(db, "rbp_scratch_ucomisd", "UCOMISD");
    let loads: Vec<MachInst> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| (*address == compare).then_some(inst.clone()))
        .collect();
    assert!(
        loads.iter().any(|inst| matches!(
            inst,
            MachInst::Mload(MemoryChunk::MFloat64, Addressing::Aindexed(8), args, Mreg::FP0)
                if args.as_ref() == &[Mreg::BP]
        )),
        "unproved RBP compare did not load FP0: {loads:#?}"
    );

    let branch = instruction_address(db, "rbp_scratch_ucomisd", "JA");
    let conditions: Vec<MachInst> = db
        .rel_iter::<(Address, MachInst)>("mach_inst")
        .filter_map(|(address, inst)| (*address == branch).then_some(inst.clone()))
        .collect();
    assert!(
        conditions.iter().any(|inst| matches!(
            inst,
            MachInst::Mcond(_, args, _) if args.contains(&Mreg::FP0)
        )),
        "memory compare consumed undefined FP0: {conditions:#?}"
    );
}

fn assert_shared_and_indirect_stack_sites_are_rejected(db: &DecompileDB) {
    let shared_b = function_span(db, "shared_stack_b");
    let shared_access = memory_access_address(db, "shared_stack_b", "RSP", 8);
    let shared_a = function_span(db, "shared_stack_a");
    assert!(db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .any(|(node, func)| (*node, *func) == (shared_access, shared_a.0)));
    assert_stack_unsupported(db, "shared_stack_a", shared_access);
    assert_stack_unsupported(db, "shared_stack_b", shared_access);
    assert_ne!(shared_a.0, shared_b.0);

    let indirect = memory_access_address(db, "unresolved_indirect_stack", "RSP", 8);
    let indirect_span = function_span(db, "unresolved_indirect_stack");
    assert!(
        db.rel_iter::<(Address, Address)>("rsp_frame_at")
            .any(|(access, func)| (*access, *func) == (indirect, indirect_span.0)),
        "fixture must retain a direct safe path so the indirect-edge gate is exercised"
    );
    assert_stack_unsupported(db, "unresolved_indirect_stack", indirect);

    let shared_addr32 = memory_access_address(db, "addr32_shared_ebp_scratch", "EBP", 8);
    let addr32_frame = function_span(db, "addr32_shared_ebp_frame");
    let addr32_scratch = function_span(db, "addr32_shared_ebp_scratch");
    assert_ne!(addr32_frame.0, addr32_scratch.0);
    assert!(db
        .rel_iter::<(Address, Address)>("instr_in_function")
        .any(|(node, func)| (*node, *func) == (shared_addr32, addr32_frame.0)));
    assert_addr32_unsupported(db, "addr32_shared_ebp_frame", shared_addr32);
    assert_addr32_unsupported(db, "addr32_shared_ebp_scratch", shared_addr32);

    let unresolved_addr32 = memory_access_address(db, "addr32_unresolved_ebp", "EBP", 8);
    let unresolved_addr32_span = function_span(db, "addr32_unresolved_ebp");
    let rcx_operands: HashSet<Symbol> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .filter_map(|(operand, register)| (*register == "RCX").then_some(*operand))
        .collect();
    let rbp_operands: HashSet<Symbol> = db
        .rel_iter::<(Symbol, &'static str)>("op_register")
        .filter_map(|(operand, register)| (*register == "RBP").then_some(*operand))
        .collect();
    assert!(
        db.rel_iter::<InstructionRow>("instruction").any(
            |(address, _, _, mnemonic, op1, op2, op3, op4, _, _)| {
                let operands = [*op1, *op2, *op3, *op4];
                in_span(*address, unresolved_addr32_span)
                    && *address < unresolved_addr32
                    && *mnemonic == "MOV"
                    && operands
                        .iter()
                        .any(|operand| rcx_operands.contains(operand))
                    && operands
                        .iter()
                        .any(|operand| rbp_operands.contains(operand))
            }
        ),
        "fixture must initialize EBP from a non-stack scratch value"
    );
    assert!(
        db.rel_iter::<(Address, Address, Symbol)>("ddisasm_cfg_edge")
            .any(|(source, _, kind)| {
                in_span(*source, unresolved_addr32_span) && *kind == "indirect"
            }),
        "fixture must retain an unresolved indirect CFG edge"
    );
    assert_addr32_unsupported(db, "addr32_unresolved_ebp", unresolved_addr32);
}

fn assert_asm_rerun_uses_decoder_owned_register_effects(object: &Path) {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    AbiPass.run(&mut db);

    let decoded_use: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("decoded_reg_use")
        .copied()
        .collect();
    let decoded_def: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("decoded_reg_def")
        .copied()
        .collect();

    AsmPass.run(&mut db);
    let first_use: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("asm_reg_use")
        .copied()
        .collect();
    let first_def: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("asm_reg_def")
        .copied()
        .collect();
    assert_eq!(first_use, decoded_use);
    assert_eq!(first_def, decoded_def);

    AsmPass.run(&mut db);
    let second_use: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("asm_reg_use")
        .copied()
        .collect();
    let second_def: HashSet<_> = db
        .rel_iter::<(Address, Mreg)>("asm_reg_def")
        .copied()
        .collect();
    assert_eq!(second_use, decoded_use);
    assert_eq!(second_def, decoded_def);
}

fn assert_unsupported_rsp_suppresses_only_affected_functions(object: &Path) {
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    db.run_pipeline(object, false, false);

    assert_all_structured_unsupported_sites_are_filtered(&db);

    // This path uses the scheduled/parallel pipeline rather than manually
    // ordered passes, guarding the Asm seed -> RTL public diagnostic edge.
    let unhandled_addr32 = memory_access_address(&db, "addr32_unhandled_memory", "EAX", 0);
    assert_addr32_unsupported(&db, "addr32_unhandled_memory", unhandled_addr32);

    let selected: Vec<&str> = db
        .cast_selected_functions
        .iter()
        .map(|func| func.name.as_str())
        .collect();
    assert!(!selected
        .iter()
        .any(|name| is_fixture_name(name, "rsp_after_mov_load")));
    assert!(!selected
        .iter()
        .any(|name| is_fixture_name(name, "rsp_after_pop_imul")));
    assert!(!selected
        .iter()
        .any(|name| is_fixture_name(name, "rsp_after_mov_two_accesses")));
    for unsupported in [
        "rbp_from_invalid_rsp",
        "rsp_loop_drift",
        "rsp_popw_sp",
        "rsp_indexed_imul",
        "rsp_indexed_cvtsi",
        "rsp_indexed_sse",
        "rsp_indexed_lea",
        "shared_stack_a",
        "shared_stack_b",
        "unresolved_indirect_stack",
        "unsupported_internal_callee",
        "rsp_diamond_stress",
        "addr32_esp_unsupported",
        "addr32_frame_ebp_unsupported",
        "addr32_shared_ebp_frame",
        "addr32_shared_ebp_scratch",
        "addr32_unresolved_ebp",
        "addr32_unhandled_memory",
    ] {
        assert!(
            !selected
                .iter()
                .any(|name| is_fixture_name(name, unsupported)),
            "unsupported function {unsupported} reached selection"
        );
    }
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "rsp_safe_cvtsi")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "rbp_scratch_load")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "rsp_pushw_balanced")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "rbp_scratch_ucomisd")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "safe_calls_unsupported")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "safe_sibling_return")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "addr32_scratch_ebp")));
    assert!(selected
        .iter()
        .any(|name| is_fixture_name(name, "addr32_safe_sibling")));

    let omitted = function_span(&db, "unsupported_internal_callee").0;
    // Equal-length aliases are deliberately inserted in reverse lexical
    // order. The provider name must remain authoritative at every call and
    // declaration site.
    db.rel_push("ident_to_symbol", (omitted as usize, "unsupported_alias_b"));
    db.rel_push("ident_to_symbol", (omitted as usize, "unsupported_alias_a"));

    let output = std::env::temp_dir().join(format!(
        "manifold_asm_stack_safety_export_{}.json",
        std::process::id()
    ));
    manifold::export_clight_json(
        &db,
        output.to_str().expect("temporary JSON path is UTF-8"),
    )
    .expect("stack-safety JSON export failed");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).expect("read stack-safety JSON"))
            .expect("parse stack-safety JSON");
    let _ = std::fs::remove_file(&output);

    let json_function_names: Vec<&str> = json["functions"]
        .as_array()
        .expect("functions array")
        .iter()
        .filter_map(|function| function["name"].as_str())
        .collect();
    assert!(json_function_names
        .iter()
        .any(|name| is_fixture_name(name, "safe_calls_unsupported")));
    assert!(json_function_names
        .iter()
        .any(|name| is_fixture_name(name, "safe_sibling_return")));
    assert!(!json_function_names
        .iter()
        .any(|name| is_fixture_name(name, "unsupported_internal_callee")));

    let extern_names: Vec<&str> = json["externals"]
        .as_array()
        .expect("externals array")
        .iter()
        .filter_map(|declaration| declaration["name"].as_str())
        .collect();
    assert!(
        extern_names
            .iter()
            .any(|name| is_fixture_name(name, "unsupported_internal_callee")),
        "safe caller lost the omitted internal declaration: {extern_names:?}"
    );
    assert!(!extern_names
        .iter()
        .any(|name| name.starts_with("unsupported_alias_")));

    let safe_caller = json["functions"]
        .as_array()
        .expect("functions array")
        .iter()
        .find(|function| {
            function["name"]
                .as_str()
                .is_some_and(|name| is_fixture_name(name, "safe_calls_unsupported"))
        })
        .expect("safe caller in JSON");
    assert!(
        contains_named_evar(&safe_caller["body"], "unsupported_internal_callee")
            || contains_named_evar(&safe_caller["body"], "coff_fn_unsupported_internal_callee"),
        "provider identity was not used at the direct call site"
    );
    assert!(!contains_named_evar(
        &safe_caller["body"],
        "unsupported_alias_a"
    ));
    assert!(!contains_named_evar(
        &safe_caller["body"],
        "unsupported_alias_b"
    ));

    let addr32_indexed = json["functions"]
        .as_array()
        .expect("functions array")
        .iter()
        .find(|function| {
            function["name"]
                .as_str()
                .is_some_and(|name| is_fixture_name(name, "addr32_eax_indexed"))
        })
        .expect("addr32 indexed function in JSON");
    assert!(
        contains_addr32_modulo_zero_extend(&addr32_indexed["body"]),
        "addr32 export lost per-operand uint32 truncation, modulo-2^32 arithmetic, or final zero extension"
    );

    let unsupported_json = json["unsupported_functions"]
        .as_array()
        .expect("unsupported_functions array");
    let unsupported_names: HashSet<&str> = unsupported_json
        .iter()
        .filter_map(|record| record["name"].as_str())
        .collect();
    assert!(unsupported_names
        .iter()
        .all(|name| !json_function_names.contains(name)));

    let reasons: HashSet<&str> = unsupported_json
        .iter()
        .map(|record| {
            record["reason"]
                .as_str()
                .expect("unsupported reason string")
        })
        .collect();
    assert_eq!(
        reasons,
        HashSet::from(["unsupported-addr32-address", "unsupported-stack-address"])
    );

    let unsupported_rows: Vec<(u64, u64, String)> = unsupported_json
        .iter()
        .map(|record| {
            (
                parse_hex_address(&record["address"]),
                parse_hex_address(&record["access_address"]),
                record["reason"]
                    .as_str()
                    .expect("unsupported reason string")
                    .to_owned(),
            )
        })
        .collect();
    let mut sorted_unsupported_rows = unsupported_rows.clone();
    sorted_unsupported_rows.sort();
    sorted_unsupported_rows.dedup();
    assert_eq!(unsupported_rows, sorted_unsupported_rows);

    let externals = json["externals"].as_array().expect("externals array");
    let external_keys: Vec<(String, u64, String, String)> = externals
        .iter()
        .map(|declaration| {
            (
                declaration["name"]
                    .as_str()
                    .expect("external name")
                    .to_owned(),
                declaration["param_count"]
                    .as_u64()
                    .expect("external parameter count"),
                declaration["return_type"].to_string(),
                declaration["param_types"].to_string(),
            )
        })
        .collect();
    let mut sorted_external_keys = external_keys.clone();
    sorted_external_keys.sort();
    sorted_external_keys.dedup();
    assert_eq!(external_keys, sorted_external_keys);

    let invalid_output = std::env::temp_dir().join(format!(
        "manifold_asm_stack_safety_invalid_reason_{}.json",
        std::process::id()
    ));
    let omitted_access = memory_access_address(&db, "unsupported_internal_callee", "RSP", 8);
    db.rel_push(
        "unsupported_stack_address",
        (omitted, omitted_access, "unsupported-test-reason"),
    );
    let invalid_error = manifold::export_clight_json(
        &db,
        invalid_output
            .to_str()
            .expect("temporary invalid JSON path is UTF-8"),
    )
    .expect_err("exporter accepted an unknown unsupported reason code");
    assert!(
        invalid_error.contains("unknown reason code")
            && invalid_error.contains("unsupported-test-reason"),
        "unexpected invalid-reason diagnostic: {invalid_error}"
    );
    let _ = std::fs::remove_file(&invalid_output);
}

#[test]
fn unsafe_bp_rsp_shortcuts_fall_back_to_pointer_loads() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("asm-stack-safety".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let db = load_rtl_relations(&object);
            let cases = [
                UnsafeCase {
                    name: "rsp_after_pop_imul",
                    mnemonic: "IMUL",
                    op: Operation::Omulimm(7),
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::SP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rsp_after_mov_cvtsi_sd",
                    mnemonic: "CVTSI2SD",
                    op: Operation::Ofloatofint,
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::SP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rsp_after_and_sse",
                    mnemonic: "ADDSD",
                    op: Operation::Oaddf,
                    chunk: MemoryChunk::MFloat64,
                    base: Mreg::SP,
                    disp: 40,
                    unary: false,
                },
                UnsafeCase {
                    name: "rsp_dynamic_cvtsi_ss",
                    mnemonic: "CVTSI2SS",
                    op: Operation::Osingleofint,
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::SP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rsp_branch_imul",
                    mnemonic: "IMUL",
                    op: Operation::Omulimm(9),
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::SP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rbp_conditional_imul",
                    mnemonic: "IMUL",
                    op: Operation::Omulimm(11),
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::BP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rbp_clobbered_cvtsi_sd",
                    mnemonic: "CVTSI2SD",
                    op: Operation::Ofloatofint,
                    chunk: MemoryChunk::MInt32,
                    base: Mreg::BP,
                    disp: 40,
                    unary: true,
                },
                UnsafeCase {
                    name: "rbp_conditional_sse",
                    mnemonic: "MULSD",
                    op: Operation::Omulf,
                    chunk: MemoryChunk::MFloat64,
                    base: Mreg::BP,
                    disp: 40,
                    unary: false,
                },
            ];
            for case in &cases {
                assert_unsafe_case_uses_generic_pointer_path(&db, case);
            }
            assert_plain_memory_fallbacks(&db);
            assert_rejected_rmw_chain_is_atomic(&db);
            assert_safe_stack_shortcuts_remain(&db);
            assert_cfg_and_width_safety(&db);
            assert_indexed_rsp_is_never_scalarized(&db);
            assert_unproved_rbp_float_compare_loads_fp0(&db);
            assert_shared_and_indirect_stack_sites_are_rejected(&db);
            assert_addr32_semantics(&db);
            assert_cmp_cross_node_safety(&db);
            assert_all_structured_unsupported_sites_are_filtered(&db);
        })
        .expect("failed to spawn asm stack-safety test thread")
        .join()
        .expect("asm stack-safety test thread panicked");
}

#[test]
fn asm_register_effect_snapshot_is_idempotent() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("asm-register-idempotence".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_asm_rerun_uses_decoder_owned_register_effects(&object))
        .expect("failed to spawn Asm register-idempotence test thread")
        .join()
        .expect("Asm register-idempotence test thread panicked");
}

#[test]
fn unsupported_rsp_is_structured_and_sibling_safe() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("asm-stack-unsupported-selection".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_unsupported_rsp_suppresses_only_affected_functions(&object))
        .expect("failed to spawn unsupported-stack export test thread")
        .join()
        .expect("unsupported-stack export test thread panicked");
}
