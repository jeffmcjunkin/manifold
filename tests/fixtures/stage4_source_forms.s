        # Deterministic x86-64 COFF fixture for the Stage-4 source-shape
        # portfolio. Each function lives in an independent COMDAT-like text
        # section so the authenticated COFF range map has exact boundaries.

        .section .text$stage4_affine64,"xr"
        .globl stage4_affine64
        .def stage4_affine64; .scl 2; .type 32; .endef
stage4_affine64:
        leaq 12(%rcx,%rdx,4), %rax
        movq %rax, (%r8)
        movq %rax, 8(%r8)
        retq

        .section .text$stage4_affine32,"xr"
        .globl stage4_affine32
        .def stage4_affine32; .scl 2; .type 32; .endef
stage4_affine32:
        leal 12(%rcx,%rdx,4), %eax
        movl %eax, (%r8)
        movl %eax, 4(%r8)
        retq

        .section .text$stage4_zero_xor32,"xr"
        .globl stage4_zero_xor32
        .def stage4_zero_xor32; .scl 2; .type 32; .endef
stage4_zero_xor32:
        xorl %eax, %eax
        retq

        .section .text$stage4_zero_sub64,"xr"
        .globl stage4_zero_sub64
        .def stage4_zero_sub64; .scl 2; .type 32; .endef
stage4_zero_sub64:
        subq %rax, %rax
        retq

        .section .text$stage4_add_reg32,"xr"
        .globl stage4_add_reg32
        .def stage4_add_reg32; .scl 2; .type 32; .endef
stage4_add_reg32:
        movl (%rcx), %eax
        addl %edx, %eax
        retq

        .section .text$stage4_add_reg64,"xr"
        .globl stage4_add_reg64
        .def stage4_add_reg64; .scl 2; .type 32; .endef
stage4_add_reg64:
        movq (%rcx), %rax
        addq %rdx, %rax
        retq

        .section .text$stage4_add_imm64,"xr"
        .globl stage4_add_imm64
        .def stage4_add_imm64; .scl 2; .type 32; .endef
stage4_add_imm64:
        movq (%rcx), %rax
        addq $7, %rax
        retq

        .section .text$stage4_sub_imm32,"xr"
        .globl stage4_sub_imm32
        .def stage4_sub_imm32; .scl 2; .type 32; .endef
stage4_sub_imm32:
        movl (%rcx), %eax
        subl $5, %eax
        retq

        .section .text$stage4_mul_reg64,"xr"
        .globl stage4_mul_reg64
        .def stage4_mul_reg64; .scl 2; .type 32; .endef
stage4_mul_reg64:
        movq (%rcx), %rax
        imulq %rdx, %rax
        retq

        .section .text$stage4_mul_imm32,"xr"
        .globl stage4_mul_imm32
        .def stage4_mul_imm32; .scl 2; .type 32; .endef
stage4_mul_imm32:
        movl (%rcx), %eax
        imull $3, %eax
        retq

        .section .text$stage4_and_imm64,"xr"
        .globl stage4_and_imm64
        .def stage4_and_imm64; .scl 2; .type 32; .endef
stage4_and_imm64:
        movq (%rcx), %rax
        andq $255, %rax
        movq %rax, (%rdx)
        movq %r8, %rax
        retq

        .section .text$stage4_or_reg64,"xr"
        .globl stage4_or_reg64
        .def stage4_or_reg64; .scl 2; .type 32; .endef
stage4_or_reg64:
        movq (%rcx), %rax
        orq %rdx, %rax
        retq

        .section .text$stage4_xor_imm64,"xr"
        .globl stage4_xor_imm64
        .def stage4_xor_imm64; .scl 2; .type 32; .endef
stage4_xor_imm64:
        movq (%rcx), %rax
        xorq $31, %rax
        retq

        # Same-function disjoint roots. Each must be offered independently;
        # no alternative may combine both source choices.
        .section .text$stage4_two_roots,"xr"
        .globl stage4_two_roots
        .def stage4_two_roots; .scl 2; .type 32; .endef
stage4_two_roots:
        movl (%rcx), %eax
        addl $3, %eax
        movl %eax, (%r8)
        movl 4(%rcx), %eax
        imull $5, %eax
        movl %eax, 4(%r8)
        movl %r9d, %eax
        retq

        # Explicit exclusions: memory destination, three-operand IMUL,
        # stack-relative LEA, and byte/word zeroing are outside the sealed v1
        # Stage-4 machine contracts and must retain only the cumulative v3
        # portfolio.
        .section .text$stage4_memory_rmw_control,"xr"
        .globl stage4_memory_rmw_control
        .def stage4_memory_rmw_control; .scl 2; .type 32; .endef
stage4_memory_rmw_control:
        addl $1, (%rcx)
        movl (%rcx), %eax
        retq

        .section .text$stage4_three_operand_mul_control,"xr"
        .globl stage4_three_operand_mul_control
        .def stage4_three_operand_mul_control; .scl 2; .type 32; .endef
stage4_three_operand_mul_control:
        imull $3, %ecx, %eax
        retq

        .section .text$stage4_stack_lea_control,"xr"
        .globl stage4_stack_lea_control
        .def stage4_stack_lea_control; .scl 2; .type 32; .endef
stage4_stack_lea_control:
        leaq 16(%rsp), %rax
        retq

        .section .text$stage4_narrow_zero_control,"xr"
        .globl stage4_narrow_zero_control
        .def stage4_narrow_zero_control; .scl 2; .type 32; .endef
stage4_narrow_zero_control:
        xorw %ax, %ax
        movzwl %ax, %eax
        retq
