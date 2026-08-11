        .section .text$stage3_movsxd64,"xr"
        .globl stage3_movsxd64
        .def stage3_movsxd64; .scl 2; .type 32; .endef
stage3_movsxd64:
        movslq 4(%rcx), %rax
        addq %rdx, %rax
        retq

        # EAX is consumed once as signed32 by the condition and independently
        # stored through the architecturally zeroed RAX carrier. Returning a
        # separate zero keeps the final ABI from becoming an inferred narrow
        # return and makes the two-stage sign32->uint32->uint64 chain observable.
        .section .text$stage3_movsx32_wide,"xr"
        .globl stage3_movsx32_wide
        .def stage3_movsx32_wide; .scl 2; .type 32; .endef
stage3_movsx32_wide:
        movsbl 1(%rcx), %eax
        cmpl $0, %eax
        jge 1f
        movl $1, 8(%rdx)
1:
        movq %rax, (%rdx)
        xorl %eax, %eax
        retq

        # Explicit MOVZX into EAX has the same architectural ZeroUpper32 result
        # stage, but its source access remains an unsigned byte.
        .section .text$stage3_movzx32_wide,"xr"
        .globl stage3_movzx32_wide
        .def stage3_movzx32_wide; .scl 2; .type 32; .endef
stage3_movzx32_wide:
        movzbl 2(%rcx), %eax
        movq %rax, (%rdx)
        xorl %eax, %eax
        retq

        # A plain MOV r32,m32 root fans out through two exact qword moves. The
        # moves may coalesce, but the root must retain three distinct final
        # 64-bit terminal stores, proving a bounded implicit-zero use forest
        # without relying on the return ABI.
        .section .text$stage3_plain32_fanout,"xr"
        .globl stage3_plain32_fanout
        .def stage3_plain32_fanout; .scl 2; .type 32; .endef
stage3_plain32_fanout:
        movl 4(%rcx), %eax
        movq %rax, %r8
        movq %rax, %r9
        movq %rax, (%rdx)
        movq %r8, 8(%rdx)
        movq %r9, 16(%rdx)
        xorl %eax, %eax
        retq

        # Two same-root extents seed the existing layout solver. The first
        # MOV access must admit a field-preferred profile only if that solver's
        # Efield/base+disp+width evidence closes exactly.
        .section .text$stage3_field_profile,"xr"
        .globl stage3_field_profile
        .def stage3_field_profile; .scl 2; .type 32; .endef
stage3_field_profile:
        movl 4(%rcx), %eax
        addl 8(%rcx), %eax
        retq

        # A partial-register merge after the load is outside the closed plan.
        .section .text$stage3_partial_merge_control,"xr"
        .globl stage3_partial_merge_control
        .def stage3_partial_merge_control; .scl 2; .type 32; .endef
stage3_partial_merge_control:
        movzbl 3(%rcx), %eax
        movb %dl, %al
        addq %r8, %rax
        retq
