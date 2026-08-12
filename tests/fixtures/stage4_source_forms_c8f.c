struct struct_1 {
    int ofs_0;
    int ofs_4;
};

struct struct_2 {
    unsigned char _opaque;
};

struct struct_3 {
    int ofs_0;
};

struct struct_4 {
    __int64 ofs_0;
};

__int64 coff_fn_stage4_affine64(__int64 p0, __int64 p1, struct struct_2 *p2);

int coff_fn_stage4_affine32(__int64 p0, __int64 p1, struct struct_1 *p2);

int coff_fn_stage4_zero_xor32(void);

__int64 coff_fn_stage4_zero_sub64(void);

int coff_fn_stage4_add_reg32(int *p0, __int64 p1);

__int64 coff_fn_stage4_add_reg64(struct struct_4 *p0, __int64 p1);

__int64 coff_fn_stage4_add_imm64(struct struct_4 *p0);

int coff_fn_stage4_sub_imm32(int *p0);

__int64 coff_fn_stage4_mul_reg64(struct struct_4 *p0, __int64 p1);

int coff_fn_stage4_mul_imm32(int *p0);

__int64 coff_fn_stage4_and_imm64(struct struct_4 *p0, struct struct_4 *p1, __int64 p2);

__int64 coff_fn_stage4_or_reg64(struct struct_4 *p0, __int64 p1);

__int64 coff_fn_stage4_xor_imm64(struct struct_4 *p0);

__int64 coff_fn_stage4_two_roots(struct struct_1 *p0, __int64 p1, struct struct_1 *p2, __int64 p3);

int coff_fn_stage4_memory_rmw_control(int *p0);

int coff_fn_stage4_three_operand_mul_control(int p0);

__int64 coff_fn_stage4_stack_lea_control(void);

unsigned short coff_fn_stage4_narrow_zero_control(void);

__int64 coff_fn_stage4_affine64(__int64 p0, __int64 p1, struct struct_2 *p2)
{
    __int64 var_0;

    var_0 = p0 + p1 * 4 + 12;
    *(__int64 *)p2 = var_0;
    *(__int64 *)((char *)p2 + 8) = var_0;
    return var_0;
}

int coff_fn_stage4_affine32(__int64 p0, __int64 p1, struct struct_1 *p2)
{
    int var_0;

    var_0 = p0 + p1 * 4 + 12;
    ((struct struct_1 *)p2)->ofs_0 = var_0;
    ((struct struct_1 *)p2)->ofs_4 = (unsigned int)var_0;
    return var_0;
}

int coff_fn_stage4_zero_xor32(void)
{
    return 0;
}

__int64 coff_fn_stage4_zero_sub64(void)
{
    return 0;
}

int coff_fn_stage4_add_reg32(int *p0, __int64 p1)
{
    return ((struct struct_3 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_add_reg64(struct struct_4 *p0, __int64 p1)
{
    return ((struct struct_4 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_add_imm64(struct struct_4 *p0)
{
    return ((struct struct_4 *)p0)->ofs_0;
}

int coff_fn_stage4_sub_imm32(int *p0)
{
    return ((struct struct_3 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_mul_reg64(struct struct_4 *p0, __int64 p1)
{
    return ((struct struct_4 *)p0)->ofs_0;
}

int coff_fn_stage4_mul_imm32(int *p0)
{
    return ((struct struct_3 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_and_imm64(struct struct_4 *p0, struct struct_4 *p1, __int64 p2)
{
    __int64 var_0;

    var_0 = ((struct struct_4 *)p0)->ofs_0;
    ((struct struct_4 *)p1)->ofs_0 = var_0;
    return p2;
}

__int64 coff_fn_stage4_or_reg64(struct struct_4 *p0, __int64 p1)
{
    return ((struct struct_4 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_xor_imm64(struct struct_4 *p0)
{
    return ((struct struct_4 *)p0)->ofs_0;
}

__int64 coff_fn_stage4_two_roots(struct struct_1 *p0, __int64 p1, struct struct_1 *p2, __int64 p3)
{
    int var_0;

    var_0 = ((struct struct_1 *)p0)->ofs_0;
    ((struct struct_1 *)p2)->ofs_0 = var_0;
    var_0 = ((struct struct_1 *)p0)->ofs_4;
    ((struct struct_1 *)p2)->ofs_4 = var_0;
    return p3;
}

int coff_fn_stage4_memory_rmw_control(int *p0)
{
    int var_0;

    var_0 = ((struct struct_3 *)p0)->ofs_0;
    var_0 = var_0 + 1;
    ((struct struct_3 *)p0)->ofs_0 = var_0;
    return ((struct struct_3 *)p0)->ofs_0;
}

int coff_fn_stage4_three_operand_mul_control(int p0)
{
    return p0 * 3;
}

__int64 coff_fn_stage4_stack_lea_control(void)
{
    int var_0;

    return (__int64)&var_0;
}

unsigned short coff_fn_stage4_narrow_zero_control(void)
{
    return 0;
}
