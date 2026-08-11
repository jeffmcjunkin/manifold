struct struct_1 {
    unsigned char _pad_0[12];
    int ofs_12;
};

struct struct_2 {
    unsigned char _pad_0[8];
    __int64 ofs_8;
};

struct struct_3 {
    unsigned char _pad_0[5];
    unsigned char ofs_5;
};

struct struct_4 {
    int ofs_0;
};

struct struct_5 {
    int i_0;
    unsigned char ofs_4;
};

int coff_fn_scalar_lvalue_plain32(int *p0, __int64 p1);

__int64 coff_fn_scalar_lvalue_plain64(struct struct_2 *p0, int p1);

char coff_fn_scalar_lvalue_movzx_indexed(const char *p0, __int64 p1);

char coff_fn_scalar_lvalue_movsx64(const char *p0, __int64 p1, __int64 p2);

int coff_fn_scalar_lvalue_store16(void *p0, int p1, short p2);

int coff_fn_scalar_lvalue_inline_index(int *p0, __int64 p1);

int coff_fn_scalar_lvalue_base718_high8_control(char *p0);

unsigned short coff_fn_scalar_lvalue_base718_word_dest_control(const char *p0);

__int64 coff_fn_scalar_lvalue_base718_wide_use_control(int *p0, __int64 p1);

int coff_fn_scalar_lvalue_plain32(int *p0, __int64 p1)
{
    return ((struct struct_1 *)((char *)p0 + p1 * 2))->ofs_12;
}

__int64 coff_fn_scalar_lvalue_plain64(struct struct_2 *p0, int p1)
{
    return ((struct struct_2 *)((char *)p0 + p1 * 4))->ofs_8;
}

char coff_fn_scalar_lvalue_movzx_indexed(const char *p0, __int64 p1)
{
    return *(unsigned char *)((char *)((char *)p0 + p1 * 4) + -3);
}

char coff_fn_scalar_lvalue_movsx64(const char *p0, __int64 p1, __int64 p2)
{
    return (char)((struct struct_3 *)((char *)p0 + p1 * 2))->ofs_5;
}

int coff_fn_scalar_lvalue_store16(void *p0, int p1, short p2)
{
    *(short *)((char *)((char *)p0 + p1 * 4) + -6) = p2;
    return 0;
}

int coff_fn_scalar_lvalue_inline_index(int *p0, __int64 p1)
{
    return ((struct struct_4 *)((int *)((char *)p0 + (__int64)24) + p1))->ofs_0;
}

int coff_fn_scalar_lvalue_base718_high8_control(char *p0)
{
    ((struct struct_5 *)p0)->ofs_4 = (unsigned char)0;
    return 0;
}

unsigned short coff_fn_scalar_lvalue_base718_word_dest_control(const char *p0)
{
    return (unsigned short)((struct struct_5 *)p0)->ofs_4;
}

__int64 coff_fn_scalar_lvalue_base718_wide_use_control(int *p0, __int64 p1)
{
    return (__int64)((struct struct_4 *)p0)->ofs_0;
}
