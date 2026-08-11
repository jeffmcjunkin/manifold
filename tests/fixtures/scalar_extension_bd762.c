struct struct_1 {
    char ofs_0;
    unsigned char _pad_1[7];
    int ofs_8;
};

struct struct_2 {
    int i_0;
    int ofs_4;
    int ofs_8;
};

struct struct_3 {
    int i_0;
    int ofs_4;
};

struct struct_4 {
    unsigned char uc_0;
    unsigned char ofs_1;
};

struct struct_5 {
    short s_0;
    unsigned char ofs_2;
};

struct struct_6 {
    __int64 ofs_0;
};

struct struct_7 {
    __int64 ofs_0;
    __int64 ofs_8;
    __int64 ofs_16;
};

__int64 coff_fn_stage3_movsxd64(int *p0, __int64 p1);

int coff_fn_stage3_movsx32_wide(const char *p0, struct struct_1 *p1);

int coff_fn_stage3_movzx32_wide(const char *p0, struct struct_6 *p1);

int coff_fn_stage3_plain32_fanout(int *p0, struct struct_7 *p1);

int coff_fn_stage3_field_profile(struct struct_2 *p0);

__int64 coff_fn_stage3_partial_merge_control(const char *p0, __int64 p1, __int64 p2);

__int64 coff_fn_stage3_movsxd64(int *p0, __int64 p1)
{
    return (__int64)((struct struct_3 *)p0)->ofs_4;
}

int coff_fn_stage3_movsx32_wide(const char *p0, struct struct_1 *p1)
{
    __int64 var_0;

    var_0 = ((struct struct_4 *)p0)->ofs_1;
    if ((int)var_0 >= 0) {
    }
    else {
        ((struct struct_1 *)p1)->ofs_8 = 1;
    }
    *(__int64 *)p1 = var_0;
    return 0;
}

int coff_fn_stage3_movzx32_wide(const char *p0, struct struct_6 *p1)
{
    __int64 var_0;

    var_0 = ((struct struct_5 *)p0)->ofs_2;
    ((struct struct_6 *)p1)->ofs_0 = var_0;
    return 0;
}

int coff_fn_stage3_plain32_fanout(int *p0, struct struct_7 *p1)
{
    __int64 var_0;

    var_0 = (__int64)((struct struct_3 *)p0)->ofs_4;
    ((struct struct_7 *)p1)->ofs_0 = var_0;
    ((struct struct_7 *)p1)->ofs_8 = var_0;
    ((struct struct_7 *)p1)->ofs_16 = var_0;
    return 0;
}

int coff_fn_stage3_field_profile(struct struct_2 *p0)
{
    int var_0;

    var_0 = ((struct struct_2 *)p0)->ofs_4;
    var_0 = var_0 + (__int64)((struct struct_2 *)p0)->ofs_8;
    return var_0;
}

__int64 coff_fn_stage3_partial_merge_control(const char *p0, __int64 p1, __int64 p2)
{
    return p1;
}
