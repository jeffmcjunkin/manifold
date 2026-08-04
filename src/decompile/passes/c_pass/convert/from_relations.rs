use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::c_pass::types::{
    AssignOp, BinaryOp, CBlockItem, CExpr, CStmt, CType, ExprTransform, FloatLiteral,
    FloatLiteralSuffix, FuncDef, FuncParam, IntLiteral, IntLiteralBase, IntLiteralSuffix, IntSize,
    Label, Signedness, SourceLoc, StmtTransform, StorageClass, StringLiteral, TypeQualifiers,
    UnaryOp, VarDecl,
};
use crate::decompile::passes::c_pass::TranslationUnit;
use crate::decompile::passes::clight_select::query::GlobalData;
use crate::decompile::passes::clight_select::select::SelectedFunction;
use crate::x86::types as clight;
use crate::x86::types::*;
use log::debug;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

pub(crate) type FunctionObjectTypes = HashMap<Address, HashMap<String, CType>>;

/// Addressable Clight locals are explicit pipeline objects.  Register values use
/// `Etempvar`; the `Evar` form is local only when stack recovery declared that
/// identifier in the current function.  Keep this provenance separate from the
/// global symbol-name map so an unnamed direct callee is never mistaken for a
/// local merely because it has no symbol.
pub(crate) fn local_evar_ids_by_function(db: &DecompileDB) -> HashMap<Address, HashSet<Ident>> {
    let mut out: HashMap<Address, HashSet<Ident>> = HashMap::new();
    for (func, _node, _offset, reg) in db.rel_iter::<(Address, Address, i64, RTLReg)>("stack_var") {
        out.entry(*func)
            .or_default()
            .insert(crate::decompile::passes::csh_pass::ident_from_reg(*reg));
    }
    for (func, buffers) in &db.stack_struct_buffers {
        let ids = out.entry(*func).or_default();
        ids.extend(
            buffers
                .keys()
                .map(|reg| crate::decompile::passes::csh_pass::ident_from_reg(*reg)),
        );
    }
    for (func, arrays) in &db.stack_array_buffers {
        let ids = out.entry(*func).or_default();
        ids.extend(
            arrays
                .keys()
                .map(|reg| crate::decompile::passes::csh_pass::ident_from_reg(*reg)),
        );
    }
    out
}

// GCC hot/cold splitting emits `<base>_cold` companions with trivial abort/unreachable bodies; drop only when name ends in `_cold` AND body matches that trivial pattern (empty body is kept, not silently deleted).
fn has_cold_suffix(name: &str) -> bool {
    name.len() > 5 && name.ends_with("_cold")
}

fn is_trivial_cold_body(body: &CStmt) -> bool {
    // Unwrap a single-statement Block/Sequence wrapper, then require exactly one trivial call.
    match body {
        CStmt::Block(items) => {
            let stmts: Vec<&CStmt> = items
                .iter()
                .filter_map(|item| match item {
                    CBlockItem::Stmt(s) => Some(s),
                    CBlockItem::Decl(_) => None,
                })
                .collect();
            stmts.len() == 1 && is_trivial_cold_call(stmts[0])
        }
        CStmt::Sequence(stmts) => stmts.len() == 1 && is_trivial_cold_call(&stmts[0]),
        other => is_trivial_cold_call(other),
    }
}

fn is_trivial_cold_call(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Expr(CExpr::Call(func, args)) => {
            if !args.is_empty() {
                return false;
            }
            if let CExpr::Var(name) = func.as_ref() {
                name == "abort" || name == "__builtin_unreachable"
            } else {
                false
            }
        }
        _ => false,
    }
}

fn convert_clight_type(ty: &crate::x86::types::ClightType) -> CType {
    use crate::x86::types::*;
    match ty {
        crate::x86::types::ClightType::Tvoid => CType::Void,
        crate::x86::types::ClightType::Tint(size, sign, _attr) => {
            let c_size = match size {
                ClightIntSize::I8 => crate::decompile::passes::c_pass::types::IntSize::Char,
                ClightIntSize::I16 => crate::decompile::passes::c_pass::types::IntSize::Short,
                ClightIntSize::I32 | ClightIntSize::IBool => {
                    crate::decompile::passes::c_pass::types::IntSize::Int
                }
            };
            let c_sign = match sign {
                ClightSignedness::Signed => {
                    crate::decompile::passes::c_pass::types::Signedness::Signed
                }
                ClightSignedness::Unsigned => {
                    crate::decompile::passes::c_pass::types::Signedness::Unsigned
                }
            };
            CType::Int(c_size, c_sign)
        }
        crate::x86::types::ClightType::Tlong(sign, _attr) => {
            let c_sign = match sign {
                ClightSignedness::Signed => {
                    crate::decompile::passes::c_pass::types::Signedness::Signed
                }
                ClightSignedness::Unsigned => {
                    crate::decompile::passes::c_pass::types::Signedness::Unsigned
                }
            };
            CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Long,
                c_sign,
            )
        }
        crate::x86::types::ClightType::Tint128(sign, _attr) => {
            let c_sign = match sign {
                ClightSignedness::Signed => {
                    crate::decompile::passes::c_pass::types::Signedness::Signed
                }
                ClightSignedness::Unsigned => {
                    crate::decompile::passes::c_pass::types::Signedness::Unsigned
                }
            };
            CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Int128,
                c_sign,
            )
        }
        crate::x86::types::ClightType::Tfloat(size, _attr) => {
            let c_size = match size {
                ClightFloatSize::F32 => crate::decompile::passes::c_pass::types::FloatSize::Float,
                ClightFloatSize::F64 => crate::decompile::passes::c_pass::types::FloatSize::Double,
            };
            CType::Float(c_size)
        }
        crate::x86::types::ClightType::Tpointer(inner, attr) => {
            let qualifiers = TypeQualifiers {
                is_volatile: attr.attr_volatile,
                ..TypeQualifiers::none()
            };
            CType::Pointer(Box::new(convert_clight_type(inner)), qualifiers)
        }
        crate::x86::types::ClightType::Tarray(inner, size, _attr) => {
            CType::Array(Box::new(convert_clight_type(inner)), Some(*size as usize))
        }
        crate::x86::types::ClightType::Tfunction(params, ret, cc) => {
            let c_params: Vec<CType> = params.iter().map(convert_clight_type).collect();
            // Carry the calling convention through: varargs gives variadic (T, ...), unproto gives true K&R () (unspecified args, distinct from (void)).
            CType::Function(
                Box::new(convert_clight_type(ret)),
                c_params,
                cc.varargs.is_some(),
                cc.unproto,
            )
        }
        crate::x86::types::ClightType::Tstruct(id, _attr) => {
            CType::Struct(format!("struct_{:x}", id))
        }
        crate::x86::types::ClightType::Tunion(id, _attr) => CType::Union(format!("union_{}", id)),
    }
}

fn clight_expr_to_ctype(expr: &clight::ClightExpr) -> CType {
    let clight_ty = match expr {
        clight::ClightExpr::EconstInt(_, ty)
        | clight::ClightExpr::EconstFloat(_, ty)
        | clight::ClightExpr::EconstSingle(_, ty)
        | clight::ClightExpr::EconstLong(_, ty)
        | clight::ClightExpr::Evar(_, ty)
        | clight::ClightExpr::EvarSymbol(_, ty)
        | clight::ClightExpr::Etempvar(_, ty)
        | clight::ClightExpr::Ederef(_, ty)
        | clight::ClightExpr::Eaddrof(_, ty)
        | clight::ClightExpr::Eunop(_, _, ty)
        | clight::ClightExpr::Ebinop(_, _, _, ty)
        | clight::ClightExpr::Ecast(_, ty)
        | clight::ClightExpr::Efield(_, _, ty)
        | clight::ClightExpr::Esizeof(_, ty)
        | clight::ClightExpr::Ealignof(_, ty)
        | clight::ClightExpr::Econdition(_, _, _, ty) => ty,
    };
    convert_clight_type(clight_ty)
}

// Well-known variadic libc/coreutils functions whose recovered prototypes carry only the fixed leading params, so calls passing the variadic tail need the prototype declared variadic.
fn is_known_variadic_fn(name: &str) -> bool {
    matches!(
        name,
        "printf"
            | "fprintf"
            | "sprintf"
            | "snprintf"
            | "dprintf"
            | "scanf"
            | "fscanf"
            | "sscanf"
            | "__printf_chk"
            | "__fprintf_chk"
            | "__sprintf_chk"
            | "__snprintf_chk"
            | "__isoc99_scanf"
            | "__isoc99_fscanf"
            | "__isoc99_sscanf"
            | "error"
            | "error_at_line"
            | "open"
            | "openat"
            | "fcntl"
            | "ioctl"
            | "syslog"
            | "asprintf"
            | "execl"
            | "execlp"
            | "execle"
    )
}

fn convert_xtype(xt: &XType) -> CType {
    match xt {
        XType::Xvoid => CType::Void,
        XType::Xint8signed => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Char,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xint8unsigned => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Char,
            crate::decompile::passes::c_pass::types::Signedness::Unsigned,
        ),
        XType::Xint16signed => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Short,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xint16unsigned => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Short,
            crate::decompile::passes::c_pass::types::Signedness::Unsigned,
        ),
        XType::Xint => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Int,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xintunsigned => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Int,
            crate::decompile::passes::c_pass::types::Signedness::Unsigned,
        ),
        XType::Xlong => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Long,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xlongunsigned => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Long,
            crate::decompile::passes::c_pass::types::Signedness::Unsigned,
        ),
        XType::Xptr => CType::Pointer(Box::new(CType::Void), TypeQualifiers::none()),
        XType::Xcharptr => CType::Pointer(
            Box::new(CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Char,
                crate::decompile::passes::c_pass::types::Signedness::Signed,
            )),
            TypeQualifiers::none(),
        ),
        XType::Xcharptrptr => CType::Pointer(
            Box::new(CType::Pointer(
                Box::new(CType::Int(
                    crate::decompile::passes::c_pass::types::IntSize::Char,
                    crate::decompile::passes::c_pass::types::Signedness::Signed,
                )),
                TypeQualifiers::none(),
            )),
            TypeQualifiers::none(),
        ),
        XType::Xintptr => CType::Pointer(
            Box::new(CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Int,
                crate::decompile::passes::c_pass::types::Signedness::Signed,
            )),
            TypeQualifiers::none(),
        ),
        XType::Xfloatptr => CType::Pointer(
            Box::new(CType::Float(
                crate::decompile::passes::c_pass::types::FloatSize::Double,
            )),
            TypeQualifiers::none(),
        ),
        XType::Xsingleptr => CType::Pointer(
            Box::new(CType::Float(
                crate::decompile::passes::c_pass::types::FloatSize::Float,
            )),
            TypeQualifiers::none(),
        ),
        XType::Xfuncptr => CType::Pointer(
            Box::new(CType::func_unprototyped(CType::Void)),
            TypeQualifiers::none(),
        ),
        XType::Xfloat => CType::Float(crate::decompile::passes::c_pass::types::FloatSize::Double),
        XType::Xsingle => CType::Float(crate::decompile::passes::c_pass::types::FloatSize::Float),
        XType::Xany32 => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Int,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xany64 => CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Long,
            crate::decompile::passes::c_pass::types::Signedness::Signed,
        ),
        XType::Xbool => CType::Bool,
        XType::XstructPtr(id) => CType::Pointer(
            Box::new(CType::Struct(format!("struct_{:x}", id))),
            TypeQualifiers::none(),
        ),
    }
}

/// Return type for a called-but-not-emitted callee's prototype, taken from the recovered signature rather than a hardcoded int, so a pointer/long-returning internal is not narrowed by insert_casts.
fn recovered_ret_ctype(map: &HashMap<String, XType>, name: &str) -> CType {
    map.get(name)
        .and_then(|xt| match xt {
            XType::Xvoid => None,
            XType::XstructPtr(_) => Some(CType::Pointer(
                Box::new(CType::Void),
                TypeQualifiers::none(),
            )),
            other => Some(convert_xtype(other)),
        })
        .unwrap_or(CType::Int(
            crate::decompile::passes::c_pass::types::IntSize::Int,
            Signedness::Signed,
        ))
}

/// Policy for locally-DEFINED functions colliding with a header declaration: "filter" drops the body, otherwise (and by default) it is emitted renamed <name>_local.
struct LocalDefPolicy {
    renames: HashMap<String, String>,
    filtered: HashSet<String>,
}

/// The canonical type spellings used by header_functions.json "globals" `type` fields (exposed as header_db().variables) -- a closed set kept tiny on purpose (header-declared globals are scalars/pointers, not aggregates).
fn parse_canonical_var_type(spelling: &str) -> Option<CType> {
    match spelling.trim() {
        "char **" => Some(CType::ptr(CType::ptr(CType::char_signed()))),
        "char *" => Some(CType::ptr(CType::char_signed())),
        "void *" => Some(CType::ptr(CType::Void)),
        "int" => Some(CType::int()),
        "long" => Some(CType::long()),
        _ => None,
    }
}

fn header_collision_policy(selected_functions: &[SelectedFunction]) -> LocalDefPolicy {
    let hdb = crate::decompile::passes::c_pass::header_db::header_db();
    let is_provided =
        |n: &str| hdb.functions.contains(n) || hdb.prefixes.iter().any(|p| n.starts_with(p));
    let mut policy = LocalDefPolicy {
        renames: HashMap::new(),
        filtered: HashSet::new(),
    };
    for f in selected_functions {
        if f.name.starts_with("FUN_") {
            continue;
        }
        let name = sanitize_c_symbol_name(&f.name);
        if !is_provided(&name) {
            continue;
        }
        if hdb.local_def_filter.contains(name.as_str()) {
            policy.filtered.insert(name);
        } else {
            policy
                .renames
                .insert(name.clone(), format!("{}_local", name));
        }
    }
    policy
}

/// Whether C emission can rely on a declaration outside this translation
/// unit. Header groups with `header: null` intentionally have no such source,
/// so their called names must still receive a compatible declaration here.
fn function_declaration_is_externally_provided(name: &str) -> bool {
    let hdb = crate::decompile::passes::c_pass::header_db::header_db();
    hdb.prefixes.iter().any(|prefix| name.starts_with(prefix))
        || hdb
            .includes
            .iter()
            .any(|(_, functions)| functions.iter().any(|function| *function == name))
}

pub fn build_cast_from_relations(
    db: &DecompileDB,
    selected_functions: &[SelectedFunction],
    globals: &[GlobalData],
    id_to_name: &HashMap<usize, String>,
    edges: &[(crate::x86::types::Node, crate::x86::types::Node)],
    struct_fields: &HashMap<String, HashMap<String, CType>>,
) -> TranslationUnit {
    let mut ctx = ConversionContext::new(id_to_name.clone());
    let local_evar_ids = local_evar_ids_by_function(db);
    ctx.func_renames = header_collision_policy(selected_functions).renames;
    for global in globals {
        if global.is_string {
            ctx.string_globals.insert(global.id);
        }
    }
    for (label, content, _size) in db.rel_iter::<(String, String, usize)>("string_data") {
        ctx.string_label_to_content
            .insert(label.clone(), content.clone());
    }

    {
        let mut addr_name_map: HashMap<u64, String> = HashMap::new();
        for (addr, name, _entry_node) in db.rel_iter::<(Address, Symbol, Node)>("emit_function") {
            addr_name_map.insert(*addr as u64, sanitize_c_symbol_name(name));
        }
        for func in selected_functions {
            addr_name_map.insert(func.address as u64, sanitize_c_symbol_name(&func.name));
        }
        let mut sorted: Vec<(u64, String)> = addr_name_map.into_iter().collect();
        for (_, name) in sorted.iter_mut() {
            if let Some(renamed) = ctx.func_renames.get(name) {
                *name = renamed.clone();
            }
        }
        sorted.sort_by_key(|(addr, _)| *addr);
        ctx.func_addrs_sorted = sorted;
    }

    let mut stmt_map: HashMap<crate::x86::types::Node, CStmt> = HashMap::new();
    for func in selected_functions {
        ctx.enter_function(func.address, local_evar_ids.get(&func.address));
        // func.statements is a HashMap; iterating it directly leaks non-deterministic order into convert_stmt -> record_var_type (which is first-wins), so a variable's recorded C type flips between runs. Walk nodes in sorted order.
        let mut nodes: Vec<crate::x86::types::Node> = func.statements.keys().copied().collect();
        nodes.sort();
        for node in nodes {
            let clight_stmt = &func.statements[&node];
            let converted = convert_stmt(clight_stmt, &mut ctx);
            stmt_map.insert(node, converted);
        }
    }

    build_translation_unit_from_stmt_map_with_types(
        db,
        selected_functions,
        globals,
        id_to_name,
        &stmt_map,
        edges,
        &ctx.var_types,
        ctx.function_object_types(),
        &HashMap::new(),
        struct_fields,
    )
}

pub fn build_translation_unit_from_stmt_map_with_types(
    db: &DecompileDB,
    selected_functions: &[SelectedFunction],
    globals: &[GlobalData],
    id_to_name: &HashMap<usize, String>,
    stmt_map: &HashMap<crate::x86::types::Node, CStmt>,
    edges: &[(crate::x86::types::Node, crate::x86::types::Node)],
    var_types: &HashMap<String, CType>,
    function_object_types: &FunctionObjectTypes,
    optimized_node_to_func: &HashMap<crate::x86::types::Node, crate::x86::types::Address>,
    struct_fields: &HashMap<String, HashMap<String, CType>>,
) -> TranslationUnit {
    let mut tu = TranslationUnit::new();
    let mut func_var_types = var_types.clone();
    let local_evar_ids = local_evar_ids_by_function(db);

    let dwarf_param_names: HashMap<(Address, usize), String> = db
        .rel_iter::<(Address, usize, Symbol)>("dwarf_func_param_name")
        .map(|&(addr, idx, name)| ((addr, idx), name.to_string()))
        .collect();

    let mut global_names: HashSet<String> = globals
        .iter()
        .map(|g| sanitize_c_symbol_name(&g.name))
        .collect();
    for func in selected_functions {
        global_names.insert(sanitize_c_symbol_name(&func.name));
    }
    for name in id_to_name.values() {
        global_names.insert(sanitize_c_symbol_name(name));
    }

    let known_global_types: HashMap<String, CType> = db
        .rel_iter::<(Symbol, XType)>("known_global_type")
        .map(|(name, xtype)| (sanitize_c_symbol_name(name), convert_xtype(xtype)))
        .collect();

    // Extern return types for the final usage-based fallback must pass through
    // the same exact-loader/conflict reducer as declarations.  Reading the
    // raw name-keyed table here would let address/kind or sanitizer collisions
    // pick an arbitrary return type late in C emission.
    let extern_return_types: HashMap<String, CType> = known_loader_signatures_from_db(db)
        .into_iter()
        .filter_map(|(name, (ret_type, _, _))| {
            let ty = convert_xtype(&ret_type);
            (!matches!(ty, CType::Void)).then_some((name, ty))
        })
        .collect();

    // Build global load chunk map: prefer integer over float (SSE bulk copies produce spurious floats).
    let global_chunks: HashMap<usize, MemoryChunk> = {
        let mut chunks: HashMap<usize, Vec<MemoryChunk>> = HashMap::new();
        for (id, chunk) in db.rel_iter::<(Ident, MemoryChunk)>("global_load_chunk") {
            chunks.entry(*id).or_default().push(*chunk);
        }
        chunks
            .into_iter()
            .map(|(id, mut chunk_list)| {
                // Filter out generic MAny/Unknown when specific chunks exist
                let has_specific = chunk_list.iter().any(|c| {
                    !matches!(
                        c,
                        MemoryChunk::MAny32 | MemoryChunk::MAny64 | MemoryChunk::Unknown
                    )
                });
                if has_specific {
                    chunk_list.retain(|c| {
                        !matches!(
                            c,
                            MemoryChunk::MAny32 | MemoryChunk::MAny64 | MemoryChunk::Unknown
                        )
                    });
                }
                // Filter out float chunks when integer chunks exist (SSE bulk copies produce MFloat64)
                let has_int = chunk_list.iter().any(|c| {
                    matches!(
                        c,
                        MemoryChunk::MBool
                            | MemoryChunk::MInt8Signed
                            | MemoryChunk::MInt8Unsigned
                            | MemoryChunk::MInt16Signed
                            | MemoryChunk::MInt16Unsigned
                            | MemoryChunk::MInt32
                            | MemoryChunk::MInt64
                    )
                });
                if has_int {
                    chunk_list
                        .retain(|c| !matches!(c, MemoryChunk::MFloat32 | MemoryChunk::MFloat64));
                }
                // Among remaining, prefer the largest integer type; use the chunk itself as a tiebreak so signed/unsigned pairs of the same width don't flip across parallel-Ascent runs.
                let best = chunk_list
                    .iter()
                    .max_by_key(|c| {
                        (
                            match c {
                                MemoryChunk::MBool => 0,
                                MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned => 1,
                                MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => 2,
                                MemoryChunk::MInt32 => 3,
                                MemoryChunk::MFloat32 => 4,
                                MemoryChunk::MFloat64 => 5,
                                MemoryChunk::MInt64 => 6,
                                MemoryChunk::MAny32 => 7,
                                MemoryChunk::MAny64 => 8,
                                MemoryChunk::Unknown => 9,
                            },
                            **c,
                        )
                    })
                    .copied()
                    .unwrap_or(MemoryChunk::Unknown);
                (id, best)
            })
            .collect()
    };

    // Rodata FP constants recovered from the binary (address, width, raw IEEE-754 bits), keyed by address so an unnamed .rodata slot emits a typed static const instead of a zeroed long.
    let rodata_fp_consts: HashMap<u64, (usize, u64)> = db
        .rel_iter::<(u64, usize, u64)>("rodata_fp_const")
        .map(|(addr, width, bits)| (*addr, (*width, *bits)))
        .collect();
    // Build a static const double/float VarDecl for a recovered rodata FP constant, or None for a non-finite bit pattern, which is more likely a misclassified bitmask.
    let make_rodata_fp_decl =
        |name: String,
         width: usize,
         bits: u64|
         -> Option<crate::decompile::passes::c_pass::types::TopLevelDecl> {
            use crate::decompile::passes::c_pass::types::{
                FloatLiteral, Initializer, TopLevelDecl,
            };
            let (ty, value, suffix) = if width == 4 {
                (
                    CType::float(),
                    f32::from_bits(bits as u32) as f64,
                    FloatLiteralSuffix::F,
                )
            } else {
                (
                    CType::double(),
                    f64::from_bits(bits),
                    FloatLiteralSuffix::None,
                )
            };
            if !value.is_finite() {
                return None;
            }
            Some(TopLevelDecl::VarDecl(VarDecl {
                name,
                ty,
                storage_class: StorageClass::Static,
                qualifiers: TypeQualifiers {
                    is_const: true,
                    ..TypeQualifiers::none()
                },
                init: Some(Initializer::Expr(CExpr::FloatLit(FloatLiteral {
                    value,
                    suffix,
                }))),
                loc: SourceLoc::unknown(),
            }))
        };

    // Build global pointer sets from rtl_pass analysis
    let global_ptr_ids: HashSet<usize> = db
        .rel_iter::<(Ident,)>("emit_global_is_ptr")
        .map(|(id,)| *id)
        .collect();
    let global_char_ptr_ids: HashSet<usize> = db
        .rel_iter::<(Ident,)>("emit_global_is_char_ptr")
        .map(|(id,)| *id)
        .collect();

    // SR-1: type global declarations from recovered layouts so emit_global_struct_fields keeps the struct definition alive through clight_emit's referenced-filter; catalog guard prevents naming a struct that extract_struct_definitions won't emit (incomplete type at TU end).
    let global_struct_decl_ids: HashMap<usize, usize> = {
        let catalog_ids: HashSet<usize> = db
            .rel_iter::<(u64, usize, usize, usize)>("global_struct_catalog")
            .map(|(_, id, _, _)| *id)
            .collect();
        let mut m: HashMap<usize, usize> = HashMap::new();
        for (ident, sid, _fields) in db
            .rel_iter::<(Ident, usize, Arc<Vec<(i64, Ident, MemoryChunk)>>)>(
                "emit_global_struct_fields",
            )
        {
            if !catalog_ids.contains(sid) {
                continue;
            }
            // Min-sid per ident: relation is set-semantics and may carry several rows; arrival order must not pick the winner.
            m.entry(*ident)
                .and_modify(|e| {
                    if *sid < *e {
                        *e = *sid;
                    }
                })
                .or_insert(*sid);
        }
        m
    };
    // Same for recovered global arrays: smallest (elem_size, count) per ident -- weakest claim, deterministic across the multi-valued relation.
    let global_array_decl: HashMap<usize, (usize, usize)> = {
        let mut m: HashMap<usize, (usize, usize)> = HashMap::new();
        for (ident, elem_size, count) in db.rel_iter::<(Ident, usize, usize)>("is_global_array") {
            if *elem_size == 0 || *count == 0 {
                continue;
            }
            let cand = (*elem_size, *count);
            m.entry(*ident)
                .and_modify(|e| {
                    if cand < *e {
                        *e = cand;
                    }
                })
                .or_insert(cand);
        }
        m
    };
    let ctype_of_chunk = |chunk: &MemoryChunk| -> CType {
        match chunk {
            // c26: MBool is a 1-byte store; emit _Bool rather than widening to a 4-byte int. _Bool is the true storage type and is consistent with the 1-byte chunk_byte_size used for array elements.
            MemoryChunk::MBool => CType::Bool,
            MemoryChunk::MInt32 | MemoryChunk::MAny32 => CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Int,
                crate::decompile::passes::c_pass::types::Signedness::Signed,
            ),
            MemoryChunk::MInt8Signed => CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Char,
                crate::decompile::passes::c_pass::types::Signedness::Signed,
            ),
            MemoryChunk::MInt8Unsigned => CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Char,
                crate::decompile::passes::c_pass::types::Signedness::Unsigned,
            ),
            MemoryChunk::MInt16Signed => CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Short,
                crate::decompile::passes::c_pass::types::Signedness::Signed,
            ),
            MemoryChunk::MInt16Unsigned => CType::Int(
                crate::decompile::passes::c_pass::types::IntSize::Short,
                crate::decompile::passes::c_pass::types::Signedness::Unsigned,
            ),
            MemoryChunk::MFloat32 => {
                CType::Float(crate::decompile::passes::c_pass::types::FloatSize::Float)
            }
            MemoryChunk::MFloat64 => {
                CType::Float(crate::decompile::passes::c_pass::types::FloatSize::Double)
            }
            _ => CType::long(),
        }
    };
    let chunk_byte_size = |chunk: &MemoryChunk| -> usize {
        match chunk {
            MemoryChunk::MBool | MemoryChunk::MInt8Signed | MemoryChunk::MInt8Unsigned => 1,
            MemoryChunk::MInt16Signed | MemoryChunk::MInt16Unsigned => 2,
            MemoryChunk::MInt32 | MemoryChunk::MFloat32 | MemoryChunk::MAny32 => 4,
            MemoryChunk::MInt64 | MemoryChunk::MFloat64 | MemoryChunk::MAny64 => 8,
            MemoryChunk::Unknown => 0,
        }
    };

    // c26: ELF symbol storage size per global, keyed by id since a global's id IS its address; multi-valued, so keep the smallest as the tightest deterministic upper bound.
    let global_sym_size: HashMap<usize, usize> = {
        let mut m: HashMap<usize, usize> = HashMap::new();
        for (addr, size, _name) in db.rel_iter::<(Address, usize, Symbol)>("symbol_size") {
            if *size == 0 {
                continue;
            }
            m.entry(*addr as usize)
                .and_modify(|e| {
                    if *size < *e {
                        *e = *size;
                    }
                })
                .or_insert(*size);
        }
        m
    };
    // c26: smallest recorded load/store chunk per global, the narrowest width actually touched, preferred over global_chunks' LARGEST pick which promotes a 1-byte global to long.
    let global_narrow_chunk: HashMap<usize, MemoryChunk> = {
        let mut groups: HashMap<usize, Vec<MemoryChunk>> = HashMap::new();
        for (id, chunk) in db.rel_iter::<(Ident, MemoryChunk)>("global_load_chunk") {
            groups.entry(*id).or_default().push(*chunk);
        }
        groups
            .into_iter()
            .map(|(id, mut chunks)| {
                let has_specific = chunks.iter().any(|c| {
                    !matches!(
                        c,
                        MemoryChunk::MAny32 | MemoryChunk::MAny64 | MemoryChunk::Unknown
                    )
                });
                if has_specific {
                    chunks.retain(|c| {
                        !matches!(
                            c,
                            MemoryChunk::MAny32 | MemoryChunk::MAny64 | MemoryChunk::Unknown
                        )
                    });
                }
                chunks.sort();
                (id, chunks[0])
            })
            .collect()
    };
    // c26: scalar global type, with st_size authoritative when it is a clean scalar size, else the narrowest recorded chunk and then long; recovers a 1-byte char global instead of widening.
    let scalar_global_ctype = |id: usize| -> CType {
        let narrow = global_narrow_chunk.get(&id).copied();
        if let Some(&size) = global_sym_size.get(&id) {
            if matches!(size, 1 | 2 | 4 | 8) {
                if let Some(chunk) = narrow {
                    if chunk_byte_size(&chunk) == size {
                        return ctype_of_chunk(&chunk);
                    }
                }
                return match size {
                    1 => CType::char_unsigned(),
                    2 => CType::Int(
                        crate::decompile::passes::c_pass::types::IntSize::Short,
                        crate::decompile::passes::c_pass::types::Signedness::Signed,
                    ),
                    4 => CType::int(),
                    _ => CType::long(),
                };
            }
        }
        match narrow {
            Some(chunk) => ctype_of_chunk(&chunk),
            None => CType::long(),
        }
    };

    // Names the emitted functions write or address-take, used to keep a data global defined rather than extern only when this TU owns its storage.
    let mutated_globals: HashSet<String> = {
        let mut m = HashSet::new();
        for stmt in stmt_map.values() {
            collect_mutated_names_in_stmt(stmt, &mut m);
        }
        m
    };

    for global in globals {
        // String literals emit separately and read-only scalars are constant-folded, but a writable .data scalar keeps its storage and must be a real initialized declaration.
        if global.is_string || (global.scalar_value.is_some() && !global.scalar_writable) {
            continue;
        }
        // A recovered rodata FP constant reaches here with no scalar_value, so emit a typed static const with the real bits instead of a loader-zeroed long L_<addr>.
        if let Some(addr) = global
            .name
            .strip_prefix("L_")
            .and_then(|h| u64::from_str_radix(h, 16).ok())
        {
            if let Some(&(width, bits)) = rodata_fp_consts.get(&addr) {
                if let Some(decl) =
                    make_rodata_fp_decl(sanitize_c_symbol_name(&global.name), width, bits)
                {
                    tu.decls.push(decl);
                    continue;
                }
            }
        }
        let sanitized_name = sanitize_c_symbol_name(&global.name);
        // stdin/stdout/stderr are declared by <stdio.h>; emitting our own definition would conflict with the header and shadow libc's real FILE*.
        if is_libc_stdio_global(&global.name) {
            continue;
        }
        let (ty, init) = if !global.pointer_init.is_empty() {
            // Relocated pointer-valued global from ELF ground truth: a single slot is a scalar void*, a run is a pointer table emitted as long name[N] = {...} so the runtime bytes are reconstructed.
            use crate::decompile::passes::c_pass::types::{InitItem, Initializer};
            if global.pointer_init.len() == 1 {
                let e = global.pointer_init[0].clone();
                let init_expr = CExpr::Cast(CType::ptr(CType::Void), Box::new(e));
                (CType::ptr(CType::Void), Some(Initializer::Expr(init_expr)))
            } else {
                let items: Vec<InitItem> = global
                    .pointer_init
                    .iter()
                    .map(|e| InitItem {
                        designator: None,
                        init: Initializer::Expr(CExpr::Cast(CType::long(), Box::new(e.clone()))),
                    })
                    .collect();
                (
                    CType::Array(Box::new(CType::long()), Some(global.pointer_init.len())),
                    Some(Initializer::List(items)),
                )
            }
        } else if let Some(scalar) = global
            .scalar_value
            .as_ref()
            .filter(|_| global.scalar_writable)
        {
            // Writable initialized scalar global: width from the narrowest recorded chunk or ELF symbol size, initializer from the .data bytes, declared here so runtime stores are preserved.
            (
                scalar_global_ctype(global.id),
                Some(crate::decompile::passes::c_pass::types::Initializer::Expr(
                    scalar.to_cexpr(),
                )),
            )
        } else if let Some(known_ty) = known_global_types.get(&sanitized_name) {
            (known_ty.clone(), None)
        } else if let Some(&sid) = global_struct_decl_ids.get(&global.id) {
            // SR-1: naming the struct on the declaration keeps the definition alive through the referenced/needs_layout filter; use-site compatibility is handled by clight_emit's recovered-global rewrite.
            (CType::Struct(format!("struct_{:x}", sid)), None)
        } else if let Some(&(elem_size, count)) = global_array_decl.get(&global.id) {
            // SR-1: non-scalar strides (e.g. 12-byte struct elements) have no honest scalar element type, so declare the full extent as a byte array rather than an undersized `int name[count]` that would let a recompile overlap the following symbol.
            let (elem_ty, decl_count) = match elem_size {
                1 | 2 | 4 | 8 => {
                    let elem_ty = match global_chunks.get(&global.id) {
                        Some(chunk) if chunk_byte_size(chunk) == elem_size => ctype_of_chunk(chunk),
                        _ => match elem_size {
                            1 => CType::Int(
                                crate::decompile::passes::c_pass::types::IntSize::Char,
                                crate::decompile::passes::c_pass::types::Signedness::Unsigned,
                            ),
                            2 => CType::Int(
                                crate::decompile::passes::c_pass::types::IntSize::Short,
                                crate::decompile::passes::c_pass::types::Signedness::Signed,
                            ),
                            8 => CType::long(),
                            _ => CType::int(),
                        },
                    };
                    (elem_ty, count)
                }
                _ => (
                    CType::Int(
                        crate::decompile::passes::c_pass::types::IntSize::Char,
                        crate::decompile::passes::c_pass::types::Signedness::Unsigned,
                    ),
                    elem_size * count,
                ),
            };
            (CType::Array(Box::new(elem_ty), Some(decl_count)), None)
        } else if global_char_ptr_ids.contains(&global.id) {
            (
                CType::Pointer(
                    Box::new(CType::Int(
                        crate::decompile::passes::c_pass::types::IntSize::Char,
                        crate::decompile::passes::c_pass::types::Signedness::Signed,
                    )),
                    TypeQualifiers::none(),
                ),
                None,
            )
        } else if global.is_pointer || global_ptr_ids.contains(&global.id) {
            (CType::ptr(CType::Void), None)
        } else {
            // c26: scalar global. Width comes from st_size / the narrowest recorded chunk (scalar_global_ctype), not the widened global_chunks pick, so a 1-byte global stays _Bool/char instead of becoming int/long.
            (scalar_global_ctype(global.id), None)
        };

        // A header-declared global VARIABLE must use the header's canonical type, since the declaration cannot be suppressed and any mismatch is a hard error; initialized data keeps the recovered type.
        let ty = match crate::decompile::passes::c_pass::header_db::header_db()
            .variables
            .get(sanitized_name.as_str())
        {
            Some(spelling) if init.is_none() => parse_canonical_var_type(spelling).unwrap_or(ty),
            _ => ty,
        };

        // A content-less scalar this TU only reads is emitted extern ONLY when a linked library actually defines it (curated gnulib/CRT globals or real-named libc globals); everything else keeps its own definition.
        let is_lib = db.is_lib_global(&global.name)
            || (uses_real_libc_name(&sanitized_name) && !is_libc_stdio_global(&sanitized_name));
        let (storage_class, init) = if is_lib {
            (StorageClass::Extern, None)
        } else {
            (StorageClass::default(), init)
        };
        tu.add_global_var(VarDecl {
            name: sanitized_name,
            ty,
            storage_class,
            qualifiers: TypeQualifiers::none(),
            init,
            loc: SourceLoc::unknown(),
        });
    }

    // instr_in_function is multi-valued (shared PLT/thunk nodes); keep the smallest owning address so the node->function mapping is identical across runs.
    let mut node_to_func: HashMap<crate::x86::types::Node, crate::x86::types::Address> =
        HashMap::new();
    for (node, func) in db.rel_iter::<(Node, Address)>("instr_in_function") {
        node_to_func
            .entry(*node)
            .and_modify(|e| {
                if *func < *e {
                    *e = *func;
                }
            })
            .or_insert(*func);
    }

    let recovered_func_names = recover_func_names_from_assert(db);

    let string_map = crate::decompile::passes::c_pass::helpers::build_string_literal_map(db);

    let known_func_names: HashMap<String, usize> = selected_functions
        .iter()
        .map(|f| (sanitize_c_symbol_name(&f.name), f.param_regs.len()))
        .chain(
            id_to_name
                .values()
                .filter(|name| !is_generated_label(name))
                .map(|name| (sanitize_c_symbol_name(name), usize::MAX)),
        )
        .collect();

    let local_def_policy = header_collision_policy(selected_functions);
    let func_renames = &local_def_policy.renames;

    // Loop-structure facts for top-level ordering, keyed by synth-masked header to member addresses, keeping a rotated loop's body contiguous and ahead of its exit instead of address tie-breaking.
    const ORDER_SYNTH_MASK: u64 = !((1u64 << 62) | (1u64 << 63));
    // loop_members must be keyed through the SAME normalization order_nodes_dfs uses (sseq-head resolve then synth-mask), or a header folded into an sseq bundle reads as loop-free and its body orders after its exit.
    let order_sseq_member_to_head: HashMap<crate::x86::types::Node, crate::x86::types::Node> = {
        let mut m: HashMap<crate::x86::types::Node, crate::x86::types::Node> = HashMap::new();
        for f in selected_functions {
            for (&head, members) in &f.sseq_groups {
                for &mem in members {
                    if mem != head {
                        m.entry(mem)
                            .and_modify(|h| {
                                if head < *h {
                                    *h = head;
                                }
                            })
                            .or_insert(head);
                    }
                }
            }
        }
        m
    };
    let norm_order_id = |n: crate::x86::types::Node| -> crate::x86::types::Node {
        order_sseq_member_to_head.get(&n).copied().unwrap_or(n) & ORDER_SYNTH_MASK
    };
    let loop_members: HashMap<crate::x86::types::Node, HashSet<crate::x86::types::Node>> = {
        let mut m: HashMap<crate::x86::types::Node, HashSet<crate::x86::types::Node>> =
            HashMap::new();
        for (_func, header, member) in db.rel_iter::<(Address, Node, Node)>("loop_body") {
            let h = norm_order_id(*header);
            let mem = norm_order_id(*member);
            m.entry(h).or_default().insert(mem);
        }
        m
    };
    // Innermost natural loop per normalized node, memoized once from loop_members (smallest body, ties on header address), turning order_nodes_dfs's per-node rescan into an O(1) lookup.
    let node_innermost_loop: HashMap<crate::x86::types::Node, crate::x86::types::Node> = {
        let mut best: HashMap<crate::x86::types::Node, (usize, crate::x86::types::Node)> =
            HashMap::new();
        for (header, members) in &loop_members {
            let cand = (members.len(), *header);
            for &node in members {
                best.entry(node)
                    .and_modify(|cur| {
                        if cand < *cur {
                            *cur = cand;
                        }
                    })
                    .or_insert(cand);
            }
        }
        best.into_iter()
            .map(|(node, (_, header))| (node, header))
            .collect()
    };

    let mut emitted_func_names: HashSet<String> = HashSet::new();
    for func in selected_functions {
        let mut scoped_function_object_types = function_object_types
            .get(&func.address)
            .cloned()
            .unwrap_or_default();
        let params: Vec<FuncParam> = func
            .param_regs
            .iter()
            .zip(func.param_types.iter())
            .map(|(reg, pty)| {
                let name = param_name_for_reg(*reg);
                FuncParam::named(name, convert_param_type_from_param(pty))
            })
            .collect();

        let param_name_set: HashSet<String> =
            params.iter().filter_map(|p| p.name.clone()).collect();

        let func_addr = func.address;

        let nodes_set: HashSet<_> = stmt_map
            .keys()
            .copied()
            .filter(|n| {
                func.statements.contains_key(n)
                    || node_to_func.get(n) == Some(&func_addr)
                    || optimized_node_to_func.get(n) == Some(&func_addr)
            })
            .collect();

        // sseq bundling drops members whose clight_succ edges still point at them, so remap member->synth edges to the head; min-head-wins is order-independent, since a member can sit in two bundles.
        let mut sseq_member_to_head: HashMap<crate::x86::types::Node, crate::x86::types::Node> =
            HashMap::new();
        for (&head, members) in &func.sseq_groups {
            for &m in members {
                if m != head {
                    sseq_member_to_head
                        .entry(m)
                        .and_modify(|h| {
                            if head < *h {
                                *h = head;
                            }
                        })
                        .or_insert(head);
                }
            }
        }
        const SYNTH_BIT: u64 = 1u64 << 62;
        let remapped_edges: Vec<(crate::x86::types::Node, crate::x86::types::Node)> = edges
            .iter()
            .flat_map(|&(s, d)| {
                let mut out: Vec<(crate::x86::types::Node, crate::x86::types::Node)> = vec![(s, d)];
                let dst_is_synth = (d & SYNTH_BIT) != 0;
                let src_is_synth = (s & SYNTH_BIT) != 0;
                // synth-targeted edge from a bundle member: also publish from the head
                if dst_is_synth {
                    if let Some(&head) = sseq_member_to_head.get(&s) {
                        if head != s {
                            out.push((head, d));
                        }
                    }
                }
                // synth-source edge to an external (non-bundle) consumer: keep as-is (synth nodes aren't bundle members, so no remap needed for src).
                let _ = src_is_synth;
                out
            })
            .filter(|(s, d)| s != d)
            .collect();

        let entry_node = func.address as crate::x86::types::Node;
        // Normalize a top-level node id to the address loop_body is keyed by: resolve an sseq member to its bundle head, then mask both synthetic-address bits so synth nodes land on their base.
        let order_normalize = |n: crate::x86::types::Node| -> crate::x86::types::Node {
            let resolved = sseq_member_to_head.get(&n).copied().unwrap_or(n);
            resolved & ORDER_SYNTH_MASK
        };
        let nodes = order_nodes_dfs(
            entry_node,
            &nodes_set,
            &remapped_edges,
            |n| stmt_map.get(&n).map_or(false, is_unconditional_exit),
            &loop_members,
            &node_innermost_loop,
            order_normalize,
        );
        if let Ok(pat) = std::env::var("MANIFOLD_PR_DUMP") {
            if !pat.is_empty() && func.name.contains(&pat) {
                eprintln!(
                    "[PR-NODES] fn {} ({:#x}) n={}: {:x?}",
                    func.name,
                    func_addr,
                    nodes.len(),
                    nodes
                );
            }
        }

        let mut body_items = Vec::new();
        let mut body_terminated = false;
        for &node in &nodes {
            if let Some(stmt) = stmt_map.get(&node) {
                if matches!(stmt, CStmt::Empty) {
                    continue;
                }
                if body_terminated {
                    // After unconditional exit, only keep statements with named labels (goto targets).
                    if contains_named_label(stmt) {
                        body_items.push(CBlockItem::Stmt(stmt.clone()));
                        // The labeled section might end with a terminator, so re-check.
                        body_terminated = is_unconditional_exit(stmt);
                    }
                } else {
                    body_items.push(CBlockItem::Stmt(stmt.clone()));
                    if is_unconditional_exit(stmt) {
                        body_terminated = true;
                    }
                }
            }
        }

        let _before_count = body_items.len();
        body_items = simplify_fallthrough_gotos_in_block(body_items);

        for (i, item) in body_items.iter().enumerate() {
            if i + 1 < body_items.len() {
                if let CBlockItem::Stmt(CStmt::If(_, then_s, Some(else_s))) = item {
                    if let CBlockItem::Stmt(next_stmt) = &body_items[i + 1] {
                        if let Some(next_label) = get_first_label_from_stmt(next_stmt) {
                            if let CStmt::Goto(else_target) = &**else_s {
                                if normalize_label(else_target) == normalize_label(&next_label) {
                                    debug!("[EMIT-UNSIMPLIFIED] else goto {} should have been simplified to next label {}",
                                        else_target, next_label);
                                }
                            }
                            if let CStmt::Goto(then_target) = &**then_s {
                                if normalize_label(then_target) == normalize_label(&next_label) {
                                    debug!("[EMIT-UNSIMPLIFIED] then goto {} should have been simplified to next label {}",
                                        then_target, next_label);
                                }
                            }
                        }
                    }
                }
            }
        }

        let expected_stmts = func.statements.len();
        let actual_leaf_count = count_leaf_stmts_in_block(&body_items);
        let is_incomplete =
            body_items.is_empty() || (expected_stmts > 4 && actual_leaf_count * 2 < expected_stmts);
        if is_incomplete {
            let covered_nodes: HashSet<crate::x86::types::Node> = nodes.iter().copied().collect();

            let mut fb_ctx = ConversionContext::new(id_to_name.clone());
            fb_ctx.enter_function(func.address, local_evar_ids.get(&func.address));
            fb_ctx.func_renames = local_def_policy.renames.clone();
            {
                let mut addr_name_map: HashMap<u64, String> = HashMap::new();
                for (addr, name, _entry_node) in
                    db.rel_iter::<(Address, Symbol, Node)>("emit_function")
                {
                    addr_name_map.insert(*addr as u64, sanitize_c_symbol_name(name));
                }
                for f in selected_functions {
                    addr_name_map.insert(f.address as u64, sanitize_c_symbol_name(&f.name));
                }
                let mut sorted: Vec<(u64, String)> = addr_name_map.into_iter().collect();
                for (_, name) in sorted.iter_mut() {
                    if let Some(renamed) = func_renames.get(name) {
                        *name = renamed.clone();
                    }
                }
                sorted.sort_by_key(|(addr, _)| *addr);
                fb_ctx.func_addrs_sorted = sorted;
            }
            let mut fb_nodes: Vec<_> = func
                .statements
                .keys()
                .copied()
                .filter(|n| !covered_nodes.contains(n))
                .collect();
            fb_nodes.sort();
            for node in fb_nodes {
                if let Some(cl_stmt) = func.statements.get(&node) {
                    let cstmt = convert_stmt(cl_stmt, &mut fb_ctx);
                    let cstmt =
                        crate::decompile::passes::c_pass::helpers::map_stmt_exprs(&cstmt, &|e| {
                            crate::decompile::passes::c_pass::helpers::inline_string_literals(
                                e,
                                &string_map,
                            )
                        });
                    let cstmt = strip_trivial_casts(&cstmt, &fb_ctx.var_types);
                    if !matches!(cstmt, CStmt::Empty) {
                        body_items.push(CBlockItem::Stmt(cstmt));
                    }
                }
            }
            if let Some(fallback_types) = fb_ctx.function_object_types.get(&func.address) {
                for (name, ty) in fallback_types {
                    scoped_function_object_types
                        .entry(name.clone())
                        .or_insert_with(|| ty.clone());
                }
            }
            for (name, ty) in fb_ctx.var_types {
                func_var_types.entry(name).or_insert(ty);
            }
        }

        rewrite_tailcall_gotos(&mut body_items, &known_func_names, &func.name);

        ensure_goto_label_consistency(&mut body_items);

        deduplicate_labels(&mut body_items);

        strip_dead_labels_in_block(&mut body_items);

        let body = if body_items.is_empty() {
            continue;
        } else if body_items.len() == 1 {
            match body_items.remove(0) {
                CBlockItem::Stmt(s) => s,
                other => CStmt::Block(vec![other]),
            }
        } else {
            CStmt::Block(body_items)
        };

        let mut body = crate::decompile::passes::c_pass::helpers::flatten_blocks_and_cleanup(&body);
        body = eliminate_dead_code(&body);

        simplify_xor_self_in_stmt(&mut body);
        strip_dead_expr_stmts(&mut body);

        // Per-function local variable types seeded from clight_select's refined candidates, built BEFORE strip_trivial_casts so a (long)float_var is not reduced to a bare float under a pointer cast.
        let mut local_var_types = func_var_types.clone();
        for (reg, candidates) in &func.var_type_candidates {
            let idx = func.var_decl_idx.get(reg).copied().unwrap_or(0);
            if let Some(type_str) = candidates.get(idx).or_else(|| func.var_types.get(reg)) {
                let var_name = crate::decompile::passes::c_pass::helpers::param_name_for_reg(*reg);
                let ctype =
                    crate::decompile::passes::c_pass::helpers::xtype_string_to_ctype(type_str);
                local_var_types.insert(var_name, ctype);
            }
        }

        body = strip_trivial_casts(&body, &local_var_types);
        body = forward_return_value(&body);

        let is_reconciled_void = db
            .rel_iter::<(Address,)>("emit_function_void")
            .any(|&(addr,)| addr == func.address);

        // Return type from signature_pass; the relation is multi-valued, so reduce with the same .min() query.rs uses and every consumer selects the identical type.
        let return_type = if is_reconciled_void {
            CType::Void
        } else if let Some(xtype) = db
            .rel_iter::<(Address, XType)>("emit_function_return_type_xtype")
            .filter(|(a, _)| *a == func.address)
            .map(|(_, t)| *t)
            .min()
        {
            convert_xtype(&xtype)
        } else {
            convert_clight_type(&func.return_type)
        };

        if is_reconciled_void {
            strip_return_values_in_stmt(&mut body);
        }
        if !matches!(return_type, CType::Void) {
            fix_bare_returns_in_stmt(&mut body);
            // Repair a shared-return-merge orphan: a value-returning function never legitimately ends in a fall-through, so append the body's own tail return to a trailing non-diverging block.
            fix_falloff_return_orphan(&mut body);
        }

        if is_dead_expr_stmt(&body) {
            continue;
        }

        // local_var_types was seeded from func.var_type_candidates above; continue refining with decl_solve overrides, usage inference, and stack buffers.

        // Phase 5 decl authority (pointerness): a register the solve typed int_* but that is dereferenced or stored through is really a pointer, seeded void * BEFORE usage inference so the pointee refines.
        if let Some(forced) = db.decl_solve_force_ptr_regs.get(&func.address) {
            for reg in forced {
                let var_name = crate::decompile::passes::c_pass::helpers::param_name_for_reg(*reg);
                local_var_types.insert(var_name, CType::ptr(CType::Void));
            }
        }

        infer_var_types_from_usage(&body, &mut local_var_types, &extern_return_types);

        // Phase 5 decl authority (force_long_registers): registers proven integer by an integer-only operator override the solver and usage inference, the register analogue of field force-long.
        if let Some(forced) = db.decl_solve_force_long_regs.get(&func.address) {
            for reg in forced {
                let var_name = crate::decompile::passes::c_pass::helpers::param_name_for_reg(*reg);
                local_var_types.insert(var_name, CType::long());
            }
        }

        // Address-taken stack regions sized as a byte buffer are declared unsigned char[N], so raw byte-offset stores stay in-bounds and (long)var / &var stay valid via array decay.
        if let Some(buffers) = db.stack_struct_buffers.get(&func.address) {
            for (reg, size) in buffers {
                let var_name = crate::decompile::passes::c_pass::helpers::param_name_for_reg(*reg);
                local_var_types.insert(
                    var_name,
                    CType::Array(Box::new(CType::char_unsigned()), Some(*size as usize)),
                );
            }
        }

        // Runtime-indexed stack arrays declare with the element type so arr + i strides by element width; a stack_struct_buffers byte-buffer entry is more specific and wins.
        if let Some(arrays) = db.stack_array_buffers.get(&func.address) {
            for (reg, (chunk, count)) in arrays {
                let var_name = crate::decompile::passes::c_pass::helpers::param_name_for_reg(*reg);
                if matches!(local_var_types.get(&var_name), Some(CType::Array(..))) {
                    continue;
                }
                let elem = ctype_from_memchunk(chunk);
                local_var_types.insert(var_name, CType::Array(Box::new(elem), Some(*count)));
            }
        }

        // An actual local Scall target must have a callable declaration.  This
        // evidence is scoped by function and call position, so it safely wins
        // even when scalar/storage heuristics classified the same register.
        retain_clight_function_object_types(
            &body,
            &scoped_function_object_types,
            &mut local_var_types,
        );

        // A scalar-promoted store to an array local's base is illegal C, so reconcile it into the matching element store var[0] = val; keyed on the array-typed local with a bare-Var LHS.
        let array_locals: HashSet<String> = local_var_types
            .iter()
            .filter_map(|(n, t)| matches!(t, CType::Array(..)).then(|| n.clone()))
            .collect();
        if !array_locals.is_empty() {
            body = crate::decompile::passes::c_pass::helpers::map_stmt_exprs(&body, &|e| {
                if let CExpr::Assign(op, lhs, rhs) = e {
                    if let CExpr::Var(name) = lhs.as_ref() {
                        if array_locals.contains(name) {
                            let elem_lhs = CExpr::Index(
                                Box::new(CExpr::Var(name.clone())),
                                Box::new(CExpr::int(0)),
                            );
                            return Some(CExpr::Assign(*op, Box::new(elem_lhs), rhs.clone()));
                        }
                    }
                }
                None
            });
        }

        let bool_vars = detect_bool_variables(&body);

        let mut params: Vec<FuncParam> = params
            .into_iter()
            .map(|mut p| {
                if let Some(name) = &p.name {
                    if p.ty == CType::long() {
                        if let Some(inferred_ty) = local_var_types.get(name) {
                            p.ty = inferred_ty.clone();
                        } else if bool_vars.contains(name.as_str()) {
                            p.ty = CType::int();
                        }
                    }
                }
                p
            })
            .collect();

        // const-qualify a read-only char * parameter whose name is never a write target in the emitted body, sound because the mutated-name set comes from the same body that will be compiled.
        {
            let mut mutated_names: HashSet<String> = HashSet::new();
            collect_mutated_names_in_stmt(&body, &mut mutated_names);
            let const_char_ptr = CType::Pointer(
                Box::new(CType::Qualified(
                    Box::new(CType::char_signed()),
                    crate::decompile::passes::c_pass::types::TypeQualifiers {
                        is_const: true,
                        is_volatile: false,
                        is_restrict: false,
                    },
                )),
                crate::decompile::passes::c_pass::types::TypeQualifiers::none(),
            );
            for p in &mut params {
                if let Some(name) = &p.name {
                    if p.ty == CType::ptr(CType::char_signed()) && !mutated_names.contains(name) {
                        p.ty = const_char_ptr.clone();
                    }
                }
            }
        }

        // Compilability: now that var/param C types are final, cast to long the pointer operands of any arithmetic C forbids (ptr*int, ptr+ptr, ...); runs before VarRenamer so type map keys still match the variable names used in `body`.
        let body = {
            let mut arith_types = local_var_types.clone();
            for p in &params {
                if let Some(name) = &p.name {
                    arith_types.insert(name.clone(), p.ty.clone());
                }
            }
            repair_arith_stmt(&body, &arith_types)
        };

        let mut local_names = HashSet::new();
        collect_var_names_from_stmt(&body, &mut local_names);
        // __builtin_* markers are never locals: synthesizing one as a zero-init local lets the optimizer fold the guard it protects into if (0), so they are declared at file scope instead.
        local_names.retain(|n| {
            !param_name_set.contains(n) && !global_names.contains(n) && !n.starts_with("__builtin_")
        });

        let mut local_vars: Vec<VarDecl> = local_names
            .into_iter()
            .map(|name| {
                let ty = local_var_types
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(CType::int);
                VarDecl {
                    name,
                    ty,
                    storage_class: StorageClass::default(),
                    qualifiers: TypeQualifiers::none(),
                    init: None,
                    loc: SourceLoc::unknown(),
                }
            })
            .collect();
        local_vars.sort_by(|a, b| a.name.cmp(&b.name));

        if std::env::var("DET_FP").is_ok() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let body_str = crate::decompile::passes::c_pass::print::print_stmt(&body);
            let mut lv: Vec<String> = local_vars.iter().map(|v| format!("{:?}", v)).collect();
            lv.sort();
            let mut h = DefaultHasher::new();
            body_str.hash(&mut h);
            lv.join(",").hash(&mut h);
            eprintln!(
                "[FP-PR] {:016x} {} {:016x}",
                func_addr,
                func.name,
                h.finish()
            );
        }
        if let Ok(pat) = std::env::var("MANIFOLD_PR_DUMP") {
            if !pat.is_empty() && func.name.contains(&pat) {
                let body_str = crate::decompile::passes::c_pass::print::print_stmt(&body);
                let mut lv: Vec<String> = local_vars.iter().map(|v| format!("{:?}", v)).collect();
                lv.sort();
                eprintln!(
                    "[PR-DUMP] fn {} ({:#x}) locals:\n{}\n[PR-DUMP] body:\n{}",
                    func.name,
                    func_addr,
                    lv.join("\n"),
                    body_str
                );
            }
        }
        let mut renamer = VarRenamer::new();
        for (i, p) in params.iter().enumerate() {
            if let Some(name) = &p.name {
                let dwarf_name = dwarf_param_names.get(&(func_addr, i)).map(|s| s.as_str());
                renamer.seed_param(name, i, dwarf_name);
            }
        }
        // Reserve every pass-through (non-compacted) name first so counter-based compaction of synthetic regs cannot collide with a literal short var_N.
        for v in &local_vars {
            renamer.reserve(&v.name);
        }
        for v in &local_vars {
            renamer.ensure_mapping(&v.name);
        }
        let body = StmtTransform::transform_stmt(&mut renamer, body);
        let params: Vec<FuncParam> = params
            .iter()
            .map(|p| {
                let new_name = p.name.as_ref().map(|n| renamer.rename(n));
                FuncParam::new(new_name, p.ty.clone())
            })
            .collect();
        let local_vars: Vec<VarDecl> = local_vars
            .into_iter()
            .map(|mut v| {
                v.name = renamer.rename(&v.name);
                v
            })
            .collect();

        // Use-before-def guard: a local that is read but never assigned, address-taken or inc/dec'd has no producer, so zero-initialize it; a defensive last resort, with the real fixes upstream.
        let mut local_origin_names: HashSet<String> = HashSet::new();
        for p in &params {
            if let Some(n) = &p.name {
                local_origin_names.insert(n.clone());
            }
        }
        collect_var_origins_stmt(&body, &mut local_origin_names);
        let local_vars: Vec<VarDecl> = local_vars
            .into_iter()
            .map(|mut v| {
                if v.init.is_none() && !local_origin_names.contains(&v.name) {
                    v.init = zero_initializer_for_var(&v.ty);
                }
                v
            })
            .collect();

        let func_name = if func.name.starts_with("FUN_") {
            recovered_func_names
                .get(&func.address)
                .cloned()
                .unwrap_or_else(|| func.name.clone())
        } else {
            func.name.clone()
        };
        let func_name = sanitize_c_symbol_name(&func_name);
        // Header-collision policy: a filtered definition is dropped entirely and its call sites bind to the system header, while a kept one is renamed <name>_local to match its converted call sites.
        if local_def_policy.filtered.contains(&func_name) {
            continue;
        }
        let func_name = func_renames.get(&func_name).cloned().unwrap_or(func_name);

        if !emitted_func_names.insert(func_name.clone()) {
            continue;
        }

        if has_cold_suffix(&func_name) && is_trivial_cold_body(&body) {
            continue;
        }

        let func_def = FuncDef {
            name: func_name,
            return_type,
            params,
            is_variadic: false,
            storage_class: StorageClass::default(),
            body,
            local_vars,
            loc: SourceLoc::unknown(),
        };

        if std::env::var("DET_FP").is_ok() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let body_str = crate::decompile::passes::c_pass::print::print_stmt(&func_def.body);
            let mut lv: Vec<String> = func_def
                .local_vars
                .iter()
                .map(|v| format!("{:?}", v))
                .collect();
            lv.sort();
            let mut h = DefaultHasher::new();
            body_str.hash(&mut h);
            lv.join(",").hash(&mut h);
            eprintln!(
                "[FP-E] {:016x} {} {:016x}",
                func_addr,
                func_def.name,
                h.finish()
            );
        }
        tu.add_function(func_def);
    }

    // Suppress our declaration only when an emitted header or a compiler
    // builtin really supplies one. A data-driven header:null entry has no
    // external declaration and must follow the ordinary compatible-decl path.
    let is_compiler_provided = function_declaration_is_externally_provided;

    // This is the sole typed-declaration authority for loader-owned and
    // ordinary curated externs.  Its projection already vetoes address/kind,
    // sanitizer, signature, and variadic conflicts.
    let known_loader_signatures = known_loader_signatures_from_db(db);
    let exact_import_pointer_object_names = exact_import_pointer_call_names(db);

    let mut resolved_extern_names = BTreeSet::new();
    for (name, _, _, _) in
        db.rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>("resolved_extern_signature")
    {
        resolved_extern_names.insert(sanitize_c_symbol_name(name));
    }
    for sanitized_name in resolved_extern_names {
        if emitted_func_names.contains(&sanitized_name)
            || is_compiler_provided(&sanitized_name)
            || exact_import_pointer_object_names.contains(&sanitized_name)
        {
            continue;
        }
        let Some((ret_type, param_types, variadic)) =
            known_loader_signatures.get(&sanitized_name)
        else {
            continue;
        };
        let ret_ctype = convert_xtype(ret_type);
        let params: Vec<FuncParam> = param_types
            .iter()
            .enumerate()
            .map(|(i, t)| FuncParam::named(format!("arg{}", i), convert_xtype(t)))
            .collect();

        let mut decl = crate::decompile::passes::c_pass::types::FuncDecl::new(
            sanitized_name,
            ret_ctype,
            params,
        );
        decl.is_variadic = *variadic;
        tu.add_func_decl(decl);
    }

    // IL-1: join per-call-site argument type evidence (width+class lattice, see ArgEvidence) to produce typed prototypes for called-but-not-emitted functions when every call site agrees; K&R fallback covers the rest.
    let joined_call_sigs: HashMap<String, Vec<CType>> = {
        // Callee return types and global variable types visible to the compiler (definitions first, then resolved externs/globals already in the TU).
        let mut callee_ret: HashMap<String, CType> = HashMap::new();
        let mut global_types: HashMap<String, CType> = HashMap::new();
        for decl in tu.decls.iter() {
            match decl {
                crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDef(f) => {
                    callee_ret.insert(f.name.clone(), f.return_type.clone());
                    global_types.insert(
                        f.name.clone(),
                        CType::Function(
                            Box::new(f.return_type.clone()),
                            f.params.iter().map(|param| param.ty.clone()).collect(),
                            f.is_variadic,
                            false,
                        ),
                    );
                }
                crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDecl(d) => {
                    callee_ret
                        .entry(d.name.clone())
                        .or_insert_with(|| d.return_type.clone());
                    global_types.entry(d.name.clone()).or_insert_with(|| {
                        CType::Function(
                            Box::new(d.return_type.clone()),
                            d.params.iter().map(|param| param.ty.clone()).collect(),
                            d.is_variadic,
                            d.unspecified_params,
                        )
                    });
                }
                crate::decompile::passes::c_pass::types::TopLevelDecl::VarDecl(v) => {
                    global_types.insert(v.name.clone(), v.ty.clone());
                }
                _ => {}
            }
        }
        let mut call_evidence: HashMap<String, Vec<Vec<ArgEvidence>>> = HashMap::new();
        for decl in tu.decls.iter() {
            if let crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDef(fdef) = decl {
                let mut local_types: HashMap<String, CType> = HashMap::new();
                for p in &fdef.params {
                    if let Some(n) = &p.name {
                        local_types.insert(n.clone(), p.ty.clone());
                    }
                }
                for v in &fdef.local_vars {
                    local_types.insert(v.name.clone(), v.ty.clone());
                }
                let env = ArgEvidenceEnv {
                    local_types,
                    global_types: &global_types,
                    callee_ret: &callee_ret,
                };
                collect_call_arg_evidence_in_stmt(&fdef.body, &env, &mut call_evidence);
            }
        }
        join_call_site_evidence_by_loader_identity(db, &call_evidence)
    };
    // Recovered return type per callee name, most-refined candidate winning to match clight_select's pick, so a skip-listed internal declares its true return width instead of int.
    let recovered_ret_by_name: HashMap<String, XType> = {
        let prio = |t: &XType| {
            (
                crate::decompile::passes::clight_pass::xtype_refine_priority(t),
                *t,
            )
        };
        let mut by_addr: HashMap<Address, XType> = HashMap::new();
        for (addr, xt) in db.rel_iter::<(Address, XType)>("emit_function_return_type_xtype") {
            by_addr
                .entry(*addr)
                .and_modify(|cur| {
                    if prio(xt) > prio(cur) {
                        *cur = *xt;
                    }
                })
                .or_insert(*xt);
        }
        let mut by_name: HashMap<String, XType> = HashMap::new();
        for (addr, name, _node) in db.rel_iter::<(Address, Symbol, Node)>("emit_function") {
            if let Some(xt) = by_addr.get(addr) {
                by_name.insert(sanitize_c_symbol_name(name), *xt);
            }
        }
        by_name
    };

    // Curated signatures are authoritative and keyed by every exact sanitized
    // loader identity.  Shared call-site inference is consulted only for names
    // with no such declaration.
    let shared_fixed_arities = infer_shared_fixed_arities_from_db(db);
    let known_variadic_names: HashSet<String> = known_loader_signatures
        .iter()
        .filter_map(|(name, (_, _, variadic))| (*variadic).then_some(name.clone()))
        .collect();
    let known_sig_by_name: HashMap<String, (XType, Arc<Vec<XType>>)> =
        known_loader_signatures
            .into_iter()
            .map(|(name, (ret, params, _))| (name, (ret, params)))
            .collect();
    let known_decl = |name: &str| -> Option<crate::decompile::passes::c_pass::types::FuncDecl> {
        let (ret, param_tys) = known_sig_by_name.get(name)?;
        let params: Vec<FuncParam> = param_tys
            .iter()
            .enumerate()
            .map(|(i, t)| FuncParam::named(format!("arg{}", i), convert_xtype(t)))
            .collect();
        let mut decl = crate::decompile::passes::c_pass::types::FuncDecl::new(
            name.to_string(),
            convert_xtype(ret),
            params,
        );
        decl.is_variadic = known_variadic_names.contains(name);
        Some(decl)
    };

    // Typed prototype from joined call-site evidence; None when evidence is missing, arity disagrees, any position is Poison, or the callee is a known variadic.
    let evidence_decl = |name: &str| -> Option<crate::decompile::passes::c_pass::types::FuncDecl> {
        if is_known_variadic_fn(name) {
            return None;
        }
        let param_tys = joined_call_sigs.get(name)?;
        let params: Vec<FuncParam> = param_tys
            .iter()
            .enumerate()
            .map(|(i, t)| FuncParam::named(format!("arg{}", i), t.clone()))
            .collect();
        // Return type from the recovered signature (see recovered_ret_ctype); `int` only when unknown.
        Some(crate::decompile::passes::c_pass::types::FuncDecl::new(
            name.to_string(),
            recovered_ret_ctype(&recovered_ret_by_name, name),
            params,
        ))
    };
    let shared_evidence_decl =
        |name: &str| -> Option<crate::decompile::passes::c_pass::types::FuncDecl> {
            let arity = *shared_fixed_arities.get(name)?;
            let decl = evidence_decl(name)?;
            (decl.params.len() == arity).then_some(decl)
        };

    let resolved_extern_names: HashSet<String> = db
        .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>("resolved_extern_signature")
        .map(|(n, _, _, _)| sanitize_c_symbol_name(n))
        .collect();
    let mut unknown_extern_names: Vec<String> = db
        .rel_iter::<(Symbol,)>("unknown_extern")
        .map(|(n,)| sanitize_c_symbol_name(n))
        .collect();
    unknown_extern_names.sort();
    unknown_extern_names.dedup();
    for sanitized_name in unknown_extern_names {
        if emitted_func_names.contains(&sanitized_name)
            || is_compiler_provided(&sanitized_name)
            || exact_import_pointer_object_names.contains(&sanitized_name)
        {
            continue;
        }
        if resolved_extern_names.contains(&sanitized_name) {
            continue;
        }

        // A curated extern gets its authoritative typed prototype.  An
        // otherwise unknown shared callee gets a fixed prototype only from an
        // anchored coherent cohort; conflicts and single-site observations
        // retain a K&R declaration and their per-site argument evidence.
        let ret_ctype = recovered_ret_ctype(&recovered_ret_by_name, &sanitized_name);
        let decl = if known_sig_by_name.contains_key(&sanitized_name) {
            known_decl(&sanitized_name)
                .or_else(|| evidence_decl(&sanitized_name))
                .unwrap_or_else(|| {
                    crate::decompile::passes::c_pass::types::FuncDecl::new_unspecified(
                        sanitized_name.clone(),
                        ret_ctype.clone(),
                    )
                })
        } else {
            shared_evidence_decl(&sanitized_name).unwrap_or_else(|| {
                crate::decompile::passes::c_pass::types::FuncDecl::new_unspecified(
                    sanitized_name.clone(),
                    ret_ctype.clone(),
                )
            })
        };
        tu.add_func_decl(decl);
    }

    // Emit forward declarations for all called but undeclared/undefined functions to eliminate "undeclared function" clang errors.
    {
        let mut called_funcs: HashSet<String> = HashSet::new();
        for decl in tu.decls.iter() {
            if let crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDef(fdef) = decl {
                let local_names: HashSet<String> = fdef
                    .params
                    .iter()
                    .filter_map(|param| param.name.clone())
                    .chain(fdef.local_vars.iter().map(|var| var.name.clone()))
                    .collect();
                collect_nonlocal_called_names_in_stmt(
                    &fdef.body,
                    &local_names,
                    &mut called_funcs,
                );
            }
        }
        let declared: HashSet<String> = tu.symbols.keys().cloned().collect();
        // Collect forward declarations and insert them before function definitions so they're visible at the point of use.
        let mut forward_decls = Vec::new();
        let mut sorted_called_funcs: Vec<&String> = called_funcs.iter().collect();
        sorted_called_funcs.sort();
        for name in sorted_called_funcs {
            let is_label = name.starts_with("L_")
                || (name.starts_with('L')
                    && name.len() > 1
                    && name[1..].chars().all(|c| c.is_ascii_hexdigit()));
            if !declared.contains(name)
                && !is_label
                && !is_compiler_provided(name)
                && !emitted_func_names.contains(name)
                && !exact_import_pointer_object_names.contains(name)
            {
                // Called-but-undeclared external: a curated one keeps its typed prototype, an unknown one gets a K&R name(); guarded to externals, with the return type from the recovered signature.
                let ret_ctype = recovered_ret_ctype(&recovered_ret_by_name, name);
                let decl = if known_sig_by_name.contains_key(name) {
                    known_decl(name)
                        .or_else(|| evidence_decl(name))
                        .unwrap_or_else(|| {
                            crate::decompile::passes::c_pass::types::FuncDecl::new_unspecified(
                                name.clone(),
                                ret_ctype.clone(),
                            )
                        })
                } else {
                    shared_evidence_decl(name).unwrap_or_else(|| {
                        crate::decompile::passes::c_pass::types::FuncDecl::new_unspecified(
                            name.clone(),
                            ret_ctype.clone(),
                        )
                    })
                };
                forward_decls
                    .push(crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDecl(decl));
            }
        }
        for decl in forward_decls {
            if let crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDecl(fd) = decl {
                tu.add_func_decl(fd);
            }
        }
    }

    // Forward declarations for internal functions, so a call preceding the definition does not get a conflicting implicit int name(); pushed directly to preserve the symbol->definition mapping.
    {
        let mut internal_decls = Vec::new();
        for decl in tu.decls.iter() {
            if let crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDef(fdef) = decl {
                if fdef.name == "main" {
                    continue;
                }
                let mut fd = crate::decompile::passes::c_pass::types::FuncDecl::new(
                    fdef.name.clone(),
                    fdef.return_type.clone(),
                    fdef.params.clone(),
                );
                fd.is_variadic = fdef.is_variadic;
                internal_decls.push(fd);
            }
        }
        for fd in internal_decls {
            tu.decls
                .push(crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDecl(fd));
        }
    }

    // Compilability: insert explicit casts at int<->pointer
    // assignment/argument/return mismatches after declaration recovery.  Call
    // argument vectors are immutable evidence; declaration arity is handled
    // by a per-site unprototyped callee cast or by declining a fixed
    // declaration, never by rewriting the vector.
    {
        use crate::decompile::passes::c_pass::types::TopLevelDecl;
        let mut callee_params: CalleeParams = HashMap::new();
        let mut callee_ret: HashMap<String, CType> = HashMap::new();
        let mut global_types: HashMap<String, CType> = HashMap::new();
        for decl in tu.decls.iter() {
            match decl {
                TopLevelDecl::FuncDef(f) => {
                    callee_params.insert(
                        f.name.clone(),
                        (
                            f.params.iter().map(|p| p.ty.clone()).collect(),
                            f.is_variadic,
                            false,
                        ),
                    );
                    callee_ret.insert(f.name.clone(), f.return_type.clone());
                }
                TopLevelDecl::FuncDecl(d) => {
                    callee_params.entry(d.name.clone()).or_insert_with(|| {
                        (
                            d.params.iter().map(|p| p.ty.clone()).collect(),
                            d.is_variadic,
                            d.unspecified_params,
                        )
                    });
                    callee_ret
                        .entry(d.name.clone())
                        .or_insert_with(|| d.return_type.clone());
                }
                TopLevelDecl::VarDecl(v) => {
                    global_types.insert(v.name.clone(), v.ty.clone());
                }
                _ => {}
            }
        }
        for decl in tu.decls.iter_mut() {
            if let TopLevelDecl::FuncDef(f) = decl {
                let mut types = global_types.clone();
                for p in &f.params {
                    if let Some(n) = &p.name {
                        types.insert(n.clone(), p.ty.clone());
                    }
                }
                for v in &f.local_vars {
                    types.insert(v.name.clone(), v.ty.clone());
                }
                let ret = f.return_type.clone();
                let types = CastTypes {
                    variables: &types,
                    struct_fields,
                };
                f.body = insert_casts_stmt(&f.body, &types, &callee_params, &callee_ret, &ret);
            }
        }
    }

    // Compilability: declare any referenced-but-undeclared file-scope identifier as a tentative long; synthesized locals are excluded, since an undeclared one is a recovery gap to fix upstream.
    {
        use crate::decompile::passes::c_pass::types::TopLevelDecl;
        let mut declared: HashSet<String> = HashSet::new();
        let mut locals: HashSet<String> = HashSet::new();
        for decl in tu.decls.iter() {
            match decl {
                TopLevelDecl::FuncDef(f) => {
                    declared.insert(f.name.clone());
                    for p in &f.params {
                        if let Some(n) = &p.name {
                            locals.insert(n.clone());
                        }
                    }
                    for v in &f.local_vars {
                        locals.insert(v.name.clone());
                    }
                }
                TopLevelDecl::FuncDecl(d) => {
                    declared.insert(d.name.clone());
                }
                TopLevelDecl::VarDecl(v) => {
                    declared.insert(v.name.clone());
                }
                _ => {}
            }
        }
        let mut referenced: HashSet<String> = HashSet::new();
        for decl in tu.decls.iter() {
            if let TopLevelDecl::FuncDef(f) = decl {
                collect_var_names_from_stmt(&f.body, &mut referenced);
            }
        }
        // __builtin_overflow, the opaque marker for the x86 OF flag, is declared once on use as a static volatile object so it recompiles and the optimizer cannot collapse the guarded branch.
        let uses_overflow_marker = referenced.contains("__builtin_overflow");
        let mut undeclared: Vec<String> = referenced
            .into_iter()
            .filter(|n| {
                !declared.contains(n)
                    && !locals.contains(n)
                    && is_global_like_name(n)
                    && !is_libc_stdio_global(n)
            })
            .collect();
        undeclared.sort();
        undeclared.dedup();
        for name in undeclared {
            // An undeclared L_<addr> label holding a recovered .rodata FP constant emits a typed read-only definition, so uses read the real constant rather than a zero-initialized long.
            if let Some(addr) = name
                .strip_prefix("L_")
                .and_then(|h| u64::from_str_radix(h, 16).ok())
            {
                if let Some(&(width, bits)) = rodata_fp_consts.get(&addr) {
                    if let Some(decl) = make_rodata_fp_decl(name.clone(), width, bits) {
                        tu.decls.push(decl);
                        continue;
                    }
                }
            }
            // A skip-listed library function that is only ADDRESS-TAKEN gets an extern function declaration, or a tentative long object makes &name point at a fresh .bss slot instead of the function.
            if db.should_skip_function(&name) {
                // Prefer the curated signature (void close_stdout(void), ...) over a K&R `int name()`.
                let decl = known_decl(&name).unwrap_or_else(|| {
                    crate::decompile::passes::c_pass::types::FuncDecl::new_unspecified(
                        name.clone(),
                        CType::Int(
                            crate::decompile::passes::c_pass::types::IntSize::Int,
                            Signedness::Signed,
                        ),
                    )
                });
                tu.add_func_decl(decl);
                continue;
            }
            // A header-declared global reaching the undeclared path must use the header's canonical type, since a long tentative definition against an extern int is a hard conflicting-types error.
            let var_ty = crate::decompile::passes::c_pass::header_db::header_db()
                .variables
                .get(name.as_str())
                .and_then(|spelling| parse_canonical_var_type(spelling))
                .unwrap_or_else(CType::long);
            // A library/CRT-provided global reaching the undeclared path must be extern, or a local tentative definition is a multiple-definition error at link.
            let storage_class = if db.is_lib_global(&name)
                || (uses_real_libc_name(&name) && !is_libc_stdio_global(&name))
            {
                StorageClass::Extern
            } else {
                StorageClass::default()
            };
            tu.decls.push(TopLevelDecl::VarDecl(VarDecl {
                name,
                ty: var_ty,
                storage_class,
                qualifiers: TypeQualifiers::none(),
                init: None,
                loc: SourceLoc::unknown(),
            }));
        }

        if uses_overflow_marker {
            tu.decls.push(TopLevelDecl::VarDecl(VarDecl {
                name: "__builtin_overflow".to_string(),
                ty: CType::Int(
                    crate::decompile::passes::c_pass::types::IntSize::Char,
                    Signedness::Unsigned,
                ),
                storage_class: StorageClass::Static,
                qualifiers: TypeQualifiers {
                    is_volatile: true,
                    ..TypeQualifiers::none()
                },
                init: None,
                loc: SourceLoc::unknown(),
            }));
        }
    }

    // Drop captures of void-returning calls: lhs = free(p) is a frontend error, so emit the call as a bare statement; only direct assignment-of-call statements are rewritten.
    let void_fns: HashSet<String> = known_sig_by_name
        .iter()
        .filter(|(_, (ret, _))| matches!(ret, XType::Xvoid))
        .map(|(n, _)| n.clone())
        .collect();
    if !void_fns.is_empty() {
        for decl in tu.decls.iter_mut() {
            if let crate::decompile::passes::c_pass::types::TopLevelDecl::FuncDef(f) = decl {
                drop_void_captures_stmt(&mut f.body, &void_fns);
            }
        }
    }

    tu
}

fn callee_name_through_casts(e: &CExpr) -> Option<&str> {
    match e {
        CExpr::Var(name) => Some(name.as_str()),
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => callee_name_through_casts(inner),
        _ => None,
    }
}

// The innermost Call(Var(name), ...) of a possibly cast-wrapped call expression, returning the callee name, or None when it is not such a call.
fn call_name_through_casts(e: &CExpr) -> Option<&str> {
    match e {
        CExpr::Call(callee, _) => callee_name_through_casts(callee),
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => call_name_through_casts(inner),
        _ => None,
    }
}

// The bare `Call(...)` node inside a cast/paren-wrapped call expression.
fn unwrap_to_call(e: CExpr) -> CExpr {
    match e {
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => unwrap_to_call(*inner),
        other => other,
    }
}

fn drop_void_captures_stmt(stmt: &mut CStmt, void_fns: &HashSet<String>) {
    match stmt {
        CStmt::Expr(CExpr::Assign(_, _, rhs)) => {
            if call_name_through_casts(rhs).is_some_and(|n| void_fns.contains(n)) {
                let call = unwrap_to_call((**rhs).clone());
                *stmt = CStmt::Expr(call);
            }
        }
        CStmt::If(_, then_s, else_s) => {
            drop_void_captures_stmt(then_s, void_fns);
            if let Some(e) = else_s {
                drop_void_captures_stmt(e, void_fns);
            }
        }
        CStmt::Switch(_, body) => drop_void_captures_stmt(body, void_fns),
        CStmt::While(_, body) | CStmt::DoWhile(body, _) => drop_void_captures_stmt(body, void_fns),
        CStmt::For(_, _, _, body) => drop_void_captures_stmt(body, void_fns),
        CStmt::Labeled(_, inner) => drop_void_captures_stmt(inner, void_fns),
        CStmt::Block(items) => {
            for item in items.iter_mut() {
                if let CBlockItem::Stmt(s) = item {
                    drop_void_captures_stmt(s, void_fns);
                }
            }
        }
        CStmt::Sequence(stmts) => {
            for s in stmts.iter_mut() {
                drop_void_captures_stmt(s, void_fns);
            }
        }
        _ => {}
    }
}

fn param_name_for_reg(reg: RTLReg) -> String {
    let ident = ident_from_reg(reg);
    format!("var_{}", ident)
}

pub(crate) fn convert_param_type_from_param(param: &ParamType) -> CType {
    match param {
        ParamType::StructPointer(struct_id) => {
            let struct_name = format!("struct_{:x}", struct_id);
            CType::ptr(CType::Struct(struct_name))
        }
        ParamType::Pointer => CType::ptr(CType::Void),
        ParamType::Typed(xtype) => convert_xtype(xtype),
        ParamType::Integer | ParamType::Unknown => CType::int(),
    }
}

// Collect locals with an origin in the body (assignment LHS, address-taken, or inc/dec target) for the use-before-def guard; a referenced local with none of these comes from nowhere.
fn collect_var_origins_expr(expr: &CExpr, out: &mut HashSet<String>) {
    match expr {
        CExpr::Assign(_, lhs, rhs) => {
            if let CExpr::Var(n) = lhs.as_ref() {
                out.insert(n.clone());
            }
            collect_var_origins_expr(lhs, out);
            collect_var_origins_expr(rhs, out);
        }
        CExpr::Unary(op, inner) => {
            if matches!(
                op,
                UnaryOp::AddrOf
                    | UnaryOp::PreInc
                    | UnaryOp::PreDec
                    | UnaryOp::PostInc
                    | UnaryOp::PostDec
            ) {
                if let CExpr::Var(n) = inner.as_ref() {
                    out.insert(n.clone());
                }
            }
            collect_var_origins_expr(inner, out);
        }
        CExpr::Binary(_, l, r) | CExpr::Index(l, r) => {
            collect_var_origins_expr(l, out);
            collect_var_origins_expr(r, out);
        }
        CExpr::Ternary(c, t, e) => {
            collect_var_origins_expr(c, out);
            collect_var_origins_expr(t, out);
            collect_var_origins_expr(e, out);
        }
        CExpr::Call(f, args) => {
            collect_var_origins_expr(f, out);
            for a in args {
                collect_var_origins_expr(a, out);
            }
        }
        CExpr::Cast(_, inner)
        | CExpr::Paren(inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _)
        | CExpr::SizeofExpr(inner) => {
            collect_var_origins_expr(inner, out);
        }
        CExpr::StmtExpr(stmts, fin) => {
            for s in stmts {
                collect_var_origins_stmt(s, out);
            }
            collect_var_origins_expr(fin, out);
        }
        _ => {}
    }
}

fn collect_var_origins_stmt(stmt: &CStmt, out: &mut HashSet<String>) {
    use crate::decompile::passes::c_pass::types::ForInit;
    match stmt {
        CStmt::Expr(e) | CStmt::Return(Some(e)) => collect_var_origins_expr(e, out),
        CStmt::If(c, t, e) => {
            collect_var_origins_expr(c, out);
            collect_var_origins_stmt(t, out);
            if let Some(s) = e {
                collect_var_origins_stmt(s, out);
            }
        }
        CStmt::Switch(e, body) => {
            collect_var_origins_expr(e, out);
            collect_var_origins_stmt(body, out);
        }
        CStmt::While(c, body) | CStmt::DoWhile(body, c) => {
            collect_var_origins_expr(c, out);
            collect_var_origins_stmt(body, out);
        }
        CStmt::For(init, cond, update, body) => {
            match init {
                Some(ForInit::Expr(e)) => collect_var_origins_expr(e, out),
                Some(ForInit::Decl(decls)) => {
                    for d in decls {
                        out.insert(d.name.clone());
                    }
                }
                None => {}
            }
            if let Some(c) = cond {
                collect_var_origins_expr(c, out);
            }
            if let Some(u) = update {
                collect_var_origins_expr(u, out);
            }
            collect_var_origins_stmt(body, out);
        }
        CStmt::Block(items) => {
            for item in items {
                match item {
                    CBlockItem::Stmt(s) => collect_var_origins_stmt(s, out),
                    CBlockItem::Decl(decls) => {
                        for d in decls {
                            out.insert(d.name.clone());
                        }
                    }
                }
            }
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_var_origins_stmt(s, out);
            }
        }
        CStmt::Labeled(_, inner) => collect_var_origins_stmt(inner, out),
        _ => {}
    }
}

// Zero initializer for a "variable from nowhere": scalars/pointers get `= 0`, aggregates get `= {0}` (both valid C); function/void-typed declarations are left untouched.
fn zero_initializer_for_var(
    ty: &CType,
) -> Option<crate::decompile::passes::c_pass::types::Initializer> {
    use crate::decompile::passes::c_pass::types::{InitItem, Initializer};
    match ty {
        CType::Void | CType::Function(..) => None,
        CType::Struct(_) | CType::Union(_) | CType::Array(..) => {
            Some(Initializer::List(vec![InitItem {
                designator: None,
                init: Initializer::Expr(CExpr::int(0)),
            }]))
        }
        _ => Some(Initializer::Expr(CExpr::int(0))),
    }
}

fn collect_var_names_from_stmt(stmt: &CStmt, vars: &mut HashSet<String>) {
    match stmt {
        CStmt::Empty | CStmt::Continue | CStmt::Break | CStmt::Goto(_) => {}
        CStmt::Expr(e) | CStmt::Return(Some(e)) => collect_var_names_from_expr(e, vars),
        CStmt::Return(None) => {}
        CStmt::If(cond, then_s, else_s) => {
            collect_var_names_from_expr(cond, vars);
            collect_var_names_from_stmt(then_s, vars);
            if let Some(e) = else_s {
                collect_var_names_from_stmt(e, vars);
            }
        }
        CStmt::Switch(expr, body) => {
            collect_var_names_from_expr(expr, vars);
            collect_var_names_from_stmt(body, vars);
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            collect_var_names_from_stmt(body, vars);
            collect_var_names_from_expr(cond, vars);
        }
        CStmt::For(init, cond, update, body) => {
            if let Some(init) = init {
                match init {
                    crate::decompile::passes::c_pass::types::ForInit::Expr(e) => {
                        collect_var_names_from_expr(e, vars)
                    }
                    crate::decompile::passes::c_pass::types::ForInit::Decl(decls) => {
                        for d in decls {
                            vars.insert(d.name.clone());
                        }
                    }
                }
            }
            if let Some(cond) = cond {
                collect_var_names_from_expr(cond, vars);
            }
            if let Some(update) = update {
                collect_var_names_from_expr(update, vars);
            }
            collect_var_names_from_stmt(body, vars);
        }
        CStmt::Labeled(_, inner) => collect_var_names_from_stmt(inner, vars),
        CStmt::Block(items) => {
            for item in items {
                match item {
                    CBlockItem::Stmt(s) => collect_var_names_from_stmt(s, vars),
                    CBlockItem::Decl(decls) => {
                        for d in decls {
                            vars.insert(d.name.clone());
                        }
                    }
                }
            }
        }
        CStmt::Decl(decls) => {
            for d in decls {
                vars.insert(d.name.clone());
            }
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_var_names_from_stmt(s, vars);
            }
        }
    }
}

/// The base symbol(s) an lvalue ultimately names (`x`, `*x`, `x.f`, `x[i]`, `(T)x` all base on `x`).
fn collect_lvalue_base_names(expr: &CExpr, out: &mut HashSet<String>) {
    match expr {
        CExpr::Var(n) => {
            out.insert(n.clone());
        }
        CExpr::Unary(_, e)
        | CExpr::Cast(_, e)
        | CExpr::Paren(e)
        | CExpr::Member(e, _)
        | CExpr::MemberPtr(e, _) => collect_lvalue_base_names(e, out),
        CExpr::Index(a, _) => collect_lvalue_base_names(a, out),
        _ => {}
    }
}

/// Collect names appearing in a MUTATION context (assignment LHS, & operand, inc/dec target), distinguishing a library-provided global from one this TU owns; over-collecting only keeps a global defined.
fn collect_mutated_names_in_expr(expr: &CExpr, out: &mut HashSet<String>) {
    match expr {
        CExpr::Assign(_, lhs, rhs) => {
            collect_lvalue_base_names(lhs, out);
            collect_mutated_names_in_expr(lhs, out);
            collect_mutated_names_in_expr(rhs, out);
        }
        CExpr::Unary(op, inner) => {
            if matches!(
                op,
                UnaryOp::AddrOf
                    | UnaryOp::PreInc
                    | UnaryOp::PreDec
                    | UnaryOp::PostInc
                    | UnaryOp::PostDec
            ) {
                collect_lvalue_base_names(inner, out);
            }
            collect_mutated_names_in_expr(inner, out);
        }
        CExpr::Binary(_, a, b) | CExpr::Index(a, b) => {
            collect_mutated_names_in_expr(a, out);
            collect_mutated_names_in_expr(b, out);
        }
        CExpr::Cast(_, e)
        | CExpr::Paren(e)
        | CExpr::SizeofExpr(e)
        | CExpr::Member(e, _)
        | CExpr::MemberPtr(e, _) => collect_mutated_names_in_expr(e, out),
        CExpr::Ternary(c, t, e) => {
            collect_mutated_names_in_expr(c, out);
            collect_mutated_names_in_expr(t, out);
            collect_mutated_names_in_expr(e, out);
        }
        CExpr::Call(f, args) => {
            collect_mutated_names_in_expr(f, out);
            for a in args {
                collect_mutated_names_in_expr(a, out);
            }
        }
        _ => {}
    }
}

fn collect_mutated_names_in_stmt(stmt: &CStmt, out: &mut HashSet<String>) {
    match stmt {
        CStmt::Expr(e) | CStmt::Return(Some(e)) => collect_mutated_names_in_expr(e, out),
        CStmt::If(cond, then_s, else_s) => {
            collect_mutated_names_in_expr(cond, out);
            collect_mutated_names_in_stmt(then_s, out);
            if let Some(e) = else_s {
                collect_mutated_names_in_stmt(e, out);
            }
        }
        CStmt::Switch(expr, body) => {
            collect_mutated_names_in_expr(expr, out);
            collect_mutated_names_in_stmt(body, out);
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            collect_mutated_names_in_stmt(body, out);
            collect_mutated_names_in_expr(cond, out);
        }
        CStmt::For(init, cond, update, body) => {
            if let Some(crate::decompile::passes::c_pass::types::ForInit::Expr(e)) = init {
                collect_mutated_names_in_expr(e, out);
            }
            if let Some(cond) = cond {
                collect_mutated_names_in_expr(cond, out);
            }
            if let Some(update) = update {
                collect_mutated_names_in_expr(update, out);
            }
            collect_mutated_names_in_stmt(body, out);
        }
        CStmt::Labeled(_, inner) => collect_mutated_names_in_stmt(inner, out),
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    collect_mutated_names_in_stmt(s, out);
                }
            }
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_mutated_names_in_stmt(s, out);
            }
        }
        _ => {}
    }
}

fn infer_var_types_from_usage(
    stmt: &CStmt,
    var_types: &mut HashMap<String, CType>,
    extern_return_types: &HashMap<String, CType>,
) {
    match stmt {
        CStmt::Expr(e) => infer_types_from_expr(e, var_types, extern_return_types),
        CStmt::Return(Some(e)) => infer_types_from_expr(e, var_types, extern_return_types),
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    infer_var_types_from_usage(s, var_types, extern_return_types);
                }
            }
        }
        CStmt::If(cond, then_s, else_s) => {
            infer_types_from_expr(cond, var_types, extern_return_types);
            infer_var_types_from_usage(then_s, var_types, extern_return_types);
            if let Some(e) = else_s {
                infer_var_types_from_usage(e, var_types, extern_return_types);
            }
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            infer_types_from_expr(cond, var_types, extern_return_types);
            infer_var_types_from_usage(body, var_types, extern_return_types);
        }
        CStmt::For(init, cond, update, body) => {
            if let Some(fi) = init {
                if let crate::decompile::passes::c_pass::types::ForInit::Expr(e) = fi {
                    infer_types_from_expr(e, var_types, extern_return_types);
                }
            }
            if let Some(e) = cond {
                infer_types_from_expr(e, var_types, extern_return_types);
            }
            if let Some(e) = update {
                infer_types_from_expr(e, var_types, extern_return_types);
            }
            infer_var_types_from_usage(body, var_types, extern_return_types);
        }
        CStmt::Switch(e, body) => {
            infer_types_from_expr(e, var_types, extern_return_types);
            infer_var_types_from_usage(body, var_types, extern_return_types);
        }
        CStmt::Labeled(_, inner) => {
            infer_var_types_from_usage(inner, var_types, extern_return_types)
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                infer_var_types_from_usage(s, var_types, extern_return_types);
            }
        }
        _ => {}
    }
}

fn insert_or_refine(var_types: &mut HashMap<String, CType>, name: String, new_ty: CType) {
    match var_types.get(&name) {
        None => {
            var_types.insert(name, new_ty);
        }
        Some(existing) => {
            if *existing == CType::ptr(CType::Void)
                && new_ty != CType::ptr(CType::Void)
                && new_ty.is_pointer()
            {
                var_types.insert(name, new_ty);
            }
            // NOTE (plan 5.4): the int/long -> pointer FLIP is retired. Pointerness is decided by the solver and force_ptr_regs; usage inference defers on that axis and only refines the pointee or width.
            else if *existing == CType::long() && !new_ty.is_pointer() && new_ty != CType::long()
            {
                var_types.insert(name, new_ty);
            } else if existing.is_generic_long_ptr()
                && new_ty.is_pointer()
                && !new_ty.is_generic_long_ptr()
            {
                var_types.insert(name, new_ty);
            }
        }
    }
}

fn extract_var_name(expr: &CExpr) -> Option<&str> {
    match expr {
        CExpr::Var(name) => Some(name.as_str()),
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => extract_var_name(inner),
        _ => None,
    }
}

// A 32/64-bit int/long, the default scalar the seeding hands an untyped result reg; sub-int, float and pointer types are excluded so the pointer-return override never downgrades them.
fn is_default_int_scalar_ty(t: Option<&CType>) -> bool {
    use crate::decompile::passes::c_pass::types::IntSize;
    matches!(t, Some(CType::Int(IntSize::Int | IntSize::Long, _)))
}

fn infer_types_from_expr(
    expr: &CExpr,
    var_types: &mut HashMap<String, CType>,
    extern_return_types: &HashMap<String, CType>,
) {
    match expr {
        CExpr::Unary(UnaryOp::Deref, inner) => {
            if let CExpr::Cast(ty, cast_inner) = inner.as_ref() {
                if let Some(name) = extract_var_name(cast_inner) {
                    if ty.is_pointer() {
                        insert_or_refine(var_types, name.to_string(), ty.clone());
                    }
                }
            } else if let CExpr::Paren(paren_inner) = inner.as_ref() {
                if let CExpr::Binary(BinaryOp::Add, lhs, _) | CExpr::Binary(BinaryOp::Sub, lhs, _) =
                    paren_inner.as_ref()
                {
                    if let CExpr::Cast(ty, cast_inner) = lhs.as_ref() {
                        if let Some(name) = extract_var_name(cast_inner) {
                            if ty.is_pointer() {
                                insert_or_refine(var_types, name.to_string(), ty.clone());
                            }
                        }
                    }
                }
            } else if let CExpr::Var(name) = inner.as_ref() {
                var_types
                    .entry(name.clone())
                    .or_insert_with(|| CType::ptr(CType::Void));
            }
            infer_types_from_expr(inner, var_types, extern_return_types);
        }
        CExpr::Unary(UnaryOp::Not, inner) => {
            if let Some(name) = extract_var_name(inner) {
                var_types.entry(name.to_string()).or_insert(CType::Bool);
            }
            infer_types_from_expr(inner, var_types, extern_return_types);
        }
        CExpr::Assign(_, lhs, rhs) => {
            if let CExpr::Var(name) = lhs.as_ref() {
                if let CExpr::Cast(ty, _) = rhs.as_ref() {
                    if ty.is_pointer() {
                        insert_or_refine(var_types, name.clone(), ty.clone());
                    }
                }
                if matches!(rhs.as_ref(), CExpr::StringLit(_)) {
                    insert_or_refine(var_types, name.clone(), CType::ptr(CType::char_signed()));
                }
                if let CExpr::Call(callee, _) = rhs.as_ref() {
                    if let CExpr::Var(func_name) = callee.as_ref() {
                        if let Some(ret_ty) = extern_return_types.get(func_name.as_str()) {
                            // The result of a known POINTER-returning call is that pointer (a declared return type, not a cast inference); overrides only a default int/long scalar that would truncate it.
                            if ret_ty.is_pointer() && is_default_int_scalar_ty(var_types.get(name))
                            {
                                var_types.insert(name.clone(), ret_ty.clone());
                            } else {
                                var_types.entry(name.clone()).or_insert(ret_ty.clone());
                            }
                        }
                    }
                }
            }
            infer_types_from_expr(lhs, var_types, extern_return_types);
            infer_types_from_expr(rhs, var_types, extern_return_types);
        }
        CExpr::Call(func, args) => {
            if let CExpr::Var(func_name) = func.as_ref() {
                infer_call_arg_types(func_name, args, var_types);
            }
            infer_types_from_expr(func, var_types, extern_return_types);
            for arg in args {
                infer_types_from_expr(arg, var_types, extern_return_types);
            }
        }
        CExpr::MemberPtr(inner, _) => {
            if let CExpr::Var(name) = inner.as_ref() {
                var_types
                    .entry(name.clone())
                    .or_insert_with(|| CType::ptr(CType::Void));
            }
            infer_types_from_expr(inner, var_types, extern_return_types);
        }
        CExpr::Index(base, index) => {
            if let CExpr::Var(name) = base.as_ref() {
                var_types
                    .entry(name.clone())
                    .or_insert_with(|| CType::ptr(CType::Void));
            }
            infer_types_from_expr(base, var_types, extern_return_types);
            infer_types_from_expr(index, var_types, extern_return_types);
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Paren(inner)
        | CExpr::SizeofExpr(inner)
        | CExpr::Member(inner, _) => {
            infer_types_from_expr(inner, var_types, extern_return_types);
        }
        CExpr::Binary(_, lhs, rhs) => {
            infer_types_from_expr(lhs, var_types, extern_return_types);
            infer_types_from_expr(rhs, var_types, extern_return_types);
        }
        CExpr::Ternary(c, t, e) => {
            infer_types_from_expr(c, var_types, extern_return_types);
            infer_types_from_expr(t, var_types, extern_return_types);
            infer_types_from_expr(e, var_types, extern_return_types);
        }
        _ => {}
    }
}

fn infer_call_arg_types(func_name: &str, args: &[CExpr], var_types: &mut HashMap<String, CType>) {
    let char_ptr = || CType::ptr(CType::char_signed());
    let void_ptr = || CType::ptr(CType::Void);
    let str_first: &[&str] = &[
        "strlen",
        "strcmp",
        "strncmp",
        "strcpy",
        "strdup",
        "strndup",
        "strcat",
        "strncat",
        "strchr",
        "strrchr",
        "strstr",
        "strtol",
        "strtoul",
        "strtoll",
        "strtoull",
        "strtod",
        "atoi",
        "atol",
        "atoll",
        "fputs",
        "puts",
        "printf",
        "sprintf",
        "snprintf",
        "__sprintf_chk",
        "__snprintf_chk",
        "__fprintf_chk",
        "__printf_chk",
        "dcgettext",
        "gettext",
        "ngettext",
        "dgettext",
        "setlocale",
        "getenv",
        "putenv",
        "opendir",
        "stat",
        "lstat",
        "access",
        "unlink",
        "rmdir",
        "mkdir",
        "fopen",
        "freopen",
        "remove",
        "rename",
        "error",
        "error_at_line",
    ];
    if str_first.contains(&func_name) {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
    }
    let str_second: &[&str] = &[
        "strcmp",
        "strncmp",
        "strcpy",
        "strncpy",
        "strcat",
        "strncat",
        "strstr",
        "fprintf",
        "sprintf",
        "snprintf",
        "fopen",
        "freopen",
        "rename",
        "__sprintf_chk",
        "__snprintf_chk",
    ];
    if str_second.contains(&func_name) {
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
    }
    if func_name == "fprintf" {
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
    }
    if func_name == "snprintf" || func_name == "__snprintf_chk" {
        if let Some(name) = args.get(2).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
    }
    if matches!(func_name, "memset" | "memcpy" | "memmove" | "memcmp") {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), void_ptr());
        }
        if matches!(func_name, "memcpy" | "memmove" | "memcmp") {
            if let Some(name) = args.get(1).and_then(extract_var_name) {
                insert_or_refine(var_types, name.to_string(), void_ptr());
            }
        }
    }
    if matches!(func_name, "memset" | "memcpy" | "memmove" | "memcmp") {
        if let Some(name) = args.get(2).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::ulong());
        }
    }
    if func_name == "error" || func_name == "error_at_line" {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if func_name == "getopt_long" || func_name == "getopt" {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(
                var_types,
                name.to_string(),
                CType::ptr(CType::ptr(CType::char_signed())),
            );
        }
        if let Some(name) = args.get(2).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
    }
    if matches!(func_name, "exit" | "_exit" | "_Exit") {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if matches!(func_name, "fwrite" | "fread") {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), void_ptr());
        }
    }
    if func_name == "fgets" {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), char_ptr());
        }
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if func_name == "setlocale" {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if func_name == "open" {
        if let Some(name) = args.get(1).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if matches!(
        func_name,
        "close" | "read" | "write" | "dup" | "dup2" | "fcntl" | "ioctl" | "isatty"
    ) {
        if let Some(name) = args.first().and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
    if matches!(
        func_name,
        "strtol" | "strtoul" | "strtoll" | "strtoull" | "strtod"
    ) {
        if let Some(name) = args.get(2).and_then(extract_var_name) {
            insert_or_refine(var_types, name.to_string(), CType::int());
        }
    }
}

fn detect_bool_variables(stmt: &CStmt) -> HashSet<&str> {
    let mut bool_uses: HashMap<&str, (usize, usize)> = HashMap::new();
    collect_bool_evidence(stmt, &mut bool_uses, false);
    bool_uses
        .into_iter()
        .filter(|(_, (bool_count, non_bool_count))| *bool_count > 0 && *non_bool_count == 0)
        .map(|(name, _)| name)
        .collect()
}

fn collect_bool_evidence<'a>(
    stmt: &'a CStmt,
    evidence: &mut HashMap<&'a str, (usize, usize)>,
    _in_bool_ctx: bool,
) {
    match stmt {
        CStmt::If(cond, then_s, else_s) => {
            collect_bool_evidence_expr(cond, evidence, true);
            collect_bool_evidence(then_s, evidence, false);
            if let Some(e) = else_s {
                collect_bool_evidence(e, evidence, false);
            }
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            collect_bool_evidence_expr(cond, evidence, true);
            collect_bool_evidence(body, evidence, false);
        }
        CStmt::For(init, cond, update, body) => {
            if let Some(fi) = init {
                if let crate::decompile::passes::c_pass::types::ForInit::Expr(e) = fi {
                    collect_bool_evidence_expr(e, evidence, false);
                }
            }
            if let Some(e) = cond {
                collect_bool_evidence_expr(e, evidence, true);
            }
            if let Some(e) = update {
                collect_bool_evidence_expr(e, evidence, false);
            }
            collect_bool_evidence(body, evidence, false);
        }
        CStmt::Expr(e) => collect_bool_evidence_expr(e, evidence, false),
        CStmt::Return(Some(e)) => collect_bool_evidence_expr(e, evidence, false),
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    collect_bool_evidence(s, evidence, false);
                }
            }
        }
        CStmt::Switch(e, body) => {
            collect_bool_evidence_expr(e, evidence, false);
            collect_bool_evidence(body, evidence, false);
        }
        CStmt::Labeled(_, inner) => collect_bool_evidence(inner, evidence, false),
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_bool_evidence(s, evidence, false);
            }
        }
        _ => {}
    }
}

fn collect_bool_evidence_expr<'a>(
    expr: &'a CExpr,
    evidence: &mut HashMap<&'a str, (usize, usize)>,
    in_bool_ctx: bool,
) {
    match expr {
        CExpr::Var(name) => {
            let entry = evidence.entry(name.as_str()).or_insert((0, 0));
            if in_bool_ctx {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
        CExpr::Unary(UnaryOp::Not, inner) => {
            collect_bool_evidence_expr(inner, evidence, true);
        }
        CExpr::Binary(op, lhs, rhs) if matches!(op, BinaryOp::Eq | BinaryOp::Ne) => {
            let lhs_is_bool_const =
                matches!(rhs.as_ref(), CExpr::IntLit(lit) if lit.value == 0 || lit.value == 1);
            let rhs_is_bool_const =
                matches!(lhs.as_ref(), CExpr::IntLit(lit) if lit.value == 0 || lit.value == 1);
            collect_bool_evidence_expr(lhs, evidence, lhs_is_bool_const || in_bool_ctx);
            collect_bool_evidence_expr(rhs, evidence, rhs_is_bool_const || in_bool_ctx);
        }
        CExpr::Binary(BinaryOp::And | BinaryOp::Or, lhs, rhs) => {
            collect_bool_evidence_expr(lhs, evidence, true);
            collect_bool_evidence_expr(rhs, evidence, true);
        }
        CExpr::Assign(_, lhs, rhs) => {
            let rhs_is_bool = matches!(rhs.as_ref(), CExpr::IntLit(lit) if lit.value == 0 || lit.value == 1)
                || matches!(rhs.as_ref(), CExpr::Binary(op, _, _) if matches!(op,
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le |
                    BinaryOp::Gt | BinaryOp::Ge | BinaryOp::And | BinaryOp::Or))
                || matches!(rhs.as_ref(), CExpr::Unary(UnaryOp::Not, _));
            collect_bool_evidence_expr(lhs, evidence, rhs_is_bool);
            collect_bool_evidence_expr(rhs, evidence, false);
        }
        CExpr::Binary(_, lhs, rhs) | CExpr::Index(lhs, rhs) => {
            collect_bool_evidence_expr(lhs, evidence, false);
            collect_bool_evidence_expr(rhs, evidence, false);
        }
        CExpr::Ternary(c, t, e) => {
            collect_bool_evidence_expr(c, evidence, true);
            collect_bool_evidence_expr(t, evidence, in_bool_ctx);
            collect_bool_evidence_expr(e, evidence, in_bool_ctx);
        }
        CExpr::Call(func, args) => {
            collect_bool_evidence_expr(func, evidence, false);
            for arg in args {
                collect_bool_evidence_expr(arg, evidence, false);
            }
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Paren(inner)
        | CExpr::SizeofExpr(inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _) => {
            collect_bool_evidence_expr(inner, evidence, false);
        }
        _ => {}
    }
}

fn strip_trivial_casts(stmt: &CStmt, var_types: &HashMap<String, CType>) -> CStmt {
    match stmt {
        CStmt::Expr(e) => CStmt::Expr(strip_trivial_casts_expr(e, var_types)),
        CStmt::Block(items) => CStmt::Block(
            items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => CBlockItem::Stmt(strip_trivial_casts(s, var_types)),
                    other => other.clone(),
                })
                .collect(),
        ),
        CStmt::If(cond, then_s, else_s) => CStmt::If(
            strip_trivial_casts_expr(cond, var_types),
            Box::new(strip_trivial_casts(then_s, var_types)),
            else_s
                .as_ref()
                .map(|e| Box::new(strip_trivial_casts(e, var_types))),
        ),
        CStmt::While(cond, body) => CStmt::While(
            strip_trivial_casts_expr(cond, var_types),
            Box::new(strip_trivial_casts(body, var_types)),
        ),
        CStmt::DoWhile(body, cond) => CStmt::DoWhile(
            Box::new(strip_trivial_casts(body, var_types)),
            strip_trivial_casts_expr(cond, var_types),
        ),
        CStmt::For(init, cond, update, body) => CStmt::For(
            init.clone(),
            cond.as_ref()
                .map(|e| strip_trivial_casts_expr(e, var_types)),
            update
                .as_ref()
                .map(|e| strip_trivial_casts_expr(e, var_types)),
            Box::new(strip_trivial_casts(body, var_types)),
        ),
        CStmt::Return(Some(e)) => CStmt::Return(Some(strip_trivial_casts_expr(e, var_types))),
        CStmt::Switch(e, body) => CStmt::Switch(
            strip_trivial_casts_expr(e, var_types),
            Box::new(strip_trivial_casts(body, var_types)),
        ),
        CStmt::Labeled(lbl, inner) => {
            CStmt::Labeled(lbl.clone(), Box::new(strip_trivial_casts(inner, var_types)))
        }
        CStmt::Sequence(stmts) => CStmt::Sequence(
            stmts
                .iter()
                .map(|s| strip_trivial_casts(s, var_types))
                .collect(),
        ),
        other => other.clone(),
    }
}

// True for a floating-point local: a (long)float_var is the load-bearing float->integer truncation, and stripping it under an outer pointer cast makes an illegal float->pointer conversion.
fn var_is_float_typed(var_types: &HashMap<String, CType>, name: &str) -> bool {
    matches!(var_types.get(name), Some(CType::Float(_)))
}

fn strip_trivial_casts_expr(expr: &CExpr, var_types: &HashMap<String, CType>) -> CExpr {
    fn unwrap_parens(expr: &CExpr) -> &CExpr {
        match expr {
            CExpr::Paren(inner) => unwrap_parens(inner),
            other => other,
        }
    }

    match expr {
        CExpr::Cast(ty, inner) => {
            let inner_stripped = strip_trivial_casts_expr(inner, var_types);

            if is_long_type(ty) {
                let unwrapped_inner = unwrap_parens(&inner_stripped);
                match unwrapped_inner {
                    // Keep (long)var when var is float/double: dropping it leaves a bare float under any outer pointer cast, and the (long) is what makes the subsequent int->pointer cast legal.
                    CExpr::Var(name) if var_is_float_typed(var_types, name) => {}
                    CExpr::Var(_) => return inner_stripped,
                    CExpr::IntLit(_) => return inner_stripped,
                    CExpr::StringLit(_) => return inner_stripped,
                    CExpr::Cast(inner_ty, _) if is_long_type(inner_ty) => return inner_stripped,
                    // Likewise keep (long)(float)x: the inner float-cast is the float->int step, so stripping the (long) would put a float directly under an outer pointer cast.
                    CExpr::Cast(inner_ty, _) if inner_ty.is_float() => {}
                    // Strip (long) wrapping pointer-typed expressions from coerce_ptr_to_long in clight_pass to avoid int/ptr conversion errors.
                    CExpr::Cast(inner_ty, _) if inner_ty.is_pointer() => return inner_stripped,
                    _ => {}
                }
            }
            if ty.is_pointer() {
                let unwrapped = unwrap_parens(&inner_stripped);
                if let CExpr::IntLit(lit) = unwrapped {
                    if lit.value != 0 {
                        return inner_stripped;
                    }
                    // Keep (void*)0 as null pointer constant
                }
            }
            CExpr::Cast(ty.clone(), Box::new(inner_stripped))
        }
        CExpr::Assign(op, lhs, rhs) => CExpr::Assign(
            *op,
            Box::new(strip_trivial_casts_expr(lhs, var_types)),
            Box::new(strip_trivial_casts_expr(rhs, var_types)),
        ),
        CExpr::Binary(op, lhs, rhs) => CExpr::Binary(
            *op,
            Box::new(strip_trivial_casts_expr(lhs, var_types)),
            Box::new(strip_trivial_casts_expr(rhs, var_types)),
        ),
        CExpr::Unary(op, inner) => {
            CExpr::Unary(*op, Box::new(strip_trivial_casts_expr(inner, var_types)))
        }
        CExpr::Call(func, args) => CExpr::Call(
            Box::new(strip_trivial_casts_expr(func, var_types)),
            args.iter()
                .map(|a| strip_trivial_casts_expr(a, var_types))
                .collect(),
        ),
        CExpr::Paren(inner) => CExpr::Paren(Box::new(strip_trivial_casts_expr(inner, var_types))),
        CExpr::Index(arr, idx) => CExpr::Index(
            Box::new(strip_trivial_casts_expr(arr, var_types)),
            Box::new(strip_trivial_casts_expr(idx, var_types)),
        ),
        CExpr::Member(inner, field) => CExpr::Member(
            Box::new(strip_trivial_casts_expr(inner, var_types)),
            field.clone(),
        ),
        CExpr::MemberPtr(inner, field) => CExpr::MemberPtr(
            Box::new(strip_trivial_casts_expr(inner, var_types)),
            field.clone(),
        ),
        CExpr::Ternary(cond, then_e, else_e) => CExpr::Ternary(
            Box::new(strip_trivial_casts_expr(cond, var_types)),
            Box::new(strip_trivial_casts_expr(then_e, var_types)),
            Box::new(strip_trivial_casts_expr(else_e, var_types)),
        ),
        other => other.clone(),
    }
}

// Last top-level return <expr> in a body block (the function's tail return), scanning only the immediate item list so it picks the function-level exit, not an early in-cascade return.
fn last_toplevel_return_expr(body: &CStmt) -> Option<CExpr> {
    let items = match body {
        CStmt::Block(items) => items,
        _ => return None,
    };
    let mut found: Option<CExpr> = None;
    for it in items {
        if let CBlockItem::Stmt(s) = it {
            match s {
                CStmt::Return(Some(e)) => found = Some(e.clone()),
                // A labeled block whose inner directly returns also exposes a tail return.
                CStmt::Labeled(_, inner) => {
                    if let CStmt::Return(Some(e)) = &**inner {
                        found = Some(e.clone());
                    }
                }
                _ => {}
            }
        }
    }
    found
}

// Repair the shared-return-merge orphan: append the tail return inside a final labeled block that does not end in an unconditional exit, always a defect in a value-returning function.
fn fix_falloff_return_orphan(body: &mut CStmt) {
    let ret_expr = match last_toplevel_return_expr(body) {
        Some(e) => e,
        None => return,
    };
    let items = match body {
        CStmt::Block(items) => items,
        _ => return,
    };
    let Some(CBlockItem::Stmt(last)) = items.last_mut() else {
        return;
    };
    let CStmt::Labeled(_, inner) = last else {
        return;
    };
    if is_unconditional_exit(inner) {
        return;
    }
    // Append `return <tail expr>` to the labeled block's body, preserving its existing statements.
    let appended = CStmt::Return(Some(ret_expr));
    let new_inner = match &**inner {
        CStmt::Block(inner_items) => {
            let mut v = inner_items.clone();
            v.push(CBlockItem::Stmt(appended));
            CStmt::Block(v)
        }
        CStmt::Empty => appended,
        other => CStmt::Block(vec![
            CBlockItem::Stmt(other.clone()),
            CBlockItem::Stmt(appended),
        ]),
    };
    **inner = new_inner;
}

fn fix_bare_returns_in_stmt(stmt: &mut CStmt) {
    match stmt {
        CStmt::Return(None) => {
            *stmt = CStmt::Return(Some(CExpr::int(0)));
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    fix_bare_returns_in_stmt(s);
                }
            }
        }
        CStmt::If(_, then_s, else_s) => {
            fix_bare_returns_in_stmt(then_s);
            if let Some(e) = else_s {
                fix_bare_returns_in_stmt(e);
            }
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) => fix_bare_returns_in_stmt(body),
        CStmt::For(_, _, _, body) => fix_bare_returns_in_stmt(body),
        CStmt::Switch(_, body) => fix_bare_returns_in_stmt(body),
        CStmt::Labeled(_, inner) => fix_bare_returns_in_stmt(inner),
        CStmt::Sequence(stmts) => {
            for s in stmts {
                fix_bare_returns_in_stmt(s);
            }
        }
        _ => {}
    }
}

fn strip_return_values_in_stmt(stmt: &mut CStmt) {
    match stmt {
        CStmt::Return(val @ Some(_)) => {
            *val = None;
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    strip_return_values_in_stmt(s);
                }
            }
        }
        CStmt::If(_, then_s, else_s) => {
            strip_return_values_in_stmt(then_s);
            if let Some(e) = else_s {
                strip_return_values_in_stmt(e);
            }
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) => strip_return_values_in_stmt(body),
        CStmt::For(_, _, _, body) => strip_return_values_in_stmt(body),
        CStmt::Switch(_, body) => strip_return_values_in_stmt(body),
        CStmt::Labeled(_, inner) => strip_return_values_in_stmt(inner),
        CStmt::Sequence(stmts) => {
            for s in stmts {
                strip_return_values_in_stmt(s);
            }
        }
        _ => {}
    }
}

pub(crate) fn sanitize_c_ident(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else if i > 0 {
            out.push('_');
        }
    }
    if out.is_empty() || out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    out
}

/// libc stdio FILE* globals keep their REAL name and are left UNDEFINED so <stdio.h>'s extern binds them to libc; renaming them silently disconnects them and any fputs on them segfaults.
pub(crate) fn is_libc_stdio_global(name: &str) -> bool {
    let base = name.split('@').next().unwrap_or(name);
    crate::decompile::passes::c_pass::header_db::header_db()
        .stdio_globals
        .contains(base)
}

/// libc globals that must keep their REAL name (the stdio FILE*s plus the getopt globals and environ) so they bind to libc; the getopt/environ ones are emitted as a plain extern.
fn uses_real_libc_name(name: &str) -> bool {
    let base = name.split('@').next().unwrap_or(name);
    crate::decompile::passes::c_pass::header_db::header_db()
        .real_libc_globals
        .contains(base)
}

pub(crate) fn sanitize_c_symbol_name(name: &str) -> String {
    if uses_real_libc_name(name) {
        return name.split('@').next().unwrap_or(name).to_string();
    }
    // Globals colliding with a libc symbol but holding a PROGRAM copy are renamed <name>_sym so the recovered definition neither binds to nor shadows the libc one.
    let reserved = &crate::decompile::passes::c_pass::header_db::header_db().reserved_globals;

    let base_name = name.split('@').next().unwrap_or(name);
    if reserved.contains(base_name) {
        let mut out = sanitize_c_ident(base_name);
        out.push_str("_sym");
        return out;
    }

    let mut out = sanitize_c_ident(name);
    if reserved.contains(out.as_str()) {
        out.push_str("_sym");
    }
    out
}

struct VarRenamer {
    map: HashMap<String, String>,
    counter: usize,
    used_names: HashSet<String>,
}

impl VarRenamer {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            counter: 0,
            used_names: HashSet::new(),
        }
    }

    fn seed_param(&mut self, name: &str, index: usize, dwarf_name: Option<&str>) {
        if self.map.contains_key(name) {
            return;
        }
        let short = if let Some(dname) = dwarf_name {
            let sanitized = sanitize_c_ident(dname);
            if self.used_names.contains(&sanitized) {
                format!("p{}", index)
            } else {
                sanitized
            }
        } else {
            format!("p{}", index)
        };
        self.used_names.insert(short.clone());
        self.map.insert(name.to_string(), short);
    }

    // ensure_mapping compacts a name only when it is a synthetic-reg name var_<5+ digit decimal> (fresh_xtl_reg sets bit 63, producing ~19-digit values); everything else (short var_N for real machine regs, params, globals) passes through rename() unchanged.
    fn is_renameable(name: &str) -> bool {
        match name.strip_prefix("var_") {
            Some(suffix) => suffix.len() > 4 && suffix.chars().all(|c| c.is_ascii_digit()),
            None => false,
        }
    }

    // Reserve a pass-through name so counter-based compaction never collides with it: short var_N names (e.g. var_0 from RTLReg 0) keep their literal spelling, so a compacted synthetic reg must not also be assigned that spelling.
    fn reserve(&mut self, name: &str) {
        if !Self::is_renameable(name) {
            self.used_names.insert(name.to_string());
        }
    }

    fn ensure_mapping(&mut self, name: &str) {
        if self.map.contains_key(name) {
            return;
        }
        if Self::is_renameable(name) {
            // Skip counter values already taken by a reserved pass-through name or a previously compacted reg, so two distinct slots never share one name.
            let short = loop {
                let candidate = format!("var_{:x}", self.counter);
                self.counter += 1;
                if !self.used_names.contains(&candidate) {
                    break candidate;
                }
            };
            self.used_names.insert(short.clone());
            self.map.insert(name.to_string(), short);
        }
    }

    fn rename(&self, name: &str) -> String {
        self.map
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }
}

impl ExprTransform for VarRenamer {
    fn transform_expr(&mut self, expr: CExpr) -> CExpr {
        match expr {
            CExpr::Var(name) => {
                self.ensure_mapping(&name);
                CExpr::Var(self.rename(&name))
            }
            other => self.walk_expr(other),
        }
    }
}

impl StmtTransform for VarRenamer {
    fn transform_stmt(&mut self, stmt: CStmt) -> CStmt {
        match stmt {
            CStmt::Decl(decls) => CStmt::Decl(
                decls
                    .into_iter()
                    .map(|mut d| {
                        self.ensure_mapping(&d.name);
                        d.name = self.rename(&d.name);
                        d.init = d.init.map(|i| self.transform_initializer(i));
                        d
                    })
                    .collect(),
            ),
            other => self.walk_stmt(other),
        }
    }
}

fn is_long_type(ty: &CType) -> bool {
    use crate::decompile::passes::c_pass::types::{IntSize, Signedness};
    matches!(
        ty,
        CType::Int(IntSize::Long, Signedness::Signed)
            | CType::Int(IntSize::Long, Signedness::Unsigned)
            | CType::Int(IntSize::LongLong, Signedness::Signed)
            | CType::Int(IntSize::LongLong, Signedness::Unsigned)
    )
}

fn collect_var_names_from_expr(expr: &CExpr, vars: &mut HashSet<String>) {
    match expr {
        CExpr::Var(name) => {
            // Collect ALL referenced names; a __builtin_* identifier is a compiler builtin, filtered out at the locals call-site but kept at the undeclared-globals one so the marker is declared.
            vars.insert(name.clone());
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Paren(inner)
        | CExpr::SizeofExpr(inner) => collect_var_names_from_expr(inner, vars),
        CExpr::MemberPtr(inner, _) | CExpr::Member(inner, _) => {
            collect_var_names_from_expr(inner, vars)
        }
        CExpr::Binary(_, lhs, rhs) | CExpr::Assign(_, lhs, rhs) | CExpr::Index(lhs, rhs) => {
            collect_var_names_from_expr(lhs, vars);
            collect_var_names_from_expr(rhs, vars);
        }
        CExpr::Ternary(c, t, e) => {
            collect_var_names_from_expr(c, vars);
            collect_var_names_from_expr(t, vars);
            collect_var_names_from_expr(e, vars);
        }
        CExpr::Call(func, args) => {
            // A bare-name callee is usually a direct call needing no VarDecl, but an indirect call's target is a var_* local that DOES need declaring; collect that, skip symbols, and recurse into complex callees.
            let callee_var = match func.as_ref() {
                CExpr::Var(n) => Some(n),
                CExpr::Cast(_, inner) => match inner.as_ref() {
                    CExpr::Var(n) => Some(n),
                    _ => None,
                },
                _ => None,
            };
            match callee_var {
                Some(n) if n.starts_with("var_") => {
                    vars.insert(n.clone());
                }
                Some(_) => {}
                None => collect_var_names_from_expr(func, vars),
            }
            for a in args {
                collect_var_names_from_expr(a, vars);
            }
        }
        CExpr::CompoundLit(_, inits) => {
            for init in inits {
                collect_var_names_from_initializer(init, vars);
            }
        }
        CExpr::Generic(sel, arms) => {
            collect_var_names_from_expr(sel, vars);
            for (_, arm_expr) in arms {
                collect_var_names_from_expr(arm_expr, vars);
            }
        }
        CExpr::StmtExpr(stmts, tail) => {
            for s in stmts {
                collect_var_names_from_stmt(s, vars);
            }
            collect_var_names_from_expr(tail, vars);
        }
        CExpr::SizeofType(_)
        | CExpr::AlignofType(_)
        | CExpr::IntLit(_)
        | CExpr::FloatLit(_)
        | CExpr::StringLit(_)
        | CExpr::CharLit(_) => {}
    }
}

fn collect_var_names_from_initializer(
    init: &crate::decompile::passes::c_pass::types::Initializer,
    vars: &mut HashSet<String>,
) {
    match init {
        crate::decompile::passes::c_pass::types::Initializer::Expr(e) => {
            collect_var_names_from_expr(e, vars)
        }
        crate::decompile::passes::c_pass::types::Initializer::List(items) => {
            for item in items {
                collect_var_names_from_initializer(&item.init, vars);
            }
        }
        crate::decompile::passes::c_pass::types::Initializer::String(_) => {}
    }
}

pub struct ConversionContext {
    pub id_to_name: HashMap<usize, String>,
    #[allow(dead_code)]
    pub global_names: HashSet<String>,
    pub string_globals: HashSet<usize>,
    #[allow(dead_code)]
    name_counter: usize,
    temp_names: HashMap<usize, String>,
    pub var_types: HashMap<String, CType>,
    pub string_label_to_content: HashMap<String, String>,
    pub suppress_string_literals: bool,
    pub func_addrs_sorted: Vec<(u64, String)>,
    /// Locally-DEFINED functions colliding with a header declaration, mapped to their emitted name (atexit -> atexit_local), applied to every symbol reference so call sites track the rename.
    pub func_renames: HashMap<String, String>,
    /// CAST-2: label id -> final injective C identifier, computed once so Slabel(id) and Sgoto(id) always resolve to the SAME name while distinct ids never share one.
    label_disambig: HashMap<usize, String>,
    current_function: Option<Address>,
    current_local_evar_ids: HashSet<Ident>,
    function_object_types: FunctionObjectTypes,
}

impl ConversionContext {
    pub fn new(id_to_name: HashMap<usize, String>) -> Self {
        let global_names = id_to_name.values().cloned().collect();
        let label_disambig = Self::build_label_disambig(&id_to_name);
        Self {
            id_to_name,
            global_names,
            string_globals: HashSet::new(),
            name_counter: 0,
            temp_names: HashMap::new(),
            var_types: HashMap::new(),
            string_label_to_content: HashMap::new(),
            suppress_string_literals: false,
            func_addrs_sorted: Vec::new(),
            func_renames: HashMap::new(),
            label_disambig,
            current_function: None,
            current_local_evar_ids: HashSet::new(),
            function_object_types: HashMap::new(),
        }
    }

    // CAST-2: the un-disambiguated label spelling for a raw id_to_name entry. label_name resolves through this, then through label_disambig for colliding ids.
    fn label_name_base(raw: &str) -> String {
        if raw.chars().all(|c| c.is_ascii_digit()) {
            format!("L{}", raw)
        } else {
            sanitize_c_ident(raw)
        }
    }

    // CAST-2: resolve label ids that sanitize to the same identifier; the smallest id in a colliding group keeps the plain name and the rest get <base>_<id>, sorted so the choice is order-independent.
    fn build_label_disambig(id_to_name: &HashMap<usize, String>) -> HashMap<usize, String> {
        let mut by_base: HashMap<String, Vec<usize>> = HashMap::new();
        for (id, raw) in id_to_name {
            by_base
                .entry(Self::label_name_base(raw))
                .or_default()
                .push(*id);
        }
        let mut out: HashMap<usize, String> = HashMap::new();
        for (base, mut ids) in by_base {
            if ids.len() <= 1 {
                continue;
            }
            ids.sort();
            for &id in ids.iter().skip(1) {
                out.insert(id, format!("{}_{}", base, id));
            }
        }
        out
    }

    pub fn record_var_type(&mut self, var_name: &str, ty: CType) {
        self.var_types.entry(var_name.to_string()).or_insert(ty);
    }

    pub(crate) fn enter_function(
        &mut self,
        address: Address,
        local_evar_ids: Option<&HashSet<Ident>>,
    ) {
        self.current_function = Some(address);
        self.current_local_evar_ids.clear();
        if let Some(ids) = local_evar_ids {
            self.current_local_evar_ids.extend(ids.iter().copied());
        }
    }

    fn is_current_local_evar(&self, id: Ident) -> bool {
        self.current_local_evar_ids.contains(&id)
    }

    fn local_evar_name(&self, id: Ident) -> String {
        format!("var_{}", id)
    }

    fn record_current_function_object_type(&mut self, name: String, mut ty: CType) {
        let Some(address) = self.current_function else {
            return;
        };
        if let CType::Function(..) = ty {
            ty = CType::ptr(ty);
        }
        if !ctype_involves_function(&ty) {
            return;
        }
        self.function_object_types
            .entry(address)
            .or_default()
            .entry(name)
            .or_insert(ty);
    }

    pub(crate) fn function_object_types(&self) -> &FunctionObjectTypes {
        &self.function_object_types
    }

    pub fn temp_name(&mut self, id: usize) -> String {
        if let Some(name) = self.temp_names.get(&id) {
            return name.clone();
        }
        let name = format!("var_{}", id);
        self.temp_names.insert(id, name.clone());
        name
    }

    pub fn var_name(&self, id: usize) -> String {
        self.id_to_name
            .get(&id)
            .map(|name| sanitize_c_symbol_name(name))
            .unwrap_or_else(|| format!("var_{}", id))
    }

    pub fn label_name(&self, id: usize) -> String {
        // CAST-2: a colliding id resolves through the per-context disambiguation map (built once in new); Slabel(id) and Sgoto(id) hit the same entry, so a goto and its target always agree.
        if let Some(name) = self.label_disambig.get(&id) {
            return name.clone();
        }
        match self.id_to_name.get(&id) {
            Some(raw) => Self::label_name_base(raw),
            None => format!("L{}", id),
        }
    }

    pub fn resolve_l_label(&self, name: &str) -> String {
        let hex_str = if name.starts_with("L_") {
            &name[2..]
        } else {
            return name.to_string();
        };
        if hex_str.is_empty() || !hex_str.chars().all(|c| c.is_ascii_hexdigit()) {
            return sanitize_c_ident(name);
        }
        let addr = match u64::from_str_radix(hex_str, 16) {
            Ok(a) => a,
            Err(_) => return sanitize_c_ident(name),
        };
        match self
            .func_addrs_sorted
            .binary_search_by_key(&addr, |(a, _)| *a)
        {
            Ok(idx) => sanitize_c_symbol_name(&self.func_addrs_sorted[idx].1),
            Err(idx) => {
                if idx > 0 {
                    sanitize_c_symbol_name(&self.func_addrs_sorted[idx - 1].1)
                } else {
                    sanitize_c_ident(name)
                }
            }
        }
    }
}

fn convert_unary_op(op: &clight::ClightUnaryOp) -> UnaryOp {
    match op {
        clight::ClightUnaryOp::Onotbool => UnaryOp::Not,
        clight::ClightUnaryOp::Onotint => UnaryOp::BitNot,
        clight::ClightUnaryOp::Oneg => UnaryOp::Neg,
        clight::ClightUnaryOp::Oabsfloat => UnaryOp::Plus,
    }
}

fn cexpr_is_pointer(e: &CExpr, types: &HashMap<String, CType>) -> bool {
    match e {
        CExpr::Cast(ty, _) => ty.is_pointer(),
        CExpr::Var(name) => types.get(name).map_or(false, |t| t.is_pointer()),
        CExpr::Paren(inner) => cexpr_is_pointer(inner, types),
        CExpr::Unary(UnaryOp::AddrOf, _) => true,
        // ptr +/- int yields a pointer; comparisons/other ops do not.
        CExpr::Binary(BinaryOp::Add | BinaryOp::Sub, l, r) => {
            cexpr_is_pointer(l, types) || cexpr_is_pointer(r, types)
        }
        _ => false,
    }
}

// Cast pointer operands of machine address arithmetic to long so the arithmetic is valid C; MEASURED non-subsumable (bypassing it regressed +3010 invalid-binop) and disjoint from decl-level force_long.
fn repair_pointer_arith(
    op: BinaryOp,
    lhs: CExpr,
    lhs_ptr: bool,
    rhs: CExpr,
    rhs_ptr: bool,
) -> (CExpr, CExpr) {
    let to_long = |e: CExpr| CExpr::Cast(CType::long(), Box::new(e));
    match op {
        BinaryOp::Mul
        | BinaryOp::Div
        | BinaryOp::Mod
        | BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr => (
            if lhs_ptr { to_long(lhs) } else { lhs },
            if rhs_ptr { to_long(rhs) } else { rhs },
        ),
        // ptr + ptr is invalid; ptr - ptr we normalize to an integer (byte) difference.
        BinaryOp::Add | BinaryOp::Sub if lhs_ptr && rhs_ptr => (to_long(lhs), to_long(rhs)),
        // int - ptr is invalid C (a pointer may only be subtracted from a pointer); cast the pointer operand to long so it becomes integer subtraction (ptr - int and ptr + int are kept).
        BinaryOp::Sub if rhs_ptr && !lhs_ptr => (lhs, to_long(rhs)),
        _ => (lhs, rhs),
    }
}

// Walk an expression bottom-up, repairing invalid pointer arithmetic at every Binary node. The Clight type annotation carried by an expression node (every variant's last field).
fn clight_expr_ctype(e: &clight::ClightExpr) -> &clight::ClightType {
    use clight::ClightExpr::*;
    match e {
        EconstInt(_, t)
        | EconstFloat(_, t)
        | EconstSingle(_, t)
        | EconstLong(_, t)
        | Evar(_, t)
        | EvarSymbol(_, t)
        | Etempvar(_, t)
        | Ederef(_, t)
        | Eaddrof(_, t)
        | Eunop(_, _, t)
        | Ebinop(_, _, _, t)
        | Ecast(_, t)
        | Efield(_, _, t)
        | Esizeof(_, t)
        | Ealignof(_, t)
        | Econdition(_, _, _, t) => t,
    }
}

/// Strip a materialized annotation cast on a bare variable when the final decl renders it redundant: `(T *)v` with v declared exactly `T *` is just `v`.
fn strip_matching_var_cast(e: CExpr, types: &HashMap<String, CType>) -> CExpr {
    if let CExpr::Cast(cty, inner) = &e {
        if let CExpr::Var(name) = inner.as_ref() {
            if types.get(name) == Some(cty) {
                return (**inner).clone();
            }
        }
    }
    e
}

/// A string literal is a char * base that code indexes into, never the integer offset; paren/cast-transparent, targeting the case cexpr_is_pointer does not classify as a pointer at all.
fn is_string_base(e: &CExpr) -> bool {
    match e {
        CExpr::StringLit(_) => true,
        CExpr::Paren(inner) | CExpr::Cast(_, inner) => is_string_base(inner),
        _ => false,
    }
}

/// An "indexable base" of pointer arithmetic that must stay a pointer: a string literal, &x, or an array-declared Var; the caller integerizes the OTHER operand instead of scalarizing both.
fn is_indexable_base(e: &CExpr, types: &HashMap<String, CType>) -> bool {
    match e {
        CExpr::Unary(UnaryOp::AddrOf, _) => true,
        CExpr::Var(name) => matches!(types.get(name), Some(CType::Array(..))),
        // A cast whose TARGET is a pointer type is itself a pointer base, even when an outer integer cast later scalarizes it (`(long)(T *)p`): recursing lets the inner `(T *)` cast still register as the base.
        CExpr::Cast(ty, inner) => ty.is_pointer() || is_indexable_base(inner, types),
        CExpr::Paren(inner) => is_indexable_base(inner, types),
        _ => is_string_base(e),
    }
}

/// Undo a scalarizing integer cast wrapping a pointer base -- `(long)(T *)p` -> `(T *)p` -- so an address base survives as a pointer through additive arithmetic. Strips only when the outer cast is integer and the inner is itself an indexable (pointer) base; otherwise returns `e` unchanged.
fn strip_scalarizing_cast(e: CExpr, types: &HashMap<String, CType>) -> CExpr {
    if let CExpr::Cast(ty, inner) = &e {
        if ty.is_integer() && is_indexable_base(inner, types) {
            return (**inner).clone();
        }
    }
    e
}

/// Peel leading casts/parens to the underlying value, so a mis-typed pointer index `(long)(void *)v` can be re-expressed cleanly as `(long)v`.
fn peel_casts(e: &CExpr) -> &CExpr {
    match e {
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => peel_casts(inner),
        _ => e,
    }
}

/// True when the innermost pointer cast of `e` is `void *` -- the degenerate pointer type a mis-typed integer index usually carries.
fn inner_ptr_is_void(e: &CExpr) -> bool {
    match e {
        CExpr::Cast(CType::Pointer(inner, _), _) if matches!(**inner, CType::Void) => true,
        CExpr::Cast(_, inner) | CExpr::Paren(inner) => inner_ptr_is_void(inner),
        _ => false,
    }
}

/// True when a pointee type SCALES under C pointer arithmetic (stride > 1): anything but char (stride 1) and void (GNU stride 1) and an array pointee. A scaling pointee is what makes a raw byte offset get re-multiplied.
fn pointee_scales_ctype(pointee: &CType) -> bool {
    !matches!(
        pointee,
        CType::Void | CType::Int(IntSize::Char, _) | CType::Array(..)
    )
}

/// True when e is the char*-rebased byte-address shape this fix materializes, returning the inner char* expression and T *, so a further byte offset FOLDS in instead of re-scaling.
fn peel_char_byte_rebase(e: &CExpr) -> Option<(CType, CExpr)> {
    let (ptr_ty, inner) = match e {
        CExpr::Cast(CType::Pointer(pointee, q), inner) if pointee_scales_ctype(pointee) => {
            (CType::Pointer(pointee.clone(), *q), inner.as_ref())
        }
        _ => return None,
    };
    let byte_expr = match inner {
        CExpr::Paren(p) => p.as_ref(),
        other => other,
    };
    // The byte expression must itself be additive arithmetic whose left operand is a `(char *)` base, i.e. already byte-addressed.
    if let CExpr::Binary(BinaryOp::Add | BinaryOp::Sub, l, _) = byte_expr {
        let l_is_char_ptr = matches!(
            l.as_ref(),
            CExpr::Cast(CType::Pointer(p, _), _) if matches!(p.as_ref(), CType::Int(IntSize::Char, _))
        );
        if l_is_char_ptr {
            return Some((ptr_ty, byte_expr.clone()));
        }
    }
    None
}

/// A pointer base whose sibling integral operand is a raw BYTE offset that must not be re-scaled: a bare T * Var whose B2 rebase did not fire, or the char*-rebased shape this fix produces.
fn scaling_ptr_byte_base(
    e: &CExpr,
    types: &HashMap<String, CType>,
) -> Option<(CType, Option<CExpr>)> {
    match e {
        CExpr::Paren(inner) => scaling_ptr_byte_base(inner, types),
        CExpr::Var(name) => match types.get(name) {
            Some(CType::Pointer(pointee, _)) if pointee_scales_ctype(pointee) => {
                Some((types.get(name).cloned().unwrap(), None))
            }
            _ => None,
        },
        _ => peel_char_byte_rebase(e).map(|(ty, inner)| (ty, Some(inner))),
    }
}

/// Apply a raw BYTE offset to a scaling pointer base exactly once through char*, folding into an already-rebased inner expression; byte-correct under any pointee size since char* has stride 1.
fn rebase_byte_add_through_char(
    op: BinaryOp,
    base: CExpr,
    inner_byte_expr: Option<CExpr>,
    offset: CExpr,
    ptr_ty: CType,
) -> CExpr {
    let off = CExpr::Cast(CType::long(), Box::new(offset));
    let byte_addr = match inner_byte_expr {
        Some(inner) => CExpr::Binary(op, Box::new(inner), Box::new(off)),
        None => {
            let base_char = CExpr::Cast(CType::ptr(CType::char_signed()), Box::new(base));
            CExpr::Binary(op, Box::new(base_char), Box::new(off))
        }
    };
    CExpr::Cast(ptr_ty, Box::new(CExpr::Paren(Box::new(byte_addr))))
}

/// CType for a memory chunk (element type of a runtime-indexed stack array).
fn ctype_from_memchunk(chunk: &MemoryChunk) -> CType {
    use crate::decompile::passes::c_pass::types::{FloatSize, IntSize, Signedness};
    match chunk {
        MemoryChunk::MBool => CType::Bool,
        MemoryChunk::MInt8Signed => CType::Int(IntSize::Char, Signedness::Signed),
        MemoryChunk::MInt8Unsigned => CType::Int(IntSize::Char, Signedness::Unsigned),
        MemoryChunk::MInt16Signed => CType::Int(IntSize::Short, Signedness::Signed),
        MemoryChunk::MInt16Unsigned => CType::Int(IntSize::Short, Signedness::Unsigned),
        MemoryChunk::MInt32 | MemoryChunk::MAny32 => CType::Int(IntSize::Int, Signedness::Signed),
        MemoryChunk::MFloat32 => CType::Float(FloatSize::Float),
        MemoryChunk::MFloat64 => CType::Float(FloatSize::Double),
        _ => CType::long(),
    }
}

fn repair_arith_expr(e: &CExpr, types: &HashMap<String, CType>) -> CExpr {
    // &arr on an array-typed local is pointer-to-array, so &arr + i strides by whole arrays; rewrite it to the bare name, which decays to an element pointer.
    if let CExpr::Unary(UnaryOp::AddrOf, inner) = e {
        if let CExpr::Var(name) = inner.as_ref() {
            if matches!(types.get(name), Some(CType::Array(..))) {
                return CExpr::Var(name.clone());
            }
        }
    }
    match e {
        CExpr::Binary(op, l, r) => {
            let l_raw = repair_arith_expr(l, types);
            let r_raw = repair_arith_expr(r, types);
            // Byte-offset re-scaling fix on the UN-stripped children: re-express a scaling base plus a raw byte offset through char*, since stripping would forge a bare Var and hide a genuine element index.
            if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                let l_byte_base = scaling_ptr_byte_base(&l_raw, types);
                let r_byte_base = scaling_ptr_byte_base(&r_raw, types);
                match (l_byte_base, r_byte_base) {
                    (Some((lty, inner)), None) if !cexpr_is_pointer(&r_raw, types) => {
                        return rebase_byte_add_through_char(*op, l_raw, inner, r_raw, lty);
                    }
                    // offset + base (Add is commutative; Sub of int - ptr is not a base subtraction).
                    (None, Some((rty, inner)))
                        if matches!(op, BinaryOp::Add) && !cexpr_is_pointer(&l_raw, types) =>
                    {
                        return rebase_byte_add_through_char(*op, r_raw, inner, l_raw, rty);
                    }
                    _ => {}
                }
            }
            let l = strip_matching_var_cast(l_raw, types);
            let r = r_raw;
            // An indexable base added to a pointer-TYPED operand means that operand is really the integer offset, so keep the base and cast the offset to long rather than scalarizing both.
            if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
                let l_base = is_indexable_base(&l, types);
                let r_base = is_indexable_base(&r, types);
                let to_long = |e: CExpr| CExpr::Cast(CType::long(), Box::new(e));
                if l_base ^ r_base {
                    match op {
                        // base +/- offset / offset + base: keep the base, integerize the pointer offset.
                        BinaryOp::Add if l_base && cexpr_is_pointer(&r, types) => {
                            return CExpr::Binary(*op, Box::new(l), Box::new(to_long(r)));
                        }
                        BinaryOp::Add if r_base && cexpr_is_pointer(&l, types) => {
                            return CExpr::Binary(*op, Box::new(to_long(l)), Box::new(r));
                        }
                        BinaryOp::Sub if l_base && cexpr_is_pointer(&r, types) => {
                            return CExpr::Binary(*op, Box::new(l), Box::new(to_long(r)));
                        }
                        // ptr - base: a pointer minus a base is a mismatched difference; normalize to an integer one.
                        BinaryOp::Sub if r_base && cexpr_is_pointer(&l, types) => {
                            return CExpr::Binary(*op, Box::new(to_long(l)), Box::new(to_long(r)));
                        }
                        // A scalarized pointer base plus a plain integer offset: the outer (long) hid the pointer from the arms above, so strip the base's scalarizing cast to keep base + off a T * + long.
                        BinaryOp::Add if l_base && !cexpr_is_pointer(&r, types) => {
                            return CExpr::Binary(
                                *op,
                                Box::new(strip_scalarizing_cast(l, types)),
                                Box::new(r),
                            );
                        }
                        BinaryOp::Add if r_base && !cexpr_is_pointer(&l, types) => {
                            return CExpr::Binary(
                                *op,
                                Box::new(l),
                                Box::new(strip_scalarizing_cast(r, types)),
                            );
                        }
                        BinaryOp::Sub if l_base && !cexpr_is_pointer(&r, types) => {
                            return CExpr::Binary(
                                *op,
                                Box::new(strip_scalarizing_cast(l, types)),
                                Box::new(r),
                            );
                        }
                        _ => {}
                    }
                }
                // Both operands carry pointer casts, and ptr + ptr is never valid C, so keep the typed pointer as the base and integerize the void*-cast operand.
                if matches!(op, BinaryOp::Add)
                    && is_indexable_base(&l, types)
                    && is_indexable_base(&r, types)
                {
                    let l_void = inner_ptr_is_void(&l);
                    let r_void = inner_ptr_is_void(&r);
                    if r_void && !l_void {
                        return CExpr::Binary(
                            *op,
                            Box::new(strip_scalarizing_cast(l, types)),
                            Box::new(to_long(peel_casts(&r).clone())),
                        );
                    }
                    if l_void && !r_void {
                        return CExpr::Binary(
                            *op,
                            Box::new(to_long(peel_casts(&l).clone())),
                            Box::new(strip_scalarizing_cast(r, types)),
                        );
                    }
                }
            }
            let lp = cexpr_is_pointer(&l, types);
            let rp = cexpr_is_pointer(&r, types);
            let (l, r) = repair_pointer_arith(*op, l, lp, r, rp);
            CExpr::Binary(*op, Box::new(l), Box::new(r))
        }
        CExpr::Unary(op, inner) => {
            let inner = repair_arith_expr(inner, types);
            // `-ptr` and `~ptr` are invalid C; cast the pointer operand to long (Deref/AddrOf/Not are valid on pointers and left alone).
            let inner = if matches!(op, UnaryOp::Neg | UnaryOp::BitNot)
                && cexpr_is_pointer(&inner, types)
            {
                CExpr::Cast(CType::long(), Box::new(inner))
            } else {
                inner
            };
            // Strip the materialized access cast (Ederef conversion) when the final decl makes it redundant: `*(T *)v` with v declared exactly `T *` prints `*v`.
            let inner = if matches!(op, UnaryOp::Deref) {
                strip_matching_var_cast(inner, types)
            } else {
                inner
            };
            CExpr::Unary(*op, Box::new(inner))
        }
        CExpr::Assign(op, l, r) => CExpr::Assign(
            *op,
            Box::new(repair_arith_expr(l, types)),
            Box::new(repair_arith_expr(r, types)),
        ),
        CExpr::Ternary(c, t, f) => CExpr::Ternary(
            Box::new(repair_arith_expr(c, types)),
            Box::new(repair_arith_expr(t, types)),
            Box::new(repair_arith_expr(f, types)),
        ),
        CExpr::Call(f, args) => CExpr::Call(
            Box::new(repair_arith_expr(f, types)),
            args.iter().map(|a| repair_arith_expr(a, types)).collect(),
        ),
        CExpr::Cast(ty, inner) => {
            CExpr::Cast(ty.clone(), Box::new(repair_arith_expr(inner, types)))
        }
        CExpr::Member(inner, fld) => {
            // `e.field` where e is a pointer must be `e->field`; type recovery sometimes declares the base as a pointer while emitting a value member access ("X is a pointer; use ->").
            let inner = repair_arith_expr(inner, types);
            if cexpr_is_pointer(&inner, types) {
                CExpr::MemberPtr(Box::new(inner), fld.clone())
            } else {
                CExpr::Member(Box::new(inner), fld.clone())
            }
        }
        CExpr::MemberPtr(inner, fld) => {
            CExpr::MemberPtr(Box::new(repair_arith_expr(inner, types)), fld.clone())
        }
        CExpr::Index(a, i) => CExpr::Index(
            Box::new(repair_arith_expr(a, types)),
            Box::new(repair_arith_expr(i, types)),
        ),
        CExpr::SizeofExpr(inner) => CExpr::SizeofExpr(Box::new(repair_arith_expr(inner, types))),
        CExpr::Paren(inner) => CExpr::Paren(Box::new(repair_arith_expr(inner, types))),
        CExpr::StmtExpr(stmts, inner) => CExpr::StmtExpr(
            stmts.iter().map(|s| repair_arith_stmt(s, types)).collect(),
            Box::new(repair_arith_expr(inner, types)),
        ),
        other => other.clone(),
    }
}

// Walk a statement tree, applying repair_arith_expr to every contained expression.
fn repair_arith_stmt(stmt: &CStmt, types: &HashMap<String, CType>) -> CStmt {
    match stmt {
        CStmt::Expr(e) => CStmt::Expr(repair_arith_expr(e, types)),
        CStmt::Block(items) => CStmt::Block(
            items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => CBlockItem::Stmt(repair_arith_stmt(s, types)),
                    other => other.clone(),
                })
                .collect(),
        ),
        CStmt::If(cond, then_s, else_s) => CStmt::If(
            repair_arith_expr(cond, types),
            Box::new(repair_arith_stmt(then_s, types)),
            else_s
                .as_ref()
                .map(|e| Box::new(repair_arith_stmt(e, types))),
        ),
        CStmt::While(cond, body) => CStmt::While(
            repair_arith_expr(cond, types),
            Box::new(repair_arith_stmt(body, types)),
        ),
        CStmt::DoWhile(body, cond) => CStmt::DoWhile(
            Box::new(repair_arith_stmt(body, types)),
            repair_arith_expr(cond, types),
        ),
        CStmt::For(init, cond, update, body) => CStmt::For(
            init.clone(),
            cond.as_ref().map(|c| repair_arith_expr(c, types)),
            update.as_ref().map(|u| repair_arith_expr(u, types)),
            Box::new(repair_arith_stmt(body, types)),
        ),
        CStmt::Return(Some(e)) => CStmt::Return(Some(repair_arith_expr(e, types))),
        CStmt::Switch(e, body) => {
            // `switch (ptr)` is invalid C (switch quantity not an integer); cast to long.
            let e = repair_arith_expr(e, types);
            let e = if cexpr_is_pointer(&e, types) {
                CExpr::Cast(CType::long(), Box::new(e))
            } else {
                e
            };
            CStmt::Switch(e, Box::new(repair_arith_stmt(body, types)))
        }
        CStmt::Labeled(lbl, inner) => {
            CStmt::Labeled(lbl.clone(), Box::new(repair_arith_stmt(inner, types)))
        }
        CStmt::Sequence(stmts) => {
            CStmt::Sequence(stmts.iter().map(|s| repair_arith_stmt(s, types)).collect())
        }
        other => other.clone(),
    }
}

// Compilability: insert explicit casts at int<->pointer mismatches, which modern clang and gcc reject as hard errors even under -w; float<->pointer and unknown-class operands are left alone.

#[derive(PartialEq, Eq, Clone, Copy)]
enum ScalarClass {
    Ptr,
    Int,
    Float,
    Other,
}

fn ctype_scalar_class(t: &CType) -> ScalarClass {
    match t {
        CType::Pointer(..) | CType::Array(..) => ScalarClass::Ptr,
        CType::Int(..) | CType::Bool | CType::Enum(..) => ScalarClass::Int,
        CType::Float(..) => ScalarClass::Float,
        _ => ScalarClass::Other,
    }
}

// Best-effort static class of an expression's value; Other == unknown (left untouched).
fn cexpr_scalar_class(
    e: &CExpr,
    types: &HashMap<String, CType>,
    callee_ret: &HashMap<String, CType>,
) -> ScalarClass {
    match e {
        CExpr::Cast(ty, _) => ctype_scalar_class(ty),
        CExpr::Var(name) => types
            .get(name)
            .map(ctype_scalar_class)
            .unwrap_or(ScalarClass::Other),
        CExpr::Paren(inner) => cexpr_scalar_class(inner, types, callee_ret),
        CExpr::IntLit(_) => ScalarClass::Int,
        CExpr::FloatLit(_) => ScalarClass::Float,
        CExpr::StringLit(_) => ScalarClass::Ptr,
        CExpr::Unary(UnaryOp::AddrOf, _) => ScalarClass::Ptr,
        CExpr::Unary(UnaryOp::Neg | UnaryOp::BitNot | UnaryOp::Not | UnaryOp::Plus, _) => {
            ScalarClass::Int
        }
        CExpr::Binary(BinaryOp::Add | BinaryOp::Sub, l, r) => {
            if cexpr_scalar_class(l, types, callee_ret) == ScalarClass::Ptr
                || cexpr_scalar_class(r, types, callee_ret) == ScalarClass::Ptr
            {
                ScalarClass::Ptr
            } else {
                ScalarClass::Int
            }
        }
        CExpr::Binary(..) => ScalarClass::Int,
        CExpr::Call(f, _) => match f.as_ref() {
            CExpr::Var(n) => callee_ret
                .get(n)
                .map(ctype_scalar_class)
                .unwrap_or(ScalarClass::Other),
            _ => ScalarClass::Other,
        },
        _ => ScalarClass::Other,
    }
}

// Best-effort exact type for scalar expressions whose type is explicit in the
// C AST. This intentionally stays narrower than full expression inference.
fn cexpr_known_ctype(
    e: &CExpr,
    types: &HashMap<String, CType>,
    callee_ret: &HashMap<String, CType>,
) -> Option<CType> {
    match e {
        CExpr::Cast(ty, _) => Some(ty.clone()),
        CExpr::Var(name) => types.get(name).cloned(),
        CExpr::Paren(inner) => cexpr_known_ctype(inner, types, callee_ret),
        CExpr::Unary(UnaryOp::AddrOf, inner) => {
            cexpr_known_ctype(inner, types, callee_ret).map(CType::ptr)
        }
        CExpr::Call(f, _) => match f.as_ref() {
            CExpr::Var(name) => callee_ret.get(name).cloned(),
            _ => None,
        },
        _ => None,
    }
}

// Wrap `e` in a cast to `target` at a genuine pointer<->integer mismatch, or
// when two statically-known pointer types differ. The latter is codegen-neutral
// in C and required when the same output is compiled as C++.
fn coerce_scalar(
    target: &CType,
    e: CExpr,
    types: &HashMap<String, CType>,
    callee_ret: &HashMap<String, CType>,
) -> CExpr {
    let target_class = ctype_scalar_class(target);
    let source_class = cexpr_scalar_class(&e, types, callee_ret);
    let scalar_mismatch = matches!(
        (target_class, source_class),
        (ScalarClass::Ptr, ScalarClass::Int) | (ScalarClass::Int, ScalarClass::Ptr)
    );
    let pointer_type_mismatch = target_class == ScalarClass::Ptr
        && source_class == ScalarClass::Ptr
        && cexpr_known_ctype(&e, types, callee_ret)
            .map(|source| source != *target)
            .unwrap_or(false);
    if !scalar_mismatch && !pointer_type_mismatch {
        return e;
    }
    // 0 is a valid null-pointer constant and a valid integer; it never needs a cast.
    if let CExpr::IntLit(l) = &e {
        if l.value == 0 {
            return e;
        }
    }
    CExpr::Cast(target.clone(), Box::new(e))
}

type StructFieldTypes = HashMap<String, HashMap<String, CType>>;

struct CastTypes<'a> {
    variables: &'a HashMap<String, CType>,
    struct_fields: &'a StructFieldTypes,
}

fn aggregate_field_ctype(
    aggregate: &CType,
    field: &str,
    struct_fields: &StructFieldTypes,
) -> Option<CType> {
    match aggregate {
        CType::Struct(name) | CType::Union(name) => struct_fields.get(name)?.get(field).cloned(),
        _ => None,
    }
}

// A cast is useful exact evidence for a member base even though the cast
// expression is not itself an assignable lvalue: `((struct S *)p)->field` is
// the common recovered form.
fn member_base_ctype(e: &CExpr, types: &CastTypes<'_>) -> Option<CType> {
    match e {
        CExpr::Cast(ty, _) => Some(ty.clone()),
        _ => lvalue_ctype(e, types),
    }
}

// Type of an lvalue, for casting an assignment's RHS to it. Field lookup is
// deliberately limited to statically named recovered structs; index and other
// expression forms remain unknown rather than triggering broad inference.
fn lvalue_ctype(e: &CExpr, types: &CastTypes<'_>) -> Option<CType> {
    match e {
        CExpr::Var(n) => types.variables.get(n).cloned(),
        CExpr::Paren(inner) => lvalue_ctype(inner, types),
        CExpr::Unary(UnaryOp::Deref, inner) => match member_base_ctype(inner, types) {
            Some(CType::Pointer(t, _)) => Some(*t),
            _ => None,
        },
        CExpr::Member(inner, field) => {
            let aggregate = member_base_ctype(inner, types)?;
            aggregate_field_ctype(&aggregate, field, types.struct_fields)
        }
        CExpr::MemberPtr(inner, field) => match member_base_ctype(inner, types) {
            Some(CType::Pointer(aggregate, _)) => {
                aggregate_field_ctype(&aggregate, field, types.struct_fields)
            }
            _ => None,
        },
        _ => None,
    }
}

// name -> (param types, is_variadic, is_unspecified_K&R)
type CalleeParams = HashMap<String, (Vec<CType>, bool, bool)>;

fn insert_casts_expr(
    e: &CExpr,
    types: &CastTypes<'_>,
    callee_params: &CalleeParams,
    callee_ret: &HashMap<String, CType>,
) -> CExpr {
    match e {
        CExpr::Call(f, args) => {
            let nf = insert_casts_expr(f, types, callee_params, callee_ret);
            let mut nargs: Vec<CExpr> = args
                .iter()
                .map(|a| insert_casts_expr(a, types, callee_params, callee_ret))
                .collect();
            if let CExpr::Var(fname) = &nf {
                if let Some((ptypes, _variadic, unspecified)) = callee_params.get(fname) {
                    // K&R `f()` is unchecked; for variadics, ptypes covers only the fixed leading parameters, so `ptypes.get(i)` naturally leaves the variadic tail untouched.
                    if !unspecified {
                        for (i, a) in nargs.iter_mut().enumerate() {
                            if let Some(pt) = ptypes.get(i) {
                                let arg = std::mem::replace(a, CExpr::int(0));
                                *a = coerce_scalar(pt, arg, types.variables, callee_ret);
                            }
                        }
                    }
                }
            }
            CExpr::Call(Box::new(nf), nargs)
        }
        CExpr::Assign(op, lhs, rhs) => {
            let nlhs = insert_casts_expr(lhs, types, callee_params, callee_ret);
            let nrhs = insert_casts_expr(rhs, types, callee_params, callee_ret);
            if *op == AssignOp::Assign {
                if let Some(lty) = lvalue_ctype(&nlhs, types) {
                    let nrhs = coerce_scalar(&lty, nrhs, types.variables, callee_ret);
                    return CExpr::Assign(*op, Box::new(nlhs), Box::new(nrhs));
                }
            }
            CExpr::Assign(*op, Box::new(nlhs), Box::new(nrhs))
        }
        CExpr::Unary(op, i) => CExpr::Unary(
            *op,
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
        ),
        CExpr::Binary(op, l, r) => CExpr::Binary(
            *op,
            Box::new(insert_casts_expr(l, types, callee_params, callee_ret)),
            Box::new(insert_casts_expr(r, types, callee_params, callee_ret)),
        ),
        CExpr::Ternary(c, t, f) => CExpr::Ternary(
            Box::new(insert_casts_expr(c, types, callee_params, callee_ret)),
            Box::new(insert_casts_expr(t, types, callee_params, callee_ret)),
            Box::new(insert_casts_expr(f, types, callee_params, callee_ret)),
        ),
        CExpr::Cast(ty, i) => CExpr::Cast(
            ty.clone(),
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
        ),
        CExpr::Member(i, fld) => CExpr::Member(
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
            fld.clone(),
        ),
        CExpr::MemberPtr(i, fld) => CExpr::MemberPtr(
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
            fld.clone(),
        ),
        CExpr::Index(a, i) => CExpr::Index(
            Box::new(insert_casts_expr(a, types, callee_params, callee_ret)),
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
        ),
        CExpr::SizeofExpr(i) => CExpr::SizeofExpr(Box::new(insert_casts_expr(
            i,
            types,
            callee_params,
            callee_ret,
        ))),
        CExpr::Paren(i) => CExpr::Paren(Box::new(insert_casts_expr(
            i,
            types,
            callee_params,
            callee_ret,
        ))),
        CExpr::StmtExpr(stmts, i) => CExpr::StmtExpr(
            stmts
                .iter()
                .map(|s| insert_casts_stmt(s, types, callee_params, callee_ret, &CType::Void))
                .collect(),
            Box::new(insert_casts_expr(i, types, callee_params, callee_ret)),
        ),
        other => other.clone(),
    }
}

fn insert_casts_stmt(
    stmt: &CStmt,
    types: &CastTypes<'_>,
    callee_params: &CalleeParams,
    callee_ret: &HashMap<String, CType>,
    ret_type: &CType,
) -> CStmt {
    match stmt {
        CStmt::Expr(e) => CStmt::Expr(insert_casts_expr(e, types, callee_params, callee_ret)),
        CStmt::Block(items) => CStmt::Block(
            items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => CBlockItem::Stmt(insert_casts_stmt(
                        s,
                        types,
                        callee_params,
                        callee_ret,
                        ret_type,
                    )),
                    other => other.clone(),
                })
                .collect(),
        ),
        CStmt::If(c, t, e) => CStmt::If(
            insert_casts_expr(c, types, callee_params, callee_ret),
            Box::new(insert_casts_stmt(
                t,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
            e.as_ref().map(|x| {
                Box::new(insert_casts_stmt(
                    x,
                    types,
                    callee_params,
                    callee_ret,
                    ret_type,
                ))
            }),
        ),
        CStmt::While(c, b) => CStmt::While(
            insert_casts_expr(c, types, callee_params, callee_ret),
            Box::new(insert_casts_stmt(
                b,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
        ),
        CStmt::DoWhile(b, c) => CStmt::DoWhile(
            Box::new(insert_casts_stmt(
                b,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
            insert_casts_expr(c, types, callee_params, callee_ret),
        ),
        CStmt::For(init, c, u, b) => CStmt::For(
            init.clone(),
            c.as_ref()
                .map(|x| insert_casts_expr(x, types, callee_params, callee_ret)),
            u.as_ref()
                .map(|x| insert_casts_expr(x, types, callee_params, callee_ret)),
            Box::new(insert_casts_stmt(
                b,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
        ),
        CStmt::Switch(e, b) => CStmt::Switch(
            insert_casts_expr(e, types, callee_params, callee_ret),
            Box::new(insert_casts_stmt(
                b,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
        ),
        CStmt::Labeled(lbl, inner) => CStmt::Labeled(
            lbl.clone(),
            Box::new(insert_casts_stmt(
                inner,
                types,
                callee_params,
                callee_ret,
                ret_type,
            )),
        ),
        CStmt::Sequence(ss) => CStmt::Sequence(
            ss.iter()
                .map(|s| insert_casts_stmt(s, types, callee_params, callee_ret, ret_type))
                .collect(),
        ),
        CStmt::Return(Some(e)) => {
            let ne = insert_casts_expr(e, types, callee_params, callee_ret);
            let ne = if matches!(ret_type, CType::Void) {
                ne
            } else {
                coerce_scalar(ret_type, ne, types.variables, callee_ret)
            };
            CStmt::Return(Some(ne))
        }
        other => other.clone(),
    }
}

// Whether a referenced-but-undeclared identifier should become a file-scope object; synthesized local names are excluded, since masking one with a global would hide the real recovery gap.
fn is_global_like_name(n: &str) -> bool {
    if n.is_empty() {
        return false;
    }
    if n.starts_with("var_") || n.starts_with("tmp") {
        return false;
    }
    // Compiler builtins (the opaque markers for un-encodable ops) are neither file-scope objects nor locals, and declaring one as a tentative long would let the optimizer constant-fold it.
    if n.starts_with("__builtin_") {
        return false;
    }
    if n.starts_with('p') && n.len() >= 2 && n[1..].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    true
}

fn convert_binary_op(op: &clight::ClightBinaryOp) -> BinaryOp {
    match op {
        clight::ClightBinaryOp::Oadd => BinaryOp::Add,
        clight::ClightBinaryOp::Osub => BinaryOp::Sub,
        clight::ClightBinaryOp::Omul => BinaryOp::Mul,
        clight::ClightBinaryOp::Odiv => BinaryOp::Div,
        clight::ClightBinaryOp::Omod => BinaryOp::Mod,
        clight::ClightBinaryOp::Oand => BinaryOp::BitAnd,
        clight::ClightBinaryOp::Oor => BinaryOp::BitOr,
        clight::ClightBinaryOp::Oxor => BinaryOp::BitXor,
        clight::ClightBinaryOp::Oshl => BinaryOp::Shl,
        clight::ClightBinaryOp::Oshr => BinaryOp::Shr,
        clight::ClightBinaryOp::Oeq => BinaryOp::Eq,
        clight::ClightBinaryOp::One => BinaryOp::Ne,
        clight::ClightBinaryOp::Olt => BinaryOp::Lt,
        clight::ClightBinaryOp::Ogt => BinaryOp::Gt,
        clight::ClightBinaryOp::Ole => BinaryOp::Le,
        clight::ClightBinaryOp::Oge => BinaryOp::Ge,
    }
}

pub fn convert_expr(expr: &clight::ClightExpr, ctx: &mut ConversionContext) -> CExpr {
    match expr {
        clight::ClightExpr::EconstInt(val, _ty) => CExpr::IntLit(IntLiteral {
            value: *val as i128,
            suffix: IntLiteralSuffix::None,
            base: IntLiteralBase::Decimal,
        }),
        clight::ClightExpr::EconstLong(val, _ty) => CExpr::IntLit(IntLiteral {
            value: *val as i128,
            suffix: IntLiteralSuffix::L,
            base: IntLiteralBase::Decimal,
        }),
        clight::ClightExpr::EconstFloat(val, _ty) => CExpr::FloatLit(FloatLiteral {
            value: val.0,
            suffix: FloatLiteralSuffix::None,
        }),
        clight::ClightExpr::EconstSingle(val, _ty) => CExpr::FloatLit(FloatLiteral {
            value: val.0 as f64,
            suffix: FloatLiteralSuffix::F,
        }),
        clight::ClightExpr::Evar(id, _ty) => {
            let is_local = ctx.is_current_local_evar(*id);
            let raw_name = if is_local {
                ctx.local_evar_name(*id)
            } else {
                ctx.id_to_name
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("var_{}", id))
            };
            if !is_local && !ctx.suppress_string_literals {
                if let Some(content) = ctx.string_label_to_content.get(&raw_name) {
                    return CExpr::StringLit(StringLiteral {
                        value: content.trim_end_matches('\0').to_string(),
                        is_wide: false,
                    });
                }
            }
            let name = if is_local {
                ctx.local_evar_name(*id)
            } else {
                ctx.var_name(*id)
            };
            CExpr::Var(name)
        }
        clight::ClightExpr::EvarSymbol(name, _ty) => {
            if !ctx.suppress_string_literals {
                if let Some(content) = ctx.string_label_to_content.get(name) {
                    return CExpr::StringLit(StringLiteral {
                        value: content.trim_end_matches('\0').to_string(),
                        is_wide: false,
                    });
                }
            }
            let sanitized = sanitize_c_symbol_name(name);
            // Track header-collision renames (local `atexit` definition -> `atexit_local`) so calls and address-of references target the renamed local definition.
            match ctx.func_renames.get(&sanitized) {
                Some(renamed) => CExpr::Var(renamed.clone()),
                None => CExpr::Var(sanitized),
            }
        }
        clight::ClightExpr::Etempvar(id, _ty) => CExpr::Var(ctx.temp_name(*id)),
        clight::ClightExpr::Ederef(inner, ty) => {
            // Materialize the node's ACCESS type as *(T *)base: always for a bare variable, whose decl may lawfully diverge, and for a composite base unless its annotation already points to the access type.
            let access_castable = matches!(
                ty,
                clight::ClightType::Tint(..)
                    | clight::ClightType::Tlong(..)
                    | clight::ClightType::Tfloat(..)
                    | clight::ClightType::Tpointer(..)
            );
            if access_castable {
                let need_cast = match inner.as_ref() {
                    clight::ClightExpr::Etempvar(..) | clight::ClightExpr::Evar(..) => true,
                    other => !matches!(
                        clight_expr_ctype(other),
                        clight::ClightType::Tpointer(p, _) if p.as_ref() == ty
                    ),
                };
                if need_cast {
                    let cty = CType::ptr(convert_clight_type(ty));
                    return CExpr::Unary(
                        UnaryOp::Deref,
                        Box::new(CExpr::Cast(cty, Box::new(convert_expr(inner, ctx)))),
                    );
                }
            }
            CExpr::Unary(UnaryOp::Deref, Box::new(convert_expr(inner, ctx)))
        }
        clight::ClightExpr::Eaddrof(inner, _ty) => {
            match inner.as_ref() {
                clight::ClightExpr::EvarSymbol(name, _) => {
                    if let Some(content) = ctx.string_label_to_content.get(name) {
                        return CExpr::StringLit(StringLiteral {
                            value: content.trim_end_matches('\0').to_string(),
                            is_wide: false,
                        });
                    }
                }
                clight::ClightExpr::Evar(id, _) => {
                    let raw_name = ctx
                        .id_to_name
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| format!("var_{}", id));
                    if let Some(content) = ctx.string_label_to_content.get(&raw_name) {
                        return CExpr::StringLit(StringLiteral {
                            value: content.trim_end_matches('\0').to_string(),
                            is_wide: false,
                        });
                    }
                }
                _ => {}
            }
            CExpr::Unary(UnaryOp::AddrOf, Box::new(convert_expr(inner, ctx)))
        }
        clight::ClightExpr::Eunop(op, inner, _ty) => {
            if matches!(op, clight::ClightUnaryOp::Oabsfloat) {
                CExpr::Call(
                    Box::new(CExpr::Var("__builtin_fabs".to_string())),
                    vec![convert_expr(inner, ctx)],
                )
            } else {
                CExpr::Unary(convert_unary_op(op), Box::new(convert_expr(inner, ctx)))
            }
        }
        clight::ClightExpr::Ebinop(op, lhs, rhs, _ty) => {
            // B2 mis-scaling fix: byte-addressed candidates annotate their base void * without a cast node, so materialize (RT)((char *)v + k) and the C stays byte-correct under any final declaration.
            if matches!(
                op,
                clight::ClightBinaryOp::Oadd | clight::ClightBinaryOp::Osub
            ) {
                let (bare, vt) = match lhs.as_ref() {
                    clight::ClightExpr::Etempvar(_, vt) | clight::ClightExpr::Evar(_, vt) => {
                        (true, vt)
                    }
                    other => (false, clight_expr_ctype(other)),
                };
                if let clight::ClightType::Tpointer(p, _) = vt {
                    // A void pointee means byte-add, so scale through char *; a composite base realizes its own annotation faithfully except void *, which C cannot scale or deref through.
                    let is_void = matches!(p.as_ref(), clight::ClightType::Tvoid);
                    // The OFFSET operand of pointer arithmetic must print as an integer, since a pointer-annotated rhs is a byte count wearing the wrong type; Osub keeps a pointer rhs as legal pointer difference.
                    let rhs_is_ptr =
                        matches!(clight_expr_ctype(rhs), clight::ClightType::Tpointer(..))
                            && matches!(op, clight::ClightBinaryOp::Oadd);
                    let rhs_field_int = matches!(rhs.as_ref(), clight::ClightExpr::Efield(..))
                        && matches!(
                            clight_expr_ctype(rhs),
                            clight::ClightType::Tint(..) | clight::ClightType::Tlong(..)
                        );
                    let rhs_is_ptr = rhs_is_ptr || rhs_field_int;
                    if bare || is_void || rhs_is_ptr {
                        let base_cty = if is_void {
                            CType::ptr(CType::char_signed())
                        } else {
                            convert_clight_type(vt)
                        };
                        let base = CExpr::Cast(base_cty.clone(), Box::new(convert_expr(lhs, ctx)));
                        let offset = if rhs_is_ptr {
                            CExpr::Cast(CType::long(), Box::new(convert_expr(rhs, ctx)))
                        } else {
                            convert_expr(rhs, ctx)
                        };
                        let sum =
                            CExpr::Binary(convert_binary_op(op), Box::new(base), Box::new(offset));
                        // In C the sum already has the base cast's type; an outer cast only adds information when the annotated result type differs.
                        let result_cty = convert_clight_type(_ty);
                        return if result_cty == base_cty {
                            sum
                        } else {
                            CExpr::Cast(result_cty, Box::new(CExpr::Paren(Box::new(sum))))
                        };
                    }
                }
            }
            let is_comparison = matches!(
                op,
                clight::ClightBinaryOp::Oeq
                    | clight::ClightBinaryOp::One
                    | clight::ClightBinaryOp::Olt
                    | clight::ClightBinaryOp::Ogt
                    | clight::ClightBinaryOp::Ole
                    | clight::ClightBinaryOp::Oge
            );
            let prev_suppress = ctx.suppress_string_literals;
            if is_comparison {
                ctx.suppress_string_literals = true;
            }
            let lhs_expr = convert_expr(lhs, ctx);
            let rhs_expr = convert_expr(rhs, ctx);
            ctx.suppress_string_literals = prev_suppress;
            CExpr::Binary(
                convert_binary_op(op),
                Box::new(lhs_expr),
                Box::new(rhs_expr),
            )
        }
        clight::ClightExpr::Ecast(inner, ty) => {
            CExpr::Cast(convert_clight_type(ty), Box::new(convert_expr(inner, ctx)))
        }
        clight::ClightExpr::Efield(inner, field_id, field_ty) => {
            let inner_expr = convert_expr(inner, ctx);
            let field_name = crate::decompile::passes::csh_pass::field_ident_to_name(*field_id);
            let inner_ct = clight_expr_to_ctype(inner);

            match &inner_expr {
                CExpr::Unary(UnaryOp::Deref, ptr_expr) => {
                    // The field access is well-typed in clight (base derefs to a struct/union), but the emitted C declaration of the base pointer can disagree (type recovery may type it as int*/void*); cast the base to the struct pointer so `->` type-checks.
                    match inner_ct {
                        ct @ (CType::Struct(_) | CType::Union(_)) => {
                            let sptr = CType::Pointer(Box::new(ct), TypeQualifiers::none());
                            CExpr::MemberPtr(
                                Box::new(CExpr::Cast(sptr, ptr_expr.clone())),
                                field_name,
                            )
                        }
                        // The base derefs to a non-aggregate, so lower ->ofs_N to a typed byte-offset deref *(FieldTy *)((char *)base + N); the field ident encodes N and FieldTy is always a concrete scalar.
                        _ => {
                            let offset = *field_id as i64;
                            let field_ctype = convert_clight_type(field_ty);
                            let char_ptr = CType::Pointer(
                                Box::new(CType::char_signed()),
                                TypeQualifiers::none(),
                            );
                            let byte_addr = CExpr::Binary(
                                BinaryOp::Add,
                                Box::new(CExpr::Cast(char_ptr, ptr_expr.clone())),
                                Box::new(CExpr::int(offset)),
                            );
                            let field_ptr =
                                CType::Pointer(Box::new(field_ctype), TypeQualifiers::none());
                            CExpr::Unary(
                                UnaryOp::Deref,
                                Box::new(CExpr::Cast(field_ptr, Box::new(byte_addr))),
                            )
                        }
                    }
                }
                _ => CExpr::Member(Box::new(inner_expr), field_name),
            }
        }
        clight::ClightExpr::Esizeof(ty, _result_ty) => CExpr::SizeofType(convert_clight_type(ty)),
        clight::ClightExpr::Ealignof(ty, _result_ty) => CExpr::AlignofType(convert_clight_type(ty)),
        clight::ClightExpr::Econdition(cond, true_val, false_val, _ty) => CExpr::Ternary(
            Box::new(convert_expr(cond, ctx)),
            Box::new(convert_expr(true_val, ctx)),
            Box::new(convert_expr(false_val, ctx)),
        ),
    }
}

/// Eliminate dead code after unconditional exits, preserving labeled goto targets.
fn eliminate_dead_code(stmt: &CStmt) -> CStmt {
    match stmt {
        CStmt::Block(items) => {
            let pruned = prune_block_items(items);
            match pruned.len() {
                0 => CStmt::Empty,
                _ => CStmt::Block(pruned),
            }
        }
        CStmt::Sequence(stmts) => {
            let items: Vec<CBlockItem> =
                stmts.iter().map(|s| CBlockItem::Stmt(s.clone())).collect();
            let pruned = prune_block_items(&items);
            let stmts: Vec<CStmt> = pruned
                .into_iter()
                .filter_map(|item| match item {
                    CBlockItem::Stmt(s) => Some(s),
                    _ => None,
                })
                .collect();
            match stmts.len() {
                0 => CStmt::Empty,
                1 => stmts.into_iter().next().unwrap(),
                _ => CStmt::Sequence(stmts),
            }
        }
        CStmt::If(cond, then_s, else_s) => CStmt::If(
            cond.clone(),
            Box::new(eliminate_dead_code(then_s)),
            else_s.as_ref().map(|e| Box::new(eliminate_dead_code(e))),
        ),
        CStmt::While(cond, body) => CStmt::While(cond.clone(), Box::new(eliminate_dead_code(body))),
        CStmt::DoWhile(body, cond) => {
            CStmt::DoWhile(Box::new(eliminate_dead_code(body)), cond.clone())
        }
        CStmt::For(init, cond, update, body) => CStmt::For(
            init.clone(),
            cond.clone(),
            update.clone(),
            Box::new(eliminate_dead_code(body)),
        ),
        CStmt::Switch(expr, body) => {
            // Don't prune inside switch bodies (all case/default labels are reachable via dispatch); only recurse into individual case bodies, not the sequential structure.
            CStmt::Switch(expr.clone(), Box::new(eliminate_dead_code_in_switch(body)))
        }
        CStmt::Labeled(label, inner) => {
            CStmt::Labeled(label.clone(), Box::new(eliminate_dead_code(inner)))
        }
        other => other.clone(),
    }
}

/// Recurse into switch body items without pruning between them (all cases are dispatch targets).
fn eliminate_dead_code_in_switch(stmt: &CStmt) -> CStmt {
    match stmt {
        CStmt::Block(items) => {
            let cleaned: Vec<CBlockItem> = items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => CBlockItem::Stmt(eliminate_dead_code(s)),
                    other => other.clone(),
                })
                .collect();
            CStmt::Block(cleaned)
        }
        CStmt::Sequence(stmts) => {
            CStmt::Sequence(stmts.iter().map(|s| eliminate_dead_code(s)).collect())
        }
        other => eliminate_dead_code(other),
    }
}

/// Prune block items after unconditional exits, preserving named labels (goto targets).
fn prune_block_items(items: &[CBlockItem]) -> Vec<CBlockItem> {
    let mut result = Vec::new();
    let mut terminated = false;
    for item in items {
        match item {
            CBlockItem::Stmt(s) => {
                // First, recursively eliminate dead code within the statement
                let cleaned = eliminate_dead_code(s);
                if matches!(cleaned, CStmt::Empty) {
                    continue;
                }
                if terminated {
                    // Preserve labeled stmts and loop/switch constructs after terminators.
                    if contains_named_label(&cleaned) || is_control_flow_construct(&cleaned) {
                        let exits = is_unconditional_exit(&cleaned);
                        result.push(CBlockItem::Stmt(cleaned));
                        terminated = exits;
                    }
                } else {
                    let exits = is_unconditional_exit(&cleaned);
                    result.push(CBlockItem::Stmt(cleaned));
                    if exits {
                        terminated = true;
                    }
                }
            }
            CBlockItem::Decl(decls) => {
                // Always keep declarations (they don't produce dead code)
                result.push(CBlockItem::Decl(decls.clone()));
            }
        }
    }
    result
}

/// True if stmt is a loop/switch that should survive DCE after a terminator.
fn is_control_flow_construct(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::While(_, _)
        | CStmt::DoWhile(_, _)
        | CStmt::For(_, _, _, _)
        | CStmt::Switch(_, _) => true,
        CStmt::Labeled(_, inner) => is_control_flow_construct(inner),
        _ => false,
    }
}

/// Returns true if stmt unconditionally exits (return/goto/break/continue).
fn is_unconditional_exit(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Return(_) | CStmt::Goto(_) | CStmt::Break | CStmt::Continue => true,
        // Labeled stmt exit status depends on inner stmt (subsequent code still dead if inner exits).
        CStmt::Labeled(_, inner) => is_unconditional_exit(inner),
        CStmt::Block(items) => {
            // A block exits if its last statement exits
            items
                .iter()
                .rev()
                .find_map(|item| match item {
                    CBlockItem::Stmt(s) if !matches!(s, CStmt::Empty) => {
                        Some(is_unconditional_exit(s))
                    }
                    _ => None,
                })
                .unwrap_or(false)
        }
        CStmt::Sequence(stmts) => stmts
            .iter()
            .rev()
            .find_map(|s| {
                if !matches!(s, CStmt::Empty) {
                    Some(is_unconditional_exit(s))
                } else {
                    None
                }
            })
            .unwrap_or(false),
        CStmt::If(_, then_s, Some(else_s)) => {
            is_unconditional_exit(then_s) && is_unconditional_exit(else_s)
        }
        _ => false,
    }
}

/// Returns true if a CStmt contains a Named label (goto target), ignoring case/default labels (used to check switch bodies without false positives from case labels).
fn contains_goto_target(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Labeled(Label::Named(_), _) => true,
        CStmt::Labeled(_, inner) => contains_goto_target(inner),
        CStmt::Block(items) => items.iter().any(|item| match item {
            CBlockItem::Stmt(s) => contains_goto_target(s),
            _ => false,
        }),
        CStmt::Sequence(stmts) => stmts.iter().any(contains_goto_target),
        CStmt::If(_, then_s, else_s) => {
            contains_goto_target(then_s)
                || else_s.as_ref().map_or(false, |e| contains_goto_target(e))
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) | CStmt::For(_, _, _, body) => {
            contains_goto_target(body)
        }
        CStmt::Switch(_, body) => contains_goto_target(body),
        _ => false,
    }
}

/// Returns true if a CStmt contains any reachable label (named goto targets, case/default).
fn contains_named_label(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Labeled(Label::Named(_), _) => true,
        CStmt::Labeled(Label::Case(_), _) | CStmt::Labeled(Label::Default, _) => true,
        CStmt::Labeled(_, inner) => contains_named_label(inner),
        CStmt::Block(items) => items.iter().any(|item| match item {
            CBlockItem::Stmt(s) => contains_named_label(s),
            _ => false,
        }),
        CStmt::Sequence(stmts) => stmts.iter().any(contains_named_label),
        CStmt::If(_, then_s, else_s) => {
            contains_named_label(then_s)
                || else_s.as_ref().map_or(false, |e| contains_named_label(e))
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) | CStmt::For(_, _, _, body) => {
            contains_named_label(body)
        }
        // Dead switch only preserved if it contains a Named goto target (case labels aren't external).
        CStmt::Switch(_, body) => contains_goto_target(body),
        _ => false,
    }
}

fn clight_callee_leaf(expr: &clight::ClightExpr) -> &clight::ClightExpr {
    match expr {
        clight::ClightExpr::Ecast(inner, _) => clight_callee_leaf(inner),
        _ => expr,
    }
}

fn record_local_callee_function_type(callee: &clight::ClightExpr, ctx: &mut ConversionContext) {
    let name = match clight_callee_leaf(callee) {
        // Register-valued Clight expressions are locals by construction.
        clight::ClightExpr::Etempvar(id, _) => Some(ctx.temp_name(*id)),
        // Addressable Evar locals require explicit per-function stack provenance.
        clight::ClightExpr::Evar(id, _) if ctx.is_current_local_evar(*id) => {
            Some(ctx.local_evar_name(*id))
        }
        _ => None,
    };
    if let Some(name) = name {
        ctx.record_current_function_object_type(name, clight_expr_to_ctype(callee));
    }
}

fn convert_callee_expr(expr: &clight::ClightExpr, ctx: &mut ConversionContext) -> CExpr {
    // A provider cast does not turn a direct address callee into a local
    // function-pointer object.  Emit the designator bare so direct-call
    // evidence and forward-declaration recovery continue to recognize it.
    if let clight::ClightExpr::Evar(id, _) = clight_callee_leaf(expr) {
        if !ctx.is_current_local_evar(*id) && !ctx.id_to_name.contains_key(id) {
            return CExpr::Var(format!("FUN_{:x}", id));
        }
    }
    match expr {
        // Preserve casts around genuine indirect callees.
        clight::ClightExpr::Ecast(inner, ty) => CExpr::Cast(
            convert_clight_type(ty),
            Box::new(convert_callee_expr(inner, ctx)),
        ),
        _ => convert_expr(expr, ctx),
    }
}

fn direct_unprototyped_callee_type(expr: &clight::ClightExpr) -> Option<CType> {
    let ty = match expr {
        clight::ClightExpr::Evar(_, ty) | clight::ClightExpr::EvarSymbol(_, ty) => ty,
        _ => return None,
    };
    let cty = convert_clight_type(ty);
    matches!(
        &cty,
        CType::Pointer(inner, _) if matches!(inner.as_ref(), CType::Function(_, _, _, true))
    )
    .then_some(cty)
}

pub fn convert_stmt(stmt: &clight::ClightStmt, ctx: &mut ConversionContext) -> CStmt {
    match stmt {
        clight::ClightStmt::Sskip => CStmt::Empty,

        clight::ClightStmt::Sassign(lhs, rhs) => {
            let lhs_expr = convert_expr(lhs, ctx);
            let rhs_expr = convert_expr(rhs, ctx);
            CStmt::Expr(CExpr::Assign(
                AssignOp::Assign,
                Box::new(lhs_expr),
                Box::new(rhs_expr),
            ))
        }

        clight::ClightStmt::Sset(id, expr) => {
            let var_name = ctx.temp_name(*id);
            let var_type = clight_expr_to_ctype(expr);
            ctx.record_var_type(&var_name, var_type);
            let rhs_expr = convert_expr(expr, ctx);
            CStmt::Expr(CExpr::Assign(
                AssignOp::Assign,
                Box::new(CExpr::Var(var_name.clone())),
                Box::new(rhs_expr),
            ))
        }

        clight::ClightStmt::Scall(dst, func, args) => {
            record_local_callee_function_type(func, ctx);
            let direct_unprototyped_type = direct_unprototyped_callee_type(func);
            let func_expr = convert_callee_expr(func, ctx);
            let func_expr = if let CExpr::Var(ref name) = func_expr {
                let resolved = ctx.resolve_l_label(name);
                if resolved != *name {
                    CExpr::Var(resolved)
                } else {
                    func_expr
                }
            } else {
                func_expr
            };
            // A short or otherwise incoherent exact call carries an
            // unprototyped annotation whose parameter slots describe this
            // site's real arguments. Materialize it as a cast so a fixed
            // declaration (including one supplied by a system header) cannot
            // make the recovered call ill-formed or tempt a later pass to
            // synthesize missing arguments.
            let func_expr = match direct_unprototyped_type {
                Some(ty) => CExpr::Cast(ty, Box::new(func_expr)),
                None => func_expr,
            };
            let arg_exprs: Vec<CExpr> = args.iter().map(|a| convert_expr(a, ctx)).collect();
            let call_expr = CExpr::Call(Box::new(func_expr), arg_exprs);

            if let Some(id) = dst {
                let var_name = ctx.temp_name(*id);
                CStmt::Expr(CExpr::Assign(
                    AssignOp::Assign,
                    Box::new(CExpr::Var(var_name)),
                    Box::new(call_expr),
                ))
            } else {
                CStmt::Expr(call_expr)
            }
        }

        clight::ClightStmt::Sbuiltin(dst, ef, _tys, args) => {
            let func_name = external_func_name(ef);
            let arg_exprs: Vec<CExpr> = args.iter().map(|a| convert_expr(a, ctx)).collect();
            let call_expr = CExpr::Call(Box::new(CExpr::Var(func_name)), arg_exprs);

            if let Some(id) = dst {
                let var_name = ctx.temp_name(*id);
                CStmt::Expr(CExpr::Assign(
                    AssignOp::Assign,
                    Box::new(CExpr::Var(var_name)),
                    Box::new(call_expr),
                ))
            } else {
                CStmt::Expr(call_expr)
            }
        }

        clight::ClightStmt::Ssequence(stmts) => {
            let mut converted: Vec<CBlockItem> = Vec::new();
            let mut terminated = false;
            for s in stmts {
                let c = convert_stmt(s, ctx);
                if terminated {
                    // After unconditional exit, keep named-label stmts (goto targets) and loop/switch constructs (bodies may host code reachable via gotos/cases).
                    if contains_named_label(&c) || is_control_flow_construct(&c) {
                        let exits = is_unconditional_exit(&c);
                        converted.push(CBlockItem::Stmt(c));
                        // A preserved label can restart reachability for following fallthrough code.
                        terminated = exits;
                    }
                } else {
                    let exits = is_unconditional_exit(&c);
                    converted.push(CBlockItem::Stmt(c));
                    if exits {
                        terminated = true;
                    }
                }
            }
            if converted.len() == 1 {
                // RB-1: len()==1 guard makes next() infallible; both arms push CBlockItem::Stmt so the non-Stmt arm is unreachable.
                match converted.into_iter().next().unwrap() {
                    CBlockItem::Stmt(s) => s,
                    _ => unreachable!("Ssequence conversion pushes only CBlockItem::Stmt"),
                }
            } else {
                CStmt::Block(converted)
            }
        }

        clight::ClightStmt::Sifthenelse(cond, then_stmt, else_stmt) => {
            let cond_expr = convert_expr(cond, ctx);
            let then_s = convert_stmt(then_stmt, ctx);
            let else_s = convert_stmt(else_stmt, ctx);

            let else_opt = if matches!(else_s, CStmt::Empty) {
                None
            } else {
                Some(Box::new(else_s))
            };

            CStmt::If(cond_expr, Box::new(then_s), else_opt)
        }

        clight::ClightStmt::Sloop(body, cont) => {
            let body_stmt = convert_stmt(body, ctx);
            let cont_stmt = convert_stmt(cont, ctx);
            let has_update = !matches!(cont_stmt, CStmt::Empty);
            let update = if has_update {
                extract_loop_update(&cont_stmt)
            } else {
                None
            };

            if let Some((cond, rest_body, is_pre)) = extract_loop_condition(&body_stmt) {
                if is_pre {
                    if has_update {
                        CStmt::For(None, Some(cond), update, Box::new(rest_body))
                    } else {
                        CStmt::While(cond, Box::new(rest_body))
                    }
                } else if has_update {
                    // Post-condition with update becomes for(;1;update) loop
                    CStmt::For(None, Some(CExpr::int(1)), update, Box::new(body_stmt))
                } else {
                    CStmt::DoWhile(Box::new(rest_body), cond)
                }
            } else if has_update {
                CStmt::For(None, Some(CExpr::int(1)), update, Box::new(body_stmt))
            } else {
                CStmt::While(CExpr::int(1), Box::new(body_stmt))
            }
        }

        clight::ClightStmt::Sbreak => CStmt::Break,
        clight::ClightStmt::Scontinue => CStmt::Continue,

        clight::ClightStmt::Sreturn(None) => CStmt::Return(None),
        clight::ClightStmt::Sreturn(Some(expr)) => CStmt::Return(Some(convert_expr(expr, ctx))),

        clight::ClightStmt::Sswitch(expr, cases) => {
            if let Some(range_if) = try_switch_to_range_if(expr, cases, ctx) {
                range_if
            } else {
                let switch_expr = convert_expr(expr, ctx);
                let body = convert_switch_cases(cases, ctx);
                CStmt::Switch(switch_expr, Box::new(body))
            }
        }

        clight::ClightStmt::Slabel(label_id, inner) => {
            let label_name = ctx.label_name(*label_id);
            let inner_stmt = convert_stmt(inner, ctx);
            CStmt::Labeled(Label::Named(label_name), Box::new(inner_stmt))
        }

        clight::ClightStmt::Sgoto(label_id) => {
            let label_name = ctx.label_name(*label_id);
            CStmt::Goto(label_name)
        }
    }
}

fn extract_loop_condition(body: &CStmt) -> Option<(CExpr, CStmt, bool)> {
    if let Some(cond) = extract_condition_break(body) {
        return Some((cond, CStmt::Empty, true));
    }

    let stmts: Vec<&CStmt> = match body {
        CStmt::Sequence(stmts) if stmts.len() >= 2 => stmts.iter().collect(),
        CStmt::Block(items) if items.len() >= 1 => items
            .iter()
            .filter_map(|item| match item {
                CBlockItem::Stmt(s) => Some(s),
                _ => None,
            })
            .collect(),
        _ => return None,
    };

    if stmts.is_empty() {
        return None;
    }

    if stmts.len() == 1 {
        if let Some(cond) = extract_condition_break(stmts[0]) {
            return Some((cond, CStmt::Empty, true));
        }
        return None;
    }

    if let Some(cond) = extract_condition_break(stmts[0]) {
        let rest: Vec<CStmt> = stmts[1..].iter().map(|s| (*s).clone()).collect();
        let rest_stmt = if rest.len() == 1 {
            rest.into_iter().next().unwrap()
        } else {
            CStmt::Sequence(rest)
        };
        return Some((cond, rest_stmt, true));
    }

    // RB-1: last() is infallible -- is_empty() and len()==1 returns above guarantee >= 2 elements here.
    if let Some(cond) = extract_condition_break(stmts.last().unwrap()) {
        let rest: Vec<CStmt> = stmts[..stmts.len() - 1]
            .iter()
            .map(|s| (*s).clone())
            .collect();
        let rest_stmt = if rest.len() == 1 {
            rest.into_iter().next().unwrap()
        } else {
            CStmt::Sequence(rest)
        };
        return Some((cond, rest_stmt, false));
    }

    None
}

fn extract_condition_break(stmt: &CStmt) -> Option<CExpr> {
    match stmt {
        CStmt::If(cond, then_br, Some(else_br))
            if matches!(**then_br, CStmt::Empty) && matches!(**else_br, CStmt::Break) =>
        {
            Some(cond.clone())
        }
        CStmt::If(cond, then_br, None) if matches!(**then_br, CStmt::Break) => {
            Some(negate_cond(cond))
        }
        CStmt::If(cond, then_br, Some(else_br))
            if matches!(**then_br, CStmt::Break) && matches!(**else_br, CStmt::Empty) =>
        {
            Some(negate_cond(cond))
        }
        _ => None,
    }
}

fn negate_cond(cond: &CExpr) -> CExpr {
    match cond {
        CExpr::Unary(UnaryOp::Not, inner) => *inner.clone(),
        CExpr::Binary(BinaryOp::Eq, a, b) => CExpr::Binary(BinaryOp::Ne, a.clone(), b.clone()),
        CExpr::Binary(BinaryOp::Ne, a, b) => CExpr::Binary(BinaryOp::Eq, a.clone(), b.clone()),
        CExpr::Binary(BinaryOp::Lt, a, b) => CExpr::Binary(BinaryOp::Ge, a.clone(), b.clone()),
        CExpr::Binary(BinaryOp::Le, a, b) => CExpr::Binary(BinaryOp::Gt, a.clone(), b.clone()),
        CExpr::Binary(BinaryOp::Gt, a, b) => CExpr::Binary(BinaryOp::Le, a.clone(), b.clone()),
        CExpr::Binary(BinaryOp::Ge, a, b) => CExpr::Binary(BinaryOp::Lt, a.clone(), b.clone()),
        _ => CExpr::Unary(UnaryOp::Not, Box::new(cond.clone())),
    }
}

fn extract_loop_update(stmt: &CStmt) -> Option<CExpr> {
    match stmt {
        CStmt::Expr(e) => Some(e.clone()),
        CStmt::Block(items) if items.len() == 1 => match &items[0] {
            CBlockItem::Stmt(s) => extract_loop_update(s),
            _ => None,
        },
        CStmt::Block(items) => {
            let exprs: Vec<CExpr> = items
                .iter()
                .filter_map(|item| match item {
                    CBlockItem::Stmt(s) => extract_loop_update(s),
                    _ => None,
                })
                .collect();
            if exprs.len() == items.len() && !exprs.is_empty() {
                Some(
                    exprs
                        .into_iter()
                        .reduce(|acc, e| CExpr::Binary(BinaryOp::Comma, Box::new(acc), Box::new(e)))
                        // RB-1: reduce(None) only on empty iterator; excluded by the !exprs.is_empty() guard above.
                        .unwrap(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

fn convert_switch_cases(
    cases: &clight::ClightLabeledStatements,
    ctx: &mut ConversionContext,
) -> CStmt {
    // Convert each (label, body) pair first.
    let converted: Vec<(Label, CStmt)> = cases
        .iter()
        .map(|(label, stmt)| {
            let label = match label {
                Some(val) => Label::Case(CExpr::int(*val)),
                None => Label::Default,
            };
            (label, convert_stmt(stmt, ctx))
        })
        .collect();

    // Collapse maximal runs of consecutive cases sharing an identical non-fall-through body into one stacked-label arm; labels are nested so later cleanup treats each arm as a single unit.
    let mut block_items = Vec::new();
    let mut i = 0;
    while i < converted.len() {
        let mut j = i + 1;
        if switch_body_exits(&converted[i].1) {
            while j < converted.len() && converted[j].1 == converted[i].1 {
                j += 1;
            }
        }
        let mut arm = converted[j - 1].1.clone();
        for (label, _) in converted[i..j].iter().rev() {
            arm = CStmt::Labeled(label.clone(), Box::new(arm));
        }
        block_items.push(CBlockItem::Stmt(arm));
        i = j;
    }

    CStmt::Block(block_items)
}

/// True if `stmt` ends in an unconditional control transfer, so control never falls off its end. Stacking consecutive same-bodied switch cases under shared labels is sound only when the shared body cannot fall through into the following case.
fn switch_body_exits(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Goto(_) | CStmt::Break | CStmt::Continue | CStmt::Return(_) => true,
        CStmt::Labeled(_, inner) => switch_body_exits(inner),
        CStmt::Sequence(stmts) => stmts.last().map_or(false, switch_body_exits),
        CStmt::Block(items) => items
            .iter()
            .rev()
            .find_map(|item| match item {
                CBlockItem::Stmt(s) => Some(s),
                _ => None,
            })
            .map_or(false, switch_body_exits),
        _ => false,
    }
}

/// Number of disjoint value ranges above which a single-target switch is left as a switch rather than expanded into an `||` chain of range tests (keeps a genuinely sparse set readable as a switch).
const MAX_RANGE_IF_RUNS: usize = 8;

/// Converts a single-target switch (all cases share one non-fall-through body) into a range `if` test. Requires: duplicable scrutinee (evaluated multiple times), non-negative case values (so `>=`/`<=` matches the switch's signedness semantics), and every emitted range covers only real case values.
fn try_switch_to_range_if(
    expr: &clight::ClightExpr,
    cases: &clight::ClightLabeledStatements,
    ctx: &mut ConversionContext,
) -> Option<CStmt> {
    if cases.is_empty() || !clight_scrutinee_dupable(expr) {
        return None;
    }

    let mut values: Vec<i64> = Vec::new();
    let mut arm_body: Option<&clight::ClightStmt> = None;
    let mut default_body: Option<&clight::ClightStmt> = None;
    for (label, body) in cases {
        match label {
            Some(v) => {
                match arm_body {
                    None => arm_body = Some(body),
                    Some(b) if b == body => {}
                    // A second distinct target means a real multi-way dispatch; keep the switch.
                    Some(_) => return None,
                }
                values.push(*v);
            }
            None => {
                if default_body.is_some() {
                    return None;
                }
                default_body = Some(body);
            }
        }
    }

    let arm_body = arm_body?;
    if !clight_stmt_exits(arm_body) {
        return None;
    }
    if matches!(default_body, Some(d) if !clight_stmt_exits(d)) {
        return None;
    }

    values.sort_unstable();
    values.dedup();
    if values.first().map_or(true, |&v| v < 0) {
        return None;
    }

    // Collapse the sorted values into maximal runs of contiguous integers.
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for &v in &values {
        match runs.last_mut() {
            Some(run) if v == run.1 + 1 => run.1 = v,
            _ => runs.push((v, v)),
        }
    }
    if runs.len() > MAX_RANGE_IF_RUNS {
        return None;
    }

    let scrutinee = convert_expr(expr, ctx);
    let mut condition: Option<CExpr> = None;
    for (lo, hi) in runs {
        let part = if lo == hi {
            CExpr::Binary(
                BinaryOp::Eq,
                Box::new(scrutinee.clone()),
                Box::new(CExpr::int(lo)),
            )
        } else {
            // Wrap each `e>=lo && e<=hi` group in parens so a multi-range condition reads as `(e>=lo && e<=hi) || (e>=lo2 && e<=hi2)` rather than relying on C's &&-over-|| precedence.
            CExpr::Paren(Box::new(CExpr::Binary(
                BinaryOp::And,
                Box::new(CExpr::Binary(
                    BinaryOp::Ge,
                    Box::new(scrutinee.clone()),
                    Box::new(CExpr::int(lo)),
                )),
                Box::new(CExpr::Binary(
                    BinaryOp::Le,
                    Box::new(scrutinee.clone()),
                    Box::new(CExpr::int(hi)),
                )),
            )))
        };
        condition = Some(match condition {
            None => part,
            Some(c) => CExpr::Binary(BinaryOp::Or, Box::new(c), Box::new(part)),
        });
    }

    let then_branch = convert_stmt(arm_body, ctx);
    let else_branch = default_body.map(|d| Box::new(convert_stmt(d, ctx)));
    Some(CStmt::If(condition?, Box::new(then_branch), else_branch))
}

/// True if expr can be evaluated more than once without changing behavior; excludes memory reads so a range if never duplicates a load that could alias or be volatile.
fn clight_scrutinee_dupable(expr: &clight::ClightExpr) -> bool {
    match expr {
        clight::ClightExpr::EconstInt(_, _)
        | clight::ClightExpr::EconstLong(_, _)
        | clight::ClightExpr::EconstFloat(_, _)
        | clight::ClightExpr::EconstSingle(_, _)
        | clight::ClightExpr::Evar(_, _)
        | clight::ClightExpr::EvarSymbol(_, _)
        | clight::ClightExpr::Etempvar(_, _) => true,
        clight::ClightExpr::Ecast(inner, _) | clight::ClightExpr::Eunop(_, inner, _) => {
            clight_scrutinee_dupable(inner)
        }
        clight::ClightExpr::Ebinop(_, lhs, rhs, _) => {
            clight_scrutinee_dupable(lhs) && clight_scrutinee_dupable(rhs)
        }
        _ => false,
    }
}

/// True if a Clight statement ends in an unconditional control transfer (mirrors `switch_body_exits` at the Clight level, before conversion to the C AST).
fn clight_stmt_exits(stmt: &clight::ClightStmt) -> bool {
    match stmt {
        clight::ClightStmt::Sgoto(_)
        | clight::ClightStmt::Sbreak
        | clight::ClightStmt::Scontinue
        | clight::ClightStmt::Sreturn(_) => true,
        clight::ClightStmt::Slabel(_, inner) => clight_stmt_exits(inner),
        clight::ClightStmt::Ssequence(stmts) => stmts.last().map_or(false, clight_stmt_exits),
        _ => false,
    }
}

fn external_func_name(ef: &ExternalFunction) -> String {
    match ef {
        ExternalFunction::EFExternal(n, _)
        | ExternalFunction::EFBuiltin(n, _)
        | ExternalFunction::EFRuntime(n, _)
        | ExternalFunction::EFInlineAsm(n, _, _) => sanitize_c_symbol_name(n),
        ExternalFunction::EFVLoad(_) => "__builtin_vload".into(),
        ExternalFunction::EFVStore(_) => "__builtin_vstore".into(),
        ExternalFunction::EFMalloc => "malloc".into(),
        ExternalFunction::EFFree => "free".into(),
        ExternalFunction::EFMemcpy(_, _) => "memcpy".into(),
        ExternalFunction::EFAnnot(_, n, _) | ExternalFunction::EFAnnotVal(_, n, _) => {
            sanitize_c_symbol_name(n)
        }
        ExternalFunction::EFDebug(_, _, _) => "__builtin_debug".into(),
    }
}

/// RPO over statement-bearing nodes walking the FULL clight_succ edge set, with loop-structure-aware successor ordering that places a loop body contiguously ahead of its exit.
fn order_nodes_dfs(
    entry: crate::x86::types::Node,
    nodes: &HashSet<crate::x86::types::Node>,
    edges: &[(crate::x86::types::Node, crate::x86::types::Node)],
    is_exit: impl Fn(crate::x86::types::Node) -> bool,
    loop_members: &HashMap<crate::x86::types::Node, HashSet<crate::x86::types::Node>>,
    node_innermost_loop: &HashMap<crate::x86::types::Node, crate::x86::types::Node>,
    normalize: impl Fn(crate::x86::types::Node) -> crate::x86::types::Node,
) -> Vec<crate::x86::types::Node> {
    let mut adjacency: HashMap<crate::x86::types::Node, Vec<crate::x86::types::Node>> =
        HashMap::new();
    for (src, dst) in edges {
        if src != dst {
            adjacency.entry(*src).or_default().push(*dst);
        }
    }

    // Loop-structure-aware successor ordering, inverted because the DFS reverses twice: the body successor must be ordered LAST to emit first. Address ascending remains the tiebreak.
    for (src, succs) in adjacency.iter_mut() {
        let src_norm = normalize(*src);
        // Innermost natural loop containing src (smallest body, ties on header address); None degrades the comparator to pure address order, and it is precomputed for O(1) lookup.
        let inner_body: Option<&HashSet<crate::x86::types::Node>> = node_innermost_loop
            .get(&src_norm)
            .and_then(|header| loop_members.get(header));
        succs.sort_by(|a, b| {
            // 0 = leaves the innermost loop (exit, ordered first); 1 = stays inside it (body, ordered last). Both 0 (loop-free node) preserves the prior ascending-address order.
            let key = |n: &crate::x86::types::Node| -> u8 {
                match inner_body {
                    Some(body) if body.contains(&normalize(*n)) => 1,
                    _ => 0,
                }
            };
            key(a).cmp(&key(b)).then_with(|| a.cmp(b))
        });
        succs.dedup();
    }

    let mut visited = HashSet::new();
    let mut post_order = Vec::new();
    let mut stack = Vec::new();

    enum Action {
        Visit(crate::x86::types::Node),
        PostVisit(crate::x86::types::Node),
    }

    // Always start at the true entry so its successor chain orders every reachable member, even when the entry itself carries no statement.
    stack.push(Action::Visit(entry));

    while let Some(action) = stack.pop() {
        match action {
            Action::Visit(u) => {
                if !visited.contains(&u) {
                    visited.insert(u);
                    stack.push(Action::PostVisit(u));
                    if let Some(succs) = adjacency.get(&u) {
                        for &v in succs.iter().rev() {
                            if !visited.contains(&v) {
                                stack.push(Action::Visit(v));
                            }
                        }
                    }
                }
            }
            Action::PostVisit(u) => {
                if nodes.contains(&u) {
                    post_order.push(u);
                }
            }
        }
    }

    let mut result: Vec<_> = post_order.into_iter().rev().collect();

    // Disconnected members (edges lost to bundling or genuinely dead) are inserted BEFORE the trailing exit run, not after it -- placing them after would let the consumer's dead-statement trim silently drop them. exec_order_key is a last-resort presentation tiebreak, not a reachability decision.
    let result_set: HashSet<_> = result.iter().copied().collect();
    let all_remaining: Vec<_> = nodes.difference(&result_set).copied().collect();
    // A disconnected node that is itself a LOOP HEADER is placed right after the reachable node edging into its body, or its Sloop floats to the wrong place and becomes unreachable.
    let mut targeted: Vec<(usize, crate::x86::types::Node)> = Vec::new();
    let mut general_remaining: Vec<crate::x86::types::Node> = Vec::new();
    for &r in &all_remaining {
        let rk = normalize(r);
        let anchor = loop_members
            .get(&rk)
            .and_then(|body| {
                result.iter().position(|&e| {
                    adjacency.get(&e).map_or(false, |succs| {
                        succs.iter().any(|s| body.contains(&normalize(*s)))
                    })
                })
            })
            .or_else(|| {
                // Threaded-entry fallback: anchor to the LAST result node whose forward edge-walk reaches a body member first, the loop's immediate preheader, so preheader init is not emitted after the loop.
                loop_members.get(&rk).and_then(|body| {
                    let result_set: HashSet<crate::x86::types::Node> =
                        result.iter().copied().collect();
                    result.iter().rposition(|&e| {
                        let mut stack: Vec<crate::x86::types::Node> =
                            adjacency.get(&e).map(|s| s.clone()).unwrap_or_default();
                        let mut seen: HashSet<crate::x86::types::Node> = HashSet::new();
                        while let Some(n) = stack.pop() {
                            if !seen.insert(n) {
                                continue;
                            }
                            if body.contains(&normalize(n)) {
                                return true;
                            }
                            // Stop at any other emitted node: control reached a different top-level statement first, so e is not the loop's entry edge.
                            if result_set.contains(&n) {
                                continue;
                            }
                            if let Some(succs) = adjacency.get(&n) {
                                stack.extend(succs.iter().copied());
                            }
                        }
                        false
                    })
                })
            });
        match anchor {
            Some(pos) => targeted.push((pos, r)),
            None => general_remaining.push(r),
        }
    }
    // Apply targeted insertions descending so earlier positions stay valid.
    targeted.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    for (pos, r) in targeted {
        result.insert(pos + 1, r);
    }
    if !general_remaining.is_empty() {
        general_remaining.sort_by_key(|&n| crate::util::exec_order_key(n));
        let mut insert_at = result.len();
        while insert_at > 0 && is_exit(result[insert_at - 1]) {
            insert_at -= 1;
        }
        result.splice(insert_at..insert_at, general_remaining);
    }

    result
}

fn simplify_fallthrough_gotos_in_block(items: Vec<CBlockItem>) -> Vec<CBlockItem> {
    let mut count = 0;
    simplify_fallthrough_gotos_in_block_with_next(items, None, &mut count)
}

fn simplify_fallthrough_gotos_in_block_with_next(
    items: Vec<CBlockItem>,
    outer_next_label: Option<&str>,
    count: &mut usize,
) -> Vec<CBlockItem> {
    if items.is_empty() {
        return items;
    }

    let mut result = Vec::with_capacity(items.len());
    let mut i = 0;

    while i < items.len() {
        let current = &items[i];

        let next_label = if i + 1 < items.len() {
            get_first_label_from_item(&items[i + 1])
        } else {
            outer_next_label.map(|s| s.to_string())
        };

        match current {
            CBlockItem::Stmt(stmt) => {
                let simplified = simplify_stmt_fallthrough(stmt, next_label.as_deref(), count);
                result.push(CBlockItem::Stmt(simplified));
            }
            other => result.push(other.clone()),
        }
        i += 1;
    }

    result
}

fn simplify_stmt_fallthrough(stmt: &CStmt, next_label: Option<&str>, count: &mut usize) -> CStmt {
    match stmt {
        CStmt::Goto(target) => {
            if let Some(next_lbl) = next_label {
                if normalize_label(target) == normalize_label(next_lbl) {
                    *count += 1;
                    return CStmt::Empty;
                }
            }
            stmt.clone()
        }

        CStmt::If(cond, then_s, Some(else_s)) => {
            let simplified_then = simplify_stmt_fallthrough(then_s, next_label, count);
            let simplified_else = simplify_stmt_fallthrough(else_s, next_label, count);

            if let Some(next_lbl) = next_label {
                let else_is_fallthrough = is_goto_to_label(&simplified_else, next_lbl);
                let then_is_fallthrough = is_goto_to_label(&simplified_then, next_lbl);

                if else_is_fallthrough && then_is_fallthrough {
                    *count += 1;
                    return CStmt::Empty;
                }

                if else_is_fallthrough {
                    *count += 1;
                    return CStmt::If(cond.clone(), Box::new(simplified_then), None);
                }

                if then_is_fallthrough {
                    let negated_cond = negate_condition(cond);
                    *count += 1;
                    return CStmt::If(negated_cond, Box::new(simplified_else), None);
                }
            }

            CStmt::If(
                cond.clone(),
                Box::new(simplified_then),
                Some(Box::new(simplified_else)),
            )
        }

        CStmt::If(cond, then_s, None) => {
            let simplified_then = simplify_stmt_fallthrough(then_s, next_label, count);
            CStmt::If(cond.clone(), Box::new(simplified_then), None)
        }

        CStmt::Labeled(label, inner) => {
            let simplified_inner = simplify_stmt_fallthrough(inner, next_label, count);
            CStmt::Labeled(label.clone(), Box::new(simplified_inner))
        }

        CStmt::Block(items) => {
            let simplified =
                simplify_fallthrough_gotos_in_block_with_next(items.clone(), next_label, count);
            CStmt::Block(simplified)
        }

        CStmt::Sequence(stmts) => {
            if stmts.is_empty() {
                return CStmt::Empty;
            }
            let mut result = Vec::with_capacity(stmts.len());
            for (j, s) in stmts.iter().enumerate() {
                let next = if j + 1 < stmts.len() {
                    get_first_label_from_stmt(&stmts[j + 1])
                } else {
                    next_label.map(|s| s.to_string())
                };
                result.push(simplify_stmt_fallthrough(s, next.as_deref(), count));
            }
            CStmt::Sequence(result)
        }

        // Recurse into loop/switch bodies with next_label = None, since a goto at a loop body's end falls to the back-edge and a fall-through past a case label is meaningful.
        CStmt::While(cond, body) => CStmt::While(
            cond.clone(),
            Box::new(simplify_stmt_fallthrough(body, None, count)),
        ),

        CStmt::DoWhile(body, cond) => CStmt::DoWhile(
            Box::new(simplify_stmt_fallthrough(body, None, count)),
            cond.clone(),
        ),

        CStmt::For(init, cond, update, body) => CStmt::For(
            init.clone(),
            cond.clone(),
            update.clone(),
            Box::new(simplify_stmt_fallthrough(body, None, count)),
        ),

        CStmt::Switch(disc, body) => CStmt::Switch(
            disc.clone(),
            Box::new(simplify_stmt_fallthrough(body, None, count)),
        ),

        other => other.clone(),
    }
}

fn is_goto_to_label(stmt: &CStmt, label: &str) -> bool {
    match stmt {
        CStmt::Goto(target) => {
            let target_norm = normalize_label(target);
            let label_norm = normalize_label(label);
            target_norm == label_norm
        }
        CStmt::Labeled(_, inner) => is_goto_to_label(inner, label),
        CStmt::Block(items) if items.len() == 1 => {
            if let CBlockItem::Stmt(s) = &items[0] {
                is_goto_to_label(s, label)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn normalize_label(label: &str) -> String {
    let s = label.strip_prefix('.').unwrap_or(label);
    let s = s.strip_prefix('L').unwrap_or(s);
    let s = s.strip_prefix('_').unwrap_or(s);
    if s.chars().all(|c| c.is_ascii_hexdigit()) {
        s.to_lowercase()
    } else {
        label.to_lowercase()
    }
}

fn get_first_label_from_item(item: &CBlockItem) -> Option<String> {
    match item {
        CBlockItem::Stmt(stmt) => get_first_label_from_stmt(stmt),
        _ => None,
    }
}

fn get_first_label_from_stmt(stmt: &CStmt) -> Option<String> {
    match stmt {
        CStmt::Labeled(Label::Named(name), _) => Some(name.clone()),
        CStmt::Block(items) if !items.is_empty() => get_first_label_from_item(&items[0]),
        CStmt::Sequence(stmts) if !stmts.is_empty() => get_first_label_from_stmt(&stmts[0]),
        _ => None,
    }
}

fn negate_condition(cond: &CExpr) -> CExpr {
    match cond {
        CExpr::Unary(UnaryOp::Not, inner) => (**inner).clone(),
        CExpr::Binary(BinaryOp::Eq, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Ne, lhs.clone(), rhs.clone())
        }
        CExpr::Binary(BinaryOp::Ne, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Eq, lhs.clone(), rhs.clone())
        }
        CExpr::Binary(BinaryOp::Lt, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Ge, lhs.clone(), rhs.clone())
        }
        CExpr::Binary(BinaryOp::Ge, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Lt, lhs.clone(), rhs.clone())
        }
        CExpr::Binary(BinaryOp::Gt, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Le, lhs.clone(), rhs.clone())
        }
        CExpr::Binary(BinaryOp::Le, lhs, rhs) => {
            CExpr::Binary(BinaryOp::Gt, lhs.clone(), rhs.clone())
        }
        other => CExpr::Unary(UnaryOp::Not, Box::new(other.clone())),
    }
}

fn simplify_xor_self_in_expr(expr: &mut CExpr) {
    match expr {
        CExpr::Binary(_, lhs, rhs) => {
            simplify_xor_self_in_expr(lhs);
            simplify_xor_self_in_expr(rhs);
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Paren(inner)
        | CExpr::SizeofExpr(inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _) => simplify_xor_self_in_expr(inner),
        CExpr::Assign(_, lhs, rhs) => {
            simplify_xor_self_in_expr(lhs);
            simplify_xor_self_in_expr(rhs);
        }
        CExpr::Index(arr, idx) => {
            simplify_xor_self_in_expr(arr);
            simplify_xor_self_in_expr(idx);
        }
        CExpr::Ternary(c, t, e) => {
            simplify_xor_self_in_expr(c);
            simplify_xor_self_in_expr(t);
            simplify_xor_self_in_expr(e);
        }
        CExpr::Call(func, args) => {
            simplify_xor_self_in_expr(func);
            for arg in args {
                simplify_xor_self_in_expr(arg);
            }
        }
        _ => {}
    }
    if let CExpr::Binary(BinaryOp::BitXor, lhs, rhs) = expr {
        if lhs == rhs {
            *expr = CExpr::IntLit(IntLiteral {
                value: 0,
                suffix: IntLiteralSuffix::None,
                base: IntLiteralBase::Decimal,
            });
        }
    }
}

fn simplify_xor_self_in_stmt(stmt: &mut CStmt) {
    match stmt {
        CStmt::Expr(e) | CStmt::Return(Some(e)) => simplify_xor_self_in_expr(e),
        CStmt::If(cond, then_s, else_s) => {
            simplify_xor_self_in_expr(cond);
            simplify_xor_self_in_stmt(then_s);
            if let Some(e) = else_s {
                simplify_xor_self_in_stmt(e);
            }
        }
        CStmt::While(cond, body) => {
            simplify_xor_self_in_expr(cond);
            simplify_xor_self_in_stmt(body);
        }
        CStmt::DoWhile(body, cond) => {
            simplify_xor_self_in_stmt(body);
            simplify_xor_self_in_expr(cond);
        }
        CStmt::For(init, cond, update, body) => {
            if let Some(crate::decompile::passes::c_pass::types::ForInit::Expr(e)) = init {
                simplify_xor_self_in_expr(e);
            }
            if let Some(c) = cond {
                simplify_xor_self_in_expr(c);
            }
            if let Some(u) = update {
                simplify_xor_self_in_expr(u);
            }
            simplify_xor_self_in_stmt(body);
        }
        CStmt::Switch(e, body) => {
            simplify_xor_self_in_expr(e);
            simplify_xor_self_in_stmt(body);
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    simplify_xor_self_in_stmt(s);
                }
            }
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                simplify_xor_self_in_stmt(s);
            }
        }
        CStmt::Labeled(_, inner) => simplify_xor_self_in_stmt(inner),
        _ => {}
    }
}

fn strip_dead_expr_stmts(stmt: &mut CStmt) {
    if let CStmt::Expr(e) = &*stmt {
        if !e.has_side_effects() {
            *stmt = CStmt::Empty;
            return;
        }
    }

    match stmt {
        CStmt::Block(items) => {
            for item in items.iter_mut() {
                if let CBlockItem::Stmt(s) = item {
                    strip_dead_expr_stmts(s);
                }
            }
            items.retain(|item| match item {
                CBlockItem::Stmt(s) => !matches!(s, CStmt::Empty),
                _ => true,
            });
        }
        CStmt::Sequence(stmts) => {
            for s in stmts.iter_mut() {
                strip_dead_expr_stmts(s);
            }
            stmts.retain(|s| !matches!(s, CStmt::Empty));
        }
        CStmt::If(_, then_s, else_s) => {
            strip_dead_expr_stmts(then_s);
            if let Some(e) = else_s {
                strip_dead_expr_stmts(e);
            }
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) => strip_dead_expr_stmts(body),
        CStmt::For(_, _, _, body) => strip_dead_expr_stmts(body),
        CStmt::Switch(_, body) => strip_dead_expr_stmts(body),
        CStmt::Labeled(_, inner) => strip_dead_expr_stmts(inner),
        _ => {}
    }
}

pub(crate) fn is_dead_expr_stmt(stmt: &CStmt) -> bool {
    match stmt {
        CStmt::Empty => true,
        CStmt::Expr(e) => !e.has_side_effects(),
        CStmt::Block(items) => items.iter().all(|item| match item {
            CBlockItem::Stmt(s) => is_dead_expr_stmt(s),
            CBlockItem::Decl(decls) => decls.is_empty(),
        }),
        CStmt::Sequence(stmts) => stmts.iter().all(is_dead_expr_stmt),
        _ => false,
    }
}

fn recover_func_names_from_assert(db: &DecompileDB) -> HashMap<u64, String> {
    use crate::x86::types::ClightStmt;

    let mut result: HashMap<u64, String> = HashMap::new();

    let mut string_addr_to_content: HashMap<usize, String> = HashMap::new();
    for (label, content, _size) in db.rel_iter::<(String, String, usize)>("string_data") {
        let hex_str = label.trim_start_matches(".L_").trim_start_matches("L_");
        if let Ok(addr) = u64::from_str_radix(hex_str, 16) {
            string_addr_to_content.insert(addr as usize, content.clone());
        }
    }

    let mut assert_fail_addrs: HashSet<u64> = HashSet::new();
    for (addr, name) in db.rel_iter::<(Address, Symbol)>("plt_entry") {
        if *name == "__assert_fail" {
            assert_fail_addrs.insert(*addr);
        }
    }
    let mut assert_fail_idents: HashSet<usize> = HashSet::new();
    for (id, name) in db.rel_iter::<(Ident, Symbol)>("ident_to_symbol") {
        if name.contains("__assert_fail") || name.contains("assert_fail") {
            assert_fail_idents.insert(*id);
            assert_fail_addrs.insert(*id as u64);
        }
    }

    if assert_fail_addrs.is_empty() && assert_fail_idents.is_empty() {
        return result;
    }

    let mut func_stmts: HashMap<u64, Vec<&ClightStmt>> = HashMap::new();
    for (func_addr, _node, stmt) in db.rel_iter::<(Address, Node, ClightStmt)>("emit_clight_stmt") {
        func_stmts.entry(*func_addr).or_default().push(stmt);
    }

    for (func_addr, stmts) in &func_stmts {
        let mut reg_to_addr: HashMap<usize, usize> = HashMap::new();
        for stmt in stmts {
            collect_sset_addr_defs(stmt, &mut reg_to_addr);
        }

        for stmt in stmts {
            if let Some(name) = extract_assert_func_name(
                stmt,
                &assert_fail_addrs,
                &assert_fail_idents,
                &reg_to_addr,
                &string_addr_to_content,
            ) {
                if is_valid_c_identifier(&name) {
                    result.entry(*func_addr).or_insert(name);
                }
            }
        }
    }

    if !result.is_empty() {
        log::info!(
            "Recovered {} function names from __assert_fail calls",
            result.len()
        );
    }
    result
}

fn collect_sset_addr_defs(
    stmt: &crate::x86::types::ClightStmt,
    reg_to_addr: &mut HashMap<usize, usize>,
) {
    use crate::x86::types::ClightStmt;
    match stmt {
        ClightStmt::Sset(reg, expr) => {
            if let Some(addr) = extract_addrof_ident(expr) {
                reg_to_addr.insert(*reg, addr);
            }
        }
        ClightStmt::Ssequence(stmts) => {
            for s in stmts {
                collect_sset_addr_defs(s, reg_to_addr);
            }
        }
        _ => {}
    }
}

fn extract_addrof_ident(expr: &crate::x86::types::ClightExpr) -> Option<usize> {
    use crate::x86::types::ClightExpr;
    match expr {
        ClightExpr::Eaddrof(inner, _) => match inner.as_ref() {
            ClightExpr::Evar(ident, _) => Some(*ident),
            _ => None,
        },
        ClightExpr::Ecast(inner, _) => extract_addrof_ident(inner),
        _ => None,
    }
}

fn extract_assert_func_name(
    stmt: &crate::x86::types::ClightStmt,
    assert_addrs: &HashSet<u64>,
    assert_idents: &HashSet<usize>,
    reg_to_addr: &HashMap<usize, usize>,
    string_map: &HashMap<usize, String>,
) -> Option<String> {
    use crate::x86::types::*;

    match stmt {
        ClightStmt::Scall(_, callee, args) if args.len() >= 4 => {
            let is_assert = match callee {
                ClightExpr::Evar(ident, _) => {
                    assert_addrs.contains(&(*ident as u64)) || assert_idents.contains(ident)
                }
                ClightExpr::EvarSymbol(name, _) => name.contains("assert_fail"),
                _ => false,
            };
            if !is_assert {
                return None;
            }

            let fourth_arg = &args[3];
            resolve_string_from_expr(fourth_arg, reg_to_addr, string_map)
        }
        ClightStmt::Ssequence(stmts) => {
            for s in stmts {
                if let Some(name) = extract_assert_func_name(
                    s,
                    assert_addrs,
                    assert_idents,
                    reg_to_addr,
                    string_map,
                ) {
                    return Some(name);
                }
            }
            None
        }
        _ => None,
    }
}

fn resolve_string_from_expr(
    expr: &crate::x86::types::ClightExpr,
    reg_to_addr: &HashMap<usize, usize>,
    string_map: &HashMap<usize, String>,
) -> Option<String> {
    use crate::x86::types::ClightExpr;
    match expr {
        ClightExpr::Eaddrof(inner, _) => {
            if let ClightExpr::Evar(ident, _) = inner.as_ref() {
                return string_map.get(ident).cloned();
            }
            None
        }
        ClightExpr::Etempvar(reg, _) => {
            if let Some(addr) = reg_to_addr.get(reg) {
                return string_map.get(addr).cloned();
            }
            None
        }
        ClightExpr::Ecast(inner, _) => resolve_string_from_expr(inner, reg_to_addr, string_map),
        _ => None,
    }
}

pub(crate) fn is_valid_c_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    // RB-1: infallible -- the is_empty() return above guarantees a first char.
    let first = s.chars().next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn strip_dead_labels_in_block(items: &mut Vec<CBlockItem>) {
    use crate::decompile::passes::c_pass::helpers::{
        collect_all_labels, collect_goto_targets, strip_dead_labels,
    };

    let mut goto_targets: HashSet<String> = HashSet::new();
    let mut all_labels: HashSet<String> = HashSet::new();
    for item in items.iter() {
        if let CBlockItem::Stmt(s) = item {
            for target in collect_goto_targets(s) {
                goto_targets.insert(target);
            }
            for label in collect_all_labels(s) {
                all_labels.insert(label);
            }
        }
    }

    let dead_labels: HashSet<String> = all_labels.difference(&goto_targets).cloned().collect();
    if dead_labels.is_empty() {
        return;
    }

    for item in items.iter_mut() {
        if let CBlockItem::Stmt(s) = item {
            *s = strip_dead_labels(s, &dead_labels);
        }
    }
}

fn count_leaf_stmts_in_block(items: &[CBlockItem]) -> usize {
    items
        .iter()
        .map(|item| match item {
            CBlockItem::Stmt(s) => count_leaf_stmts(s),
            CBlockItem::Decl(_) => 1,
        })
        .sum()
}

fn count_leaf_stmts(stmt: &CStmt) -> usize {
    match stmt {
        CStmt::Sequence(stmts) => stmts.iter().map(count_leaf_stmts).sum(),
        CStmt::Block(items) => count_leaf_stmts_in_block(items),
        CStmt::If(_, then_s, else_s) => {
            count_leaf_stmts(then_s) + else_s.as_ref().map_or(0, |e| count_leaf_stmts(e))
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) | CStmt::For(_, _, _, body) => {
            count_leaf_stmts(body)
        }
        CStmt::Switch(_, body) => count_leaf_stmts(body),
        CStmt::Labeled(_, inner) => count_leaf_stmts(inner),
        CStmt::Empty => 0,
        _ => 1,
    }
}

fn is_generated_label(name: &str) -> bool {
    if name.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if name.starts_with('L') && name[1..].chars().all(|c| c.is_ascii_digit()) {
        return !name[1..].is_empty();
    }
    if name.starts_with("L_") && name[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        return !name[2..].is_empty();
    }
    if name.starts_with("FUN_") && name[4..].chars().all(|c| c.is_ascii_hexdigit()) {
        return !name[4..].is_empty();
    }
    false
}

fn rewrite_tailcall_gotos(
    body_items: &mut Vec<CBlockItem>,
    func_names: &HashMap<String, usize>,
    current_func: &str,
) {
    for item in body_items.iter_mut() {
        if let CBlockItem::Stmt(stmt) = item {
            rewrite_tailcall_gotos_in_stmt(stmt, func_names, current_func);
        }
    }
}

fn rewrite_tailcall_gotos_in_stmt(
    stmt: &mut CStmt,
    func_names: &HashMap<String, usize>,
    current_func: &str,
) {
    match stmt {
        CStmt::Goto(target)
            if func_names.contains_key(target.as_str())
                && target != current_func
                && func_names.get(target.as_str()) == Some(&0) =>
        {
            *stmt = CStmt::Sequence(vec![
                CStmt::Expr(CExpr::call(CExpr::var(target.clone()), vec![])),
                CStmt::Return(None),
            ]);
        }
        CStmt::If(_, then_s, else_s) => {
            rewrite_tailcall_gotos_in_stmt(then_s, func_names, current_func);
            if let Some(e) = else_s {
                rewrite_tailcall_gotos_in_stmt(e, func_names, current_func);
            }
        }
        CStmt::While(_, body) | CStmt::DoWhile(body, _) | CStmt::For(_, _, _, body) => {
            rewrite_tailcall_gotos_in_stmt(body, func_names, current_func);
        }
        CStmt::Switch(_, body) => rewrite_tailcall_gotos_in_stmt(body, func_names, current_func),
        CStmt::Sequence(stmts) => {
            for s in stmts.iter_mut() {
                rewrite_tailcall_gotos_in_stmt(s, func_names, current_func);
            }
        }
        CStmt::Block(items) => {
            for item in items.iter_mut() {
                if let CBlockItem::Stmt(s) = item {
                    rewrite_tailcall_gotos_in_stmt(s, func_names, current_func);
                }
            }
        }
        CStmt::Labeled(_, inner) => rewrite_tailcall_gotos_in_stmt(inner, func_names, current_func),
        _ => {}
    }
}

/// Collect all function names called in a statement tree.
fn collect_called_names_in_stmt(stmt: &CStmt, names: &mut HashSet<String>) {
    match stmt {
        CStmt::Expr(expr) => collect_called_names_in_expr(expr, names),
        CStmt::Return(Some(expr)) => collect_called_names_in_expr(expr, names),
        CStmt::If(cond, then_s, else_s) => {
            collect_called_names_in_expr(cond, names);
            collect_called_names_in_stmt(then_s, names);
            if let Some(e) = else_s {
                collect_called_names_in_stmt(e, names);
            }
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            collect_called_names_in_expr(cond, names);
            collect_called_names_in_stmt(body, names);
        }
        CStmt::For(_, cond, update, body) => {
            if let Some(c) = cond {
                collect_called_names_in_expr(c, names);
            }
            if let Some(u) = update {
                collect_called_names_in_expr(u, names);
            }
            collect_called_names_in_stmt(body, names);
        }
        CStmt::Switch(expr, body) => {
            collect_called_names_in_expr(expr, names);
            collect_called_names_in_stmt(body, names);
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_called_names_in_stmt(s, names);
            }
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    collect_called_names_in_stmt(s, names);
                }
            }
        }
        CStmt::Labeled(_, inner) => collect_called_names_in_stmt(inner, names),
        _ => {}
    }
}

fn collect_nonlocal_called_names_in_stmt(
    stmt: &CStmt,
    local_names: &HashSet<String>,
    names: &mut HashSet<String>,
) {
    let mut candidates = HashSet::new();
    collect_called_names_in_stmt(stmt, &mut candidates);
    names.extend(
        candidates
            .into_iter()
            .filter(|name| !local_names.contains(name)),
    );
}

fn collect_called_names_in_expr(expr: &CExpr, names: &mut HashSet<String>) {
    match expr {
        CExpr::Call(callee, args) => {
            if let Some(name) = callee_name_through_casts(callee) {
                names.insert(name.to_string());
            }
            collect_called_names_in_expr(callee, names);
            for arg in args {
                collect_called_names_in_expr(arg, names);
            }
        }
        CExpr::Assign(_, lhs, rhs) => {
            collect_called_names_in_expr(lhs, names);
            collect_called_names_in_expr(rhs, names);
        }
        CExpr::Binary(_, lhs, rhs) => {
            collect_called_names_in_expr(lhs, names);
            collect_called_names_in_expr(rhs, names);
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _) => {
            collect_called_names_in_expr(inner, names);
        }
        CExpr::Ternary(a, b, c) => {
            collect_called_names_in_expr(a, names);
            collect_called_names_in_expr(b, names);
            collect_called_names_in_expr(c, names);
        }
        _ => {}
    }
}

// IL-1: forward-declaration parameter types from a JOIN over call-site evidence; every site must agree on arity and yield classifiable evidence, else the K&R int f() fallback stands.

/// Per-position evidence as a fail-closed join lattice.  Register classes and
/// pointer kinds never convert one another: contradictory observations are
/// `Poison`, which forces an unprototyped declaration.
#[derive(Debug, Clone, PartialEq)]
enum ArgEvidence {
    Bottom,
    /// Integer constant zero.  It may become an integer or a null object/code
    /// pointer without changing the GP argument class, but it must never join
    /// floating evidence and move that call position into XMM registers.
    GpZero,
    Int(u8, Signedness),
    Float(u8),
    /// Object-pointer class; differing concrete pointees conservatively join
    /// to `None` (emitted `void *`) without crossing into function pointers.
    ObjectPtr(Option<CType>),
    FunctionPtr(Option<CType>),
    Poison,
}

impl ArgEvidence {
    /// Lattice join: commutative, associative, idempotent -- result is independent of call-site arrival order.
    fn join(self, other: ArgEvidence) -> ArgEvidence {
        use ArgEvidence::*;
        match (self, other) {
            (Poison, _) | (_, Poison) => Poison,
            (Bottom, x) | (x, Bottom) => x,
            (GpZero, GpZero) => GpZero,
            (GpZero, value @ Int(_, _)) | (value @ Int(_, _), GpZero) => value,
            (GpZero, value @ ObjectPtr(_)) | (value @ ObjectPtr(_), GpZero) => value,
            (GpZero, value @ FunctionPtr(_)) | (value @ FunctionPtr(_), GpZero) => value,
            (GpZero, Float(_)) | (Float(_), GpZero) => Poison,
            (Int(w1, s1), Int(w2, s2)) => {
                if w1 == w2 {
                    let s = if s1 == s2 { s1 } else { Signedness::Signed };
                    Int(w1, s)
                } else if w1 > w2 {
                    Int(w1, s1)
                } else {
                    Int(w2, s2)
                }
            }
            (Float(w1), Float(w2)) => Float(w1.max(w2)),
            (ObjectPtr(a), ObjectPtr(b)) => {
                if a == b {
                    ObjectPtr(a)
                } else {
                    ObjectPtr(None)
                }
            }
            (FunctionPtr(a), FunctionPtr(b)) if a == b => FunctionPtr(a),
            (FunctionPtr(_), FunctionPtr(_)) => Poison,
            // GP integer, XMM float, object pointer, and function pointer are
            // distinct evidence classes.  No join may silently reclassify a
            // call position or launder an invalid implicit conversion.
            (Int(_, _), Float(_)) | (Float(_), Int(_, _))
            | (Int(_, _), ObjectPtr(_)) | (ObjectPtr(_), Int(_, _))
            | (Int(_, _), FunctionPtr(_)) | (FunctionPtr(_), Int(_, _))
            | (Float(_), ObjectPtr(_)) | (ObjectPtr(_), Float(_))
            | (Float(_), FunctionPtr(_)) | (FunctionPtr(_), Float(_))
            | (ObjectPtr(_), FunctionPtr(_)) | (FunctionPtr(_), ObjectPtr(_)) => Poison,
        }
    }

    /// The C type to emit for a fully-joined position; poisoned or
    /// signature-less function-pointer evidence forces the whole declaration
    /// to the K&R fallback.
    fn to_ctype(&self) -> Option<CType> {
        use crate::decompile::passes::c_pass::types::{FloatSize, IntSize};
        match self {
            // Bottom = no evidence: conservative default (long accepts both ints and pointers untruncated).
            ArgEvidence::Bottom => Some(CType::long()),
            ArgEvidence::GpZero => Some(CType::int()),
            ArgEvidence::Int(w, s) => {
                let sz = match *w {
                    0..=8 => IntSize::Char,
                    9..=16 => IntSize::Short,
                    17..=32 => IntSize::Int,
                    _ => IntSize::Long,
                };
                Some(CType::Int(sz, *s))
            }
            ArgEvidence::Float(w) => Some(CType::Float(if *w <= 32 {
                FloatSize::Float
            } else {
                FloatSize::Double
            })),
            ArgEvidence::ObjectPtr(Some(ty)) | ArgEvidence::FunctionPtr(Some(ty)) => {
                Some(ty.clone())
            }
            ArgEvidence::ObjectPtr(None) => Some(CType::ptr(CType::Void)),
            ArgEvidence::FunctionPtr(None) => None,
            ArgEvidence::Poison => None,
        }
    }
}

fn ctype_involves_function(ty: &CType) -> bool {
    match ty {
        CType::Function(_, _, _, _) => true,
        CType::Pointer(inner, _) | CType::Array(inner, _) | CType::Qualified(inner, _) => {
            ctype_involves_function(inner)
        }
        _ => false,
    }
}

fn ctype_is_function_type(ty: &CType) -> bool {
    match ty {
        CType::Function(_, _, _, _) => true,
        CType::Qualified(inner, _) => ctype_is_function_type(inner),
        _ => false,
    }
}

fn retain_clight_function_object_types(
    body: &CStmt,
    clight_var_types: &HashMap<String, CType>,
    local_var_types: &mut HashMap<String, CType>,
) {
    let mut body_names = HashSet::new();
    collect_var_names_from_stmt(body, &mut body_names);
    for name in body_names {
        if let Some(ty) = clight_var_types.get(&name) {
            if ctype_involves_function(ty) {
                local_var_types.insert(name, ty.clone());
            }
        }
    }
}

/// Classify a declared C type as argument evidence.
fn arg_evidence_of_ctype(ty: &CType) -> ArgEvidence {
    use crate::decompile::passes::c_pass::types::{FloatSize, IntSize};
    match ty {
        CType::Int(sz, s) => {
            let w = match sz {
                IntSize::Char => 8,
                IntSize::Short => 16,
                IntSize::Int => 32,
                IntSize::Long | IntSize::LongLong => 64,
                IntSize::Int128 => 128,
            };
            ArgEvidence::Int(w, *s)
        }
        CType::Bool => ArgEvidence::Int(8, Signedness::Unsigned),
        CType::Enum(_) => ArgEvidence::Int(32, Signedness::Signed),
        CType::Float(FloatSize::Float) => ArgEvidence::Float(32),
        CType::Float(_) => ArgEvidence::Float(64),
        CType::Pointer(inner, _) if ctype_is_function_type(inner) => {
            ArgEvidence::FunctionPtr(Some(ty.clone()))
        }
        CType::Pointer(_, _) => ArgEvidence::ObjectPtr(Some(ty.clone())),
        // Arrays and function designators decay to pointers in argument position.
        CType::Array(inner, _) => {
            ArgEvidence::ObjectPtr(Some(CType::ptr(inner.as_ref().clone())))
        }
        CType::Function(_, _, _, _) => {
            ArgEvidence::FunctionPtr(Some(CType::ptr(ty.clone())))
        }
        CType::Qualified(inner, _) => arg_evidence_of_ctype(inner),
        // void/struct/union/typedef: not a scalar class we can safely type a parameter from.
        _ => ArgEvidence::Poison,
    }
}

/// Usual arithmetic conversion for one expression.  This is deliberately
/// separate from the cross-call-site evidence join: an `int + float`
/// expression has a real floating result, while observing int at one call and
/// float at another is a prototype conflict.
fn arithmetic_arg_evidence(left: ArgEvidence, right: ArgEvidence) -> ArgEvidence {
    match (left, right) {
        (ArgEvidence::Bottom, value) | (value, ArgEvidence::Bottom) => value,
        (ArgEvidence::GpZero, ArgEvidence::GpZero) => ArgEvidence::GpZero,
        (ArgEvidence::GpZero, value @ ArgEvidence::Int(_, _))
        | (value @ ArgEvidence::Int(_, _), ArgEvidence::GpZero) => value,
        (ArgEvidence::GpZero, ArgEvidence::Float(width))
        | (ArgEvidence::Float(width), ArgEvidence::GpZero) => ArgEvidence::Float(width),
        (left @ ArgEvidence::Int(_, _), right @ ArgEvidence::Int(_, _))
        | (left @ ArgEvidence::Float(_), right @ ArgEvidence::Float(_)) => {
            left.join(right)
        }
        (ArgEvidence::Int(_, _), ArgEvidence::Float(width))
        | (ArgEvidence::Float(width), ArgEvidence::Int(_, _)) => ArgEvidence::Float(width),
        _ => ArgEvidence::Poison,
    }
}

fn integer_binary_arg_evidence(left: ArgEvidence, right: ArgEvidence) -> ArgEvidence {
    let is_integer = |evidence: &ArgEvidence| {
        matches!(
            evidence,
            ArgEvidence::Bottom | ArgEvidence::GpZero | ArgEvidence::Int(_, _)
        )
    };
    if is_integer(&left) && is_integer(&right) {
        arithmetic_arg_evidence(left, right)
    } else {
        ArgEvidence::Poison
    }
}

/// Type environment for classifying call arguments: exactly the declarations the C compiler sees; locals shadow globals.
struct ArgEvidenceEnv<'a> {
    local_types: HashMap<String, CType>,
    global_types: &'a HashMap<String, CType>,
    callee_ret: &'a HashMap<String, CType>,
}

impl ArgEvidenceEnv<'_> {
    fn var_type(&self, name: &str) -> Option<&CType> {
        self.local_types
            .get(name)
            .or_else(|| self.global_types.get(name))
    }
}

fn address_of_arg_evidence(inner: &CExpr, env: &ArgEvidenceEnv) -> ArgEvidence {
    match inner {
        CExpr::Paren(inner) => address_of_arg_evidence(inner, env),
        CExpr::Var(name) => match env.var_type(name) {
            Some(ty @ CType::Function(_, _, _, _)) => {
                ArgEvidence::FunctionPtr(Some(CType::ptr(ty.clone())))
            }
            Some(ty) => ArgEvidence::ObjectPtr(Some(CType::ptr(ty.clone()))),
            None if env.callee_ret.contains_key(name) => ArgEvidence::FunctionPtr(None),
            None => ArgEvidence::Poison,
        },
        // `&*p` retains the pointer value and, critically, its object-vs-code
        // pointer class.  Treating every non-variable lvalue as `void *` would
        // silently turn `&*fp` into an object pointer.
        CExpr::Unary(UnaryOp::Deref, pointer) => match arg_evidence_of_expr(pointer, env) {
            evidence @ (ArgEvidence::ObjectPtr(_) | ArgEvidence::FunctionPtr(_)) => evidence,
            _ => ArgEvidence::Poison,
        },
        // Member/index lvalue types are not carried by this late C AST.  Their
        // address might be an object pointer or a function designator, so the
        // only sound evidence is a veto.
        _ => ArgEvidence::Poison,
    }
}

/// Classify one call-site argument expression; returns `Poison` when the declared type cannot be determined.
fn arg_evidence_of_expr(expr: &CExpr, env: &ArgEvidenceEnv) -> ArgEvidence {
    use crate::decompile::passes::c_pass::types::UnaryOp;
    match expr {
        CExpr::Cast(ty, _) => arg_evidence_of_ctype(ty),
        CExpr::StringLit(_) => {
            ArgEvidence::ObjectPtr(Some(CType::ptr(CType::char_signed())))
        }
        // A literal zero is a GP-class value and a null pointer constant.  It
        // is not lattice bottom: a fixed float prototype would change the
        // original call's register class during recompilation.
        CExpr::IntLit(l) if l.value == 0 => ArgEvidence::GpZero,
        CExpr::IntLit(l) => {
            use crate::decompile::passes::c_pass::types::IntLiteralSuffix as S;
            let (suffix_wide, unsigned) = match l.suffix {
                S::None => (false, false),
                S::U => (false, true),
                S::L | S::LL => (true, false),
                _ => (true, true),
            };
            let fits32 = l.value >= i32::MIN as i128 && l.value <= u32::MAX as i128;
            let w = if suffix_wide || !fits32 { 64 } else { 32 };
            let s = if unsigned {
                Signedness::Unsigned
            } else {
                Signedness::Signed
            };
            ArgEvidence::Int(w, s)
        }
        CExpr::FloatLit(l) => {
            use crate::decompile::passes::c_pass::types::FloatLiteralSuffix as F;
            ArgEvidence::Float(if matches!(l.suffix, F::F) { 32 } else { 64 })
        }
        // A C character literal has type int.
        CExpr::CharLit(_) => ArgEvidence::Int(32, Signedness::Signed),
        CExpr::Var(name) => env
            .var_type(name)
            .map(arg_evidence_of_ctype)
            .unwrap_or(ArgEvidence::Poison),
        CExpr::Paren(inner) => arg_evidence_of_expr(inner, env),
        CExpr::Unary(UnaryOp::AddrOf, inner) => address_of_arg_evidence(inner, env),
        CExpr::Unary(UnaryOp::Deref, inner) => match arg_evidence_of_expr(inner, env) {
            ArgEvidence::ObjectPtr(Some(CType::Pointer(p, _)))
            | ArgEvidence::FunctionPtr(Some(CType::Pointer(p, _))) => {
                arg_evidence_of_ctype(&p)
            }
            _ => ArgEvidence::Poison,
        },
        CExpr::Unary(UnaryOp::Not, _) => ArgEvidence::Int(32, Signedness::Signed),
        CExpr::Unary(UnaryOp::Neg | UnaryOp::Plus, inner) => {
            match arg_evidence_of_expr(inner, env) {
                ev @ (ArgEvidence::Int(_,_)
                | ArgEvidence::Float(_)
                | ArgEvidence::Bottom
                | ArgEvidence::GpZero) => ev,
                _ => ArgEvidence::Poison,
            }
        }
        CExpr::Unary(UnaryOp::BitNot, inner) => match arg_evidence_of_expr(inner, env) {
            ArgEvidence::GpZero => ArgEvidence::Int(32, Signedness::Signed),
            ev @ (ArgEvidence::Int(_, _) | ArgEvidence::Bottom) => ev,
            _ => ArgEvidence::Poison,
        },
        CExpr::Unary(_, _) => ArgEvidence::Poison,
        CExpr::Binary(op, l, r) => {
            use crate::decompile::passes::c_pass::types::BinaryOp as B;
            match op {
                // Comparisons and logical connectives have type int.
                B::Eq | B::Ne | B::Lt | B::Le | B::Gt | B::Ge | B::And | B::Or => {
                    let left = arg_evidence_of_expr(l, env);
                    let right = arg_evidence_of_expr(r, env);
                    if matches!(left, ArgEvidence::Poison)
                        || matches!(right, ArgEvidence::Poison)
                    {
                        ArgEvidence::Poison
                    } else {
                        ArgEvidence::Int(32, Signedness::Signed)
                    }
                }
                B::Shl | B::Shr => {
                    let left = arg_evidence_of_expr(l, env);
                    let right = arg_evidence_of_expr(r, env);
                    if matches!(
                        right,
                        ArgEvidence::Bottom | ArgEvidence::GpZero | ArgEvidence::Int(_, _)
                    ) {
                        match left {
                            ev @ (ArgEvidence::Int(_, _) | ArgEvidence::Bottom) => ev,
                            ArgEvidence::GpZero => {
                                ArgEvidence::Int(32, Signedness::Signed)
                            }
                            _ => ArgEvidence::Poison,
                        }
                    } else {
                        ArgEvidence::Poison
                    }
                }
                B::Add | B::Sub => {
                    let le = arg_evidence_of_expr(l, env);
                    let re = arg_evidence_of_expr(r, env);
                    match (le, re) {
                        // ptr - ptr is ptrdiff_t; ptr +/- int keeps the pointer type.
                        (ArgEvidence::ObjectPtr(_), ArgEvidence::ObjectPtr(_))
                            if matches!(op, B::Sub) => {
                            ArgEvidence::Int(64, Signedness::Signed)
                        }
                        (p @ ArgEvidence::ObjectPtr(_), ArgEvidence::Int(_, _) | ArgEvidence::Bottom | ArgEvidence::GpZero)
                        | (ArgEvidence::Int(_, _) | ArgEvidence::Bottom | ArgEvidence::GpZero, p @ ArgEvidence::ObjectPtr(_)) => {
                            p
                        }
                        (
                            le @ (ArgEvidence::Int(_, _)
                            | ArgEvidence::Float(_)
                            | ArgEvidence::Bottom
                            | ArgEvidence::GpZero),
                            re @ (ArgEvidence::Int(_, _)
                            | ArgEvidence::Float(_)
                            | ArgEvidence::Bottom
                            | ArgEvidence::GpZero),
                        ) => arithmetic_arg_evidence(le, re),
                        _ => ArgEvidence::Poison,
                    }
                }
                B::Mul | B::Div => {
                    let le = arg_evidence_of_expr(l, env);
                    let re = arg_evidence_of_expr(r, env);
                    match (&le, &re) {
                        (
                            ArgEvidence::Int(_, _) | ArgEvidence::Float(_) | ArgEvidence::Bottom | ArgEvidence::GpZero,
                            ArgEvidence::Int(_, _) | ArgEvidence::Float(_) | ArgEvidence::Bottom | ArgEvidence::GpZero,
                        ) => arithmetic_arg_evidence(le, re),
                        _ => ArgEvidence::Poison,
                    }
                }
                B::Mod | B::BitAnd | B::BitOr | B::BitXor => integer_binary_arg_evidence(
                    arg_evidence_of_expr(l, env),
                    arg_evidence_of_expr(r, env),
                ),
                _ => ArgEvidence::Poison,
            }
        }
        CExpr::Ternary(_, t, e) => {
            let te = arg_evidence_of_expr(t, env);
            let ee = arg_evidence_of_expr(e, env);
            te.join(ee)
        }
        CExpr::Call(callee, _) => match callee.as_ref() {
            CExpr::Var(name) => env
                .callee_ret
                .get(name)
                .map(arg_evidence_of_ctype)
                .unwrap_or(ArgEvidence::Poison),
            _ => ArgEvidence::Poison,
        },
        CExpr::SizeofType(_) | CExpr::SizeofExpr(_) | CExpr::AlignofType(_) => {
            // size_t.
            ArgEvidence::Int(64, Signedness::Unsigned)
        }
        _ => ArgEvidence::Poison,
    }
}

/// Collect per-call-site argument evidence vectors keyed by callee name (deterministic TU walk order).
fn collect_call_arg_evidence_in_stmt(
    stmt: &CStmt,
    env: &ArgEvidenceEnv,
    sites: &mut HashMap<String, Vec<Vec<ArgEvidence>>>,
) {
    match stmt {
        CStmt::Expr(expr) => collect_call_arg_evidence_in_expr(expr, env, sites),
        CStmt::Return(Some(expr)) => collect_call_arg_evidence_in_expr(expr, env, sites),
        CStmt::If(cond, then_s, else_s) => {
            collect_call_arg_evidence_in_expr(cond, env, sites);
            collect_call_arg_evidence_in_stmt(then_s, env, sites);
            if let Some(e) = else_s {
                collect_call_arg_evidence_in_stmt(e, env, sites);
            }
        }
        CStmt::While(cond, body) | CStmt::DoWhile(body, cond) => {
            collect_call_arg_evidence_in_expr(cond, env, sites);
            collect_call_arg_evidence_in_stmt(body, env, sites);
        }
        CStmt::For(_, cond, update, body) => {
            if let Some(c) = cond {
                collect_call_arg_evidence_in_expr(c, env, sites);
            }
            if let Some(u) = update {
                collect_call_arg_evidence_in_expr(u, env, sites);
            }
            collect_call_arg_evidence_in_stmt(body, env, sites);
        }
        CStmt::Switch(expr, body) => {
            collect_call_arg_evidence_in_expr(expr, env, sites);
            collect_call_arg_evidence_in_stmt(body, env, sites);
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                collect_call_arg_evidence_in_stmt(s, env, sites);
            }
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    collect_call_arg_evidence_in_stmt(s, env, sites);
                }
            }
        }
        CStmt::Labeled(_, inner) => collect_call_arg_evidence_in_stmt(inner, env, sites),
        _ => {}
    }
}

fn collect_call_arg_evidence_in_expr(
    expr: &CExpr,
    env: &ArgEvidenceEnv,
    sites: &mut HashMap<String, Vec<Vec<ArgEvidence>>>,
) {
    match expr {
        CExpr::Call(callee, args) => {
            if let Some(name) = callee_name_through_casts(callee) {
                if env.local_types.contains_key(name) {
                    for arg in args {
                        collect_call_arg_evidence_in_expr(arg, env, sites);
                    }
                    return;
                }
                let evidence: Vec<ArgEvidence> =
                    args.iter().map(|a| arg_evidence_of_expr(a, env)).collect();
                sites.entry(name.to_string()).or_default().push(evidence);
            }
            collect_call_arg_evidence_in_expr(callee, env, sites);
            for arg in args {
                collect_call_arg_evidence_in_expr(arg, env, sites);
            }
        }
        CExpr::Assign(_, lhs, rhs) | CExpr::Binary(_, lhs, rhs) => {
            collect_call_arg_evidence_in_expr(lhs, env, sites);
            collect_call_arg_evidence_in_expr(rhs, env, sites);
        }
        CExpr::Unary(_, inner)
        | CExpr::Cast(_, inner)
        | CExpr::Member(inner, _)
        | CExpr::MemberPtr(inner, _)
        | CExpr::Paren(inner)
        | CExpr::SizeofExpr(inner) => {
            collect_call_arg_evidence_in_expr(inner, env, sites);
        }
        CExpr::Index(arr, idx) => {
            collect_call_arg_evidence_in_expr(arr, env, sites);
            collect_call_arg_evidence_in_expr(idx, env, sites);
        }
        CExpr::Ternary(a, b, c) => {
            collect_call_arg_evidence_in_expr(a, env, sites);
            collect_call_arg_evidence_in_expr(b, env, sites);
            collect_call_arg_evidence_in_expr(c, env, sites);
        }
        _ => {}
    }
}

/// Infer only the arity cohort that is safe to share across call sites.
///
/// Every site must have one dense, unambiguous vector and every non-empty
/// position must be anchored by an explicit register setup or a uniquely
/// owned entry-SP store somewhere in the cohort.  A forwarded-only thunk may
/// therefore reuse an anchored site's vector, while two forwarded-live-in-only
/// sites cannot establish a prototype.  Two agreeing empty sites are the one
/// anchor-free case: they establish the useful fixed `(void)` prototype.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ExactLoaderIdentity {
    address: Address,
    kind: LoaderSymbolKind,
}

impl ExactLoaderIdentity {
    fn from_row(
        address: Address,
        kind: LoaderSymbolKind,
        _provider: Symbol,
        _original: Symbol,
    ) -> Self {
        Self {
            address,
            kind,
        }
    }
}

fn exact_import_pointer_call_addresses(db: &DecompileDB) -> HashSet<Address> {
    db.rel_iter::<(Node, Address, LoaderSymbolKind, Symbol, Symbol)>("call_loader_identity")
        .filter_map(|(_, address, kind, _, _)| {
            (*kind == LoaderSymbolKind::ImportPointer).then_some(*address)
        })
        .collect()
}

fn exact_import_pointer_call_names(db: &DecompileDB) -> HashSet<String> {
    db.rel_iter::<(Node, Address, LoaderSymbolKind, Symbol, Symbol)>("call_loader_identity")
        .filter_map(|(_, _, kind, provider, original)| {
            (*kind == LoaderSymbolKind::ImportPointer).then_some([
                sanitize_c_symbol_name(provider),
                sanitize_c_symbol_name(original),
            ])
        })
        .flatten()
        .collect()
}

fn is_shadowed_same_address_function(
    identity: &ExactLoaderIdentity,
    exact_import_pointer_addresses: &HashSet<Address>,
) -> bool {
    identity.kind == LoaderSymbolKind::Function
        && exact_import_pointer_addresses.contains(&identity.address)
}

fn project_exact_identity_values<T: Clone>(
    identity_names: &BTreeMap<ExactLoaderIdentity, BTreeSet<String>>,
    values: &BTreeMap<ExactLoaderIdentity, T>,
) -> HashMap<String, T> {
    let mut owners: BTreeMap<String, BTreeSet<ExactLoaderIdentity>> = BTreeMap::new();
    for (identity, names) in identity_names {
        for name in names {
            owners
                .entry(name.clone())
                .or_default()
                .insert(identity.clone());
        }
    }

    let mut projected = HashMap::new();
    for (name, identities) in owners {
        if identities.len() != 1 {
            continue;
        }
        let identity = identities.iter().next().unwrap();
        if let Some(value) = values.get(identity) {
            projected.insert(name, value.clone());
        }
    }
    projected
}

fn insert_identity_names(
    names: &mut BTreeMap<ExactLoaderIdentity, BTreeSet<String>>,
    identity: &ExactLoaderIdentity,
    provider: Symbol,
    original: Symbol,
) {
    let entry = names.entry(identity.clone()).or_default();
    entry.insert(sanitize_c_symbol_name(provider));
    entry.insert(sanitize_c_symbol_name(original));
}

/// Find emitted spellings that are not uniquely owned by their loader
/// identity even though the loader relation itself has only one owner.  The C
/// AST is keyed by sanitized spelling, so a plain symbol at another address
/// makes that spelling ambiguous just as surely as a second loader identity
/// would.  Keep aliases at the same address: they still denote the same exact
/// loader object, and the symbol relations do not carry a finer kind tag.
fn loader_name_collisions_with_unowned_symbols(
    db: &DecompileDB,
    identity_names: &BTreeMap<ExactLoaderIdentity, BTreeSet<String>>,
) -> HashSet<String> {
    let mut loader_addresses: BTreeMap<String, BTreeSet<Address>> = BTreeMap::new();
    for (identity, names) in identity_names {
        for name in names {
            loader_addresses
                .entry(name.clone())
                .or_default()
                .insert(identity.address);
        }
    }

    let mut collisions = HashSet::new();
    let mut observe = |address: Address, symbol: Symbol| {
        let name = sanitize_c_symbol_name(symbol);
        if loader_addresses
            .get(&name)
            .is_some_and(|addresses| !addresses.contains(&address))
        {
            collisions.insert(name);
        }
    };
    for &(ident, symbol) in db.rel_iter::<(Ident, Symbol)>("ident_to_symbol") {
        observe(ident as Address, symbol);
    }
    for &(address, symbol, _) in db.rel_iter::<(Address, Symbol, Symbol)>("symbols") {
        observe(address, symbol);
    }
    collisions
}

fn infer_shared_fixed_arities(
    call_sites: &[(Node, ExactLoaderIdentity)],
    provenance: &[(Node, usize, RTLReg, CallArgProvenance)],
    known_variadic: &HashSet<ExactLoaderIdentity>,
    veto_sites: &HashSet<Node>,
) -> BTreeMap<ExactLoaderIdentity, usize> {
    let mut sites_by_callee: BTreeMap<ExactLoaderIdentity, BTreeSet<Node>> = BTreeMap::new();
    let mut identities_by_site: BTreeMap<Node, BTreeSet<ExactLoaderIdentity>> = BTreeMap::new();
    for (node, identity) in call_sites {
        sites_by_callee
            .entry(identity.clone())
            .or_default()
            .insert(*node);
        identities_by_site
            .entry(*node)
            .or_default()
            .insert(identity.clone());
    }
    let ambiguous_sites: HashSet<Node> = identities_by_site
        .into_iter()
        .filter_map(|(node, identities)| (identities.len() != 1).then_some(node))
        .collect();
    let mut provenance_by_site: HashMap<
        Node,
        Vec<(usize, RTLReg, CallArgProvenance)>,
    > = HashMap::new();
    for &(node, position, reg, source) in provenance {
        provenance_by_site
            .entry(node)
            .or_default()
            .push((position, reg, source));
    }

    let mut inferred = BTreeMap::new();
    for (callee, sites) in sites_by_callee {
        if sites.len() < 2 || known_variadic.contains(&callee) {
            continue;
        }

        let mut common_arity: Option<usize> = None;
        let mut anchored_positions = BTreeSet::new();
        let mut coherent = true;
        for site in sites {
            if veto_sites.contains(&site) || ambiguous_sites.contains(&site) {
                coherent = false;
                break;
            }
            let rows = provenance_by_site
                .get(&site)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let Some(args) =
                crate::decompile::passes::rtl_optimize_pass::coherent_call_vector(
                    rows.iter().copied(),
                )
            else {
                coherent = false;
                break;
            };
            match common_arity {
                Some(arity) if arity != args.len() => {
                    coherent = false;
                    break;
                }
                None => common_arity = Some(args.len()),
                Some(_) => {}
            }
            anchored_positions.extend(
                rows.iter()
                    .filter(|(_, _, source)| source.is_anchor())
                    .map(|(position, _, _)| *position),
            );
        }

        let Some(arity) = common_arity else { continue };
        if coherent
            && (arity == 0 || (0..arity).all(|position| anchored_positions.contains(&position)))
        {
            inferred.insert(callee, arity);
        }
    }
    inferred
}

/// Deterministic authoritative signatures keyed by every exact loader
/// spelling C emission may retain.  A connected identity component is omitted
/// wholesale when its signature or variadic facts conflict.
pub(crate) fn known_loader_signatures_from_db(
    db: &DecompileDB,
) -> HashMap<String, (XType, Arc<Vec<XType>>, bool)> {
    let exact_import_pointer_addresses = exact_import_pointer_call_addresses(db);
    let mut identity_names: BTreeMap<ExactLoaderIdentity, BTreeSet<String>> = BTreeMap::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_symbol_identity") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
    }
    let mut exact_rows: BTreeMap<
        ExactLoaderIdentity,
        BTreeSet<(XType, Arc<Vec<XType>>, bool)>,
    > = BTreeMap::new();
    let mut exact_vetoes = HashSet::new();
    for (address, kind, provider, original, arity, ret, params, variadic) in db
        .rel_iter::<(
            Address,
            LoaderSymbolKind,
            Symbol,
            Symbol,
            usize,
            XType,
            Arc<Vec<XType>>,
            bool,
        )>("known_loader_signature")
    {
        let identity = ExactLoaderIdentity::from_row(*address, *kind, *provider, *original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, *provider, *original);
        if *arity != params.len() {
            exact_vetoes.insert(identity);
            continue;
        }
        exact_rows
            .entry(identity)
            .or_default()
            .insert((*ret, params.clone(), *variadic));
    }
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_signature_conflict") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
        exact_vetoes.insert(identity);
    }
    let mut exact_variadic = HashSet::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("known_loader_variadic") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
        exact_variadic.insert(identity);
    }
    let exact_values: BTreeMap<ExactLoaderIdentity, (XType, Arc<Vec<XType>>, bool)> =
        exact_rows
            .into_iter()
            .filter_map(|(identity, rows)| {
                if rows.len() != 1 || exact_vetoes.contains(&identity) {
                    return None;
                }
                let value = rows.into_iter().next().unwrap();
                if exact_variadic.contains(&identity) != value.2 {
                    return None;
                }
                Some((identity, value))
            })
            .collect();
    let mut out = project_exact_identity_values(&identity_names, &exact_values);
    let name_collisions = loader_name_collisions_with_unowned_symbols(db, &identity_names);
    out.retain(|name, _| !name_collisions.contains(name));

    // Curated signatures not owned by a loader identity remain available by
    // their exact raw symbol.  Sanitizer collisions and signature/varargs
    // conflicts are vetoes, never deterministic picks.
    let loader_names: HashSet<String> = identity_names
        .values()
        .flat_map(|names| names.iter().cloned())
        .collect();
    let mut variadic_counts: BTreeMap<Symbol, BTreeSet<usize>> = BTreeMap::new();
    for &(name, fixed_count) in db.rel_iter::<(Symbol, usize)>("known_varargs_function") {
        variadic_counts.entry(name).or_default().insert(fixed_count);
    }
    let mut raw_signature_rows: BTreeMap<
        Symbol,
        BTreeSet<(usize, XType, Arc<Vec<XType>>)>,
    > = BTreeMap::new();
    for (name, arity, ret, params) in
        db.rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>("known_extern_signature")
    {
        raw_signature_rows
            .entry(*name)
            .or_default()
            .insert((*arity, *ret, params.clone()));
    }
    let mut raw_owners: BTreeMap<String, BTreeSet<Symbol>> = BTreeMap::new();
    let mut ordinary_rows: BTreeMap<
        String,
        BTreeSet<(XType, Arc<Vec<XType>>, bool)>,
    > = BTreeMap::new();
    for (name, facts) in raw_signature_rows {
        let sanitized = sanitize_c_symbol_name(name);
        if loader_names.contains(&sanitized) {
            continue;
        }
        raw_owners.entry(sanitized.clone()).or_default().insert(name);
        if facts.len() != 1 {
            continue;
        }
        let (arity, ret, params) = facts.into_iter().next().unwrap();
        if arity != params.len() {
            continue;
        }
        let counts = variadic_counts.get(name);
        if counts.map_or(false, |counts| counts.len() != 1) {
            continue;
        }
        let variadic = match counts.and_then(|counts| counts.iter().next().copied()) {
            Some(fixed_count) if fixed_count == params.len() => true,
            Some(_) => continue,
            None => false,
        };
        ordinary_rows
            .entry(sanitized)
            .or_default()
            .insert((ret, params, variadic));
    }
    for (name, rows) in ordinary_rows {
        if raw_owners.get(&name).map_or(0, |owners| owners.len()) == 1
            && rows.len() == 1
        {
            out.insert(name, rows.into_iter().next().unwrap());
        }
    }
    out
}

/// Authoritative prototypes usable for data globals invoked as functions.
/// Only loader-classified import-pointer objects qualify; a plain OBJECT whose
/// spelling happens to match a curated function must use call-site evidence.
pub(crate) fn known_import_pointer_signatures_from_db(
    db: &DecompileDB,
) -> HashMap<String, (XType, Arc<Vec<XType>>, bool)> {
    let exact_import_pointer_addresses = exact_import_pointer_call_addresses(db);
    let mut identity_names: BTreeMap<ExactLoaderIdentity, BTreeSet<String>> = BTreeMap::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_symbol_identity") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
    }
    let import_identities: BTreeMap<ExactLoaderIdentity, ()> = identity_names
        .keys()
        .filter(|identity| identity.kind == LoaderSymbolKind::ImportPointer)
        .cloned()
        .map(|identity| (identity, ()))
        .collect();
    let allowed_names: HashSet<String> =
        project_exact_identity_values(&identity_names, &import_identities)
            .into_keys()
            .collect();
    let mut signatures = known_loader_signatures_from_db(db);
    signatures.retain(|name, _| allowed_names.contains(name));
    signatures
}

/// Relation-backed entry point shared by direct external declarations and the
/// data-global/function-pointer declaration solver.
fn infer_shared_fixed_arities_from_db_filtered(
    db: &DecompileDB,
    required_kind: Option<LoaderSymbolKind>,
) -> HashMap<String, usize> {
    let exact_import_pointer_addresses = exact_import_pointer_call_addresses(db);
    let mut identity_names: BTreeMap<ExactLoaderIdentity, BTreeSet<String>> = BTreeMap::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_symbol_identity") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
    }
    let mut call_sites = Vec::new();
    for &(node, address, kind, provider, original) in db.rel_iter::<(
        Node,
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("call_loader_identity_candidate") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
        call_sites.push((node, identity));
    }
    // Hand-authored DBs and older serialized stages may contain only the
    // resolved relation.  Union it without changing exact identity semantics.
    for &(node, address, kind, provider, original) in db.rel_iter::<(
        Node,
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("call_loader_identity") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        if is_shadowed_same_address_function(&identity, &exact_import_pointer_addresses) {
            continue;
        }
        insert_identity_names(&mut identity_names, &identity, provider, original);
        call_sites.push((node, identity));
    }
    call_sites.sort();
    call_sites.dedup();

    let mut veto_sites: HashSet<Node> = db
        .rel_iter::<(Node,)>("call_forwarding_unresolved")
        .map(|(node,)| *node)
        .collect();
    veto_sites.extend(
        db.rel_iter::<(Node,)>("call_loader_identity_ambiguous")
            .map(|(node,)| *node),
    );
    if !db.abi().uses_shared_arg_slots() {
        // SysV float arguments live in an independent compact XMM sequence.
        // Until that sequence carries the same per-position provenance as GP
        // arguments, do not mistake a float-bearing callee for a shared
        // zero/fewer-argument prototype.  Its per-site RTL evidence is kept
        // and C emission remains safely unprototyped.
        veto_sites.extend(
            db.rel_iter::<(Node, usize)>("call_has_float_arg_evidence")
                .map(|(node, _)| *node),
        );
    }
    let provenance: Vec<(Node, usize, RTLReg, CallArgProvenance)> = db
        .rel_iter::<(Node, usize, RTLReg, CallArgProvenance)>("call_arg_provenance")
        .copied()
        .collect();
    let provenance_keys: HashSet<(Node, usize, RTLReg)> = provenance
        .iter()
        .map(|&(node, position, reg, _)| (node, position, reg))
        .collect();
    for &(node, position, reg) in
        db.rel_iter::<(Node, usize, RTLReg)>("call_arg_mapping")
    {
        if !provenance_keys.contains(&(node, position, reg)) {
            veto_sites.insert(node);
        }
    }
    let mut provenance_rows_by_site: HashMap<
        Node,
        Vec<(usize, RTLReg, CallArgProvenance)>,
    > = HashMap::new();
    for &(node, position, reg, source) in &provenance {
        provenance_rows_by_site
            .entry(node)
            .or_default()
            .push((position, reg, source));
    }
    let mut candidate_vectors_by_site: BTreeMap<
        Node,
        BTreeSet<Arc<Vec<RTLReg>>>,
    > = BTreeMap::new();
    for (node, args) in
        db.rel_iter::<(Node, Arc<Vec<RTLReg>>)>("call_args_collected_candidate")
    {
        candidate_vectors_by_site
            .entry(*node)
            .or_default()
            .insert(args.clone());
    }
    let exact_call_nodes: BTreeSet<Node> = call_sites.iter().map(|(node, _)| *node).collect();
    for node in exact_call_nodes {
        let rows = provenance_rows_by_site
            .get(&node)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let coherent = crate::decompile::passes::rtl_optimize_pass::coherent_call_vector(
            rows.iter().copied(),
        );
        let candidate = candidate_vectors_by_site.get(&node).and_then(|vectors| {
            (vectors.len() == 1).then(|| vectors.iter().next().unwrap())
        });
        if coherent.is_none() || coherent.as_ref() != candidate {
            veto_sites.insert(node);
        }
    }
    let mut known_variadic = HashSet::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("known_loader_variadic") {
        known_variadic.insert(ExactLoaderIdentity::from_row(
            address, kind, provider, original,
        ));
    }
    // Signature conflicts share the same fixed-arity veto path.  Calling the
    // set `known_variadic` here means "never infer fixed", not that the final C
    // declaration is necessarily variadic.
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_signature_conflict") {
        known_variadic.insert(ExactLoaderIdentity::from_row(
            address, kind, provider, original,
        ));
    }
    let mut exact_signature_facts: BTreeMap<
        ExactLoaderIdentity,
        BTreeSet<(usize, XType, Arc<Vec<XType>>, bool)>,
    > = BTreeMap::new();
    for (address, kind, provider, original, arity, ret, params, variadic) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
        usize,
        XType,
        Arc<Vec<XType>>,
        bool,
    )>("known_loader_signature") {
        let identity = ExactLoaderIdentity::from_row(
            *address, *kind, *provider, *original,
        );
        if *arity != params.len() {
            known_variadic.insert(identity);
            continue;
        }
        exact_signature_facts
            .entry(identity)
            .or_default()
            .insert((*arity, *ret, params.clone(), *variadic));
    }
    for (identity, facts) in exact_signature_facts {
        if facts.len() != 1
            || facts
                .iter()
                .any(|(_, _, _, variadic)| *variadic)
        {
            known_variadic.insert(identity);
        }
    }
    let mut exact = infer_shared_fixed_arities(
        &call_sites,
        &provenance,
        &known_variadic,
        &veto_sites,
    );
    if let Some(required_kind) = required_kind {
        exact.retain(|identity, _| identity.kind == required_kind);
    }
    let mut projected = project_exact_identity_values(&identity_names, &exact);
    let name_collisions = loader_name_collisions_with_unowned_symbols(db, &identity_names);
    projected.retain(|name, _| !name_collisions.contains(name));
    projected
}

pub(crate) fn infer_shared_fixed_arities_from_db(
    db: &DecompileDB,
) -> HashMap<String, usize> {
    infer_shared_fixed_arities_from_db_filtered(db, None)
}

/// Shared fixed arities usable for a global object called through memory.
/// Function-entry cohorts and ordinary data symbols are deliberately outside
/// this projection, even when their emitted spellings coincide.
pub(crate) fn infer_shared_import_pointer_fixed_arities_from_db(
    db: &DecompileDB,
) -> HashMap<String, usize> {
    infer_shared_fixed_arities_from_db_filtered(
        db,
        Some(LoaderSymbolKind::ImportPointer),
    )
}

/// Join per-site evidence into a parameter type vector per callee; a callee gets a typed prototype only when arity agrees across all sites and no position joins to Poison.
fn join_call_site_evidence(
    sites_by_callee: &HashMap<String, Vec<Vec<ArgEvidence>>>,
) -> HashMap<String, Vec<CType>> {
    let mut joined: HashMap<String, Vec<CType>> = HashMap::new();
    for (name, sites) in sites_by_callee {
        if let Some(param_types) = join_one_callee_evidence(sites) {
            joined.insert(name.clone(), param_types);
        }
    }
    joined
}

fn join_one_callee_evidence(sites: &[Vec<ArgEvidence>]) -> Option<Vec<CType>> {
    let arity = sites.first()?.len();
    if sites.iter().any(|site| site.len() != arity) {
        return None;
    }
    let mut acc = vec![ArgEvidence::Bottom; arity];
    for site in sites {
        for (slot, evidence) in acc.iter_mut().zip(site) {
            *slot = std::mem::replace(slot, ArgEvidence::Bottom).join(evidence.clone());
        }
    }
    acc.iter().map(ArgEvidence::to_ctype).collect()
}

/// Loader aliases of one callable object share one evidence join.  Conversely,
/// a spelling owned by multiple address+kind identities is omitted rather
/// than allowing each display name to select a different prototype.
fn join_call_site_evidence_by_loader_identity(
    db: &DecompileDB,
    raw: &HashMap<String, Vec<Vec<ArgEvidence>>>,
) -> HashMap<String, Vec<CType>> {
    let mut identity_names: BTreeMap<ExactLoaderIdentity, BTreeSet<String>> = BTreeMap::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_symbol_identity") {
        let identity = ExactLoaderIdentity::from_row(address, kind, provider, original);
        insert_identity_names(&mut identity_names, &identity, provider, original);
    }
    let mut owners: BTreeMap<String, BTreeSet<ExactLoaderIdentity>> = BTreeMap::new();
    for (identity, names) in &identity_names {
        for name in names {
            owners
                .entry(name.clone())
                .or_default()
                .insert(identity.clone());
        }
    }
    let name_collisions = loader_name_collisions_with_unowned_symbols(db, &identity_names);

    let mut ordinary = HashMap::new();
    let mut exact: BTreeMap<ExactLoaderIdentity, Vec<Vec<ArgEvidence>>> = BTreeMap::new();
    for (name, sites) in raw {
        if name_collisions.contains(name) {
            continue;
        }
        match owners.get(name) {
            None => {
                ordinary.insert(name.clone(), sites.clone());
            }
            Some(identities) if identities.len() == 1 => {
                exact
                    .entry(identities.iter().next().unwrap().clone())
                    .or_default()
                    .extend(sites.iter().cloned());
            }
            Some(_) => {}
        }
    }

    let mut joined = join_call_site_evidence(&ordinary);
    for (identity, sites) in exact {
        let Some(types) = join_one_callee_evidence(&sites) else {
            continue;
        };
        if let Some(names) = identity_names.get(&identity) {
            for name in names {
                if !name_collisions.contains(name)
                    && owners.get(name).map_or(0, |identities| identities.len()) == 1
                {
                    joined.insert(name.clone(), types.clone());
                }
            }
        }
    }
    joined
}

/// Rename duplicate labels within a function body so each is unique: first occurrence keeps its name, subsequent get `_2`, `_3`, etc.
fn deduplicate_labels(body_items: &mut Vec<CBlockItem>) {
    // Pass 1: collect all label occurrences in order
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    let mut rename_map: HashMap<(String, usize), String> = HashMap::new();

    for item in body_items.iter() {
        if let CBlockItem::Stmt(stmt) = item {
            count_labels_in_stmt(stmt, &mut label_counts, &mut rename_map);
        }
    }

    // Only proceed if there are duplicates
    if rename_map.is_empty() {
        return;
    }

    // Build a positional rename map: track label occurrences and rename duplicates
    let mut occurrence: HashMap<String, usize> = HashMap::new();
    for item in body_items.iter_mut() {
        if let CBlockItem::Stmt(stmt) = item {
            *stmt = rename_duplicate_labels_in_stmt(stmt, &rename_map, &mut occurrence);
        }
    }
}

fn count_labels_in_stmt(
    stmt: &CStmt,
    counts: &mut HashMap<String, usize>,
    rename_map: &mut HashMap<(String, usize), String>,
) {
    match stmt {
        CStmt::Labeled(Label::Named(name), inner) => {
            let occ = counts.entry(name.clone()).or_insert(0);
            *occ += 1;
            if *occ > 1 {
                rename_map.insert((name.clone(), *occ), format!("{}_{}", name, occ));
            }
            count_labels_in_stmt(inner, counts, rename_map);
        }
        CStmt::If(_, then_s, else_s) => {
            count_labels_in_stmt(then_s, counts, rename_map);
            if let Some(e) = else_s {
                count_labels_in_stmt(e, counts, rename_map);
            }
        }
        CStmt::While(_, body)
        | CStmt::DoWhile(body, _)
        | CStmt::For(_, _, _, body)
        | CStmt::Switch(_, body) => {
            count_labels_in_stmt(body, counts, rename_map);
        }
        CStmt::Sequence(stmts) => {
            for s in stmts {
                count_labels_in_stmt(s, counts, rename_map);
            }
        }
        CStmt::Block(items) => {
            for item in items {
                if let CBlockItem::Stmt(s) = item {
                    count_labels_in_stmt(s, counts, rename_map);
                }
            }
        }
        _ => {}
    }
}

fn rename_duplicate_labels_in_stmt(
    stmt: &CStmt,
    rename_map: &HashMap<(String, usize), String>,
    occurrence: &mut HashMap<String, usize>,
) -> CStmt {
    match stmt {
        CStmt::Labeled(Label::Named(name), inner) => {
            let occ = occurrence.entry(name.clone()).or_insert(0);
            *occ += 1;
            let new_name = if *occ > 1 {
                rename_map
                    .get(&(name.clone(), *occ))
                    .cloned()
                    .unwrap_or_else(|| name.clone())
            } else {
                name.clone()
            };
            let inner = rename_duplicate_labels_in_stmt(inner, rename_map, occurrence);
            CStmt::Labeled(Label::Named(new_name), Box::new(inner))
        }
        CStmt::Goto(target) => {
            // Gotos to duplicate labels redirect to the first occurrence (only duplicates are renamed)
            CStmt::Goto(target.clone())
        }
        CStmt::If(cond, then_s, else_s) => {
            let then_s = Box::new(rename_duplicate_labels_in_stmt(
                then_s, rename_map, occurrence,
            ));
            let else_s = else_s
                .as_ref()
                .map(|e| Box::new(rename_duplicate_labels_in_stmt(e, rename_map, occurrence)));
            CStmt::If(cond.clone(), then_s, else_s)
        }
        CStmt::While(cond, body) => CStmt::While(
            cond.clone(),
            Box::new(rename_duplicate_labels_in_stmt(
                body, rename_map, occurrence,
            )),
        ),
        CStmt::DoWhile(body, cond) => CStmt::DoWhile(
            Box::new(rename_duplicate_labels_in_stmt(
                body, rename_map, occurrence,
            )),
            cond.clone(),
        ),
        CStmt::For(i, c, u, body) => CStmt::For(
            i.clone(),
            c.clone(),
            u.clone(),
            Box::new(rename_duplicate_labels_in_stmt(
                body, rename_map, occurrence,
            )),
        ),
        CStmt::Switch(expr, body) => CStmt::Switch(
            expr.clone(),
            Box::new(rename_duplicate_labels_in_stmt(
                body, rename_map, occurrence,
            )),
        ),
        CStmt::Sequence(stmts) => CStmt::Sequence(
            stmts
                .iter()
                .map(|s| rename_duplicate_labels_in_stmt(s, rename_map, occurrence))
                .collect(),
        ),
        CStmt::Block(items) => CStmt::Block(
            items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => {
                        CBlockItem::Stmt(rename_duplicate_labels_in_stmt(s, rename_map, occurrence))
                    }
                    other => other.clone(),
                })
                .collect(),
        ),
        CStmt::Labeled(label, inner) => CStmt::Labeled(
            label.clone(),
            Box::new(rename_duplicate_labels_in_stmt(
                inner, rename_map, occurrence,
            )),
        ),
        _ => stmt.clone(),
    }
}

fn ensure_goto_label_consistency(body_items: &mut Vec<CBlockItem>) {
    let mut goto_targets: HashSet<String> = HashSet::new();
    let mut label_defs: HashSet<String> = HashSet::new();

    for item in body_items.iter() {
        if let CBlockItem::Stmt(stmt) = item {
            for target in crate::decompile::passes::c_pass::helpers::collect_goto_targets(stmt) {
                goto_targets.insert(target);
            }
            for label in crate::decompile::passes::c_pass::helpers::collect_all_labels(stmt) {
                label_defs.insert(label);
            }
        }
    }

    let missing_targets: HashSet<String> = goto_targets.difference(&label_defs).cloned().collect();
    if !missing_targets.is_empty() {
        for item in body_items.iter_mut() {
            if let CBlockItem::Stmt(stmt) = item {
                *stmt = remove_gotos_to_missing_labels(stmt, &missing_targets);
            }
        }
    }
}

fn remove_gotos_to_missing_labels(stmt: &CStmt, missing: &HashSet<String>) -> CStmt {
    match stmt {
        CStmt::Goto(target) if missing.contains(target) => CStmt::Empty,
        CStmt::If(cond, then_s, else_s) => CStmt::If(
            cond.clone(),
            Box::new(remove_gotos_to_missing_labels(then_s, missing)),
            else_s
                .as_ref()
                .map(|e| Box::new(remove_gotos_to_missing_labels(e, missing))),
        ),
        CStmt::While(cond, body) => CStmt::While(
            cond.clone(),
            Box::new(remove_gotos_to_missing_labels(body, missing)),
        ),
        CStmt::DoWhile(body, cond) => CStmt::DoWhile(
            Box::new(remove_gotos_to_missing_labels(body, missing)),
            cond.clone(),
        ),
        CStmt::For(init, cond, update, body) => CStmt::For(
            init.clone(),
            cond.clone(),
            update.clone(),
            Box::new(remove_gotos_to_missing_labels(body, missing)),
        ),
        CStmt::Switch(expr, body) => CStmt::Switch(
            expr.clone(),
            Box::new(remove_gotos_to_missing_labels(body, missing)),
        ),
        CStmt::Sequence(stmts) => CStmt::Sequence(
            stmts
                .iter()
                .map(|s| remove_gotos_to_missing_labels(s, missing))
                .collect(),
        ),
        CStmt::Block(items) => CStmt::Block(
            items
                .iter()
                .map(|item| match item {
                    CBlockItem::Stmt(s) => {
                        CBlockItem::Stmt(remove_gotos_to_missing_labels(s, missing))
                    }
                    other => other.clone(),
                })
                .collect(),
        ),
        CStmt::Labeled(l, inner) => CStmt::Labeled(
            l.clone(),
            Box::new(remove_gotos_to_missing_labels(inner, missing)),
        ),
        other => other.clone(),
    }
}

// Return-value forwarding: when `if (...) { var = val; } else { var = val2; } return var;` transform to `if (...) { return val; } else { return val2; }`
fn forward_return_value(stmt: &CStmt) -> CStmt {
    match stmt {
        CStmt::Block(items) => {
            let new_items = forward_return_in_block_items(items);
            CStmt::Block(new_items)
        }
        CStmt::Sequence(stmts) => {
            let items: Vec<CBlockItem> =
                stmts.iter().map(|s| CBlockItem::Stmt(s.clone())).collect();
            let new_items = forward_return_in_block_items(&items);
            let new_stmts: Vec<CStmt> = new_items
                .into_iter()
                .filter_map(|i| match i {
                    CBlockItem::Stmt(s) => Some(s),
                    _ => None,
                })
                .collect();
            if new_stmts.len() == 1 {
                new_stmts.into_iter().next().unwrap()
            } else {
                CStmt::Sequence(new_stmts)
            }
        }
        CStmt::If(cond, then_s, else_s) => CStmt::If(
            cond.clone(),
            Box::new(forward_return_value(then_s)),
            else_s.as_ref().map(|s| Box::new(forward_return_value(s))),
        ),
        CStmt::While(cond, body) => {
            CStmt::While(cond.clone(), Box::new(forward_return_value(body)))
        }
        CStmt::DoWhile(body, cond) => {
            CStmt::DoWhile(Box::new(forward_return_value(body)), cond.clone())
        }
        CStmt::For(init, cond, upd, body) => CStmt::For(
            init.clone(),
            cond.clone(),
            upd.clone(),
            Box::new(forward_return_value(body)),
        ),
        CStmt::Labeled(l, inner) => {
            CStmt::Labeled(l.clone(), Box::new(forward_return_value(inner)))
        }
        _ => stmt.clone(),
    }
}

fn forward_return_in_block_items(items: &[CBlockItem]) -> Vec<CBlockItem> {
    let mut result = items.to_vec();
    // Look for: ..., If(cond, then, else), Return(var) where If assigns var in every branch
    let mut i = 0;
    while i + 1 < result.len() {
        let var_match = match (&result[i], &result[i + 1]) {
            (
                CBlockItem::Stmt(if_stmt @ CStmt::If(_, _, Some(_))),
                CBlockItem::Stmt(CStmt::Return(Some(CExpr::Var(ret_var)))),
            ) => {
                if all_branches_assign_last(if_stmt, ret_var) {
                    Some(ret_var.clone())
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(var_name) = var_match {
            if let CBlockItem::Stmt(if_stmt) = &result[i] {
                let transformed = replace_last_assign_with_return(if_stmt, &var_name);
                result[i] = CBlockItem::Stmt(forward_return_value(&transformed));
                result.remove(i + 1);
                continue;
            }
        }
        i += 1;
    }
    for item in &mut result {
        if let CBlockItem::Stmt(s) = item {
            *s = forward_return_value(s);
        }
    }
    result
}

// Check if every branch of the if-else tree assigns to `var` as its last effective statement
fn all_branches_assign_last(stmt: &CStmt, var: &str) -> bool {
    match stmt {
        CStmt::If(_, then_s, Some(else_s)) => {
            last_stmt_assigns(then_s, var) && last_stmt_assigns(else_s, var)
        }
        _ => false,
    }
}

fn last_stmt_assigns(stmt: &CStmt, var: &str) -> bool {
    match stmt {
        CStmt::Expr(CExpr::Assign(AssignOp::Assign, lhs, _)) => {
            matches!(lhs.as_ref(), CExpr::Var(v) if v == var)
        }
        CStmt::If(_, _, Some(_)) => all_branches_assign_last(stmt, var),
        CStmt::Block(items) => items
            .iter()
            .rev()
            .find_map(|i| match i {
                CBlockItem::Stmt(s) if !matches!(s, CStmt::Empty) => Some(s),
                _ => None,
            })
            .map_or(false, |s| last_stmt_assigns(s, var)),
        CStmt::Sequence(stmts) => stmts.last().map_or(false, |s| last_stmt_assigns(s, var)),
        _ => false,
    }
}

// Replace the last `var = val` in each branch with `return val`
fn replace_last_assign_with_return(stmt: &CStmt, var: &str) -> CStmt {
    match stmt {
        CStmt::If(cond, then_s, Some(else_s)) => CStmt::If(
            cond.clone(),
            Box::new(replace_last_assign_with_return(then_s, var)),
            Some(Box::new(replace_last_assign_with_return(else_s, var))),
        ),
        CStmt::Expr(CExpr::Assign(AssignOp::Assign, lhs, rhs)) if matches!(lhs.as_ref(), CExpr::Var(v) if v == var) => {
            CStmt::Return(Some(*rhs.clone()))
        }
        CStmt::Block(items) => {
            let mut new_items = items.clone();
            if let Some(last_stmt_idx) = new_items
                .iter()
                .rposition(|i| matches!(i, CBlockItem::Stmt(s) if !matches!(s, CStmt::Empty)))
            {
                if let CBlockItem::Stmt(s) = &new_items[last_stmt_idx] {
                    new_items[last_stmt_idx] =
                        CBlockItem::Stmt(replace_last_assign_with_return(s, var));
                }
            }
            CStmt::Block(new_items)
        }
        CStmt::Sequence(stmts) => {
            let mut new_stmts = stmts.clone();
            if let Some(last) = new_stmts.last_mut() {
                *last = replace_last_assign_with_return(last, var);
            }
            CStmt::Sequence(new_stmts)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod callee_identity_tests {
    use super::*;

    fn function_pointer_type() -> ClightType {
        ClightType::Tpointer(
            Arc::new(ClightType::Tfunction(
                Arc::new(vec![ClightType::Tlong(
                    ClightSignedness::Signed,
                    ClightAttr::default(),
                )]),
                Arc::new(ClightType::Tvoid),
                CallConv::default(),
            )),
            ClightAttr::default(),
        )
    }

    fn actual_site_unprototyped_function_pointer_type(params: Vec<ClightType>) -> ClightType {
        ClightType::Tpointer(
            Arc::new(ClightType::Tfunction(
                Arc::new(params),
                Arc::new(ClightType::Tvoid),
                CallConv {
                    unproto: true,
                    ..CallConv::default()
                },
            )),
            ClightAttr::default(),
        )
    }

    #[test]
    fn unmapped_nonlocal_evar_callee_stays_direct() {
        let function_address: Address = 0x401000;
        let callee_id: Ident = 0x402345;
        let mut context = ConversionContext::new(HashMap::new());
        context.enter_function(function_address, None);

        let converted = convert_stmt(
            &ClightStmt::Scall(
                None,
                ClightExpr::Evar(callee_id, function_pointer_type()),
                vec![],
            ),
            &mut context,
        );

        assert_eq!(
            converted,
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var("FUN_402345".to_string())),
                vec![],
            ))
        );
        assert!(
            context
                .function_object_types()
                .get(&function_address)
                .is_none(),
            "a direct callee must not acquire a local object declaration"
        );
        let mut local_names = HashSet::new();
        collect_var_names_from_stmt(&converted, &mut local_names);
        assert!(local_names.is_empty());
        let mut called = HashSet::new();
        collect_nonlocal_called_names_in_stmt(&converted, &local_names, &mut called);
        assert_eq!(called, HashSet::from(["FUN_402345".to_string()]));
    }

    #[test]
    fn cast_wrapped_unmapped_nonlocal_evar_stays_direct() {
        let function_address: Address = 0x401000;
        let callee_id: Ident = 0x402346;
        let function_pointer = function_pointer_type();
        let callee = ClightExpr::Ecast(
            Box::new(ClightExpr::Evar(callee_id, function_pointer.clone())),
            function_pointer,
        );
        let mut context = ConversionContext::new(HashMap::new());
        context.enter_function(function_address, None);

        let converted = convert_stmt(&ClightStmt::Scall(None, callee, vec![]), &mut context);

        assert_eq!(
            converted,
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var("FUN_402346".to_string())),
                vec![],
            )),
            "a provider cast must not hide a direct callee from declaration recovery"
        );
        let mut called = HashSet::new();
        collect_nonlocal_called_names_in_stmt(&converted, &HashSet::new(), &mut called);
        assert_eq!(called, HashSet::from(["FUN_402346".to_string()]));
    }

    #[test]
    fn direct_unprototyped_call_materializes_a_call_site_cast() {
        let function_address: Address = 0x401000;
        let clight_type = actual_site_unprototyped_function_pointer_type(vec![
            ClightType::Tlong(ClightSignedness::Signed, ClightAttr::default()),
        ]);
        let expected_type = convert_clight_type(&clight_type);
        let mut context = ConversionContext::new(HashMap::new());
        context.enter_function(function_address, None);

        let converted = convert_stmt(
            &ClightStmt::Scall(
                None,
                ClightExpr::EvarSymbol("short_exact".to_string(), clight_type),
                vec![ClightExpr::EconstLong(
                    7,
                    ClightType::Tlong(
                        ClightSignedness::Signed,
                        ClightAttr::default(),
                    ),
                )],
            ),
            &mut context,
        );

        assert_eq!(
            converted,
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Cast(
                    expected_type,
                    Box::new(CExpr::Var("short_exact".to_string())),
                )),
                vec![CExpr::IntLit(IntLiteral {
                    value: 7,
                    suffix: IntLiteralSuffix::L,
                    base: IntLiteralBase::Decimal,
                })],
            ))
        );
        let mut called = HashSet::new();
        collect_nonlocal_called_names_in_stmt(&converted, &HashSet::new(), &mut called);
        assert_eq!(called, HashSet::from(["short_exact".to_string()]));
    }

    #[test]
    fn declared_local_evar_callee_keeps_pointer_type() {
        let function_address: Address = 0x401000;
        let local_id: Ident = 7;
        let clight_type = function_pointer_type();
        let expected_type = convert_clight_type(&clight_type);
        let mut context = ConversionContext::new(HashMap::new());
        context.enter_function(function_address, Some(&HashSet::from([local_id])));

        let converted = convert_stmt(
            &ClightStmt::Scall(None, ClightExpr::Evar(local_id, clight_type), vec![]),
            &mut context,
        );

        assert_eq!(
            converted,
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var("var_7".to_string())),
                vec![],
            ))
        );
        let scoped_types = context
            .function_object_types()
            .get(&function_address)
            .expect("function-local callee type");
        assert_eq!(scoped_types.get("var_7"), Some(&expected_type));
        let mut reconciled_types = HashMap::from([("var_7".to_string(), CType::int())]);
        retain_clight_function_object_types(&converted, scoped_types, &mut reconciled_types);
        assert_eq!(reconciled_types.get("var_7"), Some(&expected_type));

        let locals = HashSet::from(["var_7".to_string()]);
        let mut called = HashSet::new();
        collect_nonlocal_called_names_in_stmt(&converted, &locals, &mut called);
        assert!(called.is_empty());
    }

    #[test]
    fn etempvar_indirect_callee_keeps_pointer_type() {
        let function_address: Address = 0x401000;
        let temp_id: Ident = 13;
        let clight_type = function_pointer_type();
        let expected_type = convert_clight_type(&clight_type);
        let callee = ClightExpr::Ecast(
            Box::new(ClightExpr::Etempvar(
                temp_id,
                ClightType::Tlong(ClightSignedness::Signed, ClightAttr::default()),
            )),
            clight_type,
        );
        let mut context = ConversionContext::new(HashMap::new());
        context.enter_function(function_address, None);

        let converted = convert_stmt(&ClightStmt::Scall(None, callee, vec![]), &mut context);

        assert_eq!(
            converted,
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Cast(
                    expected_type.clone(),
                    Box::new(CExpr::Var("var_13".to_string())),
                )),
                vec![],
            ))
        );
        let scoped_types = context
            .function_object_types()
            .get(&function_address)
            .expect("function-local callee type");
        assert_eq!(scoped_types.get("var_13"), Some(&expected_type));
        let mut reconciled_types = HashMap::from([("var_13".to_string(), CType::long())]);

        retain_clight_function_object_types(&converted, scoped_types, &mut reconciled_types);

        assert_eq!(reconciled_types.get("var_13"), Some(&expected_type));
    }

    #[test]
    fn actual_local_callee_type_wins_conflicting_scalar_authority() {
        let function_pointer = convert_clight_type(&function_pointer_type());
        let body = CStmt::Expr(CExpr::Call(
            Box::new(CExpr::Var("var_21".to_string())),
            vec![],
        ));
        let call_target_types =
            HashMap::from([("var_21".to_string(), function_pointer.clone())]);
        // Model a late integer-only declaration heuristic classifying the same
        // register as a scalar before call-target reconciliation runs.
        let mut reconciled_types = HashMap::from([("var_21".to_string(), CType::long())]);

        retain_clight_function_object_types(
            &body,
            &call_target_types,
            &mut reconciled_types,
        );

        assert_eq!(reconciled_types.get("var_21"), Some(&function_pointer));
    }

    #[test]
    fn local_indirect_callee_is_not_an_external_forward_declaration() {
        let body = CStmt::Sequence(vec![
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var("var_0".to_string())),
                vec![CExpr::int(1)],
            )),
            CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var("external_target".to_string())),
                vec![],
            )),
        ]);
        let locals = HashSet::from(["var_0".to_string()]);
        let mut called = HashSet::new();

        collect_nonlocal_called_names_in_stmt(&body, &locals, &mut called);

        assert_eq!(called, HashSet::from(["external_target".to_string()]));
    }

    #[test]
    fn local_indirect_callee_does_not_contribute_direct_call_signature_evidence() {
        let local_function_pointer = CType::ptr(CType::Function(
            Box::new(CType::Void),
            vec![CType::long()],
            false,
            false,
        ));
        let local_types = HashMap::from([("var_0".to_string(), local_function_pointer)]);
        let global_types = HashMap::new();
        let callee_ret = HashMap::new();
        let environment = ArgEvidenceEnv {
            local_types,
            global_types: &global_types,
            callee_ret: &callee_ret,
        };
        let expression = CExpr::Call(
            Box::new(CExpr::Paren(Box::new(CExpr::Cast(
                CType::ptr(CType::Function(
                    Box::new(CType::Void),
                    vec![CType::long()],
                    false,
                    true,
                )),
                Box::new(CExpr::Var("var_0".to_string())),
            )))),
            vec![CExpr::int(1)],
        );
        let mut sites = HashMap::new();

        collect_call_arg_evidence_in_expr(&expression, &environment, &mut sites);

        assert!(
            sites.is_empty(),
            "an indirect local must not synthesize a file-scope function prototype"
        );
    }

    #[test]
    fn cast_wrapped_nonlocal_callee_contributes_actual_argument_evidence() {
        let global_types = HashMap::new();
        let callee_ret = HashMap::new();
        let environment = ArgEvidenceEnv {
            local_types: HashMap::new(),
            global_types: &global_types,
            callee_ret: &callee_ret,
        };
        let expression = CExpr::Call(
            Box::new(CExpr::Paren(Box::new(CExpr::Cast(
                CType::ptr(CType::Function(
                    Box::new(CType::int()),
                    vec![CType::long()],
                    false,
                    true,
                )),
                Box::new(CExpr::Var("headerless_external".to_string())),
            )))),
            vec![CExpr::int(1)],
        );
        let mut sites = HashMap::new();

        collect_call_arg_evidence_in_expr(&expression, &environment, &mut sites);

        assert_eq!(
            sites.get("headerless_external"),
            Some(&vec![vec![ArgEvidence::Int(32, Signedness::Signed)]])
        );
    }
}

#[cfg(test)]
mod arg_evidence_tests {
    use super::*;
    use crate::decompile::passes::c_pass::types::{IntSize, TopLevelDecl};
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    fn i32_() -> ArgEvidence {
        ArgEvidence::Int(32, Signedness::Signed)
    }
    fn i64_() -> ArgEvidence {
        ArgEvidence::Int(64, Signedness::Signed)
    }
    fn u32_() -> ArgEvidence {
        ArgEvidence::Int(32, Signedness::Unsigned)
    }
    fn f64_() -> ArgEvidence {
        ArgEvidence::Float(64)
    }
    fn ptr_char() -> ArgEvidence {
        ArgEvidence::ObjectPtr(Some(CType::ptr(CType::char_signed())))
    }
    fn ptr_int() -> ArgEvidence {
        ArgEvidence::ObjectPtr(Some(CType::ptr(CType::int())))
    }
    fn fn_ptr(params: Vec<CType>) -> ArgEvidence {
        ArgEvidence::FunctionPtr(Some(CType::ptr(CType::Function(
            Box::new(CType::Void),
            params,
            false,
            false,
        ))))
    }

    #[test]
    fn join_int_widths() {
        // int32 | int64 -> int64 (the FIXPLAN 5.5 example).
        assert_eq!(i32_().join(i64_()), i64_());
        // equal width, sign disagreement -> signed.
        assert_eq!(u32_().join(i32_()), i32_());
        assert_eq!(
            i32_().join(i64_()).to_ctype(),
            Some(CType::Int(IntSize::Long, Signedness::Signed))
        );
    }

    #[test]
    fn class_conflicts_poison_without_changing_register_class() {
        for (left, right) in [
            (i64_(), ptr_char()),
            (f64_(), ptr_char()),
            (i64_(), f64_()),
            (ArgEvidence::GpZero, f64_()),
            (fn_ptr(vec![]), ptr_char()),
        ] {
            assert_eq!(left.clone().join(right.clone()), ArgEvidence::Poison);
            assert_eq!(right.join(left), ArgEvidence::Poison);
        }
        let function_type = CType::Function(Box::new(CType::Void), Vec::new(), false, false);
        assert!(matches!(
            arg_evidence_of_ctype(&CType::ptr(CType::ptr(function_type))),
            ArgEvidence::ObjectPtr(_)
        ));
    }

    #[test]
    fn arithmetic_conversion_is_separate_from_evidence_join() {
        assert_eq!(
            arithmetic_arg_evidence(i32_(), ArgEvidence::Float(32)),
            ArgEvidence::Float(32)
        );
        assert_eq!(i32_().join(ArgEvidence::Float(32)), ArgEvidence::Poison);
    }

    #[test]
    fn code_pointer_address_expressions_preserve_pointer_kind() {
        let function = CType::Function(
            Box::new(CType::int()),
            vec![CType::long()],
            false,
            false,
        );
        let function_pointer = CType::ptr(function.clone());
        let local_types = HashMap::from([("fp".to_string(), function_pointer.clone())]);
        let global_types = HashMap::from([("callback".to_string(), function.clone())]);
        let callee_ret = HashMap::from([("callback".to_string(), CType::int())]);
        let env = ArgEvidenceEnv {
            local_types,
            global_types: &global_types,
            callee_ret: &callee_ret,
        };

        let address_of_deref = CExpr::Unary(
            UnaryOp::AddrOf,
            Box::new(CExpr::Unary(
                UnaryOp::Deref,
                Box::new(CExpr::Var("fp".to_string())),
            )),
        );
        let address_of_callback = CExpr::Unary(
            UnaryOp::AddrOf,
            Box::new(CExpr::Var("callback".to_string())),
        );
        let address_of_pointer_object = CExpr::Unary(
            UnaryOp::AddrOf,
            Box::new(CExpr::Var("fp".to_string())),
        );

        let expected = ArgEvidence::FunctionPtr(Some(function_pointer.clone()));
        assert_eq!(arg_evidence_of_expr(&address_of_deref, &env), expected);
        assert_eq!(arg_evidence_of_expr(&address_of_callback, &env), expected);
        assert!(matches!(
            arg_evidence_of_expr(&address_of_pointer_object, &env),
            ArgEvidence::ObjectPtr(_)
        ));
        assert_eq!(
            address_of_arg_evidence(
                &CExpr::Member(Box::new(CExpr::Var("unknown".to_string())), "cb".to_string()),
                &env,
            ),
            ArgEvidence::Poison
        );
    }

    #[test]
    fn floating_operands_poison_integer_only_operators() {
        let local_types = HashMap::new();
        let global_types = HashMap::new();
        let callee_ret = HashMap::new();
        let env = ArgEvidenceEnv {
            local_types,
            global_types: &global_types,
            callee_ret: &callee_ret,
        };
        let floating = || {
            CExpr::FloatLit(FloatLiteral {
                value: 1.5,
                suffix: FloatLiteralSuffix::None,
            })
        };
        for op in [
            BinaryOp::Mod,
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
        ] {
            let expression = CExpr::Binary(
                op,
                Box::new(floating()),
                Box::new(CExpr::int(1)),
            );
            assert_eq!(
                arg_evidence_of_expr(&expression, &env),
                ArgEvidence::Poison,
                "left float accepted by {op:?}"
            );
        }
        for op in [BinaryOp::Shl, BinaryOp::Shr, BinaryOp::Mod] {
            let expression = CExpr::Binary(
                op,
                Box::new(CExpr::int(1)),
                Box::new(floating()),
            );
            assert_eq!(
                arg_evidence_of_expr(&expression, &env),
                ArgEvidence::Poison,
                "right float accepted by {op:?}"
            );
        }
    }

    #[test]
    fn join_pointers() {
        // identical pointer types stay exact; differing ones unify to void*.
        assert_eq!(ptr_char().join(ptr_char()), ptr_char());
        assert_eq!(
            ptr_char().join(ptr_int()),
            ArgEvidence::ObjectPtr(None)
        );
        assert_eq!(
            ArgEvidence::ObjectPtr(None).to_ctype(),
            Some(CType::ptr(CType::Void))
        );
        let fp = fn_ptr(vec![]);
        assert_eq!(fp.clone().join(ptr_char()), ArgEvidence::Poison);
        assert_eq!(fp.clone().join(fp.clone()), fp);
        assert_eq!(fn_ptr(vec![]).join(fn_ptr(vec![CType::long()])), ArgEvidence::Poison);
    }

    #[test]
    fn join_bottom_and_poison() {
        // True bottom constrains nothing; GP zero can become an integer/null
        // pointer but cannot migrate to the floating register class.
        assert_eq!(ArgEvidence::Bottom.join(ptr_char()), ptr_char());
        assert_eq!(ArgEvidence::Bottom.join(i32_()), i32_());
        assert_eq!(ArgEvidence::Bottom.to_ctype(), Some(CType::long()));
        assert_eq!(ArgEvidence::GpZero.clone().join(ptr_char()), ptr_char());
        assert_eq!(ArgEvidence::GpZero.clone().join(i32_()), i32_());
        assert_eq!(ArgEvidence::GpZero.join(f64_()), ArgEvidence::Poison);
        assert_eq!(ArgEvidence::Poison.join(i64_()), ArgEvidence::Poison);
        assert_eq!(ArgEvidence::Poison.to_ctype(), None);
    }

    #[test]
    fn join_is_commutative_associative_idempotent() {
        let samples = [
            ArgEvidence::Bottom,
            ArgEvidence::GpZero,
            i32_(),
            i64_(),
            u32_(),
            ArgEvidence::Float(32),
            f64_(),
            ptr_char(),
            ptr_int(),
            ArgEvidence::ObjectPtr(None),
            fn_ptr(vec![]),
            fn_ptr(vec![CType::long()]),
            ArgEvidence::FunctionPtr(None),
            ArgEvidence::Poison,
        ];
        for a in &samples {
            assert_eq!(a.clone().join(a.clone()), *a, "idempotence {:?}", a);
            for b in &samples {
                assert_eq!(
                    a.clone().join(b.clone()),
                    b.clone().join(a.clone()),
                    "commutativity {:?} {:?}",
                    a,
                    b
                );
                for c in &samples {
                    let ab_c = a.clone().join(b.clone()).join(c.clone());
                    let a_bc = a.clone().join(b.clone().join(c.clone()));
                    assert_eq!(ab_c, a_bc, "associativity {:?} {:?} {:?}", a, b, c);

                    let expected = a.clone().join(b.clone()).join(c.clone());
                    for permutation in [
                        [a, b, c],
                        [a, c, b],
                        [b, a, c],
                        [b, c, a],
                        [c, a, b],
                        [c, b, a],
                    ] {
                        let actual = permutation
                            .into_iter()
                            .cloned()
                            .fold(ArgEvidence::Bottom, ArgEvidence::join);
                        assert_eq!(actual, expected, "permutation {:?}", permutation);
                    }
                }
            }
        }
    }

    #[test]
    fn join_sites_requires_arity_agreement() {
        let mut sites: HashMap<String, Vec<Vec<ArgEvidence>>> = HashMap::new();
        sites.insert("f".to_string(), vec![vec![i32_()], vec![i64_()]]);
        sites.insert("g".to_string(), vec![vec![i32_()], vec![i32_(), i32_()]]);
        sites.insert("h".to_string(), vec![vec![ArgEvidence::Poison]]);
        sites.insert("z".to_string(), vec![vec![], vec![]]);
        let joined = join_call_site_evidence(&sites);
        assert_eq!(
            joined.get("f"),
            Some(&vec![CType::Int(IntSize::Long, Signedness::Signed)])
        );
        assert!(
            !joined.contains_key("g"),
            "arity mismatch must fall back to K&R"
        );
        assert!(!joined.contains_key("h"), "poison must fall back to K&R");
        assert_eq!(
            joined.get("z"),
            Some(&vec![]),
            "consistent zero-arity -> (void)"
        );
    }

    #[test]
    fn call_evidence_joins_aliases_and_vetoes_identity_collisions() {
        let mut db = DecompileDB::default();
        db.rel_push(
            "loader_symbol_identity",
            (
                0x9000u64,
                LoaderSymbolKind::Function,
                "alias_provider" as Symbol,
                "alias_raw" as Symbol,
            ),
        );
        let conflicting_aliases = HashMap::from([
            ("alias_provider".to_string(), vec![vec![i32_()]]),
            ("alias_raw".to_string(), vec![vec![f64_()]]),
        ]);
        let joined = join_call_site_evidence_by_loader_identity(&db, &conflicting_aliases);
        assert!(!joined.contains_key("alias_provider"));
        assert!(!joined.contains_key("alias_raw"));

        db.rel_push(
            "loader_symbol_identity",
            (
                0xa000u64,
                LoaderSymbolKind::ImportPointer,
                "alias_provider" as Symbol,
                "second_raw" as Symbol,
            ),
        );
        let collided = HashMap::from([("alias_provider".to_string(), vec![vec![i32_()]])]);
        assert!(join_call_site_evidence_by_loader_identity(&db, &collided).is_empty());
    }

    fn call_rows(
        node: Node,
        arity: usize,
        provenance: CallArgProvenance,
    ) -> Vec<(Node, usize, RTLReg, CallArgProvenance)> {
        (0..arity)
            .map(|position| (node, position, 0x1000 + node + position as u64, provenance))
            .collect()
    }

    fn identity(address: Address, name: &str) -> ExactLoaderIdentity {
        let _ = name;
        ExactLoaderIdentity {
            address,
            kind: LoaderSymbolKind::Function,
        }
    }

    #[test]
    fn shared_zero_argument_sites_establish_void() {
        let callee = identity(0x1000, "zero");
        let sites = vec![(1, callee.clone()), (2, callee.clone())];
        let inferred =
            infer_shared_fixed_arities(&sites, &[], &HashSet::new(), &HashSet::new());
        assert_eq!(inferred.get(&callee), Some(&0));
    }

    #[test]
    fn anchored_site_can_corroborate_forwarding_thunk() {
        let callee = identity(0x2000, "thunked");
        let sites = vec![(1, callee.clone()), (2, callee.clone())];
        let mut rows = call_rows(1, 2, CallArgProvenance::ForwardedEntry);
        rows.extend(call_rows(2, 2, CallArgProvenance::ExplicitRegister));
        assert_eq!(
            infer_shared_fixed_arities(&sites, &rows, &HashSet::new(), &HashSet::new())
                .get(&callee),
            Some(&2)
        );

        let forwarded_only: Vec<_> = call_rows(1, 2, CallArgProvenance::ForwardedEntry)
            .into_iter()
            .chain(call_rows(2, 2, CallArgProvenance::ForwardedEntry))
            .collect();
        assert!(infer_shared_fixed_arities(
            &sites,
            &forwarded_only,
            &HashSet::new(),
            &HashSet::new(),
        )
            .get(&callee)
            .is_none());
    }

    #[test]
    fn shared_arities_zero_through_eight_require_real_positions() {
        for arity in 0..=8 {
            let name = format!("arity_{arity}");
            let callee = identity(0x3000 + arity as u64, &name);
            let sites = vec![(10, callee.clone()), (20, callee.clone())];
            let mut rows = Vec::new();
            for node in [10, 20] {
                rows.extend((0..arity).map(|position| {
                    let source = if position < 4 {
                        CallArgProvenance::ExplicitRegister
                    } else {
                        CallArgProvenance::EntrySpStore
                    };
                    (node, position, node + position as u64, source)
                }));
            }
            assert_eq!(
                infer_shared_fixed_arities(
                    &sites,
                    &rows,
                    &HashSet::new(),
                    &HashSet::new(),
                )
                .get(&callee),
                Some(&arity),
                "arity {arity}"
            );
        }
    }

    #[test]
    fn stale_high_registers_and_ambiguous_stack_stores_fail_closed() {
        let callee = identity(0x4000, "stale");
        let sites = vec![(1, callee.clone()), (2, callee)];
        let stale = vec![
            (1, 2, 12, CallArgProvenance::ForwardedEntry),
            (1, 3, 13, CallArgProvenance::ForwardedEntry),
            (2, 2, 22, CallArgProvenance::ForwardedEntry),
            (2, 3, 23, CallArgProvenance::ForwardedEntry),
        ];
        assert!(infer_shared_fixed_arities(
            &sites,
            &stale,
            &HashSet::new(),
            &HashSet::new(),
        )
        .is_empty());

        let mut ambiguous = call_rows(1, 5, CallArgProvenance::ExplicitRegister);
        ambiguous.push((1, 4, 0xdead, CallArgProvenance::EntrySpStore));
        ambiguous.extend(call_rows(2, 5, CallArgProvenance::ExplicitRegister));
        assert!(infer_shared_fixed_arities(
            &sites,
            &ambiguous,
            &HashSet::new(),
            &HashSet::new(),
        )
        .is_empty());
    }

    #[test]
    fn variadic_and_conflicting_cohorts_remain_unspecified_order_independently() {
        let conflict = identity(0x5000, "conflict");
        let mut sites = vec![(1, conflict.clone()), (2, conflict.clone())];
        let mut rows = call_rows(1, 1, CallArgProvenance::ExplicitRegister);
        rows.extend(call_rows(2, 2, CallArgProvenance::ExplicitRegister));
        let expected =
            infer_shared_fixed_arities(&sites, &rows, &HashSet::new(), &HashSet::new());
        assert!(!expected.contains_key(&conflict));

        let mut rng = SmallRng::seed_from_u64(0x5eed_c011_ec7);
        for _ in 0..128 {
            sites.shuffle(&mut rng);
            rows.shuffle(&mut rng);
            assert_eq!(
                infer_shared_fixed_arities(
                    &sites,
                    &rows,
                    &HashSet::new(),
                    &HashSet::new(),
                ),
                expected
            );
        }

        let varfun = identity(0x6000, "varfun");
        let variadic_sites = vec![(3, varfun.clone()), (4, varfun.clone())];
        let variadic_rows: Vec<_> = call_rows(3, 1, CallArgProvenance::ExplicitRegister)
            .into_iter()
            .chain(call_rows(4, 1, CallArgProvenance::ExplicitRegister))
            .collect();
        assert!(infer_shared_fixed_arities(
            &variadic_sites,
            &variadic_rows,
            &HashSet::from([varfun]),
            &HashSet::new(),
        )
        .is_empty());
    }

    #[test]
    fn veto_and_identity_ambiguity_are_never_dropped_from_the_cohort() {
        let first = identity(0x7000, "same_display");
        let second = identity(0x8000, "same_display");
        let sites = vec![
            (1, first.clone()),
            (2, first.clone()),
            (2, second.clone()),
            (3, second.clone()),
        ];
        let rows: Vec<_> = [1u64, 2, 3]
            .into_iter()
            .flat_map(|node| call_rows(node, 1, CallArgProvenance::ExplicitRegister))
            .collect();
        let inferred = infer_shared_fixed_arities(
            &sites,
            &rows,
            &HashSet::new(),
            &HashSet::from([2u64]),
        );
        assert!(inferred.is_empty());
    }

    #[test]
    fn shared_arity_follows_exact_loader_identity_and_rejects_conflict() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        db.rel_push(
            "loader_symbol_identity",
            (
                0x5000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_target" as Symbol,
                "__imp_target" as Symbol,
            ),
        );
        for node in [1u64, 2] {
            db.rel_push(
                "call_loader_identity_candidate",
                (
                    node,
                    0x5000u64,
                    LoaderSymbolKind::ImportPointer,
                    "coff_ext_target" as Symbol,
                    "__imp_target" as Symbol,
                ),
            );
            db.rel_push(
                "call_arg_provenance",
                (
                    node,
                    0usize,
                    0x100 + node,
                    CallArgProvenance::ExplicitRegister,
                ),
            );
            db.rel_push(
                "call_args_collected_candidate",
                (node, Arc::new(vec![0x100 + node])),
            );
        }

        let inferred = infer_shared_fixed_arities_from_db(&db);
        assert_eq!(inferred.get("coff_ext_target"), Some(&1));
        assert_eq!(inferred.get("__imp_target"), Some(&1));
        let import_only = infer_shared_import_pointer_fixed_arities_from_db(&db);
        assert_eq!(import_only.get("coff_ext_target"), Some(&1));
        assert_eq!(import_only.get("__imp_target"), Some(&1));

        for node in [3u64, 4] {
            db.rel_push(
                "call_loader_identity_candidate",
                (
                    node,
                    0x5000u64,
                    LoaderSymbolKind::ImportPointer,
                    "coff_ext_target" as Symbol,
                    "__imp_target" as Symbol,
                ),
            );
            for position in 0..2usize {
                db.rel_push(
                    "call_arg_provenance",
                    (
                        node,
                        position,
                        0x200 + node + position as u64,
                        CallArgProvenance::ExplicitRegister,
                    ),
                );
            }
            db.rel_push(
                "call_args_collected_candidate",
                (
                    node,
                    Arc::new(vec![0x200 + node, 0x200 + node + 1]),
                ),
            );
        }
        let conflicting = infer_shared_fixed_arities_from_db(&db);
        assert!(!conflicting.contains_key("coff_ext_target"));
        assert!(!conflicting.contains_key("__imp_target"));
    }

    #[test]
    fn unresolved_forwarding_sites_are_not_reclassified_as_void() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        db.rel_push(
            "loader_symbol_identity",
            (
                0x6000u64,
                LoaderSymbolKind::Function,
                "forwarded_only" as Symbol,
                "forwarded_only" as Symbol,
            ),
        );
        for node in [10u64, 20] {
            db.rel_push(
                "call_loader_identity_candidate",
                (
                    node,
                    0x6000u64,
                    LoaderSymbolKind::Function,
                    "forwarded_only" as Symbol,
                    "forwarded_only" as Symbol,
                ),
            );
        }
        assert!(!infer_shared_fixed_arities_from_db(&db)
            .contains_key("forwarded_only"));
        for node in [10u64, 20] {
            db.rel_push(
                "call_args_collected_candidate",
                (node, Arc::new(Vec::<RTLReg>::new())),
            );
        }
        assert_eq!(
            infer_shared_fixed_arities_from_db(&db).get("forwarded_only"),
            Some(&0)
        );

        db.rel_push("call_forwarding_unresolved", (10u64,));
        assert!(!infer_shared_fixed_arities_from_db(&db)
            .contains_key("forwarded_only"));
    }

    #[test]
    fn address_kind_and_sanitizer_collisions_veto_projection() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        for (address, kind, provider, original) in [
            (
                0x7000u64,
                LoaderSymbolKind::Function,
                "same.name" as Symbol,
                "first" as Symbol,
            ),
            (
                0x7000u64,
                LoaderSymbolKind::ImportPointer,
                "same_name" as Symbol,
                "second" as Symbol,
            ),
            (
                0x8000u64,
                LoaderSymbolKind::Function,
                "same_name" as Symbol,
                "third" as Symbol,
            ),
        ] {
            db.rel_push(
                "loader_symbol_identity",
                (address, kind, provider, original),
            );
            db.rel_push(
                "known_loader_signature",
                (
                    address,
                    kind,
                    provider,
                    original,
                    1usize,
                    XType::Xint,
                    Arc::new(vec![XType::Xany64]),
                    false,
                ),
            );
            for node in [address + 1, address + 2] {
                db.rel_push(
                    "call_loader_identity_candidate",
                    (node, address, kind, provider, original),
                );
                db.rel_push(
                    "call_arg_provenance",
                    (
                        node,
                        0usize,
                        node + 0x100,
                        CallArgProvenance::ExplicitRegister,
                    ),
                );
                db.rel_push(
                    "call_args_collected_candidate",
                    (node, Arc::new(vec![node + 0x100])),
                );
            }
        }

        assert!(!infer_shared_fixed_arities_from_db(&db).contains_key("same_name"));
        assert!(!known_loader_signatures_from_db(&db).contains_key("same_name"));
    }

    #[test]
    fn ordinary_symbol_sanitizer_collisions_veto_all_loader_projections() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        for (address, kind, provider, first_node) in [
            (
                0xb000u64,
                LoaderSymbolKind::Function,
                "punctuated.name" as Symbol,
                101u64,
            ),
            (
                0xd000u64,
                LoaderSymbolKind::ImportPointer,
                "coff.ptr" as Symbol,
                201u64,
            ),
        ] {
            db.rel_push(
                "loader_symbol_identity",
                (address, kind, provider, provider),
            );
            db.rel_push(
                "known_loader_signature",
                (
                    address,
                    kind,
                    provider,
                    provider,
                    1usize,
                    XType::Xint,
                    Arc::new(vec![XType::Xany64]),
                    false,
                ),
            );
            for node in [first_node, first_node + 1] {
                db.rel_push(
                    "call_loader_identity_candidate",
                    (node, address, kind, provider, provider),
                );
                db.rel_push(
                    "call_arg_provenance",
                    (
                        node,
                        0usize,
                        node + 0x1000,
                        CallArgProvenance::ExplicitRegister,
                    ),
                );
                db.rel_push(
                    "call_args_collected_candidate",
                    (node, Arc::new(vec![node + 0x1000])),
                );
            }
        }

        // Neither ordinary relation carries loader kind.  A foreign address
        // with the same emitted spelling is therefore an ambiguity, not an
        // alias of the callable identity.
        db.rel_push(
            "ident_to_symbol",
            (0xc000usize as Ident, "punctuated_name" as Symbol),
        );
        db.rel_push(
            "symbols",
            (0xe000u64, "coff_ptr" as Symbol, "Beg" as Symbol),
        );

        let signatures = known_loader_signatures_from_db(&db);
        let arities = infer_shared_fixed_arities_from_db(&db);
        for name in ["punctuated_name", "coff_ptr"] {
            assert!(!signatures.contains_key(name), "signature leaked for {name}");
            assert!(!arities.contains_key(name), "arity leaked for {name}");
            let raw = HashMap::from([(name.to_string(), vec![vec![i32_()]])]);
            assert!(
                join_call_site_evidence_by_loader_identity(&db, &raw).is_empty(),
                "argument evidence leaked for {name}"
            );
        }
    }

    #[test]
    fn aliases_of_one_address_and_kind_share_the_exact_identity() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        for (provider, original) in [
            ("alias_a" as Symbol, "raw_a" as Symbol),
            ("alias_b" as Symbol, "raw_b" as Symbol),
        ] {
            db.rel_push(
                "loader_symbol_identity",
                (
                    0x9000u64,
                    LoaderSymbolKind::Function,
                    provider,
                    original,
                ),
            );
        }
        for (node, provider, original) in [
            (1u64, "alias_a" as Symbol, "raw_a" as Symbol),
            (2u64, "alias_b" as Symbol, "raw_b" as Symbol),
        ] {
            db.rel_push(
                "call_loader_identity_candidate",
                (
                    node,
                    0x9000u64,
                    LoaderSymbolKind::Function,
                    provider,
                    original,
                ),
            );
            db.rel_push(
                "call_arg_provenance",
                (
                    node,
                    0usize,
                    0xa000 + node,
                    CallArgProvenance::ExplicitRegister,
                ),
            );
            db.rel_push(
                "call_args_collected_candidate",
                (node, Arc::new(vec![0xa000 + node])),
            );
        }

        let inferred = infer_shared_fixed_arities_from_db(&db);
        for name in ["alias_a", "raw_a", "alias_b", "raw_b"] {
            assert_eq!(inferred.get(name), Some(&1));
        }
        assert!(infer_shared_import_pointer_fixed_arities_from_db(&db).is_empty());
    }

    #[test]
    fn exact_known_signature_projection_vetoes_stale_fixed_conflicts() {
        let mut valid = DecompileDB::default();
        valid.rel_push(
            "loader_symbol_identity",
            (
                0xa000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_printf" as Symbol,
                "__imp_printf" as Symbol,
            ),
        );
        valid.rel_push(
            "known_loader_signature",
            (
                0xa000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_printf" as Symbol,
                "__imp_printf" as Symbol,
                1usize,
                XType::Xint,
                Arc::new(vec![XType::Xcharptr]),
                true,
            ),
        );
        valid.rel_push(
            "known_loader_variadic",
            (
                0xa000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_printf" as Symbol,
                "__imp_printf" as Symbol,
            ),
        );
        let projected = known_loader_signatures_from_db(&valid);
        assert_eq!(
            projected.get("coff_ext_printf"),
            Some(&(
                XType::Xint,
                Arc::new(vec![XType::Xcharptr]),
                true,
            ))
        );

        valid.rel_push(
            "loader_signature_conflict",
            (
                0xa000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_printf" as Symbol,
                "__imp_printf" as Symbol,
            ),
        );
        assert!(!known_loader_signatures_from_db(&valid).contains_key("coff_ext_printf"));

        let mut stale = DecompileDB::default();
        stale.rel_push(
            "loader_symbol_identity",
            (
                0xb000u64,
                LoaderSymbolKind::Function,
                "stale" as Symbol,
                "stale" as Symbol,
            ),
        );
        stale.rel_push(
            "known_loader_signature",
            (
                0xb000u64,
                LoaderSymbolKind::Function,
                "stale" as Symbol,
                "stale" as Symbol,
                1usize,
                XType::Xint,
                Arc::new(vec![XType::Xany64]),
                false,
            ),
        );
        stale.rel_push(
            "known_loader_variadic",
            (
                0xb000u64,
                LoaderSymbolKind::Function,
                "stale" as Symbol,
                "stale" as Symbol,
            ),
        );
        assert!(!known_loader_signatures_from_db(&stale).contains_key("stale"));

        let mut plain_object = DecompileDB::default();
        plain_object.rel_push(
            "known_extern_signature",
            (
                "memcpy" as Symbol,
                3usize,
                XType::Xptr,
                Arc::new(vec![XType::Xptr, XType::Xptr, XType::Xany64]),
            ),
        );
        assert!(known_loader_signatures_from_db(&plain_object).contains_key("memcpy"));
        assert!(known_import_pointer_signatures_from_db(&plain_object).is_empty());
    }

    #[test]
    fn exact_import_pointer_call_shadows_same_address_stale_function_projection() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        let address = 0xc000u64;
        let node = 0xc010u64;
        let provider = "coff_ext_callback" as Symbol;
        let original = "__imp_callback" as Symbol;
        for kind in [
            LoaderSymbolKind::ImportPointer,
            LoaderSymbolKind::Function,
        ] {
            db.rel_push(
                "loader_symbol_identity",
                (address, kind, provider, original),
            );
        }
        db.rel_push(
            "call_loader_identity",
            (
                node,
                address,
                LoaderSymbolKind::ImportPointer,
                provider,
                original,
            ),
        );
        db.rel_push(
            "known_loader_signature",
            (
                address,
                LoaderSymbolKind::ImportPointer,
                provider,
                original,
                1usize,
                XType::Xvoid,
                Arc::new(vec![XType::Xany64]),
                false,
            ),
        );
        db.rel_push(
            "known_loader_signature",
            (
                address,
                LoaderSymbolKind::Function,
                provider,
                original,
                0usize,
                XType::Xint,
                Arc::new(Vec::<XType>::new()),
                false,
            ),
        );

        let all = known_loader_signatures_from_db(&db);
        let imports = known_import_pointer_signatures_from_db(&db);
        let expected = (
            XType::Xvoid,
            Arc::new(vec![XType::Xany64]),
            false,
        );
        for name in [provider, original] {
            let name = sanitize_c_symbol_name(name);
            assert_eq!(all.get(&name), Some(&expected));
            assert_eq!(imports.get(&name), Some(&expected));
        }

        // Even if stale metadata also publishes a direct extern signature, C
        // emission must retain the IAT object declaration and must not add a
        // same-name function declaration beside it.
        db.rel_push(
            "resolved_extern_signature",
            (
                original,
                1usize,
                XType::Xvoid,
                Arc::new(vec![XType::Xany64]),
            ),
        );
        let global = GlobalData {
            id: address as Ident,
            name: original.to_string(),
            is_string: false,
            content: Vec::new(),
            is_pointer: true,
            scalar_value: None,
            scalar_writable: false,
            pointer_init: Vec::new(),
        };
        let tu = build_translation_unit_from_stmt_map_with_types(
            &db,
            &[],
            &[global],
            &HashMap::from([(address as Ident, original.to_string())]),
            &HashMap::new(),
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        assert!(tu.decls.iter().any(|decl| {
            matches!(
                decl,
                TopLevelDecl::VarDecl(var) if var.name == sanitize_c_symbol_name(original)
            )
        }));
        assert!(!tu.decls.iter().any(|decl| {
            matches!(
                decl,
                TopLevelDecl::FuncDecl(func) if func.name == sanitize_c_symbol_name(original)
            )
        }));
    }

    #[test]
    fn headerless_compiler_provided_call_gets_a_data_driven_declaration() {
        let hdb = crate::decompile::passes::c_pass::header_db::header_db();
        let mut headerless: Vec<&'static str> = hdb
            .functions
            .iter()
            .copied()
            .filter(|name| !function_declaration_is_externally_provided(name))
            .collect();
        headerless.sort_unstable();
        let name = *headerless
            .first()
            .expect("header database must exercise a header:null function");

        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        let params = Arc::new(vec![XType::Xany64]);
        db.rel_push(
            "known_extern_signature",
            (name, 1usize, XType::Xint, params.clone()),
        );
        // This relation is produced only for a resolved external call, so the
        // fixture models a called header:null entry without naming one.
        db.rel_push(
            "resolved_extern_signature",
            (name, 1usize, XType::Xint, params),
        );

        let mut tu = build_translation_unit_from_stmt_map_with_types(
            &db,
            &[],
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &[],
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
        );
        tu.add_function(FuncDef {
            name: "headerless_caller".to_string(),
            return_type: CType::Void,
            params: vec![],
            is_variadic: false,
            storage_class: StorageClass::Auto,
            body: CStmt::Expr(CExpr::Call(
                Box::new(CExpr::Var(name.to_string())),
                vec![CExpr::int(1)],
            )),
            local_vars: vec![],
            loc: SourceLoc::unknown(),
        });
        assert!(tu.decls.iter().any(|decl| {
            matches!(
                decl,
                TopLevelDecl::FuncDecl(func)
                    if func.name == name
                        && func.params.len() == 1
                        && !func.unspecified_params
            )
        }));
        let output = crate::decompile::passes::c_pass::print::print_translation_unit(&tu);
        assert!(output.contains(&format!("int {name}(long arg0);")));
        assert!(!output.contains("#include"));
    }
}

#[cfg(test)]
mod cast_insertion_tests {
    use super::*;

    fn one_field(field: &str, ty: CType) -> StructFieldTypes {
        HashMap::from([(
            "struct_1".to_string(),
            HashMap::from([(field.to_string(), ty)]),
        )])
    }

    #[test]
    fn member_lvalue_uses_recovered_integer_field_type() {
        let pointer = CType::ptr(CType::Void);
        let variables = HashMap::from([
            ("value".to_string(), CType::Struct("struct_1".to_string())),
            ("p0".to_string(), pointer),
        ]);
        let struct_fields = one_field("ofs_8", CType::long());
        let types = CastTypes {
            variables: &variables,
            struct_fields: &struct_fields,
        };
        let expr = CExpr::Assign(
            AssignOp::Assign,
            Box::new(CExpr::Member(
                Box::new(CExpr::Var("value".to_string())),
                "ofs_8".to_string(),
            )),
            Box::new(CExpr::Var("p0".to_string())),
        );

        let actual = insert_casts_expr(&expr, &types, &HashMap::new(), &HashMap::new());

        let CExpr::Assign(_, _, rhs) = actual else {
            panic!("assignment was not preserved");
        };
        assert_eq!(
            *rhs,
            CExpr::Cast(CType::long(), Box::new(CExpr::Var("p0".to_string())))
        );
    }

    #[test]
    fn member_ptr_lvalue_uses_recovered_pointer_field_type() {
        let pointer = CType::ptr(CType::Void);
        let struct_pointer = CType::ptr(CType::Struct("struct_1".to_string()));
        let variables = HashMap::from([
            ("var_0".to_string(), CType::long()),
            ("p0".to_string(), CType::long()),
        ]);
        let struct_fields = one_field("ofs_8", pointer.clone());
        let types = CastTypes {
            variables: &variables,
            struct_fields: &struct_fields,
        };
        let expr = CExpr::Assign(
            AssignOp::Assign,
            Box::new(CExpr::MemberPtr(
                Box::new(CExpr::Cast(
                    struct_pointer,
                    Box::new(CExpr::Var("var_0".to_string())),
                )),
                "ofs_8".to_string(),
            )),
            Box::new(CExpr::Var("p0".to_string())),
        );

        let actual = insert_casts_expr(&expr, &types, &HashMap::new(), &HashMap::new());

        let CExpr::Assign(_, _, rhs) = actual else {
            panic!("assignment was not preserved");
        };
        assert_eq!(
            *rhs,
            CExpr::Cast(pointer, Box::new(CExpr::Var("p0".to_string())))
        );
    }

    #[test]
    fn address_of_uses_final_declared_type_for_pointer_cast() {
        let declared = CType::ptr(CType::Struct("struct_6".to_string()));
        let variables = HashMap::from([("var_0".to_string(), declared.clone())]);
        let struct_fields = HashMap::new();
        let types = CastTypes {
            variables: &variables,
            struct_fields: &struct_fields,
        };
        let address_of = CExpr::Unary(UnaryOp::AddrOf, Box::new(CExpr::Var("var_0".to_string())));
        let expr = CExpr::Assign(
            AssignOp::Assign,
            Box::new(CExpr::Var("var_0".to_string())),
            Box::new(address_of.clone()),
        );

        let actual = insert_casts_expr(&expr, &types, &HashMap::new(), &HashMap::new());

        let CExpr::Assign(_, _, rhs) = actual else {
            panic!("assignment was not preserved");
        };
        assert_eq!(*rhs, CExpr::Cast(declared, Box::new(address_of)));
    }
}
