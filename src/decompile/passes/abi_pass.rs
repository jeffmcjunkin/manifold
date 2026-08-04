// ABI pass: populates register conventions, known function signatures, and noreturn functions into DecompileDB.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::pass::IRPass;
use crate::x86::types::{Address, LoaderSymbolKind, Symbol, XType};

pub struct AbiPass;

impl IRPass for AbiPass {
    fn name(&self) -> &'static str { "abi" }

    fn run(&self, db: &mut DecompileDB) {
        populate_arg_ret_regs(db);
        populate_known_noreturn(db);
        populate_known_func_param_is_ptr(db);
        populate_known_func_returns(db);
        populate_hardcoded_signatures(db);
        populate_known_global_types(db);
        populate_known_func_param_pointee_size(db);
        bind_loader_symbol_identities(db);
    }

    fn outputs(&self) -> &'static [&'static str] {
        &[
            "is_arg_reg", "is_xmm_arg_reg", "is_float_arg_reg",
            "is_caller_saved", "is_callee_saved",
            "abi_int_arg_position", "abi_float_arg_position",
            "abi_shared_arg_slots", "abi_first_stack_arg_position",
            "abi_outgoing_stack_base", "abi_incoming_sp_stack_base",
            "abi_incoming_bp_stack_base", "abi_stack_slot_size",
            "is_known_noreturn_function",
            "known_func_param_is_ptr", "known_func_returns_ptr",
            "known_func_returns_long",
            "known_extern_signature", "known_global_type",
            "known_varargs_function", "known_func_param_pointee_size",
            "known_loader_signature", "known_loader_variadic",
            "loader_signature_conflict",
        ]
    }

    fn extra_reads(&self) -> &'static [&'static str] {
        &["loader_symbol_identity"]
    }
}

/// Spellings implied by the COFF symbol decoration grammar, in preference
/// order.  These are format identities, not guesses based on a function-name
/// prefix: import-pointer decoration, one C external-name underscore, and a
/// numeric stdcall byte-count suffix are the only transformations accepted.
fn coff_identity_candidates(
    original: &str,
    is_coff_family: bool,
    has_legacy_x86_decoration: bool,
) -> Vec<String> {
    fn push_unique(out: &mut Vec<String>, value: &str) {
        if !value.is_empty() && !out.iter().any(|existing| existing == value) {
            out.push(value.to_string());
        }
    }

    fn without_stdcall_suffix(value: &str) -> Option<&str> {
        let (base, bytes) = value.rsplit_once('@')?;
        (!base.is_empty() && !bytes.is_empty() && bytes.bytes().all(|b| b.is_ascii_digit()))
            .then_some(base)
    }

    let mut out = Vec::new();
    push_unique(&mut out, original);

    if !is_coff_family {
        return out;
    }

    // Both spellings occur in Microsoft/LLVM COFF symbol tables.  The first
    // leaves any ordinary C leading underscore for the next grammar step.
    for prefix in ["__imp_", "_imp__"] {
        if let Some(base) = original.strip_prefix(prefix) {
            push_unique(&mut out, base);
        }
    }

    if !has_legacy_x86_decoration {
        return out;
    }

    // Legacy i386 COFF additionally decorates stdcall byte counts and ordinary
    // C externals.  A bounded loop is enough because each successful step
    // shortens a name.
    let mut cursor = 0usize;
    while cursor < out.len() {
        let value = out[cursor].clone();
        if let Some(base) = without_stdcall_suffix(&value) {
            push_unique(&mut out, base);
        }
        if value.starts_with('_') && !value.starts_with("__") {
            push_unique(&mut out, &value[1..]);
        }
        cursor += 1;
    }
    out
}

/// Resolve curated ABI facts independently for every exact loader identity.
/// Provider-name aliases are only a compatibility projection and are emitted
/// when that provider denotes exactly one `(address, kind)` object.
/// All arity-sensitive consumers use `known_loader_signature` directly.
fn bind_loader_symbol_identities(db: &mut DecompileDB) {
    let (is_coff_family, has_legacy_x86_decoration) = db
        .target_abi
        .as_ref()
        .map(|abi| {
            (
                matches!(
                    abi.format,
                    crate::abi::BinaryFormat::Coff | crate::abi::BinaryFormat::Pe
                ),
                abi.arch == crate::abi::Arch::X86_32,
            )
        })
        .unwrap_or((false, false));
    let mut identity_names: BTreeMap<
        (Address, LoaderSymbolKind),
        BTreeSet<(Symbol, Symbol)>,
    > = BTreeMap::new();
    for &(address, kind, provider, original) in db.rel_iter::<(
        Address,
        LoaderSymbolKind,
        Symbol,
        Symbol,
    )>("loader_symbol_identity") {
        identity_names
            .entry((address, kind))
            .or_default()
            .insert((provider, original));
    }
    if identity_names.is_empty() {
        return;
    }

    let signatures: Vec<(Symbol, usize, XType, Arc<Vec<XType>>)> = db
        .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>("known_extern_signature")
        .cloned()
        .collect();
    let signatures_by_name: HashMap<&str, Vec<(usize, XType, Arc<Vec<XType>>)>> = {
        let mut out: HashMap<&str, Vec<(usize, XType, Arc<Vec<XType>>)>> = HashMap::new();
        for (name, arity, ret, params) in &signatures {
            out.entry(*name)
                .or_default()
                .push((*arity, *ret, params.clone()));
        }
        out
    };

    let varargs_by_name: HashMap<Symbol, BTreeSet<usize>> = {
        let mut out: HashMap<Symbol, BTreeSet<usize>> = HashMap::new();
        for &(name, count) in db.rel_iter::<(Symbol, usize)>("known_varargs_function") {
            out.entry(name).or_default().insert(count);
        }
        out
    };
    let ptr_params: Vec<(Symbol, usize)> = db
        .rel_iter::<(Symbol, usize)>("known_func_param_is_ptr")
        .cloned()
        .collect();
    let pointee_sizes: Vec<(Symbol, usize, usize)> = db
        .rel_iter::<(Symbol, usize, usize)>("known_func_param_pointee_size")
        .cloned()
        .collect();
    let returns_ptr: HashSet<Symbol> = db
        .rel_iter::<(Symbol,)>("known_func_returns_ptr")
        .map(|(name,)| *name)
        .collect();
    let returns_long: HashSet<Symbol> = db
        .rel_iter::<(Symbol,)>("known_func_returns_long")
        .map(|(name,)| *name)
        .collect();
    let noreturn: HashSet<Symbol> = db
        .rel_iter::<(Symbol,)>("is_known_noreturn_function")
        .map(|(name,)| *name)
        .collect();

    #[derive(Clone)]
    struct IdentityBinding {
        address: Address,
        kind: LoaderSymbolKind,
        names: BTreeSet<(Symbol, Symbol)>,
        signature: Option<(usize, XType, Arc<Vec<XType>>, bool)>,
        vararg_count: Option<usize>,
        variadic: bool,
        signature_conflict: bool,
        pointer_positions: BTreeSet<usize>,
        pointee_sizes: BTreeSet<(usize, usize)>,
        returns_ptr: bool,
        returns_long: bool,
        noreturn: bool,
    }

    let mut bindings = Vec::new();
    for ((address, kind), names) in identity_names {
        let candidates: BTreeSet<String> = names
            .iter()
            .flat_map(|(_, original)| {
                coff_identity_candidates(
                    original,
                    is_coff_family,
                    has_legacy_x86_decoration,
                )
            })
            .collect();
        let mut matched_signatures: BTreeSet<(usize, XType, Arc<Vec<XType>>)> = BTreeSet::new();
        for candidate in &candidates {
            if let Some(rows) = signatures_by_name.get(candidate.as_str()) {
                matched_signatures.extend(rows.iter().cloned());
            }
        }
        let candidate_names: HashSet<&str> = candidates.iter().map(String::as_str).collect();
        let vararg_counts: BTreeSet<usize> = candidates
            .iter()
            .filter_map(|candidate| varargs_by_name.get(candidate.as_str()))
            .flat_map(|counts| counts.iter().copied())
            .collect();
        let vararg_count = (vararg_counts.len() == 1)
            .then(|| *vararg_counts.iter().next().unwrap());
        let has_signature_fact = !matched_signatures.is_empty();
        let signature = if matched_signatures.len() == 1 && vararg_counts.len() <= 1 {
            let (arity, ret, params) = matched_signatures.into_iter().next().unwrap();
            (arity == params.len()
                && vararg_count.map_or(true, |fixed_count| fixed_count == arity))
                .then_some((arity, ret, params, vararg_count.is_some()))
        } else {
            None
        };
        let signature_conflict = (has_signature_fact && signature.is_none())
            || vararg_counts.len() > 1;
        let pointer_positions = ptr_params
            .iter()
            .filter_map(|(name, position)| {
                candidate_names.contains(name).then_some(*position)
            })
            .collect();
        let pointee_sizes = pointee_sizes
            .iter()
            .filter_map(|(name, position, size)| {
                candidate_names
                    .contains(name)
                    .then_some((*position, *size))
            })
            .collect();

        bindings.push(IdentityBinding {
            address,
            kind,
            names,
            signature,
            vararg_count,
            variadic: !vararg_counts.is_empty(),
            signature_conflict,
            pointer_positions,
            pointee_sizes,
            returns_ptr: returns_ptr
                .iter()
                .any(|name| candidate_names.contains(name)),
            returns_long: returns_long
                .iter()
                .any(|name| candidate_names.contains(name)),
            noreturn: noreturn.iter().any(|name| candidate_names.contains(name)),
        });
    }

    for binding in &bindings {
        for &(provider, original) in &binding.names {
            if binding.variadic {
                db.rel_push(
                    "known_loader_variadic",
                    (binding.address, binding.kind, provider, original),
                );
            }
            if binding.signature_conflict {
                db.rel_push(
                    "loader_signature_conflict",
                    (binding.address, binding.kind, provider, original),
                );
            }
            if let Some((arity, ret, params, variadic)) = &binding.signature {
                db.rel_push(
                    "known_loader_signature",
                    (
                        binding.address,
                        binding.kind,
                        provider,
                        original,
                        *arity,
                        *ret,
                        params.clone(),
                        *variadic,
                    ),
                );
            }
        }
    }

    // A spelling collision is not an identity relation.  Keep compatibility
    // name facts only for the singleton projection; exact consumers above can
    // still use both colliding objects by address and kind.
    let mut by_provider: BTreeMap<Symbol, BTreeSet<usize>> = BTreeMap::new();
    for (index, binding) in bindings.iter().enumerate() {
        for &(provider, _) in &binding.names {
            by_provider.entry(provider).or_default().insert(index);
        }
    }
    for (provider, provider_bindings) in by_provider {
        if provider_bindings.len() != 1 {
            continue;
        }
        let binding = &bindings[*provider_bindings.iter().next().unwrap()];
        if let Some((arity, ret, params, _)) = &binding.signature {
            db.rel_push(
                "known_extern_signature",
                (provider, *arity, *ret, params.clone()),
            );
        }
        if let Some(fixed_count) = binding.vararg_count {
            db.rel_push("known_varargs_function", (provider, fixed_count));
        }
        for &position in &binding.pointer_positions {
            db.rel_push("known_func_param_is_ptr", (provider, position));
        }
        let mut sizes_by_position: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for &(position, size) in &binding.pointee_sizes {
            sizes_by_position.entry(position).or_default().insert(size);
        }
        for (position, sizes) in sizes_by_position {
            if sizes.len() == 1 {
                db.rel_push(
                    "known_func_param_pointee_size",
                    (provider, position, *sizes.iter().next().unwrap()),
                );
            }
        }
        if binding.returns_ptr && !binding.returns_long {
            db.rel_push("known_func_returns_ptr", (provider,));
        } else if binding.returns_long && !binding.returns_ptr {
            db.rel_push("known_func_returns_long", (provider,));
        }
        if binding.noreturn {
            db.rel_push("is_known_noreturn_function", (provider,));
        }
    }
}


// Populate argument, return, and caller-saved register facts from the ABI config.
fn populate_arg_ret_regs(db: &mut DecompileDB) {
    let cfg = db.abi().clone();
    for (pos, reg) in cfg.int_arg_regs.iter().enumerate() {
        db.rel_push("is_arg_reg", (*reg,));
        db.rel_push("abi_int_arg_position", (*reg, pos));
    }
    for (pos, reg) in cfg.float_arg_regs.iter().enumerate() {
        db.rel_push("is_xmm_arg_reg", (*reg,));
        db.rel_push("is_float_arg_reg", (*reg,));
        db.rel_push("abi_float_arg_position", (*reg, pos));
    }
    for reg in &cfg.caller_saved {
        db.rel_push("is_caller_saved", (*reg,));
    }
    for reg in &cfg.callee_saved {
        db.rel_push("is_callee_saved", (*reg,));
    }
    if cfg.uses_shared_arg_slots() {
        db.rel_push("abi_shared_arg_slots", (true,));
    }
    db.rel_push("abi_first_stack_arg_position", (cfg.first_stack_arg_position(),));
    db.rel_push("abi_outgoing_stack_base", (cfg.outgoing_stack_arg_base(),));
    db.rel_push("abi_incoming_sp_stack_base", (cfg.incoming_sp_stack_arg_base(),));
    db.rel_push("abi_incoming_bp_stack_base", (cfg.incoming_bp_stack_arg_base(),));
    db.rel_push("abi_stack_slot_size", (cfg.pointer_size as i64,));
}


// Register the always-noreturn functions from abi's single ALWAYS_NORETURN_FUNCS list; status-dependent error-family functions are decided at the call site instead.
fn populate_known_noreturn(db: &mut DecompileDB) {
    for name in crate::abi::ALWAYS_NORETURN_FUNCS {
        db.rel_push("is_known_noreturn_function", (*name as Symbol,));
    }
}


// Register known pointer parameters for libc/coreutils/gnulib functions.
fn populate_known_func_param_is_ptr(db: &mut DecompileDB) {
    let entries: &[(Symbol, usize)] = &[
        ("strlen", 0),
        ("strcpy", 0), ("strcpy", 1),
        ("strcat", 0), ("strcat", 1),
        ("memset", 0),
        ("memcpy", 0), ("memcpy", 1),
        ("memmove", 0), ("memmove", 1),
        ("memchr", 0),
        ("memcmp", 0), ("memcmp", 1),
        ("strdup", 0), ("strndup", 0),
        ("strstr", 0), ("strstr", 1),
        ("strtol", 0), ("strtol", 1),
        ("strtoul", 0), ("strtoul", 1),
        // ABI-1: strerror(int) and malloc(size_t) take a SCALAR param 0; forcing it to pointer contradicted the signatures below and wrongly drove must_be_ptr/is_ptr/Xptr.
        ("free", 0),
        ("realloc", 0),
        ("reallocarray", 0),
        ("printf", 0),
        ("fprintf", 0), ("fprintf", 1),
        ("snprintf", 0), ("snprintf", 2),
        ("sprintf", 0), ("sprintf", 1),
        ("sscanf", 0), ("sscanf", 1),
        ("fopen", 0), ("fopen", 1),
        ("fclose", 0),
        ("fflush", 0),
        ("ferror", 0), ("feof", 0), ("fileno", 0),
        ("puts", 0),
        ("fputs", 0), ("fputs", 1),
        ("fputc", 1),
        ("fgets", 0), ("fgets", 2),
        ("fread", 0), ("fread", 3),
        ("fwrite", 0), ("fwrite", 3),
        ("fseek", 0),
        ("ftell", 0),
        ("__sprintf_chk", 0), ("__sprintf_chk", 3),
        ("__snprintf_chk", 0), ("__snprintf_chk", 4),
        ("__fprintf_chk", 0), ("__fprintf_chk", 2),
        ("__printf_chk", 1),
        ("__memcpy_chk", 0), ("__memcpy_chk", 1),
        ("__memmove_chk", 0), ("__memmove_chk", 1),
        ("__memset_chk", 0),
        ("__strcpy_chk", 0), ("__strcpy_chk", 1),
        ("__strcat_chk", 0), ("__strcat_chk", 1),
        ("strtok", 0), ("strtok", 1),
        ("getenv", 0), ("setenv", 0), ("setenv", 1),
        ("setlocale", 1),
        ("getopt_long", 1), ("getopt_long", 2), ("getopt_long", 3), ("getopt_long", 4),
        ("atexit", 0), ("signal", 1),
        ("opendir", 0), ("closedir", 0), ("readdir", 0),
        ("access", 0), ("chmod", 0), ("chown", 0),
        ("readlink", 0), ("readlink", 1),
        ("readlinkat", 1), ("readlinkat", 2),
        ("stat", 0), ("stat", 1), ("lstat", 0), ("lstat", 1),
        ("fstatat", 1), ("fstatat", 2),
        ("localtime", 0), ("localtime_r", 0), ("localtime_r", 1),
        ("localtime_rz", 0), ("localtime_rz", 1), ("localtime_rz", 2),
        ("strftime", 0), ("strftime", 2), ("strftime", 3),
        ("nstrftime", 0), ("nstrftime", 2), ("nstrftime", 3),
        ("time", 0),
        ("umaxtostr", 1),
        ("imaxtostr", 1),
        ("offtostr", 1),
        ("uinttostr", 1),
        ("timetostr", 1),
        ("human_readable", 1),
        ("quotearg_style", 1),
        ("quotearg_buffer", 0), ("quotearg_buffer", 2),
        ("quote", 0),
        // ABI-1: getpwuid(uid_t) and getgrgid(gid_t) take a SCALAR ID param 0; forcing it to pointer wrongly drove must_be_ptr/is_ptr/Xptr.
        ("error", 2),
        ("__errno_location", 0),
        ("dcgettext", 0), ("dcgettext", 1),
    ];
    for &(name, idx) in entries {
        db.rel_push("known_func_param_is_ptr", (name, idx));
    }
}


// Register known return types (ptr, long, int) for libc/coreutils functions.
fn populate_known_func_returns(db: &mut DecompileDB) {
    let returns_ptr: &[Symbol] = &[
        "malloc", "calloc", "realloc", "reallocarray",
        "strdup", "strndup", "strcpy", "strcat",
        "memcpy", "memset", "memmove", "memchr",
        "fopen", "getenv",
        "strchr", "strrchr", "strstr", "strtok",
        "realpath", "getcwd",
        "readdir", "opendir",
        "strerror", "setlocale",
        "localtime", "localtime_r",
        "getpwuid", "getgrgid",
        "__errno_location",
        "signal", "dcgettext",
        "fdopen", "freopen", "fdopendir",
        "mempcpy", "rawmemchr", "memrchr",
        "stpcpy", "textdomain", "bindtextdomain",
        "nl_langinfo", "dcngettext",
        "localeconv", "newlocale",
        "__ctype_b_loc", "__ctype_toupper_loc", "__ctype_tolower_loc",
        "canonicalize_file_name",
        "getpwnam", "getgrnam",
        "gmtime_r", "aligned_alloc",
        "xmalloc", "ximalloc", "xrealloc", "xirealloc",
        "xcalloc", "xicalloc", "xzalloc", "xizalloc",
        "xstrdup", "xstrndup",
        "xmemdup", "ximemdup", "ximemdup0",
        "xcharalloc",
        "xreallocarray", "xireallocarray",
        "xnmalloc", "xinmalloc",
        "xnrealloc",
        "x2realloc", "x2nrealloc",
        "xpalloc",
        "imalloc", "irealloc", "icalloc", "ireallocarray",
        "tzalloc",
        "umaxtostr", "imaxtostr", "offtostr",
        "uinttostr", "timetostr",
        "localtime_rz", "human_readable",
        "quotearg_style", "quote",
        // gnulib author-name helpers return const char *; without the returns-ptr fact the result defaults to int and gets read-cast (int), truncating the pointer.
        "proper_name", "proper_name_lite", "proper_name_utf8",
    ];
    for name in returns_ptr {
        db.rel_push("known_func_returns_ptr", (*name,));
    }

    let returns_long: &[Symbol] = &[
        "strlen", "ftell", "fread", "fwrite",
        "lseek", "read", "write",
        "readlink", "readlinkat",
        "strftime", "nstrftime",
        "quotearg_buffer",
        "__getdelim", "pathconf", "clock",
        "strtoimax",
    ];
    for name in returns_long {
        db.rel_push("known_func_returns_long", (*name,));
    }
}


// Register full extern signatures (param count, return type, param types) for known functions.
fn populate_hardcoded_signatures(db: &mut DecompileDB) {
    use XType::*;

    let signatures: &[(&str, usize, XType, &[XType])] = &[
        ("memset",   3, Xptr,   &[Xptr, Xint, Xany64]),
        ("memcpy",   3, Xptr,   &[Xptr, Xptr, Xany64]),
        ("memmove",  3, Xptr,   &[Xptr, Xptr, Xany64]),
        ("memcmp",   3, Xint,   &[Xptr, Xptr, Xany64]),
        ("strlen",   1, Xany64, &[Xcharptr]),
        ("strcpy",   2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("strncpy",  3, Xcharptr, &[Xcharptr, Xcharptr, Xany64]),
        ("strcmp",    2, Xint,   &[Xcharptr, Xcharptr]),
        ("strncmp",  3, Xint,   &[Xcharptr, Xcharptr, Xany64]),
        ("strchr",   2, Xcharptr, &[Xcharptr, Xint]),
        ("strrchr",  2, Xcharptr, &[Xcharptr, Xint]),
        ("strcat",   2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("strncat",  3, Xcharptr, &[Xcharptr, Xcharptr, Xany64]),
        ("strdup",   1, Xcharptr, &[Xcharptr]),
        ("strndup",  2, Xcharptr, &[Xcharptr, Xany64]),
        ("strstr",   2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("strtol",   3, Xlong,  &[Xcharptr, Xptr, Xint]),
        ("strtoul",  3, Xlongunsigned, &[Xcharptr, Xptr, Xint]),
        ("strerror", 1, Xcharptr, &[Xint]),
        ("printf",   1, Xint,   &[Xcharptr]),
        ("fprintf",  2, Xint,   &[Xptr, Xcharptr]),
        ("snprintf", 3, Xint,   &[Xcharptr, Xany64, Xcharptr]),
        ("sprintf",  2, Xint,   &[Xcharptr, Xcharptr]),
        ("puts",     1, Xint,   &[Xcharptr]),
        ("fputs",    2, Xint,   &[Xcharptr, Xptr]),
        ("fputc",    2, Xint,   &[Xint, Xptr]),
        ("putchar",  1, Xint,   &[Xint]),
        ("fwrite",   4, Xany64, &[Xptr, Xany64, Xany64, Xptr]),
        ("fread",    4, Xany64, &[Xptr, Xany64, Xany64, Xptr]),
        ("fopen",    2, Xptr,   &[Xcharptr, Xcharptr]),
        ("fclose",   1, Xint,   &[Xptr]),
        ("fflush",   1, Xint,   &[Xptr]),
        ("ferror",   1, Xint,   &[Xptr]),
        ("feof",     1, Xint,   &[Xptr]),
        ("fileno",   1, Xint,   &[Xptr]),
        ("__sprintf_chk",  4, Xint, &[Xcharptr, Xint, Xany64, Xcharptr]),
        ("__snprintf_chk", 5, Xint, &[Xcharptr, Xany64, Xint, Xany64, Xcharptr]),
        ("__fprintf_chk",  3, Xint, &[Xptr, Xint, Xcharptr]),
        ("__printf_chk",   2, Xint, &[Xint, Xcharptr]),
        ("__memcpy_chk",   4, Xptr, &[Xptr, Xptr, Xany64, Xany64]),
        ("__memmove_chk",  4, Xptr, &[Xptr, Xptr, Xany64, Xany64]),
        ("__memset_chk",   4, Xptr, &[Xptr, Xint, Xany64, Xany64]),
        ("__strcpy_chk",   3, Xcharptr, &[Xcharptr, Xcharptr, Xany64]),
        ("__strcat_chk",   3, Xcharptr, &[Xcharptr, Xcharptr, Xany64]),
        ("malloc",   1, Xptr,   &[Xany64]),
        ("calloc",   2, Xptr,   &[Xany64, Xany64]),
        ("realloc",  2, Xptr,   &[Xptr, Xany64]),
        ("free",     1, Xvoid,  &[Xptr]),
        ("reallocarray", 3, Xptr, &[Xptr, Xany64, Xany64]),
        ("alloca",   1, Xptr,   &[Xany64]),
        ("exit",     1, Xvoid,  &[Xint]),
        ("abort",    0, Xvoid,  &[]),
        ("getenv",   1, Xcharptr, &[Xcharptr]),
        ("setenv",   3, Xint,   &[Xcharptr, Xcharptr, Xint]),
        ("setlocale", 2, Xcharptr, &[Xint, Xcharptr]),
        ("getopt_long", 5, Xint, &[Xint, Xcharptr, Xcharptr, Xptr, Xptr]),
        ("atexit",   1, Xint,   &[Xptr]),
        ("signal",   2, Xptr,   &[Xint, Xptr]),
        ("raise",    1, Xint,   &[Xint]),
        ("open",     2, Xint,   &[Xcharptr, Xint]),
        ("close",    1, Xint,   &[Xint]),
        ("read",     3, Xlong,  &[Xint, Xptr, Xany64]),
        ("write",    3, Xlong,  &[Xint, Xptr, Xany64]),
        ("lseek",    3, Xlong,  &[Xint, Xlong, Xint]),
        ("stat",     2, Xint,   &[Xcharptr, Xptr]),
        ("fstat",    2, Xint,   &[Xint, Xptr]),
        ("lstat",    2, Xint,   &[Xcharptr, Xptr]),
        ("fstatat",  4, Xint,   &[Xint, Xcharptr, Xptr, Xint]),
        ("readlink", 3, Xlong,  &[Xcharptr, Xcharptr, Xany64]),
        ("readlinkat", 4, Xlong, &[Xint, Xcharptr, Xcharptr, Xany64]),
        ("opendir",  1, Xptr,   &[Xcharptr]),
        ("closedir", 1, Xint,   &[Xptr]),
        ("readdir",  1, Xptr,   &[Xptr]),
        ("access",   2, Xint,   &[Xcharptr, Xint]),
        // mode_t/uid_t/gid_t are unsigned 32-bit in the libc prototypes.
        ("chmod",    2, Xint,   &[Xcharptr, Xintunsigned]),
        ("chown",    3, Xint,   &[Xcharptr, Xintunsigned, Xintunsigned]),
        ("isatty",   1, Xint,   &[Xint]),
        ("time",       1, Xlong, &[Xptr]),
        ("localtime",  1, Xptr,  &[Xptr]),
        ("localtime_r", 2, Xptr, &[Xptr, Xptr]),
        ("localtime_rz", 3, Xptr, &[Xptr, Xptr, Xptr]),
        ("strftime",   4, Xany64, &[Xptr, Xany64, Xptr, Xptr]),
        ("nstrftime",  6, Xany64, &[Xptr, Xany64, Xptr, Xptr, Xint, Xint]),
        ("human_readable", 5, Xcharptr, &[Xany64, Xcharptr, Xint, Xany64, Xany64]),
        ("umaxtostr",  2, Xcharptr, &[Xany64, Xcharptr]),
        ("imaxtostr",  2, Xcharptr, &[Xlong, Xcharptr]),
        ("offtostr",   2, Xcharptr, &[Xlong, Xcharptr]),
        ("uinttostr",  2, Xcharptr, &[Xany64, Xcharptr]),
        ("timetostr",  2, Xcharptr, &[Xlong, Xcharptr]),
        ("quotearg_style", 2, Xcharptr, &[Xint, Xcharptr]),
        ("quotearg_buffer", 5, Xany64, &[Xcharptr, Xany64, Xcharptr, Xany64, Xptr]),
        ("quote",      1, Xcharptr, &[Xcharptr]),
        // uid_t/gid_t are unsigned 32-bit; signed Xint here mis-signed comparisons and %u formatting.
        ("getpwuid",   1, Xptr, &[Xintunsigned]),
        ("getgrgid",   1, Xptr, &[Xintunsigned]),
        ("getuid",     0, Xint, &[]),
        ("getgid",     0, Xint, &[]),
        ("geteuid",    0, Xint, &[]),
        ("getegid",    0, Xint, &[]),
        ("error",      3, Xvoid, &[Xint, Xint, Xcharptr]),
        ("__errno_location", 0, Xptr, &[]),
        ("dcgettext",  3, Xcharptr, &[Xcharptr, Xcharptr, Xint]),
        ("__libc_start_main", 6, Xint, &[Xptr, Xint, Xptr, Xptr, Xptr, Xptr]),
        ("__cxa_finalize",    1, Xvoid,  &[Xptr]),
        ("__stack_chk_fail",  0, Xvoid,  &[]),
        ("__cxa_atexit",      3, Xint,   &[Xptr, Xptr, Xptr]),
        ("__ctype_get_mb_cur_max", 0, Xany64, &[]),
        ("__ctype_b_loc",     0, Xptr,   &[]),
        ("__ctype_toupper_loc", 0, Xptr, &[]),
        ("__ctype_tolower_loc", 0, Xptr, &[]),
        ("__fpending",        1, Xany64, &[Xptr]),
        ("__freading",        1, Xint,   &[Xptr]),
        ("__fpurge",          1, Xvoid,  &[Xptr]),
        ("__overflow",        2, Xint,   &[Xptr, Xint]),
        ("__uflow",           1, Xint,   &[Xptr]),
        ("__assert_fail",     4, Xvoid,  &[Xcharptr, Xcharptr, Xintunsigned, Xcharptr]),
        ("__getdelim",        4, Xlong,  &[Xptr, Xptr, Xint, Xptr]),
        ("_exit",             1, Xvoid,  &[Xint]),
        ("textdomain",        1, Xcharptr, &[Xcharptr]),
        ("bindtextdomain",    2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("nl_langinfo",       1, Xcharptr, &[Xint]),
        ("dcngettext",        5, Xcharptr, &[Xcharptr, Xcharptr, Xcharptr, Xlongunsigned, Xint]),
        ("localeconv",        0, Xptr,   &[]),
        ("newlocale",         3, Xptr,   &[Xint, Xcharptr, Xptr]),
        ("mbsinit",           1, Xint,   &[Xptr]),
        ("mbrtoc32",          4, Xany64, &[Xptr, Xcharptr, Xany64, Xptr]),
        ("iswprint",          1, Xint,   &[Xint]),
        ("iswcntrl",          1, Xint,   &[Xint]),
        ("wcwidth",           1, Xint,   &[Xint]),
        ("fputs_unlocked",    2, Xint,   &[Xcharptr, Xptr]),
        ("fputc_unlocked",    2, Xint,   &[Xint, Xptr]),
        ("fwrite_unlocked",   4, Xany64, &[Xptr, Xany64, Xany64, Xptr]),
        ("fread_unlocked",    4, Xany64, &[Xptr, Xany64, Xany64, Xptr]),
        ("fflush_unlocked",   1, Xint,   &[Xptr]),
        ("clearerr_unlocked", 1, Xvoid,  &[Xptr]),
        ("fseeko",            3, Xint,   &[Xptr, Xlong, Xint]),
        ("fdopen",            2, Xptr,   &[Xint, Xcharptr]),
        ("freopen",           3, Xptr,   &[Xcharptr, Xcharptr, Xptr]),
        ("setvbuf",           4, Xint,   &[Xptr, Xcharptr, Xint, Xany64]),
        ("strtoumax",         3, Xlongunsigned, &[Xcharptr, Xptr, Xint]),
        ("strtoimax",         3, Xlong,  &[Xcharptr, Xptr, Xint]),
        ("strspn",            2, Xany64, &[Xcharptr, Xcharptr]),
        ("strcspn",           2, Xany64, &[Xcharptr, Xcharptr]),
        ("strcoll",           2, Xint,   &[Xcharptr, Xcharptr]),
        ("stpcpy",            2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("strnlen",           2, Xany64, &[Xcharptr, Xany64]),
        ("mempcpy",           3, Xptr,   &[Xptr, Xptr, Xany64]),
        ("rawmemchr",         2, Xptr,   &[Xptr, Xint]),
        ("memrchr",           3, Xptr,   &[Xptr, Xint, Xany64]),
        ("fcntl",             2, Xint,   &[Xint, Xint]),
        ("ioctl",             2, Xint,   &[Xint, Xlongunsigned]),
        ("openat",            3, Xint,   &[Xint, Xcharptr, Xint]),
        ("posix_fadvise",     4, Xint,   &[Xint, Xlong, Xlong, Xint]),
        ("dup2",              2, Xint,   &[Xint, Xint]),
        ("ftruncate",         2, Xint,   &[Xint, Xlong]),
        ("fchdir",            1, Xint,   &[Xint]),
        ("chdir",             1, Xint,   &[Xcharptr]),
        ("getcwd",            2, Xcharptr, &[Xcharptr, Xany64]),
        ("dirfd",             1, Xint,   &[Xptr]),
        ("fdopendir",         1, Xptr,   &[Xint]),
        ("faccessat",         4, Xint,   &[Xint, Xcharptr, Xint, Xint]),
        ("unlink",            1, Xint,   &[Xcharptr]),
        ("unlinkat",          3, Xint,   &[Xint, Xcharptr, Xint]),
        ("mkdir",             2, Xint,   &[Xcharptr, Xintunsigned]),
        ("renameat",          4, Xint,   &[Xint, Xcharptr, Xint, Xcharptr]),
        ("renameat2",         5, Xint,   &[Xint, Xcharptr, Xint, Xcharptr, Xintunsigned]),
        ("linkat",            5, Xint,   &[Xint, Xcharptr, Xint, Xcharptr, Xint]),
        ("fchown",            3, Xint,   &[Xint, Xintunsigned, Xintunsigned]),
        ("fchownat",          5, Xint,   &[Xint, Xcharptr, Xintunsigned, Xintunsigned, Xint]),
        ("pathconf",          2, Xlong,  &[Xcharptr, Xint]),
        ("canonicalize_file_name", 1, Xcharptr, &[Xcharptr]),
        ("sigaction",         3, Xint,   &[Xint, Xptr, Xptr]),
        ("sigprocmask",       3, Xint,   &[Xint, Xptr, Xptr]),
        ("sigemptyset",       1, Xint,   &[Xptr]),
        ("sigaddset",         2, Xint,   &[Xptr, Xint]),
        ("sigismember",       2, Xint,   &[Xptr, Xint]),
        ("kill",              2, Xint,   &[Xint, Xint]),
        ("fork",              0, Xint,   &[]),
        ("execvp",            2, Xint,   &[Xcharptr, Xptr]),
        ("waitpid",           3, Xint,   &[Xint, Xptr, Xint]),
        ("unsetenv",          1, Xint,   &[Xcharptr]),
        ("aligned_alloc",     2, Xptr,   &[Xany64, Xany64]),
        ("clock_gettime",     2, Xint,   &[Xint, Xptr]),
        ("gettimeofday",      2, Xint,   &[Xptr, Xptr]),
        ("gmtime_r",          2, Xptr,   &[Xptr, Xptr]),
        ("clock",             0, Xlong,  &[]),
        ("uname",             1, Xint,   &[Xptr]),
        ("getpagesize",       0, Xint,   &[]),
        ("umask",             1, Xint,   &[Xintunsigned]),
        ("getpwnam",          1, Xptr,   &[Xcharptr]),
        ("getgrnam",          1, Xptr,   &[Xcharptr]),
        ("endpwent",          0, Xvoid,  &[]),
        ("endgrent",          0, Xvoid,  &[]),
        ("qsort",             4, Xvoid,  &[Xptr, Xany64, Xany64, Xptr]),
        ("fnmatch",           3, Xint,   &[Xcharptr, Xcharptr, Xint]),
        ("rpmatch",           1, Xint,   &[Xcharptr]),
        ("strtok_r",          3, Xcharptr, &[Xcharptr, Xcharptr, Xptr]),
        ("gettext",           1, Xcharptr, &[Xcharptr]),
        ("ngettext",          3, Xcharptr, &[Xcharptr, Xcharptr, Xlongunsigned]),
        ("dgettext",          2, Xcharptr, &[Xcharptr, Xcharptr]),
        ("quotearg_n_style",  3, Xcharptr, &[Xint, Xint, Xcharptr]),
        ("quote_n",           2, Xcharptr, &[Xint, Xcharptr]),
        ("canonicalize_filename_mode", 2, Xcharptr, &[Xcharptr, Xint]),
        ("mmap",              6, Xptr,   &[Xptr, Xany64, Xint, Xint, Xint, Xlong]),
        ("memchr",            3, Xptr,   &[Xptr, Xint, Xany64]),
        ("atoi",              1, Xint,   &[Xcharptr]),
        ("atol",              1, Xlong,  &[Xcharptr]),
        ("getopt",            3, Xint,   &[Xint, Xcharptr, Xcharptr]),
        ("putenv",            1, Xint,   &[Xcharptr]),
        ("rmdir",             1, Xint,   &[Xcharptr]),
        ("remove",            1, Xint,   &[Xcharptr]),
        ("rename",            2, Xint,   &[Xcharptr, Xcharptr]),
        ("dup",               1, Xint,   &[Xint]),
        ("pipe",              1, Xint,   &[Xptr]),
        ("fseek",             3, Xint,   &[Xptr, Xlong, Xint]),
        ("fgetc",             1, Xint,   &[Xptr]),
        ("getc",              1, Xint,   &[Xptr]),
        ("getchar",           0, Xint,   &[]),
        ("toupper",           1, Xint,   &[Xint]),
        ("tolower",           1, Xint,   &[Xint]),
        ("isspace",           1, Xint,   &[Xint]),
        ("isdigit",           1, Xint,   &[Xint]),
        ("isalpha",           1, Xint,   &[Xint]),
        ("isalnum",           1, Xint,   &[Xint]),
        ("mbstowcs",          3, Xany64, &[Xptr, Xcharptr, Xany64]),
        ("ftell",             1, Xlong,  &[Xptr]),
        ("sysconf",           1, Xlong,  &[Xint]),
        ("munmap",            2, Xint,   &[Xptr, Xany64]),
        ("sscanf",            2, Xint,   &[Xcharptr, Xcharptr]),
        // <math.h> (mirrors header_functions.json); Xfloat == C double, and the true fixed arity clamps call sites whose argument count was over-recovered.
        ("pow",               2, Xfloat, &[Xfloat, Xfloat]),
        ("fmod",              2, Xfloat, &[Xfloat, Xfloat]),
        ("atan2",             2, Xfloat, &[Xfloat, Xfloat]),
        ("sqrt",              1, Xfloat, &[Xfloat]),
        ("sin",               1, Xfloat, &[Xfloat]),
        ("cos",               1, Xfloat, &[Xfloat]),
        ("tan",               1, Xfloat, &[Xfloat]),
        ("asin",              1, Xfloat, &[Xfloat]),
        ("acos",              1, Xfloat, &[Xfloat]),
        ("atan",              1, Xfloat, &[Xfloat]),
        ("sinh",              1, Xfloat, &[Xfloat]),
        ("cosh",              1, Xfloat, &[Xfloat]),
        ("tanh",              1, Xfloat, &[Xfloat]),
        ("exp",               1, Xfloat, &[Xfloat]),
        ("log",               1, Xfloat, &[Xfloat]),
        ("log10",             1, Xfloat, &[Xfloat]),
        ("fabs",              1, Xfloat, &[Xfloat]),
        ("ceil",              1, Xfloat, &[Xfloat]),
        ("floor",             1, Xfloat, &[Xfloat]),
        ("round",             1, Xfloat, &[Xfloat]),
        // `f` (float) variants -- Xsingle == C float for ret/args.
        ("powf",              2, Xsingle, &[Xsingle, Xsingle]),
        ("fmodf",             2, Xsingle, &[Xsingle, Xsingle]),
        ("atan2f",            2, Xsingle, &[Xsingle, Xsingle]),
        ("sqrtf",             1, Xsingle, &[Xsingle]),
        ("sinf",              1, Xsingle, &[Xsingle]),
        ("cosf",              1, Xsingle, &[Xsingle]),
        ("tanf",              1, Xsingle, &[Xsingle]),
        ("asinf",             1, Xsingle, &[Xsingle]),
        ("acosf",             1, Xsingle, &[Xsingle]),
        ("atanf",             1, Xsingle, &[Xsingle]),
        ("sinhf",             1, Xsingle, &[Xsingle]),
        ("coshf",             1, Xsingle, &[Xsingle]),
        ("tanhf",             1, Xsingle, &[Xsingle]),
        ("expf",              1, Xsingle, &[Xsingle]),
        ("logf",              1, Xsingle, &[Xsingle]),
        ("log10f",            1, Xsingle, &[Xsingle]),
        ("fabsf",             1, Xsingle, &[Xsingle]),
        ("ceilf",             1, Xsingle, &[Xsingle]),
        ("floorf",            1, Xsingle, &[Xsingle]),
        ("roundf",            1, Xsingle, &[Xsingle]),
        // Long-double (l) variants; the prototype is header-suppressed, so the Xfloat shape only feeds arity normalization and float-typed args.
        ("powl",              2, Xfloat, &[Xfloat, Xfloat]),
        ("fmodl",             2, Xfloat, &[Xfloat, Xfloat]),
        ("atan2l",            2, Xfloat, &[Xfloat, Xfloat]),
        ("sqrtl",             1, Xfloat, &[Xfloat]),
        ("sinl",              1, Xfloat, &[Xfloat]),
        ("cosl",              1, Xfloat, &[Xfloat]),
        ("tanl",              1, Xfloat, &[Xfloat]),
        ("asinl",             1, Xfloat, &[Xfloat]),
        ("acosl",             1, Xfloat, &[Xfloat]),
        ("atanl",             1, Xfloat, &[Xfloat]),
        ("sinhl",             1, Xfloat, &[Xfloat]),
        ("coshl",             1, Xfloat, &[Xfloat]),
        ("tanhl",             1, Xfloat, &[Xfloat]),
        ("expl",              1, Xfloat, &[Xfloat]),
        ("logl",              1, Xfloat, &[Xfloat]),
        ("log10l",            1, Xfloat, &[Xfloat]),
        ("fabsl",             1, Xfloat, &[Xfloat]),
        ("ceill",             1, Xfloat, &[Xfloat]),
        ("floorl",            1, Xfloat, &[Xfloat]),
        ("roundl",            1, Xfloat, &[Xfloat]),
    ];

    for &(name, param_count, ret_type, param_types) in signatures {
        db.rel_push("known_extern_signature", (
            name,
            param_count,
            ret_type,
            Arc::new(param_types.to_vec()),
        ));
    }

    // Known variadic functions: (name, declared_param_count)
    let varargs: &[(&str, usize)] = &[
        ("printf", 1), ("fprintf", 2), ("sprintf", 2), ("snprintf", 3),
        ("scanf", 1), ("fscanf", 2), ("sscanf", 2),
        ("__printf_chk", 2), ("__fprintf_chk", 3), ("__sprintf_chk", 4), ("__snprintf_chk", 5),
        ("error", 3), ("error_at_line", 5),
        ("dprintf", 2),
        ("syslog", 2),
        ("open", 2), ("openat", 3), ("fcntl", 2), ("ioctl", 2),
        ("execl", 2), ("execlp", 2), ("execle", 2),
    ];
    for &(name, fixed_args) in varargs {
        db.rel_push("known_varargs_function", (name, fixed_args));
    }
}


// Required pointee size of struct-writing extern out-params (name, param_index, bytes); propagation into stack_struct_buffers is still TODO in struct_recovery_pass.
fn populate_known_func_param_pointee_size(db: &mut DecompileDB) {
    // (name, out-param index, struct size in bytes on x86-64 Linux / glibc)
    const STAT_SIZE: usize = 144;     // struct stat
    const TM_SIZE: usize = 56;        // struct tm
    const UTSNAME_SIZE: usize = 390;  // struct utsname (6 * _UTSNAME_LENGTH=65)
    let entries: &[(Symbol, usize, usize)] = &[
        ("stat", 1, STAT_SIZE),
        ("lstat", 1, STAT_SIZE),
        ("fstat", 1, STAT_SIZE),
        ("fstatat", 2, STAT_SIZE),
        ("localtime_r", 1, TM_SIZE),   // localtime_r(const time_t*, struct tm*)
        ("localtime_rz", 2, TM_SIZE),  // localtime_rz(timezone_t, const time_t*, struct tm*)
        ("gmtime_r", 1, TM_SIZE),      // gmtime_r(const time_t*, struct tm*)
        ("uname", 0, UTSNAME_SIZE),
    ];
    for &(name, idx, size) in entries {
        db.rel_push("known_func_param_pointee_size", (name, idx, size));
    }
}


// Register known types for well-known global variables (stdout, optarg, etc.).
fn populate_known_global_types(db: &mut DecompileDB) {
    use XType::*;

    // REMOVED only Version, a GENERIC identifier that force-typed unrelated program types to char*; program_name and exit_failure are KEPT as specific gnulib names whose recovered evidence is too weak.
    let known_globals: &[(&str, XType)] = &[
        ("stdout", Xptr),
        ("stderr", Xptr),
        ("stdin", Xptr),
        ("optarg", Xcharptr),
        ("optind", Xint),
        ("opterr", Xint),
        ("optopt", Xint),
        ("exit_failure", Xint),
        ("program_name", Xcharptr),
        ("program_invocation_name", Xcharptr),
        ("program_invocation_short_name", Xcharptr),
        ("__ctype_b_loc", Xptr),
        ("errno", Xint),
    ];

    for &(name, ref xtype) in known_globals {
        db.rel_push("known_global_type", (name, xtype.clone()));
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    #[test]
    fn coff_decoration_candidates_are_format_bounded() {
        let candidates = coff_identity_candidates("__imp__memcpy@24", true, true);
        assert!(candidates.iter().any(|name| name == "memcpy"));
        assert!(!candidates.iter().any(|name| name == "cpy"));

        let x64 = coff_identity_candidates("__imp__memcpy@24", true, false);
        assert!(!x64.iter().any(|name| name == "memcpy"));

        let ordinary = coff_identity_candidates("domain_specific_prefix_memcpy", true, false);
        assert_eq!(ordinary, vec!["domain_specific_prefix_memcpy"]);

        let non_coff = coff_identity_candidates("__imp_memcpy", false, false);
        assert_eq!(non_coff, vec!["__imp_memcpy"]);
    }

    #[test]
    fn original_import_identity_binds_provider_prototype() {
        let mut db = DecompileDB::default();
        db.target_abi = Some(crate::abi::AbiConfig::win64());
        db.rel_push(
            "loader_symbol_identity",
            (
                0x1000u64,
                LoaderSymbolKind::ImportPointer,
                "coff_ext_memcpy" as Symbol,
                "__imp_memcpy" as Symbol,
            ),
        );
        db.rel_push(
            "known_extern_signature",
            (
                "memcpy" as Symbol,
                3usize,
                XType::Xptr,
                Arc::new(vec![XType::Xptr, XType::Xptr, XType::Xany64]),
            ),
        );

        bind_loader_symbol_identities(&mut db);

        assert!(db
            .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>(
                "known_extern_signature"
            )
            .any(|(name, arity, _, _)| *name == "coff_ext_memcpy" && *arity == 3));
        assert!(db
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
            .any(|(address, kind, _, original, arity, _, _, variadic)| {
                *address == 0x1000
                    && *kind == LoaderSymbolKind::ImportPointer
                    && *original == "__imp_memcpy"
                    && *arity == 3
                    && !*variadic
            }));
    }

    #[test]
    fn exact_loader_identity_survives_display_sanitization() {
        let mut db = DecompileDB::default();
        db.rel_push(
            "loader_symbol_identity",
            (
                0x2000u64,
                LoaderSymbolKind::Function,
                "worker_constprop_0" as Symbol,
                "worker.constprop.0" as Symbol,
            ),
        );
        db.rel_push(
            "known_extern_signature",
            (
                "worker.constprop.0" as Symbol,
                1usize,
                XType::Xint,
                Arc::new(vec![XType::Xlong]),
            ),
        );

        bind_loader_symbol_identities(&mut db);

        assert!(db
            .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>(
                "known_extern_signature"
            )
            .any(|(name, arity, _, _)| *name == "worker_constprop_0" && *arity == 1));
    }

    #[test]
    fn provider_collisions_and_variadic_conflicts_fail_closed() {
        let mut db = DecompileDB::default();
        for (address, kind, original) in [
            (0x3000u64, LoaderSymbolKind::Function, "printf"),
            (0x4000u64, LoaderSymbolKind::ImportPointer, "puts"),
        ] {
            db.rel_push(
                "loader_symbol_identity",
                (address, kind, "shared_provider" as Symbol, original as Symbol),
            );
        }
        db.rel_push(
            "known_extern_signature",
            (
                "printf" as Symbol,
                1usize,
                XType::Xint,
                Arc::new(vec![XType::Xcharptr]),
            ),
        );
        db.rel_push("known_varargs_function", ("printf" as Symbol, 1usize));
        db.rel_push("known_varargs_function", ("printf" as Symbol, 2usize));
        db.rel_push(
            "known_extern_signature",
            (
                "puts" as Symbol,
                1usize,
                XType::Xint,
                Arc::new(vec![XType::Xcharptr]),
            ),
        );

        bind_loader_symbol_identities(&mut db);

        assert!(!db
            .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>(
                "known_extern_signature"
            )
            .any(|(name, _, _, _)| *name == "shared_provider"));
        assert!(!db
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
            .any(|(address, _, _, _, _, _, _, _)| *address == 0x3000));
        assert!(db
            .rel_iter::<(Address, LoaderSymbolKind, Symbol, Symbol)>(
                "known_loader_variadic"
            )
            .any(|(address, _, _, _)| *address == 0x3000));
        assert!(db
            .rel_iter::<(Address, LoaderSymbolKind, Symbol, Symbol)>(
                "loader_signature_conflict"
            )
            .any(|(address, _, _, _)| *address == 0x3000));
    }
}
