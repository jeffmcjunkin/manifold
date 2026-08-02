// Export selected Clight IR to JSON mirroring CompCert's Clight AST for OCaml-side reconstruction.

use crate::decompile::elevator::DecompileDB;
use crate::decompile::passes::clight_select::query::{
    build_param_xtypes, build_rtl_to_mreg_at_entry, extract_callee_signatures, extract_globals,
    extract_struct_definitions, insert_preferred_symbol_name,
};
use crate::decompile::passes::clight_select::select::{
    select_clight_stmts, validate_partial_unsupported_functions, SelectedFunction,
};
use crate::x86::types::*;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub const CLIGHT_EXPORT_SCHEMA_ID: &str = "manifold-clight-v2";
pub const PARTIAL_SUPPRESSION_CERTIFICATE_ID: &str =
    "atomic-unsupported-address-suppression-v1";

/// Export the selected Clight IR from the decompile DB to a JSON file.
pub fn export_clight_json(db: &DecompileDB, output_path: &str) -> Result<(), String> {
    // Diagnostics
    let csharp_count = db
        .rel_iter::<(Node, CsharpminorStmt)>("csharp_stmt")
        .count();
    let clight_count = db.rel_iter::<(Node, ClightStmt)>("clight_stmt").count();
    let emit_count = db
        .rel_iter::<(Address, Node, ClightStmt)>("emit_clight_stmt")
        .count();
    let var_type_count = db
        .rel_iter::<(RTLReg, XType)>("emit_var_type_candidate")
        .count();
    eprintln!("=== Clight Pipeline Diagnostics ===");
    eprintln!("  csharp_stmt:                    {}", csharp_count);
    eprintln!("  clight_stmt:                    {}", clight_count);
    eprintln!("  emit_clight_stmt:               {}", emit_count);
    eprintln!("  emit_var_type:                  {}", var_type_count);
    eprintln!("=== End Diagnostics ===\n");

    let selected_functions = select_clight_stmts(db)?;

    let binary_path = db.binary_path.as_ref().ok_or("binary_path not set")?;

    // Build id_to_name mapping
    let mut id_to_name: HashMap<usize, String> = HashMap::new();
    for (id, name) in db.rel_iter::<(Ident, Symbol)>("ident_to_symbol") {
        insert_preferred_symbol_name(&mut id_to_name, *id, name);
    }
    let mut symbol_names: HashMap<usize, String> = HashMap::new();
    for (addr, name, _) in db.rel_iter::<(Address, Symbol, Symbol)>("symbols") {
        insert_preferred_symbol_name(&mut symbol_names, *addr as usize, name);
    }
    for (id, name) in symbol_names {
        id_to_name.insert(id, name);
    }
    for func in &selected_functions {
        id_to_name
            .entry(func.address as usize)
            .or_insert_with(|| func.name.clone());
    }

    // Use the same emit_function authority and FUN_ fallback as normal
    // function extraction. func_span can contain aliases and its parallel
    // relation order is not a deterministic provider-name choice.
    let mut emit_name_rows: Vec<(Address, String)> = db
        .rel_iter::<(Address, Symbol, Node)>("emit_function")
        .map(|(address, name, _)| {
            let final_name = if name.starts_with("FUN_") {
                id_to_name
                    .get(&(*address as usize))
                    .cloned()
                    .unwrap_or_else(|| (*name).to_string())
            } else {
                (*name).to_string()
            };
            (*address, final_name)
        })
        .collect();
    emit_name_rows.sort();
    emit_name_rows.dedup();
    let mut provider_names: HashMap<Address, String> = HashMap::new();
    for (address, name) in emit_name_rows {
        if let Some(previous) = provider_names.insert(address, name.clone()) {
            if previous != name {
                return Err(format!(
                    "emit_function gives address 0x{address:x} conflicting provider names"
                ));
            }
        }
    }
    // Provider identity is authoritative for emitted definitions, direct
    // Evar call targets, and any declaration synthesized for an omitted
    // internal function.
    for (address, name) in &provider_names {
        id_to_name.insert(*address as usize, name.clone());
    }

    let external_funcs: std::collections::HashSet<u64> = db
        .rel_iter::<(Address,)>("is_external_function")
        .map(|(a,)| *a)
        .collect();

    const CRT_FUNCTIONS: &[&str] = &[
        "_start",
        "_init",
        "_fini",
        "__libc_csu_init",
        "__libc_csu_fini",
        "__libc_start_main",
        "deregister_tm_clones",
        "register_tm_clones",
        "__do_global_dtors_aux",
        "frame_dummy",
        "__x86.get_pc_thunk.bx",
    ];

    // Partial metadata describes definitions in this exact JSON document, not
    // merely functions which survived statement selection. Apply the final
    // emission filter before classifying any diagnostic.
    let internal_functions: Vec<&SelectedFunction> = selected_functions
        .iter()
        .filter(|func| {
            !external_funcs.contains(&func.address)
                && !func.name.starts_with('.')
                && !func.name.starts_with("__")
                && !CRT_FUNCTIONS.contains(&func.name.as_str())
                && !func.name.starts_with("FUN_")
                && !db.should_skip_function(func.name.as_str())
        })
        .collect();
    let emitted_addresses: std::collections::HashSet<Address> = internal_functions
        .iter()
        .map(|function| function.address)
        .collect();

    let partial_validation = validate_partial_unsupported_functions(db);
    if !partial_validation.orphan_certificates.is_empty()
        || !partial_validation.orphan_provenance_sites.is_empty()
        || !partial_validation.orphan_budget_functions.is_empty()
        || !partial_validation.orphan_details.is_empty()
        || !partial_validation.unknown_details.is_empty()
    {
        return Err(format!(
            "invalid partial-suppression relation bundle: orphan certificates={:?}, provenance={:?}, budgets={:?}, details={:?}; unknown details={:?}",
            partial_validation.orphan_certificates,
            partial_validation.orphan_provenance_sites,
            partial_validation.orphan_budget_functions,
            partial_validation.orphan_details,
            partial_validation.unknown_details,
        ));
    }
    let mut unsupported_stack_rows: Vec<(Address, Address, String)> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .map(|(func, access, reason)| (*func, *access, (*reason).to_string()))
        .collect();
    unsupported_stack_rows.sort();
    unsupported_stack_rows.dedup();
    let mut unsupported_details: BTreeMap<(Address, Address), Vec<String>> = BTreeMap::new();
    for (func, access, detail) in
        db.rel_iter::<(Address, Address, Symbol)>("unsupported_address_detail")
    {
        unsupported_details
            .entry((*func, *access))
            .or_default()
            .push((*detail).to_string());
    }
    for details in unsupported_details.values_mut() {
        details.sort();
        details.dedup();
    }

    let mut unsupported_functions = Vec::with_capacity(unsupported_stack_rows.len());
    for (func, access, reason) in &unsupported_stack_rows {
        if !matches!(
            reason.as_str(),
            "unsupported-stack-address" | "unsupported-addr32-address"
        ) {
            return Err(format!(
                "unsupported function 0x{func:x} has unknown reason code {reason:?}"
            ));
        }
        let name = provider_names.get(func).ok_or_else(|| {
            format!("unsupported function 0x{func:x} has no emit_function provider name")
        })?;
        let base = json!({
            "name": name,
            "address": format!("0x{func:x}"),
            "access_address": format!("0x{access:x}"),
            "reason": reason,
            "details": unsupported_details
                .get(&(*func, *access))
                .cloned()
                .unwrap_or_default(),
        });
        if !emitted_addresses.contains(func) {
            unsupported_functions.push(base);
            continue;
        }
        if !partial_validation.partial_functions.contains_key(func) {
            return Err(format!(
                "emitted partial function 0x{func:x} failed strict suppression-certificate validation"
            ));
        }
    }

    // Schema v2 groups all sites for a function into one stable record. Site
    // rows and instruction provenance are sorted by the shared validator; the
    // counts and policy bounds certify their function-wide union exactly.
    let mut partial_functions = Vec::new();
    for (func, certificate) in &partial_validation.partial_functions {
        if !emitted_addresses.contains(func) {
            continue;
        }
        let name = provider_names.get(func).ok_or_else(|| {
            format!("partial function 0x{func:x} has no emit_function provider name")
        })?;
        let diagnostics: Vec<Value> = certificate
            .sites
            .iter()
            .map(|site| {
                json!({
                    "access_address": format!("0x{:x}", site.access),
                    "reason": site.reason,
                    "details": unsupported_details
                        .get(&(*func, site.access))
                        .cloned()
                        .unwrap_or_default(),
                    "suppressed_instructions": site
                        .suppressed_nodes
                        .iter()
                        .map(|node| format!("0x{node:x}"))
                        .collect::<Vec<_>>(),
                })
            })
            .collect();
        partial_functions.push(json!({
            "name": name,
            "address": format!("0x{func:x}"),
            "diagnostics": diagnostics,
            "certification": {
                "kind": PARTIAL_SUPPRESSION_CERTIFICATE_ID,
                "atomic_chain_suppressed": true,
                "all_function_diagnostics_certified": true,
                "suppressed_instruction_count": certificate.suppressed_instruction_count,
                "owned_instruction_count": certificate.owned_instruction_count,
                "max_suppressed_instruction_count": crate::decompile::passes::rtl_pass::MAX_PARTIAL_SUPPRESSED_INSTRUCTIONS,
                "min_owned_instructions_per_suppressed": crate::decompile::passes::rtl_pass::MIN_OWNED_INSTRUCTIONS_PER_SUPPRESSED,
            }
        }));
    }

    // Extract struct definitions (composites)
    let struct_defs = extract_struct_definitions(db);
    let composites: Vec<Value> = struct_defs
        .iter()
        .map(|s| {
            let fields: Vec<Value> = s
                .definition
                .fields
                .iter()
                .map(|f| {
                    json!({
                        "name": f.name,
                        "ty": serialize_ctype_from_field(&f.ty)
                    })
                })
                .collect();
            json!({
                "id": s.struct_id,
                "name": format!("struct_{:x}", s.struct_id),
                "su": "Struct",
                "members": fields
            })
        })
        .collect();

    // Extract globals
    let globals = extract_globals(db, binary_path).unwrap_or_default();
    let globals_json: Vec<Value> = globals
        .iter()
        .map(|g| {
            let mut obj = json!({
                "name": g.name,
                "id": g.id,
                "is_string": g.is_string,
                "is_pointer": g.is_pointer,
            });
            if g.is_string {
                let s = String::from_utf8_lossy(&g.content)
                    .trim_end_matches('\0')
                    .to_string();
                obj["string_value"] = json!(s);
            }
            obj
        })
        .collect();

    // Collect names of functions that are actually called from internal functions
    let internal_names: std::collections::HashSet<&str> =
        internal_functions.iter().map(|f| f.name.as_str()).collect();
    let mut called_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for func in &internal_functions {
        for stmt in func.statements.values() {
            collect_called_functions(stmt, &mut called_names, &id_to_name);
        }
    }

    // Build external function declarations (only for actually-called externals)
    let mut known_extern_rows: Vec<(String, usize, XType, Vec<XType>)> = db
        .rel_iter::<(Symbol, usize, XType, Arc<Vec<XType>>)>("known_extern_signature")
        .filter(|(name, _, _, _)| called_names.contains(*name) && !internal_names.contains(*name))
        .map(|(name, param_count, ret_type, param_types)| {
            (
                name.to_string(),
                *param_count,
                *ret_type,
                param_types.as_ref().clone(),
            )
        })
        .collect();
    known_extern_rows.sort();
    known_extern_rows.dedup();
    let mut extern_sigs: Vec<Value> = known_extern_rows
        .into_iter()
        .map(|(name, param_count, ret_type, param_types)| {
            json!({
                "name": name,
                "param_count": param_count,
                "return_type": serialize_xtype(&ret_type),
                "param_types": param_types.iter().map(serialize_xtype).collect::<Vec<_>>()
            })
        })
        .collect();

    // A selected sibling may call an internal function omitted by the
    // structured unsupported-address path. Such a callee still needs a real
    // declaration: otherwise the JSON-to-C consumer either relies on an
    // implicit-int declaration or fails C++ compilation. Reuse the same
    // deterministic recovered signatures that type direct calls.
    let declared_names: std::collections::HashSet<String> = extern_sigs
        .iter()
        .filter_map(|value| value.get("name")?.as_str().map(str::to_owned))
        .collect();
    let mut provider_addresses_by_name: HashMap<String, Vec<Address>> = HashMap::new();
    for (address, name) in &provider_names {
        provider_addresses_by_name
            .entry(name.clone())
            .or_default()
            .push(*address);
    }
    for addresses in provider_addresses_by_name.values_mut() {
        addresses.sort_unstable();
        addresses.dedup();
    }
    let param_xtypes = build_param_xtypes(db);
    let rtl_to_mreg = build_rtl_to_mreg_at_entry(db);
    let recovered_sigs = extract_callee_signatures(db, &param_xtypes, &rtl_to_mreg);
    let unsupported_addresses: std::collections::HashSet<Address> = db
        .rel_iter::<(Address, Address, Symbol)>("unsupported_stack_address")
        .map(|(func, _, _)| *func)
        .collect();
    let mut omitted_called_names: Vec<String> = called_names
        .iter()
        .filter(|name| !internal_names.contains(name.as_str()) && !declared_names.contains(*name))
        .cloned()
        .collect();
    omitted_called_names.sort();
    omitted_called_names.dedup();
    for name in omitted_called_names {
        let Some(addresses) = provider_addresses_by_name.get(&name) else {
            continue;
        };
        let unsupported: Vec<Address> = addresses
            .iter()
            .copied()
            .filter(|address| unsupported_addresses.contains(address))
            .collect();
        if unsupported.is_empty() {
            continue;
        }
        if unsupported.len() != 1 {
            return Err(format!(
                "called omitted internal {name} has ambiguous provider addresses: {unsupported:x?}"
            ));
        }
        let address = unsupported[0];
        let signature = recovered_sigs.get(&(address as Ident)).ok_or_else(|| {
            format!("called omitted internal {name} at 0x{address:x} has no recovered signature")
        })?;
        let mut param_types = signature.param_types.clone();
        param_types.resize(signature.param_count, XType::Xint);
        param_types.truncate(signature.param_count);
        extern_sigs.push(json!({
            "name": name,
            "param_count": signature.param_count,
            "return_type": serialize_xtype(&signature.return_type),
            "param_types": param_types.iter().map(serialize_xtype).collect::<Vec<_>>()
        }));
    }
    extern_sigs.sort_by_cached_key(external_decl_sort_key);
    extern_sigs.dedup();

    // Serialize internal functions
    let functions_json: Vec<Value> = internal_functions
        .iter()
        .map(|func| serialize_function(func, &id_to_name))
        .collect();

    let program = json!({
        "compcert_clight": true,
        "manifold_clight_schema": CLIGHT_EXPORT_SCHEMA_ID,
        "arch": "x86_64",
        "composites": composites,
        "globals": globals_json,
        "externals": extern_sigs,
        "functions": functions_json,
        "unsupported_functions": unsupported_functions,
        "partial_functions": partial_functions,
    });

    let json_str = serde_json::to_string_pretty(&program)
        .map_err(|e| format!("JSON serialization error: {}", e))?;

    std::fs::write(output_path, json_str)
        .map_err(|e| format!("Failed to write {}: {}", output_path, e))?;

    Ok(())
}

fn external_decl_sort_key(value: &Value) -> (String, u64, String, String) {
    (
        value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        value
            .get("param_count")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        value
            .get("return_type")
            .map(Value::to_string)
            .unwrap_or_default(),
        value
            .get("param_types")
            .map(Value::to_string)
            .unwrap_or_default(),
    )
}

fn serialize_function(func: &SelectedFunction, id_to_name: &HashMap<usize, String>) -> Value {
    // Build a local name map including param/temp register names for consistent identifiers.
    let mut local_names = id_to_name.clone();
    for reg in &func.param_regs {
        local_names
            .entry(*reg as usize)
            .or_insert_with(|| format!("param_{}", reg));
    }
    let param_set: std::collections::HashSet<RTLReg> = func.param_regs.iter().copied().collect();
    for reg in &func.used_regs {
        if !param_set.contains(reg) {
            local_names
                .entry(*reg as usize)
                .or_insert_with(|| format!("var_{}", reg));
        }
    }

    // Serialize parameters
    let params: Vec<Value> = func
        .param_regs
        .iter()
        .zip(func.param_types.iter())
        .map(|(reg, pty)| {
            json!({
                "id": *reg as usize,
                "name": resolve_name(*reg as usize, &local_names),
                "ty": serialize_param_type(pty)
            })
        })
        .collect();

    // Collect local variables (temps): all used regs minus params
    let mut temps: Vec<Value> = func
        .used_regs
        .iter()
        .filter(|r| !param_set.contains(r))
        .map(|reg| {
            let ty = func
                .var_types
                .get(reg)
                .map(|s| xtype_string_to_clight_type(s))
                .unwrap_or_else(|| json!({"tag": "Tlong", "sign": "Unsigned"}));
            json!({
                "id": *reg as usize,
                "name": resolve_name(*reg as usize, &local_names),
                "ty": ty
            })
        })
        .collect();
    temps.sort_by_key(|v| v["id"].as_u64().unwrap_or(0));

    // Build the function body: right-fold Ssequence over sorted nodes
    let body = serialize_function_body(func, &local_names);

    // Serialize successors (CFG edges)
    let mut cfg_edges: Vec<(Node, Node)> = func
        .successors
        .iter()
        .flat_map(|(src, dsts)| dsts.iter().map(move |dst| (*src, *dst)))
        .collect();
    cfg_edges.sort_unstable();
    let cfg: Vec<Value> = cfg_edges
        .into_iter()
        .map(|(src, dst)| json!([src, dst]))
        .collect();

    json!({
        "name": func.name,
        "address": format!("0x{:x}", func.address),
        "entry_node": func.entry_node,
        "return_type": serialize_clight_type(&func.return_type),
        "callconv": "cc_default",
        "params": params,
        "temps": temps,
        "vars": [],
        "body": body,
        "cfg": cfg,
        "stack_size": func.stack_size,
    })
}

/// Serialize all statements for a function, providing both per-node and flattened CFG representations.
fn serialize_function_body(func: &SelectedFunction, id_to_name: &HashMap<usize, String>) -> Value {
    // Per-node statements (preserves CFG structure)
    let mut node_stmts: Vec<(Node, Value)> = func
        .statements
        .iter()
        .map(|(node, stmt)| (*node, serialize_stmt(stmt, id_to_name)))
        .collect();
    node_stmts.sort_by_key(|(n, _)| *n);

    // Build a right-folded Ssequence from address-ordered statements
    if node_stmts.is_empty() {
        return json!({"tag": "Sskip"});
    }

    // Return as a structured object with both representations
    json!({
        "node_stmts": node_stmts.iter().map(|(n, s)| {
            json!({"node": *n, "stmt": s.clone()})
        }).collect::<Vec<_>>(),
        "sequence": fold_ssequence(&node_stmts.iter().map(|(_, s)| s.clone()).collect::<Vec<_>>()),
    })
}

/// Right-fold a list of statements into nested Ssequence (CompCert's binary form).
fn fold_ssequence(stmts: &[Value]) -> Value {
    match stmts.len() {
        0 => json!({"tag": "Sskip"}),
        1 => stmts[0].clone(),
        _ => json!({
            "tag": "Ssequence",
            "s1": stmts[0].clone(),
            "s2": fold_ssequence(&stmts[1..])
        }),
    }
}

fn serialize_stmt(stmt: &ClightStmt, id_to_name: &HashMap<usize, String>) -> Value {
    match stmt {
        ClightStmt::Sskip => json!({"tag": "Sskip"}),

        ClightStmt::Sassign(lhs, rhs) => json!({
            "tag": "Sassign",
            "lhs": serialize_expr(lhs, id_to_name),
            "rhs": serialize_expr(rhs, id_to_name),
        }),

        ClightStmt::Sset(id, expr) => json!({
            "tag": "Sset",
            "id": *id,
            "name": resolve_name(*id, id_to_name),
            "rhs": serialize_expr(expr, id_to_name),
        }),

        ClightStmt::Scall(opt_id, func_expr, args) => {
            let mut obj = json!({
                "tag": "Scall",
                "func": serialize_expr(func_expr, id_to_name),
                "args": args.iter().map(|a| serialize_expr(a, id_to_name)).collect::<Vec<_>>(),
            });
            if let Some(id) = opt_id {
                obj["result_id"] = json!(*id);
                obj["result_name"] = json!(resolve_name(*id, id_to_name));
            }
            obj
        }

        ClightStmt::Sbuiltin(opt_id, ef, tys, args) => {
            let mut obj = json!({
                "tag": "Sbuiltin",
                "ef": serialize_external_function(ef),
                "arg_types": tys.iter().map(serialize_clight_type).collect::<Vec<_>>(),
                "args": args.iter().map(|a| serialize_expr(a, id_to_name)).collect::<Vec<_>>(),
            });
            if let Some(id) = opt_id {
                obj["result_id"] = json!(*id);
            }
            obj
        }

        ClightStmt::Ssequence(stmts) => {
            let serialized: Vec<Value> = stmts
                .iter()
                .map(|s| serialize_stmt(s, id_to_name))
                .collect();
            fold_ssequence(&serialized)
        }

        ClightStmt::Sifthenelse(cond, then_s, else_s) => json!({
            "tag": "Sifthenelse",
            "cond": serialize_expr(cond, id_to_name),
            "then": serialize_stmt(then_s, id_to_name),
            "else": serialize_stmt(else_s, id_to_name),
        }),

        ClightStmt::Sloop(s1, s2) => json!({
            "tag": "Sloop",
            "body1": serialize_stmt(s1, id_to_name),
            "body2": serialize_stmt(s2, id_to_name),
        }),

        ClightStmt::Sbreak => json!({"tag": "Sbreak"}),
        ClightStmt::Scontinue => json!({"tag": "Scontinue"}),

        ClightStmt::Sreturn(opt_expr) => {
            let mut obj = json!({"tag": "Sreturn"});
            if let Some(expr) = opt_expr {
                obj["expr"] = serialize_expr(expr, id_to_name);
            }
            obj
        }

        ClightStmt::Sswitch(expr, cases) => json!({
            "tag": "Sswitch",
            "expr": serialize_expr(expr, id_to_name),
            "cases": cases.iter().map(|(label, s)| {
                json!({
                    "label": match label {
                        Some(z) => json!(*z),
                        None => json!("default"),
                    },
                    "stmt": serialize_stmt(s, id_to_name),
                })
            }).collect::<Vec<_>>(),
        }),

        ClightStmt::Slabel(id, s) => json!({
            "tag": "Slabel",
            "label": *id,
            "label_name": resolve_name(*id, id_to_name),
            "stmt": serialize_stmt(s, id_to_name),
        }),

        ClightStmt::Sgoto(id) => json!({
            "tag": "Sgoto",
            "label": *id,
            "label_name": resolve_name(*id, id_to_name),
        }),
    }
}

fn serialize_expr(expr: &ClightExpr, id_to_name: &HashMap<usize, String>) -> Value {
    match expr {
        ClightExpr::EconstInt(v, ty) => json!({
            "tag": "Econst_int",
            "value": *v,
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::EconstLong(v, ty) => json!({
            "tag": "Econst_long",
            "value": *v,
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::EconstFloat(v, ty) => json!({
            "tag": "Econst_float",
            "value": v.0,
            "bits": v.0.to_bits(),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::EconstSingle(v, ty) => json!({
            "tag": "Econst_single",
            "value": v.0,
            "bits": v.0.to_bits(),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Evar(id, ty) => json!({
            "tag": "Evar",
            "id": *id,
            "name": resolve_name(*id, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::EvarSymbol(name, ty) => json!({
            "tag": "Evar",
            "name": name,
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Etempvar(id, ty) => json!({
            "tag": "Etempvar",
            "id": *id,
            "name": resolve_name(*id, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Ederef(inner, ty) => json!({
            "tag": "Ederef",
            "expr": serialize_expr(inner, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Eaddrof(inner, ty) => json!({
            "tag": "Eaddrof",
            "expr": serialize_expr(inner, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Eunop(op, inner, ty) => json!({
            "tag": "Eunop",
            "op": serialize_unop(op),
            "expr": serialize_expr(inner, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Ebinop(op, lhs, rhs, ty) => json!({
            "tag": "Ebinop",
            "op": serialize_binop(op),
            "lhs": serialize_expr(lhs, id_to_name),
            "rhs": serialize_expr(rhs, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Ecast(inner, ty) => json!({
            "tag": "Ecast",
            "expr": serialize_expr(inner, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Efield(base, field_id, ty) => json!({
            "tag": "Efield",
            "base": serialize_expr(base, id_to_name),
            "field": *field_id,
            "field_name": resolve_name(*field_id, id_to_name),
            "ty": serialize_clight_type(ty),
        }),

        ClightExpr::Esizeof(ty1, ty2) => json!({
            "tag": "Esizeof",
            "sizeof_ty": serialize_clight_type(ty1),
            "ty": serialize_clight_type(ty2),
        }),

        ClightExpr::Ealignof(ty1, ty2) => json!({
            "tag": "Ealignof",
            "alignof_ty": serialize_clight_type(ty1),
            "ty": serialize_clight_type(ty2),
        }),

        ClightExpr::Econdition(cond, then_e, else_e, ty) => json!({
            "tag": "Econdition",
            "cond": serialize_expr(cond, id_to_name),
            "then": serialize_expr(then_e, id_to_name),
            "else": serialize_expr(else_e, id_to_name),
            "ty": serialize_clight_type(ty),
        }),
    }
}

fn serialize_clight_type(ty: &ClightType) -> Value {
    match ty {
        ClightType::Tvoid => json!({"tag": "Tvoid"}),

        ClightType::Tint(size, sign, attr) => json!({
            "tag": "Tint",
            "size": match size {
                ClightIntSize::I8 => "I8",
                ClightIntSize::I16 => "I16",
                ClightIntSize::I32 => "I32",
                ClightIntSize::IBool => "IBool",
            },
            "sign": match sign {
                ClightSignedness::Signed => "Signed",
                ClightSignedness::Unsigned => "Unsigned",
            },
            "attr": serialize_attr(attr),
        }),

        ClightType::Tlong(sign, attr) => json!({
            "tag": "Tlong",
            "sign": match sign {
                ClightSignedness::Signed => "Signed",
                ClightSignedness::Unsigned => "Unsigned",
            },
            "attr": serialize_attr(attr),
        }),

        ClightType::Tint128(sign, attr) => json!({
            "tag": "Tint128",
            "sign": match sign {
                ClightSignedness::Signed => "Signed",
                ClightSignedness::Unsigned => "Unsigned",
            },
            "attr": serialize_attr(attr),
        }),

        ClightType::Tfloat(size, attr) => json!({
            "tag": "Tfloat",
            "size": match size {
                ClightFloatSize::F32 => "F32",
                ClightFloatSize::F64 => "F64",
            },
            "attr": serialize_attr(attr),
        }),

        ClightType::Tpointer(inner, attr) => json!({
            "tag": "Tpointer",
            "inner": serialize_clight_type(inner),
            "attr": serialize_attr(attr),
        }),

        ClightType::Tarray(elem, size, attr) => json!({
            "tag": "Tarray",
            "elem": serialize_clight_type(elem),
            "size": *size,
            "attr": serialize_attr(attr),
        }),

        ClightType::Tfunction(param_types, ret_type, cc) => json!({
            "tag": "Tfunction",
            "params": param_types.iter().map(serialize_clight_type).collect::<Vec<_>>(),
            "return": serialize_clight_type(ret_type),
            "cc": serialize_callconv(cc),
        }),

        ClightType::Tstruct(id, attr) => json!({
            "tag": "Tstruct",
            "id": *id,
            "attr": serialize_attr(attr),
        }),

        ClightType::Tunion(id, attr) => json!({
            "tag": "Tunion",
            "id": *id,
            "attr": serialize_attr(attr),
        }),
    }
}

fn serialize_attr(attr: &ClightAttr) -> Value {
    if !attr.attr_volatile && attr.attr_alignas.is_none() {
        return json!(null);
    }
    let mut obj = json!({});
    if attr.attr_volatile {
        obj["volatile"] = json!(true);
    }
    if let Some(align) = attr.attr_alignas {
        obj["alignas"] = json!(align);
    }
    obj
}

fn serialize_callconv(cc: &CallConv) -> Value {
    if cc.varargs.is_none() && !cc.unproto && !cc.structured_ret {
        return json!("cc_default");
    }
    json!({
        "varargs": cc.varargs,
        "unproto": cc.unproto,
        "structret": cc.structured_ret,
    })
}

fn serialize_unop(op: &ClightUnaryOp) -> &'static str {
    match op {
        ClightUnaryOp::Onotbool => "Onotbool",
        ClightUnaryOp::Onotint => "Onotint",
        ClightUnaryOp::Oneg => "Oneg",
        ClightUnaryOp::Oabsfloat => "Oabsfloat",
    }
}

fn serialize_binop(op: &ClightBinaryOp) -> &'static str {
    match op {
        ClightBinaryOp::Oadd => "Oadd",
        ClightBinaryOp::Osub => "Osub",
        ClightBinaryOp::Omul => "Omul",
        ClightBinaryOp::Odiv => "Odiv",
        ClightBinaryOp::Omod => "Omod",
        ClightBinaryOp::Oand => "Oand",
        ClightBinaryOp::Oor => "Oor",
        ClightBinaryOp::Oxor => "Oxor",
        ClightBinaryOp::Oshl => "Oshl",
        ClightBinaryOp::Oshr => "Oshr",
        ClightBinaryOp::Oeq => "Oeq",
        ClightBinaryOp::One => "One",
        ClightBinaryOp::Olt => "Olt",
        ClightBinaryOp::Ogt => "Ogt",
        ClightBinaryOp::Ole => "Ole",
        ClightBinaryOp::Oge => "Oge",
    }
}

fn serialize_param_type(pty: &ParamType) -> Value {
    match pty {
        ParamType::Pointer => json!({"tag": "Tpointer", "inner": {"tag": "Tvoid"}, "attr": null}),
        ParamType::StructPointer(id) => json!({
            "tag": "Tpointer",
            "inner": {"tag": "Tstruct", "id": *id, "attr": null},
            "attr": null,
        }),
        ParamType::Typed(xtype) => xtype_to_clight_type_json(xtype),
        ParamType::Integer => json!({"tag": "Tlong", "sign": "Unsigned", "attr": null}),
        ParamType::Unknown => json!({"tag": "Tlong", "sign": "Unsigned", "attr": null}),
    }
}

fn serialize_xtype(xtype: &XType) -> Value {
    json!(match xtype {
        XType::Xbool => "Xbool",
        XType::Xint8signed => "Xint8signed",
        XType::Xint8unsigned => "Xint8unsigned",
        XType::Xint16signed => "Xint16signed",
        XType::Xint16unsigned => "Xint16unsigned",
        XType::Xint => "Xint",
        XType::Xintunsigned => "Xintunsigned",
        XType::Xlong => "Xlong",
        XType::Xlongunsigned => "Xlongunsigned",
        XType::Xfloat => "Xfloat",
        XType::Xsingle => "Xsingle",
        XType::Xptr => "Xptr",
        XType::Xcharptr => "Xcharptr",
        XType::Xcharptrptr => "Xcharptrptr",
        XType::Xintptr => "Xintptr",
        XType::Xfloatptr => "Xfloatptr",
        XType::Xsingleptr => "Xsingleptr",
        XType::Xfuncptr => "Xfuncptr",
        XType::Xvoid => "Xvoid",
        XType::Xany32 => "Xany32",
        XType::Xany64 => "Xany64",
        XType::XstructPtr(_) => "Xptr",
    })
}

fn xtype_to_clight_type_json(xtype: &XType) -> Value {
    match xtype {
        XType::Xbool => json!({"tag": "Tint", "size": "IBool", "sign": "Unsigned", "attr": null}),
        XType::Xint8signed => json!({"tag": "Tint", "size": "I8", "sign": "Signed", "attr": null}),
        XType::Xint8unsigned => {
            json!({"tag": "Tint", "size": "I8", "sign": "Unsigned", "attr": null})
        }
        XType::Xint16signed => {
            json!({"tag": "Tint", "size": "I16", "sign": "Signed", "attr": null})
        }
        XType::Xint16unsigned => {
            json!({"tag": "Tint", "size": "I16", "sign": "Unsigned", "attr": null})
        }
        XType::Xint | XType::Xany32 => {
            json!({"tag": "Tint", "size": "I32", "sign": "Signed", "attr": null})
        }
        XType::Xintunsigned => {
            json!({"tag": "Tint", "size": "I32", "sign": "Unsigned", "attr": null})
        }
        XType::Xlong | XType::Xany64 => json!({"tag": "Tlong", "sign": "Signed", "attr": null}),
        XType::Xlongunsigned => json!({"tag": "Tlong", "sign": "Unsigned", "attr": null}),
        XType::Xfloat => json!({"tag": "Tfloat", "size": "F64", "attr": null}),
        XType::Xsingle => json!({"tag": "Tfloat", "size": "F32", "attr": null}),
        XType::Xptr
        | XType::Xcharptr
        | XType::Xcharptrptr
        | XType::Xintptr
        | XType::Xfloatptr
        | XType::Xsingleptr
        | XType::Xfuncptr => {
            json!({"tag": "Tpointer", "inner": {"tag": "Tvoid"}, "attr": null})
        }
        XType::XstructPtr(id) => json!({
            "tag": "Tpointer",
            "inner": {"tag": "Tstruct", "id": *id, "attr": null},
            "attr": null,
        }),
        XType::Xvoid => json!({"tag": "Tvoid"}),
    }
}

fn xtype_string_to_clight_type(s: &str) -> Value {
    match s {
        "int_IBool" => json!({"tag": "Tint", "size": "IBool", "sign": "Unsigned", "attr": null}),
        "int_I8" => json!({"tag": "Tint", "size": "I8", "sign": "Signed", "attr": null}),
        "int_I8_unsigned" => json!({"tag": "Tint", "size": "I8", "sign": "Unsigned", "attr": null}),
        "int_I16" => json!({"tag": "Tint", "size": "I16", "sign": "Signed", "attr": null}),
        "int_I16_unsigned" => {
            json!({"tag": "Tint", "size": "I16", "sign": "Unsigned", "attr": null})
        }
        "int_I32" => json!({"tag": "Tint", "size": "I32", "sign": "Signed", "attr": null}),
        "int_I32_unsigned" => {
            json!({"tag": "Tint", "size": "I32", "sign": "Unsigned", "attr": null})
        }
        "int_I64" => json!({"tag": "Tlong", "sign": "Signed", "attr": null}),
        "int_I64_unsigned" => json!({"tag": "Tlong", "sign": "Unsigned", "attr": null}),
        "float_F64" => json!({"tag": "Tfloat", "size": "F64", "attr": null}),
        "float_F32" => json!({"tag": "Tfloat", "size": "F32", "attr": null}),
        "ptr_I64" | "ptr_char" | "ptr_int" | "ptr_double" | "ptr_float" => {
            json!({"tag": "Tpointer", "inner": {"tag": "Tvoid"}, "attr": null})
        }
        "void" => json!({"tag": "Tvoid"}),
        s if s.starts_with("ptr_struct_") => {
            let id_str = s.trim_start_matches("ptr_struct_");
            let id = usize::from_str_radix(id_str, 16).unwrap_or(0);
            json!({
                "tag": "Tpointer",
                "inner": {"tag": "Tstruct", "id": id, "attr": null},
                "attr": null,
            })
        }
        _ => json!({"tag": "Tlong", "sign": "Unsigned", "attr": null}),
    }
}

fn serialize_external_function(ef: &ExternalFunction) -> Value {
    match ef {
        ExternalFunction::EFExternal(name, sig) => json!({
            "tag": "EF_external",
            "name": name.as_ref(),
            "sig": serialize_signature(sig),
        }),
        ExternalFunction::EFBuiltin(name, sig) => json!({
            "tag": "EF_builtin",
            "name": name.as_ref(),
            "sig": serialize_signature(sig),
        }),
        ExternalFunction::EFRuntime(name, sig) => json!({
            "tag": "EF_runtime",
            "name": name.as_ref(),
            "sig": serialize_signature(sig),
        }),
        ExternalFunction::EFVLoad(chunk) => json!({
            "tag": "EF_vload",
            "chunk": serialize_chunk(chunk),
        }),
        ExternalFunction::EFVStore(chunk) => json!({
            "tag": "EF_vstore",
            "chunk": serialize_chunk(chunk),
        }),
        ExternalFunction::EFMalloc => json!({"tag": "EF_malloc"}),
        ExternalFunction::EFFree => json!({"tag": "EF_free"}),
        ExternalFunction::EFMemcpy(sz, al) => json!({
            "tag": "EF_memcpy",
            "size": *sz,
            "align": *al,
        }),
        _ => json!({"tag": "EF_unknown"}),
    }
}

fn serialize_signature(sig: &Signature) -> Value {
    json!({
        "args": sig.sig_args.iter().map(serialize_xtype).collect::<Vec<_>>(),
        "res": serialize_xtype(&sig.sig_res),
        "cc": serialize_callconv(&sig.sig_cc),
    })
}

fn serialize_chunk(chunk: &MemoryChunk) -> &'static str {
    match chunk {
        MemoryChunk::MBool => "Mbool",
        MemoryChunk::MInt8Signed => "Mint8signed",
        MemoryChunk::MInt8Unsigned => "Mint8unsigned",
        MemoryChunk::MInt16Signed => "Mint16signed",
        MemoryChunk::MInt16Unsigned => "Mint16unsigned",
        MemoryChunk::MInt32 => "Mint32",
        MemoryChunk::MInt64 => "Mint64",
        MemoryChunk::MFloat32 => "Mfloat32",
        MemoryChunk::MFloat64 => "Mfloat64",
        MemoryChunk::MAny32 => "Many32",
        MemoryChunk::MAny64 => "Many64",
        MemoryChunk::Unknown => "Many64",
    }
}

/// Convert a FieldType (from struct recovery) to a Clight JSON type.
fn serialize_ctype_from_field(ty: &crate::decompile::passes::c_pass::types::CType) -> Value {
    use crate::decompile::passes::c_pass::types::{CType, FloatSize, IntSize, Signedness};
    match ty {
        CType::Void => json!({"tag": "Tvoid"}),
        CType::Bool => json!({"tag": "Tint", "size": "IBool", "sign": "Unsigned", "attr": null}),
        CType::Int(size, sign) => {
            let s = match size {
                IntSize::Char => "I8",
                IntSize::Short => "I16",
                IntSize::Int => "I32",
                IntSize::Long | IntSize::LongLong => {
                    return json!({
                        "tag": "Tlong",
                        "sign": match sign { Signedness::Signed => "Signed", Signedness::Unsigned => "Unsigned" },
                        "attr": null,
                    })
                }
                IntSize::Int128 => {
                    return json!({
                        "tag": "Tint128",
                        "sign": match sign { Signedness::Signed => "Signed", Signedness::Unsigned => "Unsigned" },
                        "attr": null,
                    })
                }
            };
            json!({
                "tag": "Tint",
                "size": s,
                "sign": match sign { Signedness::Signed => "Signed", Signedness::Unsigned => "Unsigned" },
                "attr": null,
            })
        }
        CType::Float(size) => json!({
            "tag": "Tfloat",
            "size": match size { FloatSize::Float => "F32", FloatSize::Double | FloatSize::LongDouble => "F64" },
            "attr": null,
        }),
        CType::Pointer(inner, _) => json!({
            "tag": "Tpointer",
            "inner": serialize_ctype_from_field(inner),
            "attr": null,
        }),
        CType::Array(elem, size) => json!({
            "tag": "Tarray",
            "elem": serialize_ctype_from_field(elem),
            "size": size.unwrap_or(0),
            "attr": null,
        }),
        CType::Struct(name) => json!({
            "tag": "Tstruct",
            "name": name,
            "attr": null,
        }),
        CType::Union(name) => json!({
            "tag": "Tunion",
            "name": name,
            "attr": null,
        }),
        CType::Function(ret, params, variadic, unprototyped) => json!({
            "tag": "Tfunction",
            "return": serialize_ctype_from_field(ret),
            "params": params.iter().map(serialize_ctype_from_field).collect::<Vec<_>>(),
            "cc": if *unprototyped {
                json!("cc_unproto")
            } else if *variadic {
                json!({"varargs": 0})
            } else {
                json!("cc_default")
            },
        }),
        CType::Enum(name) => {
            json!({"tag": "Tint", "size": "I32", "sign": "Signed", "attr": null, "enum": name})
        }
        CType::TypedefName(name) => {
            json!({"tag": "Tint", "size": "I32", "sign": "Signed", "attr": null, "typedef": name})
        }
        CType::Qualified(inner, _) => serialize_ctype_from_field(inner),
    }
}

/// Collect names of functions called via Scall/EvarSymbol in a statement tree.
fn collect_called_functions(
    stmt: &ClightStmt,
    names: &mut std::collections::HashSet<String>,
    id_to_name: &HashMap<usize, String>,
) {
    match stmt {
        ClightStmt::Scall(_, func_expr, _) => match func_expr {
            ClightExpr::EvarSymbol(name, _) => {
                names.insert(name.clone());
            }
            ClightExpr::Evar(id, _) => {
                names.insert(
                    id_to_name
                        .get(id)
                        .cloned()
                        .unwrap_or_else(|| id.to_string()),
                );
            }
            _ => {}
        },
        ClightStmt::Ssequence(stmts) => {
            for s in stmts {
                collect_called_functions(s, names, id_to_name);
            }
        }
        ClightStmt::Sifthenelse(_, s1, s2) => {
            collect_called_functions(s1, names, id_to_name);
            collect_called_functions(s2, names, id_to_name);
        }
        ClightStmt::Sloop(s1, s2) => {
            collect_called_functions(s1, names, id_to_name);
            collect_called_functions(s2, names, id_to_name);
        }
        ClightStmt::Slabel(_, s) => collect_called_functions(s, names, id_to_name),
        ClightStmt::Sswitch(_, cases) => {
            for (_, s) in cases {
                collect_called_functions(s, names, id_to_name);
            }
        }
        _ => {}
    }
}

fn resolve_name(id: usize, id_to_name: &HashMap<usize, String>) -> String {
    id_to_name
        .get(&id)
        .cloned()
        .unwrap_or_else(|| format!("_{}", id))
}
