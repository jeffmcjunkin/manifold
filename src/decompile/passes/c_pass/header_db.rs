// Single parsed view of header_functions.json, the one source of truth for system-header knowledge so the emitter's prototype suppression, renames, and #include selection can never drift.
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub struct HeaderDb {
    /// Every function name the headers declare (union of all groups). The prototype-suppression set.
    pub functions: HashSet<&'static str>,
    /// Name prefixes that are compiler-provided (e.g. "__builtin_").
    pub prefixes: Vec<&'static str>,
    /// header -> names, for include triggering (groups with "header": null are excluded).
    pub includes: Vec<(&'static str, Vec<&'static str>)>,
    /// Header-declared global VARIABLES -> canonical C type spelling ("char **", "int", ...).
    pub variables: HashMap<&'static str, &'static str>,
    /// Local definitions under a header-declared name whose body is known libc behavior: the definition is dropped and call sites bind to the header's declaration.
    pub local_def_filter: HashSet<&'static str>,
    /// Local definitions under a header-declared name that must be kept: emitted renamed (`<name>_local`). Colliding names in neither set default to this policy.
    pub local_def_decompile: HashSet<&'static str>,
    /// libc stdio FILE* globals (kind "stdio_file"): keep their real name AND are left undefined so <stdio.h>'s `extern FILE *stdout;` binds them to libc at link.
    pub stdio_globals: HashSet<&'static str>,
    /// libc globals that must keep their REAL name to bind to the library symbol (stdio FILE*s + getopt globals + environ): kinds "stdio_file" and "libc_extern".
    pub real_libc_globals: HashSet<&'static str>,
    /// Names that collide with a libc global and must be `_sym`-renamed when emitted as a program's own definition: kinds "stdio_file", "libc_extern", "reserved_rename".
    pub reserved_globals: HashSet<&'static str>,
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

pub fn header_db() -> &'static HeaderDb {
    static DB: OnceLock<HeaderDb> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = HeaderDb {
            functions: HashSet::new(),
            prefixes: Vec::new(),
            includes: Vec::new(),
            variables: HashMap::new(),
            local_def_filter: HashSet::new(),
            local_def_decompile: HashSet::new(),
            stdio_globals: HashSet::new(),
            real_libc_globals: HashSet::new(),
            reserved_globals: HashSet::new(),
        };
        let json_str = include_str!("../../../data/json/header_functions.json");
        let parsed: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("Failed to parse header_functions.json: {}", e);
                return db;
            }
        };
        // header -> accumulated names (several groups can share one header, e.g. stdio + stdio_unlocked, or a default group + its _GNU_SOURCE sibling).
        let mut by_header: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
        for (_group, spec) in parsed.get("groups").and_then(|g| g.as_object()).into_iter().flatten() {
            let names: Vec<&'static str> = spec
                .get("names")
                .and_then(|n| n.as_array())
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str())
                .map(leak)
                .collect();
            db.functions.extend(names.iter().copied());
            if let Some(h) = spec.get("header").and_then(|h| h.as_str()) {
                by_header.entry(leak(h)).or_default().extend(names.iter().copied());
            }
        }
        let mut headers: Vec<_> = by_header.into_iter().collect();
        headers.sort_by_key(|(h, _)| *h);
        db.includes = headers;
        // Header-declared / libc globals: a single table whose "kind" derives every per-global policy set the emitter needs, plus an optional canonical "type" spelling.
        for (name, spec) in parsed.get("globals").and_then(|v| v.as_object()).into_iter().flatten() {
            if !spec.is_object() {
                continue;
            }
            let name = leak(name);
            if let Some(t) = spec.get("type").and_then(|t| t.as_str()) {
                db.variables.insert(name, leak(t));
            }
            match spec.get("kind").and_then(|k| k.as_str()) {
                Some("stdio_file") => {
                    db.stdio_globals.insert(name);
                    db.real_libc_globals.insert(name);
                    db.reserved_globals.insert(name);
                }
                Some("libc_extern") => {
                    db.real_libc_globals.insert(name);
                    db.reserved_globals.insert(name);
                }
                Some("reserved_rename") => {
                    db.reserved_globals.insert(name);
                }
                _ => {}
            }
        }
        if let Some(ld) = parsed.get("local_definitions") {
            for (key, set) in [("filter", &mut db.local_def_filter), ("decompile", &mut db.local_def_decompile)] {
                for n in ld.get(key).and_then(|v| v.as_array()).into_iter().flatten() {
                    if let Some(s) = n.as_str() {
                        set.insert(leak(s));
                    }
                }
            }
        }
        for p in parsed.get("prefixes").and_then(|p| p.as_array()).into_iter().flatten() {
            if let Some(s) = p.as_str() {
                db.prefixes.push(leak(s));
            }
        }
        db
    })
}
