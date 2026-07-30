//! Deterministic in-memory loader for native AMD64 COFF objects.
//!
//! A relocatable COFF object has section-relative symbol values, zero section
//! virtual addresses, and unapplied implicit-addend relocations.  Feeding that
//! representation directly to Capstone makes every section overlap at zero and
//! makes a zero-filled `REL32` call appear to call its fall-through address.
//! This module turns a private copy of the input into a linked-image view while
//! retaining an explicit map back to every original section offset.  The input
//! file on disk is never modified.

use std::collections::{BTreeMap, HashMap, HashSet};

use object::{
    Architecture, BinaryFormat, Object, ObjectSection, ObjectSymbol, RelocationFlags,
    RelocationKind, RelocationTarget, SectionIndex, SectionKind, SymbolKind, SymbolSection,
};
use serde::Serialize;

use crate::decompile::elevator::DecompileDB;

/// Kept below 4 GiB because a COFF section header stores its virtual address in
/// 32 bits.  All synthetic addresses remain within the signed REL32 window.
const COFF_IMAGE_BASE: u64 = 0x1000_0000;
const SECTION_GRANULARITY: u64 = 0x1000;
const EXTERNAL_STRIDE: u64 = 0x10;
pub const COFF_LOADER_ID: &str = "amd64-coff-image-v1";
pub const MANIFOLD_UPSTREAM_COMMIT: &str = "fc944d043895ab453ce55a3f63eb4f2a922c7502";

#[derive(Debug, Clone, Serialize)]
pub struct CoffSectionMap {
    pub index: usize,
    pub name: String,
    pub kind: String,
    pub original_file_offset: Option<u64>,
    pub original_offset_start: u64,
    pub original_offset_end: u64,
    pub mapped_va_start: u64,
    pub mapped_va_end: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoffFunctionMap {
    pub original_name: String,
    pub provider_name: String,
    pub section_index: usize,
    pub section_offset: u64,
    pub original_size: u64,
    pub manifold_size: u64,
    pub mapped_entry: u64,
    pub mapped_end: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoffSymbolMap {
    pub original_name: String,
    pub provider_name: String,
    pub kind: String,
    pub defined: bool,
    pub section_index: Option<usize>,
    pub section_offset: Option<u64>,
    pub mapped_address: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CoffExternalKind {
    Function,
    ImportPointer,
    Data,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoffExternalMap {
    pub original_name: String,
    pub provider_name: String,
    pub kind: CoffExternalKind,
    pub synthetic_address: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoffRelocationMap {
    pub section_index: usize,
    pub section_name: String,
    pub section_offset: u64,
    pub mapped_field_va: u64,
    pub relocation_type: String,
    pub width_bits: u8,
    pub target_original_name: String,
    pub target_mapped_address: u64,
    pub encoded_value: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CoffAddressMap {
    pub schema: &'static str,
    pub loader_id: &'static str,
    pub architecture: &'static str,
    pub image_base: u64,
    pub sections: Vec<CoffSectionMap>,
    pub functions: Vec<CoffFunctionMap>,
    pub symbols: Vec<CoffSymbolMap>,
    pub externs: Vec<CoffExternalMap>,
    pub relocations: Vec<CoffRelocationMap>,
}

impl CoffAddressMap {
    pub fn write_json(&self, path: &std::path::Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
            }
        }
        let data = serde_json::to_vec_pretty(self)
            .map_err(|e| format!("failed to serialize COFF address map: {e}"))?;
        std::fs::write(path, data)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))
    }
}

/// Loader state used after the patched private image has been reparsed.  All
/// externally visible association is keyed by mapped address plus original
/// name, never by function ordinal.
#[derive(Debug, Clone)]
pub struct CoffImage {
    pub address_map: CoffAddressMap,
    provider_names: HashMap<(u64, String), String>,
    function_sizes: HashMap<(u64, String), u64>,
}

impl CoffImage {
    pub fn provider_name<'a>(&'a self, mapped_address: u64, original: &str) -> Option<&'a str> {
        self.provider_names
            .get(&(mapped_address, original.to_string()))
            .map(String::as_str)
    }

    pub fn function_size(&self, mapped_address: u64, original: &str) -> Option<u64> {
        self.function_sizes
            .get(&(mapped_address, original.to_string()))
            .copied()
    }

    /// Add the synthetic side of undefined COFF symbols.  Defined symbols are
    /// already visible through the rebased object parser.
    pub fn load_synthetic_symbols(&self, db: &mut DecompileDB) {
        for ext in &self.address_map.externs {
            let name: &'static str = Box::leak(ext.provider_name.clone().into_boxed_str());
            let addr = ext.synthetic_address;
            db.rel_push("symbols", (addr, name, "Beg"));
            match ext.kind {
                CoffExternalKind::Function => {
                    // plt_block marks a named callable target as external, so
                    // function inference never emits a body for the synthetic
                    // address even though direct_call retains the exact edge.
                    db.rel_push("plt_block", (addr, name));
                    db.rel_push(
                        "symbol_table",
                        (addr, 0usize, "FUNC", "GLOBAL", "UNDEF", 0usize,
                         ".coff.extern", 0usize, name),
                    );
                }
                CoffExternalKind::ImportPointer => {
                    db.rel_push("pointer_to_external_symbol", (addr, name));
                    db.rel_push(
                        "symbol_table",
                        (addr, 8usize, "OBJECT", "GLOBAL", "UNDEF", 0usize,
                         ".coff.iat", 0usize, name),
                    );
                }
                CoffExternalKind::Data => {
                    db.rel_push(
                        "symbol_table",
                        (addr, 8usize, "OBJECT", "GLOBAL", "UNDEF", 0usize,
                         ".coff.extern", 0usize, name),
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct SectionPlan {
    index: usize,
    name: String,
    kind: SectionKind,
    original_address: u64,
    size: u64,
    file_offset: Option<u64>,
    mapped_address: u64,
    va_field_offset: usize,
}

#[derive(Debug, Clone)]
struct SymbolPlan {
    original_name: String,
    kind: SymbolKind,
    section: SymbolSection,
    original_address: u64,
    mapped_address: u64,
}

#[derive(Debug, Clone)]
struct ExternalPlan {
    original_name: String,
    provider_source_name: String,
    kind: CoffExternalKind,
    mapped_address: u64,
}

#[derive(Debug)]
struct BytePatch {
    offset: usize,
    bytes: Vec<u8>,
}

/// Prepare a private COFF image in place.  Non-COFF inputs are returned
/// unchanged.  The caller reparses `data` after this function returns.
pub fn prepare_image(data: &mut Vec<u8>) -> Result<Option<CoffImage>, String> {
    let (mut patches, image) = {
        let obj = object::File::parse(&data[..])
            .map_err(|e| format!("invalid object: {e}"))?;
        if obj.format() != BinaryFormat::Coff {
            return Ok(None);
        }
        if obj.architecture() != Architecture::X86_64 {
            return Err(format!(
                "unsupported COFF target {:?}; only native AMD64 COFF objects are supported",
                obj.architecture()
            ));
        }
        plan_image(&obj, &data[..])?
    };

    patches.sort_by_key(|p| p.offset);
    for patch in patches {
        let end = patch
            .offset
            .checked_add(patch.bytes.len())
            .ok_or("COFF patch offset overflow")?;
        let dst = data
            .get_mut(patch.offset..end)
            .ok_or_else(|| format!("COFF patch outside input at 0x{:x}", patch.offset))?;
        dst.copy_from_slice(&patch.bytes);
    }
    Ok(Some(image))
}

fn plan_image(obj: &object::File<'_>, data: &[u8]) -> Result<(Vec<BytePatch>, CoffImage), String> {
    let va_field_offsets = section_va_field_offsets(obj, data)?;
    let mut cursor = COFF_IMAGE_BASE;
    let mut sections = Vec::new();
    for section in obj.sections() {
        let alignment = section.align().max(SECTION_GRANULARITY);
        cursor = align_up(cursor, alignment)?;
        let size = section.size();
        let mapped_address = cursor;
        // Keep at least one byte of identity for empty sections, then place the
        // next section on a page boundary.  Runtime bytes never overlap.
        cursor = cursor
            .checked_add(size.max(1))
            .ok_or("COFF section layout overflow")?;
        if mapped_address > u32::MAX as u64 {
            return Err("COFF synthetic section address exceeds 32-bit header field".into());
        }
        let (file_offset, _) = section.file_range().unwrap_or((0, 0));
        sections.push(SectionPlan {
            index: section.index().0,
            name: section.name().unwrap_or("").to_string(),
            kind: section.kind(),
            original_address: section.address(),
            size,
            file_offset: section.file_range().map(|_| file_offset),
            mapped_address,
            va_field_offset: *va_field_offsets
                .get(&section.index().0)
                .ok_or_else(|| format!("missing COFF header for section {}", section.index().0))?,
        });
    }
    let section_by_index: HashMap<usize, &SectionPlan> =
        sections.iter().map(|s| (s.index, s)).collect();

    let mut symbols: HashMap<usize, SymbolPlan> = HashMap::new();
    for sym in obj.symbols() {
        let original_name = sym.name().unwrap_or("").to_string();
        let section = sym.section();
        let mapped_address = match section {
            SymbolSection::Section(idx) => {
                let sec = section_by_index
                    .get(&idx.0)
                    .ok_or_else(|| format!("symbol {original_name:?} has invalid section {}", idx.0))?;
                let offset = sym.address().checked_sub(sec.original_address).ok_or_else(|| {
                    format!("symbol {original_name:?} lies before section {}", idx.0)
                })?;
                sec.mapped_address
                    .checked_add(offset)
                    .ok_or("COFF symbol address overflow")?
            }
            SymbolSection::Absolute => sym.address(),
            _ => 0,
        };
        symbols.insert(
            sym.index().0,
            SymbolPlan {
                original_name,
                kind: sym.kind(),
                section,
                original_address: sym.address(),
                mapped_address,
            },
        );
    }

    // Classify undefined symbols from both their COFF type and their use.  Some
    // producers omit the function type, but an E8/E9 REL32 field is definitive.
    let mut undefined_kinds: BTreeMap<String, CoffExternalKind> = BTreeMap::new();
    for sym in symbols.values() {
        if !matches!(sym.section, SymbolSection::Undefined | SymbolSection::Common) {
            continue;
        }
        if sym.original_name.is_empty() {
            continue;
        }
        let kind = if sym.original_name.starts_with("__imp_") {
            CoffExternalKind::ImportPointer
        } else if sym.kind == SymbolKind::Text {
            CoffExternalKind::Function
        } else {
            CoffExternalKind::Data
        };
        merge_external_kind(&mut undefined_kinds, &sym.original_name, kind);
    }
    for section in obj.sections() {
        let sec_data = section.data().unwrap_or(&[]);
        for (offset, reloc) in section.relocations() {
            if reloc.kind() != RelocationKind::Relative || reloc.size() != 32 || offset == 0 {
                continue;
            }
            let direct_transfer = sec_data
                .get(offset as usize - 1)
                .map_or(false, |b| *b == 0xe8 || *b == 0xe9);
            if !direct_transfer {
                continue;
            }
            let RelocationTarget::Symbol(index) = reloc.target() else { continue };
            if let Some(sym) = symbols.get(&index.0) {
                if matches!(sym.section, SymbolSection::Undefined | SymbolSection::Common)
                    && !sym.original_name.starts_with("__imp_")
                {
                    merge_external_kind(
                        &mut undefined_kinds,
                        &sym.original_name,
                        CoffExternalKind::Function,
                    );
                }
            }
        }
    }

    let mut external_cursor = align_up(cursor, SECTION_GRANULARITY)?;
    let mut externals = Vec::new();
    for (original_name, kind) in undefined_kinds {
        let provider_base_name = original_name
            .strip_prefix("__imp_")
            .unwrap_or(&original_name)
            .to_string();
        let provider_source_name = match kind {
            CoffExternalKind::Function | CoffExternalKind::ImportPointer => {
                format!("coff_ext_{provider_base_name}")
            }
            CoffExternalKind::Data => format!("coff_data_ext_{provider_base_name}"),
        };
        externals.push(ExternalPlan {
            original_name,
            provider_source_name,
            kind,
            mapped_address: external_cursor,
        });
        external_cursor = external_cursor
            .checked_add(EXTERNAL_STRIDE)
            .ok_or("COFF external layout overflow")?;
    }
    if external_cursor > u32::MAX as u64 {
        return Err("COFF synthetic extern address exceeds 32-bit layout".into());
    }
    let external_by_name: HashMap<&str, &ExternalPlan> = externals
        .iter()
        .map(|e| (e.original_name.as_str(), e))
        .collect();

    // Allocate one injective, downstream-idempotent C provider name per symbol
    // identity.  Every name starts in an explicit COFF namespace so Clight's
    // generic `.`, `__`, and `FUN_` filters cannot discard an authoritative
    // Windows symbol.  Address suffixes resolve sanitizer collisions.
    let mut name_seeds: Vec<(u64, String, String)> = symbols
        .values()
        .filter(|s| {
            s.mapped_address != 0
                && !s.original_name.is_empty()
                && !matches!(s.kind, SymbolKind::File | SymbolKind::Section)
        })
        .map(|s| {
            (
                s.mapped_address,
                s.original_name.clone(),
                defined_provider_source_name(s),
            )
        })
        .collect();
    name_seeds.extend(
        externals
            .iter()
            .map(|e| {
                (
                    e.mapped_address,
                    e.original_name.clone(),
                    e.provider_source_name.clone(),
                )
            }),
    );
    name_seeds.sort();
    name_seeds.dedup();
    let mut provider_by_identity: HashMap<(u64, String), String> = HashMap::new();
    let mut used_provider: HashSet<String> = HashSet::new();
    for (address, original, provider_source) in name_seeds {
        let base = c_safe_name(&provider_source);
        let mut candidate = base.clone();
        if used_provider.contains(&candidate) {
            candidate = format!("{}_coff_{:x}", base, address);
            let mut collision = 2usize;
            while used_provider.contains(&candidate) {
                candidate = format!("{}_coff_{:x}_{}", base, address, collision);
                collision += 1;
            }
        }
        used_provider.insert(candidate.clone());
        provider_by_identity.insert((address, original), candidate);
    }

    let mut function_rows: Vec<(usize, u64, String, String)> = Vec::new();
    for sym in symbols.values() {
        let SymbolSection::Section(section_index) = sym.section else { continue };
        if sym.kind != SymbolKind::Text || sym.original_name.is_empty() {
            continue;
        }
        let provider = provider_by_identity
            .get(&(sym.mapped_address, sym.original_name.clone()))
            .cloned()
            .ok_or_else(|| format!("missing provider for function {:?}", sym.original_name))?;
        function_rows.push((
            section_index.0,
            sym.mapped_address,
            sym.original_name.clone(),
            provider,
        ));
    }
    function_rows.sort();

    let mut functions = Vec::new();
    let mut function_sizes = HashMap::new();
    for (section_index, mapped_entry, original_name, provider_name) in &function_rows {
        let sec = section_by_index
            .get(section_index)
            .ok_or_else(|| format!("missing function section {section_index}"))?;
        let next = function_rows
            .iter()
            .filter(|(idx, addr, _, _)| idx == section_index && addr > mapped_entry)
            .map(|(_, addr, _, _)| *addr)
            .min()
            .unwrap_or_else(|| sec.mapped_address.saturating_add(sec.size));
        let (mapped_end, size) = infer_function_extent(
            sec.mapped_address,
            sec.size,
            *mapped_entry,
            Some(next),
        )
        .map_err(|e| format!("invalid COFF function {original_name:?}: {e}"))?;
        functions.push(CoffFunctionMap {
            original_name: original_name.clone(),
            provider_name: provider_name.clone(),
            section_index: *section_index,
            section_offset: mapped_entry - sec.mapped_address,
            original_size: size,
            manifold_size: size,
            mapped_entry: *mapped_entry,
            mapped_end,
        });
        function_sizes.insert((*mapped_entry, original_name.clone()), size);
    }

    let mut external_maps = Vec::new();
    for ext in &externals {
        let provider_name = provider_by_identity
            .get(&(ext.mapped_address, ext.original_name.clone()))
            .cloned()
            .ok_or_else(|| format!("missing provider for external {:?}", ext.original_name))?;
        external_maps.push(CoffExternalMap {
            original_name: ext.original_name.clone(),
            provider_name,
            kind: ext.kind,
            synthetic_address: ext.mapped_address,
        });
    }

    let mut symbol_maps = Vec::new();
    for sym in symbols.values() {
        if sym.mapped_address == 0
            || sym.original_name.is_empty()
            || matches!(sym.kind, SymbolKind::File | SymbolKind::Section)
        {
            continue;
        }
        let (defined, section_index, section_offset) = match sym.section {
            SymbolSection::Section(idx) => {
                let sec = section_by_index
                    .get(&idx.0)
                    .ok_or_else(|| format!("missing symbol section {}", idx.0))?;
                (
                    true,
                    Some(idx.0),
                    Some(sym.mapped_address - sec.mapped_address),
                )
            }
            SymbolSection::Absolute => (true, None, None),
            SymbolSection::Undefined | SymbolSection::Common => continue,
            _ => continue,
        };
        let provider_name = provider_by_identity
            .get(&(sym.mapped_address, sym.original_name.clone()))
            .cloned()
            .ok_or_else(|| format!("missing provider for symbol {:?}", sym.original_name))?;
        symbol_maps.push(CoffSymbolMap {
            original_name: sym.original_name.clone(),
            provider_name,
            kind: defined_symbol_kind_name(sym.kind).to_string(),
            defined,
            section_index,
            section_offset,
            mapped_address: sym.mapped_address,
        });
    }
    for ext in &external_maps {
        symbol_maps.push(CoffSymbolMap {
            original_name: ext.original_name.clone(),
            provider_name: ext.provider_name.clone(),
            kind: external_kind_name(ext.kind).to_string(),
            defined: false,
            section_index: None,
            section_offset: None,
            mapped_address: ext.synthetic_address,
        });
    }

    let mut patches: Vec<BytePatch> = sections
        .iter()
        .map(|section| BytePatch {
            offset: section.va_field_offset,
            bytes: (section.mapped_address as u32).to_le_bytes().to_vec(),
        })
        .collect();

    let mut relocation_maps = Vec::new();
    for section in obj.sections() {
        let sec = section_by_index
            .get(&section.index().0)
            .ok_or_else(|| format!("missing relocation section {}", section.index().0))?;
        let Some((raw_file_offset, raw_file_size)) = section.file_range() else {
            if section.relocations().next().is_some() {
                return Err(format!("section {} has relocations but no raw bytes", sec.name));
            }
            continue;
        };
        for (offset, reloc) in section.relocations() {
            let RelocationTarget::Symbol(symbol_index) = reloc.target() else {
                return Err(format!(
                    "unsupported non-symbol COFF relocation in section {} at +0x{offset:x}",
                    sec.name
                ));
            };
            let target = symbols.get(&symbol_index.0).ok_or_else(|| {
                format!("COFF relocation references missing symbol {}", symbol_index.0)
            })?;
            let (target_address, target_section, target_section_offset) = match target.section {
                SymbolSection::Section(idx) => {
                    let target_sec = section_by_index
                        .get(&idx.0)
                        .ok_or_else(|| format!("missing target section {}", idx.0))?;
                    (
                        target.mapped_address,
                        Some(idx.0),
                        Some(target.mapped_address - target_sec.mapped_address),
                    )
                }
                SymbolSection::Absolute => (target.original_address, None, None),
                SymbolSection::Undefined | SymbolSection::Common => {
                    let ext = external_by_name.get(target.original_name.as_str()).ok_or_else(|| {
                        format!("missing synthetic target for {:?}", target.original_name)
                    })?;
                    (ext.mapped_address, None, None)
                }
                _ => {
                    return Err(format!(
                        "unsupported target section {:?} for symbol {:?}",
                        target.section, target.original_name
                    ));
                }
            };
            let width = reloc.size();
            if !matches!(width, 7 | 16 | 32 | 64) {
                return Err(format!(
                    "unsupported COFF relocation width {width} for {:?}",
                    target.original_name
                ));
            }
            let byte_width = if width == 7 { 1 } else { (width / 8) as usize };
            if offset + byte_width as u64 > raw_file_size {
                return Err(format!(
                    "COFF relocation in {} at +0x{offset:x} exceeds section bytes",
                    sec.name
                ));
            }
            let file_offset = raw_file_offset
                .checked_add(offset)
                .ok_or("COFF relocation file offset overflow")? as usize;
            let raw_bytes = data
                .get(file_offset..file_offset + byte_width)
                .ok_or("COFF relocation field outside input")?;
            let raw_unsigned = read_unsigned(raw_bytes);
            let place = sec.mapped_address + offset;
            let value = relocation_value(
                reloc.kind(),
                width,
                raw_unsigned,
                reloc.addend(),
                place,
                target_address,
                target_section,
                target_section_offset,
            )?;
            let encoded = encode_relocation_value(reloc.kind(), width, value, raw_unsigned)?;
            let encoded_bytes = encoded.to_le_bytes();
            patches.push(BytePatch {
                offset: file_offset,
                bytes: encoded_bytes[..byte_width].to_vec(),
            });
            relocation_maps.push(CoffRelocationMap {
                section_index: sec.index,
                section_name: sec.name.clone(),
                section_offset: offset,
                mapped_field_va: place,
                relocation_type: relocation_name(reloc.flags(), reloc.kind(), reloc.addend()),
                width_bits: width,
                target_original_name: target.original_name.clone(),
                target_mapped_address: target_address,
                encoded_value: encoded,
            });
        }
    }

    let section_maps = sections
        .iter()
        .map(|s| CoffSectionMap {
            index: s.index,
            name: s.name.clone(),
            kind: format!("{:?}", s.kind),
            original_file_offset: s.file_offset,
            original_offset_start: 0,
            original_offset_end: s.size,
            mapped_va_start: s.mapped_address,
            mapped_va_end: s.mapped_address + s.size,
        })
        .collect();
    functions.sort_by_key(|f| (f.mapped_entry, f.original_name.clone()));
    symbol_maps.sort_by_key(|s| (s.mapped_address, s.original_name.clone()));
    external_maps.sort_by_key(|e| e.synthetic_address);
    relocation_maps.sort_by_key(|r| (r.section_index, r.section_offset));

    let address_map = CoffAddressMap {
        schema: "manifold.coff-address-map.v1",
        loader_id: COFF_LOADER_ID,
        architecture: "x86_64-pc-windows-msvc",
        image_base: COFF_IMAGE_BASE,
        sections: section_maps,
        functions,
        symbols: symbol_maps,
        externs: external_maps,
        relocations: relocation_maps,
    };
    Ok((
        patches,
        CoffImage {
            address_map,
            provider_names: provider_by_identity,
            function_sizes,
        },
    ))
}

fn merge_external_kind(
    kinds: &mut BTreeMap<String, CoffExternalKind>,
    name: &str,
    kind: CoffExternalKind,
) {
    kinds
        .entry(name.to_string())
        .and_modify(|existing| {
            if kind == CoffExternalKind::ImportPointer
                || (kind == CoffExternalKind::Function && *existing == CoffExternalKind::Data)
            {
                *existing = kind;
            }
        })
        .or_insert(kind);
}

fn defined_provider_source_name(sym: &SymbolPlan) -> String {
    let prefix = match sym.kind {
        SymbolKind::Text => "coff_fn_",
        SymbolKind::Data | SymbolKind::Tls => "coff_data_",
        _ => "coff_sym_",
    };
    format!("{prefix}{}", sym.original_name)
}

fn defined_symbol_kind_name(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Text => "function",
        SymbolKind::Data => "data",
        SymbolKind::Tls => "tls",
        SymbolKind::Label => "label",
        _ => "symbol",
    }
}

fn external_kind_name(kind: CoffExternalKind) -> &'static str {
    match kind {
        CoffExternalKind::Function => "function",
        CoffExternalKind::ImportPointer => "import_pointer",
        CoffExternalKind::Data => "data",
    }
}

fn infer_function_extent(
    section_start: u64,
    section_size: u64,
    entry: u64,
    next_entry: Option<u64>,
) -> Result<(u64, u64), String> {
    let section_end = section_start
        .checked_add(section_size)
        .ok_or("section extent overflow")?;
    if section_size == 0 || entry < section_start || entry >= section_end {
        return Err(format!(
            "entry 0x{entry:x} is outside nonempty section [0x{section_start:x}, 0x{section_end:x})"
        ));
    }
    let end = next_entry.unwrap_or(section_end);
    if end <= entry || end > section_end {
        return Err(format!(
            "inferred end 0x{end:x} does not form a positive in-section extent from 0x{entry:x}"
        ));
    }
    Ok((end, end - entry))
}

fn c_safe_name(name: &str) -> String {
    crate::decompile::passes::c_pass::convert::from_relations::sanitize_c_symbol_name(name)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, String> {
    if alignment == 0 {
        return Ok(value);
    }
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| "COFF alignment overflow".to_string())
    }
}

fn section_va_field_offsets(
    obj: &object::File<'_>,
    data: &[u8],
) -> Result<HashMap<usize, usize>, String> {
    let base = data.as_ptr() as usize;
    let mut result = HashMap::new();
    match obj {
        object::File::Coff(file) => {
            for section in file.sections() {
                let ptr = &section.coff_section().virtual_address as *const _ as usize;
                let offset = ptr
                    .checked_sub(base)
                    .ok_or("COFF section header is outside input")?;
                result.insert(section.index().0, offset);
            }
        }
        object::File::CoffBig(file) => {
            for section in file.sections() {
                let ptr = &section.coff_section().virtual_address as *const _ as usize;
                let offset = ptr
                    .checked_sub(base)
                    .ok_or("bigobj section header is outside input")?;
                result.insert(section.index().0, offset);
            }
        }
        _ => return Err("internal error: COFF section offsets requested for non-COFF input".into()),
    }
    Ok(result)
}

fn read_unsigned(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

fn sign_extend(value: u64, bits: u8) -> i128 {
    let shift = 128 - bits as u32;
    ((value as i128) << shift) >> shift
}

fn relocation_value(
    kind: RelocationKind,
    width: u8,
    raw: u64,
    explicit_addend: i64,
    place: u64,
    target: u64,
    target_section: Option<usize>,
    target_section_offset: Option<u64>,
) -> Result<i128, String> {
    let addend = explicit_addend as i128;
    match kind {
        RelocationKind::Relative => Ok(
            target as i128 + sign_extend(raw, width) + addend - place as i128,
        ),
        RelocationKind::Absolute => {
            Ok(target as i128 + sign_extend(raw, width) + addend)
        }
        RelocationKind::ImageOffset => {
            Ok(target as i128 - COFF_IMAGE_BASE as i128 + sign_extend(raw, width) + addend)
        }
        RelocationKind::SectionIndex => {
            let index = target_section.ok_or("SECTION relocation targets no COFF section")?;
            Ok(index as i128 + raw as i128 + addend)
        }
        RelocationKind::SectionOffset => {
            let offset = target_section_offset
                .ok_or("SECREL relocation targets no COFF section")?;
            Ok(offset as i128 + sign_extend(raw, width) + addend)
        }
        other => Err(format!("unsupported AMD64 COFF relocation kind {other:?}")),
    }
}

fn encode_relocation_value(
    kind: RelocationKind,
    width: u8,
    value: i128,
    original: u64,
) -> Result<u64, String> {
    if kind == RelocationKind::Relative {
        let min = -(1i128 << (width - 1));
        let max = (1i128 << (width - 1)) - 1;
        if value < min || value > max {
            return Err(format!("COFF relative relocation value {value} does not fit i{width}"));
        }
    } else {
        let max = (1i128 << width) - 1;
        if value < 0 || value > max {
            return Err(format!("COFF relocation value {value} does not fit u{width}"));
        }
    }
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let encoded = (value as u128 as u64) & mask;
    if width == 7 {
        // AMD64 SECREL7 occupies only the low seven bits of its byte.
        Ok((original & 0x80) | encoded)
    } else {
        Ok(encoded)
    }
}

fn relocation_name(flags: RelocationFlags, kind: RelocationKind, addend: i64) -> String {
    if let RelocationFlags::Coff { typ } = flags {
        use object::pe::*;
        return match typ {
            IMAGE_REL_AMD64_ADDR64 => "IMAGE_REL_AMD64_ADDR64",
            IMAGE_REL_AMD64_ADDR32 => "IMAGE_REL_AMD64_ADDR32",
            IMAGE_REL_AMD64_ADDR32NB => "IMAGE_REL_AMD64_ADDR32NB",
            IMAGE_REL_AMD64_REL32 => "IMAGE_REL_AMD64_REL32",
            IMAGE_REL_AMD64_REL32_1 => "IMAGE_REL_AMD64_REL32_1",
            IMAGE_REL_AMD64_REL32_2 => "IMAGE_REL_AMD64_REL32_2",
            IMAGE_REL_AMD64_REL32_3 => "IMAGE_REL_AMD64_REL32_3",
            IMAGE_REL_AMD64_REL32_4 => "IMAGE_REL_AMD64_REL32_4",
            IMAGE_REL_AMD64_REL32_5 => "IMAGE_REL_AMD64_REL32_5",
            IMAGE_REL_AMD64_SECTION => "IMAGE_REL_AMD64_SECTION",
            IMAGE_REL_AMD64_SECREL => "IMAGE_REL_AMD64_SECREL",
            IMAGE_REL_AMD64_SECREL7 => "IMAGE_REL_AMD64_SECREL7",
            _ => return format!("COFF_{typ:#x}_{kind:?}_ADDEND_{addend}"),
        }
        .to_string();
    }
    format!("{kind:?}_ADDEND_{addend}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel32_external_call_uses_field_address_and_does_not_call_fallthrough() {
        let place = 0x1000_0029;
        let target = 0x1000_2000;
        let displacement = relocation_value(
            RelocationKind::Relative,
            32,
            0,
            -4,
            place,
            target,
            None,
            None,
        )
        .unwrap();
        assert_eq!(displacement, target as i128 - (place + 4) as i128);
        assert_ne!(displacement, 0, "an unapplied relocation calls fall-through");
    }

    #[test]
    fn rel32_4_defined_data_reference_resolves_to_mapped_data_section() {
        let place = 0x1000_0040;
        let data_symbol = 0x1000_1020;
        assert_eq!(
            relocation_value(
                RelocationKind::Relative,
                32,
                0,
                -8,
                place,
                data_symbol,
                Some(2),
                Some(0x20),
            )
            .unwrap(),
            data_symbol as i128 - (place + 8) as i128
        );
    }

    #[test]
    fn provider_namespace_survives_generic_clight_name_filters() {
        let sym = SymbolPlan {
            original_name: "__FUN_.hidden".to_string(),
            kind: SymbolKind::Text,
            section: SymbolSection::Section(SectionIndex(1)),
            original_address: 0,
            mapped_address: COFF_IMAGE_BASE,
        };
        let provider = c_safe_name(&defined_provider_source_name(&sym));
        assert!(provider.starts_with("coff_fn_"));
        assert!(!provider.starts_with("__"));
        assert!(!provider.starts_with("FUN_"));
        assert!(!provider.starts_with('.'));
    }

    #[test]
    fn function_extents_are_positive_and_zero_extent_is_rejected() {
        assert_eq!(
            infer_function_extent(0x1000, 0x40, 0x1010, Some(0x1030)).unwrap(),
            (0x1030, 0x20)
        );
        assert!(infer_function_extent(0x1000, 0x40, 0x1040, None).is_err());
        assert!(infer_function_extent(0x1000, 0, 0x1000, None).is_err());
        assert!(infer_function_extent(0x1000, 0x40, 0x1020, Some(0x1020)).is_err());
    }

    #[test]
    fn v1_function_schema_has_only_explicit_original_and_manifold_sizes() {
        let row = CoffFunctionMap {
            original_name: "f".to_string(),
            provider_name: "coff_fn_f".to_string(),
            section_index: 1,
            section_offset: 0,
            original_size: 4,
            manifold_size: 4,
            mapped_entry: COFF_IMAGE_BASE,
            mapped_end: COFF_IMAGE_BASE + 4,
        };
        let value = serde_json::to_value(row).unwrap();
        let object = value.as_object().unwrap();
        assert_eq!(object.get("original_size").unwrap(), 4);
        assert_eq!(object.get("manifold_size").unwrap(), 4);
        assert!(!object.contains_key("size"));
    }

    #[test]
    fn addr_and_section_relocations_use_internal_image_coordinates() {
        let target = COFF_IMAGE_BASE + 0x2345;
        assert_eq!(
            relocation_value(
                RelocationKind::ImageOffset,
                32,
                3,
                0,
                0,
                target,
                Some(2),
                Some(0x345),
            )
            .unwrap(),
            0x2348
        );
        assert_eq!(
            relocation_value(
                RelocationKind::SectionIndex,
                16,
                0,
                0,
                0,
                target,
                Some(2),
                Some(0x345),
            )
            .unwrap(),
            2
        );
        assert_eq!(
            relocation_value(
                RelocationKind::SectionOffset,
                32,
                5,
                0,
                0,
                target,
                Some(2),
                Some(0x345),
            )
            .unwrap(),
            0x34a
        );
    }

    #[test]
    fn relative_encoding_rejects_out_of_range_targets() {
        assert!(encode_relocation_value(RelocationKind::Relative, 32, 1i128 << 31, 0).is_err());
        assert_eq!(
            encode_relocation_value(RelocationKind::Relative, 32, -4, 0).unwrap(),
            0xffff_fffc
        );
    }
}
