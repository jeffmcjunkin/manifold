//! Native AMD64 PE metadata ingestion: IAT slot names, exports, and exception-directory extents, at preferred virtual addresses.

use std::mem;

use object::read::pe::{ImageNtHeaders, ImageOptionalHeader};
use object::{LittleEndian as LE, Object, ObjectSection};

use crate::decompile::elevator::DecompileDB;
use crate::x86::types::{Address, Symbol};

const UNW_FLAG_CHAININFO: u8 = 0x4;

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Reject PE variants the current x86-64 pipeline does not model; non-PE inputs are a no-op.
pub fn validate_native_pe(obj: &object::File<'_>) -> Result<(), String> {
    match obj {
        object::File::Pe64(pe) => {
            if pe
                .data_directory(object::pe::IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR)
                .is_some()
            {
                return Err("managed CLR images are not native AMD64 PE binaries".to_string());
            }
            Ok(())
        }
        object::File::Pe32(_) => {
            Err("PE32 is not supported; only native AMD64 PE32+ images are supported".to_string())
        }
        _ => Ok(()),
    }
}

/// Load PE-only metadata and return block leaders contributed by it.
pub fn load_metadata(db: &mut DecompileDB, obj: &object::File<'_>) -> Result<Vec<Address>, String> {
    let pe = match obj {
        object::File::Pe64(pe) => pe,
        _ => return Ok(Vec::new()),
    };

    let image_base = pe.nt_headers().optional_header().image_base();
    let mut leaders = load_runtime_functions(db, pe, image_base)?;
    load_imports(db, pe, image_base)?;

    let code_ranges: Vec<(u64, u64)> = obj
        .sections()
        .filter(|s| s.kind() == object::SectionKind::Text)
        .map(|s| (s.address(), s.address().saturating_add(s.size())))
        .collect();

    for export in obj
        .exports()
        .map_err(|e| format!("invalid PE export table: {e}"))?
    {
        let addr = export.address();
        if !code_ranges
            .iter()
            .any(|(start, end)| addr >= *start && addr < *end)
        {
            continue;
        }
        let raw_name = String::from_utf8_lossy(export.name()).into_owned();
        if raw_name.is_empty() {
            continue;
        }
        let name: Symbol = leak(raw_name);
        db.rel_push("symbols", (addr, name, "Beg"));
        db.rel_push(
            "symbol_table",
            (
                addr, 0usize, "FUNC", "GLOBAL", "DEFAULT", 0usize, "", 0usize, name,
            ),
        );
        db.rel_push("known_function_entry", (addr,));
        leaders.push(addr);
    }

    // A PE with no entry yields the image base (the DOS header, not code), so text-section membership is the only sound test.
    let entry = obj.entry();
    if code_ranges
        .iter()
        .any(|(start, end)| entry >= *start && entry < *end)
    {
        db.rel_push("known_function_entry", (entry,));
        leaders.push(entry);
    }

    leaders.sort_unstable();
    leaders.dedup();
    Ok(leaders)
}

fn load_runtime_functions(
    db: &mut DecompileDB,
    pe: &object::read::pe::PeFile64<'_>,
    image_base: u64,
) -> Result<Vec<Address>, String> {
    let directory = match pe.data_directory(object::pe::IMAGE_DIRECTORY_ENTRY_EXCEPTION) {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };
    let data = directory
        .data(pe.data(), &pe.section_table())
        .map_err(|e| format!("invalid PE exception directory: {e}"))?;
    let entry_size = mem::size_of::<object::pe::ImageRuntimeFunctionEntry>();
    if data.len() % entry_size != 0 {
        return Err(format!(
            "exception directory size {} is not a multiple of RUNTIME_FUNCTION size {}",
            data.len(),
            entry_size
        ));
    }

    let sections = pe.section_table();
    let mut leaders = Vec::new();
    for raw in data.chunks_exact(entry_size) {
        let begin_rva = u32::from_le_bytes(raw[0..4].try_into().unwrap());
        let end_rva = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        let unwind_rva = u32::from_le_bytes(raw[8..12].try_into().unwrap()) & !0x3;
        if begin_rva == 0 || begin_rva >= end_rva {
            continue;
        }

        let begin = image_base
            .checked_add(u64::from(begin_rva))
            .ok_or_else(|| format!("PE runtime-function start RVA {begin_rva:#x} overflows"))?;
        let end = image_base
            .checked_add(u64::from(end_rva))
            .ok_or_else(|| format!("PE runtime-function end RVA {end_rva:#x} overflows"))?;
        db.rel_push("known_function_range", (begin, end));
        // Every fragment boundary helps the block builder, but only primary entries seed functions; CHAININFO fragments point back to the primary.
        leaders.push(begin);
        let chained = sections
            .pe_data_at(pe.data(), unwind_rva)
            .and_then(|bytes| bytes.first().copied())
            .map(|version_and_flags| ((version_and_flags >> 3) & 0x1f) & UNW_FLAG_CHAININFO != 0)
            .unwrap_or(false);
        if !chained {
            db.rel_push("known_function_entry", (begin,));
        }
    }

    log::debug!(
        "PE exception directory: {} runtime-function records",
        data.len() / entry_size
    );
    Ok(leaders)
}

fn load_imports(
    db: &mut DecompileDB,
    pe: &object::read::pe::PeFile64<'_>,
    image_base: u64,
) -> Result<(), String> {
    let table = match pe
        .import_table()
        .map_err(|e| format!("invalid PE import table: {e}"))?
    {
        Some(t) => t,
        None => return Ok(()),
    };
    let mut descriptors = table
        .descriptors()
        .map_err(|e| format!("invalid PE import descriptors: {e}"))?;
    let thunk_size = mem::size_of::<object::pe::ImageThunkData64>() as u64;
    let mut count = 0usize;

    while let Some(desc) = descriptors
        .next()
        .map_err(|e| format!("invalid PE import descriptor: {e}"))?
    {
        let library = table
            .name(desc.name.get(LE))
            .map_err(|e| format!("invalid PE import library name: {e}"))?;
        let library = String::from_utf8_lossy(library);
        let mut lookup_rva = desc.original_first_thunk.get(LE);
        if lookup_rva == 0 {
            lookup_rva = desc.first_thunk.get(LE);
        }
        let mut thunks = table
            .thunks(lookup_rva)
            .map_err(|e| format!("invalid PE import thunk list: {e}"))?;
        let first_thunk = u64::from(desc.first_thunk.get(LE));
        let mut index = 0u64;
        while let Some(thunk) = thunks
            .next::<object::pe::ImageNtHeaders64>()
            .map_err(|e| format!("invalid PE import thunk: {e}"))?
        {
            let name = match table
                .import::<object::pe::ImageNtHeaders64>(thunk)
                .map_err(|e| format!("invalid PE import name: {e}"))?
            {
                object::read::pe::Import::Name(_, bytes) => {
                    String::from_utf8_lossy(bytes).into_owned()
                }
                object::read::pe::Import::Ordinal(ordinal) => {
                    format!("{}_ordinal_{}", library.replace('.', "_"), ordinal)
                }
            };
            let thunk_offset = index
                .checked_mul(thunk_size)
                .and_then(|offset| first_thunk.checked_add(offset))
                .and_then(|rva| image_base.checked_add(rva))
                .ok_or_else(|| format!("PE IAT slot {index} address overflows"))?;
            let slot = thunk_offset;
            db.rel_push("pointer_to_external_symbol", (slot, leak(name)));
            count += 1;
            index += 1;
        }
    }

    log::debug!("PE imports: {} IAT slots", count);
    Ok(())
}
