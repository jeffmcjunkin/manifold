use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::decompile::elevator::DecompileDB;
use manifold::decompile::passes::abi_pass::AbiPass;
use manifold::decompile::passes::pass::IRPass;
use manifold::abi::BinaryFormat;
use manifold::mreg::Mreg;
use manifold::x86::types::{Address, Symbol};

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") || !command_exists("lld-link") {
        eprintln!("skipping PE test: clang/lld-link unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!("manifold_pe_fixture_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dll_c = dir.join("imported.c");
    let exe_c = dir.join("fixture.c");
    let dll_obj = dir.join("imported.obj");
    let exe_obj = dir.join("fixture.obj");
    let dll = dir.join("imported.dll");
    let import_lib = dir.join("imported.lib");
    let exe = dir.join("fixture.exe");

    std::fs::write(
        &dll_c,
        "__declspec(dllexport) int Imported(int x) { return x + 11; }\n",
    )
    .unwrap();
    std::fs::write(
        &exe_c,
        r#"
__declspec(dllimport) int Imported(int);
int _fltused;
__declspec(dllexport) __declspec(noinline) int add_five(int a, int b, int c, int d, int e) {
    return a + b + c + d + e;
}
__declspec(dllexport) __declspec(noinline) int first_and_fifth(int a, int b, int c, int d, int e) {
    return a + e;
}
__declspec(dllexport) __declspec(noinline) double mixed_five(int a, double b, int c, double d, double e) {
    return a + b + c + d + e;
}
__declspec(dllexport) int exported(int x) {
    return add_five(x, 2, 3, 4, 5) + first_and_fifth(x, 20, 30, 40, 50) + Imported(x);
}
__declspec(dllexport) double exported_float(int x, double e) {
    return mixed_five(x, 2.5, 3, 4.5, e);
}
void entry(void) { (void)exported(7); }
"#,
    )
    .unwrap();

    let compile = |source: &Path, output: &Path| {
        Command::new("clang")
            .args(["--target=x86_64-pc-windows-msvc", "-O1", "-c"])
            .arg(source)
            .arg("-o")
            .arg(output)
            .status()
            .unwrap()
    };
    assert!(compile(&dll_c, &dll_obj).success());
    assert!(Command::new("lld-link")
        .args(["/dll", "/noentry", "/nodefaultlib"])
        .arg(format!("/out:{}", dll.display()))
        .arg(format!("/implib:{}", import_lib.display()))
        .arg(&dll_obj)
        .status()
        .unwrap()
        .success());
    assert!(compile(&exe_c, &exe_obj).success());
    assert!(Command::new("lld-link")
        .args(["/entry:entry", "/subsystem:console", "/nodefaultlib"])
        .arg(format!("/out:{}", exe.display()))
        .arg(&exe_obj)
        .arg(&import_lib)
        .status()
        .unwrap()
        .success());
    Some(exe)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
}

#[test]
fn pe_loader_recovers_iat_exports_unwind_and_win64_abi() {
    let Some(exe) = fixture() else { return };
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, exe);

    assert_eq!(db.abi().format, BinaryFormat::Pe);
    assert!(db
        .rel_iter::<(Address, Symbol)>("pointer_to_external_symbol")
        .any(|(_, name)| *name == "Imported"));
    assert!(db
        .rel_iter::<(Address, Symbol, Symbol)>("symbols")
        .any(|(_, name, _)| *name == "exported"));
    assert!(db
        .rel_iter::<(Address, Address)>("known_function_range")
        .any(|(start, end)| start < end));
    assert!(db
        .rel_iter::<(Address,)>("known_function_entry")
        .any(|(address,)| *address >= 0x1_0000_0000));

    AbiPass.run(&mut db);
    assert!(db
        .rel_iter::<(Mreg, usize)>("abi_int_arg_position")
        .any(|(reg, pos)| *reg == Mreg::CX && *pos == 0));
    assert!(db
        .rel_iter::<(Mreg, usize)>("abi_float_arg_position")
        .any(|(reg, pos)| *reg == Mreg::X2 && *pos == 2));
    assert!(db
        .rel_iter::<(i64,)>("abi_outgoing_stack_base")
        .any(|(v,)| *v == 32));
    assert!(db
        .rel_iter::<(i64,)>("abi_incoming_sp_stack_base")
        .any(|(v,)| *v == 40));
    assert!(db
        .rel_iter::<(Mreg,)>("is_callee_saved")
        .any(|(r,)| *r == Mreg::SI));

    // ABI state is per decompilation: loading an ELF immediately afterward
    // must not retain the Windows calling convention.
    if Path::new("/bin/true").exists() {
        let mut elf = DecompileDB::default();
        manifold::decompile::disassembly::load_from_binary(&mut elf, Path::new("/bin/true"));
        assert_eq!(elf.abi().format, BinaryFormat::Elf);
        assert_eq!(db.abi().format, BinaryFormat::Pe);
    }
}

#[test]
fn pe_pipeline_emits_export_and_named_import_call() {
    let Some(exe) = fixture() else { return };
    let exe = exe.to_path_buf();
    std::thread::Builder::new()
        .name("pe-pipeline".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let mut db = DecompileDB::default();
            manifold::decompile::disassembly::load_from_binary(&mut db, &exe);
            manifold::decompile::disassembly::load_preset(&mut db);
            db.run_pipeline(&exe, false, false);

            let tu = db
                .cast_optimized_translation_unit
                .as_ref()
                .expect("PE pipeline must emit a translation unit");
            let text = manifold::decompile::passes::c_pass::print_translation_unit(tu);
            assert!(text.contains("exported("), "missing PE export in:\n{text}");
            assert!(
                text.contains("Imported("),
                "IAT call was not named in:\n{text}"
            );
            assert!(
                text.contains("add_five(p0, 2, 3, 4, 5)"),
                "Win64 stack argument value was not preserved in:\n{text}"
            );
            // Clang sees the callee body and is free to omit setup for its
            // unused b/c/d parameters even without inlining it. The binary
            // therefore contains only positions 0 and 4. The decompiler must
            // preserve that gap as five source-level arguments and keep the
            // real fifth value in the final slot.
            let sparse_call = text
                .lines()
                .find(|line| line.contains("= first_and_fifth("))
                .expect("missing call to first_and_fifth");
            let sparse_args = sparse_call
                .split_once("first_and_fifth(")
                .and_then(|(_, tail)| tail.split_once(')'))
                .map(|(args, _)| args)
                .expect("malformed first_and_fifth call");
            assert_eq!(
                sparse_args.split(',').count(),
                5,
                "Win64 sparse parameter slots were compacted in:\n{text}"
            );
            assert_eq!(
                sparse_args.rsplit(',').next().map(str::trim),
                Some("50"),
                "Win64 fifth argument moved out of position 4 in:\n{text}"
            );

            let sparse_body_start = text
                .match_indices("first_and_fifth(")
                .find_map(|(start, _)| {
                    let suffix = &text[start..];
                    let brace = suffix.find('{')?;
                    let semicolon = suffix.find(';').unwrap_or(usize::MAX);
                    (brace < semicolon).then_some(start + brace + 1)
                })
                .expect("missing first_and_fifth definition");
            let sparse_body = &text[sparse_body_start
                ..sparse_body_start
                    + text[sparse_body_start..]
                        .find("\n}")
                        .expect("unterminated first_and_fifth definition")];
            assert!(
                sparse_body.contains("p0") && sparse_body.contains("p4"),
                "Win64 sparse callee dataflow did not bind positions 0 and 4:\n{sparse_body}"
            );

            let mixed_call = text
                .lines()
                .find(|line| line.contains("= mixed_five(") || line.contains("return mixed_five("))
                .unwrap_or_else(|| panic!("missing call to mixed_five in:\n{text}"));
            let mixed_args = mixed_call
                .split_once("mixed_five(")
                .and_then(|(_, tail)| tail.trim().strip_suffix(");"))
                .expect("malformed mixed_five call");
            let mixed_args: Vec<&str> = mixed_args.split(',').map(str::trim).collect();
            assert_eq!(
                mixed_args.len(),
                5,
                "Win64 mixed GP/XMM slots did not form one five-argument sequence:\n{text}"
            );
            assert!(
                mixed_args[0].contains("p0"),
                "Win64 live-in RCX was not forwarded in slot 0:\n{text}"
            );
            assert!(
                mixed_args[4].contains("p1"),
                "Win64 live-in double was not preserved in stack slot 4:\n{text}"
            );
            for operation in [
                "var_0 = p0 + p1;",
                "var_0 = var_0 + p2;",
                "var_0 = var_0 + p3;",
                "var_0 = var_0 + p4;",
            ] {
                assert!(
                    text.contains(operation),
                    "Win64 parameter dataflow lost `{operation}` in:\n{text}"
                );
            }

            let add_five = db
                .rel_iter::<(Address, Symbol, Symbol)>("symbols")
                .find_map(|(addr, name, _)| (*name == "add_five").then_some(*addr))
                .expect("missing add_five export");
            assert!(db
                .rel_iter::<(Address, usize)>("emit_function_param_count")
                .any(|(addr, count)| *addr == add_five && *count == 5));
            assert!(
                db.rel_iter::<(manifold::x86::types::Node, Address)>("call_target_func")
                    .filter(|(_, target)| *target == add_five)
                    .any(|(call, _)| db
                        .rel_iter::<(
                            manifold::x86::types::Node,
                            usize,
                            manifold::x86::types::RTLReg
                        )>("call_arg")
                        .any(|(arg_call, position, _)| arg_call == call && *position == 4)),
                "Win64 fifth argument was not recovered at shared position 4"
            );

            let first_and_fifth = db
                .rel_iter::<(Address, Symbol, Symbol)>("symbols")
                .find_map(|(addr, name, _)| (*name == "first_and_fifth").then_some(*addr))
                .expect("missing first_and_fifth export");
            assert!(db
                .rel_iter::<(Address, usize)>("emit_function_param_count")
                .any(|(addr, count)| *addr == first_and_fifth && *count == 5));

            let mixed_five = db
                .rel_iter::<(Address, Symbol, Symbol)>("symbols")
                .find_map(|(addr, name, _)| (*name == "mixed_five").then_some(*addr))
                .expect("missing mixed_five export");
            assert!(db
                .rel_iter::<(Address, usize)>("emit_function_param_count")
                .any(|(addr, count)| *addr == mixed_five && *count == 5));
        })
        .expect("failed to spawn PE pipeline test thread")
        .join()
        .expect("PE pipeline test thread panicked");
}
