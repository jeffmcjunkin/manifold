use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use manifold::abi::BinaryFormat;
use manifold::decompile::elevator::DecompileDB;

fn command_exists(name: &str) -> bool {
    Command::new(name)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn build_fixture() -> Option<PathBuf> {
    if !command_exists("clang") || !command_exists("clang++") {
        eprintln!("skipping final-TU struct-field test: clang/clang++ unavailable");
        return None;
    }

    let dir = std::env::temp_dir().join(format!(
        "manifold_final_tu_struct_fields_fixture_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let source = dir.join("fixture.s");
    let object = dir.join("fixture.obj");

    // VS2013-style Win64 parameter spills followed by indexed loads.  Across
    // this TU, decl-solve recognizes stack slot +8 as a pointer field even
    // though the initial recovered layout represents it as an integer.
    std::fs::write(
        &source,
        r#"
        .text

        .globl final_tu_blend
        .def final_tu_blend; .scl 2; .type 32; .endef
final_tu_blend:
        movq %rsp, %rax
        movl %r8d, 24(%rax)
        movl %edx, 16(%rax)
        movq %rcx, 8(%rax)
        movl %edx, %ecx
        movq 8(%rax), %rdx
        movq (%rdx), %rax
        movl (%rax,%rcx,4), %eax
        xorl 24(%rsp), %eax
        andl 8(%rdx), %eax
        retq

        .globl final_tu_indexed_imul
        .def final_tu_indexed_imul; .scl 2; .type 32; .endef
final_tu_indexed_imul:
        movq %rsp, %rax
        movl %edx, 16(%rax)
        movq %rcx, 8(%rax)
        movq 8(%rax), %rax
        movl %edx, %ecx
        imull $13, (%rax,%rcx,4), %eax
        retq

        .globl final_tu_indexed_or
        .def final_tu_indexed_or; .scl 2; .type 32; .endef
final_tu_indexed_or:
        movq %rsp, %rax
        movl %r8d, 24(%rax)
        movl %edx, 16(%rax)
        movq %rcx, 8(%rax)
        movl %r8d, %eax
        movl %edx, %edx
        leal (%rax,%rax,4), %ecx
        movq 8(%rsp), %rax
        movl (%rax,%rdx,4), %eax
        orl %ecx, %eax
        retq

        .globl final_tu_scale
        .def final_tu_scale; .scl 2; .type 32; .endef
final_tu_scale:
        movq %rsp, %rax
        movl %r8d, 24(%rax)
        movl %edx, 16(%rax)
        movq %rcx, 8(%rax)
        movq %rcx, %rax
        movq (%rcx), %rcx
        movl %edx, %edx
        imull $11, (%rcx,%rdx,4), %eax
        addl 24(%rsp), %eax
        retq
"#,
    )
    .ok()?;

    let status = Command::new("clang")
        .args(["--target=x86_64-pc-windows-msvc", "-c"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .ok()?;
    if !status.success() {
        eprintln!("skipping final-TU struct-field test: fixture assembly failed");
        return None;
    }
    Some(object)
}

fn fixture() -> Option<&'static Path> {
    static FIXTURE: OnceLock<Option<PathBuf>> = OnceLock::new();
    FIXTURE.get_or_init(build_fixture).as_deref()
}

fn assert_selected_field_types_drive_assignment_casts(object: &Path) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .stack_size(64 * 1024 * 1024)
        .build()
        .expect("failed to build final-TU test thread pool");
    let mut db = DecompileDB::default();
    manifold::decompile::disassembly::load_from_binary(&mut db, object);
    manifold::decompile::disassembly::load_preset(&mut db);
    pool.install(|| db.run_pipeline(object, true, false));

    let tu = db
        .cast_optimized_translation_unit
        .as_ref()
        .expect("pipeline must emit an optimized translation unit");
    let text = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        tu,
        BinaryFormat::Coff,
    );
    let raw_tu = db
        .cast_raw_translation_unit
        .as_ref()
        .expect("trace pipeline must emit a raw translation unit");
    let raw_text = manifold::decompile::passes::c_pass::print_translation_unit_for_format(
        raw_tu,
        BinaryFormat::Coff,
    );

    assert!(
        text.contains("void *ofs_8;"),
        "fixture must exercise decl-solve's final pointer field selection:\n{text}"
    );
    assert!(
        text.contains("->ofs_8 = (void *)p0;"),
        "integer Win64 parameter must be cast to the final pointer field type:\n{text}"
    );
    assert!(
        raw_text.contains("->ofs_8 = (void *)p0;"),
        "raw TU wrapper must consume the same selected field types:\n{raw_text}"
    );
    assert!(
        !text.contains("->ofs_8 = p0;"),
        "no assignment may retain the pre-selection integer field type:\n{text}"
    );

    let output = object.with_extension("generated.cpp");
    std::fs::write(&output, &text).expect("failed to write generated C++ fixture output");
    let compiled = Command::new("clang++")
        .args([
            "--target=x86_64-pc-windows-msvc",
            "-fms-extensions",
            "-Wno-everything",
            "-x",
            "c++",
            "-fsyntax-only",
        ])
        .arg(&output)
        .output()
        .expect("failed to run clang++ over generated output");
    assert!(
        compiled.status.success(),
        "final TU does not compile as C++:\n{}\n{text}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn selected_struct_field_types_compile_through_the_real_coff_pipeline() {
    let Some(object) = fixture() else { return };
    let object = object.to_path_buf();
    std::thread::Builder::new()
        .name("final-tu-struct-field-output".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || assert_selected_field_types_drive_assignment_casts(&object))
        .expect("failed to spawn final-TU test thread")
        .join()
        .expect("final-TU test thread panicked");
}
