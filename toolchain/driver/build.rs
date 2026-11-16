//! Embeds the explanations `wrela explain` prints (explain/<code>.wrela), one per diagnostic
//! code, as a table: `src/explain.rs` reads them. Also embeds the conformance suite's cases, for
//! `wrela primer`, and the version of the WASM engine that computes a constant, for the
//! constants cache's keys.

use sha2::{Digest, Sha256};
use std::fmt::Write;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    // The explanations: each by its code, the file's name.
    let explain = manifest.join("explain");
    let files: Vec<(String, PathBuf)> = walk(&explain, &|n| n.ends_with(".wrela"))
        .into_iter()
        .map(|(rel, path)| (rel.trim_end_matches(".wrela").to_string(), path))
        .collect();
    out("explanations.rs", &table("FILES", &files));
    // The conformance suite's cases (toolchain/tests/conformance: each a file, or a directory of
    // a package's files), by path from the suite.
    let cases = walk(&root.join("toolchain/tests/conformance"), &|n| n.ends_with(".wrela"));
    out("conformance.rs", &table("CASES", &cases));
    // The WASM engine that computes a constant, by its version (const_cache.rs's key): a value
    // depends on what runs it as well as on its WASM. The evaluator's own meaning is
    // `const_cache::MEANING`, a number raised by hand, not a hash of its sources: a change to
    // the toolchain that changes no value cooks nothing again.
    let lock = root.join("toolchain/Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let lock = std::fs::read_to_string(&lock).expect("Cargo.lock");
    let wasmtime = lock
        .split("[[package]]")
        .find(|p| p.contains("\nname = \"wasmtime\"\n"))
        .expect("wasmtime in Cargo.lock");
    let id: String =
        Sha256::digest(wasmtime.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
    println!("cargo:rustc-env=WRELA_CONST_ENGINE={id}");
}

/// The files under `base` whose names `keep` takes, at any depth, each by its path from `base`
/// with `/` between its parts, sorted by that path. Cargo runs this again when one of them, or a
/// directory's list, changes.
fn walk(base: &Path, keep: &dyn Fn(&str) -> bool) -> Vec<(String, PathBuf)> {
    fn visit(
        base: &Path,
        dir: &Path,
        keep: &dyn Fn(&str) -> bool,
        out: &mut Vec<(String, PathBuf)>,
    ) {
        println!("cargo:rerun-if-changed={}", dir.display());
        let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for e in entries.flatten() {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                visit(base, &path, keep, out);
            } else if keep(&name) {
                println!("cargo:rerun-if-changed={}", path.display());
                let rel = path.strip_prefix(base).expect("under base");
                out.push((rel.to_string_lossy().replace('\\', "/"), path));
            }
        }
    }
    let mut out = Vec::new();
    visit(base, base, keep, &mut out);
    out.sort();
    out
}

/// The Rust table `name` of `files`: each one's key, and its text.
fn table(name: &str, files: &[(String, PathBuf)]) -> String {
    let mut out = format!("pub(crate) const {name}: &[(&str, &str)] = &[\n");
    for (key, path) in files {
        writeln!(out, "    ({key:?}, include_str!({:?})),", path.display().to_string())
            .expect("write to a string");
    }
    out.push_str("];\n");
    out
}

/// Writes `text` to the file `name` in cargo's `OUT_DIR`.
fn out(name: &str, text: &str) {
    let dest = Path::new(&std::env::var("OUT_DIR").expect("OUT_DIR")).join(name);
    std::fs::write(&dest, text).unwrap_or_else(|e| panic!("{}: {e}", dest.display()));
}
