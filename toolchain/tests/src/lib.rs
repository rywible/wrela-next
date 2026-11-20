//! Shared pieces of the end-to-end tests: building a package with the compiler, calling a
//! program's exports, reading case files, and running hand-written WGSL (the spike's kernels) on
//! the same GPU for comparison.
//!
//! The one check runs every retained CPU and GPU regression. Samples have one bounded
//! size; historical milestone benchmarks and full-scene tours are not part of verification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use wrela_diag::{Diagnostic, Edit, FileId};
use wrela_host::{CpuHost, Host, Value};

/// SplitMix64, for tests' random inputs.
pub mod rng;
pub mod testing;
pub use rng::Rng;
pub use testing::{files_under, par_each, par_map, par_map_with, repo_root};

/// Where this process writes its test files. The gate gives each isolated test an absolute
/// `WRELA_TEST_TMPDIR`, so processes never remove or overwrite each other's builds. Outside
/// the gate, the caller's usual scratch directory is used.
pub fn scratch_root(default: impl AsRef<Path>) -> PathBuf {
    scratch_root_in(std::env::var_os("WRELA_TEST_TMPDIR"), default.as_ref())
}

fn scratch_root_in(dir: Option<std::ffi::OsString>, default: &Path) -> PathBuf {
    match dir.filter(|d| !d.is_empty()) {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            assert!(dir.is_absolute(), "WRELA_TEST_TMPDIR must be an absolute path");
            dir
        }
        None => default.to_path_buf(),
    }
}

/// The scratch directory for pages and images, normally under `toolchain/target/tmp`.
pub fn page_scratch_root() -> PathBuf {
    scratch_root(repo_root().join("toolchain/target/tmp"))
}

/// Work done once per process for each key: a second caller with a key waits for the first's
/// work rather than racing it, and work for other keys runs at the same time.
pub struct OncePerKey(Mutex<BTreeMap<String, Arc<OnceLock<()>>>>);

impl OncePerKey {
    pub const fn new() -> OncePerKey {
        OncePerKey(Mutex::new(BTreeMap::new()))
    }

    /// Does `work` for `key` unless it's done (or another thread is doing it: this waits).
    pub fn run(&self, key: &str, work: impl FnOnce()) {
        let once =
            self.0.lock().unwrap_or_else(|p| p.into_inner()).entry(key.into()).or_default().clone();
        once.get_or_init(work);
    }
}

impl Default for OncePerKey {
    fn default() -> OncePerKey {
        OncePerKey::new()
    }
}

/// Builds the package in `pkg` as `wrela build --testing` does (with its `@testing` exports),
/// without writing its files: the output, or the errors rendered.
pub fn build(pkg: &Path) -> Result<wrela_driver::Output, String> {
    let built = wrela_driver::build_for_tests(pkg, false);
    if built.has_errors() {
        return Err(wrela_diag::render::render_all(&built.sources, &built.diagnostics));
    }
    for (name, bytes) in &built.files {
        if name.ends_with(".metal") && bytes.len() > 288 * 1024 {
            return Err(format!("{name}: {} bytes, over the 288 KiB shader budget", bytes.len()));
        }
    }
    offline_check(&built)?;
    Ok(built)
}

/// Every shader of a build through its platform's own offline tool, where this machine has it
/// (`wrela_shader::offline`): Apple's `metal` compiler, with warnings as errors, or `spirv-val`.
pub fn offline_check(built: &wrela_driver::Output) -> Result<(), String> {
    wrela_shader::offline::check_all(built.files.iter().map(|(n, b)| (n.as_str(), b.as_slice())))
}

/// [`build`], for `target`'s shaders and limits (`wrela build --target`) rather than this
/// machine's: for a test that reads a shader's text, or one of the limits of the machine its
/// cases were written for.
pub fn build_for(pkg: &Path, target: wrela_abi::Target) -> Result<wrela_driver::Output, String> {
    let how = wrela_driver::BuildFor {
        kind: wrela_driver::BuildKind { debug: false, testing: true },
        set: Vec::new(),
        target,
        trusted_shaders: false,
        exact: false,
    };
    let built = wrela_driver::build_for(pkg, &how);
    if built.has_errors() {
        return Err(wrela_diag::render::render_all(&built.sources, &built.diagnostics));
    }
    offline_check(&built)?;
    Ok(built)
}

/// A shipped build of the package in `pkg` (`wrela_driver::build`), for `target`, errors and
/// all.
pub fn shipped_for(pkg: &Path, target: wrela_abi::Target) -> wrela_driver::Output {
    let how = wrela_driver::BuildFor {
        kind: wrela_driver::BuildKind::default(),
        set: Vec::new(),
        target,
        trusted_shaders: false,
        exact: false,
    };
    wrela_driver::build_for(pkg, &how)
}

/// Builds the package in `pkg` into `out`, as `wrela build` does. Panics with the errors if it
/// doesn't build.
pub fn must_build(pkg: &Path, out: &Path) {
    let built = build(pkg).unwrap_or_else(|e| panic!("{} doesn't build:\n{e}", pkg.display()));
    built.write_to(out).expect("write the build");
}

/// Copies the files under `from` (as [`files_under`] finds them) to the same paths under `to`.
pub fn copy_dir(from: &Path, to: &Path) {
    for file in files_under(from, &[]) {
        let dest = to.join(file.strip_prefix(from).expect("a file under `from`"));
        std::fs::create_dir_all(dest.parent().expect("a parent")).expect("make a directory");
        std::fs::copy(&file, &dest).expect("copy a file");
    }
}

/// Writes `edit` of the file `file` of the package at `pkg`, which must change it: when it was
/// written.
pub fn write_edit(pkg: &Path, file: &str, edit: impl Fn(&str) -> String) -> std::time::Instant {
    let path = pkg.join(file);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{file}: {e}"));
    let new = edit(&src);
    assert_ne!(new, src, "the edit changes {file}");
    std::fs::write(&path, new).unwrap_or_else(|e| panic!("write {file}: {e}"));
    std::time::Instant::now()
}

/// The cases of a case-file suite: the `.wrela` files and the directories in `dir`, sorted.
pub fn cases(dir: &Path) -> Vec<PathBuf> {
    let mut cases: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| p.is_dir() || p.extension().is_some_and(|x| x == "wrela"))
        .collect();
    cases.sort();
    cases
}

/// The values of a case file's `// <key>: <value>` lines, trimmed. Only the comment block at
/// the start of `text` is read.
pub fn header<'a>(text: &'a str, key: &str) -> impl Iterator<Item = &'a str> {
    let prefix = format!("// {key}:");
    text.lines()
        .take_while(|l| l.starts_with("//"))
        .filter_map(move |l| l.strip_prefix(prefix.as_str()).map(str::trim))
}

/// Applies every edit of the first fix of each diagnostic to `text` (all in one file). An edit
/// the same as the one before it (two diagnostics suggesting it) is made once; edits that
/// overlap otherwise can't both be made, so that's a test failure. Used by tests that check a
/// fix removes its diagnostic.
pub fn apply_fixes(text: &str, file: FileId, diags: &[Diagnostic]) -> String {
    let mut edits: Vec<&Edit> = diags
        .iter()
        .filter_map(|d| d.fixes.first())
        .flat_map(|f| f.edits.iter())
        .filter(|e| e.span.file == file)
        .collect();
    edits.sort_by_key(|e| (e.span.start, e.span.end));
    edits.dedup_by(|a, b| a.span == b.span && a.replacement == b.replacement);
    wrela_diag::apply_edits(text, edits)
}

/// The errors among `diags` whose only next step is the general one (`wrela explain`), each as
/// `code: message`: an error must say what to do about it (`Diagnostic::with_next_step`).
pub fn without_own_help<'a>(diags: impl IntoIterator<Item = &'a Diagnostic>) -> Vec<String> {
    diags
        .into_iter()
        .filter(|d| d.is_error() && d.has_only_general_help())
        .map(|d| format!("{}: {}", d.code.as_str(), d.message))
        .collect()
}

/// A loaded program whose exports a test calls: a [`CpuHost`] or a [`Host`].
pub trait Exports {
    fn call(&mut self, name: &str, args: &[Value]) -> wrela_host::Result<Vec<Value>>;
}

impl Exports for CpuHost {
    fn call(&mut self, name: &str, args: &[Value]) -> wrela_host::Result<Vec<Value>> {
        self.call_export(name, args)
    }
}

impl Exports for Host {
    fn call(&mut self, name: &str, args: &[Value]) -> wrela_host::Result<Vec<Value>> {
        self.call_export(name, args)
    }
}

/// Calls the export `name`, which must return one `f32`.
pub fn one_f32(host: &mut impl Exports, name: &str, args: &[Value]) -> f32 {
    match host.call(name, args).expect(name).as_slice() {
        [Value::F32(x)] => *x,
        other => panic!("{name} returned {other:?}"),
    }
}

/// Calls the export `name`, which must return one `u32` (an `i32` in WASM).
pub fn one_u32(host: &mut impl Exports, name: &str, args: &[Value]) -> u32 {
    match host.call(name, args).expect(name).as_slice() {
        [Value::I32(x)] => *x as u32,
        other => panic!("{name} returned {other:?}"),
    }
}

/// Calls the export `name`, which must return one `u64` (an `i64` in WASM).
pub fn one_u64(host: &mut impl Exports, name: &str, args: &[Value]) -> u64 {
    match host.call(name, args).expect(name).as_slice() {
        [Value::I64(x)] => *x as u64,
        other => panic!("{name} returned {other:?}"),
    }
}

/// Calls the export `name`, which must return one `vec3` (three `f32`s).
pub fn one_vec3(host: &mut impl Exports, name: &str, args: &[Value]) -> [f32; 3] {
    match host.call(name, args).expect(name).as_slice() {
        [Value::F32(x), Value::F32(y), Value::F32(z)] => [*x, *y, *z],
        other => panic!("{name} returned {other:?}"),
    }
}

/// Calls the export `name`, which must return one `vec4` (four `f32`s).
pub fn one_vec4(host: &mut impl Exports, name: &str, args: &[Value]) -> [f32; 4] {
    match host.call(name, args).expect(name).as_slice() {
        [Value::F32(x), Value::F32(y), Value::F32(z), Value::F32(w)] => [*x, *y, *z, *w],
        other => panic!("{name} returned {other:?}"),
    }
}

/// A little-endian `f32` array from bytes.
pub fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn u32s(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

pub fn bytes_of(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The larger error; NaN if either is (`f64::max` drops a NaN, which would pass every check).
pub fn worse(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) }
}

/// The manifest of the build in `dir`.
pub fn manifest(dir: &Path) -> wrela_abi::Manifest {
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest.json");
    wrela_abi::Manifest::parse(&text).expect("a valid manifest")
}

/// Builds the package at `pkg` (a path from the repo root) into `toolchain/target/tmp/<name>`. Returns the
/// directory and its path from the repo root. Each package is built once per process, and each
/// page is a copy of that build.
pub fn page(pkg: &str, name: &str) -> (PathBuf, String) {
    page_built_by(pkg, name, "release", |p| wrela_driver::build_for_tests(p, false))
}

/// [`page`] with a debug build (language.md §11's checks).
pub fn debug_page(pkg: &str, name: &str) -> (PathBuf, String) {
    page_built_by(pkg, name, "debug", |p| wrela_driver::build_for_tests(p, true))
}

/// [`page`] with a build whose WASM uses no SIMD (language.md §11).
pub fn scalar_page(pkg: &str, name: &str) -> (PathBuf, String) {
    page_built_by(pkg, name, "scalar", wrela_driver::build_without_simd)
}

/// [`page`] with a relaxed-SIMD build (`wrela_driver::build_relaxed`): `game.relaxed.wasm` beside
/// `game.wasm`.
pub fn relaxed_page(pkg: &str, name: &str) -> (PathBuf, String) {
    page_built_by(pkg, name, "relaxed", wrela_driver::build_relaxed)
}

/// [`page`], the package built by `build` (`kind` names it).
fn page_built_by(
    pkg: &str,
    name: &str,
    kind: &str,
    build: fn(&Path) -> wrela_driver::Output,
) -> (PathBuf, String) {
    let base = built_once(pkg, kind, build);
    let dir = page_scratch_root().join(name);
    std::fs::create_dir_all(&dir).expect("make the page's directory");
    let dir = dir.canonicalize().expect("the page's directory");
    let root = repo_root().canonicalize().expect("the repository's directory");
    let rel = dir
        .strip_prefix(root)
        .expect("a browser page's scratch directory must be under the repository")
        .to_string_lossy()
        .into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    copy_dir(&base, &dir);
    (dir, rel)
}

/// The package at `pkg` built by `build` (`kind` names it), once per process: its directory.
fn built_once(pkg: &str, kind: &str, build: fn(&Path) -> wrela_driver::Output) -> PathBuf {
    static BUILDS: OncePerKey = OncePerKey::new();
    // Hosts load copies of this shared build from their own scratch directories.
    let base =
        page_scratch_root().join("page-builds").join(format!("{}-{kind}", pkg.replace('/', "-")));
    BUILDS.run(&base.display().to_string(), || {
        let _ = std::fs::remove_dir_all(&base);
        let dir = repo_root().join(pkg);
        let built = build(&dir);
        if built.has_errors() {
            panic!(
                "{pkg} doesn't build:\n{}",
                wrela_diag::render::render_all(&built.sources, &built.diagnostics)
            );
        }
        if let Err(e) = offline_check(&built) {
            panic!("{pkg}'s shaders: {e}");
        }
        built.write_to(&base).expect("write the build");
    });
    base
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_test_scratch_override_is_absolute_and_otherwise_keeps_the_default() {
        let default = std::path::Path::new("/build/target/tmp");
        assert_eq!(super::scratch_root_in(None, default), default);
        assert_eq!(super::scratch_root_in(Some("".into()), default), default);
        assert_eq!(
            super::scratch_root_in(Some("/build/target/tmp-alone/7".into()), default),
            std::path::Path::new("/build/target/tmp-alone/7")
        );
        assert!(
            std::panic::catch_unwind(|| {
                super::scratch_root_in(Some("../another-test".into()), default)
            })
            .is_err(),
            "a relative override could put two programs' files in different directories"
        );
    }
}
