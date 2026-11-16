pub mod wat;

use std::ops::Deref;
use std::path::{Path, PathBuf};

/// The first-light fixture's build for this machine's target.
pub fn first_light() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/first-light")
        .join(wrela_abi::Target::HERE.name())
}

/// A shader file's name on this machine's target: `stem.metal` or `stem.spv`.
pub fn shader_file(stem: &str) -> String {
    format!("{stem}.{}", wrela_wgsl::extension(wrela_abi::Target::HERE))
}

/// A fresh, empty temporary directory for one test, removed when this drops (even if the test
/// fails).
pub struct TempDir(PathBuf);

impl Deref for TempDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempDir {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A [`TempDir`] named for the test `name`.
pub fn temp_dir(name: &str) -> TempDir {
    let dir = std::env::temp_dir().join(format!("wrela-host-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    TempDir(dir)
}

/// A copy of first-light's build.
pub fn first_light_copy(name: &str) -> TempDir {
    let dir = temp_dir(name);
    for entry in std::fs::read_dir(first_light()).expect("fixture") {
        let path = entry.expect("entry").path();
        if path.is_file() {
            std::fs::copy(&path, dir.join(path.file_name().expect("name"))).expect("copy");
        }
    }
    dir
}

/// A copy of first-light's build with its `game.wasm` replaced by `wasm`.
pub fn first_light_with(name: &str, wasm: &[u8]) -> TempDir {
    let dir = first_light_copy(name);
    std::fs::write(dir.join("game.wasm"), wasm).expect("write wasm");
    dir
}

/// A manifest of this ABI's versions for `game.wasm` on this machine's target, with `pipelines`
/// (a JSON array).
pub fn manifest(pipelines: &str) -> String {
    format!(
        r#"{{ "manifest_version": {}, "stream_version": {}, "target": "{}", "wasm": "game.wasm", "pipelines": {pipelines} }}"#,
        wrela_abi::manifest::VERSION,
        wrela_abi::stream::VERSION,
        wrela_abi::Target::HERE.name(),
    )
}

/// A build directory with the given manifest, shaders and WASM. A shader file named `*.wgsl`
/// is WGSL, which the build holds as the MSL the host takes ([`write_shader`]); any other is
/// written as it is.
pub fn build(name: &str, manifest: &str, shaders: &[(&str, &str)], wasm: &[u8]) -> TempDir {
    let dir = temp_dir(name);
    std::fs::write(dir.join("manifest.json"), manifest).expect("write manifest");
    for (file, source) in shaders {
        if file.ends_with(".wgsl") {
            write_shader(&dir, file, source);
        } else {
            std::fs::write(dir.join(file), source).expect("write shader");
        }
    }
    std::fs::write(dir.join("game.wasm"), wasm).expect("write wasm");
    dir
}

/// Writes the WGSL `source` of the shader file `file` (`x.wgsl`) of `dir`'s build as the shader
/// this machine's host takes (`x.metal` or `x.spv`), by the compiler's own writer
/// (`wrela_wgsl::shader_of_wgsl`), and the manifest's pipelines that name either as naming it,
/// with the sizes and entry points' names that shader has, and a small bound on each entry
/// point's work where it states none. A manifest the host would refuse is
/// left as it is.
pub fn write_shader(dir: &Path, file: &str, source: &str) {
    let msl_file = shader_file(file.strip_suffix(".wgsl").expect("a .wgsl file"));
    let path = dir.join("manifest.json");
    let Ok(mut m) = wrela_abi::Manifest::parse(&std::fs::read_to_string(&path).expect("manifest"))
    else {
        return;
    };
    let mut text = None;
    for p in m.pipelines.iter_mut().filter(|p| p.shader == file || p.shader == msl_file) {
        p.shader = msl_file.clone();
        // The compiler doesn't count hand-written WGSL: the tests' small kernels' most work an
        // invocation, which the host weighs a dispatch by, is stated here.
        if p.work.is_empty() {
            p.work = vec![Some(64); 2];
        }
        let (target, checks) = (wrela_abi::Target::HERE, wrela_wgsl::Checks::CHECKED);
        let written = wrela_wgsl::shader_of_wgsl(source, p, target, checks)
            .unwrap_or_else(|e| panic!("{file}: {e}"));
        text = Some(written.bytes);
    }
    if let Some(text) = text {
        std::fs::write(dir.join(&msl_file), text).expect("write shader");
        std::fs::write(&path, m.to_json()).expect("write manifest");
    }
}
