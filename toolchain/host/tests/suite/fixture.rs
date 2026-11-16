//! The checked-in fixtures are current. A fixture is a directory of sources (`game.wat`, and each
//! shader built as IR in `common::shaders`) with a build of them for each target (`metal/`,
//! `vulkan/`): its `game.wasm` the `game.wat` compiled, its shaders written by the compiler's own
//! writers for that target, its manifest valid and in canonical form.
//! `WRELA_BLESS=1` rewrites the builds' files. A Metal shader may open with comment lines of its
//! own; past its opening comments, its text is the writer's, byte for byte.

use crate::common::shaders;
use std::path::{Path, PathBuf};
use wrela_shader::build::Builder;

/// Fixture `dir`'s shader `stem`, built as IR.
fn sources(dir: &Path, stem: &str) -> Builder {
    let fixture = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    match (fixture, stem) {
        ("first-light", "first-light") => shaders::first_light_render(),
        ("first-light", "fill") => shaders::first_light_fill(),
        _ => panic!("no shader `{stem}` of fixture `{fixture}` in common::shaders"),
    }
}
use wrela_abi::{Manifest, Target};

fn fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("toolchain/fixtures")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.join("game.wat").is_file())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no fixtures in {}", root.display());
    dirs
}

/// Text after its leading comment lines.
fn uncommented(text: &str) -> &str {
    let mut rest = text;
    while rest.starts_with("//") {
        rest = rest.split_once('\n').map_or("", |(_, r)| r);
    }
    rest
}

#[test]
fn each_targets_build_is_its_sources() {
    let bless = std::env::var_os("WRELA_BLESS").is_some();
    for dir in fixtures() {
        let wasm = wat::parse_file(dir.join("game.wat")).expect("game.wat compiles");
        // The pipelines as the Metal build's manifest states them, which the Vulkan build's
        // follows.
        let template =
            std::fs::read_to_string(dir.join("metal/manifest.json")).expect("metal/manifest.json");
        let template = Manifest::parse(&template).expect("a valid manifest");
        for target in Target::ALL {
            let out = dir.join(target.name());
            let mut m = template.clone();
            m.target = target;
            let mut shaders = Vec::new();
            for p in &mut m.pipelines {
                let stem = p.shader.rsplit_once('.').map_or(&*p.shader, |(s, _)| s).to_string();
                let source = sources(&dir, &stem);
                p.shader = format!("{stem}.{}", wrela_shader::extension(target));
                let written = source
                    .write(p, target)
                    .unwrap_or_else(|e| panic!("{stem} for {}: {e}", target.name()));
                shaders.push((p.shader.clone(), written.bytes));
            }
            let json = m.to_json();
            if bless {
                std::fs::create_dir_all(&out).expect("the target's directory");
                std::fs::write(out.join("game.wasm"), &wasm).expect("game.wasm");
                std::fs::write(out.join("manifest.json"), &json).expect("manifest.json");
            }
            let here = |name: &str| out.join(name).display().to_string();
            let read = |name: &str| std::fs::read(out.join(name)).unwrap_or_default();
            assert!(read("game.wasm") == wasm, "{} is stale (WRELA_BLESS=1)", here("game.wasm"));
            let checked_in = String::from_utf8(read("manifest.json")).unwrap_or_default();
            assert_eq!(checked_in, json, "{} is stale (WRELA_BLESS=1)", here("manifest.json"));
            for (name, bytes) in shaders {
                let file = read(&name);
                let same = match target {
                    Target::Metal => {
                        uncommented(&String::from_utf8_lossy(&file))
                            == uncommented(&String::from_utf8_lossy(&bytes))
                    }
                    Target::Vulkan => file == bytes,
                };
                if bless && !same {
                    std::fs::write(out.join(&name), &bytes).expect("the shader");
                    continue;
                }
                assert!(same, "{} is stale (WRELA_BLESS=1)", here(&name));
            }
        }
    }
}
