//! `wrela shaders`: a build's shaders through the platforms' offline tools
//! (`wrela_shader::offline`), every failure named, and a directory with none refused.
use super::common::{package, run, wrela};

#[test]
fn every_shader_is_checked_and_each_failure_named() {
    let dir = package(
        "shaders-checked",
        &[
            ("pipeline_0.metal", "kernel void k() {}\n"),
            ("pipeline_1.metal", "kernel void k() {}\n"),
            ("pipeline_2.metal", "kernel void k( {\n"),
            ("manifest.json", "{}"),
        ],
    );
    let cache = dir.join("checks");
    let (code, said) = run(wrela()
        .args(["shaders", dir.to_str().unwrap(), "--jobs", "2"])
        .env("WRELA_SHADER_CHECKS", &cache));
    if code == Some(2) && said.contains("this machine hasn't") {
        eprintln!("skipped: {said}");
        return;
    }
    assert_eq!(code, Some(1), "{said}");
    assert!(said.contains("FAILED pipeline_2.metal"), "{said}");
    assert!(!said.contains("FAILED pipeline_0") && !said.contains("FAILED pipeline_1"), "{said}");
    assert!(said.contains("shaders: 2 of 3 pass"), "{said}");
}

#[test]
fn a_directory_without_shaders_is_refused() {
    let dir = package("shaders-none", &[("manifest.json", "{}")]);
    let (code, said) = run(wrela().args(["shaders", dir.to_str().unwrap()]));
    assert_eq!(code, Some(2), "{said}");
    assert!(said.contains("has no shaders"), "{said}");
    let (code, _) = run(wrela().args(["shaders"]));
    assert_eq!(code, Some(2));
}
