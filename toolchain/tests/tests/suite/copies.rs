//! A large uniform struct is read where it lies, never copied into each invocation
//! (toolchain/tests/copies; ir/src/opt.rs's `forward` and GPU `split_locals`). The volume spike
//! lost 5-8x a pass to such copies twice: a method returning a copy of `self` with a field
//! changed (`form.only(mask).sample(p)`), and a by-value call inside a `for` the compiler writes
//! out (each copy of the loop stored the struct into the same local again). Built for Metal and
//! for Vulkan, no pipeline keeps a byte (the manifest's `kept`: the arrays, structs and enums
//! its functions hold), and the Metal source declares no `Body`; on the GPU, every kernel's
//! samples are the CPU's.
//! `cargo test -p wrela-tests --test suite copies:: -- --ignored` for the GPU's.

use wrela_host::{Host, Value};
use wrela_tests::{one_u32, page};

const KERNELS: [&str; 4] = ["plain", "masked", "unrolled", "looped"];

#[test]
fn no_kernel_copies_the_uniform_body() {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/copies");
    for target in [wrela_abi::Target::Metal, wrela_abi::Target::Vulkan] {
        let built = wrela_tests::build_for(&pkg, target).expect("builds");
        let manifest = wrela_tests::built_manifest(&built);
        for k in KERNELS {
            let p = manifest.pipelines.iter().find(|p| p.name == k).expect(k);
            assert_eq!(p.kept, 0, "{k} ({target:?}) keeps {} bytes", p.kept);
            if target == wrela_abi::Target::Metal {
                let msl = String::from_utf8_lossy(wrela_tests::file(&built, &p.shader));
                let copies =
                    msl.lines().filter(|l| l.contains("_Body ") && l.contains('=')).count();
                assert_eq!(copies, 0, "{k}'s MSL copies a Body:\n{msl}");
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn every_kernel_samples_what_the_cpu_does() {
    let (dir, _) = page("toolchain/tests/copies", "copies-native");
    let mut host = Host::load(&dir).expect("load");
    let times = crate::frame_times(4);
    host.run_frames(&times, 16, 16).expect("run");
    for (k, name) in KERNELS.iter().enumerate() {
        let arg = [Value::I32(k as i32)];
        assert_eq!(one_u32(&mut host, "arrived", &arg), 64 * 1024, "{name}'s samples arrived");
        assert_eq!(one_u32(&mut host, "matched", &arg), 64 * 1024, "{name}'s are the CPU's");
    }
}

/// A copy the code needs, as it stands (a part changed at a place only the invocation knows,
/// then read again), is E0713 in a shipped build, at the copy, and no error in a check; a
/// smaller one is none.
#[test]
fn a_copy_the_code_needs_is_e0713_in_a_shipped_build() {
    let kernel = |name: &str, n: u32| {
        let dir = crate::scratch(name);
        let src = format!(
            "use std::gpu::{{GlobalId, GpuBuffer, Slots, buffer, dispatch}}

struct Given: Copy + GpuData {{
    a: [vec4; {n}],
}}

@compute(64)
fn k(g: Given, out: mut Slots<f32>, id: GlobalId) {{
    var a = g.a
    a[id.x % {n}] = vec4(f32(id.x))
    out[id] = a[(id.x + 1) % {n}].x + a[id.x % {n}].y
}}

pub fn frame(time: f32, width: u32, height: u32) {{
    var out: GpuBuffer<f32> = buffer(64)
    dispatch(k.bind(g: Given {{ a: [vec4(time); {n}] }}, out: mut out), over: 64)
}}
"
        );
        crate::write_package(&dir, name, &src);
        dir
    };
    let e0713 = |dir: &std::path::Path| -> Vec<String> {
        let out = wrela_driver::build(dir);
        let others: Vec<_> = out
            .diagnostics
            .iter()
            .filter(|d| d.is_error() && d.code != wrela_diag::codes::E0713)
            .map(|d| &d.message)
            .collect();
        assert!(others.is_empty(), "{others:?}");
        let ours = out.diagnostics.iter().filter(|d| d.code == wrela_diag::codes::E0713);
        ours.map(|d| d.message.clone()).collect()
    };
    let big = kernel("copied_big", 20);
    let msgs = e0713(&big);
    assert_eq!(msgs.len(), 1, "{msgs:?}");
    assert!(msgs[0].contains("copies an array (320 bytes) from the uniform block"), "{msgs:?}");
    let checked = wrela_driver::check_pipelines(&big);
    assert!(!checked.has_errors(), "a check isn't held to it: {:?}", checked.diagnostics);
    // Fifteen `vec4`s: under the line.
    assert_eq!(e0713(&kernel("copied_small", 15)), Vec::<String>::new());
}
