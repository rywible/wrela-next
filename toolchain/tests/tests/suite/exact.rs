//! `--exact` (language.md §18.11): a function's bits don't depend on the code around it.
//! toolchain/tests/exact's `f`, written into one kernel and kept as a call in another (300 more
//! calls of it push that kernel past the inlining limit), gives the same bits for the same 4,096
//! inputs in an exact build. In a default build it may not: said, not held.
//! `cargo test -p wrela-tests --test suite exact:: -- --ignored`.

use wrela_host::{Host, Value};

/// The fixture built for this machine, exact or not, into a scratch directory: how many of the
/// two kernels' 4,096 results differ in any bit.
fn differing(exact: bool) -> u32 {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/exact");
    let how = wrela_driver::BuildFor { exact, ..wrela_tests::testing_for(wrela_abi::Target::HERE) };
    let built = wrela_driver::build_for(&pkg, &how);
    assert!(
        !built.has_errors(),
        "{}",
        wrela_diag::render::render_all(&built.sources, &built.diagnostics)
    );
    // One kernel keeps `f` as a call, the other writes it in.
    let manifest = wrela_tests::built_manifest(&built);
    assert_eq!(manifest.pipelines.len(), 2);
    let dir = wrela_tests::page_scratch_root().join(if exact { "exact-on" } else { "exact-off" });
    let _ = std::fs::remove_dir_all(&dir);
    built.write_to(&dir).expect("write the build");
    let mut host = Host::load(&dir).expect("load");
    for _ in 0..20 {
        host.run_frames(&crate::frame_times(1), 16, 16).expect("a frame");
        match host.call_export("differ", &[]).expect("differ").as_slice() {
            // A count, 0 to 4,097; the fixture's `NOT_IN` (`u32::MAX`, -1 as an `i32`) until
            // the results arrive.
            [Value::I32(n)] if *n as u32 <= 4097 => return *n as u32,
            [Value::I32(-1)] => {}
            other => panic!("differ returned {other:?}"),
        }
    }
    panic!("the results never arrived")
}

#[test]
#[ignore = "needs a GPU"]
fn a_function_gives_the_same_bits_inlined_and_as_a_call_in_exact_mode() {
    let off = differing(false);
    eprintln!("default build: {off} of 4096 results differ between the inlined and the called `f`");
    assert_eq!(differing(true), 0, "exact build: the inlined and the called `f` differ");
}

/// A `@strict` function computes as the CPU does on every GPU (§11.3), so it may use nothing
/// whose result a driver defines: a derivative, or a texture read through a sampler's filter,
/// its own or a function's it calls. Arithmetic, and a texel read by its place, are its own.
#[test]
fn a_strict_function_refuses_what_a_driver_defines() {
    let main = r#"
use std::gpu::{Rgba8, Sampler, Texture}

fn slope(x: f32) -> f32 {
    dpdx(x)
}

@strict
fn edge(x: f32) -> f32 {
    slope(x) + 1.0
}

@strict
fn filtered(t: Texture<Rgba8>, s: Sampler, uv: vec2) -> f32 {
    t.sample(s, uv).x
}

@strict
fn own(t: Texture<Rgba8>, x: f32) -> f32 {
    t.load(x: 1, y: 2).x + x * x
}

pub fn frame(time: f32, width: u32, height: u32) {}
"#;
    let (codes, text) = super::checked(&crate::package("strict-driver", main));
    assert_eq!(codes.len(), 2, "{text}");
    assert!(text.contains("`edge` is `@strict`") && text.contains("dpdx"), "{text}");
    assert!(text.contains("`filtered` is `@strict`"), "{text}");
    assert!(!text.contains("`own`"), "{text}");
}

/// A strict kernel gives the CPU's bits (docs/plans/engine-at-build.md, phase 1 step 10):
/// toolchain/tests/strict_bits runs std's strict functions, a division, a root and a remainder for
/// 4,096 arguments (the hard ones and random normals of every exponent) in a `@strict` kernel, and
/// the CPU computes the same; no result differs but a subnormal the GPU flushes (the strict model).
#[test]
#[ignore = "needs a GPU"]
fn a_strict_kernel_gives_the_cpus_bits() {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/strict_bits");
    let built = wrela_driver::build_for(&pkg, &wrela_tests::testing_for(wrela_abi::Target::HERE));
    assert!(
        !built.has_errors(),
        "{}",
        wrela_diag::render::render_all(&built.sources, &built.diagnostics)
    );
    let dir = wrela_tests::page_scratch_root().join("strict-bits");
    let _ = std::fs::remove_dir_all(&dir);
    built.write_to(&dir).expect("write the build");
    let mut host = Host::load(&dir).expect("load");
    const FUNCTIONS: [&str; 19] = [
        "sin",
        "cos",
        "tan",
        "exp",
        "exp2",
        "log",
        "log2",
        "sinh",
        "cosh",
        "tanh",
        "atan",
        "asin",
        "acos",
        "atan2",
        "pow",
        "sqrt",
        "/",
        "%",
        "x / (1 + |x|)",
    ];
    for _ in 0..20 {
        host.run_frames(&crate::frame_times(1), 16, 16).expect("a frame");
        match host.call_export("differ", &[]).expect("differ").as_slice() {
            [Value::I32(-1), _] => {}
            [Value::I32(n), Value::I32(at)] => {
                let (arg, f) = (*at as u32 / 19, *at as u32 % 19);
                assert_eq!(
                    *n, 0,
                    "{n} results differ from the CPU's, first `{}` of argument {arg}",
                    FUNCTIONS[f as usize]
                );
                return;
            }
            other => panic!("differ returned {other:?}"),
        }
    }
    panic!("the results never arrived")
}
