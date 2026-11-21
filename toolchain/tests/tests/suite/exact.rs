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
            [Value::I32(n)] if *n as u32 <= 4097 => return *n as u32,
            _ => {}
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
