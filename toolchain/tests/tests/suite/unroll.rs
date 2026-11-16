//! Loops on the GPU written out (compiler/ir/src/unroll.rs): toolchain/tests/unroll's kernels.
//! A loop of a few passes counted from a constant to a constant is written out; one whose end
//! isn't a constant, one left early and one of more passes than the limit stay loops. Each
//! kernel's 64 results, summed, are what the CPU's arithmetic gives.
//! `cargo test -p wrela-tests --test suite unroll:: -- --ignored`.

use wrela_host::{Host, Value};
use wrela_tests::{one_u32, page};

#[test]
#[ignore = "needs a GPU"]
fn small_counted_loops_are_written_out_and_give_the_same_results() {
    let (dir, _) = page("toolchain/tests/unroll", "unroll-native");
    // The loops left in each kernel's MSL, built for Metal on any machine.
    let pkg = wrela_tests::repo_root().join("toolchain/tests/unroll");
    let metal = wrela_tests::build_for(&pkg, wrela_abi::Target::Metal).expect("builds");
    let manifest = wrela_abi::Manifest::parse(&String::from_utf8_lossy(
        &metal.files.iter().find(|(n, _)| n == "manifest.json").expect("a manifest").1,
    ))
    .expect("valid");
    let loops = |name: &str| {
        let p = manifest.pipelines.iter().find(|p| p.name == name).expect(name);
        let (_, msl) = metal.files.iter().find(|(n, _)| *n == p.shader).expect("its shader");
        String::from_utf8_lossy(msl).matches("while(true) {").count()
    };
    assert_eq!(loops("grid"), 0, "a 3 × 3 loop of constants is written out");
    assert_eq!(loops("up_to"), 1, "a loop to a value the invocation has stays");
    assert_eq!(loops("first_over"), 1, "a loop left early stays");
    assert_eq!(loops("long"), 1, "twenty passes stay a loop");

    let mut host = Host::load(&dir).expect("load");
    let times = crate::frame_times(4);
    host.run_frames(&times, 16, 16).expect("run");
    let mut sum = |k: u32| one_u32(&mut host, "sum", &[Value::I32(k as i32)]);
    let ids = 0..64u32;
    let grid: u32 = ids.clone().map(|id| 36 * (id + 1)).sum();
    let up_to: u32 = ids.clone().map(|id| (0..id % 5).sum::<u32>()).sum();
    let first_over: u32 = ids.clone().map(|id| (0..8).find(|i| i * i > id).unwrap_or(99)).sum();
    let long: u32 = ids.map(|id| (0..20).map(|i| i ^ id).sum::<u32>()).sum();
    assert_eq!(sum(0), grid, "grid");
    assert_eq!(sum(1), up_to, "up_to");
    assert_eq!(sum(2), first_over, "first_over");
    assert_eq!(sum(3), long, "long");
}
