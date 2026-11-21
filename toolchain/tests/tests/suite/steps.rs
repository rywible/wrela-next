//! Bounded compute steps (docs/rates.md): each kernel's manifest states the most work an
//! invocation can do, or that the compiler can't say, and the host weighs a dispatch by it
//! before it runs (host/src/step.rs). The kernel that took the M4 down on 2026-11-15 (a medium
//! marched at every pixel with fine steps, in one dispatch: toolchain/tests/steps/march) is
//! refused at its real size on both machines, by arithmetic here and on the GPU by the host.

use wrela_abi::Target;
use wrela_host::step::{Bound, check, ms_of};

/// The fixture `name` built for `target` with `sets`: its manifest's first pipeline's bound.
fn bound_of(name: &str, sets: &[(&str, &str)], target: Target) -> Bound {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/steps").join(name);
    let how = wrela_driver::BuildFor {
        set: sets.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        ..wrela_driver::BuildFor::new(target)
    };
    let out = wrela_driver::build_for(&pkg, &how);
    assert!(
        !out.has_errors(),
        "{name}: {}",
        wrela_diag::render::render_all(&out.sources, &out.diagnostics)
    );
    let m = wrela_tests::built_manifest(&out);
    Bound::of(&m.pipelines[0])
}

/// As first written, the reference's loop runs past what the compiler follows: no bound, so
/// it is refused at any size. With a length the compiler sees, it is refused at 1920x1080 by
/// what its bound says it could take (hundreds of ms on either machine), and runs at 64x64
/// (a millisecond or two), unless the limit is lowered.
#[test]
fn the_reference_that_took_the_machine_down_is_refused_before_it_runs() {
    for target in [Target::Metal, Target::Vulkan] {
        let first = bound_of("march", &[], target);
        let Bound::Unbounded(at) = &first else { panic!("{target:?}: {first:?}") };
        assert!(at.starts_with("main.wrela:"), "{at}");
        let e = check("refer", &first, Some(64 * 64), target).unwrap_err().to_string();
        assert!(e.contains("can't bound") && e.contains(at.as_str()), "{e}");

        let bounded = bound_of("march", &[("STEPS", "2048")], target);
        let Bound::Most(most) = bounded else { panic!("{target:?}: {bounded:?}") };
        let full = 1920 * 1080;
        assert!(ms_of(most, full, target) > 200.0, "{target:?}: {}", ms_of(most, full, target));
        let e = check("refer", &bounded, Some(full), target).unwrap_err().to_string();
        assert!(e.contains("refused a dispatch of `refer` before it ran"), "{e}");
        assert!(check("refer", &bounded, Some(64 * 64), target).is_ok());
    }
}

/// Every calibration kernel has a bound: each loop's count is a constant.
#[test]
fn the_calibration_kernels_are_bounded() {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/steps/calibrate");
    let out = wrela_tests::shipped_for(&pkg, Target::HERE);
    assert!(!out.has_errors(), "{:?}", out.diagnostics);
    let m = wrela_tests::built_manifest(&out);
    assert_eq!(m.pipelines.len(), 7);
    for p in &m.pipelines {
        assert!(matches!(Bound::of(p), Bound::Most(_)), "{}: {:?}", p.name, p.work);
    }
}
