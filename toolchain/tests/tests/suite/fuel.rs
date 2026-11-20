//! Fuel (compiler/ir/src/fuel.rs, AGENTS.md rule 2): toolchain/tests/fuel's two pipelines. The
//! one whose loop the compiler bounds within a lane's fuel is written with no counter and no
//! flag; the one it can't bound is charged each pass, draws under its fuel, and the frame whose
//! lanes go over ends the run, naming the pipeline and its pass. Over the fuel is a million
//! passes a lane: a few milliseconds without fuel, so the test is safe on a host without it.
//! `cargo test -p wrela-tests --test suite fuel:: -- --ignored`.

use wrela_host::Host;
use wrela_tests::page;

/// Fuel counts a draw with a loop the count can't bound, or whose lanes' bound is past the
/// line's work at the warm clock (`TargetLimits::lane_line_work`); one bounded within it is left
/// as written, and named (W0016) when its bound is over the line's work at an idle clock. `under_line` and
/// `over_line` sit either side of Metal's line; `fixed` is within the fuel; `march` has no
/// bound.
#[test]
fn a_draw_is_charged_only_past_the_lines_work_or_unbounded_and_named_between() {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/fuel");
    for target in [wrela_abi::Target::Metal, wrela_abi::Target::Vulkan] {
        let limits = target.limits();
        let (fuel, line) = (limits.lane_idle_work(), limits.lane_line_work());
        assert_eq!(limits.lane_fuel(), line, "a charged lane's ceiling is an uncounted one's");
        let built = wrela_tests::build_for(&pkg, target).expect("builds");
        let manifest = wrela_abi::Manifest::parse(&String::from_utf8_lossy(
            &built.files.iter().find(|(n, _)| n == "manifest.json").expect("a manifest").1,
        ))
        .expect("valid");
        let named = |name: &str| {
            built.diagnostics.iter().any(|d| {
                d.code == wrela_diag::codes::W0016 && d.message.contains(&format!("`{name}`"))
            })
        };
        for p in &manifest.pipelines {
            // (The manifest's work is counted before the shader is written; these loops have no
            // constants to fold, so it's the same count.)
            let counted = p.work.iter().any(|w| w.is_none_or(|w| w > line));
            let most = p.work.iter().flatten().copied().max().unwrap_or(0);
            assert_eq!(p.debug_flag.is_some(), counted, "{target:?} {}: {:?}", p.name, p.work);
            assert_eq!(named(&p.name), !counted && most > fuel, "{target:?} {}", p.name);
            if target == wrela_abi::Target::Metal {
                let (_, b) = built.files.iter().find(|(n, _)| *n == p.shader).expect("shader");
                assert_eq!(
                    String::from_utf8_lossy(b).contains("wrela_fuel"),
                    counted,
                    "{}",
                    p.name
                );
            }
        }
        let p = |name: &str| manifest.pipelines.iter().find(|p| p.name == name).expect(name);
        assert!(p("cover+march").debug_flag.is_some(), "{target:?}: no bound, so charged");
        if target == wrela_abi::Target::Metal {
            assert!(named("cover+under_line") && p("cover+over_line").debug_flag.is_some());
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_lane_out_of_fuel_ends_the_run_naming_its_pass() {
    let (dir, _) = page("toolchain/tests/fuel", "fuel-native");
    let mut host = Host::load(&dir).expect("load");
    host.run_frames(&crate::frame_times(2), 64, 64).expect("under its fuel, it draws");
    let e = host.run_frames(&crate::frame_times(2), 64, 64).expect_err("over it").to_string();
    assert!(
        e.contains("`cover+march`") && e.contains("`runaway`") && e.contains("ran out of fuel"),
        "{e}"
    );
}
