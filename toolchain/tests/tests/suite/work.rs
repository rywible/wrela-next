//! Work and weight (language.md §18.11, compiler/ir/src/work.rs): `wrela pipelines` counts each
//! entry point's most work in one call and each pipeline's weight, the same on every machine;
//! `@budget(work: n)` and `@budget(weight: n)` hold a GPU function to them (E0710).

use crate::{scratch, write_package};

/// A program of one kernel per body, each dispatched: `kernels` are `(name, body)`, the body
/// computing `out[id]` from `x`, an `f32` the invocation has.
fn kernels(name: &str, items: &str, kernels: &[(&str, &str)]) -> std::path::PathBuf {
    let dir = scratch(name);
    let mut src = String::from("use std::gpu::{GlobalId, GpuBuffer, Slots, buffer, dispatch}\n\n");
    src.push_str(items);
    for (k, body) in kernels {
        src.push_str(&format!(
            "\n@compute(64)\nfn {k}(out: mut Slots<f32>, id: GlobalId) {{\n    let x = f32(id.x)\n{body}\n}}\n"
        ));
    }
    src.push_str("\npub fn frame(time: f32, width: u32, height: u32) {\n    var out: GpuBuffer<f32> = buffer(64)\n");
    for (k, _) in kernels {
        src.push_str(&format!("    dispatch({k}.bind(mut out), over: 64)\n"));
    }
    src.push_str("}\n");
    write_package(&dir, name, &src);
    dir
}

/// Each kernel's work, by name.
fn work(dir: &std::path::Path) -> Vec<(String, Option<u64>, u64)> {
    let out = wrela_driver::check_pipelines(dir);
    assert!(!out.has_errors(), "{:?}", out.diagnostics);
    out.pipelines
        .iter()
        .map(|p| (p.names[0].clone(), p.work.first().copied().flatten(), p.weight))
        .collect()
}

fn of(all: &[(String, Option<u64>, u64)], k: &str) -> (Option<u64>, u64) {
    let (_, w, g) = all.iter().find(|(n, ..)| n == k).unwrap_or_else(|| panic!("{k}: {all:?}"));
    (*w, *g)
}

#[test]
fn loops_count_each_pass_and_unbounded_ones_have_no_work() {
    let dir = kernels(
        "work_loops",
        "",
        &[
            (
                "eight",
                "    var a = 0.0\n    for _i in 0..8 {\n        a = a + sin(x + f32(_i))\n    }\n    out[id] = a",
            ),
            (
                "sixteen",
                "    var a = 0.0\n    for _i in 0..16 {\n        a = a + sin(x + f32(_i))\n    }\n    out[id] = a",
            ),
            // Left early on the invocation's own value, but never past 32 passes.
            (
                "capped",
                "    var a = 0.0\n    var i = 0\n    var done = false\n    while i < 32 && !done {\n        a = a + sin(x + a)\n        done = a > 10.0\n        i = i + 1\n    }\n    out[id] = a",
            ),
            // The same with the invocation's flag first (`a && b` is `r = a; if a { r = b }`):
            // the counter still ends it, on the side where the flag is false too.
            (
                "flag_first",
                "    var a = 0.0\n    var i = 0\n    var done = false\n    while !done && i < 32 {\n        a = a + sin(x + a)\n        done = a > 10.0\n        i = i + 1\n    }\n    out[id] = a",
            ),
            // Or the other way: left when either the flag or the counter says so.
            (
                "either",
                "    var a = 0.0\n    var i = 0\n    var go = true\n    while true {\n        if !go || i >= 32 {\n            break\n        }\n        a = a + sin(x + a)\n        go = a < 10.0\n        i = i + 1\n    }\n    out[id] = a",
            ),
            // A bound the loop itself moves by the invocation's value: what the first pass knew
            // of it isn't known on the next.
            (
                "moved",
                "    var i: u32 = 0\n    var n: u32 = 4\n    while i < n {\n        i = i + 1\n        n = n + id.x\n    }\n    out[id] = f32(i)",
            ),
            // Left only on the invocation's own value: no sure exit.
            (
                "open",
                "    var a = 0.0\n    while a < 10.0 {\n        a = a + sin(x + a) + 0.5\n    }\n    out[id] = a",
            ),
        ],
    );
    let all = work(&dir);
    let (eight, sixteen) =
        (of(&all, "eight").0.expect("bounded"), of(&all, "sixteen").0.expect("bounded"));
    let per_pass = sixteen - eight;
    assert!(per_pass >= 8 * 4, "eight more passes, each a sine (4) and more: {all:?}");
    assert!(eight < sixteen && sixteen < 2 * eight + 16, "{all:?}");
    let capped = of(&all, "capped").0.expect("a counter that reaches 32 bounds it");
    assert!(capped >= 32 * 4, "{all:?}");
    let flag_first = of(&all, "flag_first").0.expect("the counter bounds it with the flag first");
    assert!(flag_first.abs_diff(capped) <= 4, "the same loop but for a `!`: {all:?}");
    of(&all, "either").0.expect("a counter in an `||` bounds it");
    assert_eq!(of(&all, "open").0, None, "{all:?}");
    assert_eq!(of(&all, "moved").0, None, "{all:?}");
}

#[test]
fn a_loop_over_a_count_known_at_run_time_is_bounded_by_its_clamp() {
    let items = "struct Steps: Copy + GpuData {\n    limit: u32,\n    signed: i32,\n}\n";
    let dir = scratch("work_clamped");
    let src = format!(
        "use std::gpu::{{GlobalId, GpuBuffer, Slots, buffer, dispatch}}\n\n{items}\n\
@compute(64)\nfn open(s: Steps, out: mut Slots<f32>, id: GlobalId) {{\n    var a = 0.0\n    for _i in 0..s.limit {{\n        a = a + sin(a)\n    }}\n    out[id] = a\n}}\n\n\
@compute(64)\nfn clamped(s: Steps, out: mut Slots<f32>, id: GlobalId) {{\n    var a = 0.0\n    for _i in 0..min(s.limit, 64) {{\n        a = a + sin(a)\n    }}\n    out[id] = a\n}}\n\n\
@compute(64)\nfn masked(s: Steps, out: mut Slots<f32>, id: GlobalId) {{\n    var a = 0.0\n    var i: u32 = 0\n    while i < (s.limit & 31) + 1 {{\n        a = a + sin(a)\n        i += 1\n    }}\n    out[id] = a\n}}\n\n\
@compute(64)\nfn signed(s: Steps, out: mut Slots<f32>, id: GlobalId) {{\n    var a = 0.0\n    for _i in 0..clamp(s.signed, 0, 16) {{\n        a = a + sin(a)\n    }}\n    out[id] = a\n}}\n\n\
@compute(64)\nfn wraps(s: Steps, out: mut Slots<f32>, id: GlobalId) {{\n    var a = 0.0\n    for _i in 0..min(s.limit, 64) - 1 {{\n        a = a + sin(a)\n    }}\n    out[id] = a\n}}\n\n\
pub fn frame(time: f32, width: u32, height: u32) {{\n    var out: GpuBuffer<f32> = buffer(64)\n    let s = Steps {{ limit: width, signed: i32(height) }}\n\
    dispatch(open.bind(s, mut out), over: 64)\n    dispatch(clamped.bind(s, mut out), over: 64)\n    dispatch(masked.bind(s, mut out), over: 64)\n\
    dispatch(signed.bind(s, mut out), over: 64)\n    dispatch(wraps.bind(s, mut out), over: 64)\n}}\n"
    );
    write_package(&dir, "work_clamped", &src);
    let all = work(&dir);
    assert_eq!(of(&all, "open").0, None, "a count read at run time: {all:?}");
    let clamped = of(&all, "clamped").0.expect("min(n, 64) bounds it");
    assert!((64 * 4..64 * 40).contains(&clamped), "64 passes of a sine and a sum: {all:?}");
    let masked = of(&all, "masked").0.expect("(n & 31) + 1 bounds it");
    assert!((32 * 4..33 * 40).contains(&masked), "at most 32 passes: {all:?}");
    let signed = of(&all, "signed").0.expect("clamp(n, 0, 16) bounds it");
    assert!((16 * 4..16 * 40).contains(&signed), "{all:?}");
    // A subtraction may wrap below zero: its count isn't bounded.
    assert_eq!(of(&all, "wraps").0, None, "{all:?}");
}

/// A side of a branch that returns ends its way through the function there: what follows the
/// branch is counted on the other side's way only, and the function's work is the most of every
/// way, as for a branch with an `else` (once, the two were summed).
#[test]
fn a_branch_that_returns_ends_its_way_there() {
    let items = "fn heavy(x: f32) -> f32 {\n    var a = x\n    for _h in 0..16 {\n        a = a + sin(a)\n    }\n    a\n}\n\nfn early(x: f32) -> f32 {\n    if x > 1.0 {\n        return heavy(x)\n    }\n    heavy(x + 1.0)\n}\n\nfn branched(x: f32) -> f32 {\n    if x > 1.0 { heavy(x) } else { heavy(x + 1.0) }\n}\n";
    let dir = kernels(
        "work_returned",
        items,
        &[("returns", "    out[id] = early(x)"), ("elsed", "    out[id] = branched(x)")],
    );
    let all = work(&dir);
    let early = of(&all, "returns").0.expect("bounded");
    let branched = of(&all, "elsed").0.expect("bounded");
    assert_eq!(early, branched, "{all:?}");
}

/// `std::derive::work` in GPU code counts a closure over what the kernel was given (a buffer it
/// reads) as the read it is; in CPU code, where a buffer is a handle, such a closure isn't GPU
/// code, and `work` says so (E0700).
#[test]
fn work_counts_a_closure_over_a_kernels_buffer() {
    let dir = scratch("work_of_a_read");
    let src = "use std::derive::work\nuse std::gpu::{GlobalId, GpuBuffer, GpuSpan, Slots, buffer, dispatch}\n\n\
fn twice(xs: GpuSpan<f32>, i: u32) -> f32 {\n    xs[i] * 2.0\n}\n\n\
@compute(64)\nfn counted(xs: GpuSpan<f32>, out: mut Slots<f32>, id: GlobalId) {\n    out[id] = f32(work(|i: u32| twice(xs, i)))\n}\n\n\
pub fn frame(time: f32, width: u32, height: u32) {\n    let xs: GpuBuffer<f32> = buffer(64)\n    var out: GpuBuffer<f32> = buffer(64)\n\
    dispatch(counted.bind(xs.items(), mut out), over: 64)\n}\n";
    write_package(&dir, "work_of_a_read", src);
    let all = work(&dir);
    assert!(of(&all, "counted").0.is_some(), "{all:?}");
    let cpu = scratch("work_of_a_handle");
    let src = "use std::derive::work\nuse std::gpu::{GpuBuffer, GpuSpan, buffer}\n\n\
fn twice(xs: GpuSpan<f32>, i: u32) -> f32 {\n    xs[i] * 2.0\n}\n\n\
pub fn frame(time: f32, width: u32, height: u32) {\n    let b: GpuBuffer<f32> = buffer(64)\n    let xs = b.items()\n\
    let w = work(|i: u32| twice(xs, i))\n    if w == 0 {\n        let c: GpuBuffer<f32> = buffer(1)\n    }\n}\n";
    write_package(&cpu, "work_of_a_handle", src);
    let (codes, all) = crate::checked(&cpu);
    assert!(codes.contains(&"E0700".to_string()), "{all}");
}

/// A bound may be loose, never low: the host's step guard trusts it. A `break` or `continue` on
/// a value the count doesn't know leaves what it knew with its loop, so what follows the loop
/// (or its next pass) is counted for every way out, not only the last pass's.
#[test]
fn a_loop_left_early_counts_what_follows_for_every_way_out() {
    // The heavy step: 16 sines.
    let items = "fn heavy(x: f32) -> f32 {\n    var a = x\n    for _h in 0..16 {\n        a = a + sin(a)\n    }\n    a\n}\n";
    let dir = kernels(
        "work_left_early",
        items,
        &[
            // The opening's cloud march as first written (2026-11-16, stated at 2,498 work for
            // about 1.3 million): on to the next marked cell, then a heavy step. At most 8 cells
            // of 4 steps run the heavy step: 32 times.
            (
                "cells",
                "    var i: u32 = 0\n    var j: u32 = 0\n    var a = x\n    for _s in 0..64 {\n        for _k in 0..8 {\n            if i >= 8 || ((id.x >> i) & u32(1)) != u32(0) {\n                break\n            }\n            i += 1\n        }\n        if i >= 8 {\n            break\n        }\n        a = heavy(a + f32(j))\n        j += 1\n        if j >= 4 {\n            j = 0\n            i += 1\n        }\n    }\n    out[id] = a",
            ),
            // The same loop with one counter: the bound it was meant to have.
            (
                "counted",
                "    var a = x\n    for k in 0..32 {\n        if ((id.x >> (k / 4)) & u32(1)) != u32(0) {\n            a = heavy(a + f32(k % 4))\n        }\n    }\n    out[id] = a",
            ),
            // A count a `break` may leave anywhere from 0 to 32, then a branch on it.
            (
                "after",
                "    var n: u32 = 0\n    for k in 0..32 {\n        if f32(k) > x {\n            break\n        }\n        n += 1\n    }\n    var a = x\n    if n < 32 {\n        a = heavy(a)\n    }\n    out[id] = a",
            ),
            // The same loop without the branch, and the heavy step alone.
            (
                "before",
                "    var n: u32 = 0\n    for k in 0..32 {\n        if f32(k) > x {\n            break\n        }\n        n += 1\n    }\n    out[id] = x + f32(n)",
            ),
            ("plain", "    out[id] = heavy(x)"),
            ("none", "    out[id] = x"),
            // A `continue` that sets the counter back: no sure exit.
            (
                "again",
                "    var i: u32 = 0\n    var a = x\n    while i < 8 {\n        a = a + sin(a)\n        if a > 0.5 {\n            i = 0\n            continue\n        }\n        i += 1\n    }\n    out[id] = a",
            ),
            // A clamp whose least is past its value's most: at least 16 passes.
            (
                "floored",
                "    var a = x\n    for _i in 0..clamp(id.x & 3, 16, 32) {\n        a = a + sin(a)\n    }\n    out[id] = a",
            ),
        ],
    );
    let all = work(&dir);
    let step = of(&all, "counted").0.expect("bounded") / 32;
    assert!(step >= 16 * 4, "a heavy step is 16 sines: {all:?}");
    let cells = of(&all, "cells").0.expect("bounded by its 64 passes");
    assert!(cells >= 32 * step, "32 heavy steps at least: {all:?}");
    let after = of(&all, "after").0.expect("bounded");
    let (before, plain) = (of(&all, "before").0.unwrap(), of(&all, "plain").0.unwrap());
    let none = of(&all, "none").0.unwrap();
    assert!(after + none >= before + plain, "the branch after the loop may run: {all:?}");
    assert_eq!(of(&all, "again").0, None, "{all:?}");
    let floored = of(&all, "floored").0.expect("bounded by its clamp");
    assert!(floored >= 16 * 4, "{all:?}");
}

#[test]
fn a_branch_counts_its_costlier_side_unless_its_condition_is_known() {
    let items = "fn heavy(x: f32) -> f32 {\n    var a = x\n    for _i in 0..32 {\n        a = sin(a) + cos(a)\n    }\n    a\n}\n";
    let dir = kernels(
        "work_branches",
        items,
        &[
            ("either", "    out[id] = if x > 3.0 { heavy(x) } else { x + 1.0 }"),
            ("never", "    out[id] = if false { heavy(x) } else { x + 1.0 }"),
            ("plain", "    out[id] = heavy(x)"),
        ],
    );
    let all = work(&dir);
    let (either, plain, never) =
        (of(&all, "either").0.unwrap(), of(&all, "plain").0.unwrap(), of(&all, "never").0.unwrap());
    assert!(either >= plain, "the costlier side counts: {all:?}");
    assert!(never * 10 < plain, "a known branch counts the side it takes: {all:?}");
}

/// A function of about `n` hundred statements, called from many places: big enough that the
/// shader is written with it as a function of its own (`opt::outlined`).
fn big(n: usize) -> String {
    let mut s = String::from("fn big(x: f32) -> f32 {\n    var a = x\n");
    for k in 0..n * 25 {
        s.push_str(&format!("    a = sin(a * {}.5) * cos(a) + sqrt(abs(a))\n", k % 7 + 1));
    }
    s.push_str("    a\n}\n");
    s
}

#[test]
fn a_function_called_from_more_places_weighs_more_unless_its_calls_merge() {
    let dir = kernels(
        "work_weight",
        &big(12),
        &[
            // Each call needs the last's result: four copies.
            ("chain", "    out[id] = big(big(big(big(x))))"),
            // Four taps of it, none needing another's: one call in a loop (`merge_calls`).
            ("taps", "    out[id] = big(x + 1.0) + big(x + 2.0) + big(x + 3.0) + big(x + 4.0)"),
            // A loop that calls it stays a loop.
            (
                "looped",
                "    var a = x\n    for _i in 0..4 {\n        a = a + big(x + f32(_i))\n    }\n    out[id] = a",
            ),
        ],
    );
    let all = work(&dir);
    let (chain, taps, looped) = (of(&all, "chain").1, of(&all, "taps").1, of(&all, "looped").1);
    assert!(chain > 3 * taps, "four copies against one: {all:?}");
    assert!(chain > 3 * looped, "a loop of one call is one copy: {all:?}");
    // About the same work either way: merging changes what's compiled, not what runs.
    let (a, b) = (of(&all, "chain").0.unwrap(), of(&all, "taps").0.unwrap());
    assert!(a.abs_diff(b) * 10 < a, "{all:?}");
}

#[test]
fn a_function_over_its_budget_is_e0710() {
    let items = "@budget(work: 50)\nfn heavy(x: f32) -> f32 {\n    var a = x\n    for _i in 0..32 {\n        a = sin(a) + cos(a)\n    }\n    a\n}\n\n@budget(weight: 10)\nfn wide(x: f32) -> f32 {\n    heavy(heavy(x))\n}\n\n@budget(work: 1000)\nfn open(x: f32) -> f32 {\n    var a = x\n    while a < 10.0 {\n        a = a + 1.0\n    }\n    a\n}\n";
    let dir = kernels("work_budget", items, &[("k", "    out[id] = wide(x) + open(x)")]);
    let out = wrela_driver::check_pipelines(&dir);
    let msgs: Vec<String> = out
        .diagnostics
        .iter()
        .filter(|d| d.code == wrela_diag::codes::E0710)
        .map(|d| d.message.clone())
        .collect();
    assert_eq!(msgs.len(), 3, "{:?}", out.diagnostics);
    assert!(
        msgs.iter().any(|m| m.contains("`heavy` can do") && m.contains("over its budget of 50")),
        "{msgs:?}"
    );
    assert!(msgs.iter().any(|m| m.contains("`wide` weighs")), "{msgs:?}");
    assert!(msgs.iter().any(|m| m.contains("`open`'s work has no bound")), "{msgs:?}");
}

/// What the work count proves leaves its check out of the shader (language.md §11.3): a loop it
/// sees end needs no bound, an index it knows is under its table's length no clamp, a division
/// by a constant other than 0 no guard. What it can't prove keeps its check: a loop with no exit
/// it's sure of, an index past what it knows, a divisor it doesn't know.
#[test]
fn checks_the_count_proves_unneeded_are_left_out() {
    let items = "const TABLE: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]\n";
    let dir = kernels(
        "work_proofs",
        items,
        &[
            (
                "proved",
                "    var acc = 0.0\n    for i in 0..min(id.x, 16) {\n        acc += TABLE[i]\n    }\n    out[id] = acc + f32(id.x / 7) + f32(id.x % 5)",
            ),
            (
                "unproved",
                "    var i = id.x\n    var steps = 0.0\n    while i > 1 {\n        i = i / 2\n        steps += TABLE[i]\n    }\n    out[id] = steps + f32(id.x / (id.y + 1))",
            ),
        ],
    );
    let built = wrela_driver::build_for(&dir, &wrela_tests::testing_for(wrela_abi::Target::Metal));
    assert!(!built.has_errors(), "{:?}", built.diagnostics);
    let msl = |entry: &str| -> String {
        built
            .files
            .iter()
            .filter(|(n, _)| n.ends_with(".metal"))
            .map(|(_, b)| String::from_utf8_lossy(b).into_owned())
            .find(|t| t.contains(&format!("kernel void {entry}_e(")))
            .unwrap_or_else(|| panic!("no shader of {entry}"))
    };
    let (proved, unproved) = (msl("proved"), msl("unproved"));
    let count = |t: &str, what: &str| t.matches(what).count();
    assert_eq!(count(&proved, "loop_bound"), 0, "the loop ends by 16:\n{proved}");
    // (`out[id]` keeps its clamp: a buffer's length isn't known when it's compiled.)
    assert_eq!(count(&proved, ", 15u)"), 0, "every index of the table is under 16:\n{proved}");
    assert_eq!(count(&proved, "naga_div("), 0, "by 7:\n{proved}");
    assert_eq!(count(&proved, "naga_mod("), 0, "by 5:\n{proved}");
    assert!(count(&unproved, "loop_bound") > 0, "no exit is sure:\n{unproved}");
    assert!(count(&unproved, ", 15u)") > 0, "`i` isn't known:\n{unproved}");
    assert!(count(&unproved, "naga_div(") > 1, "by `id.y + 1`, which may wrap:\n{unproved}");
}

/// What a strict module weighs a built-in at (`wrela_ir::work::strict_builtin_cost`, which E0712
/// and hoisting take) is the work of std's code that it becomes there: a strict kernel of one
/// call, less a strict kernel that computes only its arguments.
#[test]
fn strict_builtins_cost_what_std_s_code_does() {
    use wrela_ir::Builtin as B;
    let calls = [
        (B::Sqrt, "sqrt(x)"),
        (B::Sin, "sin(x)"),
        (B::Cos, "cos(x)"),
        (B::Tan, "tan(x)"),
        (B::Asin, "asin(x)"),
        (B::Acos, "acos(x)"),
        (B::Atan, "atan(x)"),
        (B::Atan2, "atan2(x, x * 0.5 + 0.25)"),
        (B::Exp, "exp(x)"),
        (B::Exp2, "exp2(x)"),
        (B::Log, "log(x)"),
        (B::Log2, "log2(x)"),
        (B::Sinh, "sinh(x)"),
        (B::Cosh, "cosh(x)"),
        (B::Tanh, "tanh(x)"),
        (B::Pow, "pow(x, x * 0.5 + 0.25)"),
    ];
    let dir = scratch("work_strict_builtins");
    let mut src = String::from("use std::gpu::{GlobalId, GpuBuffer, Slots, buffer, dispatch}\n");
    let mut names = vec!["none".to_string()];
    let body = |k: &str, e: &str| {
        format!(
            "\n@strict\n@compute(64)\nfn {k}(out: mut Slots<f32>, id: GlobalId) {{\n    let x = out[id]\n    out[id] = {e}\n}}\n"
        )
    };
    src.push_str(&body("none", "x"));
    src.push_str(&body("none2", "x * 0.5 + 0.25"));
    names.push("none2".to_string());
    for (i, (_, e)) in calls.iter().enumerate() {
        let k = format!("k{i}");
        src.push_str(&body(&k, e));
        names.push(k);
    }
    src.push_str("\npub fn frame(time: f32, width: u32, height: u32) {\n    var out: GpuBuffer<f32> = buffer(64)\n");
    for k in &names {
        src.push_str(&format!("    dispatch({k}.bind(mut out), over: 64)\n"));
    }
    src.push_str("}\n");
    write_package(&dir, "work_strict_builtins", &src);
    let all = work(&dir);
    let none = of(&all, "none").0.expect("bounded");
    let none2 = of(&all, "none2").0.expect("bounded");
    let mut wrong = Vec::new();
    for (i, (b, e)) in calls.iter().enumerate() {
        let base = if e.contains(',') { none2 } else { none };
        let got = of(&all, &format!("k{i}")).0.expect("bounded") - base;
        let table = wrela_ir::work::strict_builtin_cost(*b, 1);
        if got != table {
            wrong.push(format!("{e}: std's code {got}, the table {table}"));
        }
    }
    assert!(wrong.is_empty(), "the table is not std's code (the kernel of none: {none}):\n{}", wrong.join("\n"));
}
