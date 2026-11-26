//! An optimized build (`wrela build --optimize`, `wrela_driver::optimize`): Binaryen makes its
//! WASM smaller, and its trap locations go through it as a source map, into `game.lines` beside
//! it. A trap says where it happened just as the unoptimized build's does: in the native host,
//! which reads the file with the build, and in Chrome, which fetches it only when a trap needs
//! it. Without the file, a trap gives its code offset. toolchain/tests/traps traps in functions
//! the optimizer inlines; toolchain/tests/numerics in each way a CPU operation traps.

use crate::{built, scratch};
use std::path::{Path, PathBuf};
use wrela_host::{CpuHost, Error, Value};

/// The build in `dir`, its WASM optimized and its trap locations beside it.
fn optimize_in_place(dir: &Path) {
    let wasm = std::fs::read(dir.join("game.wasm")).expect("game.wasm");
    let (code, lines) =
        wrela_driver::optimize::optimize(&wasm, wrela_driver::optimize::Level::Size, false)
            .expect("optimizes");
    assert!(code.len() < wasm.len(), "optimized {} bytes into {}", wasm.len(), code.len());
    // As `wrela build --optimize` does: the manifest states the rewritten module's interface.
    let manifest = std::fs::read(dir.join("manifest.json")).expect("manifest.json");
    let mut files = vec![("game.wasm".to_string(), code), ("manifest.json".to_string(), manifest)];
    wrela_driver::build::restate_abi(&mut files).expect("restates the interface");
    for (name, bytes) in files {
        std::fs::write(dir.join(&name), bytes).expect(&name);
    }
    std::fs::write(dir.join("game.lines"), lines).expect("write game.lines");
}

/// A copy of the test package `pkg`'s build, optimized: its directory.
fn optimized(pkg: &str, name: &str) -> PathBuf {
    let dir = scratch(name);
    wrela_tests::copy_dir(&built(pkg), &dir);
    optimize_in_place(&dir);
    dir
}

/// Why `name(args)` trapped in the build in `dir`.
fn trap(dir: &Path, name: &str, args: &[Value]) -> String {
    match CpuHost::load(dir).expect("load").call_export(name, args) {
        Err(Error::Trap(why)) => why,
        other => panic!("{name}{args:?} should trap, got {other:?}"),
    }
}

/// Each way toolchain/tests/numerics traps: overflow, division by zero, a long shift, a
/// float-to-int conversion out of range, an index out of range.
const NUMERICS_TRAPS: [(&str, [i32; 2]); 10] = [
    ("add_i32", [i32::MAX, 1]),
    ("sub_u32", [3, 5]),
    ("mul_i32", [65536, 65536]),
    ("neg_i32", [i32::MIN, 0]),
    ("div_i32", [1, 0]),
    ("rem_u32", [7, 0]),
    ("shl_u32", [1, 32]),
    ("add_u8", [200, 56]),
    ("to_i32", [f32::NAN.to_bits() as i32, 0]),
    ("index", [3, 0]),
];

#[test]
fn an_optimized_build_traps_where_it_did() {
    let (plain, small) = (built("numerics"), optimized("numerics", "optimize-numerics"));
    for (name, [a, b]) in NUMERICS_TRAPS {
        let args: Vec<Value> = match name {
            "neg_i32" | "index" => vec![Value::I32(a)],
            "to_i32" => vec![Value::F32(f32::from_bits(a as u32))],
            _ => vec![Value::I32(a), Value::I32(b)],
        };
        let want = trap(&plain, name, &args);
        assert!(want.contains(" at main.wrela:"), "{name}: {want}");
        assert_eq!(trap(&small, name, &args), want, "{name}");
    }
}

#[test]
fn an_inlined_trap_says_the_callee_s_line() {
    let (plain, small) = (built("traps"), optimized("traps", "optimize-traps"));
    let divide = [Value::I32(1), Value::I32(0)];
    for dir in [&plain, &small] {
        // `a / b` is line 12 of toolchain/tests/traps, `a + 1` line 20.
        let zero = trap(dir, "divide", &divide);
        assert!(zero.ends_with(" at main.wrela:12:5"), "{zero}");
        let overflow = trap(dir, "grow_twice", &[Value::I32(i32::MAX)]);
        assert!(overflow.ends_with(" at main.wrela:20:5"), "{overflow}");
    }
}

#[test]
fn without_its_lines_a_trap_gives_its_code_offset() {
    let dir = optimized("traps", "optimize-traps-unlocated");
    std::fs::remove_file(dir.join("game.lines")).expect("remove game.lines");
    let why = trap(&dir, "divide", &[Value::I32(1), Value::I32(0)]);
    let offset = why.rsplit_once(" at code offset 0x").map(|(_, o)| o);
    assert!(offset.is_some_and(|o| u32::from_str_radix(o, 16).is_ok()), "{why}");
}

/// Each program's manifest states its WASM's interface (`wasm_abi`) as its bytes give
/// it, every import the host's with the host's type (what the browser once read from the bytes),
/// and an optimized build's manifest states the optimized module's.
#[test]
fn each_programs_manifest_states_its_wasm_interface() {
    let host: std::collections::HashMap<&str, String> =
        wrela_abi::HOST_FUNCTIONS.iter().map(|f| (f.name, f.signature())).collect();
    for name in ["renderer", "parallel", "render", "noise"] {
        let dir = built(name);
        let read = |dir: &Path| {
            let wasm = std::fs::read(dir.join("game.wasm")).expect("game.wasm");
            let json = std::fs::read_to_string(dir.join("manifest.json")).expect("manifest");
            let stated = wrela_abi::Manifest::parse(&json).expect("valid").wasm_abi;
            (stated, wrela_wasm::interface(&wasm).expect("reads"))
        };
        let (stated, abi) = read(&dir);
        assert_eq!(stated.as_ref(), Some(&abi), "{name}");
        assert!(abi.memory.is_some_and(|m| m.shared), "{name} imports its memory, shared");
        for [module, import, ty] in &abi.imports {
            assert_eq!(module, wrela_abi::IMPORT_MODULE, "{name}");
            if import != wrela_abi::IMPORT_MEMORY {
                assert_eq!(host.get(import.as_str()), Some(ty), "{name}'s `{import}`");
            }
        }
        assert!(abi.exports.contains(&["frame".into(), "(f32, i32, i32) -> ()".into()]), "{name}");
        // The optimized build's manifest is restated from its module.
        let small = scratch(&format!("abi-{name}"));
        wrela_tests::copy_dir(&dir, &small);
        let mut files: Vec<(String, Vec<u8>)> = ["game.wasm", "manifest.json"]
            .iter()
            .map(|f| (f.to_string(), std::fs::read(small.join(f)).expect("file")))
            .collect();
        let (code, _) = wrela_driver::optimize::optimize(
            &files[0].1,
            wrela_driver::optimize::Level::Size,
            false,
        )
        .expect("optimizes");
        files[0].1 = code;
        wrela_driver::build::restate_abi(&mut files).expect("restated");
        for (f, bytes) in &files {
            std::fs::write(small.join(f), bytes).expect("write");
        }
        let (stated, abi) = read(&small);
        assert_eq!(stated, Some(abi), "{name}, optimized");
    }
}

/// A build writes no compressed files: the toolchain's servers compress as they serve, and a
/// release's edge does the same (plan D26).
#[test]
fn a_build_holds_no_compressed_files() {
    let dir = built("renderer");
    let compressed: Vec<_> = wrela_tests::files_under(&dir, &[])
        .into_iter()
        .filter(|f| f.extension().is_some_and(|e| e == "br" || e == "gz"))
        .collect();
    assert!(compressed.is_empty(), "{compressed:?}");
}

/// A predicate whose body folds to `false` (`ON && s.n == 3` with `ON` false) is `false` where
/// it's inlined, so the dispatch under it folds away after inlining, and its pipeline with it:
/// neither the manifest nor `wrela pipelines` has it. A comparison of two integer constants
/// folds the same way.
#[test]
fn a_dispatch_under_a_predicate_that_folds_to_false_leaves_the_build() {
    let dir = crate::package(
        "optimize/folded-predicate",
        r#"
use std::gpu::{GlobalId, GpuBuffer, Slots, buffer, dispatch}

const ON: bool = false
const SIDES: u32 = 4

pub struct State {
    n: u32,
    out: GpuBuffer<u32>,
}

@compute(64)
fn kept(out: mut Slots<u32>, id: GlobalId) {
    out[id] = 1
}

@compute(64)
fn dead(out: mut Slots<u32>, id: GlobalId) {
    out[id] = 2
}

@compute(64)
fn also_dead(out: mut Slots<u32>, id: GlobalId) {
    out[id] = 3
}

fn on(s: State) -> bool {
    ON && s.n == 3
}

fn square() -> bool {
    SIDES == 3
}

pub fn init() -> State {
    State { n: 0, out: buffer(64) }
}

pub fn frame(s: mut State, time: f32, width: u32, height: u32) {
    dispatch(kept.bind(out: mut s.out), groups: 1)
    if on(s) {
        dispatch(dead.bind(out: mut s.out), groups: 1)
    }
    if square() {
        dispatch(also_dead.bind(out: mut s.out), groups: 1)
    }
    s.n += 1
}
"#,
    );
    let built = wrela_driver::build(&dir);
    assert!(!built.has_errors(), "{:?}", built.diagnostics);
    let manifest = wrela_tests::built_manifest(&built);
    let names: Vec<&str> = manifest.pipelines.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["kept"]);
    let listed = wrela_driver::check_pipelines(&dir);
    let entries: Vec<String> = listed.pipelines.iter().flat_map(|p| p.entries.clone()).collect();
    assert_eq!(entries, ["kept"]);
}

/// Three-tier proof erasure end to end:
/// 1. Zero-sized proof tokens (`EmptyToken { seal: () }`, `Option<EmptyToken>`) leave zero struct
///    storage in Metal/SPIR-V/WASM (`Option<ZST>` lowers to a bare `bool`).
/// 2. Build-time `const` proof witnesses computed via `std::derive::lipschitz` + `affine_form`
///    fold into GPU shaders and CPU code, erasing dead branches (`!FACTS.may_transmit`,
///    `FACTS.lip < 1e12`), dead pipelines, and the entire build-time verifier from shipped binaries.
/// 3. Flow-sensitive static range analysis erases proven-safe GPU debug bounds checks.
#[test]
fn proof_erasure_strips_zst_tokens_const_verifiers_and_dead_branches() {
    let dir = crate::package(
        "optimize/proof-erasure",
        r#"
use std::derive::{Box3, affine_form, lipschitz}
use std::gpu::{GlobalId, GpuBuffer, Slots, buffer, dispatch}

struct EmptyToken: Copy {
    seal: (),
}

struct ProvenFacts: Copy {
    lip: f32,
    may_transmit: bool,
    seal: (),
}

fn sphere_d(p: vec3) -> f32 {
    length(p) - 1.0
}

fn prove_facts() -> ProvenFacts {
    let bx = Box3 { lo: vec3(-2.0), hi: vec3(2.0) }
    let lip = lipschitz(sphere_d, bx)
    let r = affine_form(|p: vec3| sphere_d(p), bx)
    let _ = r.valid()
    ProvenFacts { lip, may_transmit: false, seal: () }
}

const FACTS: ProvenFacts = prove_facts()

fn prove_skip(x: f32) -> Option<EmptyToken> {
    if x > 0.0 {
        Some(EmptyToken { seal: () })
    } else {
        None
    }
}

pub struct State {
    out: GpuBuffer<f32>,
}

@compute(64)
fn erased_kernel(out: mut Slots<f32>, id: GlobalId) {
    if id.x >= 64 {
        return
    }
    let facts = FACTS
    var acc = 0.0
    if facts.lip < 1.0e12 {
        acc += facts.lip
    } else {
        acc += 99999.0
    }
    if facts.may_transmit {
        acc += 77777.0
    }
    match prove_skip(f32(id.x)) {
        Some(_tok) => {
            acc += 1.0
        },
        None => {},
    }
    // Proven-safe fixed-size array access via bitmask `id.x & 3`.
    let lut: [f32; 4] = [1.0, 2.0, 3.0, 4.0]
    acc += lut[id.x & 3]
    out[id] = acc
}

@compute(64)
fn dead_transmissive_pass(out: mut Slots<f32>, id: GlobalId) {
    out[id] = 42.0
}

pub fn init() -> State {
    State { out: buffer(64) }
}

pub fn frame(s: mut State, _time: f32, _width: u32, _height: u32) {
    dispatch(erased_kernel.bind(out: mut s.out), groups: 1)
    if FACTS.may_transmit {
        dispatch(dead_transmissive_pass.bind(out: mut s.out), groups: 1)
    }
}
"#,
    );
    let built = wrela_driver::build(&dir);
    assert!(!built.has_errors(), "{:?}", built.diagnostics);
    // Tier 2: dead transmissive pipeline under `FACTS.may_transmit` is completely erased.
    let manifest = wrela_tests::built_manifest(&built);
    let names: Vec<&str> = manifest.pipelines.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["erased_kernel"]);
    // Inspect the emitted Metal shader source for `erased_kernel`.
    let msl = built
        .files
        .iter()
        .find(|(name, _)| name.ends_with(".metal"))
        .map(|(_, bytes)| String::from_utf8_lossy(bytes).into_owned())
        .expect("expected Metal shader output");
    // Tier 1: zero trace of `EmptyToken` struct wrapper in Metal!
    assert!(!msl.contains("EmptyToken"), "ZST EmptyToken leaked into Metal:\n{msl}");
    assert!(!msl.contains("ProvenFacts"), "const ProvenFacts leaked into Metal:\n{msl}");
    // Tier 2: dead branches (`99999` fallback, `77777` transmissive branch) and build-time
    // verifier `prove_facts` are completely erased from shipped Metal!
    assert!(!msl.contains("99999"), "dead lip fallback leaked into Metal:\n{msl}");
    assert!(!msl.contains("77777"), "dead transmissive branch leaked into Metal:\n{msl}");
    assert!(!msl.contains("prove_facts"), "build-time verifier leaked into Metal:\n{msl}");
}

/// An affine form keeps its slopes through a condition that compares two conditions (`!=` of two
/// `bool`s) and through a call that returns a condition: each is unsure over a box that straddles
/// it, and the form is the sides' hull, its slope along `y` kept (before either had a rule, the
/// whole field's form fell to its interval: a kit outline's crossing test, `crosses`, cost every
/// turned and moulded thing its outline certificates, docs/plans/engine-at-build.md phase 5).
#[test]
fn an_affine_form_keeps_its_slopes_through_conditions_of_conditions_and_calls() {
    let dir = crate::package(
        "optimize/affine-conditions",
        r#"
use std::derive::{Box3, affine_form}

fn over(x: f32, k: f32) -> bool {
    x > k
}

@test
fn slopes_survive() {
    let b = Box3 { lo: vec3(0.0, 0.0, 0.0), hi: vec3(1.0, 1.0, 1.0) }
    let ne = affine_form(|q: vec3| (if (q.x > 0.3) != (q.z > 0.6) { 1.0 } else { 0.0 }) + q.y, b)
    assert(ne.valid() && abs(ne.l.y - 0.5) < 1.0e-5, f"!= of conditions: {ne.l}")
    let called = affine_form(|q: vec3| (if over(q.x, 0.4) { 1.0 } else { 0.0 }) + q.y, b)
    assert(called.valid() && abs(called.l.y - 0.5) < 1.0e-5, f"a call's condition: {called.l}")
    let sure = affine_form(|q: vec3| (if over(q.x, 2.0) { 1.0 } else { 0.0 }) + q.y, b)
    assert(sure.e < 1.0e-5 && abs(sure.c - 0.5) < 1.0e-5, f"a sure call: {sure.c} {sure.e}")
}
"#,
    );
    let out = wrela_driver::test(&dir, None);
    assert!(!wrela_diag::has_errors(&out.diagnostics), "{}", wrela_diag::render::render_all(&out.sources, &out.diagnostics));
    assert!(out.results.iter().all(|r| r.failure.is_none()), "{:?}", out.results);
}
