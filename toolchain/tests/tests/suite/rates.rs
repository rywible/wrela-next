//! The compiler's bar (docs/design.md; docs/rates.md): what the compiler moves to an earlier rate
//! by itself, on the fixtures that put the hand's moves back as first written
//! (toolchain/tests/rates), and what it says where it can't (W0014, language.md §18.13).

use crate::{scratch, write_package};
use wrela_abi::Target;
use wrela_diag::codes;

/// A rates fixture built for `target` with `--set MOVED=<moved>`, as a shipped build is.
fn fixture(name: &str, moved: bool, target: Target) -> wrela_driver::Output {
    let pkg = wrela_tests::repo_root().join("toolchain/tests/rates").join(name);
    let how = wrela_driver::BuildFor {
        set: vec![("MOVED".into(), moved.to_string())],
        ..wrela_driver::BuildFor::new(target)
    };
    let out = wrela_driver::build_for(&pkg, &how);
    assert!(
        !out.has_errors(),
        "{name}: {}",
        wrela_diag::render::render_all(&out.sources, &out.diagnostics)
    );
    out
}

/// The text of the build's shader for the pipeline named `pipeline` (Metal's).
fn shader(out: &wrela_driver::Output, pipeline: &str) -> String {
    let m = wrela_tests::built_manifest(out);
    let p = m.pipelines.iter().find(|p| p.name == pipeline).expect("the pipeline");
    String::from_utf8_lossy(wrela_tests::file(out, &p.shader)).into_owned()
}

/// Each pipeline's kept bytes, by name.
fn kept(out: &wrela_driver::Output) -> Vec<(String, u32)> {
    let m = wrela_tests::built_manifest(out);
    m.pipelines.iter().map(|p| (p.name.clone(), p.kept)).collect()
}

/// Row 1: a pixel program that asks for a sum over 6,561 constant waves where it uses it has the
/// shader the hand's `const` gave it, byte for byte, and no loop.
#[test]
fn a_sum_of_constants_is_worked_out_when_the_game_is_built() {
    let first = fixture("sum", false, Target::Metal);
    let moved = fixture("sum", true, Target::Metal);
    let (a, b) = (shader(&first, "cover+foam"), shader(&moved, "cover+foam"));
    assert!(!a.contains("while"), "the sum's loop is still in the pixel program:\n{a}");
    assert_eq!(a, b, "as first written and as moved differ");
    assert!(first.diagnostics.is_empty(), "{:?}", first.diagnostics);
}

/// Row 2: a table of constants read at an index the kernel works out is kept once, in Metal's
/// constant memory, not built in each invocation's: the kernel keeps no bytes for it, on either
/// target.
#[test]
fn a_constant_table_is_kept_once() {
    for target in [Target::Metal, Target::Vulkan] {
        let first = fixture("table", false, target);
        for (name, bytes) in kept(&first) {
            assert_eq!(bytes, 0, "{target:?}: `{name}` keeps {bytes} bytes an invocation");
        }
    }
    let text = shader(&fixture("table", false, Target::Metal), "step");
    // A table at program scope in the `constant` address space, read in place by the kernel.
    let table = text
        .lines()
        .find_map(|l| l.strip_prefix("constant ").and_then(|r| r.split_whitespace().nth(1)))
        .unwrap_or_else(|| panic!("no table in constant memory:\n{text}"));
    assert!(
        text.contains(&format!("{table}.inner[")),
        "the kernel doesn't read the table `{table}` in place:\n{text}"
    );
}

/// A loop of constants longer than the build runs one for is left to the GPU, and a shipped
/// build says so at the loop (W0014), naming the move: a `const`.
#[test]
fn a_loop_of_constants_past_the_fuel_is_said() {
    let dir = scratch("rates/fuel");
    write_package(
        &dir,
        "fuel",
        "use std::gpu::{ClipPosition, FragCoord, VertexIndex, begin_screen_pass, draw}

fn spin() -> f32 {
    var a = 0.5
    for _i in 0..16777216 {
        a = fract(a * 1.37 + 0.11)
    }
    a
}

@vertex
fn cover(v: VertexIndex) -> ClipPosition {
    let x = f32((v.index & 1) * 4) - 1.0
    let y = f32((v.index & 2) * 2) - 1.0
    ClipPosition { position: vec4(x, y, 0.0, 1.0) }
}

@fragment
fn paint(pixel: FragCoord) -> vec4 {
    vec4(spin() * pixel.position.x, 0.0, 0.0, 1.0)
}

pub fn frame(time: f32, width: u32, height: u32) {
    let screen = begin_screen_pass(clear: vec4(0.0))
    draw(screen, cover, paint, vertices: 3)
    screen.present()
}
",
    );
    let out = wrela_tests::shipped_for(&dir, Target::Metal);
    let said: Vec<_> = out.diagnostics.iter().filter(|d| d.code == codes::W0014).collect();
    assert_eq!(said.len(), 1, "{:?}", out.diagnostics);
    assert!(said[0].message.contains("loop of constants"), "{}", said[0].message);
    assert!(said[0].help.iter().any(|h| h.contains("`const`")), "{:?}", said[0]);
    assert!(!out.has_errors(), "a warning, not an error");
}

/// A loop to a count the draw gives is said (W0014); held to a `@budget(work: n)`, it isn't.
#[test]
fn a_count_the_draw_gives_is_said_unless_budgeted() {
    let program = |budget: &str| {
        format!(
            "use std::gpu::{{ClipPosition, FragCoord, VertexIndex, begin_screen_pass, draw}}

struct Lights: Copy + GpuData {{
    at: [vec4; 4],
    count: u32,
}}

@vertex
fn cover(v: VertexIndex) -> ClipPosition {{
    let x = f32((v.index & 1) * 4) - 1.0
    let y = f32((v.index & 2) * 2) - 1.0
    ClipPosition {{ position: vec4(x, y, 0.0, 1.0) }}
}}

{budget}
fn shine(p: vec3, lights: Lights) -> f32 {{
    var c = 0.1
    for i in 0..4 {{
        if i >= lights.count {{
            break
        }}
        let d = lights.at[i].xyz - p
        let e = exp(-dot(d, d))
        c += smoothstep(0.2, 0.3, e * pow(abs(sin(d.x)), 2.5) / length(d))
        c += smoothstep(0.5, 0.6, e * pow(abs(cos(d.y)), 1.5) / length(d + 0.1))
    }}
    c
}}

@fragment
fn lit(pixel: FragCoord, lights: Lights) -> vec4 {{
    vec4(vec3(shine(vec3(pixel.position.xy * 0.01, 0.0), lights)), 1.0)
}}

pub fn frame(time: f32, width: u32, height: u32) {{
    let screen = begin_screen_pass(clear: vec4(0.0))
    let lights = Lights {{ at: [vec4(0.0); 4], count: 0 }}
    draw(screen, cover, lit.bind(lights: lights), vertices: 3)
    screen.present()
}}
"
        )
    };
    let said = |name: &str, budget: &str| {
        let dir = scratch(name);
        write_package(&dir, "count", &program(budget));
        let out = wrela_tests::shipped_for(&dir, Target::Metal);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        out.diagnostics.into_iter().filter(|d| d.code == codes::W0014).collect::<Vec<_>>()
    };
    let loud = said("rates/count", "");
    assert_eq!(loud.len(), 1, "{loud:?}");
    assert!(loud[0].message.contains("count the command gives"), "{}", loud[0].message);
    assert!(said("rates/count-budgeted", "@budget(work: 100000)").is_empty());
}

/// Something heavy at every pixel is said at the deepest call that does it (W0014), whoever
/// owns the fix: eight value noises a pixel in a rock's look, at its call of the noise. Held to
/// a `@budget(work: n)`, it isn't.
#[test]
fn a_heavy_look_at_every_pixel_is_said_unless_budgeted() {
    let program = |budget: &str| {
        format!(
            "use std::gpu::{{ClipPosition, FragCoord, VertexIndex, begin_screen_pass, draw}}

fn hash(c: vec3i) -> f32 {{
    var h = u32(c.x) * 0x8da6b343 ^ u32(c.y) * 0xd8163841 ^ u32(c.z) * 0xcb1ab31f
    h = (h ^ (h >> 15)) * 0x2c1b3c6d
    f32(h >> 8) / 16777216.0
}}

fn noise(p: vec3) -> f32 {{
    let i = vec3i(floor(p))
    let u = fract(p)
    let a = mix(hash(i), hash(i + vec3i(1, 0, 0)), u.x)
    let b = mix(hash(i + vec3i(0, 1, 0)), hash(i + vec3i(1, 1, 0)), u.x)
    let c = mix(hash(i + vec3i(0, 0, 1)), hash(i + vec3i(1, 0, 1)), u.x)
    let d = mix(hash(i + vec3i(0, 1, 1)), hash(i + vec3i(1, 1, 1)), u.x)
    mix(mix(a, b, u.y), mix(c, d, u.y), u.z)
}}

{budget}
fn grain(p: vec3) -> f32 {{
    var sum = 0.0
    var scale = 4.0
    for _i in 0..8 {{
        sum += noise(p * scale) / scale
        scale *= 2.0
    }}
    sum
}}

@vertex
fn cover(v: VertexIndex) -> ClipPosition {{
    let x = f32((v.index & 1) * 4) - 1.0
    let y = f32((v.index & 2) * 2) - 1.0
    ClipPosition {{ position: vec4(x, y, 0.0, 1.0) }}
}}

@fragment
fn rock(pixel: FragCoord) -> vec4 {{
    vec4(vec3(grain(vec3(pixel.position.xy * 0.01, 1.0))), 1.0)
}}

pub fn frame(time: f32, width: u32, height: u32) {{
    let screen = begin_screen_pass(clear: vec4(0.0))
    draw(screen, cover, rock, vertices: 3)
    screen.present()
}}
"
        )
    };
    let said = |name: &str, budget: &str| {
        let dir = scratch(name);
        write_package(&dir, "rock", &program(budget));
        let out = wrela_tests::shipped_for(&dir, Target::Metal);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        out.diagnostics.into_iter().filter(|d| d.code == codes::W0014).collect::<Vec<_>>()
    };
    let loud = said("rates/rock", "");
    assert_eq!(loud.len(), 1, "{loud:?}");
    assert!(loud[0].message.contains("calls `noise` here 8 times"), "{}", loud[0].message);
    assert!(said("rates/rock-budgeted", "@budget(work: 100000)").is_empty());
}
