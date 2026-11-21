//! GPU programs run by the native host: a draw whose shaders read buffers (bound in the order
//! the pipeline declares them), `len()` of a storage buffer, and what the WGSL back end must
//! take care with (toolchain/tests/shaders). Need a GPU:
//! `cargo test -p wrela-tests --test suite render:: -- --ignored`.

use crate::built;
use wrela_host::{Host, Value};

#[test]
#[ignore = "needs a GPU"]
fn shaders_read_buffers() {
    let run = Host::load(built("render")).expect("load").run_frames(&[0.0], 16, 4).expect("run");
    let px = |x: usize, y: usize| &run.frame[4 * (y * 16 + x)..4 * (y * 16 + x) + 4];
    // Half of red on the left, half of blue on the right (rgba8unorm rounds 127.5 up).
    assert_eq!(px(2, 1), [128, 0, 0, 128]);
    assert_eq!(px(12, 2), [0, 0, 128, 128]);
}

#[test]
#[ignore = "needs a GPU"]
fn lengths_of_runs_and_buffers() {
    let mut host = Host::load(built("lengths")).expect("load");
    // On the CPU: a run's length, and an array's.
    match host.call_export("lens", &[]).expect("lens").as_slice() {
        [Value::F32(sum), Value::F32(len), _] => assert_eq!((*sum, *len), (10.0, 4.0)),
        other => panic!("lens returned {other:?}"),
    }
    // On the GPU: a buffer of 10 elements has length 10, whatever the host rounds its size to.
    host.run_frames(&[0.0], 4, 4).expect("run");
    let out = host.buffers()[0];
    let counts = wrela_tests::u32s(&host.read_buffer(out).expect("out"));
    assert_eq!(&counts[..4], &[10, 11, 12, 13]);
}

/// Without a GPU too: the shader writers take every shader (and the platforms' offline tools
/// what they write, `wrela_tests::offline_check`).
#[test]
fn shaders_build() {
    built("shaders");
}

/// GPU code at the edges of what the back ends lower: each builds (its WGSL valid), rather
/// than failing with an internal error, for each target. Without a GPU.
#[test]
fn gpu_code_at_its_edges_builds() {
    let kernels = [
        // A tuple none of whose elements has a value: no empty WGSL struct.
        ("unit_tuple", "", "let t = (2.0, ((), ()))\n    out[id] = t.0"),
        // A matrix element, in a pipeline that makes no `vec2` of its own.
        ("matrix_element", "", "out[id] = u.m[1][1]"),
        // A helper that never returns: no result to take from it.
        (
            "endless_helper",
            "fn spin(x: f32) -> f32 {\n    loop {\n    }\n}\n",
            "if id.x > 100000 {\n        out[id] = spin(1.0)\n    }",
        ),
        // Constant indexes past the end: they trap on the CPU, and WGSL rejects them as
        // constants, so the GPU reads them at run time (robust access).
        (
            "constant_index_outside",
            "const K: u32 = 5\n",
            "let a = [1.0, 2.0, 3.0]\n    let v = vec3(1.0, 2.0, 3.0)\n    \
             if id.x > 100000 {\n        out[id] = a[5] + v[K] + u.m[K].x + a[1]\n    }",
        ),
    ];
    for (name, items, body) in kernels {
        let src = format!(
            "use std::gpu::{{GlobalId, GpuBuffer, Slots, buffer, dispatch}}

struct U: Copy + GpuData {{
    m: mat2,
}}

{items}

@compute(64)
fn k(u: U, out: mut Slots<f32>, id: GlobalId) {{
    {body}
}}

pub fn frame(time: f32, width: u32, height: u32) {{
    var out: GpuBuffer<f32> = buffer(64)
    let u = U {{ m: mat2(vec2(1.0, 2.0), vec2(3.0, 4.0)) }}
    dispatch(k.bind(u: u, out: mut out), groups: 1)
}}
"
        );
        // Each target's writer: Metal's shading language and SPIR-V.
        for target in wrela_abi::Target::ALL {
            let pkg = crate::package(&format!("edges/{name}"), &src);
            if let Err(e) = wrela_tests::build_for(&pkg, target) {
                panic!("{name} doesn't build for {}:\n{e}", target.name());
            }
        }
    }
}

#[test]
fn many_early_returns_nest_little() {
    // 130 returns in a row, and a value from before them used after them: once inlined, each
    // return puts what follows under one more `if`, and a shader compiler takes only so many
    // levels of braces (WGSL's 127).
    let mut pick = String::from("fn pick(x: f32) -> f32 {\n    let base = x * 2.0\n");
    for k in 0..130 {
        pick += &format!("    if x < {k}.5 {{\n        return {k}.0\n    }}\n");
    }
    pick += "    base\n}\n";
    let src = format!(
        "use std::gpu::{{GlobalId, GpuBuffer, Slots, buffer, dispatch}}

{pick}
@compute(64)
fn k(out: mut Slots<f32>, id: GlobalId) {{
    out[id] = pick(f32(id.x))
}}

pub fn frame(time: f32, width: u32, height: u32) {{
    var out: GpuBuffer<f32> = buffer(64)
    dispatch(k.bind(out: mut out), groups: 1)
}}
"
    );
    let built =
        wrela_tests::build_for(&crate::package("early_returns", &src), wrela_abi::Target::Metal)
            .unwrap_or_else(|e| panic!("doesn't build:\n{e}"));
    let mut shaders = 0;
    for (path, bytes) in &built.files {
        if !path.ends_with(".metal") {
            continue;
        }
        shaders += 1;
        let (mut depth, mut deepest) = (0i32, 0);
        for c in String::from_utf8_lossy(bytes).chars() {
            match c {
                '{' => {
                    depth += 1;
                    deepest = deepest.max(depth);
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        assert!(deepest <= 32, "{path} nests {deepest} deep");
    }
    assert_eq!(shaders, 1);
}

#[test]
#[ignore = "needs a GPU"]
fn shaders_get_the_values_the_program_means() {
    let mut host = Host::load(built("shaders")).expect("load");
    let run = host.run_frames(&[0.0], 16, 4).expect("run");
    let px = |x: usize, y: usize| &run.frame[4 * (y * 16 + x)..4 * (y * 16 + x) + 4];
    // A vertex output of only a `ClipPosition` covers the screen; the generic pair, drawn with
    // `f32` then `vec4`, the bottom left and bottom right.
    assert_eq!(px(3, 0), [0, 0, 255, 255]);
    assert_eq!(px(3, 3), [255, 0, 0, 255]);
    assert_eq!(px(12, 3), [0, 255, 0, 255]);
    let out = host.buffers()[0];
    let got = wrela_tests::f32s(&host.read_buffer(out).expect("out"));
    // Integer `/ 0` gives the dividend, a shift takes its amount modulo 32, and `1.0 / 0.0` and
    // `3e38 * 10.0` are infinite; then `^` of bools, `-` of a matrix, `select` of enums, an enum
    // in a struct, a uniform after one with no value, the last row of a 80 KB table, a buffer's
    // length through a closure, a second `GlobalId`, and constants the build computed.
    let want = [7.0, 2.0, 5.0, 6.0, 1.0, -5.0, 6.0, 7.5, 3.0, 9.0, 3.0, 11.0, 9.5];
    assert_eq!(&got[..want.len()], &want);
}

/// The pipeline-count query (§18.9): each pipeline's entry points with their type arguments, and
/// where CPU code dispatches or draws it. Without a GPU.
#[test]
fn the_pipeline_count_query_reports_each_instantiation() {
    let out = wrela_driver::check(&wrela_tests::repo_root().join("toolchain/tests/shaders"));
    assert!(!out.has_errors());
    let shown: Vec<(String, Vec<String>)> =
        out.pipelines.iter().map(|p| (p.entries.join(" + "), p.sites.clone())).collect();
    let want = [
        ("step", "main.wrela:183:5"),
        ("cover + blue", "main.wrela:188:5"),
        ("half::<f32> + shade::<f32>", "main.wrela:189:5"),
        ("half::<vec4> + shade::<vec4>", "main.wrela:195:5"),
    ];
    let want: Vec<(String, Vec<String>)> =
        want.iter().map(|(e, s)| (e.to_string(), vec![s.to_string()])).collect();
    assert_eq!(shown, want);
    let shade = out.pipelines.iter().filter(|p| p.names.contains(&"shade".to_string())).count();
    assert_eq!(shade, 2, "`shade` is instantiated twice: with f32 and with vec4");
}

/// A trusted build's shaders (`wrela build --trusted-shaders`) have none of the run-time checks
/// a build's have: no index clamped into a runtime array's length, no loop bounded.
#[test]
fn a_trusted_build_writes_its_shaders_without_checks() {
    let src = "use std::gpu::{GlobalId, GpuBuffer, Slots, buffer, dispatch}

@compute(64)
fn k(a: [f32], out: mut Slots<f32>, id: GlobalId) {
    var x = 0.0
    var i: u32 = 0
    while i < u32(a[0]) {
        x += a[id.x + i]
        i += 1
    }
    out[id] = x
}

pub fn frame(time: f32, width: u32, height: u32) {
    let a: GpuBuffer<f32> = buffer(64)
    var out: GpuBuffer<f32> = buffer(64)
    dispatch(k.bind(a, out: mut out), groups: 1)
}
";
    let pkg = crate::package("trusted_shaders", src);
    // Each target's shaders, built as `wrela build --target` does.
    let shaders = |target: wrela_abi::Target, trusted_shaders: bool| -> Vec<u8> {
        let how = wrela_driver::BuildFor { trusted_shaders, ..wrela_driver::BuildFor::new(target) };
        let out = wrela_driver::build_for(&pkg, &how);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let ext = format!(".{}", wrela_shader::extension(target));
        out.files.iter().filter(|(n, _)| n.ends_with(&ext)).flat_map(|(_, b)| b.clone()).collect()
    };
    let (checked, trusted) =
        (shaders(wrela_abi::Target::Metal, false), shaders(wrela_abi::Target::Metal, true));
    let (checked, trusted) = (String::from_utf8_lossy(&checked), String::from_utf8_lossy(&trusted));
    for check in ["_buffer_sizes.size", "loop_bound"] {
        assert!(checked.contains(check), "a build checks: no `{check}`");
        assert!(!trusted.contains(check), "a trusted build doesn't: `{check}`");
    }
    // SPIR-V's checks are instructions: a trusted build's shader is smaller.
    let checked = shaders(wrela_abi::Target::Vulkan, false);
    let trusted = shaders(wrela_abi::Target::Vulkan, true);
    assert_eq!(&checked[..4], &[0x03, 0x02, 0x23, 0x07], "SPIR-V");
    assert!(trusted.len() < checked.len(), "{} against {}", trusted.len(), checked.len());
}
