//! What this machine's GPU computes, in a strict kernel (`Checks::exact`: no contraction, and
//! safe math on Metal), for the float operations whose IEEE 754 result is one correctly rounded
//! value, against the CPU's (Rust's `f32`, which rounds each correctly): a square root and a
//! reciprocal for all 2³² inputs, and a division, an add, a subtract and a multiply over a set of
//! difficult operands and a large random set from a fixed seed. An operation that gives the CPU's
//! bits for every input (a NaN for a NaN) is a safe operation: a strict function's result made
//! of safe operations has the same bits on every GPU that passes and on the CPU
//! (docs/plans/engine-at-build.md, phase 1). One that doesn't is written in software for strict
//! functions (std::strict).
//!
//! Each dispatch is `CHUNK` invocations, far under the 50 ms line on both machines.

use super::*;
use wrela_abi::manifest::{BindingStage, Pipeline};
use wrela_shader::build::{BinOp, Builder, Builtin, BuiltinInput, Scalar};

/// Invocations a dispatch: 2²² (16,384 × 4 groups of 64 along x and y).
const CHUNK: u32 = 1 << 22;
const GROUPS: [u32; 3] = [16_384, 4, 1];
const ROW: u32 = 16_384 * 64;

fn binding(b: u32) -> ResourceBinding {
    ResourceBinding {
        binding: b,
        kind: BindingKind::ReadWrite,
        stage: BindingStage::Both,
        format: None,
    }
}

/// Two strict compute pipelines: `unary` writes the bits of `sqrt(x)` and `1 / x` for
/// `x = bitcast(base + i)` (binding 0 holds `base`; 1 and 2 the results); `binary` writes
/// `a / b`, `a + b`, `a - b` and `a * b` of bindings 0 and 1 into 2 to 5.
fn strict_kernels() -> (Manifest, Vec<Vec<u8>>) {
    let checks = wrela_shader::Checks { exact: true, ..wrela_shader::Checks::CHECKED };
    let target = wrela_abi::Target::HERE;
    let mut manifest = Manifest::new("game.wasm");
    let mut shaders = Vec::new();
    // unary
    manifest.pipelines.push(Pipeline::hand_written(
        "unary",
        Stage::Compute { entry: "unary".into(), workgroup_size: [64, 1, 1] },
        (0..3).map(binding).collect(),
        vec![Some(64)],
    ));
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let f = b.scalar(Scalar::F32);
    let u3 = b.vector(Scalar::U32, 3);
    let base = b.storage("base", 0, u, false);
    let sq = b.storage("sq", 1, u, true);
    let rc = b.storage("rc", 2, u, true);
    b.compute("unary", [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |k| {
        let id = k.input(0, u3);
        let x0 = k.extract(u, id, 0);
        let y0 = k.extract(u, id, 1);
        let row = k.u32(ROW);
        let off = k.binary(BinOp::Mul, y0, row);
        let i = k.binary(BinOp::Add, x0, off);
        let zero = k.u32(0);
        let b0 = k.load(u, Builder::element(base, zero));
        let bits = k.binary(BinOp::Add, b0, i);
        let x = k.bitcast(f, bits, Scalar::F32);
        let s = k.builtin(f, Builtin::Sqrt, &[x]);
        let one = k.f32(1.0);
        let r = k.binary(BinOp::Div, one, x);
        let sb = k.bitcast(u, s, Scalar::U32);
        let rb = k.bitcast(u, r, Scalar::U32);
        k.store(Builder::element(sq, i), sb);
        k.store(Builder::element(rc, i), rb);
        k.ret(None);
    });
    let p = manifest.pipelines.last_mut().unwrap();
    p.shader = format!("unary.{}", wrela_shader::extension(target));
    shaders.push(b.write_with(p, target, checks).unwrap().bytes);
    // binary
    manifest.pipelines.push(Pipeline::hand_written(
        "binary",
        Stage::Compute { entry: "binary".into(), workgroup_size: [64, 1, 1] },
        (0..6).map(binding).collect(),
        vec![Some(96)],
    ));
    let mut b = Builder::new();
    let u = b.scalar(Scalar::U32);
    let f = b.scalar(Scalar::F32);
    let u3 = b.vector(Scalar::U32, 3);
    let xa = b.storage("xa", 0, u, false);
    let xb = b.storage("xb", 1, u, false);
    let outs: Vec<_> = (2..6).map(|k| b.storage(&format!("o{k}"), k, u, true)).collect();
    b.compute("binary", [64, 1, 1], &[BuiltinInput::GlobalInvocationId], |k| {
        let id = k.input(0, u3);
        let x0 = k.extract(u, id, 0);
        let y0 = k.extract(u, id, 1);
        let row = k.u32(ROW);
        let off = k.binary(BinOp::Mul, y0, row);
        let i = k.binary(BinOp::Add, x0, off);
        let ab = k.load(u, Builder::element(xa, i));
        let bb = k.load(u, Builder::element(xb, i));
        let a = k.bitcast(f, ab, Scalar::F32);
        let c = k.bitcast(f, bb, Scalar::F32);
        for (op, out) in [BinOp::Div, BinOp::Add, BinOp::Sub, BinOp::Mul].into_iter().zip(&outs) {
            let r = k.binary(op, a, c);
            let rb = k.bitcast(u, r, Scalar::U32);
            k.store(Builder::element(*out, i), rb);
        }
        k.ret(None);
    });
    let p = manifest.pipelines.last_mut().unwrap();
    p.shader = format!("binary.{}", wrela_shader::extension(target));
    shaders.push(b.write_with(p, target, checks).unwrap().bytes);
    (manifest, shaders)
}

fn words(v: &[u8]) -> Vec<u32> {
    v.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()
}

/// `x`, a subnormal flushed to a zero of its sign.
fn flushed(x: f32) -> f32 {
    if x.is_subnormal() { f32::from_bits(x.to_bits() & 0x8000_0000) } else { x }
}

/// The strict model of an operation, what a GPU that passes gives (`STRICT_MODEL`): its
/// subnormal operands taken as zeros of their signs, then IEEE 754's correctly rounded result,
/// but a zero of its sign wherever the exact result (`exact`, in f64, from the flushed operands)
/// is below the least normal f32 in magnitude: flushed before rounding, so a result that would
/// round up to the least normal is a zero too.
pub(crate) fn strict_model(
    op: fn(f32, f32) -> f32,
    exact: fn(f64, f64) -> f64,
    a: f32,
    b: f32,
) -> f32 {
    let (a, b) = (flushed(a), flushed(b));
    let r = op(a, b);
    let e = exact(f64::from(a), f64::from(b));
    if e != 0.0 && e.abs() < f64::from(f32::MIN_POSITIVE) {
        return if e < 0.0 { -0.0 } else { 0.0 };
    }
    flushed(r)
}

/// Whether the GPU's bits `got` are `want`: equal, or both NaNs.
fn same(got: u32, want: f32) -> bool {
    got == want.to_bits() || (want.is_nan() && f32::from_bits(got).is_nan())
}

/// What one operation gave: how many inputs, how many differ from IEEE 754's result, and how
/// many from the strict model's, with the first few of those and the farthest any is from the
/// model's (in floats between them: `u32::MAX` where one is not a number of the model's sign).
#[derive(Default, Debug)]
struct Tally {
    inputs: u64,
    ieee: u64,
    model: u64,
    worst: u32,
    first: Vec<String>,
}

impl Tally {
    fn note(&mut self, got: u32, ieee: f32, model: f32, operands: &[f32], what: &str) {
        self.inputs += 1;
        if !same(got, ieee) {
            self.ieee += 1;
        }
        if same(got, model) {
            return;
        }
        self.model += 1;
        let m = model.to_bits();
        let apart = if f32::from_bits(got).is_finite() && model.is_finite() && (got ^ m) >> 31 == 0
        {
            got.abs_diff(m)
        } else {
            u32::MAX
        };
        self.worst = self.worst.max(apart);
        if self.first.len() < 6 {
            let ops: Vec<String> =
                operands.iter().map(|x| format!("{x:e} ({:#010x})", x.to_bits())).collect();
            self.first.push(format!(
                "{what}({}) = {:e} ({got:#010x}), the model's {model:e} ({:#010x})",
                ops.join(", "),
                f32::from_bits(got),
                model.to_bits()
            ));
        }
    }

    fn say(&self, name: &str) {
        eprintln!(
            "strict {name}: {} inputs, {} differ from IEEE 754 (subnormals kept), {} from the strict model (the farthest by {} floats)",
            self.inputs, self.ieee, self.model, self.worst
        );
        for f in &self.first {
            eprintln!("  {f}");
        }
    }
}

/// The difficult operands: zeros, the least and greatest subnormals and normals, one, values
/// near powers of two, the greatest finite, infinities and a NaN, each with both signs.
fn difficult() -> Vec<f32> {
    let mut v = Vec::new();
    for bits in [
        0u32, 1, 2, 3, 0x7fffff, 0x7ffffe, 0x400000, 0x800000, 0x800001, 0xffffff, 0x1000000,
        0x3f800000, 0x3f7fffff, 0x3f800001, 0x3fffffff, 0x40000000, 0x40000001, 0x7f7fffff,
        0x7f7ffffe, 0x7f000000, 0x00ffffff, 0x0c000000, 0x34000000, 0x4b000000, 0x4b7fffff,
        0x7f800000, 0x7fc00000, 0x3eaaaaab, 0x3f2aaaab, 0x40490fdb, 0x3fb504f3,
    ] {
        v.push(f32::from_bits(bits));
        v.push(f32::from_bits(bits | 0x8000_0000));
    }
    v
}

/// A fixed sequence of 32-bit words (a 64-bit LCG's high half).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

/// Runs `binary` over the pairs `(a[i], b[i])`, a chunk at a time, and tallies its four
/// results against the CPU's.
fn run_binary(gpu: &mut Gpu, a: &[u32], b: &[u32], t: &mut [Tally; 4]) {
    let bytes = 4 * CHUNK;
    for start in (0..a.len()).step_by(CHUNK as usize) {
        let end = (start + CHUNK as usize).min(a.len());
        let mut ca = a[start..end].to_vec();
        let mut cb = b[start..end].to_vec();
        ca.resize(CHUNK as usize, 0x3f800000);
        cb.resize(CHUNK as usize, 0x3f800000);
        let le = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        gpu.write_buffer(1, 0, &le(&ca)).unwrap();
        gpu.write_buffer(2, 0, &le(&cb)).unwrap();
        let bind = |h: u32| Binding::range(h, 0, bytes);
        gpu.dispatch(1, Ok(GROUPS), &[bind(1), bind(2), bind(3), bind(4), bind(5), bind(6)], &[])
            .unwrap();
        gpu.end_frame().unwrap();
        let outs: Vec<Vec<u32>> = (3..7).map(|h| words(&gpu.read_buffer(h).unwrap())).collect();
        for i in 0..end - start {
            let (x, y) = (f32::from_bits(ca[i]), f32::from_bits(cb[i]));
            let ops: [(fn(f32, f32) -> f32, fn(f64, f64) -> f64); 4] = [
                (|a, b| a / b, |a, b| a / b),
                (|a, b| a + b, |a, b| a + b),
                (|a, b| a - b, |a, b| a - b),
                (|a, b| a * b, |a, b| a * b),
            ];
            for (k, (op, exact)) in ops.into_iter().enumerate() {
                let model = strict_model(op, exact, x, y);
                t[k].note(outs[k][i], op(x, y), model, &[x, y], ["div", "add", "sub", "mul"][k]);
            }
        }
    }
}

/// The operations a strict kernel computes as the CPU does on this machine's GPU: each of
/// `sqrt` and `1 / x` for every one of the 2³² inputs, and `/`, `+`, `-` and `*` over the
/// difficult operands' every pair and 2²⁴ random pairs (half of them with a subnormal
/// operand). Prints each operation's tally; holds the operations `SAFE` names to be the CPU's
/// to the bit. `tools/check` runs it on each machine's GPU.
#[test]
#[ignore = "needs a GPU, and a minute"]
fn the_gpu_rounds_the_safe_operations_as_the_cpu_does() {
    let (manifest, shaders) = strict_kernels();
    let mut gpu = Gpu::new(&manifest, &shaders, Timing::Off, false).unwrap();
    let bytes = 4 * CHUNK;
    for h in 1..=6 {
        gpu.create_buffer(h, bytes);
    }
    gpu.create_buffer(7, 4);
    // sqrt and 1 / x, every input.
    let mut sq = Tally::default();
    let mut rc = Tally::default();
    let all = 1u64 << 32;
    let only = std::env::var("WRELA_STRICT_OPS_STRIDE").ok().and_then(|s| s.parse::<u64>().ok());
    let stride = only.unwrap_or(1).max(1);
    let mut base: u64 = 0;
    while base < all {
        gpu.write_buffer(7, 0, &(base as u32).to_le_bytes()).unwrap();
        let bind = |h: u32, n: u32| Binding::range(h, 0, n);
        gpu.dispatch(0, Ok(GROUPS), &[bind(7, 4), bind(1, bytes), bind(2, bytes)], &[]).unwrap();
        gpu.end_frame().unwrap();
        let s = words(&gpu.read_buffer(1).unwrap());
        let r = words(&gpu.read_buffer(2).unwrap());
        for i in 0..CHUNK {
            let x = f32::from_bits((base as u32).wrapping_add(i));
            let sm = strict_model(|a, _| a.sqrt(), |a, _| a.sqrt(), x, 0.0);
            sq.note(s[i as usize], x.sqrt(), sm, &[x], "sqrt");
            let rm = strict_model(|a, _| 1.0 / a, |a, _| 1.0 / a, x, 0.0);
            rc.note(r[i as usize], 1.0 / x, rm, &[x], "recip");
        }
        base += u64::from(CHUNK) * stride;
    }
    // The binary operations.
    let hard = difficult();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for &x in &hard {
        for &y in &hard {
            a.push(x.to_bits());
            b.push(y.to_bits());
        }
    }
    let mut rng = Lcg(0x5eed_0f_57_41c7);
    for k in 0..(1u32 << 24) {
        let mut x = rng.next();
        let mut y = rng.next();
        if k & 1 == 1 {
            // A subnormal operand, or one that makes a subnormal result.
            x &= 0x807f_ffff;
            y = (y & 0x807f_ffff) | (rng.next() & 0x3f80_0000);
        }
        a.push(x);
        b.push(y);
    }
    let mut t: [Tally; 4] = Default::default();
    run_binary(&mut gpu, &a, &b, &mut t);
    // Pairs of significands, each in [1, 2): what std's division asks a GPU that doesn't round
    // its own (`std::math::div_rounded`).
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    for _ in 0..(1u32 << 22) {
        sa.push((rng.next() & 0x007f_ffff) | 0x3f80_0000);
        sb.push((rng.next() & 0x007f_ffff) | 0x3f80_0000);
    }
    let mut sig: [Tally; 4] = Default::default();
    run_binary(&mut gpu, &sa, &sb, &mut sig);
    gpu.check_errors().unwrap();
    sq.say("sqrt");
    rc.say("1 / x");
    for (k, name) in ["a / b", "a + b", "a - b", "a * b"].iter().enumerate() {
        t[k].say(name);
    }
    let tallies = [
        ("sqrt", &sq),
        ("recip", &rc),
        ("div", &t[0]),
        ("add", &t[1]),
        ("sub", &t[2]),
        ("mul", &t[3]),
    ];
    sig[0].say("a / b of significands");
    let here = wrela_abi::Target::HERE;
    for (name, tally) in tallies {
        let safe = SAFE.contains(&name) || (here.rounds_div() && DIVIDES.contains(&name));
        if safe {
            assert_eq!(
                tally.model, 0,
                "`{name}` is held safe, and this GPU differs from the strict model"
            );
        }
    }
    // A target whose division isn't rounded divides by std's code, which sets right a quotient
    // of two significands that is within four floats of the rounded one (`std::math::div_from`).
    assert!(
        sig[0].worst <= 2,
        "a quotient of two significands is {} floats from the rounded one: `std::math::div_from` \
         sets right one within four, and Vulkan allows two and a half",
        sig[0].worst
    );
}

/// The operations held to be the CPU's to the bit in a strict kernel on every machine (the test
/// above), so a strict function may use the GPU's own.
const SAFE: &[&str] = &["add", "sub", "mul"];

/// Those held so only on a target whose GPU rounds its division (`Target::rounds_div`): on
/// another a strict kernel divides by std's code.
const DIVIDES: &[&str] = &["recip", "div"];
