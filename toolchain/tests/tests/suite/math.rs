//! std's math (lib/std/math.wrela: f32 alone, the CPU's and a strict GPU kernel's) against
//! Rust's f64 functions rounded to f32: within its stated error (`std::math::STRICT_ULPS`)
//! everywhere sampled, and the same bits at zeros, infinities and NaNs; and each one-argument
//! function measured for every one of its 2³² arguments (`every_argument_is_within_its_error`,
//! ignored: some minutes on ten cores). The interval interpretation's widening relies on it. Also std's fields (lib/std/field.wrela) against references, at the edge cases
//! where a formula breaks down: `k = 0`, coincident or nested spheres, centres, and coordinates
//! past the i32 range.

use crate::built;
use wrela_host::{CpuHost, Value};
use wrela_tests::{Rng, one_f32};

/// The distance between two f32s in units in the last place of the reference.
fn ulps(got: f32, want: f32) -> f64 {
    if got == want || (got.is_nan() && want.is_nan()) {
        return 0.0;
    }
    if !got.is_finite() || !want.is_finite() {
        return f64::INFINITY;
    }
    let ulp = (f64::from(want.abs()) * f64::from(f32::EPSILON)).max(f64::from(f32::from_bits(1)));
    (f64::from(got) - f64::from(want)).abs() / ulp
}

#[test]
fn std_math_is_within_an_ulp() {
    let mut host = CpuHost::load(built("math")).expect("load");
    let mut rng = Rng::new(42);
    type One = fn(f64) -> f64;
    let ones: [(&str, One, f64, f64); 20] = [
        ("f_sinh", f64::sinh, -90.0, 90.0),
        ("f_sinh", f64::sinh, -1.0, 1.0),
        ("f_cosh", f64::cosh, -90.0, 90.0),
        ("f_cosh", f64::cosh, -1.0, 1.0),
        ("f_tanh", f64::tanh, -12.0, 12.0),
        ("f_tanh", f64::tanh, 1e-30, 1.0),
        ("f_sin", f64::sin, -1e4, 1e4),
        ("f_cos", f64::cos, -1e4, 1e4),
        ("f_tan", f64::tan, -1e2, 1e2),
        ("f_exp", f64::exp, -87.0, 88.0),
        ("f_exp2", f64::exp2, -126.0, 127.0),
        ("f_log", f64::ln, 1e-30, 1e30),
        ("f_log2", f64::log2, 1e-30, 1e30),
        ("f_atan", f64::atan, -1e3, 1e3),
        ("f_asin", f64::asin, -1.0, 1.0),
        ("f_acos", f64::acos, -1.0, 1.0),
        ("f_sin", f64::sin, -4.0, 4.0),
        // Every magnitude, and either sign: the range reduction past 2^20 is exact.
        ("f_sin", f64::sin, 1.0, 3.4e38),
        ("f_cos", f64::cos, 1.0, 3.4e38),
        ("f_tan", f64::tan, 1.0, 3.4e38),
    ];
    let mut report = Vec::new();
    for (name, f, lo, hi) in ones {
        let mut worst = 0.0f64;
        for i in 0..1024 {
            // Log-uniform over positive ranges that span decades, uniform otherwise.
            let x = if lo > 0.0 {
                (lo.ln() + (hi.ln() - lo.ln()) * rng.unit()).exp()
            } else {
                lo + (hi - lo) * rng.unit()
            } as f32;
            // The trigonometric functions' whole range takes both signs.
            let x = if hi > 1e30 && i % 2 == 1 { -x } else { x };
            let x = if i == 0 { 0.0 } else { x };
            let got = one_f32(&mut host, name, &[Value::F32(x)]);
            let want = f(f64::from(x)) as f32;
            let u = ulps(got, want);
            let most = stated(&mut host, name);
            assert!(u <= most, "{name}({x:e}) = {got:e}, want {want:e}: {u} ulps of {most}");
            worst = worst.max(u);
        }
        report.push(format!("  {name}: worst {worst:.2} ulp over [{lo:e}, {hi:e}]"));
    }
    let mut worst = (0.0f64, 0.0f64);
    for _ in 0..1024 {
        let (x, y) = ((rng.unit() * 8.0) as f32, (rng.unit() * 16.0 - 8.0) as f32);
        let p = one_f32(&mut host, "f_pow", &[Value::F32(x), Value::F32(y)]);
        let u = ulps(p, f64::from(x).powf(f64::from(y)) as f32);
        let most = stated(&mut host, "f_pow");
        assert!(u <= most, "pow({x:e}, {y:e}) = {p:e}: {u} ulps of {most}");
        worst.0 = worst.0.max(u);
        let (b, a) = ((rng.unit() * 8.0 - 4.0) as f32, (rng.unit() * 8.0 - 4.0) as f32);
        let t = one_f32(&mut host, "f_atan2", &[Value::F32(b), Value::F32(a)]);
        let u = ulps(t, f64::from(b).atan2(f64::from(a)) as f32);
        let most = stated(&mut host, "f_atan2");
        assert!(u <= most, "atan2({b:e}, {a:e}) = {t:e}: {u} ulps of {most}");
        worst.1 = worst.1.max(u);
    }
    report.push(format!("  f_pow: worst {:.2} ulp; f_atan2: worst {:.2} ulp", worst.0, worst.1));
    println!("std math against f64 references:\n{}", report.join("\n"));
}

/// `STRICT_ULPS`'s order of the functions, by their exports' names.
const STRICT_ORDER: [&str; 15] = [
    "f_sin", "f_cos", "f_tan", "f_exp", "f_exp2", "f_log", "f_log2", "f_sinh", "f_cosh", "f_tanh",
    "f_atan", "f_asin", "f_acos", "f_atan2", "f_pow",
];

/// The error std states of the function its export `name` calls, in ulps (rounded to the CPU's
/// f32 reference here, so half an ulp more).
fn stated(host: &mut CpuHost, name: &str) -> f64 {
    let k = STRICT_ORDER.iter().position(|n| *n == name).expect("a strict function") as u32;
    f64::from(one_f32(host, "stated_ulps", &[Value::I32(k as i32)])) + 0.5
}

/// Each one-argument function of `STRICT_ULPS`, for every one of the 2³² arguments, against
/// std's f64 functions (`std::math::strict_and_reference`), on every core: the most error of
/// each, which it prints, within what `STRICT_ULPS` states. (atan2 and pow: over the 2³² pairs
/// whose second argument is the first's bits with their halves swapped.)
#[test]
#[ignore = "some minutes on ten cores"]
fn every_argument_is_within_its_error() {
    let dir = built("math");
    let only: Option<u32> = std::env::var("WRELA_STRICT_ONLY").ok().and_then(|s| s.parse().ok());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut lines = Vec::new();
    let mut over = Vec::new();
    for k in 0..15u32 {
        if only.is_some_and(|o| o != k) {
            continue;
        }
        const CHUNK: u64 = 1 << 22;
        let next = std::sync::atomic::AtomicU64::new(0);
        let worst = std::sync::Mutex::new((0.0f64, 0u32));
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut host = CpuHost::load(&dir).expect("load");
                    loop {
                        let start = next.fetch_add(CHUNK, std::sync::atomic::Ordering::Relaxed);
                        if start >= 1 << 32 {
                            break;
                        }
                        let args = [
                            Value::I32(k as i32),
                            Value::I32(start as u32 as i32),
                            Value::I32(CHUNK as u32 as i32),
                        ];
                        let (u, at) =
                            match wrela_tests::Exports::call(&mut host, "worst_ulps", &args)
                                .expect("worst_ulps")
                                .as_slice()
                            {
                                [Value::F64(u), Value::I32(at)] => (*u, *at as u32),
                                other => panic!("worst_ulps returned {other:?}"),
                            };
                        let mut w = worst.lock().unwrap();
                        if u > w.0 {
                            *w = (u, at);
                        }
                    }
                });
            }
        });
        let (u, at) = *worst.lock().unwrap();
        let mut host = CpuHost::load(&dir).expect("load");
        let stated = f64::from(one_f32(&mut host, "stated_ulps", &[Value::I32(k as i32)]));
        let name = STRICT_ORDER[k as usize];
        lines.push(format!(
            "  {name}: {u:.3} ulps at most (at {:e}, bits {at:#010x}), stated {stated}",
            f32::from_bits(at)
        ));
        if u > stated {
            over.push(name);
        }
    }
    println!("std's f32 functions, every argument:\n{}", lines.join("\n"));
    assert!(over.is_empty(), "past their stated errors: {over:?}");
}

/// `sqrt_rounded`, a strict kernel's square root, is the CPU's `sqrt` (a subnormal taken as a
/// zero of its sign) for 2²⁰ arguments spread over every exponent, and `rem_exact`, its `%`, is
/// Rust's (C's `fmod`, exact) for 2¹⁶ pairs; `every_sqrt_argument_is_the_cpus` holds the root
/// for all 2³².
#[test]
fn the_strict_root_and_remainder_are_the_cpus() {
    let mut host = CpuHost::load(built("math")).expect("load");
    let (n, first) = match wrela_tests::Exports::call(
        &mut host,
        "sqrt_differences",
        &[Value::I32(0x3f00_0000), Value::I32(1 << 20)],
    )
    .expect("sqrt")
    .as_slice()
    {
        [Value::I32(n), Value::I32(f)] => (*n as u32, *f as u32),
        other => panic!("{other:?}"),
    };
    assert_eq!(n, 0, "sqrt_rounded differs first at {:#010x}", first);
    let mut rng = Rng::new(7);
    for _ in 0..(1 << 16) {
        let a = f32::from_bits(rng.next_u64() as u32);
        let b = f32::from_bits(rng.next_u64() as u32);
        let got = one_f32(&mut host, "f_rem_exact", &[Value::F32(a), Value::F32(b)]);
        let want = a % b;
        assert!(same(got, want), "rem_exact({a:e}, {b:e}) = {got:e}, Rust's {want:e}");
    }
}

/// `sqrt_rounded` for every one of the 2³² arguments.
#[test]
#[ignore = "a minute on ten cores"]
fn every_sqrt_argument_is_the_cpus() {
    let dir = built("math");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    const CHUNK: u64 = 1 << 22;
    let next = std::sync::atomic::AtomicU64::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut host = CpuHost::load(&dir).expect("load");
                loop {
                    let start = next.fetch_add(CHUNK, std::sync::atomic::Ordering::Relaxed);
                    if start >= 1 << 32 {
                        break;
                    }
                    let args = [Value::I32(start as u32 as i32), Value::I32(CHUNK as u32 as i32)];
                    match wrela_tests::Exports::call(&mut host, "sqrt_differences", &args)
                        .expect("sqrt")
                        .as_slice()
                    {
                        [Value::I32(0), _] => {}
                        [Value::I32(n), Value::I32(f)] => {
                            bad.lock().unwrap().push((*n as u32, *f as u32))
                        }
                        other => panic!("{other:?}"),
                    }
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "sqrt_rounded differs from the CPU's: {bad:x?}");
}

/// `sqrt_from` sets the root of every positive normal argument right from the root or either
/// of its neighbours: what the GPU's root, within one ulp, can be. (The CPU's root is right, so
/// `every_sqrt_argument_is_the_cpus` never reaches the correction; this does.) A quick part of
/// it runs with the other tests: the arguments about 1, 2 and 4, where the neighbour below a
/// power of two is half an ulp away, and the least and the greatest.
#[test]
fn roots_about_powers_of_two_are_set_right_from_a_neighbour() {
    let mut host = CpuHost::load(built("math")).expect("load");
    for start in [0x3f7f_0000u32, 0x3fff_0000, 0x407f_0000, 0x0080_0000, 0x7f7f_0000] {
        let args = [Value::I32(start as i32), Value::I32(1 << 17)];
        match wrela_tests::Exports::call(&mut host, "sqrt_from_differences", &args)
            .expect("sqrt_from")
            .as_slice()
        {
            [Value::I32(0), _] => {}
            [Value::I32(n), Value::I32(f)] => {
                panic!("sqrt_from is wrong {n} times from {start:#x}, first at {:#010x}", *f as u32)
            }
            other => panic!("{other:?}"),
        }
    }
}

/// `sqrt_from` for every positive normal argument, from its root and each neighbour.
#[test]
#[ignore = "a minute on ten cores"]
fn every_root_is_set_right_from_a_neighbour() {
    let dir = built("math");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    const CHUNK: u64 = 1 << 22;
    let next = std::sync::atomic::AtomicU64::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut host = CpuHost::load(&dir).expect("load");
                loop {
                    let start = next.fetch_add(CHUNK, std::sync::atomic::Ordering::Relaxed);
                    if start >= 1 << 31 {
                        break;
                    }
                    let args = [Value::I32(start as u32 as i32), Value::I32(CHUNK as u32 as i32)];
                    match wrela_tests::Exports::call(&mut host, "sqrt_from_differences", &args)
                        .expect("sqrt_from")
                        .as_slice()
                    {
                        [Value::I32(0), _] => {}
                        [Value::I32(n), Value::I32(f)] => {
                            bad.lock().unwrap().push((*n as u32, *f as u32))
                        }
                        other => panic!("{other:?}"),
                    }
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "sqrt_from is wrong: {bad:x?}");
}

/// `div_from` for sixteen million pairs (hard ones, subnormal operands, quotients about the
/// least normal and random ones), from the quotient of their significands and each float
/// within three ulps of it: the strict model's quotient every time.
#[test]
fn every_quotient_is_set_right_from_an_estimate() {
    let dir = built("math");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    const CHUNK: u64 = 1 << 18;
    const PAIRS: u64 = 1 << 24;
    let next = std::sync::atomic::AtomicU64::new(0);
    let bad = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut host = CpuHost::load(&dir).expect("load");
                loop {
                    let start = next.fetch_add(CHUNK, std::sync::atomic::Ordering::Relaxed);
                    if start >= PAIRS {
                        break;
                    }
                    let args = [Value::I32(start as u32 as i32), Value::I32(CHUNK as u32 as i32)];
                    match wrela_tests::Exports::call(&mut host, "div_from_differences", &args)
                        .expect("div_from")
                        .as_slice()
                    {
                        [Value::I32(0), _] => {}
                        [Value::I32(n), Value::I32(f)] => {
                            bad.lock().unwrap().push((start, *n as u32, *f as u32))
                        }
                        other => panic!("{other:?}"),
                    }
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    assert!(bad.is_empty(), "div_from is wrong (chunk start, how many, first): {bad:?}");
}

/// The same bits, or both NaN.
fn same(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

#[test]
fn std_math_edges_match_c() {
    let mut host = CpuHost::load(built("math")).expect("load");
    let tiny = f32::from_bits(1);
    let edges = [
        0.0,
        -0.0,
        tiny,
        -tiny,
        1e-5,
        -1e-5,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        3.0,
        -3.0,
        2.5,
        1e10,
        -1e10,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    type One = fn(f64) -> f64;
    let ones: [(&str, One); 10] = [
        ("f_sin", f64::sin),
        ("f_cos", f64::cos),
        ("f_tan", f64::tan),
        ("f_exp", f64::exp),
        ("f_exp2", f64::exp2),
        ("f_log", f64::ln),
        ("f_log2", f64::log2),
        ("f_atan", f64::atan),
        ("f_asin", f64::asin),
        ("f_acos", f64::acos),
    ];
    // Zeros, infinities and NaNs exactly; anything else within an ulp.
    let check = |what: String, got: f32, want: f32| {
        let special = want == 0.0 || !want.is_finite();
        assert!(
            if special { same(got, want) } else { ulps(got, want) <= 1.0 },
            "{what} = {got:e}, want {want:e}"
        );
    };
    for (name, f) in ones {
        for &x in &edges {
            let got = one_f32(&mut host, name, &[Value::F32(x)]);
            check(format!("{name}({x:e})"), got, f(f64::from(x)) as f32);
        }
    }
    for &x in &edges {
        for &y in &edges {
            let (a, b) = (Value::F32(x), Value::F32(y));
            let got = one_f32(&mut host, "f_pow", &[a, b]);
            check(format!("pow({x:e}, {y:e})"), got, f64::from(x).powf(f64::from(y)) as f32);
            let got = one_f32(&mut host, "f_atan2", &[a, b]);
            check(format!("atan2({x:e}, {y:e})"), got, f64::from(x).atan2(f64::from(y)) as f32);
        }
    }
}

/// Calls export `name` with f32 arguments.
fn call(host: &mut CpuHost, name: &str, args: &[f32]) -> Vec<f32> {
    let args: Vec<Value> = args.iter().map(|&x| Value::F32(x)).collect();
    host.call_export(name, &args)
        .expect(name)
        .into_iter()
        .map(|v| match v {
            Value::F32(x) => x,
            other => panic!("{name} returned {other:?}"),
        })
        .collect()
}

/// The signed distance to the hull of two spheres (a round cone), in f64: the hull is the union
/// of the spheres between them, `c(t) = a + t (b - a)` with radius `r1 + t (r2 - r1)`, and
/// `|p - c(t)| - r(t)` is convex in `t`, so a ternary search finds its minimum, which is the
/// signed distance inside as well as out.
fn round_cone_reference(a: [f32; 3], b: [f32; 3], r1: f32, r2: f32, p: [f32; 3]) -> f64 {
    let f = |t: f64| {
        let d: f64 = (0..3)
            .map(|i| {
                let (a, b) = (f64::from(a[i]), f64::from(b[i]));
                (f64::from(p[i]) - (a + t * (b - a))).powi(2)
            })
            .sum();
        d.sqrt() - (f64::from(r1) + t * f64::from(r2 - r1))
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let (m1, m2) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if f(m1) < f(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    f(0.0).min(f(1.0)).min(f(0.5 * (lo + hi)))
}

#[test]
fn std_fields_hold_at_their_edge_cases() {
    let mut host = CpuHost::load(built("math")).expect("load");
    let mut rng = Rng::new(7);
    let mut r = |lo: f64, hi: f64| (lo + (hi - lo) * rng.unit()) as f32;

    // smin: `k = 0` (or less) is `min`, not NaN; a smooth union with `k = 0` is a union.
    for k in [0.0, -0.0, -1.0] {
        for (a, b) in [(1.0f32, 2.0f32), (2.0, 1.0), (1.0, 1.0), (-3.0, 0.5)] {
            let got = call(&mut host, "f_smin", &[a, b, k])[0];
            assert_eq!(got, a.min(b), "smin({a}, {b}, {k})");
        }
    }
    assert_eq!(call(&mut host, "f_blend", &[0.0, 0.0, 0.0, 0.0])[0], 0.0);
    assert_eq!(call(&mut host, "f_blend", &[3.0, 0.0, 0.0, 0.0])[0], 1.0);
    // Two empty surfaces (infinite distances) blend to an empty one, and a huge `k` doesn't
    // overflow: on the seam it's `a - k / 4`.
    assert_eq!(call(&mut host, "f_smin", &[f32::INFINITY, f32::INFINITY, 0.1])[0], f32::INFINITY);
    assert_eq!(call(&mut host, "f_smin", &[1.0, 1.0, 1e20])[0], 1.0 - 2.5e19);

    // round_cone is exact everywhere: two spheres apart, one holding the other, the same, or
    // nearly the same (where the usual formula's terms leave f32's range).
    let mut worst = 0.0f64;
    for i in 0..2000 {
        let a = [r(-1.0, 1.0), r(-1.0, 1.0), r(-1.0, 1.0)];
        let near = |a: [f32; 3], d: f32| [a[0] + d, a[1], a[2] - d];
        let b = match i % 10 {
            0 => a,
            1 => near(a, 1e-9),
            2 => near(a, 3e-8),
            3 => near(a, 1e-6),
            _ => [r(-1.0, 1.0), r(-1.0, 1.0), r(-1.0, 1.0)],
        };
        let (r1, r2) = (r(0.0, 1.5), r(0.0, 1.5));
        let p = [r(-2.5, 2.5), r(-2.5, 2.5), r(-2.5, 2.5)];
        let args = [a[0], a[1], a[2], b[0], b[1], b[2], r1, r2, p[0], p[1], p[2]];
        let got = f64::from(call(&mut host, "f_round_cone", &args)[0]);
        let want = round_cone_reference(a, b, r1, r2, p);
        let e = (got - want).abs();
        assert!(e < 1e-5, "round_cone({a:?}, {b:?}, {r1}, {r2}) at {p:?} = {got}, want {want}");
        worst = worst.max(e);
    }
    println!("round_cone: worst error {worst:.2e} over 2000 cases");

    // The ellipsoid's bound is inside (negative) at its centre too: the smallest radius there.
    assert_eq!(call(&mut host, "f_ellipsoid", &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0])[0], -1.0);
    assert_eq!(call(&mut host, "f_ellipsoid", &[1.0, 2.0, 3.0, -0.0, 0.0, -0.0])[0], -1.0);
    assert!(call(&mut host, "f_ellipsoid", &[1.0, 2.0, 3.0, 1e-20, 0.0, 0.0])[0] < 0.0);

    // half_space is exact for a normal of any length, and keeps the same half-space.
    // Also one whose length's square would underflow or overflow.
    let planes: [([f32; 3], f32); 6] = [
        ([0.0, 2.0, 0.0], 1.0),
        ([3.0, 0.0, 4.0], -5.0),
        ([0.0, -0.5, 0.0], 0.0),
        ([0.0, 1e-23, 0.0], 0.0),
        ([3e-30, 0.0, 4e-30], 1e-30),
        ([0.0, 1e20, 1e20], 2e20),
    ];
    for (n, offset) in planes {
        let n64 = n.map(f64::from);
        let len = (n64[0] * n64[0] + n64[1] * n64[1] + n64[2] * n64[2]).sqrt();
        for p in [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]] {
            let got =
                call(&mut host, "f_half_space", &[n[0], n[1], n[2], offset, p[0], p[1], p[2]]);
            let p64 = p.map(f64::from);
            let want =
                (n64[0] * p64[0] + n64[1] * p64[1] + n64[2] * p64[2] - f64::from(offset)) / len;
            let e = (f64::from(got[0]) - want).abs();
            assert!(e < 1e-6, "half_space({n:?}, {offset}) at {p:?}: {got:?}, want {want}");
        }
    }

    // Value noise and fbm take any coordinate: past the i32 range a lattice coordinate wraps
    // to 32 bits, as the hash does (so 2^31 and -2^31 are one plane), and inf or NaN gives NaN.
    let big = 2147483648.0f32;
    for (y, z) in [(0.25, 0.5), (-7.5, 1e9)] {
        let a = call(&mut host, "f_value_noise", &[big, y, z])[0];
        let b = call(&mut host, "f_value_noise", &[-big, y, z])[0];
        assert_eq!(a, b, "value noise at x = 2^31 and -2^31");
        assert!((-1.0..=1.0).contains(&a), "{a}");
    }
    for x in [3e9, -3e9, 1e30, f32::MAX, -f32::MAX] {
        let v = call(&mut host, "f_value_noise", &[x, 0.5, 0.5])[0];
        assert!((-1.0..=1.0).contains(&v), "value noise at {x:e}: {v}");
        // fbm's octaves scale the point by up to 8 here, so only where that stays finite.
        if x.abs() < 1e37 {
            let f = call(&mut host, "f_fbm", &[x, 0.5, 0.5])[0];
            assert!(f.abs() <= 2.0, "fbm at {x:e}: {f}");
        }
    }
    for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(call(&mut host, "f_value_noise", &[x, 0.5, 0.5])[0].is_nan(), "{x}");
        assert!(call(&mut host, "f_fbm", &[0.5, x, 0.5])[0].is_nan(), "{x}");
    }
}
